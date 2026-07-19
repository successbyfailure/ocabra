"""
POST /v1/chat/completions — OpenAI chat completions endpoint.

Supports:
- Streaming (SSE)
- Tool calls
- Vision (image_url content parts)
- JSON mode (response_format)
- All standard parameters (temperature, top_p, max_tokens, stop, seed, etc.)
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Annotated, Any

import httpx
import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from ocabra.agents.chat_glue import (
    build_invoker_for_agent,
    build_subagent_runner,
    extract_per_request_headers,
    parse_allowed_tools_header,
)
from ocabra.agents.executor import AgentExecutor
from ocabra.agents.mcp_registry import get_registry as get_mcp_registry
from ocabra.agents.resolver import is_agent_model, resolve_agent
from ocabra.api._deps_auth import UserContext
from ocabra.database import AsyncSessionLocal

from ._deps import (
    STREAMING_LOAD_RESPONSE_DOC,
    build_model_status_headers,
    check_capability,
    compute_worker_key,
    ensure_worker_loaded,
    keepalive_until_done,
    get_federation_manager,
    get_model_manager,
    get_openai_user,
    get_profile_registry,
    lookup_profile,
    merge_profile_defaults,
    raise_upstream_http_error,
    resolve_profile,
    sse_ocabra_event,
    to_backend_body,
)

router = APIRouter()
logger = structlog.get_logger(__name__)


@router.post(
    "/chat/completions",
    summary="Create chat completion",
    responses=STREAMING_LOAD_RESPONSE_DOC,
)
async def chat_completions(
    request: Request,
    user: Annotated[UserContext, Depends(get_openai_user)],
) -> Any:
    """
    Generate a chat completion from a conversation history.

    **Supports**: streaming (SSE), tool/function calling, vision (image_url),
    JSON mode (`response_format`), and all standard OpenAI parameters.

    The `model` field accepts a **profile_id** (recommended) or a canonical model_id.
    Models are loaded on demand if not already in GPU.
    """
    body = await request.json()
    model_id: str = body.get("model", "")
    stream: bool = body.get("stream", False)

    model_manager = get_model_manager(request)
    profile_registry = get_profile_registry(request)

    # --- Federation: check if request should be proxied to a remote peer ---
    federation_manager = get_federation_manager(request)
    # Agent invocations (model="agent/<slug>") are local resources and must not
    # go through federation: resolve_federated would 404 because the agent
    # slug is not a profile or model_id.
    if federation_manager is not None and not is_agent_model(model_id):
        from ocabra.config import settings as _settings

        if _settings.federation_enabled:
            from ocabra.core.federation import resolve_federated

            target, peer = await resolve_federated(
                model_id, model_manager, federation_manager, profile_registry
            )
            if target == "remote":
                from ocabra.core.federation import should_fallback_to_local

                request.state.federation_remote_node_id = peer.peer_id
                if stream:
                    # SSE: can't peek status before streaming starts, so we
                    # rely on the peer's first chunk including the auth error
                    # if any. No local fallback for streaming requests.
                    return StreamingResponse(
                        federation_manager.proxy_stream(
                            peer=peer,
                            path=request.url.path,
                            body=body,
                            headers=dict(request.headers),
                        ),
                        media_type="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                    )
                try:
                    resp = await federation_manager.proxy_request(
                        peer=peer,
                        path=request.url.path,
                        body=body,
                        headers=dict(request.headers),
                    )
                except Exception as exc:
                    logger.warning(
                        "federation_peer_network_error_fallback_local",
                        peer=peer.name,
                        model_id=model_id,
                        error=str(exc),
                    )
                    request.state.federation_remote_node_id = None
                    resp = None
                if resp is not None and resp.status_code < 400:
                    return Response(
                        content=resp.content,
                        status_code=resp.status_code,
                        media_type=resp.headers.get("content-type"),
                    )
                if resp is not None and not should_fallback_to_local(resp.status_code):
                    return Response(
                        content=resp.content,
                        status_code=resp.status_code,
                        media_type=resp.headers.get("content-type"),
                    )
                if resp is not None:
                    logger.warning(
                        "federation_peer_rejected_fallback_local",
                        peer=peer.name,
                        model_id=model_id,
                        status=resp.status_code,
                        body_preview=resp.text[:200],
                    )
                    request.state.federation_remote_node_id = None
                # Fall through to local processing.
    # --- End federation hook ---

    # --- Agent dispatch: model="agent/<slug>" ---
    if is_agent_model(model_id):
        return await _dispatch_agent(
            request=request,
            user=user,
            body=body,
            stream=stream,
        )

    # Diagnostic: tells us when a request lands on the legacy path with a
    # model_id that doesn't have the agent/ prefix. Helps catch stale frontend
    # bundles that send the bare slug or the base model id.
    if isinstance(model_id, str) and (
        model_id.startswith("task-") or "agent" in model_id.lower()
    ):
        logger.warning(
            "agent_dispatch_skipped_no_prefix",
            model_id=model_id,
            user_id=user.user_id,
        )

    _content_kinds = _detect_message_content_kinds(body.get("messages") or [])
    worker_pool = request.app.state.worker_pool

    # Streaming path: pre-flush response headers (including
    # ``X-Ocabra-Expected-Wait-Seconds``) *before* triggering the load, so
    # clients see the wait estimate up front instead of after the cold start.
    if stream:
        try:
            profile = await lookup_profile(model_id, profile_registry, user=user)
        except HTTPException:
            # Legacy / model-id fallback paths aren't covered by lookup_profile.
            # Fall through to the regular resolve_profile path; the wait header
            # will be missing for these requests.
            profile = None

        if profile is not None:
            worker_key = compute_worker_key(
                profile.base_model_id, profile.load_overrides
            )
            headers = {
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            }
            headers.update(
                await build_model_status_headers(
                    model_manager, worker_key, profile.base_model_id
                )
            )
            return StreamingResponse(
                _stream_chat_with_load(
                    model_manager=model_manager,
                    worker_pool=worker_pool,
                    profile=profile,
                    worker_key=worker_key,
                    body=body,
                    content_kinds=_content_kinds,
                ),
                media_type="text/event-stream",
                headers=headers,
            )

    profile, state = await resolve_profile(
        model_id,
        model_manager,
        profile_registry,
        user=user,
    )
    check_capability(state, "chat", "chat completions")
    if body.get("tools"):
        check_capability(state, "tools", "tool calling")
    # Detect multimodal content parts inside messages and validate the model
    # advertises the matching capability. The body itself is forwarded to the
    # backend untouched: when the upstream (e.g. Ollama) starts accepting
    # ``input_audio`` / ``video_url``, oCabra will pass it through with no
    # further changes.
    if "image_url" in _content_kinds:
        check_capability(state, "vision", "image input")
    if "input_audio" in _content_kinds:
        check_capability(state, "audio_input", "audio input")
    if "video_url" in _content_kinds:
        check_capability(state, "video_input", "video input")

    merged_body = merge_profile_defaults(profile, body)
    worker_key = compute_worker_key(profile.base_model_id, profile.load_overrides)

    if stream:
        return StreamingResponse(
            _stream_chat(worker_pool, worker_key, to_backend_body(state, merged_body)),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        result = await worker_pool.forward_request(
            worker_key,
            "/v1/chat/completions",
            to_backend_body(state, merged_body),
        )
    except httpx.HTTPStatusError as exc:
        raise_upstream_http_error(exc)
    return result


def _sse_error(message: str, code: str) -> bytes:
    payload = json.dumps(
        {"error": {"message": message, "type": "server_error", "code": code}}
    )
    return f"data: {payload}\n\n".encode() + b"data: [DONE]\n\n"


async def _stream_chat(worker_pool, model_id: str, body: dict):
    """Yield SSE chunks from the resolved model worker."""
    try:
        async for chunk in worker_pool.forward_stream(model_id, "/v1/chat/completions", body):
            yield chunk
    except Exception as e:
        error_payload = json.dumps(
            {"error": {"message": str(e), "type": "server_error", "code": "stream_error"}}
        )
        yield f"data: {error_payload}\n\n".encode()
        yield b"data: [DONE]\n\n"


async def _stream_chat_with_load(
    *,
    model_manager,
    worker_pool,
    profile,
    worker_key: str,
    body: dict,
    content_kinds: set[str],
):
    """SSE generator that triggers the model load lazily.

    Flushes HTTP response headers (including ``X-Ocabra-Expected-Wait-Seconds``)
    plus a JSON-bearing SSE comment with the load status before blocking. When
    the load finishes a second comment carries the actual ``load_duration_ms``.
    Capability checks and load errors are surfaced as SSE error events instead
    of HTTP error responses, since headers have already been sent.
    """
    pre_state = await model_manager.get_state(worker_key)
    if pre_state is None and worker_key != profile.base_model_id:
        pre_state = await model_manager.get_state(profile.base_model_id)
    pre_status = pre_state.status.value if pre_state is not None else "unknown"
    expected_wait = None
    if pre_status != "loaded":
        expected_wait = await model_manager.get_expected_load_seconds(worker_key)
        if expected_wait is None and worker_key != profile.base_model_id:
            expected_wait = await model_manager.get_expected_load_seconds(
                profile.base_model_id
            )

    # First frame: comment-only SSE event with status. Forces header flush and
    # gives clients an immediate, parseable load hint.
    yield sse_ocabra_event(
        "model_loading",
        model_id=profile.base_model_id,
        worker_key=worker_key,
        status=pre_status,
        expected_wait_seconds=expected_wait,
    )

    load_started_at = time.monotonic()
    load_task = asyncio.ensure_future(
        ensure_worker_loaded(
            model_manager,
            profile.base_model_id,
            worker_key,
            profile.load_overrides,
        )
    )
    # Drip ``: keepalive`` comments while the load runs so reverse proxies
    # (NPM, Cloudflare, ...) don't close the connection on a 60s read timeout
    # for cold starts of large models.
    async for ka in keepalive_until_done(load_task):
        yield ka
    if load_task.exception() is not None:
        exc = load_task.exception()
        if isinstance(exc, HTTPException):
            detail = exc.detail.get("error", {}) if isinstance(exc.detail, dict) else {}
            yield _sse_error(
                detail.get("message") or str(exc.detail),
                detail.get("code") or "model_load_failed",
            )
        else:
            yield _sse_error(str(exc), "model_load_failed")
        return
    state = load_task.result()

    load_duration_ms = int((time.monotonic() - load_started_at) * 1000)
    yield sse_ocabra_event(
        "model_ready",
        model_id=state.model_id,
        worker_key=worker_key,
        load_duration_ms=load_duration_ms,
        was_cold_start=(pre_status != "loaded"),
    )

    try:
        check_capability(state, "chat", "chat completions")
        if body.get("tools"):
            check_capability(state, "tools", "tool calling")
        if "image_url" in content_kinds:
            check_capability(state, "vision", "image input")
        if "input_audio" in content_kinds:
            check_capability(state, "audio_input", "audio input")
        if "video_url" in content_kinds:
            check_capability(state, "video_input", "video input")
    except HTTPException as exc:
        detail = exc.detail.get("error", {}) if isinstance(exc.detail, dict) else {}
        yield _sse_error(
            detail.get("message") or str(exc.detail),
            detail.get("code") or "model_not_capable",
        )
        return

    merged_body = merge_profile_defaults(profile, body)
    backend_body = to_backend_body(state, merged_body)
    try:
        async for chunk in worker_pool.forward_stream(
            worker_key, "/v1/chat/completions", backend_body
        ):
            yield chunk
    except Exception as exc:
        yield _sse_error(str(exc), "stream_error")


async def _dispatch_agent(
    *,
    request: Request,
    user: UserContext,
    body: dict,
    stream: bool,
):
    """Resolve and execute an agent invocation (``model="agent/<slug>"``).

    Plan: docs/tasks/agents-mcp-plan.md — Fase 2/3.
    """
    from fastapi import HTTPException
    from fastapi.responses import JSONResponse

    model_id = body.get("model", "")
    async with AsyncSessionLocal() as session:
        agent = await resolve_agent(model_id, session, user=user)
    if agent is None:
        # Log enough context to debug 404s from the Playground without
        # leaking secrets. Distinguishes "agent does not exist" from "user
        # has no access" — both map to 404 client-side to avoid info leak.
        logger.warning(
            "agent_resolve_failed",
            model_id=model_id,
            user_id=user.user_id,
            user_role=user.role,
            user_group_count=len(user.group_ids or []),
        )
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "message": f"The model '{model_id}' does not exist.",
                    "type": "invalid_request_error",
                    "param": "model",
                    "code": "model_not_found",
                }
            },
        )

    registry = getattr(request.app.state, "mcp_registry", None) or get_mcp_registry()
    if registry is None:
        raise HTTPException(status_code=503, detail="MCP registry not available")

    model_manager = get_model_manager(request)
    profile_registry = get_profile_registry(request)
    invoker = await build_invoker_for_agent(
        agent,
        model_manager=model_manager,
        profile_registry=profile_registry,
        user=user,
        worker_pool=request.app.state.worker_pool,
    )

    executor = AgentExecutor(registry)
    executor._subagent_runner = build_subagent_runner(  # noqa: SLF001
        executor,
        model_manager=model_manager,
        profile_registry=profile_registry,
        user=user,
        worker_pool=request.app.state.worker_pool,
    )
    messages = body.get("messages") or []
    if not isinstance(messages, list):
        raise HTTPException(status_code=400, detail="messages must be an array")

    # Stamp the root request_stat with the agent id (consumed by the stats
    # middleware via contextvars).  Setting here means the middleware's
    # ``asyncio.create_task(_record_stat...)`` captures the live value.
    from ocabra.agents.executor import current_agent_id  # local import to avoid cycle

    current_agent_id.set(agent.id)

    request_options = {
        k: v for k, v in body.items() if k not in ("model", "messages", "stream", "tools")
    }
    caller_tools = body.get("tools") if isinstance(body.get("tools"), list) else None
    per_request_headers = extract_per_request_headers(request)
    per_request_allowed = parse_allowed_tools_header(request.headers.get("x-ocabra-allowed-tools"))
    require_approval_override = request.headers.get("x-ocabra-require-approval")
    # oCabra tool-progress events (event: ocabra.tool_*) break strict OpenAI SSE
    # parsers, so they're opt-in: only emitted when the client asks for them
    # (the Playground sets this). Default off keeps the stream standards-clean.
    emit_ocabra_events = str(
        request.headers.get("x-ocabra-stream-events", "")
    ).strip().lower() in ("1", "true", "yes", "on")

    if stream:
        from fastapi.responses import StreamingResponse as _SR

        agen = executor.run_stream(
            agent,
            messages,
            request_options,
            user,
            invoker,
            per_request_headers=per_request_headers,
            per_request_allowed_tools=per_request_allowed,
            caller_tools=caller_tools,
            require_approval_override=require_approval_override,
            emit_ocabra_events=emit_ocabra_events,
        )
        return _SR(
            agen,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    result = await executor.run(
        agent,
        messages,
        request_options,
        user,
        invoker,
        per_request_headers=per_request_headers,
        per_request_allowed_tools=per_request_allowed,
        caller_tools=caller_tools,
        require_approval_override=require_approval_override,
    )
    return JSONResponse(content=result.openai_response)


def _detect_message_content_kinds(messages: list) -> set[str]:
    """Return the set of multimodal content-part types used by the messages.

    Walks the OpenAI ``content`` arrays looking for ``{type: ...}`` parts.
    Recognised kinds: ``"image_url"``, ``"input_audio"``, ``"video_url"``.
    Plain string content is ignored.
    """
    kinds: set[str] = set()
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            kind = part.get("type")
            if kind in ("image_url", "input_audio", "video_url"):
                kinds.add(kind)
    return kinds
