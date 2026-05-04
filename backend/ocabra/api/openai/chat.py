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

import json
from typing import Annotated, Any

import httpx
import structlog
from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response, StreamingResponse

from ocabra.api._deps_auth import UserContext

from ._deps import (
    check_capability,
    compute_worker_key,
    get_federation_manager,
    get_model_manager,
    get_openai_user,
    get_profile_registry,
    merge_profile_defaults,
    raise_upstream_http_error,
    resolve_profile,
    to_backend_body,
)
from ocabra.agents.chat_glue import (
    build_invoker_for_agent,
    build_subagent_runner,
    extract_per_request_headers,
    parse_allowed_tools_header,
)
from ocabra.agents.executor import AgentExecutor
from ocabra.agents.mcp_registry import get_registry as get_mcp_registry
from ocabra.agents.resolver import is_agent_model, resolve_agent
from ocabra.database import AsyncSessionLocal

router = APIRouter()
logger = structlog.get_logger(__name__)


@router.post("/chat/completions", summary="Create chat completion")
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
                request.state.federation_remote_node_id = peer.peer_id
                if stream:
                    return StreamingResponse(
                        federation_manager.proxy_stream(peer, "POST", request.url.path, body),
                        media_type="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                    )
                resp = await federation_manager.proxy_request(
                    peer,
                    "POST",
                    request.url.path,
                    body,
                )
                return Response(
                    content=resp.content,
                    status_code=resp.status_code,
                    media_type=resp.headers.get("content-type"),
                )
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
    _content_kinds = _detect_message_content_kinds(body.get("messages") or [])
    if "image_url" in _content_kinds:
        check_capability(state, "vision", "image input")
    if "input_audio" in _content_kinds:
        check_capability(state, "audio_input", "audio input")
    if "video_url" in _content_kinds:
        check_capability(state, "video_input", "video input")

    merged_body = merge_profile_defaults(profile, body)
    worker_key = compute_worker_key(profile.base_model_id, profile.load_overrides)

    worker_pool = request.app.state.worker_pool

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
