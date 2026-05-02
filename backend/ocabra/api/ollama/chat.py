"""
POST /api/chat — Ollama chat compatibility endpoint.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from starlette.responses import StreamingResponse

from ocabra.api._deps_auth import UserContext
from ocabra.api.openai._deps import (
    check_capability,
    compute_worker_key,
    ensure_loaded,
    get_federation_manager,
    get_model_manager,
    get_profile_registry,
    merge_profile_defaults,
    resolve_profile,
)

from ._mapper import resolve_model  # noqa: I001
from ._shared import (
    apply_option_map,
    build_native_passthrough_body,
    get_ollama_user,
    iter_sse_payloads,
    now_iso_z,
)
from ocabra.agents.chat_glue import (
    build_invoker_for_agent,
    extract_per_request_headers,
    parse_allowed_tools_header,
)
from ocabra.agents.executor import AgentExecutor
from ocabra.agents.mcp_registry import get_registry as get_mcp_registry
from ocabra.agents.resolver import is_agent_model, resolve_agent
from ocabra.database import AsyncSessionLocal

router = APIRouter()


def _native_backend_model_name(requested_model: str, backend_model_id: str | None) -> str:
    return backend_model_id or requested_model


@router.post("/chat", summary="Create chat completion")
async def chat(
    request: Request,
    user: UserContext = Depends(get_ollama_user),
):
    """
    Create a chat completion in Ollama format.

    Request fields include model, messages, stream, and options.
    Streaming responses are translated from OpenAI SSE to Ollama NDJSON.
    """
    body = await request.json()
    ollama_model = str(body.get("model", ""))
    stream = bool(body.get("stream", True))

    model_manager = get_model_manager(request)
    profile_registry = get_profile_registry(request)

    # --- Federation hook ---
    # Agent invocations (model="agent/<slug>") are local; skip federation.
    federation_manager = get_federation_manager(request)
    if federation_manager is not None and not is_agent_model(ollama_model):
        from ocabra.config import settings as _settings

        if _settings.federation_enabled:
            from ocabra.core.federation import resolve_federated

            target, peer = await resolve_federated(ollama_model, model_manager, federation_manager)
            if target == "remote":
                request.state.federation_remote_node_id = peer.peer_id
                if stream:
                    return StreamingResponse(
                        federation_manager.proxy_stream(peer, "POST", request.url.path, body),
                        media_type="application/x-ndjson",
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

    # --- Agent dispatch: model="agent/<slug>" via Ollama-compat ---
    if is_agent_model(ollama_model):
        return await _dispatch_agent_ollama(
            request=request,
            user=user,
            body=body,
            ollama_model=ollama_model,
            stream=stream,
        )

    # Try profile resolution first
    try:
        profile, state = await resolve_profile(
            ollama_model,
            model_manager,
            profile_registry,
            user=user,
        )
        merged_body = merge_profile_defaults(profile, body)
        worker_key = compute_worker_key(profile.base_model_id, profile.load_overrides)
    except HTTPException as profile_exc:
        # Fallback to legacy Ollama resolution for native Ollama models
        model_id, resolved_state = await resolve_model(model_manager, ollama_model, user=user)
        if resolved_state is None:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{ollama_model}' not found",
            ) from profile_exc
        state = await ensure_loaded(model_manager, model_id)
        merged_body = body
        worker_key = model_id
        profile = None

    check_capability(state, "chat", "chat")

    worker_pool = request.app.state.worker_pool
    if state.backend_type == "ollama":
        upstream_body = build_native_passthrough_body(
            merged_body,
            model=_native_backend_model_name(ollama_model, state.backend_model_id),
            stream=stream,
            content_keys=("messages",),
            passthrough_keys=("keep_alive", "format", "think", "tools"),
        )
        if stream:
            return StreamingResponse(
                worker_pool.forward_stream(worker_key, "/api/chat", upstream_body),
                media_type="application/x-ndjson",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        return await worker_pool.forward_request(worker_key, "/api/chat", upstream_body)

    vllm_body = _build_vllm_chat_body(merged_body, state.backend_model_id, stream)

    if stream:
        return StreamingResponse(
            _stream_chat(worker_pool, worker_key, ollama_model, vllm_body),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    started_ns = time.monotonic_ns()
    result = await worker_pool.forward_request(worker_key, "/v1/chat/completions", vllm_body)
    message = _extract_chat_message(result)
    total_duration = time.monotonic_ns() - started_ns
    usage = result.get("usage") or {}
    return {
        "model": ollama_model,
        "created_at": now_iso_z(),
        "message": {"role": "assistant", "content": message},
        "done": True,
        "total_duration": total_duration,
        "load_duration": 0,
        "prompt_eval_count": int(usage.get("prompt_tokens") or 0),
        "eval_count": int(usage.get("completion_tokens") or 0),
        "eval_duration": total_duration,
    }


def _build_vllm_chat_body(payload: dict, model_id: str, stream: bool) -> dict:
    body: dict = {
        "model": model_id,
        "messages": payload.get("messages", []),
        "stream": stream,
    }

    for key in (
        "tools",
        "tool_choice",
        "response_format",
        "temperature",
        "top_p",
        "stop",
        "seed",
        "max_tokens",
    ):
        if key in payload:
            body[key] = payload[key]

    options = payload.get("options") or {}
    apply_option_map(body, options)

    return body


async def _stream_chat(
    worker_pool,
    model_id: str,
    ollama_model: str,
    body: dict,
) -> AsyncIterator[bytes]:
    started_ns = time.monotonic_ns()
    prompt_eval_count = 0
    eval_count = 0

    async for payload in iter_sse_payloads(
        worker_pool.forward_stream(model_id, "/v1/chat/completions", body)
    ):
        if payload == "[DONE]":
            total_duration = time.monotonic_ns() - started_ns
            done_payload = {
                "model": ollama_model,
                "created_at": now_iso_z(),
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "total_duration": total_duration,
                "load_duration": 0,
                "prompt_eval_count": prompt_eval_count,
                "eval_count": eval_count,
                "eval_duration": total_duration,
            }
            yield (json.dumps(done_payload) + "\n").encode("utf-8")
            return

        if not isinstance(payload, dict):
            continue

        usage = payload.get("usage") or {}
        if usage:
            prompt_eval_count = int(usage.get("prompt_tokens") or prompt_eval_count)
            eval_count = int(usage.get("completion_tokens") or eval_count)

        choices = payload.get("choices") or []
        if not choices:
            continue

        delta = choices[0].get("delta") or {}
        token = str(delta.get("content") or "")
        if token == "":
            continue

        chunk = {
            "model": ollama_model,
            "created_at": now_iso_z(),
            "message": {"role": "assistant", "content": token},
            "done": False,
        }
        yield (json.dumps(chunk) + "\n").encode("utf-8")


def _extract_chat_message(payload: dict) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return str(message.get("content") or "")


# ── Agent dispatch (Ollama adapter) ──────────────────────────────


def _ollama_messages_to_openai(messages: list[dict]) -> list[dict]:
    """Translate Ollama-style messages to OpenAI-style messages.

    Ollama supports: ``role``, ``content`` (string), ``images`` (list of b64),
    ``tool_calls``.  OpenAI expects ``content`` as string or content-parts list,
    ``role=tool`` with ``tool_call_id``.
    """
    out: list[dict] = []
    for raw in messages or []:
        if not isinstance(raw, dict):
            continue
        role = raw.get("role")
        content = raw.get("content", "")
        images = raw.get("images") or []
        msg: dict = {"role": role}
        if images:
            parts: list[dict] = []
            if isinstance(content, str) and content:
                parts.append({"type": "text", "text": content})
            for img in images:
                if isinstance(img, str) and img:
                    parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img}"},
                        }
                    )
            msg["content"] = parts
        else:
            msg["content"] = content
        if raw.get("tool_calls"):
            msg["tool_calls"] = raw["tool_calls"]
        if raw.get("tool_call_id"):
            msg["tool_call_id"] = raw["tool_call_id"]
        out.append(msg)
    return out


def _openai_response_to_ollama(payload: dict, ollama_model: str) -> dict:
    """Translate the executor's OpenAI response into Ollama's ``/api/chat`` shape."""
    msg_dict = (payload.get("choices") or [{}])[0].get("message") or {}
    out_msg: dict = {"role": "assistant", "content": msg_dict.get("content") or ""}
    if msg_dict.get("tool_calls"):
        out_msg["tool_calls"] = msg_dict["tool_calls"]
    usage = payload.get("usage") or {}
    return {
        "model": ollama_model,
        "created_at": now_iso_z(),
        "message": out_msg,
        "done": True,
        "total_duration": 0,
        "load_duration": 0,
        "prompt_eval_count": int(usage.get("prompt_tokens") or 0),
        "eval_count": int(usage.get("completion_tokens") or 0),
        "eval_duration": 0,
    }


async def _dispatch_agent_ollama(
    *,
    request: Request,
    user: UserContext,
    body: dict,
    ollama_model: str,
    stream: bool,
):
    """Run the AgentExecutor and return a response in Ollama format.

    Ollama clients send Ollama-shaped request bodies; we translate them to
    OpenAI format, run the executor, and translate the final assistant
    response back.  Streaming is exposed as NDJSON, mirroring the rest of
    ``/api/chat``.
    """
    async with AsyncSessionLocal() as session:
        agent = await resolve_agent(ollama_model, session, user=user)
    if agent is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{ollama_model}' not found",
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
    raw_messages = body.get("messages") or []
    messages = _ollama_messages_to_openai(raw_messages)

    # See the OpenAI dispatcher for the rationale behind the contextvar set.
    from ocabra.agents.executor import current_agent_id  # local import to avoid cycle

    current_agent_id.set(agent.id)

    request_options: dict = {}
    options = body.get("options") or {}
    if isinstance(options, dict):
        # Map the relevant Ollama options to OpenAI sampling args.
        for src, dst in (
            ("temperature", "temperature"),
            ("top_p", "top_p"),
            ("top_k", "top_k"),
            ("seed", "seed"),
            ("num_predict", "max_tokens"),
            ("stop", "stop"),
        ):
            if src in options:
                request_options[dst] = options[src]
    if "tools" in body and isinstance(body["tools"], list):
        caller_tools = body["tools"]
    else:
        caller_tools = None

    per_request_headers = extract_per_request_headers(request)
    per_request_allowed = parse_allowed_tools_header(request.headers.get("x-ocabra-allowed-tools"))
    require_approval_override = request.headers.get("x-ocabra-require-approval")

    if stream:

        async def _gen():
            # Build the final response synchronously then emit a single done line.
            # The OpenAI streaming variant offers token-by-token deltas; for
            # Ollama we keep parity with the legacy implementation that drains
            # to a single chunk on tool-loops.
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
            translated = _openai_response_to_ollama(result.openai_response, ollama_model)
            yield (json.dumps(translated) + "\n").encode("utf-8")

        return StreamingResponse(
            _gen(),
            media_type="application/x-ndjson",
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
    return _openai_response_to_ollama(result.openai_response, ollama_model)
