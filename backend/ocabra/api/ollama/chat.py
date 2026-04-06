"""
POST /api/chat — Ollama chat compatibility endpoint.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from starlette.responses import StreamingResponse

from ocabra.api._deps_auth import UserContext
from ocabra.api.openai._deps import (
    check_capability,
    compute_worker_key,
    ensure_loaded,
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
