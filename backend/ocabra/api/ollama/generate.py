"""
POST /api/generate — Ollama generate compatibility endpoint.
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

from ._mapper import resolve_model
from ._shared import (
    apply_option_map,
    build_native_passthrough_body,
    get_ollama_user,
    iter_sse_payloads,
    now_iso_z,
)
from .chat import _native_backend_model_name

router = APIRouter()


@router.post("/generate", summary="Generate completion")
async def generate(
    request: Request,
    user: UserContext = Depends(get_ollama_user),
):
    """
    Generate text from a prompt using Ollama-compatible request/response format.

    Request fields include model, prompt, stream, and options.
    Streaming responses are returned as NDJSON lines.
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

    check_capability(state, "completion", "text generation")

    worker_pool = request.app.state.worker_pool
    if state.backend_type == "ollama":
        upstream_body = build_native_passthrough_body(
            merged_body,
            model=_native_backend_model_name(ollama_model, state.backend_model_id),
            stream=stream,
            content_keys=("prompt",),
            passthrough_keys=(
                "suffix",
                "system",
                "template",
                "context",
                "raw",
                "format",
                "keep_alive",
                "images",
                "think",
            ),
        )
        if stream:
            return StreamingResponse(
                worker_pool.forward_stream(worker_key, "/api/generate", upstream_body),
                media_type="application/x-ndjson",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        return await worker_pool.forward_request(worker_key, "/api/generate", upstream_body)

    vllm_body = _build_vllm_generate_body(merged_body, state.backend_model_id, stream)

    if stream:
        return StreamingResponse(
            _stream_generate(worker_pool, worker_key, ollama_model, vllm_body),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    started_ns = time.monotonic_ns()
    result = await worker_pool.forward_request(worker_key, "/v1/completions", vllm_body)
    text = ""
    choices = result.get("choices") or []
    if choices:
        text = str(choices[0].get("text") or "")
    now_iso = now_iso_z()
    total_duration = time.monotonic_ns() - started_ns
    usage = result.get("usage") or {}
    return {
        "model": ollama_model,
        "created_at": now_iso,
        "response": text,
        "done": True,
        "total_duration": total_duration,
        "load_duration": 0,
        "prompt_eval_count": int(usage.get("prompt_tokens") or 0),
        "eval_count": int(usage.get("completion_tokens") or 0),
        "eval_duration": total_duration,
    }


def _build_vllm_generate_body(payload: dict, model_id: str, stream: bool) -> dict:
    body: dict = {
        "model": model_id,
        "prompt": payload.get("prompt", ""),
        "stream": stream,
    }

    # Common top-level knobs accepted by Ollama.
    for key in ("suffix", "temperature", "top_p", "stop", "seed"):
        if key in payload:
            body[key] = payload[key]

    apply_option_map(body, payload.get("options"))

    return body


async def _stream_generate(
    worker_pool,
    model_id: str,
    ollama_model: str,
    body: dict,
) -> AsyncIterator[bytes]:
    started_ns = time.monotonic_ns()
    prompt_eval_count = 0
    eval_count = 0

    async for payload in iter_sse_payloads(
        worker_pool.forward_stream(model_id, "/v1/completions", body)
    ):
        if payload == "[DONE]":
            total_duration = time.monotonic_ns() - started_ns
            done_payload = {
                "model": ollama_model,
                "created_at": now_iso_z(),
                "response": "",
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

        token = str(choices[0].get("text") or "")
        if token == "":
            continue

        chunk = {
            "model": ollama_model,
            "created_at": now_iso_z(),
            "response": token,
            "done": False,
        }
        yield (json.dumps(chunk) + "\n").encode("utf-8")
