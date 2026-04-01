"""
POST /api/generate — Ollama generate compatibility endpoint.
"""
from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator

from fastapi import APIRouter, Request
from starlette.responses import StreamingResponse

from ocabra.api.openai._deps import check_capability, ensure_loaded, get_model_manager

from ._mapper import resolve_model
from ._shared import apply_option_map, build_native_passthrough_body, iter_sse_payloads, now_iso_z

router = APIRouter()


@router.post("/generate", summary="Generate completion")
async def generate(request: Request):
    """
    Generate text from a prompt using Ollama-compatible request/response format.

    Request fields include model, prompt, stream, and options.
    Streaming responses are returned as NDJSON lines.
    """
    body = await request.json()
    ollama_model = str(body.get("model", ""))
    stream = bool(body.get("stream", True))

    model_manager = get_model_manager(request)
    model_id, _ = await resolve_model(model_manager, ollama_model)
    state = await ensure_loaded(model_manager, model_id)
    check_capability(state, "completion", "text generation")

    worker_pool = request.app.state.worker_pool
    if state.backend_type == "ollama":
        upstream_body = build_native_passthrough_body(
            body,
            model=ollama_model,
            stream=stream,
            content_keys=("prompt",),
        )
        if stream:
            return StreamingResponse(
                worker_pool.forward_stream(model_id, "/api/generate", upstream_body),
                media_type="application/x-ndjson",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        return await worker_pool.forward_request(model_id, "/api/generate", upstream_body)

    vllm_body = _build_vllm_generate_body(body, state.backend_model_id, stream)

    if stream:
        return StreamingResponse(
            _stream_generate(worker_pool, model_id, ollama_model, vllm_body),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    started_ns = time.monotonic_ns()
    result = await worker_pool.forward_request(model_id, "/v1/completions", vllm_body)
    text = ""
    choices = result.get("choices") or []
    if choices:
        text = str(choices[0].get("text") or "")
    now_iso = _now_iso_z()
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
