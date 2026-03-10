"""
POST /api/generate — Ollama generate compatibility endpoint.
"""
from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime

from fastapi import APIRouter, Request
from starlette.responses import StreamingResponse

from ocabra.api.openai._deps import check_capability, ensure_loaded, get_model_manager

from ._mapper import OllamaNameMapper

router = APIRouter()
_mapper = OllamaNameMapper()

OPTION_MAP = {
    "num_predict": "max_tokens",
    "num_ctx": "max_model_len",
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "stop": "stop",
    "seed": "seed",
    "repeat_penalty": "repetition_penalty",
}


@router.post("/generate", summary="Generate completion")
async def generate(request: Request):
    """
    Generate text from a prompt using Ollama-compatible request/response format.

    Request fields include model, prompt, stream, and options.
    Streaming responses are returned as NDJSON lines.
    """
    body = await request.json()
    ollama_model = str(body.get("model", ""))
    model_id = _mapper.to_internal(ollama_model)
    stream = bool(body.get("stream", True))

    model_manager = get_model_manager(request)
    state = await ensure_loaded(model_manager, model_id)
    check_capability(state, "completion", "text generation")

    vllm_body = _build_vllm_generate_body(body, model_id, stream)

    worker_pool = request.app.state.worker_pool
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

    options = payload.get("options") or {}
    if isinstance(options, dict):
        for key, value in options.items():
            mapped = OPTION_MAP.get(key)
            if mapped:
                body[mapped] = value

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

    async for payload in _iter_sse_payloads(
        worker_pool.forward_stream(model_id, "/v1/completions", body)
    ):
        if payload == "[DONE]":
            total_duration = time.monotonic_ns() - started_ns
            done_payload = {
                "model": ollama_model,
                "created_at": _now_iso_z(),
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
            "created_at": _now_iso_z(),
            "response": token,
            "done": False,
        }
        yield (json.dumps(chunk) + "\n").encode("utf-8")


async def _iter_sse_payloads(source: AsyncIterator[bytes]) -> AsyncIterator[dict | str]:
    """Parse `data: ...\n\n` SSE frames and yield decoded payloads."""
    buffer = ""
    async for chunk in source:
        if not chunk:
            continue
        buffer += chunk.decode("utf-8", errors="ignore")

        while "\n\n" in buffer:
            raw_event, buffer = buffer.split("\n\n", 1)
            data_lines = []
            for line in raw_event.splitlines():
                if line.startswith("data:"):
                    data_lines.append(line[5:].strip())
            if not data_lines:
                continue

            data = "\n".join(data_lines).strip()
            if not data:
                continue
            if data == "[DONE]":
                yield "[DONE]"
                continue
            try:
                yield json.loads(data)
            except json.JSONDecodeError:
                continue


def _now_iso_z() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
