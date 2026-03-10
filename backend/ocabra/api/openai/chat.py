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
from typing import Any

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from ._deps import check_capability, ensure_loaded, get_model_manager

router = APIRouter()
logger = structlog.get_logger(__name__)


@router.post("/chat/completions", summary="Create chat completion")
async def chat_completions(request: Request) -> Any:
    """
    Create a chat completion. Proxies to the model's vLLM worker.

    Triggers on-demand model loading if the model is configured but not loaded.
    """
    body = await request.json()
    model_id: str = body.get("model", "")
    stream: bool = body.get("stream", False)

    model_manager = get_model_manager(request)
    state = await ensure_loaded(model_manager, model_id)
    check_capability(state, "chat", "chat completions")

    worker_pool = request.app.state.worker_pool

    if stream:
        return StreamingResponse(
            _stream_chat(worker_pool, model_id, body),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    result = await worker_pool.forward_request(model_id, "/v1/chat/completions", body)
    return result


async def _stream_chat(worker_pool, model_id: str, body: dict):
    """Yield SSE chunks from the vLLM worker."""
    try:
        async for chunk in worker_pool.forward_stream(model_id, "/v1/chat/completions", body):
            yield chunk
    except Exception as e:
        error_payload = json.dumps({
            "error": {"message": str(e), "type": "server_error", "code": "stream_error"}
        })
        yield f"data: {error_payload}\n\n".encode()
        yield b"data: [DONE]\n\n"
