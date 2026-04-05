"""
POST /v1/completions — legacy text completions endpoint.
"""
from __future__ import annotations

import json
from typing import Annotated, Any

import httpx
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from ocabra.api._deps_auth import UserContext

from ._deps import (
    check_capability,
    ensure_loaded,
    get_model_manager,
    get_openai_user,
    raise_upstream_http_error,
    to_backend_body,
)

router = APIRouter()


@router.post("/completions", summary="Create text completion")
async def completions(
    request: Request,
    user: Annotated[UserContext, Depends(get_openai_user)],
) -> Any:
    """
    Create a text completion. Proxies to the resolved model worker (backend-agnostic).
    """
    body = await request.json()
    model_id: str = body.get("model", "")
    stream: bool = body.get("stream", False)

    model_manager = get_model_manager(request)
    state = await ensure_loaded(model_manager, model_id, user=user)
    check_capability(state, "completion", "text completions")
    model_id = state.model_id

    worker_pool = request.app.state.worker_pool

    if stream:
        return StreamingResponse(
            _stream_completions(worker_pool, model_id, to_backend_body(state, body)),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        result = await worker_pool.forward_request(model_id, "/v1/completions", to_backend_body(state, body))
    except httpx.HTTPStatusError as exc:
        raise_upstream_http_error(exc)
    return result


async def _stream_completions(worker_pool, model_id: str, body: dict):
    try:
        async for chunk in worker_pool.forward_stream(model_id, "/v1/completions", body):
            yield chunk
    except Exception as e:
        error_payload = json.dumps({
            "error": {"message": str(e), "type": "server_error", "code": "stream_error"}
        })
        yield f"data: {error_payload}\n\n".encode()
        yield b"data: [DONE]\n\n"
