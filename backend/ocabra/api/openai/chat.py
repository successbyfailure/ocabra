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
    if federation_manager is not None:
        from ocabra.config import settings as _settings

        if _settings.federation_enabled:
            from ocabra.core.federation import resolve_federated

            target, peer = await resolve_federated(model_id, model_manager, federation_manager)
            if target == "remote":
                request.state.federation_remote_node_id = peer.peer_id
                if stream:
                    return StreamingResponse(
                        federation_manager.proxy_stream(peer, "POST", request.url.path, body),
                        media_type="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                    )
                resp = await federation_manager.proxy_request(
                    peer, "POST", request.url.path, body,
                )
                return Response(
                    content=resp.content,
                    status_code=resp.status_code,
                    media_type=resp.headers.get("content-type"),
                )
    # --- End federation hook ---

    profile, state = await resolve_profile(
        model_id,
        model_manager,
        profile_registry,
        user=user,
    )
    check_capability(state, "chat", "chat completions")
    if body.get("tools"):
        check_capability(state, "tools", "tool calling")

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
