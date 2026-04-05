"""
POST /v1/embeddings — text embeddings endpoint.
"""
from __future__ import annotations

from typing import Annotated, Any

import httpx
from fastapi import APIRouter, Depends, Request

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


@router.post("/embeddings", summary="Create embeddings")
async def embeddings(
    request: Request,
    user: Annotated[UserContext, Depends(get_openai_user)],
) -> Any:
    """
    Create text embeddings. Proxies to the resolved model worker (backend-agnostic).
    Requires a model with capability embeddings=True.
    """
    body = await request.json()
    model_id: str = body.get("model", "")

    model_manager = get_model_manager(request)
    state = await ensure_loaded(model_manager, model_id, user=user)
    check_capability(state, "embeddings", "embeddings")
    model_id = state.model_id

    worker_pool = request.app.state.worker_pool
    try:
        return await worker_pool.forward_request(model_id, "/v1/embeddings", to_backend_body(state, body))
    except httpx.HTTPStatusError as exc:
        raise_upstream_http_error(exc)
