"""
POST /v1/embeddings — text embeddings endpoint.
"""
from __future__ import annotations

from typing import Any

import httpx
from fastapi import APIRouter, Request

from ._deps import (
    check_capability,
    ensure_loaded,
    get_model_manager,
    raise_upstream_http_error,
)

router = APIRouter()


@router.post("/embeddings", summary="Create embeddings")
async def embeddings(request: Request) -> Any:
    """
    Create text embeddings. Proxies to the model's vLLM worker.
    Requires a model with capability embeddings=True.
    """
    body = await request.json()
    model_id: str = body.get("model", "")

    model_manager = get_model_manager(request)
    state = await ensure_loaded(model_manager, model_id)
    check_capability(state, "embeddings", "embeddings")

    worker_pool = request.app.state.worker_pool
    try:
        return await worker_pool.forward_request(model_id, "/v1/embeddings", body)
    except httpx.HTTPStatusError as exc:
        raise_upstream_http_error(exc)
