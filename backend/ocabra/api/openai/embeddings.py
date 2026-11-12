"""
POST /v1/embeddings — text embeddings endpoint.
"""

from __future__ import annotations

import base64
import math
import struct
import time
from typing import Annotated, Any

import httpx
import structlog
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from ocabra.api._deps_auth import UserContext
from ocabra.core.request_telemetry import mark_prefill_since
from ocabra.core.worker_pool import InferenceTimeoutError, WorkerDegradedError

from ._deps import (
    NON_STREAMING_LOAD_RESPONSE_DOC,
    _openai_error,
    check_capability,
    compute_worker_key,
    get_federation_manager,
    get_model_manager,
    get_openai_user,
    get_profile_registry,
    merge_profile_defaults,
    ocabra_response_headers,
    raise_inference_dispatch_error,
    raise_upstream_http_error,
    resolve_profile,
    to_backend_body,
)
from ._federation import try_proxy_json

router = APIRouter()
logger = structlog.get_logger(__name__)


def _truncate_and_renormalize(values: list[float], dimensions: int) -> list[float]:
    truncated = values[:dimensions]
    norm = math.sqrt(sum(v * v for v in truncated))
    if norm > 0:
        truncated = [v / norm for v in truncated]
    return truncated


def _apply_requested_dimensions(result: dict, dimensions: int) -> None:
    """Truncate + L2-renormalize each embedding to ``dimensions``, in place.

    Matryoshka-trained models (Qwen3-Embedding and similar) support this
    safely. Some backends already honor OpenAI's ``dimensions`` param
    natively (Ollama's embeddings API does) — this is a no-op for those,
    since it only ever shrinks a vector that's longer than requested. It's
    only a real fix for backends that silently ignore ``dimensions``:
    reproduced live 2026-11-03 against qwen3-embedding:8b after it was
    migrated to llama_cpp, whose server has no equivalent support — every
    response kept the model's native width (4096) regardless of what was
    requested, breaking a pgvector index built for 4000-dim vectors.
    """
    data = result.get("data")
    if not isinstance(data, list):
        return
    for item in data:
        if not isinstance(item, dict):
            continue
        embedding = item.get("embedding")
        if isinstance(embedding, list) and len(embedding) > dimensions:
            item["embedding"] = _truncate_and_renormalize(embedding, dimensions)
        elif isinstance(embedding, str):
            # encoding_format="base64": float32 little-endian array, same
            # wire format OpenAI's own API uses for it.
            raw = base64.b64decode(embedding)
            if len(raw) // 4 > dimensions:
                values = list(struct.unpack(f"<{len(raw) // 4}f", raw))
                truncated = _truncate_and_renormalize(values, dimensions)
                item["embedding"] = base64.b64encode(
                    struct.pack(f"<{len(truncated)}f", *truncated)
                ).decode("ascii")


@router.post("/embeddings", summary="Create embeddings", responses=NON_STREAMING_LOAD_RESPONSE_DOC)
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
    profile_registry = get_profile_registry(request)

    fed_resp = await try_proxy_json(
        request,
        model_id=model_id,
        body=body,
        federation_manager=get_federation_manager(request),
        model_manager=model_manager,
        profile_registry=profile_registry,
        logger=logger,
    )
    if fed_resp is not None:
        return fed_resp

    profile, state = await resolve_profile(
        model_id,
        model_manager,
        profile_registry,
        user=user,
        request_body=body,
        router_resolver=getattr(request.app.state, "router_resolver", None),
        request_state=request.state,
    )
    check_capability(state, "embeddings", "embeddings")

    merged_body = merge_profile_defaults(profile, body)
    worker_key = compute_worker_key(profile.base_model_id, profile.load_overrides)

    dimensions = merged_body.get("dimensions")
    if dimensions is not None and (not isinstance(dimensions, int) or dimensions < 1):
        raise _openai_error(
            "The 'dimensions' field must be a positive integer.",
            "invalid_request_error",
            param="dimensions",
            code="invalid_dimensions",
        )

    worker_pool = request.app.state.worker_pool
    forward_started = time.monotonic()
    try:
        result = await worker_pool.forward_request(
            worker_key,
            "/v1/embeddings",
            to_backend_body(state, merged_body),
        )
    except httpx.HTTPStatusError as exc:
        raise_upstream_http_error(exc)
    except (InferenceTimeoutError, WorkerDegradedError) as exc:
        raise_inference_dispatch_error(exc)
    # Un embedding es todo prefill: con este tiempo el middleware de stats
    # calcula los tokens/s que el worker no devuelve.
    mark_prefill_since(request.state, forward_started)
    if isinstance(dimensions, int) and dimensions >= 1:
        _apply_requested_dimensions(result, dimensions)
    resp_headers = await ocabra_response_headers(
        request, model_manager, worker_key, profile.base_model_id
    )
    return JSONResponse(content=result, headers=resp_headers)
