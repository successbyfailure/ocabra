"""
POST /v1/embeddings — text embeddings endpoint.
"""

from __future__ import annotations

import base64
import math
import struct
from typing import Annotated, Any

import structlog
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from ocabra.api._deps_auth import UserContext
from ocabra.core.inference.pipeline import (
    PipelineContext,
    RequestDescriptor,
    RequestOutcome,
    run_request,
)
from ocabra.core.request_telemetry import mark_prefill_since

from ._deps import (
    NON_STREAMING_LOAD_RESPONSE_DOC,
    _openai_error,
    get_federation_manager,
    get_model_manager,
    get_openai_user,
    get_profile_registry,
    read_json_object,
)
from ._federation import try_proxy_json
from ._schemas import EmbeddingRequest, validate_body

router = APIRouter()
logger = structlog.get_logger(__name__)

# REF-01: /v1/embeddings declara su flujo y delega en ``run_request``. Rellena
# sus dos huecos de la matriz respecto a hoy: ``Desconexión`` (un cliente que se
# va cancela el embedding en curso) y la formalización del paso ``admit`` —que
# para embeddings NO añade reja de reparto justo, conservando su concurrencia
# alta (ver ``project_embeddings_inflight_cap``).
_EMBEDDINGS_DESCRIPTOR = RequestDescriptor(
    request_kind="embeddings",
    capability="embeddings",
    dispatch_path="/v1/embeddings",
    federates=True,
    honors_disconnect=True,
    applies_fair_share=False,
)


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


def _validate_dimensions(body: dict) -> None:
    dimensions = body.get("dimensions")
    if dimensions is not None and (not isinstance(dimensions, int) or dimensions < 1):
        raise _openai_error(
            "The 'dimensions' field must be a positive integer.",
            "invalid_request_error",
            param="dimensions",
            code="invalid_dimensions",
        )


@router.post("/embeddings", summary="Create embeddings", responses=NON_STREAMING_LOAD_RESPONSE_DOC)
async def embeddings(
    request: Request,
    user: Annotated[UserContext, Depends(get_openai_user)],
) -> Any:
    """
    Create text embeddings. Proxies to the resolved model worker (backend-agnostic).
    Requires a model with capability embeddings=True.
    """
    # Permissive early validation (REF-11/BUG-18): no ``model`` or no ``input``
    # is a clear 422; token-id arrays and extra fields still pass through.
    body = validate_body(EmbeddingRequest, await read_json_object(request))

    model_manager = get_model_manager(request)
    profile_registry = get_profile_registry(request)

    async def _federate(fed_model_id: str, fed_body: dict):
        return await try_proxy_json(
            request,
            model_id=fed_model_id,
            body=fed_body,
            federation_manager=get_federation_manager(request),
            model_manager=model_manager,
            profile_registry=profile_registry,
            logger=logger,
        )

    async def _pre_dispatch(ctx: PipelineContext) -> None:
        # ``dimensions`` se valida con el cuerpo ya fusionado, antes del despacho.
        _validate_dimensions(ctx.merged_body)

    outcome = await run_request(
        request,
        descriptor=_EMBEDDINGS_DESCRIPTOR,
        body=body,
        user=user,
        model_manager=model_manager,
        profile_registry=profile_registry,
        worker_pool=request.app.state.worker_pool,
        federate=_federate,
        pre_dispatch=_pre_dispatch,
    )
    if not isinstance(outcome, RequestOutcome):
        # La federación sirvió la petición: devuelve la respuesta del peer.
        return outcome

    result = outcome.result
    # Un embedding es todo prefill: con este tiempo el middleware de stats
    # calcula los tokens/s que el worker no devuelve.
    mark_prefill_since(request.state, outcome.forward_started)
    dimensions = outcome.merged_body.get("dimensions")
    if isinstance(dimensions, int) and dimensions >= 1:
        _apply_requested_dimensions(result, dimensions)
    return JSONResponse(content=result, headers=outcome.response_headers)
