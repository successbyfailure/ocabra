"""API interna de escalado de vídeo: ``POST /ocabra/video/upscale``.

Va en la API interna y no en ``/v1/*`` a propósito: no existe endpoint OpenAI
estándar para escalar vídeo y forzarlo dentro sería inventarse un contrato que
ningún cliente reconocería.

El contrato es deliberadamente pobre: entra **un segmento de vídeo sin audio**
y sale el segmento escalado. Sin estado entre peticiones, para que el
planificador siga siendo libre de expulsar el modelo entre segmentos según la
política que haya fijado el administrador. Quien trocea, remezcla el audio y
reensambla es el cliente (VHS), que ya tiene ffmpeg y NVENC afinados.
"""

from __future__ import annotations

from typing import Annotated

import httpx
import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response

from ocabra.api._deps_auth import UserContext, require_role
from ocabra.api.openai._deps import (
    check_capability,
    compute_worker_key,
    get_model_manager,
    get_profile_registry,
    resolve_profile,
)

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["video"])


@router.post(
    "/video/upscale",
    summary="Upscale a video segment",
    description=(
        "Upscale/restore one self-contained video segment. Send video only "
        "(no audio) and reassemble on the client side. Returns video/mp4."
    ),
)
async def upscale_video(
    request: Request,
    file: Annotated[UploadFile, File(description="Video segment, no audio")],
    model: Annotated[str, Form(description="Upscaler model id")],
    target_height: Annotated[int | None, Form()] = None,
    crf: Annotated[int, Form()] = 14,
    _user: UserContext = Depends(require_role("user")),
) -> Response:
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Empty video segment")

    model_manager = get_model_manager(request)
    profile_registry = get_profile_registry(request)

    profile, state = await resolve_profile(
        model, model_manager, profile_registry, user=_user
    )
    check_capability(state, "video_upscaling", "video upscaling")

    worker_key = compute_worker_key(profile.base_model_id, profile.load_overrides)
    request.state.stats_model_id = worker_key

    worker_pool = request.app.state.worker_pool
    inflight_request_id = model_manager.begin_request(worker_key)
    try:
        for attempt in range(2):
            worker = worker_pool.get_worker(worker_key)
            backend = await worker_pool.get_backend(state.backend_type)
            healthy = bool(worker) and await backend.health_check(state.backend_model_id)

            if not worker or not healthy:
                if attempt == 0:
                    reason = "worker_missing" if not worker else "worker_unhealthy"
                    await model_manager.unload(worker_key, reason=reason)
                    await model_manager.load(worker_key)
                    await model_manager.touch_last_request_at(worker_key)
                    continue
                raise HTTPException(
                    status_code=503,
                    detail=f"Upscaler worker unavailable for model '{worker_key}'",
                )

            try:
                body, stats = await backend.upscale_video(
                    state.backend_model_id,
                    payload,
                    target_height=target_height,
                    crf=crf,
                )
                break
            except httpx.TransportError as exc:
                logger.warning(
                    "upscale_worker_transport_error",
                    model_id=worker_key, attempt=attempt + 1, error=str(exc),
                )
                if attempt == 0:
                    await model_manager.unload(worker_key, reason="worker_transport_error")
                    await model_manager.load(worker_key)
                    await model_manager.touch_last_request_at(worker_key)
                    continue
                raise HTTPException(
                    status_code=503,
                    detail=f"Upscaler worker unavailable for model '{worker_key}'",
                ) from exc
            except httpx.HTTPStatusError as exc:
                # El worker ya devuelve un detalle útil (p. ej. 507 por VRAM):
                # se propaga tal cual en vez de enmascararlo con un 500.
                raise HTTPException(
                    status_code=exc.response.status_code,
                    detail=exc.response.text[:500],
                ) from exc
    finally:
        model_manager.end_request(worker_key, inflight_request_id)

    # Las estadísticas del worker (fps, VRAM, resolución) se reenvían: son lo
    # único que permite a un cliente saber qué le costó de verdad su segmento.
    return Response(content=body, media_type="video/mp4", headers=stats)
