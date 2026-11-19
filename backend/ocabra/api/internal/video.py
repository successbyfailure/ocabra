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

import time
from datetime import UTC, datetime
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
from ocabra.config import settings
from ocabra.stats.collector import record_direct_request

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["video"])

_READ_CHUNK_BYTES = 1024 * 1024

# Stats headers the upscale workers (upscaler, FlashVSR) emit and that are
# forwarded to the client. Anything else the worker answers with
# (content-length, server, date, a future debug header...) is dropped: the
# response is rebuilt here and must not inherit framing or internal headers.
_FORWARDED_STATS_HEADERS = frozenset(
    {
        "x-ocabra-frames",
        "x-ocabra-input-resolution",
        "x-ocabra-model-resolution",
        "x-ocabra-delivered-height",
        "x-ocabra-seconds",
        "x-ocabra-fps",
        "x-ocabra-vram-peak-mb",
    }
)
_MAX_HEADER_VALUE_LEN = 128


def _forwarded_headers(stats: dict) -> dict[str, str]:
    """Whitelisted worker stats headers, with values safe to re-emit."""
    out: dict[str, str] = {}
    for key, value in (stats or {}).items():
        name = str(key).lower()
        if name not in _FORWARDED_STATS_HEADERS:
            continue
        text = str(value)
        if any(ch in text for ch in "\r\n\x00") or len(text) > _MAX_HEADER_VALUE_LEN:
            continue
        out[name] = text
    return out


async def _read_capped(file: UploadFile, max_bytes: int) -> bytes:
    """Read the upload in chunks and stop with 413 as soon as it passes the cap.

    ``file.size`` (known once Starlette has spooled the part) rejects without
    reading at all; otherwise the chunked read never holds more than
    ``max_bytes`` + one chunk in memory.
    """
    too_large = HTTPException(
        status_code=413,
        detail=f"Video segment exceeds the {max_bytes // (1024 * 1024)} MB upload limit",
    )
    if file.size is not None and file.size > max_bytes:
        raise too_large
    buffer = bytearray()
    while chunk := await file.read(_READ_CHUNK_BYTES):
        buffer.extend(chunk)
        if len(buffer) > max_bytes:
            raise too_large
    return bytes(buffer)


def _frames_from(stats: dict) -> int | None:
    """Frame count the upscale worker reports (``x-ocabra-frames``)."""
    for key, value in (stats or {}).items():
        if key.lower() == "x-ocabra-frames":
            try:
                frames = int(float(value))
            except (TypeError, ValueError):
                return None
            return frames if frames > 0 else None
    return None


@router.post(
    "/video/upscale",
    summary="Upscale a video segment",
    description=(
        "Upscale/restore one self-contained video segment. Send video only "
        "(no audio) and reassemble on the client side. Returns video/mp4. "
        "Segments above VIDEO_UPSCALE_MAX_UPLOAD_MB (default 512 MB) get 413."
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
    """Upscale/restore one self-contained video segment.

    Any authenticated user (role ``user``) with access to the model; routers are
    resolved like on ``/v1``. The worker is reloaded once if it is missing,
    unhealthy or drops the connection. The request is recorded in stats as
    ``video_upscale``.

    Args:
        file: Video segment without audio (multipart).
        model: Upscaler model id or router (needs ``video_upscaling``).
        target_height: Optional output height in pixels.
        crf: Encoder CRF of the output (default 14).

    Returns:
        The upscaled segment as ``video/mp4`` with the worker's stats headers
        (``x-ocabra-frames``, ``-fps``, ``-seconds``, ``-input-resolution``,
        ``-model-resolution``, ``-delivered-height``, ``-vram-peak-mb``); no
        other worker header is forwarded.

    Raises:
        HTTPException 400: empty segment, or the model lacks the capability.
        HTTPException 413: segment larger than ``VIDEO_UPSCALE_MAX_UPLOAD_MB``.
        HTTPException 503: the upscaler worker is unavailable (``Retry-After``).
        HTTPException: the worker's own error status (e.g. 507 for VRAM) is
            passed through.
    """
    max_bytes = max(1, int(settings.video_upscale_max_upload_mb)) * 1024 * 1024
    payload = await _read_capped(file, max_bytes)
    if not payload:
        raise HTTPException(status_code=400, detail="Empty video segment")

    model_manager = get_model_manager(request)
    profile_registry = get_profile_registry(request)

    profile, state = await resolve_profile(
        model,
        model_manager,
        profile_registry,
        user=_user,
        # Routers resolve here too, as on every /v1 endpoint (BUG-20).
        router_resolver=getattr(request.app.state, "router_resolver", None),
        request_state=request.state,
        required_capability="video_upscaling",
    )
    check_capability(state, "video_upscaling", "video upscaling")

    worker_key = compute_worker_key(profile.base_model_id, profile.load_overrides)
    request.state.stats_model_id = worker_key

    worker_pool = request.app.state.worker_pool
    started_at = datetime.now(UTC)
    started = time.monotonic()
    status_code, error = 200, None
    work: dict = {"input_bytes": len(payload)}
    inflight_request_id = model_manager.begin_request(worker_key, user=_user)
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
                    headers={"Retry-After": "10"},
                )

            try:
                body, stats = await backend.upscale_video(
                    state.backend_model_id,
                    payload,
                    target_height=target_height,
                    crf=crf,
                )
                frames = _frames_from(stats)
                if frames:
                    # What the duration estimator and the Stats speed column
                    # read (frames per second).
                    work["input_frames"] = frames
                break
            except httpx.TransportError as exc:
                logger.warning(
                    "upscale_worker_transport_error",
                    model_id=worker_key,
                    attempt=attempt + 1,
                    error=str(exc),
                )
                if attempt == 0:
                    await model_manager.unload(worker_key, reason="worker_transport_error")
                    await model_manager.load(worker_key)
                    await model_manager.touch_last_request_at(worker_key)
                    continue
                raise HTTPException(
                    status_code=503,
                    detail=f"Upscaler worker unavailable for model '{worker_key}'",
                    headers={"Retry-After": "10"},
                ) from exc
            except httpx.HTTPStatusError as exc:
                # El worker ya devuelve un detalle útil (p. ej. 507 por VRAM):
                # se propaga tal cual en vez de enmascararlo con un 500.
                raise HTTPException(
                    status_code=exc.response.status_code,
                    detail=exc.response.text[:500],
                ) from exc
    except HTTPException as exc:
        status_code, error = exc.status_code, str(exc.detail)[:500]
        raise
    except Exception as exc:
        status_code, error = 500, str(exc)[:500]
        raise
    finally:
        model_manager.end_request(worker_key, inflight_request_id)
        # Not under /v1, so the stats middleware never saw these (BUG-20).
        try:
            await record_direct_request(
                request,
                model_id=worker_key,
                started_at=started_at,
                duration_ms=(time.monotonic() - started) * 1000,
                endpoint_path="/ocabra/video/upscale",
                request_kind="video_upscale",
                status_code=status_code,
                error_message=error,
                work_size_meta=work,
            )
        except Exception as exc:  # noqa: BLE001 -- stats must not break the response
            logger.warning("video_upscale_stats_failed", error=str(exc))

    # Las estadísticas del worker (fps, VRAM, resolución) se reenvían: son lo
    # único que permite a un cliente saber qué le costó de verdad su segmento.
    return Response(content=body, media_type="video/mp4", headers=_forwarded_headers(stats))
