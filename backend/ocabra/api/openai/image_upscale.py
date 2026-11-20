"""POST /v1/images/upscale — super-resolution of one still image.

Not an OpenAI endpoint (OpenAI has none for this), but it lives under ``/v1``
next to the other image routes so a client that already talks to oCabra with
an API key gets it with the same auth, model names and stats. It reuses the
video upscaler models (``realesrgan-x4plus``, ``ultrasharp-x4``…): the worker
runs the same network per tile, with no video codec in the way.

``/v1/images/edits`` is the wrong door for this on purpose: it is img2img with
a prompt and rejects these models.
"""

from __future__ import annotations

import base64
import time
from typing import Annotated, Any

import httpx
import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from ocabra.api._deps_auth import UserContext
from ocabra.config import settings
from ocabra.core.admission import EndpointInflight

from ._deps import (
    NON_STREAMING_LOAD_RESPONSE_DOC,
    _openai_error,
    build_model_status_headers,
    check_capability,
    get_federation_manager,
    get_model_manager,
    get_openai_user,
    get_profile_registry,
    raise_upstream_http_error,
    resolve_profile,
)
from ._federation import try_proxy_multipart

router = APIRouter()
logger = structlog.get_logger(__name__)

CAPABILITY = "image_upscaling"
# Backends whose adapter implements ``upscale_image``. Used to turn a request
# for, say, a chat model into a 400 *before* loading it: the capability itself
# is only known for sure once the model is loaded.
_IMAGE_UPSCALE_BACKENDS = frozenset({"upscaler"})
_OUTPUT_FORMATS = {"png": "png", "jpeg": "jpeg", "jpg": "jpeg", "webp": "webp"}
_RESPONSE_FORMATS = {"b64_json"}
# Worker errors that are the client's fault, passed on with their own status.
_CLIENT_ERROR_STATUSES = {400, 413, 422}


def _resolution(value: str | None) -> tuple[int, int] | None:
    """``"1920x1080"`` → ``(1920, 1080)``; None if it does not parse."""
    try:
        w, h = (int(part) for part in str(value).lower().split("x", 1))
    except (TypeError, ValueError):
        return None
    return (w, h) if w > 0 and h > 0 else None


def upscale_work_meta(stats: dict[str, str]) -> dict[str, Any]:
    """Size of the work, from the worker's ``x-ocabra-*`` headers, for stats.

    ``model_mp`` (what the network produced, before any reduction) is what the
    time depends on: a 1080p photo took ~10 s and a 4K one ~45 s, and with only
    ``n_images`` the two looked the same.
    """
    lower = {k.lower(): v for k, v in stats.items()}
    meta: dict[str, Any] = {}
    for prefix, key in (
        ("input", "input-resolution"),
        ("model", "model-resolution"),
        ("output", "delivered-resolution"),
    ):
        size = _resolution(lower.get(f"x-ocabra-{key}"))
        if size:
            meta[f"{prefix}_w"], meta[f"{prefix}_h"] = size
            meta[f"{prefix}_mp"] = round(size[0] * size[1] / 1_000_000, 3)
    try:
        tiles = int(lower.get("x-ocabra-tiles", ""))
    except ValueError:
        tiles = 0
    if tiles > 0:
        meta["tiles"] = tiles
    return meta


def _invalid(message: str, param: str, code: str, status_code: int = 400) -> HTTPException:
    """Client error with a ``code``: VHS could only tell a 413 by its status."""
    return _openai_error(
        message, "invalid_request_error", param=param, code=code, status_code=status_code
    )


# Worker status -> code: 413 is the input pixel cap, 400 an image it cannot
# read or an output too big for the format, 422 a bad numeric field.
_WORKER_ERROR_CODES = {400: "invalid_image", 413: "image_too_large", 422: "invalid_value"}


def _worker_detail(exc: httpx.HTTPStatusError) -> str:
    try:
        payload = exc.response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict) and isinstance(payload.get("detail"), str):
        return payload["detail"]
    return (exc.response.text or "").strip()[:500] or "Invalid image"


async def _reject_incapable_before_loading(
    request: Request, model_id: str, user: UserContext
) -> None:
    """400 ``model_not_capable`` for a plain profile of another backend.

    Without this a request naming a 20 GB chat model would load it only to be
    refused afterwards. Routers, unknown names and models the caller cannot
    see are left to :func:`resolve_profile` (routers filter their targets by
    capability; the others answer 404 there, without leaking existence).
    """
    profile_registry = get_profile_registry(request)
    model_manager = get_model_manager(request)
    try:
        profile = await profile_registry.get(model_id)
        if profile is not None:
            if not getattr(profile, "base_model_id", None):
                return
            resolver = getattr(request.app.state, "router_resolver", None)
            if resolver is not None and resolver.is_router(profile):
                return
            ids: tuple[str, ...] = (profile.profile_id, profile.base_model_id)
        else:
            # A canonical id (``upscaler/realesrgan-x4plus``).
            ids = (model_id,)
        if not user.can_access_model(*ids):
            return
        state = await model_manager.get_state(ids[-1])
    except Exception:  # noqa: BLE001 -- best-effort pre-check; resolve_profile decides
        return
    backend_type = getattr(state, "backend_type", None)
    if isinstance(backend_type, str) and backend_type not in _IMAGE_UPSCALE_BACKENDS:
        raise _openai_error(
            f"The model '{model_id}' does not support image upscaling.",
            "invalid_request_error",
            param="model",
            code="model_not_capable",
        )


@router.post(
    "/images/upscale",
    summary="Upscale an image",
    responses={
        **NON_STREAMING_LOAD_RESPONSE_DOC,
        200: {
            **NON_STREAMING_LOAD_RESPONSE_DOC[200],
            "content": {
                "image/png": {},
                "image/jpeg": {},
                "image/webp": {},
                "application/json": {},
            },
        },
    },
)
async def image_upscale(
    request: Request,
    user: Annotated[UserContext, Depends(get_openai_user)],
    image: Annotated[UploadFile | None, File(description="Image to upscale")] = None,
    model: Annotated[str | None, Form(description="Upscaler model or router")] = None,
    scale: Annotated[float | None, Form(description="Total factor, up to the model's")] = None,
    target_width: Annotated[int | None, Form(description="Max output width (px)")] = None,
    target_height: Annotated[int | None, Form(description="Max output height (px)")] = None,
    output_format: Annotated[str | None, Form(description="png (default), jpeg, webp")] = None,
    output_quality: Annotated[int | None, Form(description="1-100 for jpeg/webp")] = None,
    response_format: Annotated[str | None, Form(description="Omit for bytes, or b64_json")] = None,
) -> Any:
    """Upscale (super-resolve) one photo with an upscaler model.

    ``multipart/form-data`` with:

    * ``image`` — the photo (PNG, JPEG, WebP, BMP, TIFF… anything Pillow
      reads; EXIF orientation is applied). Required. Up to
      ``OPENAI_IMAGE_MAX_PART_SIZE_MB`` (25 MB by default) and, after
      decoding, up to 3840x2160 px of input (worker setting
      ``UPSCALER_IMAGE_MAX_PIXELS``).
    * ``model`` — an upscaler model (``realesrgan-x4plus``,
      ``ultrasharp-x4``, ``remacri-x4``, ``real-esrgan-compact-x4``,
      ``ultrasharp-v2-lite``…) or a router. Required.
    * ``scale`` — optional total factor over the input (e.g. ``2``). The model
      always runs at its native factor (4x for these) and the result is then
      reduced with Lanczos; a value above the native factor is capped at it.
    * ``target_width`` / ``target_height`` — optional bounding box for the
      output, aspect ratio kept. Like ``scale`` they only ever *reduce* the
      model's output: asking for more than the model produced returns what it
      produced (check ``X-Ocabra-Delivered-Resolution``), never a fake
      upscale. When several are given the smallest result wins.
    * ``output_format`` — ``png`` (default, lossless, keeps transparency),
      ``jpeg`` (``jpg`` accepted; transparency dropped) or ``webp`` (keeps
      transparency, at most 16383 px per side).
    * ``output_quality`` — 1-100 for jpeg/webp (default 95). Ignored for png.
    * ``response_format`` — omit it to get the image bytes; ``b64_json``
      returns OpenAI-style JSON ``{"created", "data": [{"b64_json",
      "output_format", "size"}]}``.

    Transparency: the RGB goes through the network and the alpha channel is
    resized separately with Lanczos (these networks are 3-channel).

    Returns:
        By default the image itself with ``Content-Type`` ``image/png``,
        ``image/jpeg`` or ``image/webp``; with ``response_format=b64_json``,
        ``application/json``. Both carry the worker's stats headers
        (``X-Ocabra-Input-Resolution``, ``-Model-Resolution``,
        ``-Delivered-Resolution``, ``-Delivered-Width``, ``-Delivered-Height``,
        ``-Scale``, ``-Tiles``, ``-Alpha``, ``-Output-Format``, ``-Seconds``)
        plus the usual model status / cold-start headers. Recorded in stats as
        ``image_upscale``.

    Raises:
        HTTPException 400: with ``error.code`` ``missing_required_parameter``
            (no ``image``/``model``), ``empty_image``, ``invalid_value``
            (out-of-range ``scale``/``target_*``/``output_quality``),
            ``invalid_output_format``, ``invalid_response_format``,
            ``invalid_image`` (unreadable, or output too large for the chosen
            format) or ``model_not_capable`` when the model is not an upscaler
            (FlashVSR included: it only upscales video).
        HTTPException 404: unknown model, or not accessible to the caller.
        HTTPException 413: ``file_too_large`` (upload above the size limit) or
            ``image_too_large`` (more input pixels than the worker accepts).
        HTTPException 422: a numeric field that is not a number.
        HTTPException 503: upscaler worker unavailable (``Retry-After``).
        HTTPException 507: not enough VRAM even with the smallest tiles.
    """
    if image is None:
        raise _invalid("The 'image' field is required.", "image", "missing_required_parameter")
    model_id = str(model or "").strip()
    if not model_id:
        raise _invalid("The 'model' field is required.", "model", "missing_required_parameter")
    fmt = _OUTPUT_FORMATS.get(str(output_format or "png").strip().lower())
    if fmt is None:
        raise _invalid(
            f"Invalid output_format '{output_format}'. Must be one of: png, jpeg, webp.",
            "output_format",
            "invalid_output_format",
        )
    resp_format = str(response_format).strip().lower() if response_format else None
    if resp_format is not None and resp_format not in _RESPONSE_FORMATS:
        raise _invalid(
            f"Invalid response_format '{response_format}'. Omit it for the raw image "
            "or use 'b64_json'.",
            "response_format",
            "invalid_response_format",
        )
    if scale is not None and not 0 < scale <= 16:
        raise _invalid("'scale' must be greater than 0 and at most 16.", "scale", "invalid_value")
    for name, value in (("target_width", target_width), ("target_height", target_height)):
        if value is not None and value <= 0:
            raise _invalid(
                f"'{name}' must be a positive number of pixels.", name, "invalid_value"
            )
    if output_quality is not None and not 1 <= output_quality <= 100:
        raise _invalid(
            "'output_quality' must be between 1 and 100.", "output_quality", "invalid_value"
        )

    max_mb = max(1, int(settings.openai_image_max_part_size_mb))
    payload = await image.read()
    if not payload:
        raise _invalid("The uploaded image is empty.", "image", "empty_image")
    if len(payload) > max_mb * 1024 * 1024:
        raise _invalid(
            f"The image exceeds the {max_mb} MB upload limit.", "image", "file_too_large", 413
        )

    model_manager = get_model_manager(request)
    profile_registry = get_profile_registry(request)
    filename = image.filename or "image"
    content_type = image.content_type or "application/octet-stream"

    form: dict[str, str] = {"model": model_id, "output_format": fmt}
    for name, value in (
        ("scale", scale),
        ("target_width", target_width),
        ("target_height", target_height),
        ("output_quality", output_quality),
        ("response_format", resp_format),
    ):
        if value is not None:
            form[name] = str(value)
    fed_resp = await try_proxy_multipart(
        request,
        model_id=model_id,
        files={"image": (filename, payload, content_type)},
        data=form,
        federation_manager=get_federation_manager(request),
        model_manager=model_manager,
        profile_registry=profile_registry,
        logger=logger,
        required_capability=CAPABILITY,
    )
    if fed_resp is not None:
        return Response(
            content=fed_resp.content,
            status_code=fed_resp.status_code,
            media_type=fed_resp.headers.get("content-type"),
            headers={
                k: v for k, v in fed_resp.headers.items() if k.lower().startswith("x-ocabra-")
            },
        )

    await _reject_incapable_before_loading(request, model_id, user)
    profile, state = await resolve_profile(
        model_id,
        model_manager,
        profile_registry,
        user=user,
        router_resolver=getattr(request.app.state, "router_resolver", None),
        request_state=request.state,
        required_capability=CAPABILITY,
    )
    check_capability(state, CAPABILITY, "image upscaling")
    pre_status = state.status.value
    started = time.monotonic()

    # El worker que resolve_profile ha cargado: su clave incluye el preset de
    # runtime, que compute_worker_key(base, load_overrides) no ve.
    worker_key = state.model_id
    # Multipart: the stats middleware has no JSON ``model`` to read.
    request.state.stats_model_id = profile.profile_id

    worker_pool = request.app.state.worker_pool
    inflight = EndpointInflight(request, model_manager, worker_key, user=user)
    inflight.acquire()
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
                raise _openai_error(
                    f"Upscaler worker unavailable for model '{model_id}'.",
                    "server_error",
                    code="worker_unavailable",
                    status_code=503,
                    headers={"Retry-After": "10"},
                )
            try:
                body, media_type, stats = await backend.upscale_image(
                    state.backend_model_id,
                    payload,
                    filename=filename,
                    content_type=content_type,
                    scale=scale,
                    target_width=target_width,
                    target_height=target_height,
                    output_format=fmt,
                    quality=output_quality,
                )
                break
            except httpx.TransportError as exc:
                logger.warning(
                    "image_upscale_worker_transport_error",
                    model_id=worker_key,
                    attempt=attempt + 1,
                    error=str(exc),
                )
                if attempt == 0:
                    await model_manager.unload(worker_key, reason="worker_transport_error")
                    await model_manager.load(worker_key)
                    await model_manager.touch_last_request_at(worker_key)
                    continue
                raise _openai_error(
                    f"Upscaler worker unavailable for model '{model_id}'.",
                    "server_error",
                    code="worker_unavailable",
                    status_code=503,
                    headers={"Retry-After": "10"},
                ) from exc
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status in _CLIENT_ERROR_STATUSES:
                    raise _invalid(
                        _worker_detail(exc), "image", _WORKER_ERROR_CODES[status], status
                    ) from exc
                raise_upstream_http_error(exc)
    finally:
        inflight.release()

    headers = await build_model_status_headers(
        model_manager,
        worker_key,
        profile.base_model_id,
        pre_status=pre_status,
        elapsed_ms=int((time.monotonic() - started) * 1000),
    )
    headers.update(stats)
    # Bytes delivered too: the same photo weighed 10.3 MB in PNG and 0.7 MB
    # in WebP (VHS, 20/11).
    request.state.upscale_meta = {
        **upscale_work_meta(stats),
        "output_bytes": len(body),
        "output_format": fmt,
    }
    if resp_format == "b64_json":
        size = next(
            (v for k, v in stats.items() if k.lower() == "x-ocabra-delivered-resolution"), None
        )
        item: dict[str, Any] = {
            "b64_json": base64.b64encode(body).decode("ascii"),
            "output_format": fmt,
        }
        if size:
            item["size"] = size
        return JSONResponse(content={"created": int(time.time()), "data": [item]}, headers=headers)
    return Response(content=body, media_type=media_type or f"image/{fmt}", headers=headers)
