"""``/v1/videos``: generación de vídeo asíncrona, compatible con la API de OpenAI.

``POST /v1/videos`` encola y devuelve el objeto ``video`` en ``queued``; el
cliente consulta ``GET /v1/videos/{id}`` (``status`` y ``progress``) y, en
``completed``, descarga ``GET /v1/videos/{id}/content``. La ejecución vive en
``core/video_jobs.py``.

Diferencias con OpenAI que conviene saber:

* ``seconds`` admite cualquier entero entre 1 y el tope del modelo (10 s por
  defecto), no solo 4/8/12.
* ``size`` se redondea a múltiplos de 64 (LTX-2.5 genera a media resolución
  y escala x2): ``1280x720`` sale ``1280x704``. El objeto devuelto lleva el
  tamaño real.
* El vídeo trae audio generado por el modelo (LTX-2.5 genera vídeo y sonido).
* ``seed`` (extensión) fija la semilla; el objeto devuelve la usada.
"""

from __future__ import annotations

import asyncio
import base64
import tempfile
import uuid
from pathlib import Path
from typing import Annotated, Any

import httpx
import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response
from sqlalchemy import desc, select

from ocabra.api._deps_auth import UserContext
from ocabra.api.openai._deps import (
    _openai_error,
    get_model_manager,
    get_openai_user,
    get_profile_registry,
)
from ocabra.api.openai._media_urls import url_problem
from ocabra.config import settings
from ocabra.core.image_header import image_size
from ocabra.core.model_manager import declared_capabilities
from ocabra.core.video_jobs import PendingVideo, normalize_size
from ocabra.database import AsyncSessionLocal
from ocabra.db.videos import VideoJob

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["OpenAI"])

_ID_PREFIX = "video_"


def _manager(request: Request):
    manager = getattr(request.app.state, "video_job_manager", None)
    if manager is None:
        raise _openai_error(
            "Video generation is not available on this node.",
            "server_error",
            code="video_unavailable",
            status_code=503,
        )
    return manager


def _parse_id(video_id: str) -> uuid.UUID:
    raw = video_id.removeprefix(_ID_PREFIX)
    try:
        return uuid.UUID(hex=raw) if len(raw) == 32 else uuid.UUID(raw)
    except ValueError:
        raise _not_found(video_id) from None


def _not_found(video_id: str) -> HTTPException:
    return _openai_error(
        f"No video found with id '{video_id}'.",
        "invalid_request_error",
        param="video_id",
        code="video_not_found",
        status_code=404,
    )


def _sees_all(user: UserContext) -> bool:
    return user.role in ("model_manager", "system_admin")


async def _owned_job(video_id: str, user: UserContext) -> VideoJob:
    job_id = _parse_id(video_id)
    async with AsyncSessionLocal() as session:
        job = await session.get(VideoJob, job_id)
    owner = str(job.user_id) if job is not None and job.user_id else None
    if job is None or (owner != user.user_id and not _sees_all(user)):
        raise _not_found(video_id)
    return job


def _bad(message: str, param: str) -> HTTPException:
    return _openai_error(message, "invalid_request_error", param=param, code="invalid_value")


# Campos de imagen: ``input_reference`` (OpenAI: primer fotograma) y
# ``last_frame`` (extensión: último fotograma).
_IMAGE_FIELDS = ("input_reference", "last_frame")


async def _read_body(request: Request) -> dict[str, Any]:
    """Cuerpo como dict; las imágenes ya leídas como ``bytes``."""
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("multipart/form-data") or content_type.startswith(
        "application/x-www-form-urlencoded"
    ):
        form = await request.form()
        body: dict[str, Any] = {}
        for key, value in form.items():
            if isinstance(value, str):
                body[key] = value
            elif key in _IMAGE_FIELDS:
                body[key] = await _read_upload(value, key)
        return body
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        raise _bad("The body must be JSON or multipart/form-data.", "body") from None
    if not isinstance(body, dict):
        raise _bad("The body must be a JSON object.", "body")
    for key in _IMAGE_FIELDS:
        if body.get(key):
            body[key] = await _read_image_ref(body[key], key)
    return body


def _max_image_bytes() -> int:
    return max(1, int(settings.video_max_image_mb)) * 1024 * 1024


async def _read_upload(upload: Any, param: str) -> bytes:
    limit = _max_image_bytes()
    data = await upload.read(limit + 1)
    if len(data) > limit:
        raise _bad(f"'{param}' is larger than {settings.video_max_image_mb} MB.", param)
    return data


async def _read_image_ref(value: Any, param: str) -> bytes:
    """Imagen dada como data URL, URL http(s) o ``{"image_url": ...}``."""
    if isinstance(value, dict):
        value = value.get("image_url") or value.get("url")
        if isinstance(value, dict):
            value = value.get("url")
    if not isinstance(value, str) or not value.strip():
        raise _bad(f"'{param}' must be an image file, a data: URL or an http(s) URL.", param)
    value = value.strip()
    limit = _max_image_bytes()
    if value.startswith("data:"):
        _, _, payload = value.partition(",")
        try:
            data = base64.b64decode(payload, validate=True)
        except ValueError:
            raise _bad(f"'{param}' is not valid base64.", param) from None
        if len(data) > limit:
            raise _bad(f"'{param}' is larger than {settings.video_max_image_mb} MB.", param)
        return data
    if value.startswith(("http://", "https://")):
        # Misma regla que las imágenes de chat: nada de direcciones privadas.
        problem = await url_problem(value)
        if problem:
            raise _bad(f"'{param}': {problem}", param)
        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=False) as client:
                async with client.stream("GET", value) as response:
                    response.raise_for_status()
                    chunks, total = [], 0
                    async for chunk in response.aiter_bytes():
                        total += len(chunk)
                        if total > limit:
                            raise _bad(
                                f"'{param}' is larger than {settings.video_max_image_mb} MB.",
                                param,
                            )
                        chunks.append(chunk)
                    return b"".join(chunks)
        except httpx.HTTPError as exc:
            raise _bad(f"Could not download '{param}': {exc}", param) from None
    raise _bad(f"'{param}' must be an image file, a data: URL or an http(s) URL.", param)


def _size_for_image(width: int, height: int) -> tuple[int, int]:
    """Tamaño de vídeo con la proporción de la imagen y los píxeles por defecto."""
    w_str, h_str = str(settings.video_default_size).lower().split("x")
    pixels = min(int(w_str) * int(h_str), int(settings.video_max_pixels))
    ratio = width / height
    out_w = (pixels * ratio) ** 0.5
    out_h = out_w / ratio
    w, h = normalize_size(int(out_w), int(out_h))
    while w * h > int(settings.video_max_pixels):
        w, h = normalize_size(int(w * 0.95), int(h * 0.95))
    return w, h


def _parse_params(body: dict[str, Any]) -> dict[str, Any]:
    prompt = str(body.get("prompt") or "").strip()
    if not prompt:
        raise _bad("'prompt' is required.", "prompt")
    try:
        seconds = int(str(body.get("seconds") or settings.video_default_seconds))
    except ValueError:
        raise _bad("'seconds' must be an integer number of seconds.", "seconds") from None
    max_seconds = int(settings.video_max_seconds)
    if not 1 <= seconds <= max_seconds:
        raise _bad(f"'seconds' must be between 1 and {max_seconds}.", "seconds")
    first = body.get("input_reference")
    last = body.get("last_frame")
    first_size = None
    for key, data in (("input_reference", first), ("last_frame", last)):
        if data is None or data == "":
            continue
        if not isinstance(data, bytes) or image_size(data) is None:
            raise _bad(f"'{key}' must be a PNG, JPEG, WebP or GIF image.", key)
        if key == "input_reference":
            first_size = image_size(data)
    if last and not first:
        raise _bad("'last_frame' needs 'input_reference' as the first frame.", "last_frame")
    default_size = settings.video_default_size
    if first_size and not body.get("size"):
        default_size = "{}x{}".format(*_size_for_image(*first_size))
    size = str(body.get("size") or default_size).lower().replace(" ", "")
    try:
        w_str, h_str = size.split("x")
        width, height = normalize_size(int(w_str), int(h_str))
    except ValueError:
        raise _bad("'size' must look like '1280x720'.", "size") from None
    if width * height > int(settings.video_max_pixels):
        raise _bad(
            f"'size' too large: at most {settings.video_max_pixels} pixels per frame.", "size"
        )
    seed = body.get("seed")
    try:
        seed = int(seed) if seed not in (None, "") else None
    except (TypeError, ValueError):
        raise _bad("'seed' must be an integer.", "seed") from None
    params: dict[str, Any] = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "seconds": seconds,
        "fps": 24,
        "seed": seed,
    }
    negative = str(body.get("negative_prompt") or "").strip()
    if negative:
        params["negative_prompt"] = negative
    # Las imágenes viajan al worker en base64 y solo viven en memoria.
    params["mode"] = "flf2v" if last else ("i2v" if first else "t2v")
    if first:
        params["image_b64"] = base64.b64encode(first).decode("ascii")
    if last:
        params["last_image_b64"] = base64.b64encode(last).decode("ascii")
    return params


@router.post("/videos", summary="Create a video")
async def create_video(
    request: Request,
    user: Annotated[UserContext, Depends(get_openai_user)],
) -> dict[str, Any]:
    """Queue a video generation (asynchronous, OpenAI-compatible).

    Body (JSON or multipart/form-data):
        model           — video model or profile (required, e.g. ``ltx-2.5``)
        prompt          — description of the scene, including sound (required)
        seconds         — duration, 1..``VIDEO_MAX_SECONDS`` (default 5)
        size            — ``WxH``; rounded to multiples of 64 (default 1280x704)
        negative_prompt — what to avoid (extension)
        input_reference — image used as the first frame (image-to-video): a file
                          (multipart), a ``data:`` URL, an ``http(s)`` URL or
                          ``{"image_url": ...}``. Without ``size`` the video
                          takes the image's aspect ratio.
        last_frame      — image for the last frame (extension, needs
                          ``input_reference``): the video goes from one to
                          the other
        seed            — fixed seed (extension); random when omitted

    Returns:
        The ``video`` object in ``queued``. Poll ``GET /v1/videos/{id}`` until
        ``status`` is ``completed`` or ``failed`` and then download
        ``GET /v1/videos/{id}/content``.

    Raises:
        400 for invalid parameters or a model that does not generate video,
        404 for an unknown or inaccessible model, 403 on zero-data nodes.
    """
    manager = _manager(request)
    body = await _read_body(request)
    model = str(body.get("model") or "").strip()
    if not model:
        raise _bad("'model' is required.", "model")
    params = _parse_params(body)

    # Validación rápida, sin cargar el modelo: existe, el usuario lo ve y
    # genera vídeo. La carga la hace el trabajo cuando le llega el turno.
    registry = get_profile_registry(request)
    model_manager = get_model_manager(request)
    profile = await registry.get(model)
    base_id = profile.base_model_id if profile is not None else model
    state = await model_manager.get_state(base_id) if base_id else None
    if (
        (profile is None and state is None)
        or (profile is not None and not profile.enabled)
        or not user.can_access_model(model, base_id)
    ):
        raise _openai_error(
            f"The model '{model}' does not exist.",
            "invalid_request_error",
            param="model",
            code="model_not_found",
            status_code=404,
        )
    # Las capacidades de un modelo sin cargar están vacías: las que declara
    # su backend sirven para validar antes de la primera carga.
    if state is not None and not (
        getattr(state.capabilities, "video_generation", False)
        or declared_capabilities(state).video_generation
    ):
        raise _openai_error(
            f"The model '{model}' does not generate video.",
            "invalid_request_error",
            param="model",
            code="model_not_capable",
        )

    job = VideoJob(
        id=uuid.uuid4(),
        user_id=uuid.UUID(user.user_id) if user.user_id else None,
        model=model,
        status="queued",
        progress=0,
        seconds=params["seconds"],
        width=params["width"],
        height=params["height"],
        seed=params["seed"],
    )
    async with AsyncSessionLocal() as session:
        session.add(job)
        await session.commit()
        await session.refresh(job)
    await manager.submit(
        PendingVideo(job_id=job.id, model=model, params=params, user=user, request=request)
    )
    logger.info(
        "video_job_queued",
        job_id=str(job.id),
        model=model,
        **{k: params[k] for k in ("width", "height", "seconds")},
    )
    return job.to_openai_dict()


@router.get("/videos", summary="List videos")
async def list_videos(
    request: Request,
    user: Annotated[UserContext, Depends(get_openai_user)],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    order: Annotated[str, Query(pattern="^(asc|desc)$")] = "desc",
    after: str | None = None,
) -> dict[str, Any]:
    """List the caller's videos (managers and admins see everyone's).

    Args:
        limit: page size (1-100, default 20).
        order: ``desc`` (newest first, default) or ``asc``.
        after: cursor — the id of the last video of the previous page.

    Returns:
        ``{"object": "list", "data": [video...], "has_more": bool}``.
    """
    stmt = select(VideoJob)
    if not _sees_all(user):
        if not user.user_id:
            return {"object": "list", "data": [], "has_more": False}
        stmt = stmt.where(VideoJob.user_id == uuid.UUID(user.user_id))
    async with AsyncSessionLocal() as session:
        if after:
            anchor = await session.get(VideoJob, _parse_id(after))
            if anchor is not None:
                cmp = VideoJob.created_at < anchor.created_at
                if order == "asc":
                    cmp = VideoJob.created_at > anchor.created_at
                stmt = stmt.where(cmp)
        column = VideoJob.created_at if order == "asc" else desc(VideoJob.created_at)
        rows = (await session.execute(stmt.order_by(column).limit(limit + 1))).scalars().all()
    data = [row.to_openai_dict() for row in rows[:limit]]
    return {"object": "list", "data": data, "has_more": len(rows) > limit}


@router.get("/videos/{video_id}", summary="Retrieve a video")
async def retrieve_video(
    video_id: str,
    request: Request,
    user: Annotated[UserContext, Depends(get_openai_user)],
) -> dict[str, Any]:
    """Return the ``video`` object: ``status`` (queued, in_progress,
    completed, failed), ``progress`` (0-100) and, when failed, ``error``.

    Extension: ``queue_position`` (0 = generating now) while it waits.
    """
    job = await _owned_job(video_id, user)
    payload = job.to_openai_dict()
    if job.status in ("queued", "in_progress"):
        position = _manager(request).queue_position(job.id)
        if position is not None:
            payload["queue_position"] = position
    return payload


@router.get("/videos/{video_id}/content", summary="Download video content")
async def download_video(
    video_id: str,
    user: Annotated[UserContext, Depends(get_openai_user)],
    variant: Annotated[str, Query(pattern="^(video|thumbnail)$")] = "video",
) -> Response:
    """Download the MP4 (``variant=video``, H.264 + AAC) or a JPEG thumbnail
    of its first frame (``variant=thumbnail``).

    Raises:
        404 unknown video; 409 if it is not ``completed`` yet; 410 if expired.
    """
    job = await _owned_job(video_id, user)
    if job.status != "completed":
        raise _openai_error(
            f"Video '{video_id}' is {job.status}; content is available once it is completed.",
            "invalid_request_error",
            code="video_not_ready",
            status_code=409,
        )
    path = Path(job.storage_path or "")
    if not job.storage_path or not await asyncio.to_thread(path.exists):
        raise _openai_error(
            f"Video '{video_id}' has expired.",
            "invalid_request_error",
            code="video_expired",
            status_code=410,
        )
    if variant == "video":
        return FileResponse(path, media_type="video/mp4", filename=f"{job.public_id}.mp4")
    return Response(content=await _thumbnail(path), media_type="image/jpeg")


async def _thumbnail(path: Path) -> bytes:
    with tempfile.TemporaryDirectory(prefix="ocabra_thumb_") as tmp:
        out = Path(tmp) / "thumb.jpg"
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-v", "error", "-y", "-i", str(path), "-frames:v", "1", "-q:v", "3", str(out),
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
        )  # fmt: skip
        _, err = await proc.communicate()
        if proc.returncode != 0 or not out.exists():
            raise _openai_error(
                f"Could not extract a thumbnail: {err.decode(errors='replace')[:200]}",
                "server_error", code="thumbnail_failed", status_code=500,
            )  # fmt: skip
        return out.read_bytes()


@router.delete("/videos/{video_id}", summary="Delete a video")
async def delete_video(
    video_id: str,
    user: Annotated[UserContext, Depends(get_openai_user)],
) -> dict[str, Any]:
    """Delete a finished video and its file.

    Raises:
        404 unknown video; 409 while it is queued or generating.
    """
    job = await _owned_job(video_id, user)
    if job.status in ("queued", "in_progress"):
        raise _openai_error(
            f"Video '{video_id}' is {job.status}; it can be deleted once it finishes.",
            "invalid_request_error", code="video_in_progress", status_code=409,
        )  # fmt: skip
    if job.storage_path:
        await asyncio.to_thread(Path(job.storage_path).unlink, missing_ok=True)
    async with AsyncSessionLocal() as session:
        row = await session.get(VideoJob, job.id)
        if row is not None:
            await session.delete(row)
            await session.commit()
    return {"id": job.public_id, "object": "video.deleted", "deleted": True}
