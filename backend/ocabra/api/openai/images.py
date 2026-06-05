"""
POST /v1/images/generations — image generation endpoint.
GET  /v1/images/files/{name} — serve a generated image (URL response_format).
"""

from __future__ import annotations

import base64
import binascii
import re
import time
import uuid
from pathlib import Path
from typing import Annotated, Any

import httpx
import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response

from ocabra.api._deps_auth import UserContext
from ocabra.config import settings

from ._deps import (
    _openai_error,
    check_capability,
    compute_worker_key,
    get_federation_manager,
    get_model_manager,
    get_openai_user,
    get_profile_registry,
    merge_profile_defaults,
    raise_upstream_http_error,
    resolve_profile,
)
from ._federation import try_proxy_json, try_proxy_multipart

router = APIRouter()
logger = structlog.get_logger(__name__)

# Allowed values for the OpenAI `response_format` field.
_RESPONSE_FORMATS = {"b64_json", "url"}

# Generated filenames are <uuid>.png — restrict the public route to that shape
# so the file server can never be coerced into reading arbitrary paths.
_IMAGE_NAME_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.png$")


def _ensure_output_dir() -> Path:
    path = Path(settings.image_outputs_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_image(b64_data: str) -> str:
    """Decode a base64 PNG payload and store it. Returns the bare filename."""
    try:
        raw = base64.b64decode(b64_data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(
            status_code=502,
            detail={"error": {
                "message": f"Backend returned invalid base64 image data: {exc}",
                "type": "internal_error",
            }},
        ) from exc

    out_dir = _ensure_output_dir()
    name = f"{uuid.uuid4()}.png"
    (out_dir / name).write_bytes(raw)
    return name


def _build_public_url(request: Request, name: str) -> str:
    """Build an absolute URL for a generated image filename."""
    base = (settings.public_url or "").rstrip("/")
    if not base:
        # Fall back to the request's base URL (works behind proxies that pass
        # X-Forwarded-* and Host correctly).
        base = str(request.base_url).rstrip("/")
    return f"{base}/v1/images/files/{name}"


@router.post("/images/generations", summary="Create image")
async def image_generations(
    request: Request,
    user: Annotated[UserContext, Depends(get_openai_user)],
) -> Any:
    """
    Generate images from a text prompt.
    Proxies to the Diffusers backend worker.
    Requires a model with capability image_generation=True.

    Supports OpenAI size strings: 256x256, 512x512, 1024x1024, 1792x1024, 1024x1792.
    Supports `response_format`: ``b64_json`` (default, preserves existing behavior)
    or ``url`` (image is persisted to disk and a temporary URL is returned).
    """
    body = await request.json()
    model_id: str = body.get("model", "")

    response_format = body.get("response_format", "b64_json")
    if response_format not in _RESPONSE_FORMATS:
        raise HTTPException(
            status_code=400,
            detail={"error": {
                "message": (
                    f"Invalid response_format '{response_format}'. "
                    f"Must be one of: {sorted(_RESPONSE_FORMATS)}"
                ),
                "type": "invalid_request_error",
                "param": "response_format",
            }},
        )

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
    )
    check_capability(state, "image_generation", "image generation")

    merged_body = merge_profile_defaults(profile, body)
    worker_key = compute_worker_key(profile.base_model_id, profile.load_overrides)

    # Translate OpenAI size to width/height
    size_str: str = merged_body.get("size", "1024x1024")
    width, height = _parse_size(size_str)

    worker_body = {
        "prompt": merged_body.get("prompt", ""),
        "negative_prompt": merged_body.get("negative_prompt"),
        "width": width,
        "height": height,
        "num_images": merged_body.get("n", 1),
        "num_inference_steps": merged_body.get("num_inference_steps", 20),
        "guidance_scale": merged_body.get("guidance_scale", 7.5),
        "seed": merged_body.get("seed"),
    }

    worker_pool = request.app.state.worker_pool
    # begin_request marks the model as in-flight so model_manager's idle
    # eviction loop doesn't unload it mid-generation. The stats middleware
    # tracks the same thing but keyed by the user-facing profile_id, which
    # doesn't match the canonical state key — image generation requests can
    # take minutes (sequential offload on contended GPUs), well past the
    # 300s idle_timeout, so we need the canonical-keyed marker too.
    inflight_request_id = model_manager.begin_request(worker_key)
    try:
        result = await worker_pool.forward_request(worker_key, "/generate", worker_body)
    except httpx.HTTPStatusError as exc:
        raise_upstream_http_error(exc)
    finally:
        model_manager.end_request(worker_key, inflight_request_id)

    # Translate back to OpenAI format
    images = result.get("images", [])
    if response_format == "url":
        data = []
        for img in images:
            b64 = img.get("b64_json", "")
            if not b64:
                continue
            name = _save_image(b64)
            data.append({"url": _build_public_url(request, name)})
        return {"created": int(time.time()), "data": data}

    # Default: b64_json
    return {
        "created": int(time.time()),
        "data": [{"b64_json": img.get("b64_json", "")} for img in images],
    }


@router.post("/images/edits", summary="Edit image")
async def image_edits(
    request: Request,
    image: UploadFile,
    user: Annotated[UserContext, Depends(get_openai_user)],
) -> Any:
    """
    Edit an image from a text prompt (OpenAI ``/v1/images/edits`` compatible).

    Accepts ``multipart/form-data`` with:

    * ``image`` — base PNG to edit (required).
    * ``mask`` — optional PNG; transparent pixels mark the area to repaint.
      When omitted the edit is performed as img2img over the full image.
    * ``model`` — profile_id (required).
    * ``prompt`` — edit description (required).
    * ``n``, ``size``, ``response_format``, ``user`` — same semantics as
      ``/v1/images/generations``.

    Returns 400 ``mask_unsupported`` when the loaded pipeline has no inpainting
    variant in diffusers' auto-mapping (e.g. distilled FLUX.2 Klein,
    Z-Image-Turbo), and 400 ``edit_unsupported`` when no img2img variant is
    available either.
    """
    form = await request.form(
        max_part_size=max(1, int(settings.openai_image_max_part_size_mb)) * 1024 * 1024
    )
    model_id: str = form.get("model", "")
    prompt: str = form.get("prompt", "")
    response_format: str = form.get("response_format", "b64_json")

    if response_format not in _RESPONSE_FORMATS:
        raise HTTPException(
            status_code=400,
            detail={"error": {
                "message": (
                    f"Invalid response_format '{response_format}'. "
                    f"Must be one of: {sorted(_RESPONSE_FORMATS)}"
                ),
                "type": "invalid_request_error",
                "param": "response_format",
            }},
        )
    if not prompt:
        raise _openai_error(
            "The 'prompt' field is required.",
            "invalid_request_error",
            param="prompt",
        )

    image_bytes = await image.read()
    mask_upload = form.get("mask")
    mask_bytes: bytes | None = None
    mask_filename: str | None = None
    mask_content_type: str | None = None
    # Duck-type instead of ``isinstance(UploadFile)``: starlette's UploadFile
    # and FastAPI's re-exported one can resolve to different classes in some
    # import paths, and any file-like part has ``.read()``.
    if mask_upload is not None and hasattr(mask_upload, "read"):
        mask_bytes = await mask_upload.read()
        mask_filename = getattr(mask_upload, "filename", None)
        mask_content_type = getattr(mask_upload, "content_type", None)

    model_manager = get_model_manager(request)
    profile_registry = get_profile_registry(request)

    federation_form_data: dict[str, str] = {
        "model": model_id,
        "prompt": prompt,
        "response_format": response_format,
    }
    for field in ("n", "size", "user"):
        val = form.get(field)
        if val is not None and not hasattr(val, "read"):
            federation_form_data[field] = str(val)

    federation_files: dict[str, tuple[str, bytes, str]] = {
        "image": (
            image.filename or "image.png",
            image_bytes,
            image.content_type or "image/png",
        )
    }
    if mask_bytes is not None:
        federation_files["mask"] = (
            mask_filename or "mask.png",
            mask_bytes,
            mask_content_type or "image/png",
        )

    fed_resp = await try_proxy_multipart(
        request,
        model_id=model_id,
        files=federation_files,
        data=federation_form_data,
        federation_manager=get_federation_manager(request),
        model_manager=model_manager,
        profile_registry=profile_registry,
        logger=logger,
    )
    if fed_resp is not None:
        return Response(
            content=fed_resp.content,
            status_code=fed_resp.status_code,
            media_type=fed_resp.headers.get("content-type"),
        )

    profile, state = await resolve_profile(
        model_id,
        model_manager,
        profile_registry,
        user=user,
    )
    check_capability(state, "image_generation", "image editing")

    request_body: dict[str, Any] = {"prompt": prompt}
    for field in ("negative_prompt", "num_inference_steps", "guidance_scale",
                  "strength", "seed"):
        val = form.get(field)
        if val is not None and not hasattr(val, "read"):
            request_body[field] = val
    n_val = form.get("n")
    if n_val is not None and not hasattr(n_val, "read"):
        try:
            request_body["n"] = int(n_val)
        except (TypeError, ValueError) as exc:
            raise _openai_error(
                "Field 'n' must be an integer.",
                "invalid_request_error",
                param="n",
            ) from exc

    merged_body = merge_profile_defaults(profile, request_body)
    worker_key = compute_worker_key(profile.base_model_id, profile.load_overrides)

    size_raw = form.get("size")
    width: int | None = None
    height: int | None = None
    if size_raw and not hasattr(size_raw, "read"):
        width, height = _parse_size(str(size_raw))

    worker_body: dict[str, Any] = {
        "prompt": merged_body.get("prompt", ""),
        "image_b64": base64.b64encode(image_bytes).decode("ascii"),
    }
    if mask_bytes is not None:
        worker_body["mask_b64"] = base64.b64encode(mask_bytes).decode("ascii")
    if merged_body.get("negative_prompt") is not None:
        worker_body["negative_prompt"] = merged_body["negative_prompt"]
    if width is not None and height is not None:
        worker_body["width"] = width
        worker_body["height"] = height
    if "num_inference_steps" in merged_body:
        try:
            worker_body["num_inference_steps"] = int(merged_body["num_inference_steps"])
        except (TypeError, ValueError):
            pass
    if "guidance_scale" in merged_body:
        try:
            worker_body["guidance_scale"] = float(merged_body["guidance_scale"])
        except (TypeError, ValueError):
            pass
    if "strength" in merged_body:
        try:
            worker_body["strength"] = float(merged_body["strength"])
        except (TypeError, ValueError):
            pass
    if "seed" in merged_body and merged_body["seed"] is not None:
        try:
            worker_body["seed"] = int(merged_body["seed"])
        except (TypeError, ValueError):
            pass
    worker_body["num_images"] = int(merged_body.get("n", 1) or 1)

    worker_pool = request.app.state.worker_pool
    inflight_request_id = model_manager.begin_request(worker_key)
    try:
        result = await worker_pool.forward_request(worker_key, "/edit", worker_body)
    except httpx.HTTPStatusError as exc:
        # The worker translates "mask not supported by this pipeline" /
        # "img2img not supported" into HTTP 400; surface them with a stable
        # OpenAI-style error code so clients can branch on it.
        if exc.response.status_code == 400:
            detail_text = exc.response.text or ""
            code = "mask_unsupported" if mask_bytes is not None else "edit_unsupported"
            raise _openai_error(
                detail_text.strip() or "Image editing not supported by this model.",
                "invalid_request_error",
                param="model",
                code=code,
                status_code=400,
            ) from exc
        raise_upstream_http_error(exc)
    finally:
        model_manager.end_request(worker_key, inflight_request_id)

    images = result.get("images", [])
    if response_format == "url":
        data = []
        for img in images:
            b64 = img.get("b64_json", "")
            if not b64:
                continue
            name = _save_image(b64)
            data.append({"url": _build_public_url(request, name)})
        return {"created": int(time.time()), "data": data}

    return {
        "created": int(time.time()),
        "data": [{"b64_json": img.get("b64_json", "")} for img in images],
    }


@router.get("/images/files/{name}", summary="Retrieve a generated image")
async def get_generated_image(name: str):
    """
    Serve a previously generated image.

    The URL is the one returned by ``/v1/images/generations`` when
    ``response_format=url``. Authentication is intentionally not required:
    the UUID-based filename acts as an unguessable token, and files expire
    after ``settings.image_url_ttl_seconds`` (mirroring OpenAI's 60-min URLs).
    """
    if not _IMAGE_NAME_RE.match(name):
        raise HTTPException(status_code=404, detail="Not found")

    path = Path(settings.image_outputs_dir) / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Image expired or not found")

    return FileResponse(path=path, media_type="image/png", filename=name)


def _parse_size(size: str) -> tuple[int, int]:
    _SIZE_MAP = {
        "256x256": (256, 256),
        "512x512": (512, 512),
        "1024x1024": (1024, 1024),
        "1792x1024": (1792, 1024),
        "1024x1792": (1024, 1792),
    }
    if size in _SIZE_MAP:
        return _SIZE_MAP[size]
    # Parse generic WxH
    parts = size.lower().split("x")
    if len(parts) == 2:
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            pass
    return 1024, 1024
