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
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, Response

from ocabra.api._deps_auth import UserContext
from ocabra.config import settings

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
)

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

    # --- Federation hook ---
    federation_manager = get_federation_manager(request)
    if federation_manager is not None:
        from ocabra.config import settings as _settings

        if _settings.federation_enabled:
            from ocabra.core.federation import resolve_federated

            target, peer = await resolve_federated(
                model_id, model_manager, federation_manager, profile_registry
            )
            if target == "remote":
                request.state.federation_remote_node_id = peer.peer_id
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
    try:
        result = await worker_pool.forward_request(worker_key, "/generate", worker_body)
    except httpx.HTTPStatusError as exc:
        raise_upstream_http_error(exc)

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
