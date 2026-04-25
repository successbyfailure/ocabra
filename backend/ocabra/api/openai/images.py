"""
POST /v1/images/generations — image generation endpoint.
"""

from __future__ import annotations

import time
from typing import Annotated, Any

import httpx
from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response

from ocabra.api._deps_auth import UserContext

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
    """
    body = await request.json()
    model_id: str = body.get("model", "")

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
    return {
        "created": int(time.time()),
        "data": [{"b64_json": img.get("b64_json", "")} for img in images],
    }


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
