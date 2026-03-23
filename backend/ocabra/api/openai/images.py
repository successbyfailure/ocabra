"""
POST /v1/images/generations — image generation endpoint.
"""
from __future__ import annotations

import time
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


@router.post("/images/generations", summary="Create image")
async def image_generations(request: Request) -> Any:
    """
    Generate images from a text prompt.
    Proxies to the Diffusers backend worker.
    Requires a model with capability image_generation=True.

    Supports OpenAI size strings: 256x256, 512x512, 1024x1024, 1792x1024, 1024x1792.
    """
    body = await request.json()
    model_id: str = body.get("model", "")

    model_manager = get_model_manager(request)
    state = await ensure_loaded(model_manager, model_id)
    check_capability(state, "image_generation", "image generation")
    model_id = state.model_id

    # Translate OpenAI size to width/height
    size_str: str = body.get("size", "1024x1024")
    width, height = _parse_size(size_str)

    worker_body = {
        "prompt": body.get("prompt", ""),
        "negative_prompt": body.get("negative_prompt"),
        "width": width,
        "height": height,
        "num_images": body.get("n", 1),
        "num_inference_steps": body.get("num_inference_steps", 20),
        "guidance_scale": body.get("guidance_scale", 7.5),
        "seed": body.get("seed"),
    }

    worker_pool = request.app.state.worker_pool
    try:
        result = await worker_pool.forward_request(model_id, "/generate", worker_body)
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
