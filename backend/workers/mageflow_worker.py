"""FastAPI worker for Microsoft Mage-Flow image generation.

Mage-Flow is NOT a native diffusers pipeline: it ships a custom
``MageFlowPipeline`` class (``pip install -e mage_flow``) with its own
``.generate()`` / ``.edit()`` API and pins a distinct stack (torch 2.13,
transformers 5.5, diffusers 0.38). It therefore runs in its own venv
(``/data/backends/mage/venv``) behind this dedicated worker instead of the
shared diffusers worker.

The HTTP contract mirrors ``diffusers_worker.py`` exactly so the OpenAI image
endpoints (``/v1/images/generations`` → worker ``/generate``) forward the same
``worker_body`` and parse the same ``{"images": [{"b64_json": ...}]}`` response
with no special-casing in ``api/openai/images.py``.

Attention: flash-attn is avoided entirely. The editable install's
``attn_type`` default is patched to ``"sdpa"`` (covers both the DiT and the
Qwen3-VL text encoder), so no CUDA extension needs to be compiled.
"""

import argparse
import base64
import os
import time
from dataclasses import dataclass
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import Any

import asyncio

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional runtime dependency
    torch = None


# ``num_inference_steps`` / ``guidance_scale`` defaults that ``images.py``
# sends when the caller does not specify them. They are the *diffusers*
# defaults (20 steps, 7.5 cfg) which over-process a 4-step Turbo model, so we
# treat these exact sentinel values as "unset" and fall back to the model's
# own Turbo defaults instead.
_DIFFUSERS_DEFAULT_STEPS = 20
_DIFFUSERS_DEFAULT_CFG = 7.5


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str
    negative_prompt: str | None = None
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    seed: int | None = None
    num_images: int = 1


class GenerateImage(BaseModel):
    b64_json: str


class GenerateResponse(BaseModel):
    images: list[GenerateImage]
    generation_time_ms: int
    seed_used: int


class EditRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str
    image_b64: str
    mask_b64: str | None = None
    negative_prompt: str | None = None
    width: int | None = None
    height: int | None = None
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    strength: float = 0.8
    seed: int | None = None
    num_images: int = 1


@dataclass
class WorkerState:
    model_id: str
    model_path: Path
    supports_edit: bool
    pipeline: Any = None
    load_error: str | None = None


def _resolve_steps(req_steps: int) -> int:
    if req_steps == _DIFFUSERS_DEFAULT_STEPS:
        return int(os.getenv("MAGE_DEFAULT_STEPS", "4"))
    return max(1, int(req_steps))


def _resolve_cfg(req_cfg: float) -> float:
    if abs(req_cfg - _DIFFUSERS_DEFAULT_CFG) < 1e-6:
        return float(os.getenv("MAGE_DEFAULT_CFG", "1.0"))
    return float(req_cfg)


def _make_divisible_by_16(value: int) -> int:
    return max(256, (int(value) // 16) * 16)


def load_pipeline(state: WorkerState) -> None:
    if torch is None:
        raise RuntimeError("torch is required to run mageflow_worker")

    from mage_flow import MageFlowPipeline
    from mage_flow.models.modules._attn_backend import set_attn_backend

    if torch.cuda.is_available() and str(os.getenv("MAGE_ALLOW_TF32", "true")).lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = MageFlowPipeline.from_pretrained(str(state.model_path), device=device)
    # Belt-and-suspenders: the editable install already defaults attn_type to
    # "sdpa", but force the DiT shim too so a stray flash2 config can never
    # drag in the (uncompiled) flash-attn kernel at first forward.
    set_attn_backend("sdpa")
    state.pipeline = pipeline


def pil_to_b64(image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def generate_sync(state: WorkerState, req: GenerateRequest) -> GenerateResponse:
    if state.pipeline is None:
        raise RuntimeError("Pipeline not loaded")

    import random

    seed_base = req.seed if req.seed is not None else random.randint(0, 2**31 - 1)
    steps = _resolve_steps(req.num_inference_steps)
    cfg = _resolve_cfg(req.guidance_scale)
    n = max(1, int(req.num_images))
    h, w = _make_divisible_by_16(req.height), _make_divisible_by_16(req.width)

    prompts = [req.prompt] * n
    seeds = [seed_base + i for i in range(n)]

    started = time.perf_counter()
    imgs = state.pipeline.generate(
        prompts,
        steps=steps,
        cfg=cfg,
        heights=[h] * n,
        widths=[w] * n,
        seeds=seeds,
    )
    duration_ms = int((time.perf_counter() - started) * 1000)

    images = [GenerateImage(b64_json=pil_to_b64(img)) for img in imgs]
    return GenerateResponse(
        images=images, generation_time_ms=duration_ms, seed_used=seed_base
    )


def edit_sync(state: WorkerState, req: EditRequest) -> GenerateResponse:
    if state.pipeline is None:
        raise RuntimeError("Pipeline not loaded")
    if not state.supports_edit:
        # This checkpoint is a text-to-image model. Editing needs a
        # Mage-Flow-Edit-* checkpoint loaded as a separate model. Surface a
        # 400 so images.py maps it to the stable ``edit_unsupported`` code.
        raise HTTPException(
            status_code=400,
            detail=(
                "This Mage-Flow model is text-to-image only. Load a "
                "Mage-Flow-Edit checkpoint for /v1/images/edits."
            ),
        )

    from PIL import Image

    import random

    try:
        raw = base64.b64decode(req.image_b64, validate=True)
        base_image = Image.open(BytesIO(raw)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid image_b64: {exc}") from exc

    width = _make_divisible_by_16(req.width or base_image.width)
    height = _make_divisible_by_16(req.height or base_image.height)
    if (width, height) != base_image.size:
        base_image = base_image.resize((width, height))

    seed_base = req.seed if req.seed is not None else random.randint(0, 2**31 - 1)
    steps = _resolve_steps(req.num_inference_steps)
    cfg = _resolve_cfg(req.guidance_scale)

    started = time.perf_counter()
    imgs = state.pipeline.edit(
        [req.prompt],
        [base_image],
        steps=steps,
        cfg=cfg,
        heights=[height],
        widths=[width],
        seeds=[seed_base],
    )
    duration_ms = int((time.perf_counter() - started) * 1000)

    images = [GenerateImage(b64_json=pil_to_b64(img)) for img in imgs]
    return GenerateResponse(
        images=images, generation_time_ms=duration_ms, seed_used=seed_base
    )


def create_app(state: WorkerState) -> FastAPI:
    app = FastAPI(title="oCabra Mage-Flow Worker")

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(req: GenerateRequest) -> GenerateResponse:
        if state.pipeline is None:
            raise HTTPException(
                status_code=503, detail=state.load_error or "Pipeline not ready"
            )
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, partial(generate_sync, state, req))

    @app.post("/edit", response_model=GenerateResponse)
    async def edit(req: EditRequest) -> GenerateResponse:
        if state.pipeline is None:
            raise HTTPException(
                status_code=503, detail=state.load_error or "Pipeline not ready"
            )
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, partial(edit_sync, state, req))

    @app.get("/health")
    async def health() -> dict[str, bool]:
        return {"ok": state.pipeline is not None}

    @app.get("/info")
    async def info() -> dict[str, Any]:
        vram_used_mb = 0
        if torch is not None and torch.cuda.is_available():
            vram_used_mb = int(torch.cuda.memory_allocated(0) / (1024 * 1024))
        return {
            "model_id": state.model_id,
            "pipeline_type": "MageFlowPipeline",
            "supports_edit": state.supports_edit,
            "vram_used_mb": vram_used_mb,
        }

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mage-Flow image generation worker")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    return parser.parse_args()


def main() -> None:
    if torch is None:
        raise RuntimeError("torch is required to run mageflow_worker")

    args = parse_args()
    model_path = Path(args.model_path)
    # Edit-capable checkpoints are named ``Mage-Flow-Edit-*``; the plain t2i
    # Turbo/Base checkpoints only support .generate().
    supports_edit = "edit" in model_path.name.lower()

    state = WorkerState(
        model_id=args.model_id, model_path=model_path, supports_edit=supports_edit
    )
    try:
        load_pipeline(state)
    except Exception as exc:
        state.load_error = str(exc)
        raise

    app = create_app(state)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
