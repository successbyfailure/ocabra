"""FastAPI worker for Diffusers image generation."""

import argparse
import asyncio
import base64
import json
import os
import random
import time
from dataclasses import dataclass
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional runtime dependency
    torch = None


SUPPORTED_PIPELINES = {
    "FluxPipeline",
    "StableDiffusionXLPipeline",
    "StableDiffusionPipeline",
}


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


@dataclass
class WorkerState:
    model_id: str
    model_path: Path
    pipeline_type: str
    pipeline: Any = None
    load_error: str | None = None


def detect_pipeline_class(model_path: Path) -> str:
    """Detect the diffusers pipeline class for *model_path*.

    Two layouts are supported:

    1. **HuggingFace diffusers tree** (a directory with ``model_index.json``).
       The ``_class_name`` / ``pipeline_class`` field selects the pipeline.
    2. **Single-file checkpoint** (a ``.safetensors`` / ``.ckpt`` file
       produced by Stable Diffusion WebUI / ComfyUI workflows). The pipeline
       class is inferred from the file name — SDXL when the stem contains
       ``xl`` / ``sdxl`` / ``stable-diffusion-xl``, otherwise SD1.5. Use the
       ``DIFFUSERS_PIPELINE_OVERRIDE`` env var to force a class when the
       heuristic guesses wrong.
    """
    if model_path.is_file():
        suffix = model_path.suffix.lower()
        if suffix not in {".safetensors", ".ckpt"}:
            raise ValueError(
                f"Unsupported single-file checkpoint extension '{suffix}' for {model_path}"
            )
        override = os.getenv("DIFFUSERS_PIPELINE_OVERRIDE", "").strip()
        if override:
            if override not in SUPPORTED_PIPELINES:
                raise ValueError(
                    f"DIFFUSERS_PIPELINE_OVERRIDE='{override}' is not in {SUPPORTED_PIPELINES}"
                )
            return override
        stem = model_path.stem.lower()
        if any(token in stem for token in ("sdxl", "stable-diffusion-xl", "sd_xl", "xl-base")):
            return "StableDiffusionXLPipeline"
        return "StableDiffusionPipeline"

    model_index_path = model_path / "model_index.json"
    if not model_index_path.exists():
        raise FileNotFoundError(f"model_index.json not found in {model_path}")

    with model_index_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    candidates = [
        payload.get("_class_name"),
        payload.get("pipeline_class"),
    ]

    for candidate in candidates:
        if candidate in SUPPORTED_PIPELINES:
            return candidate

    raise ValueError("Unsupported pipeline class in model_index.json")


def resolve_pipeline_class(pipeline_type: str) -> type:
    from diffusers import (
        FluxPipeline,
        StableDiffusionPipeline,
        StableDiffusionXLPipeline,
    )

    mapping = {
        "FluxPipeline": FluxPipeline,
        "StableDiffusionXLPipeline": StableDiffusionXLPipeline,
        "StableDiffusionPipeline": StableDiffusionPipeline,
    }
    if pipeline_type not in mapping:
        raise ValueError(f"Unsupported pipeline type: {pipeline_type}")
    return mapping[pipeline_type]


def env_flag(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, str(default))).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def select_torch_dtype() -> Any:
    if torch is None:
        raise RuntimeError("torch is required to run diffusers_worker")

    dtype_override = str(os.getenv("DIFFUSERS_TORCH_DTYPE", "auto")).strip().lower()
    if dtype_override in {"float16", "fp16"}:
        return torch.float16
    if dtype_override in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if dtype_override in {"float32", "fp32"}:
        return torch.float32

    if not torch.cuda.is_available():
        return torch.float32

    return torch.float16


def maybe_compile_pipeline(pipeline: Any) -> None:
    if torch is None:
        return
    if not env_flag("DIFFUSERS_ENABLE_TORCH_COMPILE", False):
        return
    if not hasattr(torch, "compile"):
        return

    target_attr = "transformer" if hasattr(pipeline, "transformer") else "unet"
    if not hasattr(pipeline, target_attr):
        return

    module = getattr(pipeline, target_attr)
    try:
        compiled = torch.compile(module, mode="reduce-overhead")
        setattr(pipeline, target_attr, compiled)
    except Exception:
        return


def maybe_enable_xformers_attention(pipeline: Any) -> None:
    if not env_flag("DIFFUSERS_ENABLE_XFORMERS", False):
        return
    enable_xformers = getattr(pipeline, "enable_xformers_memory_efficient_attention", None)
    if callable(enable_xformers):
        try:
            enable_xformers()
        except Exception:
            return


def maybe_enable_cpu_offload(pipeline: Any) -> None:
    if torch is None:
        return
    if not torch.cuda.is_available():
        return
    offload_mode = str(os.getenv("DIFFUSERS_OFFLOAD_MODE", "none")).strip().lower()
    if offload_mode == "model":
        enable_model_offload = getattr(pipeline, "enable_model_cpu_offload", None)
        if callable(enable_model_offload):
            enable_model_offload()
        return
    if offload_mode == "sequential":
        enable_sequential_offload = getattr(pipeline, "enable_sequential_cpu_offload", None)
        if callable(enable_sequential_offload):
            enable_sequential_offload()
        return


def load_pipeline(state: WorkerState) -> None:
    if torch is None:
        raise RuntimeError("torch is required to run diffusers_worker")

    pipeline_class = resolve_pipeline_class(state.pipeline_type)
    dtype = select_torch_dtype()

    if torch.cuda.is_available() and env_flag("DIFFUSERS_ALLOW_TF32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if state.model_path.is_file():
        # Single-file checkpoint (SDXL / SD1.5 .safetensors from a1111/ComfyUI).
        # ``from_single_file`` re-builds the unet/vae/text_encoder from the
        # bundled weights without expecting a HuggingFace tree on disk.
        pipeline = pipeline_class.from_single_file(
            str(state.model_path), torch_dtype=dtype
        )
    else:
        pipeline = pipeline_class.from_pretrained(
            str(state.model_path), torch_dtype=dtype
        )
    offload_mode = str(os.getenv("DIFFUSERS_OFFLOAD_MODE", "none")).strip().lower()
    if torch.cuda.is_available() and offload_mode == "none":
        pipeline = pipeline.to("cuda")

    maybe_enable_xformers_attention(pipeline)
    maybe_enable_cpu_offload(pipeline)
    maybe_compile_pipeline(pipeline)
    state.pipeline = pipeline


def pil_to_b64(image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def generate_sync(state: WorkerState, req: GenerateRequest) -> GenerateResponse:
    if torch is None:
        raise RuntimeError("torch is required to run diffusers_worker")

    if state.pipeline is None:
        raise RuntimeError("Pipeline not loaded")

    seed_used = req.seed if req.seed is not None else random.randint(0, 2**31 - 1)
    generator = None

    if torch.cuda.is_available():
        generator = torch.Generator(device="cuda").manual_seed(seed_used)
    else:
        generator = torch.Generator().manual_seed(seed_used)

    started = time.perf_counter()
    output = state.pipeline(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        width=req.width,
        height=req.height,
        num_inference_steps=req.num_inference_steps,
        guidance_scale=req.guidance_scale,
        num_images_per_prompt=req.num_images,
        generator=generator,
    )
    duration_ms = int((time.perf_counter() - started) * 1000)

    images = [GenerateImage(b64_json=pil_to_b64(image)) for image in output.images]
    return GenerateResponse(
        images=images, generation_time_ms=duration_ms, seed_used=seed_used
    )


def create_app(state: WorkerState) -> FastAPI:
    app = FastAPI(title="oCabra Diffusers Worker")

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(req: GenerateRequest) -> GenerateResponse:
        if state.pipeline is None:
            raise HTTPException(
                status_code=503, detail=state.load_error or "Pipeline not ready"
            )

        loop = asyncio.get_running_loop()
        run = partial(generate_sync, state, req)
        return await loop.run_in_executor(None, run)

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
            "pipeline_type": state.pipeline_type,
            "vram_used_mb": vram_used_mb,
        }

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diffusers image generation worker")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    return parser.parse_args()


def main() -> None:
    if torch is None:
        raise RuntimeError("torch is required to run diffusers_worker")

    args = parse_args()
    model_path = Path(args.model_path)

    pipeline_type = detect_pipeline_class(model_path)
    state = WorkerState(
        model_id=args.model_id, model_path=model_path, pipeline_type=pipeline_type
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
