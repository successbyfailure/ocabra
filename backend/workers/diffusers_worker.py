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
    "Flux2KleinPipeline",
    "StableDiffusion3Pipeline",
    "StableDiffusionXLPipeline",
    "StableDiffusionPipeline",
    "ZImagePipeline",
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


class EditRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str
    image_b64: str
    mask_b64: str | None = None
    negative_prompt: str | None = None
    # When omitted, output keeps the input image's dimensions.
    width: int | None = None
    height: int | None = None
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    # img2img strength — 0=identity, 1=fully re-imagined. Ignored by inpainting
    # pipelines that don't take this kwarg (filtered out via signature inspect).
    strength: float = 0.8
    seed: int | None = None
    num_images: int = 1


@dataclass
class WorkerState:
    model_id: str
    model_path: Path
    pipeline_type: str
    pipeline: Any = None
    load_error: str | None = None
    # Derived pipelines for image editing — built lazily on first use via
    # diffusers' AutoPipeline*.from_pipe(), which shares weights with the
    # text2img pipeline (no extra VRAM, no second load).
    img2img_pipeline: Any = None
    inpaint_pipeline: Any = None


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
    # Import lazily so a missing optional class (e.g. Flux2KleinPipeline on an
    # old diffusers) doesn't crash the worker at import time for unrelated
    # models. Each requested pipeline is fetched on demand from the package.
    import diffusers as _diffusers

    cls = getattr(_diffusers, pipeline_type, None)
    if cls is None:
        raise ValueError(
            f"Pipeline '{pipeline_type}' is not available in this diffusers "
            f"build (version={getattr(_diffusers, '__version__', 'unknown')})"
        )
    if pipeline_type not in SUPPORTED_PIPELINES:
        raise ValueError(f"Unsupported pipeline type: {pipeline_type}")
    return cls


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


def _filter_pipeline_kwargs(pipeline: Any, kwargs: dict) -> dict:
    """Drop kwargs the pipeline's ``__call__`` doesn't accept.

    Different diffusers pipelines accept different parameter sets — e.g.
    distilled FLUX.2 Klein and Z-Image Turbo don't take ``negative_prompt``
    or ``guidance_scale``. Inspect the signature once per call and pass
    only what's supported (and drop ``None`` values so we don't override
    pipeline-side defaults with explicit nulls).
    """
    import inspect

    try:
        sig = inspect.signature(pipeline.__call__)
    except (TypeError, ValueError):
        return {k: v for k, v in kwargs.items() if v is not None}

    accepts_kwargs = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    allowed = set(sig.parameters)
    return {
        k: v
        for k, v in kwargs.items()
        if v is not None and (accepts_kwargs or k in allowed)
    }


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

    # Build the kwargs we'd like to pass, then filter to those the pipeline
    # actually accepts. Distilled pipelines (Flux2Klein, Z-Image-Turbo) drop
    # ``negative_prompt`` and ``guidance_scale``; SD3 / SDXL / SD1.5 keep them.
    candidate_kwargs: dict[str, Any] = {
        "prompt": req.prompt,
        "negative_prompt": req.negative_prompt,
        "width": req.width,
        "height": req.height,
        "num_inference_steps": req.num_inference_steps,
        "guidance_scale": req.guidance_scale,
        "num_images_per_prompt": req.num_images,
        "generator": generator,
    }
    call_kwargs = _filter_pipeline_kwargs(state.pipeline, candidate_kwargs)

    started = time.perf_counter()
    output = state.pipeline(**call_kwargs)
    duration_ms = int((time.perf_counter() - started) * 1000)

    images = [GenerateImage(b64_json=pil_to_b64(image)) for image in output.images]
    return GenerateResponse(
        images=images, generation_time_ms=duration_ms, seed_used=seed_used
    )


def _decode_b64_image(data: str, *, field: str):
    """Decode a base64-encoded image payload into a PIL RGB image."""
    from PIL import Image

    try:
        raw = base64.b64decode(data, validate=True)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid base64 for '{field}': {exc}",
        ) from exc
    try:
        return Image.open(BytesIO(raw))
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot decode '{field}' as an image: {exc}",
        ) from exc


def _mask_from_alpha(mask_image):
    """Build a diffusers-style mask (white=edit) from an OpenAI-style mask.

    OpenAI's spec uses the alpha channel: transparent pixels mark the area to
    edit. Diffusers' inpainting pipelines expect a single-channel image where
    *white* pixels are the area to repaint. We invert the alpha so transparent
    (=0) becomes 255 (=edit) and opaque (=255) becomes 0 (=keep). Masks without
    an alpha channel are passed through converted to "L".
    """
    from PIL import ImageOps

    if mask_image.mode in ("RGBA", "LA") or "A" in mask_image.getbands():
        alpha = mask_image.convert("RGBA").split()[-1]
        return ImageOps.invert(alpha)
    return mask_image.convert("L")


def _derive_edit_pipeline(state: WorkerState, *, with_mask: bool) -> Any:
    """Lazily build (and cache) the img2img or inpainting variant.

    Uses ``AutoPipelineForImage2Image.from_pipe`` / ``AutoPipelineForInpainting``
    which share weights with the loaded text2img pipeline — no VRAM cost. When
    the base pipeline isn't in the auto-mapping (e.g. distilled FLUX.2 Klein or
    Z-Image-Turbo, which only ship a text2img variant) ``from_pipe`` raises
    ``ValueError`` and we propagate it as a 400 so the API layer can return a
    clean ``edit_unsupported`` / ``mask_unsupported`` error.
    """
    from diffusers import AutoPipelineForImage2Image, AutoPipelineForInpainting

    if with_mask:
        if state.inpaint_pipeline is None:
            try:
                state.inpaint_pipeline = AutoPipelineForInpainting.from_pipe(state.pipeline)
            except (ValueError, KeyError, NotImplementedError) as exc:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Inpainting (mask edit) is not supported for pipeline "
                        f"'{state.pipeline_type}': {exc}"
                    ),
                ) from exc
        return state.inpaint_pipeline

    if state.img2img_pipeline is None:
        try:
            state.img2img_pipeline = AutoPipelineForImage2Image.from_pipe(state.pipeline)
        except (ValueError, KeyError, NotImplementedError) as exc:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Image-to-image editing is not supported for pipeline "
                    f"'{state.pipeline_type}': {exc}"
                ),
            ) from exc
    return state.img2img_pipeline


def edit_sync(state: WorkerState, req: EditRequest) -> GenerateResponse:
    if torch is None:
        raise RuntimeError("torch is required to run diffusers_worker")
    if state.pipeline is None:
        raise RuntimeError("Pipeline not loaded")

    base_image = _decode_b64_image(req.image_b64, field="image_b64").convert("RGB")

    mask_image = None
    if req.mask_b64:
        raw_mask = _decode_b64_image(req.mask_b64, field="mask_b64")
        mask_image = _mask_from_alpha(raw_mask)
        if mask_image.size != base_image.size:
            mask_image = mask_image.resize(base_image.size)

    # Output keeps the input geometry unless the caller forces a size.
    width = req.width or base_image.width
    height = req.height or base_image.height
    if (width, height) != base_image.size:
        base_image = base_image.resize((width, height))
        if mask_image is not None:
            mask_image = mask_image.resize((width, height))

    edit_pipeline = _derive_edit_pipeline(state, with_mask=mask_image is not None)

    seed_used = req.seed if req.seed is not None else random.randint(0, 2**31 - 1)
    if torch.cuda.is_available():
        generator = torch.Generator(device="cuda").manual_seed(seed_used)
    else:
        generator = torch.Generator().manual_seed(seed_used)

    candidate_kwargs: dict[str, Any] = {
        "prompt": req.prompt,
        "negative_prompt": req.negative_prompt,
        "image": base_image,
        "mask_image": mask_image,
        "width": width,
        "height": height,
        "num_inference_steps": req.num_inference_steps,
        "guidance_scale": req.guidance_scale,
        "strength": req.strength,
        "num_images_per_prompt": req.num_images,
        "generator": generator,
    }
    call_kwargs = _filter_pipeline_kwargs(edit_pipeline, candidate_kwargs)

    started = time.perf_counter()
    output = edit_pipeline(**call_kwargs)
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

    @app.post("/edit", response_model=GenerateResponse)
    async def edit(req: EditRequest) -> GenerateResponse:
        if state.pipeline is None:
            raise HTTPException(
                status_code=503, detail=state.load_error or "Pipeline not ready"
            )

        loop = asyncio.get_running_loop()
        run = partial(edit_sync, state, req)
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
