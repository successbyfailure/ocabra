"""TensorRT-LLM compilation API.

Endpoints:
    POST   /ocabra/trtllm/compile                    — submit a compile job
    GET    /ocabra/trtllm/compile                     — list all jobs (history)
    GET    /ocabra/trtllm/compile/{job_id}/stream     — SSE progress stream
    DELETE /ocabra/trtllm/compile/{job_id}            — cancel a job
    DELETE /ocabra/trtllm/engines/{engine_name}       — delete engine + unregister model
"""
from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ocabra.config import settings
from ocabra.core.trtllm_compile_manager import CompileRequest
from ocabra.redis_client import subscribe

router = APIRouter(tags=["trtllm"])

# ── Pydantic schemas ─────────────────────────────────────────────


class CompileJobCreate(BaseModel):
    """Request body for POST /ocabra/trtllm/compile."""

    model_id: str = Field(
        ...,
        description="Source model identifier, e.g. 'vllm/Qwen/Qwen3.5-27B-GPTQ-Int4'",
    )
    gpu_indices: list[int] = Field(
        ...,
        description="GPU indices to use, e.g. [1] or [0, 1] for tensor parallelism",
    )
    dtype: str = Field(
        "fp16",
        description="Weight dtype: fp16 | bf16 | int8 | fp8",
    )
    max_batch_size: Annotated[int, Field(ge=1, le=256)] = 1
    max_input_len: Annotated[int, Field(ge=128)] = 2048
    max_seq_len: Annotated[int, Field(ge=256)] = 4096
    engine_name: str = Field(
        ...,
        description="Name for the compiled engine (used as directory name)",
    )


# ── Endpoints ────────────────────────────────────────────────────


@router.post("/trtllm/compile", status_code=202)
async def create_compile_job(body: CompileJobCreate, request: Request) -> dict:
    """Submit a TensorRT-LLM engine compilation job.

    Parameters:
        body: CompileJobCreate — compilation parameters.

    Returns:
        dict with job_id and stream_url for SSE progress.
    """
    manager = _get_manager(request)

    req = CompileRequest(
        source_model=body.model_id,
        engine_name=body.engine_name,
        gpu_indices=body.gpu_indices,
        dtype=body.dtype,
        max_batch_size=body.max_batch_size,
        max_input_len=body.max_input_len,
        max_seq_len=body.max_seq_len,
    )
    state = await manager.enqueue(req)
    return {
        "job_id": state.job_id,
        "stream_url": f"/ocabra/trtllm/compile/{state.job_id}/stream",
        **state.to_dict(),
    }


@router.get("/trtllm/compile")
async def list_compile_jobs(request: Request) -> list[dict]:
    """List all TensorRT-LLM compilation jobs (history + active).

    Returns:
        List of job dicts ordered by started_at descending.
    """
    manager = _get_manager(request)
    jobs = await manager.list_jobs()
    return [j.to_dict() for j in jobs]


@router.get("/trtllm/compile/{job_id}/stream")
async def stream_compile_job(job_id: str, request: Request) -> StreamingResponse:
    """Stream real-time progress for a compilation job via SSE.

    Subscribes to the Redis pub/sub channel for the job and forwards
    events until the job reaches a terminal state or the client disconnects.

    Parameters:
        job_id: Compile job identifier.

    Returns:
        Server-Sent Events stream with progress and log events.
    """
    manager = _get_manager(request)

    # Validate job exists
    state = manager.get_job(job_id)
    if state is None:
        jobs = await manager.list_jobs()
        match = next((j for j in jobs if j.job_id == job_id), None)
        if match is None:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
        state = match

    # If already terminal, return final state immediately
    if state.status in {"done", "failed", "cancelled"}:
        async def _final_stream():
            data = json.dumps(state.to_dict())
            yield f"data: {data}\n\n"

        return StreamingResponse(_final_stream(), media_type="text/event-stream")

    async def _event_generator():
        channel = f"trtllm:compile:{job_id}"
        async with subscribe(channel) as pubsub:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = await asyncio.wait_for(pubsub.get_message(ignore_subscribe_messages=True), timeout=1.0)
                except TimeoutError:
                    # Send keepalive comment
                    yield ": keepalive\n\n"
                    continue

                if msg is None:
                    yield ": keepalive\n\n"
                    continue

                data = msg.get("data", "")
                if not data:
                    continue

                yield f"data: {data}\n\n"

                # Stop streaming when job reaches terminal state
                try:
                    parsed = json.loads(data)
                    if parsed.get("status") in {"done", "failed", "cancelled"}:
                        break
                except (json.JSONDecodeError, AttributeError):
                    pass

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.delete("/trtllm/compile/{job_id}", status_code=200)
async def cancel_compile_job(job_id: str, request: Request) -> dict:
    """Cancel a pending or running compilation job.

    Parameters:
        job_id: Compile job identifier.

    Returns:
        Updated job dict with status='cancelled'.
    """
    manager = _get_manager(request)
    try:
        state = await manager.cancel(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return state.to_dict()


@router.delete("/trtllm/engines/{engine_name}", status_code=200)
async def delete_engine(engine_name: str, request: Request) -> dict:
    """Delete a compiled TensorRT-LLM engine from disk and unregister its model.

    Parameters:
        engine_name: Engine directory name (e.g. 'TinyLlama--TinyLlama-1.1B-fp16').

    Returns:
        dict with engine_name and deleted=True.
    """
    # Delete engine files from disk (use container-side path)
    if settings.tensorrt_llm_engines_dir:
        engines_base = Path(settings.tensorrt_llm_engines_dir)
    else:
        models_container = settings.tensorrt_llm_docker_models_mount_container or "/data/models"
        engines_base = Path(models_container) / "tensorrt_llm"
    engine_root = engines_base / engine_name
    if engine_root.exists() and not _is_path_within_base(engine_root, engines_base):
        raise HTTPException(
            status_code=400,
            detail="Refusing to delete a TensorRT-LLM engine path outside the configured engines directory",
        )

    # Unload + unregister from model manager
    model_manager = getattr(request.app.state, "model_manager", None)
    model_id = f"tensorrt_llm/{engine_name}"
    if model_manager is not None and model_id in model_manager._states:
        try:
            await model_manager.delete_model(model_id)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to unregister model: {exc}") from exc

    if engine_root.exists():
        try:
            await asyncio.to_thread(shutil.rmtree, str(engine_root))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to delete engine files: {exc}") from exc

    # Remove compile job records for this engine from DB
    try:
        import sqlalchemy as sa
        from ocabra.database import AsyncSessionLocal
        from ocabra.db.trtllm import TrtllmCompileJob
        async with AsyncSessionLocal() as session:
            await session.execute(
                sa.delete(TrtllmCompileJob).where(TrtllmCompileJob.engine_name == engine_name)
            )
            await session.commit()
    except Exception:
        pass  # DB cleanup is best-effort

    return {"engine_name": engine_name, "deleted": True}


@router.get("/trtllm/estimate")
async def estimate_compile(
    request: Request,
    model_id: str,
    tp_size: int = 1,
    dtype: str = "fp16",
    max_batch_size: int = 1,
    max_seq_len: int = 4096,
) -> dict:
    """Estimate VRAM required to build and serve a TRT-LLM engine.

    Args:
        model_id: Canonical model id (e.g. vllm/Qwen/Qwen3-32B-AWQ).
        tp_size: Tensor parallelism — number of GPUs.
        dtype: fp16 | bf16 | int8 | fp8.
        max_batch_size: Max concurrent requests.
        max_seq_len: Max sequence length (input + output).

    Returns:
        Estimates in MB for build phase and serve phase, per GPU and total.
    """
    gpu_indices = _parse_gpu_indices_query(request)
    return _estimate_vram(model_id, tp_size, gpu_indices, dtype, max_batch_size, max_seq_len)


def _estimate_vram(
    model_id: str,
    tp_size: int,
    gpu_indices: list[int] | None,
    dtype: str,
    max_batch_size: int,
    max_seq_len: int,
) -> dict:
    """Heuristic VRAM estimator based on empirical TRT-LLM observations.

    Key findings from real compilations:
    - Qwen3-8B  fp16  TP=1: build ~12GB, serve ~16GB
    - Qwen3-32B W4A16 TP=1: build ~19GB, serve ~41GB  (OOM on 24GB)
    - Qwen3-32B W4A16 TP=2: build ~10GB/GPU, serve ~10GB/GPU

    VRAM components:
      weights   = params * bytes_per_param / tp_size
      kv_cache  = 2 * layers * heads * head_dim * max_seq_len * max_batch_size * dtype_bytes / tp_size
      overhead  = ~2GB per GPU (TRT workspace, CUDA context, plugins)
      build_extra = ~1.5x weights during engine optimisation phase
    """
    import json as _json
    from pathlib import Path as _Path

    # ── Derive HF config path ────────────────────────────────────
    raw = model_id
    parts = raw.split("/", 1)
    if parts[0] in {"vllm", "tensorrt_llm", "llama_cpp", "sglang"}:
        raw = parts[1]
    hf_dir = raw.replace("/", "--")
    models_dir = _Path(settings.models_dir or "/data/models")
    cfg_path = models_dir / "huggingface" / hf_dir / "config.json"

    # ── Parse model config ───────────────────────────────────────
    params_b: float | None = None  # billions of params
    num_layers: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    is_awq = False
    is_gptq = False

    try:
        cfg = _json.loads(cfg_path.read_text())
        arch = (cfg.get("architectures") or [""])[0]

        # Parameter count from config fields
        hidden = cfg.get("hidden_size", 4096)
        intermediate = cfg.get("intermediate_size", hidden * 4)
        num_layers = cfg.get("num_hidden_layers", 32)
        num_heads = cfg.get("num_attention_heads", 32)
        num_kv_heads = cfg.get("num_key_value_heads") or num_heads
        vocab_size = cfg.get("vocab_size", 32000)
        head_dim = hidden // num_heads

        # Rough params estimate: attn + FFN + embedding
        attn_params = num_layers * hidden * hidden * 4  # Q,K,V,O projections (approx)
        ffn_params = num_layers * hidden * intermediate * 3  # gate, up, down
        embed_params = vocab_size * hidden * 2  # embed + lm_head
        params_b = (attn_params + ffn_params + embed_params) / 1e9

        qcfg = cfg.get("quantization_config") or {}
        qm = str(qcfg.get("quant_method") or "").lower()
        is_awq = "awq" in qm
        is_gptq = "gptq" in qm

    except Exception:
        # Fallback: estimate from model name
        name_lower = raw.lower()
        if "0.5b" in name_lower: params_b = 0.5
        elif "0.6b" in name_lower: params_b = 0.6
        elif "0.8b" in name_lower: params_b = 0.8
        elif "1b" in name_lower or "1.1b" in name_lower: params_b = 1.1
        elif "1.5b" in name_lower: params_b = 1.5
        elif "3b" in name_lower: params_b = 3.0
        elif "4b" in name_lower: params_b = 4.0
        elif "7b" in name_lower: params_b = 7.0
        elif "8b" in name_lower: params_b = 8.0
        elif "9b" in name_lower: params_b = 9.0
        elif "12b" in name_lower: params_b = 12.0
        elif "14b" in name_lower: params_b = 14.0
        elif "27b" in name_lower: params_b = 27.0
        elif "32b" in name_lower: params_b = 32.0
        elif "70b" in name_lower: params_b = 70.0
        elif "72b" in name_lower: params_b = 72.0
        else: params_b = 7.0  # safe default

        name_lower_full = model_id.lower()
        is_awq = "awq" in name_lower_full
        is_gptq = "gptq" in name_lower_full or "int4" in name_lower_full or "int8" in name_lower_full

    # ── Bytes per parameter ──────────────────────────────────────
    if is_awq or is_gptq:
        bytes_per_param = 0.5   # 4-bit
    elif dtype in ("int8",):
        bytes_per_param = 1.0
    elif dtype in ("fp8",):
        bytes_per_param = 1.0
    else:
        bytes_per_param = 2.0   # fp16 / bf16

    # ── Weights VRAM per GPU ─────────────────────────────────────
    weights_mb = (params_b * 1e9 * bytes_per_param) / (1024 ** 2) / tp_size

    # ── KV cache VRAM ────────────────────────────────────────────
    # TRT-LLM uses fp16 KV cache (2 bytes) regardless of model dtype
    kv_bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * 2  # K+V, fp16
    kv_mb = (kv_bytes_per_token * max_seq_len * max_batch_size) / (1024 ** 2) / tp_size

    # ── Overhead (CUDA context + TRT plugins + activations) ──────
    overhead_mb = 2048  # ~2GB base

    # ── Serve estimate ───────────────────────────────────────────
    serve_mb_per_gpu = weights_mb + kv_mb + overhead_mb
    serve_total_mb = serve_mb_per_gpu * tp_size

    # ── Build estimate (TRT optimization needs ~30% extra scratch) ──
    # Calibrated from real builds:
    #   Qwen3-8B fp16 TP=1: built on 24GB (peak ~22GB) → 17GB weights + 0.3x = ~24GB
    #   Qwen3-32B AWQ TP=1: built on 24GB (failed disk, not OOM) → 15GB + 0.3x = ~22GB
    build_extra_mb = weights_mb * 0.3   # TRT engine build workspace
    build_mb_per_gpu = weights_mb + build_extra_mb + overhead_mb
    build_total_mb = build_mb_per_gpu * tp_size

    # ── Disk estimate ────────────────────────────────────────────
    # Engine on disk ≈ weights * 1.05 (TRT serialization overhead)
    # Checkpoint on disk ≈ weights (temporary, deleted after build)
    engine_disk_mb = weights_mb * tp_size * 1.05
    ckpt_disk_mb = weights_mb * tp_size  # temporary

    has_config = cfg_path.exists()

    return {
        "model_id": model_id,
        "estimated_params_b": round(params_b, 1) if params_b else None,
        "quant": "awq" if is_awq else ("gptq" if is_gptq else dtype),
        "tp_size": tp_size,
        "config_found": has_config,
        "serve": {
            "vram_per_gpu_mb": round(serve_mb_per_gpu),
            "vram_total_mb": round(serve_total_mb),
            "breakdown": {
                "weights_mb": round(weights_mb),
                "kv_cache_mb": round(kv_mb),
                "overhead_mb": overhead_mb,
            },
        },
        "build": {
            "vram_per_gpu_mb": round(build_mb_per_gpu),
            "vram_total_mb": round(build_total_mb),
        },
        "disk": {
            "engine_mb": round(engine_disk_mb),
            "checkpoint_mb_temp": round(ckpt_disk_mb),
            "total_peak_mb": round(engine_disk_mb + ckpt_disk_mb),
        },
        "warnings": _estimate_warnings(
            build_mb_per_gpu, serve_mb_per_gpu, tp_size, gpu_indices, params_b or 0
        ),
    }


def _estimate_warnings(
    build_mb: float,
    serve_mb: float,
    tp_size: int,
    gpu_indices: list[int] | None,
    params_b: float,
) -> list[str]:
    import pynvml

    def _format_tight_message(kind: str, required_mb: float, gpu_index: int, free_mb: int) -> str:
        margin_mb = required_mb - free_mb
        return (
            f"Muy justo: {kind} necesita ~{required_mb / 1024:.1f}GB/GPU "
            f"y la GPU {gpu_index} tiene {free_mb / 1024:.1f}GB libres ahora "
            f"({int(round(margin_mb))} MB por debajo). Puede funcionar si liberas algo de VRAM antes de empezar."
        )

    warnings = []
    try:
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        gpus = []
        for i in range(count):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(h)
            gpus.append({"index": i, "free_mb": info.free // (1024 ** 2), "total_mb": info.total // (1024 ** 2)})
        pynvml.nvmlShutdown()

        if tp_size > len(gpus):
            warnings.append(f"tp_size={tp_size} exceeds available GPUs ({len(gpus)})")
        else:
            if gpu_indices:
                selected = [g for g in gpus if g["index"] in gpu_indices]
            else:
                selected = gpus[:tp_size]

            if len(selected) != tp_size:
                warnings.append(
                    f"Selected GPUs {gpu_indices} do not match tp_size={tp_size} or are not available"
                )
                return warnings

            smallest = min(selected, key=lambda gpu: gpu["free_mb"])
            smallest_free = smallest["free_mb"]
            smallest_index = smallest["index"]
            tight_margin_mb = 1024

            if build_mb > smallest_free:
                if build_mb - smallest_free <= tight_margin_mb:
                    warnings.append(_format_tight_message("build", build_mb, smallest_index, smallest_free))
                else:
                    warnings.append(
                        f"Build probablemente no cabe: necesita ~{build_mb/1024:.1f}GB/GPU y la GPU {smallest_index} tiene {smallest_free/1024:.1f}GB libres ahora"
                    )
            elif serve_mb > smallest_free:
                if serve_mb - smallest_free <= tight_margin_mb:
                    warnings.append(_format_tight_message("serving", serve_mb, smallest_index, smallest_free))
                else:
                    warnings.append(
                        f"Serving probablemente no cabe: necesita ~{serve_mb/1024:.1f}GB/GPU y la GPU {smallest_index} tiene {smallest_free/1024:.1f}GB libres ahora"
                    )
    except Exception:
        pass
    return warnings


def _parse_gpu_indices_query(request: Request) -> list[int] | None:
    values = request.query_params.getlist("gpu_indices")
    if not values:
        raw = request.query_params.get("gpu_indices")
        values = [raw] if raw else []

    parsed: list[int] = []
    for value in values:
        for part in str(value).split(","):
            item = part.strip()
            if not item:
                continue
            try:
                parsed.append(int(item))
            except ValueError:
                continue

    return parsed or None


def _is_path_within_base(path: Path, base: Path) -> bool:
    try:
        return path.resolve(strict=False).is_relative_to(base.resolve(strict=False))
    except (OSError, RuntimeError, ValueError):
        return False


# ── Helpers ──────────────────────────────────────────────────────


def _get_manager(request: Request):
    manager = getattr(request.app.state, "trtllm_compile_manager", None)
    if manager is None:
        raise HTTPException(
            status_code=503,
            detail="TrtllmCompileManager not initialised",
        )
    return manager
