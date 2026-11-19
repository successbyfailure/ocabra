"""Memory projection of a llama.cpp load, by llama.cpp itself.

``llama-fit-params --fit-print`` (built next to ``llama-server`` by
``install_llama_cpp.sh``) reads the GGUF and the context parameters and prints
what the load would allocate on each device and in host RAM — model weights,
context (KV + recurrent state) and compute buffers — without loading the
model. It is the same accounting llama.cpp runs when it starts.

Measured 2026-11-16 on gemma4:26b with its experts in RAM (Mapache's
configuration, 10k context, 4 slots, q8 KV) against a real load on the 3060:

    projection  GPU 2459 + 755 + 541 = 3755 MiB   host 14557 + 26 MiB
    real        GPU 3752 MiB (nvidia-smi)          host 14557 MiB mapped
    oCabra's formulas: GPU 4371 MiB, host 13809 MiB

The formulas carried hand-calibrated constants (gemma4's included a vision
projector that is now loaded on demand) and missed the sliding-window cache
growing with slots and the token embeddings left in RAM. This module is the
estimator for llama.cpp-family models whenever the tool is installed; callers
fall back to the formulas when it returns ``None``.

What it does not see: a speculative draft model (MTP/EAGLE) and a vision
projector loaded at start. Callers add those on top.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

FIT_PARAMS_BIN = "llama-fit-params"
_TIMEOUT_S = 60.0
_CACHE_MAX = 256


@dataclass(frozen=True)
class FitProjection:
    """MiB per physical GPU index and in host RAM, split by kind."""

    gpus: dict[int, dict[str, int]] = field(default_factory=dict)
    host: dict[str, int] = field(default_factory=dict)

    @property
    def vram_by_gpu(self) -> dict[int, int]:
        return {g: sum(parts.values()) for g, parts in self.gpus.items()}

    @property
    def vram_total_mb(self) -> int:
        return sum(self.vram_by_gpu.values())

    @property
    def host_mb(self) -> int:
        return sum(self.host.values())


_cache: dict[tuple, FitProjection | None] = {}
_locks: dict[tuple, asyncio.Lock] = {}


def fit_params_binary(server_bin: str | os.PathLike[str] | None) -> Path | None:
    """``llama-fit-params`` next to ``llama-server``, or ``None`` if absent."""
    if not server_bin:
        return None
    candidate = Path(server_bin).with_name(FIT_PARAMS_BIN)
    return candidate if candidate.is_file() and os.access(candidate, os.X_OK) else None


def _fmt(value: float) -> str:
    text = f"{float(value):.4f}".rstrip("0").rstrip(".")
    return text or "0"


def build_args(
    model_file: str,
    *,
    ctx_size: int,
    parallel: int,
    gpu_layers: int | None,
    n_cpu_moe: int | None = None,
    override_tensor: str | None = None,
    cache_type_k: str | None = None,
    cache_type_v: str | None = None,
    flash_attn: bool | None = None,
    batch_size: int | None = None,
    ubatch_size: int | None = None,
    split_mode: str | None = None,
    tensor_split: list[float] | None = None,
    main_gpu: int | None = None,
    lora: list[str] | None = None,
) -> list[str]:
    """CLI arguments for ``llama-fit-params --fit-print`` mirroring a launch.

    ``--kv-unified`` is not accepted by the tool; without it llama.cpp sizes
    the sliding-window cache per sequence, so the projection can be slightly
    above a unified-pool load (conservative).
    """
    args = ["-m", str(model_file), "--fit-print", "on", "-c", str(int(ctx_size))]
    args += ["-np", str(max(1, int(parallel)))]
    if gpu_layers is not None:
        args += ["-ngl", str(int(gpu_layers))]
    if n_cpu_moe:
        args += ["--n-cpu-moe", str(int(n_cpu_moe))]
    if override_tensor:
        args += ["-ot", str(override_tensor)]
    if cache_type_k:
        args += ["-ctk", str(cache_type_k)]
    if cache_type_v:
        args += ["-ctv", str(cache_type_v)]
    if flash_attn is not None:
        args += ["-fa", "on" if flash_attn else "off"]
    if batch_size:
        args += ["-b", str(int(batch_size))]
    if ubatch_size:
        args += ["-ub", str(int(ubatch_size))]
    if split_mode:
        args += ["-sm", str(split_mode)]
    if tensor_split:
        args += ["-ts", ",".join(_fmt(r) for r in tensor_split)]
    if main_gpu is not None:
        args += ["-mg", str(int(main_gpu))]
    if lora:
        args += list(lora)
    return args


def parse_output(
    stdout: str, gpu_indices: list[int], ratios: list[float] | None = None
) -> FitProjection | None:
    """Parse ``<device> <model> <context> <compute>`` lines (MiB).

    ``CUDA<k>`` is the k-th *visible* device, mapped back to ``gpu_indices``;
    ``Host`` is RAM. Returns ``None`` when nothing usable was printed.
    """
    gpus: dict[int, dict[str, int]] = {}
    host: dict[str, int] = {}
    for line in stdout.splitlines():
        parts = line.split()
        if len(parts) != 4:
            continue
        name, *numbers = parts
        try:
            model, context, compute = (int(float(n)) for n in numbers)
        except ValueError:
            continue
        values = {"model": model, "context": context, "compute": compute}
        if name.lower() in {"host", "cpu"}:
            host = values
        elif name.lower().startswith("meta") and gpu_indices:
            # ``--split-mode tensor`` groups the GPUs into one virtual device
            # printed as ``Meta()``: split it back by the tensor ratios (even
            # without them). Before this, a qwen3.8 projection lost its 17 GB.
            weights = (
                [max(0.0, float(r)) for r in ratios]
                if ratios and len(ratios) == len(gpu_indices) and sum(ratios) > 0
                else [1.0] * len(gpu_indices)
            )
            total = sum(weights)
            for gpu, weight in zip(gpu_indices, weights, strict=True):
                share = {k: int(v * weight / total) for k, v in values.items()}
                previous = gpus.get(int(gpu), {})
                gpus[int(gpu)] = {k: previous.get(k, 0) + share[k] for k in share}
        elif name.upper().startswith("CUDA"):
            try:
                ordinal = int(name[4:])
            except ValueError:
                continue
            if 0 <= ordinal < len(gpu_indices):
                gpus[int(gpu_indices[ordinal])] = values
    if not gpus and not host:
        return None
    return FitProjection(gpus=gpus, host=host)


async def project(
    server_bin: str | os.PathLike[str] | None,
    model_file: str,
    gpu_indices: list[int],
    args: list[str],
    ratios: list[float] | None = None,
) -> FitProjection | None:
    """Run the projection (cached per file, devices and arguments).

    ``gpu_indices`` are the physical GPUs the load would see
    (``CUDA_VISIBLE_DEVICES``); an empty list projects a CPU-only load.
    Never raises: any failure returns ``None`` and the caller falls back.
    """
    binary = fit_params_binary(server_bin)
    if binary is None:
        return None
    try:
        mtime = Path(model_file).stat().st_mtime_ns  # noqa: ASYNC240 — solo metadatos
    except OSError:
        return None
    key = (str(binary), str(model_file), mtime, tuple(gpu_indices), tuple(args))
    if key in _cache:
        return _cache[key]
    lock = _locks.setdefault(key, asyncio.Lock())
    async with lock:
        if key in _cache:
            return _cache[key]
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_indices) or "-1"
        lib_dir = str(binary.parent)
        env["LD_LIBRARY_PATH"] = lib_dir + (
            f":{env['LD_LIBRARY_PATH']}" if env.get("LD_LIBRARY_PATH") else ""
        )
        result: FitProjection | None = None
        try:
            proc = await asyncio.create_subprocess_exec(
                str(binary),
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=_TIMEOUT_S)
            except TimeoutError:
                proc.kill()
                await proc.wait()
                logger.warning("llama_fit_timeout", model_file=model_file)
                stdout, stderr = b"", b""
            if proc.returncode == 0:
                result = parse_output(stdout.decode(errors="replace"), list(gpu_indices), ratios)
            else:
                logger.info(
                    "llama_fit_failed",
                    model_file=model_file,
                    returncode=proc.returncode,
                    stderr=stderr.decode(errors="replace")[-300:],
                )
        except Exception as exc:  # noqa: BLE001 - an estimate must never break a load
            logger.info("llama_fit_error", model_file=model_file, error=str(exc))
        if len(_cache) >= _CACHE_MAX:
            _cache.pop(next(iter(_cache)))
        _cache[key] = result
        _locks.pop(key, None)
        return result


def args_from_options(model_file: str, options: dict[str, Any]) -> list[str]:
    """``build_args`` from the backend's resolved launch options."""
    return build_args(
        model_file,
        ctx_size=int(options.get("ctx_size") or 4096),
        parallel=int(options.get("parallel_slots") or 1),
        gpu_layers=options.get("gpu_layers"),
        n_cpu_moe=options.get("n_cpu_moe"),
        override_tensor=options.get("override_tensor"),
        cache_type_k=options.get("cache_type_k"),
        cache_type_v=options.get("cache_type_v"),
        flash_attn=bool(options.get("flash_attn"))
        if options.get("flash_attn") is not None
        else None,
        batch_size=options.get("batch_size"),
        ubatch_size=options.get("ubatch_size"),
        split_mode=options.get("split_mode"),
        tensor_split=options.get("tensor_split"),
        main_gpu=options.get("main_gpu"),
        lora=_lora_args(options.get("lora")),
    )


def _lora_args(adapters: Any) -> list[str] | None:
    if not adapters:
        return None
    from ocabra.backends.llama_cpp_backend import LlamaCppBackend

    return LlamaCppBackend.lora_args(adapters)


# ── calibration from real loads ──────────────────────────────────────────
#
# What a process really holds on a GPU differs from the projection by a
# model-dependent amount: on Mapache's GTX 1660 (2026-11-18) qwen3:30b with
# experts in RAM came out 150-190 MB *below* its projection in every config,
# gemma4:26b within ±130. 150 MB on a 6 GB card is a load refused that fits
# (qwen3:30b with 40 expert layers in RAM: refused by 34 MB). After each load
# the model manager records measured − projected per model file and GPU, and
# later projections add it. Averaged over the last loads, clamped, in memory
# (a restart starts from the plain projection again).

_OFFSET_CLAMP_MB = 512
_OFFSET_WEIGHT = 0.5  # new measurement vs history
#: A measurement further than this from the projection is not a projection
#: error but a wrong reading (another process attributed to the worker, a
#: load still allocating): ignored. Real errors measured on Mapache were
#: within ±200 MB; a misattributed embedding (3 GB) taught qwen3:30b −1024 MB
#: and its projection then admitted a load that ran out of memory.
_PLAUSIBLE_FRACTION = 0.15
_PLAUSIBLE_MIN_MB = 400
_offsets: dict[tuple[str, int], float] = {}


def calibration_offset_mb(model_file: str, gpu: int) -> int:
    return int(round(_offsets.get((str(model_file), int(gpu)), 0.0)))


def record_measurement(model_file: str, gpu: int, measured_mb: int, raw_projected_mb: int) -> int:
    """Fold one load's real VRAM on ``gpu`` into its offset; returns the new one."""
    if measured_mb <= 0 or raw_projected_mb <= 0:
        return calibration_offset_mb(model_file, gpu)
    raw_delta = measured_mb - raw_projected_mb
    if abs(raw_delta) > max(_PLAUSIBLE_MIN_MB, raw_projected_mb * _PLAUSIBLE_FRACTION):
        logger.warning(
            "llama_fit_calibration_ignored",
            model_file=model_file,
            gpu=gpu,
            measured_mb=measured_mb,
            projected_mb=raw_projected_mb,
        )
        return calibration_offset_mb(model_file, gpu)
    delta = max(-_OFFSET_CLAMP_MB, min(_OFFSET_CLAMP_MB, raw_delta))
    key = (str(model_file), int(gpu))
    previous = _offsets.get(key)
    _offsets[key] = (
        float(delta)
        if previous is None
        else previous * (1 - _OFFSET_WEIGHT) + delta * _OFFSET_WEIGHT
    )
    logger.info(
        "llama_fit_calibrated",
        model_file=model_file,
        gpu=gpu,
        measured_mb=measured_mb,
        projected_mb=raw_projected_mb,
        offset_mb=calibration_offset_mb(model_file, gpu),
    )
    return calibration_offset_mb(model_file, gpu)
