"""Pure helper functions for model manager lifecycle concerns."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ocabra.core.vram_planner import arch_from_gguf, plan_llama_cpp_vram_mb

PRISMML_DEFAULT_GPU_LAYERS = 99


def compute_worker_key(base_model_id: str, load_overrides: dict | None) -> str:
    """Derive the stable worker key for a model profile."""
    if not load_overrides:
        return base_model_id
    canonical = json.dumps(load_overrides, sort_keys=True, separators=(",", ":"))
    short_hash = hashlib.sha256(canonical.encode()).hexdigest()[:12]
    return f"{base_model_id}::{short_hash}"


def build_diarized_extra_config(base_extra_config: dict | None) -> dict:
    """Build extra_config with diarization enabled, used by profile creation."""
    merged = dict(base_extra_config or {})
    merged["diarization_enabled"] = True
    whisper_cfg = merged.get("whisper") if isinstance(merged.get("whisper"), dict) else {}
    merged["whisper"] = {**whisper_cfg, "diarizationEnabled": True}
    return merged


def resolve_bitnet_option(state, key: str, default: int) -> int:
    extra = state.extra_config if isinstance(state.extra_config, dict) else {}
    nested = extra.get("bitnet") if isinstance(extra.get("bitnet"), dict) else None
    if nested and key in nested:
        return int(nested[key])
    if key in extra:
        return int(extra[key])
    return int(default)


def is_prismml_bitnet_state(state) -> bool:
    """Detect Bonsai/PrismML before load from ids or configured GGUF path."""
    extra = state.extra_config if isinstance(state.extra_config, dict) else {}
    nested = extra.get("bitnet") if isinstance(extra.get("bitnet"), dict) else {}
    markers = (
        getattr(state, "model_id", ""),
        getattr(state, "backend_model_id", ""),
        extra.get("model_path", ""),
        nested.get("model_path", ""),
    )
    text = " ".join(str(value).lower() for value in markers if value)
    return "bonsai" in text or "q1_0" in text or "prismml" in text


def resolve_bitnet_gpu_layers(state, default_gpu_layers: int) -> int:
    extra = state.extra_config if isinstance(state.extra_config, dict) else {}
    nested = extra.get("bitnet") if isinstance(extra.get("bitnet"), dict) else None
    if nested and "gpu_layers" in nested:
        return int(nested["gpu_layers"])
    if "gpu_layers" in extra:
        return int(extra["gpu_layers"])
    if is_prismml_bitnet_state(state):
        return PRISMML_DEFAULT_GPU_LAYERS
    return int(default_gpu_layers)


def estimate_bitnet_vram_from_config(
    state,
    *,
    default_gpu_layers: int,
    default_total_layers: int = 32,
    default_model_vram_mb: int = 400,
    models_dir: str | Path | None = None,
) -> int:
    gpu_layers = resolve_bitnet_gpu_layers(state, default_gpu_layers)
    if gpu_layers <= 0:
        return 0
    total_layers = max(1, resolve_bitnet_option(state, "total_layers", default_total_layers))
    extra = state.extra_config if isinstance(state.extra_config, dict) else {}
    nested = extra.get("bitnet") if isinstance(extra.get("bitnet"), dict) else {}
    has_explicit_estimate = "model_vram_mb" in extra or "model_vram_mb" in nested
    model_vram_mb = max(1, resolve_bitnet_option(state, "model_vram_mb", default_model_vram_mb))
    if not has_explicit_estimate:
        configured_path = nested.get("model_path") or extra.get("model_path")
        candidates = [Path(str(configured_path))] if configured_path else []
        backend_id = str(getattr(state, "backend_model_id", "") or "")
        if models_dir and backend_id:
            root = Path(models_dir)
            candidates.extend(
                [
                    root / backend_id,
                    root / f"{backend_id}.gguf",
                    root / "huggingface" / backend_id.replace("/", "--"),
                ]
            )
        model_file = next((path for path in candidates if path.is_file()), None)
        if model_file is not None:
            model_vram_mb = max(
                model_vram_mb,
                int(model_file.stat().st_size / (1024 * 1024) * 1.08),
            )
    return int(model_vram_mb * min(gpu_layers, total_layers) / total_layers)


def resolve_llama_cpp_option(state, key: str, default):
    """Read a llama.cpp option from ``extra_config['llama_cpp'][key]`` or the
    top level (accepting snake_case), mirroring the backend's own resolution."""
    extra = state.extra_config if isinstance(state.extra_config, dict) else {}
    nested = extra.get("llama_cpp") if isinstance(extra.get("llama_cpp"), dict) else None
    if nested and key in nested:
        return nested[key]
    if key in extra:
        return extra[key]
    return default


def resolve_llama_cpp_gpu_layers(state, default_gpu_layers: int) -> int:
    try:
        return int(resolve_llama_cpp_option(state, "gpu_layers", default_gpu_layers))
    except (TypeError, ValueError):
        return int(default_gpu_layers)


def estimate_llama_cpp_vram_from_config(
    state,
    *,
    default_gpu_layers: int,
    default_ctx_size: int = 4096,
    default_total_layers: int = 32,
) -> int:
    """Pre-load VRAM estimate for a llama.cpp GGUF model.

    The backend itself can only estimate *after* load (its option cache is
    populated in ``load()``), so the scheduler would otherwise see 0 MB, never
    check VRAM and never evict to make room.

    When the GGUF architecture is readable we compute weights + an *exact* KV
    cache for the configured ctx_size (KV scales with context, which a flat
    file-size multiplier misses entirely — e.g. a low-KV-head model over-reserves
    and a large-context one under-reserves). Otherwise we fall back to the on-disk
    GGUF size scaled by the offloaded-layer fraction plus a small overhead.
    Returns 0 for CPU-only (gpu_layers <= 0).
    """
    gpu_layers = resolve_llama_cpp_gpu_layers(state, default_gpu_layers)
    if gpu_layers <= 0:
        return 0

    model_path = resolve_llama_cpp_option(state, "model_path", None) or resolve_llama_cpp_option(
        state, "model_file", None
    )
    size_mb = 0
    if model_path:
        try:
            size_mb = int(Path(str(model_path)).stat().st_size / (1024 * 1024))
        except OSError:
            size_mb = 0
    if size_mb <= 0:
        return 0  # unknown size → let the backend/scheduler proceed as before

    try:
        ctx_size = int(resolve_llama_cpp_option(state, "ctx_size", default_ctx_size))
    except (TypeError, ValueError):
        ctx_size = default_ctx_size

    # KV-aware path: read the architecture straight from the GGUF header.
    arch = arch_from_gguf(str(model_path)) if model_path else None
    if arch is not None:
        # Size the KV cache to the configured cache_type (quantized KV halves/
        # quarters the footprint). Without this the estimate always assumes f16
        # and over-reserves ~2x, wrongly rejecting q8 long-context loads.
        cache_type = str(
            resolve_llama_cpp_option(state, "cache_type_k", None)
            or resolve_llama_cpp_option(state, "cache_type_v", None)
            or "f16"
        ).lower()
        kv_dtype_bytes = {
            "f16": 2.0,
            "fp16": 2.0,
            "bf16": 2.0,
            "q8_0": 1.0,
            "q8": 1.0,
            "q5_0": 0.65,
            "q5_1": 0.65,
            "q4_0": 0.5,
            "q4_1": 0.5,
            "q4": 0.5,
            "iq4_nl": 0.5,
        }.get(cache_type, 2.0)
        est = plan_llama_cpp_vram_mb(
            arch, size_mb, ctx_size, gpu_layers=gpu_layers, kv_dtype_bytes=kv_dtype_bytes
        )
        if est > 0:
            return est

    # Fallback: weights-only heuristic (arch unreadable). ~1.08x leaves headroom
    # to trigger eviction without over-reserving so much that a model that really
    # fits gets bumped to tensor-parallel across a too-small GPU.
    try:
        total_layers = max(
            1, int(resolve_llama_cpp_option(state, "total_layers", default_total_layers))
        )
    except (TypeError, ValueError):
        total_layers = default_total_layers
    fraction = min(gpu_layers, total_layers) / total_layers
    return int(size_mb * fraction * 1.08)
