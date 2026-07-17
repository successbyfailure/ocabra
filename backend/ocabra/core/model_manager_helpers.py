"""Pure helper functions for model manager lifecycle concerns."""

from __future__ import annotations

from pathlib import Path

from ocabra.core.vram_planner import arch_from_gguf, plan_llama_cpp_vram_mb


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


def resolve_bitnet_gpu_layers(state, default_gpu_layers: int) -> int:
    return resolve_bitnet_option(state, "gpu_layers", default_gpu_layers)


def estimate_bitnet_vram_from_config(
    state,
    *,
    default_gpu_layers: int,
    default_total_layers: int = 32,
    default_model_vram_mb: int = 400,
) -> int:
    gpu_layers = resolve_bitnet_gpu_layers(state, default_gpu_layers)
    if gpu_layers <= 0:
        return 0
    total_layers = max(1, resolve_bitnet_option(state, "total_layers", default_total_layers))
    model_vram_mb = max(1, resolve_bitnet_option(state, "model_vram_mb", default_model_vram_mb))
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
        est = plan_llama_cpp_vram_mb(arch, size_mb, ctx_size, gpu_layers=gpu_layers)
        if est > 0:
            return est

    # Fallback: weights-only heuristic (arch unreadable). ~1.08x leaves headroom
    # to trigger eviction without over-reserving so much that a model that really
    # fits gets bumped to tensor-parallel across a too-small GPU.
    try:
        total_layers = max(1, int(resolve_llama_cpp_option(state, "total_layers", default_total_layers)))
    except (TypeError, ValueError):
        total_layers = default_total_layers
    fraction = min(gpu_layers, total_layers) / total_layers
    return int(size_mb * fraction * 1.08)
