"""Pure helper functions for model manager lifecycle concerns."""

from __future__ import annotations

from pathlib import Path


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
    default_total_layers: int = 32,
) -> int:
    """Pre-load VRAM estimate for a llama.cpp GGUF model.

    The backend itself can only estimate *after* load (its option cache is
    populated in ``load()``), so the scheduler would otherwise see 0 MB, never
    check VRAM and never evict to make room. Estimate from the on-disk GGUF size
    scaled by the fraction of layers offloaded to GPU, plus ~15% headroom for
    the CUDA context + KV cache. Returns 0 for CPU-only (gpu_layers <= 0).
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
        total_layers = max(1, int(resolve_llama_cpp_option(state, "total_layers", default_total_layers)))
    except (TypeError, ValueError):
        total_layers = default_total_layers
    fraction = min(gpu_layers, total_layers) / total_layers
    return int(size_mb * fraction * 1.15)
