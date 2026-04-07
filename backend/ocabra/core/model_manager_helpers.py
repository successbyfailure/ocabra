"""Pure helper functions for model manager lifecycle concerns."""

from __future__ import annotations


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
