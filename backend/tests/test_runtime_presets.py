from __future__ import annotations

import pytest

from ocabra.core.runtime_presets import (
    ContextCapacityError,
    ResolvedRuntimePreset,
    advertised_effective_max,
    build_runtime_key,
    resolve_max_context,
    select_runtime_preset,
)


def _preset(
    key: str,
    context: int,
    *,
    loaded_key: str | None = None,
    default: bool = False,
    enabled: bool = True,
) -> ResolvedRuntimePreset:
    return ResolvedRuntimePreset(
        preset_key=key,
        effective_context=context,
        parallelism=2,
        kv_cache_dtype="q8",
        runtime_key=loaded_key,
        is_default=default,
        enabled=enabled,
    )


def test_resolve_max_context_uses_smallest_effective_limit() -> None:
    assert resolve_max_context(
        model_max=262_144,
        backend_max=300_000,
        hardware_max=270_000,
        admin_cap=200_000,
    ) == 200_000


def test_resolve_max_context_ignores_unknown_limits() -> None:
    assert resolve_max_context(
        model_max=262_144,
        backend_max=None,
        hardware_max=180_000,
    ) == 180_000


def test_runtime_key_is_canonical_and_excludes_preset_name() -> None:
    kwargs = {
        "model_id": "gemma4:12b",
        "backend_type": "ollama",
        "effective_context": 65_536,
        "parallelism": 2,
        "kv_cache_dtype": "q8",
        "placement_policy": "forced",
        "load_flags": {"flash_attention": True, "temperature": 0.2},
    }
    first = build_runtime_key(gpu_indices=(1, 0, 1), **kwargs)
    second = build_runtime_key(gpu_indices=(0, 1), **kwargs)
    assert first == second
    assert first.startswith("runtime:")


def test_loaded_large_runtime_serves_small_request() -> None:
    presets = [
        _preset("64k", 65_536, default=True),
        _preset("max", 262_144, loaded_key="runtime:max"),
    ]
    result = select_runtime_preset(
        presets,
        required_tokens=16_000,
        loaded_runtime_keys={"runtime:max"},
    )
    assert result.preset.preset_key == "max"
    assert result.reason == "loaded"
    assert result.reused_loaded_runtime is True


def test_smallest_loaded_sufficient_runtime_wins() -> None:
    presets = [
        _preset("64k", 65_536, loaded_key="runtime:64", default=True),
        _preset("128k", 131_072, loaded_key="runtime:128"),
        _preset("max", 262_144, loaded_key="runtime:max"),
    ]
    result = select_runtime_preset(
        presets,
        required_tokens=100_000,
        loaded_runtime_keys={"runtime:64", "runtime:128", "runtime:max"},
    )
    assert result.preset.preset_key == "128k"


def test_default_wins_when_nothing_suitable_is_loaded() -> None:
    presets = [
        _preset("32k", 32_768),
        _preset("64k", 65_536, default=True),
        _preset("max", 262_144),
    ]
    result = select_runtime_preset(presets, required_tokens=8_000)
    assert result.preset.preset_key == "64k"
    assert result.reason == "default"


def test_request_above_default_uses_smallest_sufficient() -> None:
    presets = [
        _preset("64k", 65_536, default=True),
        _preset("128k", 131_072),
        _preset("max", 262_144),
    ]
    result = select_runtime_preset(presets, required_tokens=70_000)
    assert result.preset.preset_key == "128k"
    assert result.reason == "smallest_sufficient"


def test_prefer_default_can_opt_out_of_loaded_larger_runtime() -> None:
    presets = [
        _preset("64k", 65_536, default=True),
        _preset("max", 262_144, loaded_key="runtime:max"),
    ]
    result = select_runtime_preset(
        presets,
        required_tokens=16_000,
        loaded_runtime_keys={"runtime:max"},
        selection="prefer_default",
    )
    assert result.preset.preset_key == "64k"
    assert result.reused_loaded_runtime is False


def test_disabled_max_is_not_advertised_or_selected() -> None:
    presets = [
        _preset("64k", 65_536, default=True),
        _preset("max", 262_144, enabled=False),
    ]
    assert advertised_effective_max(presets) == 65_536
    with pytest.raises(ContextCapacityError):
        select_runtime_preset(presets, required_tokens=70_000)
