from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ocabra.core.runtime_preset_capacity import resolve_profile_runtime_capacities
from ocabra.core.vram_capacity import ArchResolution
from ocabra.core.vram_planner import ModelArch


def _preset(
    key: str,
    *,
    mode: str,
    tokens: int | None,
    placement: str = "auto",
    gpus: list[int] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        preset_key=key,
        context_mode=mode,
        context_tokens=tokens,
        parallelism=2,
        kv_cache_dtype="q8",
        placement_policy=placement,
        preferred_gpus=gpus,
        priority=10,
    )


@pytest.mark.asyncio
async def test_resolves_fixed_and_max_on_the_3090(monkeypatch) -> None:
    arch = ModelArch(
        layers=32,
        n_kv_heads=8,
        key_length=128,
        value_length=128,
        context_length=262_144,
    )
    monkeypatch.setattr(
        "ocabra.core.runtime_preset_capacity.resolve_arch_and_weights",
        AsyncMock(return_value=ArchResolution(arch, 10_000, "")),
    )
    profile = SimpleNamespace(
        base_model_id="ollama/gemma4:12b",
        load_overrides=None,
        runtime_policy=None,
        runtime_presets=[
            _preset("64k", mode="fixed", tokens=65_536, placement="forced", gpus=[1]),
            _preset("max", mode="max", tokens=None, placement="forced", gpus=[1]),
        ],
    )
    state = SimpleNamespace(
        backend_type="ollama",
        preferred_gpu=1,
        extra_config={},
    )
    gpu_states = [
        SimpleNamespace(index=0, total_vram_mb=12_288),
        SimpleNamespace(index=1, total_vram_mb=24_576),
    ]

    rows = await resolve_profile_runtime_capacities(profile, state, gpu_states)

    assert rows[0]["preset_key"] == "64k"
    assert rows[0]["effective_context"] == 65_536
    assert rows[0]["target_gpus"] == [1]
    assert rows[0]["fits"] is True
    assert rows[1]["effective_context"] == 113_408
    assert rows[1]["runtime_key"].startswith("runtime:")


@pytest.mark.asyncio
async def test_does_not_fake_explicit_multi_gpu_capacity(monkeypatch) -> None:
    arch = ModelArch(32, 8, 128, 128, context_length=262_144)
    monkeypatch.setattr(
        "ocabra.core.runtime_preset_capacity.resolve_arch_and_weights",
        AsyncMock(return_value=ArchResolution(arch, 10_000, "")),
    )
    profile = SimpleNamespace(
        base_model_id="ollama/gemma4:12b",
        load_overrides=None,
        runtime_policy=None,
        runtime_presets=[
            _preset("dual", mode="max", tokens=None, placement="forced", gpus=[0, 1])
        ],
    )
    state = SimpleNamespace(backend_type="ollama", preferred_gpu=1, extra_config={})
    gpus = [
        SimpleNamespace(index=0, total_vram_mb=12_288),
        SimpleNamespace(index=1, total_vram_mb=24_576),
    ]

    row = (await resolve_profile_runtime_capacities(profile, state, gpus))[0]

    assert row["resolved"] is False
    assert row["effective_context"] is None
    assert "multi-GPU" in row["warnings"][0]


@pytest.mark.asyncio
async def test_profile_policy_caps_calculated_max(monkeypatch) -> None:
    arch = ModelArch(32, 2, 128, 128, context_length=262_144)
    monkeypatch.setattr(
        "ocabra.core.runtime_preset_capacity.resolve_arch_and_weights",
        AsyncMock(return_value=ArchResolution(arch, 10_000, "")),
    )
    profile = SimpleNamespace(
        profile_id="qwen3.8:27b",
        base_model_id="ollama/qwen3.8:27b",
        load_overrides=None,
        runtime_policy={"max_context_tokens": 163_840},
        runtime_presets=[
            SimpleNamespace(
                preset_key="max",
                context_mode="max",
                context_tokens=None,
                parallelism=1,
                kv_cache_dtype="q8",
                placement_policy="forced",
                preferred_gpus=[1],
                priority=10,
            )
        ],
    )
    state = SimpleNamespace(
        backend_type="ollama",
        backend_model_id="qwen3.8:27b",
        preferred_gpu=1,
        extra_config={},
        updated_at=None,
    )
    gpus = [SimpleNamespace(index=1, total_vram_mb=24_576)]

    row = (await resolve_profile_runtime_capacities(profile, state, gpus))[0]

    assert row["hardware_max"] > 163_840
    assert row["policy_max"] == 163_840
    assert row["effective_context"] == 163_840
    assert row["fits"] is True


@pytest.mark.asyncio
async def test_runtime_parallelism_cap_is_used_for_capacity_and_identity(monkeypatch) -> None:
    arch = ModelArch(32, 8, 128, 128, context_length=262_144)
    monkeypatch.setattr(
        "ocabra.core.runtime_preset_capacity.resolve_arch_and_weights",
        AsyncMock(return_value=ArchResolution(arch, 10_000, "")),
    )
    profile = SimpleNamespace(
        base_model_id="ollama/qwen3.8:27b",
        load_overrides=None,
        runtime_policy=None,
        runtime_presets=[_preset("64k", mode="fixed", tokens=65_536)],
    )
    state = SimpleNamespace(
        backend_type="ollama",
        preferred_gpu=1,
        extra_config={"runtime_capabilities": {"max_parallelism": 1}},
    )

    row = (
        await resolve_profile_runtime_capacities(
            profile,
            state,
            [SimpleNamespace(index=1, total_vram_mb=24_576)],
        )
    )[0]

    assert row["requested_parallelism"] == 2
    assert row["effective_parallelism"] == 1
    assert row["fits"] is True
    assert any("p2" in warning and "p1" in warning for warning in row["warnings"])
