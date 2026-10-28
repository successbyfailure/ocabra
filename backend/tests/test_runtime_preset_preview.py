"""Tests for the live runtime-preset preview endpoint (ocabra/api/internal/profiles.py).

Added for the preset-editor UI work: lets the admin see a draft preset's
resolved capacity (fits / effective context / target GPUs) before saving it,
instead of save -> "Calcular capacidad" -> edit -> save again, which is what
made configuring qwen3.8:27b/gemma4:26b's preset ladders this tedious all
night (2026-10-26/27).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ocabra.api.internal import profiles as profiles_api
from ocabra.core.model_manager import ModelState
from ocabra.core.vram_planner import ModelArch
from ocabra.schemas.profiles import RuntimePresetCreate


class _FakeGpuState:
    def __init__(self, index: int, total_vram_mb: int) -> None:
        self.index = index
        self.total_vram_mb = total_vram_mb


def _make_request(profile, state, gpu_states):
    registry = SimpleNamespace(get=AsyncMock(return_value=profile))
    model_manager = SimpleNamespace(
        get_state=AsyncMock(return_value=state),
        gpu_manager=SimpleNamespace(get_all_states=AsyncMock(return_value=gpu_states)),
    )
    app_state = SimpleNamespace(profile_registry=registry, model_manager=model_manager)
    return SimpleNamespace(app=SimpleNamespace(state=app_state))


def _make_profile():
    return SimpleNamespace(
        profile_kind="execution",
        base_model_id="llama_cpp/qwen3.8-27b",
        load_overrides=None,
        runtime_policy=None,
    )


async def test_preview_resolves_a_draft_preset_without_persisting_anything():
    profile = _make_profile()
    state = ModelState(
        model_id="llama_cpp/qwen3.8-27b",
        display_name="qwen3.8:27b",
        backend_type="llama_cpp",
        backend_model_id="qwen3.8-27b",
        extra_config={
            "llama_cpp": {"model_path": "/does/not/matter/for/this/fake"},
        },
    )
    gpu_states = [_FakeGpuState(0, 12288), _FakeGpuState(1, 24576)]
    request = _make_request(profile, state, gpu_states)
    draft = RuntimePresetCreate(
        preset_key="preview",
        label="Preview",
        context_mode="fixed",
        context_tokens=32768,
        parallelism=1,
        kv_cache_dtype="q8",
        placement_policy="forced",
        preferred_gpus=[1],
    )

    arch = ModelArch(layers=65, n_kv_heads=4, key_length=256, value_length=256, context_length=262144)
    fake_resolution = SimpleNamespace(arch=arch, weights_mb=16032.0, note="")

    async def _fake_resolve_arch(_state):
        return fake_resolution

    import ocabra.core.runtime_preset_capacity as rpc

    original = rpc.resolve_arch_and_weights
    rpc.resolve_arch_and_weights = _fake_resolve_arch
    try:
        result = await profiles_api.preview_runtime_preset(
            "qwen3.8:27b", draft, request, _user=SimpleNamespace()
        )
    finally:
        rpc.resolve_arch_and_weights = original

    assert result["preset_key"] == "preview"
    assert result["requested_context"] == 32768
    assert result["target_gpus"] == [1]
    assert result["fits"] is True


async def test_preview_404s_when_profile_missing():
    registry = SimpleNamespace(get=AsyncMock(return_value=None))
    app_state = SimpleNamespace(profile_registry=registry, model_manager=SimpleNamespace())
    request = SimpleNamespace(app=SimpleNamespace(state=app_state))
    draft = RuntimePresetCreate(
        preset_key="preview", label="Preview", context_tokens=1024, parallelism=1,
    )

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        await profiles_api.preview_runtime_preset(
            "missing-profile", draft, request, _user=SimpleNamespace()
        )
    assert exc_info.value.status_code == 404
