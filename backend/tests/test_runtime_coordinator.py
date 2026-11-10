from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ocabra.core.runtime_coordinator import (
    OutputTokenLimitError,
    estimate_request_token_budget,
    plan_profile_runtime,
)


def _preset(
    key: str,
    context: int | None,
    *,
    default: bool = False,
    mode: str = "fixed",
) -> SimpleNamespace:
    return SimpleNamespace(
        preset_key=key,
        context_mode=mode,
        context_tokens=context,
        parallelism=2,
        kv_cache_dtype="q8",
        placement_policy="forced",
        preferred_gpus=[1],
        execution_policy="auto",
        load_policy="warm" if default else "on_demand",
        enabled=True,
        is_default=default,
        priority=10,
    )


def _profile() -> SimpleNamespace:
    return SimpleNamespace(
        profile_id="gemma4:12b",
        base_model_id="ollama/gemma4:12b",
        load_overrides=None,
        runtime_policy={"selection": "prefer_loaded", "max_output_tokens": 65_536},
        runtime_presets=[
            _preset("64k", 65_536, default=True),
            _preset("max", None, mode="max"),
        ],
    )


def _capacity_rows() -> list[dict]:
    return [
        {
            "preset_key": "64k",
            "resolved": True,
            "fits": True,
            "effective_context": 65_536,
            "target_gpus": [1],
            "runtime_key": "runtime:64",
        },
        {
            "preset_key": "max",
            "resolved": True,
            "fits": True,
            "effective_context": 262_144,
            "target_gpus": [1],
            "runtime_key": "runtime:max",
        },
    ]


def test_estimate_request_budget_includes_reserved_output() -> None:
    body = {
        "messages": [{"role": "user", "content": "hola " * 300}],
        "max_tokens": 8_000,
    }
    assert estimate_request_token_budget(body) > 8_000


def test_estimate_request_budget_reserves_context_for_images() -> None:
    text_only = {
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hola"}]}],
        "max_tokens": 100,
    }
    with_images = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hola"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,a"}},
                    {"type": "input_image", "image_url": "data:image/png;base64,b"},
                ],
            }
        ],
        "max_tokens": 100,
    }

    assert (
        estimate_request_token_budget(with_images) - estimate_request_token_budget(text_only) == 256
    )


def test_estimate_ollama_budget_honours_num_ctx_and_num_predict() -> None:
    body = {
        "prompt": "hola",
        "options": {"num_ctx": 65_536, "num_predict": 128},
    }

    assert estimate_request_token_budget(body) == 65_536


def test_estimate_ollama_budget_uses_num_predict_without_num_ctx() -> None:
    body = {"prompt": "hola", "options": {"num_predict": 12_000}}

    assert estimate_request_token_budget(body) > 12_000


def test_estimate_budget_output_reservation_defaults_to_configurable_setting(monkeypatch) -> None:
    """A client that sets no max_tokens/max_completion_tokens/num_predict
    gets settings.runtime_preset_default_output_reservation_tokens reserved
    for output, not a hardcoded value — lets an operator raise or lower how
    aggressively uncapped requests push toward larger (more expensive)
    presets, per user request 2026-10-28."""
    from ocabra.config import settings

    monkeypatch.setattr(settings, "runtime_preset_default_output_reservation_tokens", 777)
    body = {"messages": [{"role": "user", "content": "hola"}]}

    budget = estimate_request_token_budget(body)

    assert budget >= 777
    monkeypatch.setattr(settings, "runtime_preset_default_output_reservation_tokens", 55_000)
    assert estimate_request_token_budget(body) >= 55_000


def test_default_output_reservation_param_overrides_the_global_setting(monkeypatch) -> None:
    """2026-10-29: a per-profile override (runtime_policy.default_output_
    reservation_tokens, see plan_profile_runtime) must win over the global
    setting — this is what lets reasoning-heavy profiles (gemma4, qwen3.8)
    reserve more room by default without raising it for every other model."""
    from ocabra.config import settings

    monkeypatch.setattr(settings, "runtime_preset_default_output_reservation_tokens", 4_096)
    body = {"messages": [{"role": "user", "content": "hola"}]}

    assert estimate_request_token_budget(body) < 16_384
    assert estimate_request_token_budget(body, default_output_reservation=16_384) >= 16_384


@pytest.mark.asyncio
async def test_profile_without_presets_still_computes_admission_budget() -> None:
    profile = SimpleNamespace(
        base_model_id="llama_cpp/gemma4-26b",
        load_overrides=None,
        runtime_policy={"default_output_reservation_tokens": 16_384},
        runtime_presets=[],
    )

    plan = await plan_profile_runtime(
        profile,
        SimpleNamespace(),
        {"messages": [{"role": "user", "content": "hola"}]},
    )

    assert plan.worker_key == profile.base_model_id
    assert plan.required_tokens is not None
    assert plan.required_tokens >= 16_384


@pytest.mark.asyncio
async def test_fixed_profile_rejects_output_above_declared_hard_cap() -> None:
    profile = SimpleNamespace(
        base_model_id="llama_cpp/gemma4-26b",
        load_overrides=None,
        runtime_policy={"max_context_tokens": 262_144, "max_output_tokens": 65_536},
        runtime_presets=[],
    )

    with pytest.raises(OutputTokenLimitError) as raised:
        await plan_profile_runtime(
            profile,
            SimpleNamespace(),
            {
                "messages": [{"role": "user", "content": "hola"}],
                "max_tokens": 500_000,
            },
        )

    assert raised.value.parameter == "max_tokens"
    assert raised.value.maximum == 65_536


@pytest.mark.asyncio
async def test_default_runtime_is_selected_for_normal_request(monkeypatch) -> None:
    profile = _profile()
    state = SimpleNamespace(
        model_id=profile.base_model_id,
        backend_type="ollama",
        status=SimpleNamespace(value="configured"),
        load_policy=SimpleNamespace(value="warm"),
        extra_config={},
    )
    manager = SimpleNamespace(
        get_state=AsyncMock(return_value=state),
        gpu_manager=SimpleNamespace(get_all_states=AsyncMock(return_value=[])),
        list_states=AsyncMock(return_value=[state]),
    )
    monkeypatch.setattr(
        "ocabra.core.runtime_coordinator.runtime_capacity_cache.resolve",
        AsyncMock(return_value=_capacity_rows()),
    )

    plan = await plan_profile_runtime(
        profile,
        manager,
        {"messages": [{"role": "user", "content": "hola"}], "max_tokens": 100},
    )

    assert plan.preset_key == "64k"
    assert plan.load_policy == "warm"
    assert plan.preferred_gpu == 1
    assert plan.load_overrides["use_case"] == {
        "context": 65_536,
        "slots": 2,
        "kv_dtype": "q8",
    }


@pytest.mark.asyncio
async def test_explicit_output_above_hard_cap_is_not_reported_as_context_pressure(
    monkeypatch,
) -> None:
    profile = _profile()
    state = SimpleNamespace(
        model_id=profile.base_model_id,
        backend_type="ollama",
        status=SimpleNamespace(value="configured"),
        load_policy=SimpleNamespace(value="warm"),
        extra_config={},
    )
    manager = SimpleNamespace(
        get_state=AsyncMock(return_value=state),
        gpu_manager=SimpleNamespace(get_all_states=AsyncMock(return_value=[])),
        list_states=AsyncMock(return_value=[state]),
    )
    monkeypatch.setattr(
        "ocabra.core.runtime_coordinator.runtime_capacity_cache.resolve",
        AsyncMock(return_value=_capacity_rows()),
    )

    with pytest.raises(OutputTokenLimitError) as raised:
        await plan_profile_runtime(
            profile,
            manager,
            {
                "messages": [{"role": "user", "content": "hola"}],
                "max_tokens": 500_000,
            },
        )

    assert raised.value.parameter == "max_tokens"
    assert raised.value.requested == 500_000
    assert raised.value.maximum == 65_536


@pytest.mark.asyncio
async def test_llama_cpp_dual_preset_replaces_single_gpu_base_placement(monkeypatch) -> None:
    preset = _preset("max", 262_144, default=False)
    preset.preferred_gpus = [0, 1]
    profile = SimpleNamespace(
        profile_id="gemma4:26b",
        base_model_id="llama_cpp/gemma4-26b",
        load_overrides=None,
        runtime_policy={"selection": "prefer_loaded"},
        runtime_presets=[preset],
    )
    state = SimpleNamespace(
        model_id=profile.base_model_id,
        backend_type="llama_cpp",
        status=SimpleNamespace(value="configured"),
        load_policy=SimpleNamespace(value="warm"),
        extra_config={
            "llama_cpp": {
                "disabled_gpus": [0],
                "split_mode": "none",
                "vram_estimate_mb": 23_632,
            }
        },
    )
    gpu_states = [
        SimpleNamespace(index=0, total_vram_mb=12_288),
        SimpleNamespace(index=1, total_vram_mb=24_576),
    ]
    manager = SimpleNamespace(
        get_state=AsyncMock(return_value=state),
        gpu_manager=SimpleNamespace(get_all_states=AsyncMock(return_value=gpu_states)),
        list_states=AsyncMock(return_value=[state]),
    )
    monkeypatch.setattr(
        "ocabra.core.runtime_coordinator.runtime_capacity_cache.resolve",
        AsyncMock(
            return_value=[
                {
                    "preset_key": "max",
                    "resolved": True,
                    "fits": True,
                    "effective_context": 262_144,
                    "effective_parallelism": 1,
                    "target_gpus": [0, 1],
                    "runtime_key": "runtime:max",
                }
            ]
        ),
    )

    plan = await plan_profile_runtime(
        profile,
        manager,
        {
            "messages": [{"role": "user", "content": "hola"}],
            "max_tokens": 1,
            "ocabra": {"context_preset": "max"},
        },
    )

    llama = plan.load_overrides["llama_cpp"]
    assert llama["disabled_gpus"] == []
    assert llama["split_mode"] == "layer"
    assert llama["tensor_split"] == [1.0, 2.0]
    assert llama["main_gpu"] == 1
    assert llama["vram_estimate_mb"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("configured", "expected"),
    [("tensor", "tensor"), ("row", "row"), ("layer", "layer"), (None, "layer")],
)
async def test_llama_cpp_dual_preset_keeps_tensor_or_row_split(
    monkeypatch, configured, expected
) -> None:
    """A dual preset used to force "layer"; a profile asking for "tensor"
    (both GPUs compute every layer together) must get it."""
    preset = _preset("extended", 262_144, default=False)
    preset.preferred_gpus = [0, 1]
    profile = SimpleNamespace(
        profile_id="qwen3.8:27b",
        base_model_id="llamacpp_mks/qwen3.8:27b",
        load_overrides=None,
        runtime_policy={"selection": "prefer_loaded"},
        runtime_presets=[preset],
    )
    llama_cfg = {"tensor_split": [1, 3], "main_gpu": 1}
    if configured is not None:
        llama_cfg["split_mode"] = configured
    state = SimpleNamespace(
        model_id=profile.base_model_id,
        backend_type="llamacpp_mks",
        status=SimpleNamespace(value="configured"),
        load_policy=SimpleNamespace(value="warm"),
        extra_config={"llama_cpp": llama_cfg},
    )
    gpu_states = [
        SimpleNamespace(index=0, total_vram_mb=12_288),
        SimpleNamespace(index=1, total_vram_mb=24_576),
    ]
    manager = SimpleNamespace(
        get_state=AsyncMock(return_value=state),
        gpu_manager=SimpleNamespace(get_all_states=AsyncMock(return_value=gpu_states)),
        list_states=AsyncMock(return_value=[state]),
    )
    monkeypatch.setattr(
        "ocabra.core.runtime_coordinator.runtime_capacity_cache.resolve",
        AsyncMock(
            return_value=[
                {
                    "preset_key": "extended",
                    "resolved": True,
                    "fits": True,
                    "effective_context": 262_144,
                    "effective_parallelism": 4,
                    "target_gpus": [0, 1],
                    "runtime_key": "runtime:extended",
                }
            ]
        ),
    )

    plan = await plan_profile_runtime(
        profile,
        manager,
        {
            "messages": [{"role": "user", "content": "hola"}],
            "max_tokens": 1,
            "ocabra": {"context_preset": "extended"},
        },
    )

    assert plan.load_overrides["llama_cpp"]["split_mode"] == expected


@pytest.mark.asyncio
async def test_llama_cpp_dual_preset_keeps_calibrated_matching_split(monkeypatch) -> None:
    preset = _preset("extended", 229_376, default=False)
    preset.preferred_gpus = [0, 1]
    profile = SimpleNamespace(
        profile_id="qwen3.8:27b-abliterated",
        base_model_id="llama_cpp/qwen3.8-27b-abliterated",
        load_overrides=None,
        runtime_policy={"selection": "prefer_loaded"},
        runtime_presets=[preset],
    )
    state = SimpleNamespace(
        model_id=profile.base_model_id,
        backend_type="llama_cpp",
        status=SimpleNamespace(value="configured"),
        load_policy=SimpleNamespace(value="warm"),
        extra_config={
            "llama_cpp": {
                "tensor_split": [1, 2.3],
                "main_gpu": 1,
                "split_mode": "layer",
            }
        },
    )
    gpu_states = [
        SimpleNamespace(index=0, total_vram_mb=12_288),
        SimpleNamespace(index=1, total_vram_mb=24_576),
    ]
    manager = SimpleNamespace(
        get_state=AsyncMock(return_value=state),
        gpu_manager=SimpleNamespace(get_all_states=AsyncMock(return_value=gpu_states)),
        list_states=AsyncMock(return_value=[state]),
    )
    monkeypatch.setattr(
        "ocabra.core.runtime_coordinator.runtime_capacity_cache.resolve",
        AsyncMock(
            return_value=[
                {
                    "preset_key": "extended",
                    "resolved": True,
                    "fits": True,
                    "effective_context": 229_376,
                    "effective_parallelism": 4,
                    "target_gpus": [0, 1],
                    "runtime_key": "runtime:extended",
                }
            ]
        ),
    )

    plan = await plan_profile_runtime(
        profile,
        manager,
        {"ocabra": {"context_preset": "extended"}, "max_tokens": 1},
    )

    llama = plan.load_overrides["llama_cpp"]
    assert llama["tensor_split"] == [1.0, 2.3]
    assert llama["main_gpu"] == 1


@pytest.mark.asyncio
async def test_llama_cpp_single_preset_keeps_matching_measured_estimate(monkeypatch) -> None:
    preset = _preset("normal", 32_768, default=True)
    preset.parallelism = 4
    profile = SimpleNamespace(
        profile_id="gemma4:26b",
        base_model_id="llama_cpp/gemma4-26b",
        load_overrides=None,
        runtime_policy={"selection": "prefer_loaded"},
        runtime_presets=[preset],
    )
    state = SimpleNamespace(
        model_id=profile.base_model_id,
        backend_type="llama_cpp",
        status=SimpleNamespace(value="configured"),
        load_policy=SimpleNamespace(value="warm"),
        extra_config={
            "llama_cpp": {
                "disabled_gpus": [0],
                "vram_estimate_mb": 23_632,
            }
        },
    )
    gpu_states = [
        SimpleNamespace(index=0, total_vram_mb=12_288),
        SimpleNamespace(index=1, total_vram_mb=24_576),
    ]
    manager = SimpleNamespace(
        get_state=AsyncMock(return_value=state),
        gpu_manager=SimpleNamespace(get_all_states=AsyncMock(return_value=gpu_states)),
        list_states=AsyncMock(return_value=[state]),
    )
    monkeypatch.setattr(
        "ocabra.core.runtime_coordinator.runtime_capacity_cache.resolve",
        AsyncMock(
            return_value=[
                {
                    "preset_key": "normal",
                    "resolved": True,
                    "fits": True,
                    "effective_context": 32_768,
                    "effective_parallelism": 4,
                    "target_gpus": [1],
                    "runtime_key": "runtime:normal",
                }
            ]
        ),
    )

    plan = await plan_profile_runtime(
        profile,
        manager,
        {"messages": [{"role": "user", "content": "hola"}], "max_tokens": 1},
    )

    llama = plan.load_overrides["llama_cpp"]
    assert llama["disabled_gpus"] == [0]
    assert llama["vram_estimate_mb"] == 23_632
    assert llama["split_mode"] == "none"


@pytest.mark.asyncio
async def test_llama_cpp_unified_preset_materializes_aggregate_pool(monkeypatch) -> None:
    preset = _preset("normal", 98_304, default=True)
    preset.parallelism = 4
    profile = SimpleNamespace(
        profile_id="qwen3.8:27b",
        base_model_id="llama_cpp/qwen3.8-27b",
        load_overrides=None,
        runtime_policy={
            "selection": "prefer_loaded",
            "validated_context_tokens": {"normal": 98_304},
            "validated_vram_mb": {"normal": 24_000},
        },
        runtime_presets=[preset],
    )
    state = SimpleNamespace(
        model_id=profile.base_model_id,
        backend_type="llama_cpp",
        status=SimpleNamespace(value="configured"),
        load_policy=SimpleNamespace(value="warm"),
        extra_config={
            "llama_cpp": {
                "ctx_size": 262_144,
                "parallel_slots": 4,
                "kv_unified": True,
                "disabled_gpus": [],
            }
        },
    )
    gpu_states = [
        SimpleNamespace(index=0, total_vram_mb=12_288),
        SimpleNamespace(index=1, total_vram_mb=24_576),
    ]
    manager = SimpleNamespace(
        get_state=AsyncMock(return_value=state),
        gpu_manager=SimpleNamespace(get_all_states=AsyncMock(return_value=gpu_states)),
        list_states=AsyncMock(return_value=[state]),
    )
    monkeypatch.setattr(
        "ocabra.core.runtime_coordinator.runtime_capacity_cache.resolve",
        AsyncMock(
            return_value=[
                {
                    "preset_key": "normal",
                    "resolved": True,
                    "fits": True,
                    "effective_context": 98_304,
                    "effective_parallelism": 4,
                    "target_gpus": [1],
                    "runtime_key": "runtime:normal",
                }
            ]
        ),
    )

    plan = await plan_profile_runtime(
        profile,
        manager,
        {"messages": [{"role": "user", "content": "hola"}], "max_tokens": 1},
    )

    llama = plan.load_overrides["llama_cpp"]
    assert llama["ctx_size"] == 98_304
    assert llama["parallel_slots"] == 4
    assert plan.load_overrides["use_case"]["context_is_pool"] is True
    assert plan.load_overrides["_runtime"]["context_is_pool"] is True
    assert plan.load_overrides["_runtime"]["capacity_validated"] is True
    assert plan.load_overrides["_runtime"]["capacity_vram_mb"] == 24_000


@pytest.mark.asyncio
async def test_loaded_max_is_reused_for_small_request(monkeypatch) -> None:
    profile = _profile()
    base_state = SimpleNamespace(
        model_id=profile.base_model_id,
        backend_type="ollama",
        status=SimpleNamespace(value="configured"),
        load_policy=SimpleNamespace(value="warm"),
        extra_config={},
    )
    max_state = SimpleNamespace(
        model_id="ollama/gemma4:12b::max",
        status=SimpleNamespace(value="loaded"),
        extra_config={"_runtime": {"runtime_key": "runtime:max"}},
    )
    manager = SimpleNamespace(
        get_state=AsyncMock(return_value=base_state),
        gpu_manager=SimpleNamespace(get_all_states=AsyncMock(return_value=[])),
        list_states=AsyncMock(return_value=[base_state, max_state]),
    )
    monkeypatch.setattr(
        "ocabra.core.runtime_coordinator.runtime_capacity_cache.resolve",
        AsyncMock(return_value=_capacity_rows()),
    )

    plan = await plan_profile_runtime(
        profile,
        manager,
        {"messages": [{"role": "user", "content": "hola"}], "max_tokens": 100},
    )

    assert plan.preset_key == "max"
    assert plan.selection_reason == "loaded"


@pytest.mark.asyncio
async def test_profile_output_reservation_override_pushes_to_a_bigger_preset(monkeypatch) -> None:
    """A profile with runtime_policy.default_output_reservation_tokens set
    reserves that much output by default — even for a tiny prompt with no
    max_tokens — instead of the global 4096 fallback. Verified end-to-end
    through plan_profile_runtime (not just the estimator in isolation):
    without the override the tiny prompt fits '64k'; with a large enough
    override it no longer does and 'max' is selected instead."""
    profile = _profile()
    profile.runtime_policy = {
        "selection": "prefer_loaded",
        "default_output_reservation_tokens": 100_000,
    }
    state = SimpleNamespace(
        model_id=profile.base_model_id,
        backend_type="ollama",
        status=SimpleNamespace(value="configured"),
        load_policy=SimpleNamespace(value="warm"),
        extra_config={},
    )
    manager = SimpleNamespace(
        get_state=AsyncMock(return_value=state),
        gpu_manager=SimpleNamespace(get_all_states=AsyncMock(return_value=[])),
        list_states=AsyncMock(return_value=[state]),
    )
    monkeypatch.setattr(
        "ocabra.core.runtime_coordinator.runtime_capacity_cache.resolve",
        AsyncMock(return_value=_capacity_rows()),
    )

    # No max_tokens at all — this is exactly the "client forgot to set it"
    # case that truncated gemma4's reasoning output in production.
    plan = await plan_profile_runtime(
        profile,
        manager,
        {"messages": [{"role": "user", "content": "hola"}]},
    )

    assert plan.preset_key == "max"


@pytest.mark.asyncio
async def test_explicit_preset_override_is_honoured(monkeypatch) -> None:
    profile = _profile()
    state = SimpleNamespace(
        model_id=profile.base_model_id,
        backend_type="ollama",
        status=SimpleNamespace(value="configured"),
        load_policy=SimpleNamespace(value="warm"),
        extra_config={},
    )
    manager = SimpleNamespace(
        get_state=AsyncMock(return_value=state),
        gpu_manager=SimpleNamespace(get_all_states=AsyncMock(return_value=[])),
        list_states=AsyncMock(return_value=[state]),
    )
    monkeypatch.setattr(
        "ocabra.core.runtime_coordinator.runtime_capacity_cache.resolve",
        AsyncMock(return_value=_capacity_rows()),
    )

    plan = await plan_profile_runtime(
        profile,
        manager,
        {
            "messages": [{"role": "user", "content": "hola"}],
            "max_tokens": 100,
            "ocabra": {"context_preset": "max"},
        },
    )

    assert plan.preset_key == "max"
    assert plan.selection_reason == "explicit"


@pytest.mark.asyncio
async def test_request_size_does_not_change_runtime_worker_key(monkeypatch) -> None:
    profile = _profile()
    state = SimpleNamespace(
        model_id=profile.base_model_id,
        backend_type="ollama",
        status=SimpleNamespace(value="configured"),
        load_policy=SimpleNamespace(value="warm"),
        extra_config={},
    )
    manager = SimpleNamespace(
        get_state=AsyncMock(return_value=state),
        gpu_manager=SimpleNamespace(get_all_states=AsyncMock(return_value=[])),
        list_states=AsyncMock(return_value=[state]),
    )
    monkeypatch.setattr(
        "ocabra.core.runtime_coordinator.runtime_capacity_cache.resolve",
        AsyncMock(return_value=_capacity_rows()),
    )

    short = await plan_profile_runtime(
        profile, manager, {"messages": [{"role": "user", "content": "hola"}]}
    )
    longer = await plan_profile_runtime(
        profile,
        manager,
        {"messages": [{"role": "user", "content": "hola " * 5000}]},
    )

    assert short.preset_key == longer.preset_key == "64k"
    assert short.worker_key == longer.worker_key
