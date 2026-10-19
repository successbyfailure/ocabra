from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ocabra.core.runtime_coordinator import (
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
        runtime_policy={"selection": "prefer_loaded"},
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


def test_estimate_ollama_budget_honours_num_ctx_and_num_predict() -> None:
    body = {
        "prompt": "hola",
        "options": {"num_ctx": 65_536, "num_predict": 128},
    }

    assert estimate_request_token_budget(body) == 65_536


def test_estimate_ollama_budget_uses_num_predict_without_num_ctx() -> None:
    body = {"prompt": "hola", "options": {"num_predict": 12_000}}

    assert estimate_request_token_budget(body) > 12_000


@pytest.mark.asyncio
async def test_default_runtime_is_selected_for_normal_request(monkeypatch) -> None:
    profile = _profile()
    state = SimpleNamespace(
        model_id=profile.base_model_id,
        backend_type="ollama",
        status=SimpleNamespace(value="configured"),
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
async def test_loaded_max_is_reused_for_small_request(monkeypatch) -> None:
    profile = _profile()
    base_state = SimpleNamespace(
        model_id=profile.base_model_id,
        backend_type="ollama",
        status=SimpleNamespace(value="configured"),
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
async def test_explicit_preset_override_is_honoured(monkeypatch) -> None:
    profile = _profile()
    state = SimpleNamespace(
        model_id=profile.base_model_id,
        backend_type="ollama",
        status=SimpleNamespace(value="configured"),
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
