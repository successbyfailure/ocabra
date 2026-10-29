"""Model resolution for the OpenAI Realtime WebSocket (profiles, routers, legacy ids)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ocabra.api.openai.realtime import (
    resolve_realtime_llm_runtime_plan,
    resolve_realtime_model_id,
)


def _registry(profiles: dict) -> MagicMock:
    registry = MagicMock()
    registry.get = AsyncMock(side_effect=lambda pid: profiles.get(pid))
    return registry


def _profile(pid: str, base: str | None, enabled: bool = True) -> SimpleNamespace:
    return SimpleNamespace(profile_id=pid, base_model_id=base, enabled=enabled)


@pytest.mark.asyncio
async def test_profile_resolves_to_its_base_model() -> None:
    registry = _registry({"qwen3:8b": _profile("qwen3:8b", "ollama/qwen3:8b")})
    resolved = await resolve_realtime_model_id("qwen3:8b", profile_registry=registry, model_manager=MagicMock())
    assert resolved == "ollama/qwen3:8b"


@pytest.mark.asyncio
async def test_router_profile_resolves_through_the_router_resolver() -> None:
    router = _profile("recomendado-sota", None)
    target = _profile("qwen3.8:27b", "llama_cpp/qwen3.8-27b")
    registry = _registry({"recomendado-sota": router, "qwen3.8:27b": target})
    resolver = MagicMock()
    resolver.is_router.return_value = True
    resolver.pick = AsyncMock(return_value=SimpleNamespace(target_profile_id="qwen3.8:27b"))

    resolved = await resolve_realtime_model_id(
        "recomendado-sota", profile_registry=registry, model_manager=MagicMock(), router_resolver=resolver
    )

    assert resolved == "llama_cpp/qwen3.8-27b"


@pytest.mark.asyncio
async def test_router_without_target_returns_none_instead_of_a_modelless_session() -> None:
    registry = _registry({"recomendado-stt": _profile("recomendado-stt", None)})
    resolver = MagicMock()
    resolver.is_router.return_value = False

    resolved = await resolve_realtime_model_id(
        "recomendado-stt", profile_registry=registry, model_manager=MagicMock(), router_resolver=resolver
    )

    assert resolved is None


@pytest.mark.asyncio
async def test_router_pick_failure_returns_none() -> None:
    registry = _registry({"r": _profile("r", None)})
    resolver = MagicMock()
    resolver.is_router.return_value = True
    resolver.pick = AsyncMock(side_effect=RuntimeError("no routable target"))

    resolved = await resolve_realtime_model_id(
        "r", profile_registry=registry, model_manager=MagicMock(), router_resolver=resolver
    )

    assert resolved is None


@pytest.mark.asyncio
async def test_legacy_backend_id_falls_back_to_model_state() -> None:
    manager = MagicMock()
    manager.list_states = AsyncMock(
        return_value=[SimpleNamespace(model_id="ollama/qwen3:8b", backend_model_id="qwen3:8b")]
    )
    resolved = await resolve_realtime_model_id("qwen3:8b", profile_registry=_registry({}), model_manager=manager)
    assert resolved == "ollama/qwen3:8b"


@pytest.mark.asyncio
async def test_canonical_id_is_kept_as_is() -> None:
    resolved = await resolve_realtime_model_id(
        "vllm/org/model", profile_registry=_registry({}), model_manager=MagicMock()
    )
    assert resolved == "vllm/org/model"


# ── LLM worker-key resolution (2026-10-29) ──────────────────────────────
#
# Reproduced live: a realtime session loaded the bare base_model_id as its
# own separate worker, orphaned from whatever preset-derived instance REST
# traffic (/v1/chat/completions) was already using for the same profile —
# on a single GPU that doesn't fit both, every new realtime session evicted
# the other and cold-started again. resolve_realtime_llm_runtime_plan()
# resolves the same preset-aware worker_key REST endpoints use.


def _profile_registry_with_profiles(profiles: list) -> MagicMock:
    registry = MagicMock()
    registry.list_by_model = AsyncMock(return_value=profiles)
    return registry


@pytest.mark.asyncio
async def test_profile_without_runtime_presets_falls_back_to_base_model_id() -> None:
    profile = SimpleNamespace(profile_id="ollama/qwen3:8b", is_default=True, runtime_presets=[])
    registry = _profile_registry_with_profiles([profile])

    worker_key, load_overrides, preferred_gpu, load_policy = await resolve_realtime_llm_runtime_plan(
        "ollama/qwen3:8b", profile_registry=registry, model_manager=MagicMock()
    )

    assert worker_key == "ollama/qwen3:8b"
    assert load_overrides is None
    assert preferred_gpu is None
    assert load_policy is None


@pytest.mark.asyncio
async def test_multi_preset_profile_resolves_through_plan_profile_runtime() -> None:
    profile = SimpleNamespace(
        profile_id="gemma4:26b-llamacpp", is_default=True, runtime_presets=[object()]
    )
    registry = _profile_registry_with_profiles([profile])
    fake_plan = SimpleNamespace(
        worker_key="llama_cpp/gemma4-26b::02e08fbd7161",
        load_overrides={"llama_cpp": {"ctx_size": 65536}},
        preferred_gpu=1,
        load_policy="on_demand",
    )

    with patch(
        "ocabra.core.runtime_coordinator.plan_profile_runtime",
        new=AsyncMock(return_value=fake_plan),
    ):
        worker_key, load_overrides, preferred_gpu, load_policy = await resolve_realtime_llm_runtime_plan(
            "llama_cpp/gemma4-26b", profile_registry=registry, model_manager=MagicMock()
        )

    assert worker_key == "llama_cpp/gemma4-26b::02e08fbd7161"
    assert load_overrides == {"llama_cpp": {"ctx_size": 65536}}
    assert preferred_gpu == 1
    assert load_policy == "on_demand"


@pytest.mark.asyncio
async def test_runtime_plan_failure_falls_back_to_base_model_id_instead_of_raising() -> None:
    registry = MagicMock()
    registry.list_by_model = AsyncMock(side_effect=RuntimeError("boom"))

    worker_key, load_overrides, preferred_gpu, load_policy = await resolve_realtime_llm_runtime_plan(
        "llama_cpp/gemma4-26b", profile_registry=registry, model_manager=MagicMock()
    )

    assert worker_key == "llama_cpp/gemma4-26b"
    assert (load_overrides, preferred_gpu, load_policy) == (None, None, None)


@pytest.mark.asyncio
async def test_no_matching_profile_falls_back_to_base_model_id() -> None:
    registry = _profile_registry_with_profiles([])

    worker_key, *_rest = await resolve_realtime_llm_runtime_plan(
        "llama_cpp/orphan-model", profile_registry=registry, model_manager=MagicMock()
    )

    assert worker_key == "llama_cpp/orphan-model"
