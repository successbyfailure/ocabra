"""Model resolution for the OpenAI Realtime WebSocket (profiles, routers, legacy ids)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ocabra.api.openai.realtime import resolve_realtime_model_id


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
