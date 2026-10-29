"""Realtime sessions must accept a ROUTER profile (not just a direct model_id)
as the configured ``realtime_default_stt_model``/``realtime_default_tts_model``.

Mirrors the LLM target's resolution (``resolve_realtime_model_id()`` in
``ocabra.api.openai.realtime``): a plain profile resolves to its base model, a
router profile is resolved through the ``RouterResolver``, and anything else
(a literal model/backend id, or an unknown string) passes through unchanged.
Resolved once per connect in ``RealtimeSession._resolve_stt_tts_defaults()``,
not per-turn — same scope as the LLM path.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ocabra.core.realtime_session import RealtimeSession


def _profile(pid: str, base: str | None, enabled: bool = True) -> SimpleNamespace:
    return SimpleNamespace(profile_id=pid, base_model_id=base, enabled=enabled)


def _registry(profiles: dict) -> MagicMock:
    registry = MagicMock()
    registry.get = AsyncMock(side_effect=lambda pid: profiles.get(pid))
    return registry


def _make_session(
    *,
    monkeypatch: pytest.MonkeyPatch,
    default_stt: str = "",
    default_tts: str = "",
    profile_registry: MagicMock | None = None,
    router_resolver: MagicMock | None = None,
) -> RealtimeSession:
    monkeypatch.setattr("ocabra.config.settings.realtime_default_stt_model", default_stt)
    monkeypatch.setattr("ocabra.config.settings.realtime_default_tts_model", default_tts)

    ws = MagicMock()
    ws.send_text = AsyncMock()

    model_manager = MagicMock()
    model_manager.list_states = AsyncMock(return_value=[])

    worker_pool = MagicMock()
    worker_pool.get_worker = MagicMock(return_value=None)

    return RealtimeSession(
        ws=ws,
        model_id="vllm/some-llm",
        worker_pool=worker_pool,
        model_manager=model_manager,
        profile_registry=profile_registry,
        router_resolver=router_resolver,
    )


@pytest.mark.asyncio
async def test_router_profile_default_resolves_through_the_router_resolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router_profile = _profile("recomendado-stt", None)
    target_profile = _profile("whisper-large-v3", "whisper/openai/whisper-large-v3")
    registry = _registry({"recomendado-stt": router_profile, "whisper-large-v3": target_profile})
    resolver = MagicMock()
    resolver.is_router.return_value = True
    resolver.pick = AsyncMock(return_value=SimpleNamespace(target_profile_id="whisper-large-v3"))

    session = _make_session(
        monkeypatch=monkeypatch,
        default_stt="recomendado-stt",
        profile_registry=registry,
        router_resolver=resolver,
    )
    await session._resolve_stt_tts_defaults()  # noqa: SLF001

    assert session.stt_model_id == "whisper/openai/whisper-large-v3"
    resolver.pick.assert_awaited_once()


@pytest.mark.asyncio
async def test_non_router_profile_default_resolves_to_its_base_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _registry({"kokoro-profile": _profile("kokoro-profile", "tts/kokoro")})
    resolver = MagicMock()
    resolver.is_router.return_value = False
    resolver.pick = AsyncMock()

    session = _make_session(
        monkeypatch=monkeypatch,
        default_tts="kokoro-profile",
        profile_registry=registry,
        router_resolver=resolver,
    )
    await session._resolve_stt_tts_defaults()  # noqa: SLF001

    assert session.tts_model_id == "tts/kokoro"
    resolver.pick.assert_not_awaited()


@pytest.mark.asyncio
async def test_unconfigured_defaults_stay_none(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _registry({})

    session = _make_session(monkeypatch=monkeypatch, profile_registry=registry)
    await session._resolve_stt_tts_defaults()  # noqa: SLF001

    assert session.stt_model_id is None
    assert session.tts_model_id is None
    registry.get.assert_not_awaited()


@pytest.mark.asyncio
async def test_unknown_string_default_falls_back_to_the_literal_model_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _registry({})

    session = _make_session(
        monkeypatch=monkeypatch,
        default_stt="whisper/custom-worker-id",
        profile_registry=registry,
    )
    await session._resolve_stt_tts_defaults()  # noqa: SLF001

    assert session.stt_model_id == "whisper/custom-worker-id"


@pytest.mark.asyncio
async def test_no_injected_profile_registry_skips_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct construction without a profile_registry (e.g. existing tests that
    don't wire one up) must keep today's literal-passthrough behaviour instead
    of erroring out on a missing dependency."""
    session = _make_session(monkeypatch=monkeypatch, default_stt="some-literal-id")
    assert session.stt_model_id == "some-literal-id"

    await session._resolve_stt_tts_defaults()  # noqa: SLF001

    assert session.stt_model_id == "some-literal-id"
