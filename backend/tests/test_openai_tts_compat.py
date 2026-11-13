"""Unit tests for OpenAITTSCompatRegistry — the OpenAI TTS voice-name
compatibility shim. Deliberately independent of routers/profiles: these
tests only touch ``ModelProfile.profile_id``/``enabled`` for the
existence/enabled check in ``upsert()``, nothing router-shaped.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ocabra.api.openai.audio import _resolve_openai_tts_compat
from ocabra.core.openai_tts_compat import OpenAITTSCompatRegistry
from ocabra.db.model_config import OpenAITTSVoiceMap


class _FakeFirstResult:
    """Stands in for the ``.first()`` shape of the existence/enabled check."""

    def __init__(self, row):
        self._row = row

    def first(self):
        return self._row


class _FakeScalarResult:
    def __init__(self, item=None, items=None):
        self._item = item
        self._items = items or []

    def scalar_one_or_none(self):
        return self._item

    def scalars(self):
        return SimpleNamespace(all=lambda: self._items)


def _entry(openai_model="tts-1", voice_name="alloy", target_profile_id="piper-es-mx-claude-high"):
    return OpenAITTSVoiceMap(
        openai_model=openai_model, voice_name=voice_name, target_profile_id=target_profile_id
    )


class TestLoadAndResolve:
    @pytest.mark.asyncio
    async def test_load_all_populates_cache_by_model_and_voice(self):
        registry = OpenAITTSCompatRegistry()
        rows = [_entry(), _entry("tts-1", "echo", "glados-es-fine-tuned")]
        session = AsyncMock()
        session.execute = AsyncMock(return_value=_FakeScalarResult(items=rows))

        await registry.load_all(session)

        assert registry.resolve("tts-1", "alloy") == "piper-es-mx-claude-high"
        assert registry.resolve("tts-1", "echo") == "glados-es-fine-tuned"

    def test_resolve_returns_none_when_unmapped(self):
        registry = OpenAITTSCompatRegistry()
        registry._entries[("tts-1", "alloy")] = _entry()
        assert registry.resolve("tts-1", "shimmer") is None
        assert registry.resolve("tts-1-hd", "alloy") is None

    def test_resolve_returns_none_without_a_voice(self):
        """No ``voice`` in the request body at all (not even OpenAI's own
        'alloy' default) — nothing to look up, let the caller fall back to
        resolving the literal model id."""
        registry = OpenAITTSCompatRegistry()
        registry._entries[("tts-1", "alloy")] = _entry()
        assert registry.resolve("tts-1", None) is None
        assert registry.resolve("tts-1", "") is None

    def test_list_all_is_sorted(self):
        registry = OpenAITTSCompatRegistry()
        registry._entries[("tts-1-hd", "echo")] = _entry("tts-1-hd", "echo", "t2")
        registry._entries[("tts-1", "alloy")] = _entry("tts-1", "alloy", "t1")
        listed = registry.list_all()
        assert [(e.openai_model, e.voice_name) for e in listed] == [
            ("tts-1", "alloy"),
            ("tts-1-hd", "echo"),
        ]


class TestUpsert:
    @pytest.mark.asyncio
    async def test_rejects_missing_target(self):
        registry = OpenAITTSCompatRegistry()
        session = AsyncMock()
        session.execute = AsyncMock(return_value=_FakeFirstResult(None))

        with pytest.raises(ValueError, match="not found"):
            await registry.upsert(
                session, openai_model="tts-1", voice_name="alloy", target_profile_id="missing"
            )

    @pytest.mark.asyncio
    async def test_rejects_disabled_target(self):
        registry = OpenAITTSCompatRegistry()
        session = AsyncMock()
        session.execute = AsyncMock(
            return_value=_FakeFirstResult(SimpleNamespace(profile_id="t1", enabled=False))
        )

        with pytest.raises(ValueError, match="disabled"):
            await registry.upsert(
                session, openai_model="tts-1", voice_name="alloy", target_profile_id="t1"
            )

    @pytest.mark.asyncio
    async def test_rejects_missing_model_or_voice(self):
        registry = OpenAITTSCompatRegistry()
        session = AsyncMock()
        with pytest.raises(ValueError, match="required"):
            await registry.upsert(
                session, openai_model="", voice_name="alloy", target_profile_id="t1"
            )

    @pytest.mark.asyncio
    async def test_creates_a_new_entry_and_caches_it(self):
        registry = OpenAITTSCompatRegistry()
        session = AsyncMock()
        session.execute = AsyncMock(
            return_value=_FakeFirstResult(SimpleNamespace(profile_id="t1", enabled=True))
        )

        entry = await registry.upsert(
            session, openai_model="tts-1", voice_name="alloy", target_profile_id="t1"
        )

        assert entry.target_profile_id == "t1"
        assert registry.resolve("tts-1", "alloy") == "t1"
        session.add.assert_called_once()
        session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_overwrites_an_existing_entry_in_place(self):
        registry = OpenAITTSCompatRegistry()
        registry._entries[("tts-1", "alloy")] = _entry("tts-1", "alloy", "old-target")
        session = AsyncMock()
        session.execute = AsyncMock(
            return_value=_FakeFirstResult(SimpleNamespace(profile_id="new-target", enabled=True))
        )

        entry = await registry.upsert(
            session, openai_model="tts-1", voice_name="alloy", target_profile_id="new-target"
        )

        assert entry.target_profile_id == "new-target"
        assert registry.resolve("tts-1", "alloy") == "new-target"
        # Overwrite reuses the cached row — no new ``session.add``.
        session.add.assert_not_called()


class TestDelete:
    @pytest.mark.asyncio
    async def test_delete_removes_a_known_entry(self):
        registry = OpenAITTSCompatRegistry()
        registry._entries[("tts-1", "alloy")] = _entry()
        session = AsyncMock()
        session.execute = AsyncMock(return_value=_FakeScalarResult(_entry()))

        deleted = await registry.delete(session, openai_model="tts-1", voice_name="alloy")

        assert deleted is True
        assert registry.resolve("tts-1", "alloy") is None
        session.delete.assert_awaited_once()
        session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_delete_returns_false_for_unknown_entry(self):
        registry = OpenAITTSCompatRegistry()
        session = AsyncMock()

        deleted = await registry.delete(session, openai_model="tts-1", voice_name="alloy")

        assert deleted is False
        session.execute.assert_not_called()


# ── Glue: ``audio.py``'s ``_resolve_openai_tts_compat`` ──────────────────
#
# Translates an OpenAI official model name to an oCabra profile_id before
# the normal ``resolve_profile()`` call in speech()/list_voices(). A
# client that already sends an oCabra name must see zero behaviour
# change — that's the "separate from routers, opt-in only" requirement.


def _fake_request(compat: OpenAITTSCompatRegistry | None):
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(openai_tts_compat=compat)))


class TestAudioCompatGlue:
    def test_translates_a_pinned_openai_model_and_voice(self):
        registry = OpenAITTSCompatRegistry()
        registry._entries[("tts-1", "alloy")] = _entry("tts-1", "alloy", "piper-es-mx-claude-high")
        request = _fake_request(registry)

        assert _resolve_openai_tts_compat(request, "tts-1", "alloy") == "piper-es-mx-claude-high"

    def test_passes_through_an_ocabra_model_id_unchanged(self):
        """A client that already sends an oCabra name is unaffected, pin
        or no pin — the compat table is only ever keyed by OpenAI's own
        model names."""
        registry = OpenAITTSCompatRegistry()
        registry._entries[("tts-1", "alloy")] = _entry()
        request = _fake_request(registry)

        assert (
            _resolve_openai_tts_compat(request, "piper-es-mx-claude-high", "alloy")
            == "piper-es-mx-claude-high"
        )

    def test_passes_through_when_nothing_is_pinned_for_that_pair(self):
        registry = OpenAITTSCompatRegistry()
        request = _fake_request(registry)

        assert _resolve_openai_tts_compat(request, "tts-1", "alloy") == "tts-1"

    def test_passes_through_when_the_registry_is_unavailable(self):
        request = _fake_request(None)

        assert _resolve_openai_tts_compat(request, "tts-1", "alloy") == "tts-1"
