"""Unit tests for the SessionRegistry.

Bloque 20, Etapa 4.

Covers the two consumer contracts:

    has_active_session_holding(worker_key)
    sessions_holding(worker_key)

plus the state transitions the RealtimeSession will drive:
register → heartbeat → unregister, and the zombie-sweep safety net.

Redis persistence is exercised end-to-end by ``test_routing_e2e.py``;
here we skip it (the ``_persist`` / ``_forget`` calls swallow errors when
Redis is unavailable) and focus on the in-memory + concurrency contract.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta

import pytest

from ocabra.core.session_registry import SessionInfo, SessionRegistry


class TestRegisterHoldingLookup:
    @pytest.mark.asyncio
    async def test_register_marks_workers_as_held(self):
        registry = SessionRegistry()
        await registry.register("sess-a", workers_held=["ollama/gemma4:26b", "whisper/x"])
        assert registry.has_active_session_holding("ollama/gemma4:26b") is True
        assert registry.has_active_session_holding("whisper/x") is True
        assert registry.has_active_session_holding("not-held") is False

    @pytest.mark.asyncio
    async def test_multiple_sessions_can_hold_same_worker(self):
        """Two Realtime sessions on the same LLM must both be tracked. Losing
        one shouldn't release the worker while the other is still live."""
        registry = SessionRegistry()
        await registry.register("sess-a", workers_held=["ollama/gemma4:26b"])
        await registry.register("sess-b", workers_held=["ollama/gemma4:26b"])
        assert len(registry.sessions_holding("ollama/gemma4:26b")) == 2
        await registry.unregister("sess-a")
        assert registry.has_active_session_holding("ollama/gemma4:26b") is True
        await registry.unregister("sess-b")
        assert registry.has_active_session_holding("ollama/gemma4:26b") is False

    @pytest.mark.asyncio
    async def test_unregister_is_idempotent(self):
        registry = SessionRegistry()
        await registry.register("sess-a", workers_held=["w"])
        await registry.unregister("sess-a")
        await registry.unregister("sess-a")  # should not raise or blow up state
        assert registry.get("sess-a") is None


class TestHeartbeat:
    @pytest.mark.asyncio
    async def test_heartbeat_updates_last_activity(self):
        registry = SessionRegistry()
        await registry.register("sess-a", workers_held=["w"])
        info = registry.get("sess-a")
        assert info is not None
        first = info.last_activity_at
        await asyncio.sleep(0.01)
        await registry.heartbeat("sess-a")
        assert info.last_activity_at > first

    @pytest.mark.asyncio
    async def test_heartbeat_on_unknown_session_is_noop(self):
        """The RealtimeSession may fire heartbeats after a manual kill —
        the registry must silently ignore instead of raising."""
        registry = SessionRegistry()
        await registry.heartbeat("phantom-session")  # no exception


class TestZombieDetection:
    def test_is_zombie_flag(self):
        info = SessionInfo(
            session_id="s",
            user_id=None,
            api_key_name=None,
            workers_held=["w"],
            started_at=datetime.now(UTC) - timedelta(hours=1),
            last_activity_at=datetime.now(UTC) - timedelta(seconds=2000),
        )
        assert info.is_zombie(max_idle_s=900) is True
        assert info.is_zombie(max_idle_s=3600) is False

    @pytest.mark.asyncio
    async def test_zombie_ignored_by_has_active(self, monkeypatch):
        """A zombie must NOT be reported as an active hold — evict/router
        should behave as if the session had unregistered itself. The
        auto-sweeper will actually remove it on the next tick."""
        from ocabra.config import settings

        monkeypatch.setattr(settings, "session_max_idle_s", 1)
        registry = SessionRegistry()
        await registry.register("sess-a", workers_held=["w"])
        await asyncio.sleep(1.1)
        assert registry.has_active_session_holding("w") is False

    @pytest.mark.asyncio
    async def test_sweep_removes_zombies(self, monkeypatch):
        """The background sweeper unregisters zombies so the reverse index
        stops carrying them, matching what ``has_active_session_holding``
        already advertises."""
        from ocabra.config import settings

        monkeypatch.setattr(settings, "session_max_idle_s", 1)
        registry = SessionRegistry()
        await registry.register("sess-a", workers_held=["w"])
        await asyncio.sleep(1.1)
        await registry._sweep_once()
        assert registry.get("sess-a") is None
        assert registry.sessions_holding("w") == []


class TestPauseState:
    def test_is_paused_reports_recent_silence(self):
        info = SessionInfo(
            session_id="s",
            user_id=None,
            api_key_name=None,
            workers_held=["w"],
            started_at=datetime.now(UTC),
            last_activity_at=datetime.now(UTC) - timedelta(seconds=200),
        )
        assert info.is_paused(threshold_s=120) is True
        assert info.is_paused(threshold_s=600) is False


class TestSerialization:
    def test_roundtrip_via_dict(self):
        original = SessionInfo(
            session_id="s",
            user_id=uuid.uuid4(),
            api_key_name="k",
            workers_held=["w1", "w2"],
            started_at=datetime.now(UTC),
        )
        rebuilt = SessionInfo.from_dict(original.to_dict())
        assert rebuilt.session_id == original.session_id
        assert rebuilt.user_id == original.user_id
        assert rebuilt.workers_held == original.workers_held
        # Datetimes may lose sub-microsecond precision through isoformat →
        # from_dict; a second's accuracy is enough for our purposes.
        assert abs((rebuilt.started_at - original.started_at).total_seconds()) < 1
