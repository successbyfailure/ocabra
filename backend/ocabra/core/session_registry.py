"""Session registry — tracks live Realtime sessions holding worker models.

Bloque 20 — Etapa 4.

A Realtime session (``api/openai/realtime.py``) is a WebSocket that spans
minutes to tens of minutes, driving 2-3 backend workers (Whisper STT + LLM +
TTS) turn-by-turn. It's fundamentally different from a request/response job:

    * You can't estimate its total duration up front. Nothing tells you how
      long the user will keep talking.
    * You must NOT evict its workers while it's live. Interrupting a
      Realtime session mid-turn drops the WebSocket in a way the client
      can't recover from without a reconnect.
    * A quiet gap between turns is legitimate — the model is idle, but it
      would be wrong to steal it. The whole session is the reservation.

This registry answers two questions:

    has_active_session_holding(worker_key) -> bool
        Consumer: router + pressure_eviction. If True → don't touch.

    sessions_holding(worker_key) -> list[SessionInfo]
        Consumer: admin UI + Stats. Show which sessions are pinning what.

Redis is used as an out-of-process cache with TTL = ``session_heartbeat_ttl_s``
so a fresh api process rehydrates active sessions after a restart mid-flight.
An entry that doesn't get a heartbeat for ``session_max_idle_s`` is
auto-unregistered — protection against zombie WebSockets.

Persistence choice: **only** Redis + in-memory, no Postgres row. Sessions are
ephemeral and the sub-operations (``realtime_transcription`` etc.) already
land in ``request_stats`` for the historical trail.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import structlog

from ocabra.config import settings

logger = structlog.get_logger(__name__)


@dataclass
class SessionInfo:
    """A live Realtime session.

    ``workers_held`` are the worker_keys (canonical model_id or ``base::hash``
    form for profiles with load_overrides) that this session has reserved.
    They may be a mix of backends: e.g. Whisper STT + Gemma4 LLM + Chatterbox
    TTS all pinned by one session.
    """

    session_id: str
    user_id: uuid.UUID | None
    api_key_name: str | None
    workers_held: list[str]
    started_at: datetime
    last_activity_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def is_paused(self, threshold_s: int) -> bool:
        """Silence longer than ``threshold_s`` — worker may be shared with
        concurrent traffic, but never evicted."""
        return (datetime.now(UTC) - self.last_activity_at).total_seconds() > threshold_s

    def is_zombie(self, max_idle_s: int) -> bool:
        """No heartbeat for so long the client is presumed disconnected."""
        return (datetime.now(UTC) - self.last_activity_at).total_seconds() > max_idle_s

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "user_id": str(self.user_id) if self.user_id else None,
            "api_key_name": self.api_key_name,
            "workers_held": list(self.workers_held),
            "started_at": self.started_at.isoformat(),
            "last_activity_at": self.last_activity_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "SessionInfo":
        return cls(
            session_id=payload["session_id"],
            user_id=uuid.UUID(payload["user_id"]) if payload.get("user_id") else None,
            api_key_name=payload.get("api_key_name"),
            workers_held=list(payload.get("workers_held") or []),
            started_at=datetime.fromisoformat(payload["started_at"]),
            last_activity_at=datetime.fromisoformat(payload["last_activity_at"]),
        )


_REDIS_KEY_PREFIX = "ocabra:realtime_session:"


class SessionRegistry:
    """Live registry of Realtime sessions and the workers they hold."""

    def __init__(self) -> None:
        self._by_id: dict[str, SessionInfo] = {}
        # Reverse index for O(1) lookup by worker_key. A worker can be held
        # by multiple sessions concurrently (e.g. two admins running Realtime
        # against the same LLM); the set grows/shrinks as sessions attach.
        self._by_worker: dict[str, set[str]] = {}
        self._lock = asyncio.Lock()
        self._sweep_task: asyncio.Task | None = None

    # ── Read side ────────────────────────────────────────────

    def has_active_session_holding(self, worker_key: str) -> bool:
        """Router / pressure_eviction consumer. Fast path with no lock —
        the reverse index is fine to read racily; the worst case is a stale
        True/False for a millisecond, which never matters."""
        holders = self._by_worker.get(worker_key)
        if not holders:
            return False
        max_idle = int(settings.session_max_idle_s)
        for sid in holders:
            session = self._by_id.get(sid)
            if session is None:
                continue
            if not session.is_zombie(max_idle):
                return True
        return False

    def sessions_holding(self, worker_key: str) -> list[SessionInfo]:
        return [
            self._by_id[sid]
            for sid in self._by_worker.get(worker_key, set())
            if sid in self._by_id
        ]

    def list_sessions(self) -> list[SessionInfo]:
        return list(self._by_id.values())

    def get(self, session_id: str) -> SessionInfo | None:
        return self._by_id.get(session_id)

    # ── Write side ───────────────────────────────────────────

    async def register(
        self,
        session_id: str,
        workers_held: list[str],
        *,
        user_id: uuid.UUID | None = None,
        api_key_name: str | None = None,
    ) -> SessionInfo:
        """Called by ``RealtimeSession`` at WebSocket accept time. Persists
        to Redis so a restart can rehydrate."""
        async with self._lock:
            existing = self._by_id.get(session_id)
            info = SessionInfo(
                session_id=session_id,
                user_id=user_id,
                api_key_name=api_key_name,
                workers_held=list(workers_held),
                started_at=existing.started_at if existing else datetime.now(UTC),
            )
            self._by_id[session_id] = info
            for worker_key in workers_held:
                self._by_worker.setdefault(worker_key, set()).add(session_id)
        await self._persist(info)
        logger.info(
            "session_registered",
            session_id=session_id,
            workers=workers_held,
            api_key_name=api_key_name,
        )
        return info

    async def heartbeat(self, session_id: str) -> None:
        """Called by ``RealtimeSession`` on every chunk (STT input, LLM
        turn, TTS output). Refreshes both the in-memory last_activity_at
        and the Redis TTL."""
        info = self._by_id.get(session_id)
        if info is None:
            return
        info.last_activity_at = datetime.now(UTC)
        await self._persist(info)

    async def unregister(self, session_id: str, *, reason: str = "closed") -> None:
        """Called on WebSocket close or auto-sweep. Idempotent."""
        async with self._lock:
            info = self._by_id.pop(session_id, None)
            if info is not None:
                for worker_key in info.workers_held:
                    holders = self._by_worker.get(worker_key)
                    if holders:
                        holders.discard(session_id)
                        if not holders:
                            self._by_worker.pop(worker_key, None)
        await self._forget(session_id)
        if info is not None:
            logger.info(
                "session_unregistered",
                session_id=session_id,
                reason=reason,
                duration_s=(datetime.now(UTC) - info.started_at).total_seconds(),
            )

    # ── Background sweeper ───────────────────────────────────

    async def start(self) -> None:
        """Kick off the background zombie sweeper and rehydrate any Redis
        entries the previous process left behind. Safe to call twice."""
        if self._sweep_task is not None:
            return
        await self._rehydrate_from_redis()
        self._sweep_task = asyncio.create_task(self._sweep_loop(), name="session-registry-sweep")

    async def stop(self) -> None:
        if self._sweep_task is None:
            return
        self._sweep_task.cancel()
        try:
            await self._sweep_task
        except (asyncio.CancelledError, Exception):
            pass
        self._sweep_task = None

    async def _sweep_loop(self) -> None:
        # Sweep at least once every 30 s (short) or every quarter of the
        # zombie threshold (whichever is smaller) so a zombie is caught
        # promptly.
        interval = max(15, min(30, int(settings.session_max_idle_s) // 4))
        while True:
            try:
                await asyncio.sleep(interval)
                await self._sweep_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — never break the loop
                logger.warning("session_registry_sweep_failed", error=str(exc))

    async def _sweep_once(self) -> None:
        max_idle = int(settings.session_max_idle_s)
        stale = [
            sid
            for sid, info in list(self._by_id.items())
            if info.is_zombie(max_idle)
        ]
        for sid in stale:
            await self.unregister(sid, reason="zombie_auto_kill")

    # ── Redis persistence ────────────────────────────────────

    async def _persist(self, info: SessionInfo) -> None:
        try:
            from ocabra.redis_client import set_key

            await set_key(
                f"{_REDIS_KEY_PREFIX}{info.session_id}",
                info.to_dict(),
                ttl=int(settings.session_heartbeat_ttl_s),
            )
        except Exception as exc:  # noqa: BLE001 — memory registry still works
            logger.debug("session_registry_persist_failed", error=str(exc))

    async def _forget(self, session_id: str) -> None:
        try:
            from ocabra.redis_client import get_redis

            client = await get_redis()
            if client is not None:
                await client.delete(f"{_REDIS_KEY_PREFIX}{session_id}")
        except Exception:
            pass

    async def _rehydrate_from_redis(self) -> None:
        """On startup, restore sessions whose heartbeats are still fresh.
        Any older entries are discarded — the client already disconnected."""
        try:
            from ocabra.redis_client import get_redis

            client = await get_redis()
            if client is None:
                return
            keys = []
            async for k in client.scan_iter(f"{_REDIS_KEY_PREFIX}*"):
                keys.append(k)
            if not keys:
                return
            payloads = await client.mget(*keys)
        except Exception as exc:
            logger.warning("session_registry_rehydrate_failed", error=str(exc))
            return
        restored = 0
        cutoff = datetime.now(UTC) - timedelta(seconds=int(settings.session_max_idle_s))
        for raw in payloads:
            if raw is None:
                continue
            try:
                info = SessionInfo.from_dict(
                    json.loads(raw) if isinstance(raw, (str, bytes, bytearray)) else raw
                )
            except (ValueError, KeyError) as exc:
                logger.debug("session_registry_bad_redis_row", error=str(exc))
                continue
            if info.last_activity_at < cutoff:
                continue
            self._by_id[info.session_id] = info
            for worker_key in info.workers_held:
                self._by_worker.setdefault(worker_key, set()).add(info.session_id)
            restored += 1
        if restored:
            logger.info("session_registry_rehydrated", sessions=restored)
