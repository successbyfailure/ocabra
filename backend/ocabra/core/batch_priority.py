"""Batch-vs-interactive priority coordinator (2026-10-30 design).

``FairSlotGate`` (see ``fair_slot_gate.py``) already makes sure batch-origin
callers never win the *next* free slot while an interactive caller is
waiting — but it never preempts something already running. This module
covers the harder case: every slot on a worker is already held by batch
requests, so nothing will free up on its own any time soon. It tracks which
in-flight requests are batch-origin (so they can be cancelled) and decides,
per ``settings.batch_preempt_mode``, whether a specific one should be.

Measured motivation (2026-10-30, against ``llama_cpp/qwen3.8-27b`` with its
3-slot preset): while at least one slot is free, a concurrent long-running
request has no measurable effect on a new request's time-to-first-token
(~0.2s regardless of 0, 1 or 2 slots already busy). The moment all slots
are saturated, a new request waits almost the full remaining duration of
whichever occupant finishes first (~18.5s against a ~21s task) — that's
the wait this module exists to cut short.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import structlog

from ocabra.config import settings

logger = structlog.get_logger(__name__)


@dataclass
class _BatchTask:
    task: asyncio.Task[object]
    batch_id: str
    line_idx: int
    started_at: float = field(default_factory=time.monotonic)


class BatchPriorityCoordinator:
    """Per-worker registry of in-flight batch tasks + preemption/cooldown state."""

    def __init__(self) -> None:
        self._by_worker: dict[str, dict[int, _BatchTask]] = {}
        self._next_handle = 0
        self._cooldown_until: dict[str, float] = {}
        self._cooldown_kind: dict[str, str] = {}
        self._lock = asyncio.Lock()
        # Visibility (2026-10-30) — in-memory only, resets on restart like
        # everything else here; just enough to show "is this mechanism doing
        # anything" in the UI without building a persisted events table for
        # numbers we're still calibrating against real traffic.
        self._total_preemptions = 0
        self._recent_events: list[dict[str, object]] = []
        self._max_recent_events = 50

    def _record_event(self, **fields: object) -> None:
        self._recent_events.append({"at": time.monotonic(), **fields})
        if len(self._recent_events) > self._max_recent_events:
            self._recent_events = self._recent_events[-self._max_recent_events:]

    def status(self) -> dict[str, object]:
        """Snapshot for the admin UI: current cooldowns, recent preemption
        events, running totals. All timestamps are seconds-from-now (not
        wall clock) since the internal clock is ``time.monotonic()``."""
        now = time.monotonic()
        cooldowns = [
            {
                "workerKey": key,
                "kind": self._cooldown_kind.get(key, "slot"),
                "remainingSeconds": round(max(0.0, until - now), 1),
            }
            for key, until in self._cooldown_until.items()
            if until > now
        ]
        return {
            "totalPreemptions": self._total_preemptions,
            "activeCooldowns": cooldowns,
            "recentEvents": [
                {**event, "secondsAgo": round(now - event["at"], 1)}
                for event in reversed(self._recent_events)
            ],
        }

    @staticmethod
    def _canonical(worker_key: str) -> str:
        """Strip the ``::<hash>`` runtime-preset suffix oCabra appends to a
        loaded worker's real ``model_id`` (e.g.
        ``llama_cpp/qwen3.8-27b::781a0367a8ba``).

        The batch processor only knows the bare model string the user wrote
        in their JSONL body — it doesn't resolve profiles/presets, so it
        can't predict which suffixed variant will actually serve the
        request. Canonicalizing both sides to the bare prefix here is what
        makes registration (batch_processor.py) and lookup (worker_pool.py,
        model_manager.py — both pass the real, suffixed worker_key) agree on
        the same identity. Confirmed live 2026-10-30: without this,
        preempt_now() always found an empty bucket and silently did nothing.
        """
        return worker_key.split("::", 1)[0]

    # ── Registration (called by the batch processor around each dispatch) ──

    async def register(
        self, worker_key: str, task: asyncio.Task[object], *, batch_id: str, line_idx: int
    ) -> int:
        key = self._canonical(worker_key)
        async with self._lock:
            handle = self._next_handle
            self._next_handle += 1
            self._by_worker.setdefault(key, {})[handle] = _BatchTask(
                task=task, batch_id=batch_id, line_idx=line_idx
            )
            return handle

    async def unregister(self, worker_key: str, handle: int) -> None:
        key = self._canonical(worker_key)
        async with self._lock:
            bucket = self._by_worker.get(key)
            if bucket is not None:
                bucket.pop(handle, None)
                if not bucket:
                    self._by_worker.pop(key, None)

    # ── Cooldown (set after a preemption fires; consulted before the batch
    #    processor dispatches a new line to the same worker) ──

    def mark_cooldown(self, worker_key: str, *, kind: str) -> None:
        key = self._canonical(worker_key)
        seconds = (
            settings.batch_resume_cooldown_eviction_s
            if kind == "eviction"
            else settings.batch_resume_cooldown_slots_s
        )
        until = time.monotonic() + max(0.0, float(seconds))
        self._cooldown_until[key] = max(until, self._cooldown_until.get(key, 0.0))
        self._cooldown_kind[key] = kind
        logger.info(
            "batch_priority_cooldown_started",
            worker_key=key,
            kind=kind,
            seconds=seconds,
        )

    def in_cooldown(self, worker_key: str) -> bool:
        key = self._canonical(worker_key)
        until = self._cooldown_until.get(key)
        if until is None:
            return False
        if time.monotonic() >= until:
            self._cooldown_until.pop(key, None)
            return False
        return True

    # ── Preemption ──

    def has_batch_activity(self, worker_key: str) -> bool:
        return bool(self._by_worker.get(self._canonical(worker_key)))

    async def preempt_now(self, worker_key: str, *, estimator=None) -> int:
        """Cancel batch-origin tasks on ``worker_key`` per ``batch_preempt_mode``.

        Returns the number of tasks actually cancelled. Cancellation
        propagates down through the in-process ASGI call into the real
        backend HTTP client (same disconnect-abort contract documented in
        ``worker_pool.py``) — the GPU work actually stops, not just the
        batch processor's bookkeeping.
        """
        key = self._canonical(worker_key)
        logger.info(
            "batch_priority_preempt_now_called",
            worker_key=key,
            enabled=settings.batch_priority_enabled,
            registry_keys=list(self._by_worker.keys()),
        )
        if not settings.batch_priority_enabled:
            return 0
        mode = settings.batch_preempt_mode
        if mode == "never":
            return 0

        async with self._lock:
            bucket = list(self._by_worker.get(key, {}).items())

        if not bucket:
            return 0

        now = time.monotonic()
        target_n = max(1, round(len(bucket) * settings.batch_free_slots_pct))
        cancelled = 0
        # ``worker_key`` here is the model_id (WorkerPoolFairGates/FairSlotGate
        # key by model_id, same space model_manager.mark_request_start uses) —
        # in_flight_remaining is a per-model aggregate (oldest in-flight
        # request on this model), not per-line, so it's computed once and
        # reused for every candidate on this worker rather than per entry.
        remaining = None
        if estimator is not None and mode == "conditional":
            try:
                remaining = estimator.in_flight_remaining(worker_key, model_id=worker_key)
            except Exception:  # noqa: BLE001
                remaining = None
        for handle, entry in bucket:
            if cancelled >= target_n:
                break
            if entry.task.done():
                continue
            elapsed = now - entry.started_at
            if mode == "conditional":
                already_long = elapsed > settings.batch_preempt_min_elapsed_s
                almost_done = remaining is not None and remaining < settings.batch_preempt_grace_s
                if already_long and almost_done:
                    continue  # let it finish — cheaper than killing it this close to done
            entry.task.cancel()
            cancelled += 1
            self._total_preemptions += 1
            self._record_event(
                type="preempted",
                workerKey=worker_key,
                batchId=entry.batch_id,
                lineIdx=entry.line_idx,
                elapsedSeconds=round(elapsed, 1),
                mode=mode,
            )
            logger.info(
                "batch_priority_preempted",
                worker_key=worker_key,
                batch_id=entry.batch_id,
                line_idx=entry.line_idx,
                elapsed_s=round(elapsed, 1),
                mode=mode,
            )

        if cancelled:
            self.mark_cooldown(worker_key, kind="slot")
        return cancelled
