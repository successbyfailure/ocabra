"""Per-worker fair-share admission gate.

llama.cpp/vLLM/etc. workers run up to N requests concurrently (their
``--parallel``/``max_num_seqs``-style slot count) and queue the rest
internally — but that internal queue is plain FIFO by arrival order, with
no concept of "caller". One user firing a long backlog of requests (an
agent loop, a retry storm, a batch job) can occupy every slot indefinitely;
a second user's request then queues behind the *entire* backlog, not just
whatever was already running.

:class:`FairSlotGate` sits in front of that: oCabra holds pending requests
itself and hands out the ``slots`` concurrency budget to whichever
``user_key`` with pending work has gone longest without being served —
strict alternation, not FIFO-by-arrival. A user who already burned through
several free, uncontested slots never keeps precedence over a user who
hasn't been served at all yet, even if the first user's next request was
technically queued earlier (2026-10-28 design discussion: "las peticiones
de multiples usuarios se vayan alternando en los slots disponibles,
independientemente de que uno haya mandado una rafaga"). A new user only
ever waits for the already-*running* work currently holding the slots to
finish (unavoidable — nothing can preempt a request mid-flight) — never for
an active user's entire remaining queue.

One gate instance per worker (keyed by worker_key, which already changes
whenever a model's runtime preset changes, so a preset escalation
naturally gets a fresh gate sized for the new slot count — see
WorkerPoolFairGates).
"""

from __future__ import annotations

import asyncio
from collections import deque
from contextlib import asynccontextmanager
from typing import AsyncIterator


class FairSlotGate:
    """Admits up to ``slots`` concurrent callers; waiters are released to
    whichever ``user_key`` with pending work has gone longest without being
    served — least-recently-served-first, not arrival-order FIFO.

    Every acquire — even one that could be granted immediately — goes
    through the same queue-and-dispatch path. That uniformity is what
    makes the alternation strict: there's no special-cased "fast path"
    that would let a user's first request skip the bookkeeping and let
    their *next* request effectively cut in line ahead of someone who
    arrived in between.
    """

    def __init__(self, slots: int) -> None:
        self.slots = max(1, int(slots))
        self._in_flight = 0
        # user_key -> queue of not-yet-granted waiters for that user.
        self._queues: dict[str, deque[asyncio.Future]] = {}
        # Dispatch priority key per user_key: (0, first_seen_seq) for a
        # user who has NEVER been granted a slot, (1, last_served_seq)
        # once they have. The leading 0/1 guarantees a never-served user
        # always outranks one who's already been served at least once,
        # regardless of how the raw sequence numbers compare — a user who
        # burned through several free/uncontested slots must never keep
        # precedence over one who hasn't been served at all. Within each
        # tier, the lower sequence number (arrived earlier / served
        # longer ago) wins.
        self._first_seen: dict[str, int] = {}
        self._last_served: dict[str, int] = {}
        self._seq = 0
        self._lock = asyncio.Lock()

    def _priority_key(self, user_key: str) -> tuple[int, int]:
        if user_key in self._last_served:
            return (1, self._last_served[user_key])
        return (0, self._first_seen.get(user_key, 0))

    async def acquire(self, user_key: str) -> None:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        async with self._lock:
            if user_key not in self._first_seen:
                self._first_seen[user_key] = self._seq
                self._seq += 1
            queue = self._queues.setdefault(user_key, deque())
            queue.append(fut)
            self._dispatch_locked()
        try:
            await fut
        except asyncio.CancelledError:
            # The caller gave up before being granted a slot (e.g. client
            # disconnected while queued) — drop our own future from the
            # queue so it doesn't get "granted" into the void later and
            # silently eat a slot nobody uses. If it was already granted
            # (fut.done()) the slot is ours; release it like normal so we
            # don't leak it.
            async with self._lock:
                if fut.done() and not fut.cancelled():
                    self._in_flight -= 1
                    self._dispatch_locked()
                else:
                    queue = self._queues.get(user_key)
                    if queue is not None:
                        try:
                            queue.remove(fut)
                        except ValueError:
                            pass
                        if not queue:
                            self._queues.pop(user_key, None)
            raise

    def _dispatch_locked(self) -> None:
        """Grant queued waiters while slots are free. Caller holds ``_lock``."""
        while self._in_flight < self.slots:
            candidates = [uk for uk, q in self._queues.items() if q]
            if not candidates:
                break
            user_key = min(candidates, key=self._priority_key)
            queue = self._queues[user_key]
            fut = queue.popleft()
            if not queue:
                self._queues.pop(user_key, None)
            self._last_served[user_key] = self._seq
            self._seq += 1
            self._in_flight += 1
            if not fut.done():
                fut.set_result(None)
            else:
                # Waiter already cancelled before we got here — the slot
                # we just handed out was never actually claimed; give it
                # back and keep dispatching.
                self._in_flight -= 1

    async def release(self) -> None:
        async with self._lock:
            self._in_flight = max(0, self._in_flight - 1)
            self._dispatch_locked()

    @asynccontextmanager
    async def slot(self, user_key: str) -> AsyncIterator[None]:
        await self.acquire(user_key)
        try:
            yield
        finally:
            await self.release()


class WorkerPoolFairGates:
    """Lazily creates and caches one :class:`FairSlotGate` per worker_key.

    A worker_key already encodes the active runtime preset (see
    ``runtime_coordinator.py`` — switching presets derives a different
    worker_key), so an escalation to a bigger/smaller preset naturally
    gets a fresh gate sized for the new slot count instead of reusing a
    stale one. Gates for worker_keys that stop being used just sit idle
    (cheap — an empty gate is a few empty containers) rather than being
    actively cleaned up; that's an acceptable trade for not having to hook
    into every unload path.
    """

    def __init__(self) -> None:
        self._gates: dict[str, FairSlotGate] = {}

    def get(self, worker_key: str, slots: int) -> FairSlotGate:
        gate = self._gates.get(worker_key)
        if gate is None or gate.slots != max(1, int(slots)):
            gate = FairSlotGate(slots)
            self._gates[worker_key] = gate
        return gate
