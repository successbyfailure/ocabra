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

One gate instance per worker. For a fixed llama.cpp runtime with unified KV,
the gate keeps the same worker key and arbitrates both its stable physical
slot count and its shared token pool. Runtime presets may still derive a new
worker key for backends/models that need a different process.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass


class KVCapacityError(ValueError):
    """A single request reservation is larger than the worker's KV pool."""

    def __init__(self, requested_tokens: int, capacity_tokens: int) -> None:
        self.requested_tokens = requested_tokens
        self.capacity_tokens = capacity_tokens
        super().__init__(
            f"request needs {requested_tokens} KV tokens but the pool holds {capacity_tokens}"
        )


@dataclass
class _Waiter:
    future: asyncio.Future[AdmissionTelemetry]
    user_key: str
    is_batch: bool
    kv_tokens: int
    prompt_tokens: int
    admission_limit: int
    queued_at: float
    granted: bool = False


@dataclass(frozen=True)
class AdmissionTelemetry:
    """Facts captured atomically when the gate grants a request."""

    queue_ms: int
    physical_slots: int
    admission_limit: int
    admission_concurrency: int


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

    def __init__(
        self,
        slots: int,
        kv_capacity_tokens: int | None = None,
        *,
        medium_prefill_tokens: int | None = None,
        medium_prefill_max_concurrency: int | None = None,
        large_prefill_tokens: int | None = None,
        large_prefill_max_concurrency: int | None = None,
    ) -> None:
        self.slots = max(1, int(slots))
        self.kv_capacity_tokens = (
            None if kv_capacity_tokens is None else max(0, int(kv_capacity_tokens))
        )
        self._in_flight = 0
        self._kv_tokens_in_flight = 0
        self._kv_tokens_in_flight_batch = 0
        # How many of the currently-in-flight occupants are batch-origin —
        # tracked separately so callers (preemption logic) can tell "every
        # slot is held, and all of them are batch" from "every slot is held
        # by real interactive traffic" without walking a task registry.
        self._in_flight_batch = 0
        self.medium_prefill_tokens = self._positive_or_none(medium_prefill_tokens)
        self.medium_prefill_max_concurrency = self._bounded_limit(
            medium_prefill_max_concurrency
        )
        self.large_prefill_tokens = self._positive_or_none(large_prefill_tokens)
        self.large_prefill_max_concurrency = self._bounded_limit(
            large_prefill_max_concurrency
        )
        self._admission_limits_in_flight: dict[int, int] = {}
        # user_key -> queue of not-yet-granted waiters for that user.
        self._queues: dict[str, deque[_Waiter]] = {}
        # Dispatch priority key per user_key: (batch_tier, served_tier, seq).
        # batch_tier is 0 for interactive, 1 for batch-origin — batch NEVER
        # wins the next free slot while an interactive caller is waiting,
        # regardless of serve history (2026-10-30 batch-vs-interactive
        # priority design). Within a tier: (0, first_seen_seq) for a user
        # who has NEVER been granted a slot, (1, last_served_seq) once they
        # have — a never-served user always outranks one already served at
        # least once, so nobody keeps precedence after burning through a
        # free/uncontested run. Within that, the lower sequence number
        # (arrived earlier / served longer ago) wins.
        self._first_seen: dict[str, int] = {}
        self._last_served: dict[str, int] = {}
        self._seq = 0
        self._lock = asyncio.Lock()

    @staticmethod
    def _positive_or_none(value: int | None) -> int | None:
        if value is None:
            return None
        normalized = int(value)
        return normalized if normalized > 0 else None

    def _bounded_limit(self, value: int | None) -> int:
        if value is None:
            return self.slots
        return min(self.slots, max(1, int(value)))

    def _admission_limit_for(self, prompt_tokens: int) -> int:
        prompt_tokens = max(0, int(prompt_tokens))
        if self.large_prefill_tokens is not None and prompt_tokens >= self.large_prefill_tokens:
            return self.large_prefill_max_concurrency
        if self.medium_prefill_tokens is not None and prompt_tokens >= self.medium_prefill_tokens:
            return self.medium_prefill_max_concurrency
        return self.slots

    def _active_admission_limit(self) -> int:
        active = [limit for limit, count in self._admission_limits_in_flight.items() if count > 0]
        return min(active, default=self.slots)

    def _fits_compute(self, admission_limit: int) -> bool:
        effective_limit = min(self.slots, admission_limit, self._active_admission_limit())
        return self._in_flight < effective_limit

    def _priority_key(self, user_key: str, *, is_batch: bool) -> tuple[int, int, int]:
        batch_tier = 1 if is_batch else 0
        if user_key in self._last_served:
            return (batch_tier, 1, self._last_served[user_key])
        return (batch_tier, 0, self._first_seen.get(user_key, 0))

    def peek_priority(self, user_key: str, *, is_batch: bool = False) -> tuple[int, int, int]:
        """Read-only priority key for ``user_key`` as of right now — same
        ordering ``_dispatch_locked`` would use, without registering
        anything. For display/diagnostics (e.g. the admin dashboard's
        recent-requests queue estimate) that want to show the gate's REAL
        least-recently-served-first order instead of guessing from request
        timestamps alone.
        """
        return self._priority_key(user_key, is_batch=is_batch)

    def batch_holds_all_slots(self) -> bool:
        """True when every currently-occupied slot is batch-origin.

        Used by callers upstream to decide whether an arriving interactive
        request needs active preemption (nothing will free up naturally any
        time soon) rather than just waiting its turn.
        """
        return self._in_flight >= self.slots and self._in_flight_batch >= self._in_flight

    def batch_blocks(self, kv_tokens: int = 0, prompt_tokens: int = 0) -> bool:
        """Whether running batch work prevents this interactive admission.

        A weighted pool can be saturated before every physical slot is in use,
        so slot occupancy alone is not enough to decide whether preemption may
        help. This remains conservative: no batch reservation means no
        preemption request, even if interactive work also fills the pool.
        """
        if self.batch_holds_all_slots():
            return True
        requested = max(0, int(kv_tokens))
        kv_blocked = (
            requested > 0 and not self._fits_kv(requested) and self._kv_tokens_in_flight_batch > 0
        )
        compute_blocked = (
            not self._fits_compute(self._admission_limit_for(prompt_tokens))
            and self._in_flight_batch > 0
        )
        return kv_blocked or compute_blocked

    def compute_saturated(self, prompt_tokens: int = 0) -> bool:
        """Whether a candidate prompt is blocked by adaptive admission."""
        return not self._fits_compute(self._admission_limit_for(prompt_tokens))

    def snapshot(self) -> dict[str, int | bool | None]:
        """Return logical admission state without exposing mutable internals."""
        capacity = self.kv_capacity_tokens
        free = None if capacity is None else max(0, capacity - self._kv_tokens_in_flight)
        active_limit = self._active_admission_limit()
        return {
            "in_flight": self._in_flight,
            "queued": sum(len(queue) for queue in self._queues.values()),
            "slots": self.slots,
            "slot_saturated": self._in_flight >= self.slots,
            "admission_limit": active_limit,
            "compute_saturated": self._in_flight >= active_limit,
            "kv_capacity_tokens": capacity,
            "kv_reserved_tokens": self._kv_tokens_in_flight,
            "kv_free_tokens": free,
        }

    def is_idle(self) -> bool:
        """Return whether replacing this gate cannot orphan a lease/waiter."""
        return self._in_flight == 0 and not any(self._queues.values())

    def _fits_kv(self, kv_tokens: int) -> bool:
        return (
            self.kv_capacity_tokens is None
            or self._kv_tokens_in_flight + kv_tokens <= self.kv_capacity_tokens
        )

    async def acquire(
        self,
        user_key: str,
        *,
        is_batch: bool = False,
        kv_tokens: int = 0,
        prompt_tokens: int = 0,
    ) -> AdmissionTelemetry:
        kv_tokens = max(0, int(kv_tokens))
        if self.kv_capacity_tokens is not None and kv_tokens > self.kv_capacity_tokens:
            raise KVCapacityError(kv_tokens, self.kv_capacity_tokens)

        loop = asyncio.get_running_loop()
        waiter = _Waiter(
            future=loop.create_future(),
            user_key=user_key,
            is_batch=is_batch,
            kv_tokens=kv_tokens,
            prompt_tokens=max(0, int(prompt_tokens)),
            admission_limit=self._admission_limit_for(prompt_tokens),
            queued_at=time.monotonic(),
        )
        async with self._lock:
            if user_key not in self._first_seen:
                self._first_seen[user_key] = self._seq
                self._seq += 1
            queue = self._queues.setdefault(user_key, deque())
            queue.append(waiter)
            self._dispatch_locked()
        try:
            return await waiter.future
        except asyncio.CancelledError:
            # The caller gave up before being granted a slot (e.g. client
            # disconnected while queued) — drop our own future from the
            # queue so it doesn't get "granted" into the void later and
            # silently eat a slot nobody uses. If it was already granted
            # (fut.done()) the slot is ours; release it like normal so we
            # don't leak it.
            async with self._lock:
                if waiter.granted:
                    self._release_locked(
                        waiter.is_batch, waiter.kv_tokens, waiter.admission_limit
                    )
                    self._dispatch_locked()
                else:
                    queue = self._queues.get(user_key)
                    if queue is not None:
                        try:
                            queue.remove(waiter)
                        except ValueError:
                            pass
                        if not queue:
                            self._queues.pop(user_key, None)
            raise

    def _dispatch_locked(self) -> None:
        """Grant queued waiters while slots are free. Caller holds ``_lock``."""
        while self._in_flight < self.slots:
            # Cancelled queued futures can be completed by asyncio before the
            # acquire coroutine gets CPU time to remove itself. Discard them
            # here so they neither block a user's head nor consume capacity.
            for user_key, queue in list(self._queues.items()):
                while queue and queue[0].future.done() and not queue[0].granted:
                    queue.popleft()
                if not queue:
                    self._queues.pop(user_key, None)

            candidates = [
                user_key
                for user_key, queue in self._queues.items()
                if queue
                and self._fits_kv(queue[0].kv_tokens)
                and self._fits_compute(queue[0].admission_limit)
            ]
            if not candidates:
                break
            user_key = min(
                candidates,
                key=lambda candidate: self._priority_key(
                    candidate,
                    is_batch=self._queues[candidate][0].is_batch,
                ),
            )
            queue = self._queues[user_key]
            waiter = queue.popleft()
            if not queue:
                self._queues.pop(user_key, None)
            self._last_served[user_key] = self._seq
            self._seq += 1
            self._in_flight += 1
            self._kv_tokens_in_flight += waiter.kv_tokens
            self._admission_limits_in_flight[waiter.admission_limit] = (
                self._admission_limits_in_flight.get(waiter.admission_limit, 0) + 1
            )
            waiter.granted = True
            if waiter.is_batch:
                self._in_flight_batch += 1
                self._kv_tokens_in_flight_batch += waiter.kv_tokens
            if not waiter.future.done():
                waiter.future.set_result(
                    AdmissionTelemetry(
                        queue_ms=max(0, int((time.monotonic() - waiter.queued_at) * 1000)),
                        physical_slots=self.slots,
                        admission_limit=min(
                            waiter.admission_limit, self._active_admission_limit()
                        ),
                        admission_concurrency=self._in_flight,
                    )
                )
            else:
                # Waiter already cancelled before we got here — the slot
                # we just handed out was never actually claimed; give it
                # back and keep dispatching.
                waiter.granted = False
                self._release_locked(waiter.is_batch, waiter.kv_tokens, waiter.admission_limit)

    def _release_locked(self, is_batch: bool, kv_tokens: int, admission_limit: int) -> None:
        self._in_flight = max(0, self._in_flight - 1)
        self._kv_tokens_in_flight = max(0, self._kv_tokens_in_flight - kv_tokens)
        count = self._admission_limits_in_flight.get(admission_limit, 0)
        if count <= 1:
            self._admission_limits_in_flight.pop(admission_limit, None)
        else:
            self._admission_limits_in_flight[admission_limit] = count - 1
        if is_batch:
            self._in_flight_batch = max(0, self._in_flight_batch - 1)
            self._kv_tokens_in_flight_batch = max(0, self._kv_tokens_in_flight_batch - kv_tokens)

    async def release(
        self,
        *,
        is_batch: bool = False,
        kv_tokens: int = 0,
        admission_limit: int | None = None,
    ) -> None:
        async with self._lock:
            self._release_locked(
                is_batch,
                max(0, int(kv_tokens)),
                self.slots if admission_limit is None else int(admission_limit),
            )
            self._dispatch_locked()

    @asynccontextmanager
    async def slot(
        self,
        user_key: str,
        *,
        is_batch: bool = False,
        kv_tokens: int = 0,
        prompt_tokens: int = 0,
    ) -> AsyncIterator[AdmissionTelemetry]:
        admission = await self.acquire(
            user_key,
            is_batch=is_batch,
            kv_tokens=kv_tokens,
            prompt_tokens=prompt_tokens,
        )
        try:
            yield admission
        finally:
            await self.release(
                is_batch=is_batch,
                kv_tokens=kv_tokens,
                admission_limit=self._admission_limit_for(prompt_tokens),
            )


class WorkerPoolFairGates:
    """Lazily creates and caches one :class:`FairSlotGate` per worker_key.

    Fixed unified-KV runtimes reuse one gate. If their configured slot count
    or pool capacity changes after a drained reload, :meth:`get` replaces the
    idle gate with one sized for the new limits. A runtime preset can instead
    derive another worker key; its gate is then independent. Gates for worker
    keys that stop being used remain idle (cheap — an empty gate is a few
    containers) rather than hooking cleanup into every unload path.
    """

    def __init__(self) -> None:
        self._gates: dict[str, FairSlotGate] = {}

    def get(
        self,
        worker_key: str,
        slots: int,
        kv_capacity_tokens: int | None = None,
        *,
        medium_prefill_tokens: int | None = None,
        medium_prefill_max_concurrency: int | None = None,
        large_prefill_tokens: int | None = None,
        large_prefill_max_concurrency: int | None = None,
    ) -> FairSlotGate:
        gate = self._gates.get(worker_key)
        normalized_slots = max(1, int(slots))
        normalized_capacity = (
            None if kv_capacity_tokens is None else max(0, int(kv_capacity_tokens))
        )
        if (
            gate is None
            or gate.slots != normalized_slots
            or gate.kv_capacity_tokens != normalized_capacity
            or gate.medium_prefill_tokens != medium_prefill_tokens
            or gate.medium_prefill_max_concurrency
            != min(normalized_slots, max(1, int(medium_prefill_max_concurrency or normalized_slots)))
            or gate.large_prefill_tokens != large_prefill_tokens
            or gate.large_prefill_max_concurrency
            != min(normalized_slots, max(1, int(large_prefill_max_concurrency or normalized_slots)))
        ):
            if gate is not None and not gate.is_idle():
                raise RuntimeError(
                    f"cannot resize active admission gate '{worker_key}'; "
                    "runtime configuration must use a distinct worker key"
                )
            gate = FairSlotGate(
                normalized_slots,
                normalized_capacity,
                medium_prefill_tokens=medium_prefill_tokens,
                medium_prefill_max_concurrency=medium_prefill_max_concurrency,
                large_prefill_tokens=large_prefill_tokens,
                large_prefill_max_concurrency=large_prefill_max_concurrency,
            )
            self._gates[worker_key] = gate
        return gate

    def peek(self, worker_key: str) -> FairSlotGate | None:
        """Non-creating lookup — for read-only diagnostics that must never
        instantiate a gate for a worker that isn't actually using fairness."""
        return self._gates.get(worker_key)
