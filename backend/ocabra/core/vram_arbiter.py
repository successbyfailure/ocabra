"""Árbitro de VRAM: un único dueño de las operaciones que cambian la VRAM.

Primera versión de REF-03 (cierra lo que quedaba de BUG-23). El mutex es la
puerta de carga de siempre (``ModelManager._backend_load_gate``): una carga
la toma desde la admisión hasta el arranque del backend, y ahora también la
toman las demás operaciones que cambian la VRAM fuera de una carga
(``unload`` manual, ``ensure_vram_free`` de los servicios, el encogimiento de
``make_room_for_whisper`` y ``make_room_for_vision``, el vigilante de VRAM).
Diseño y argumento de no-interbloqueo en ``docs/tasks/vram-arbiter-plan.md``.

Lo que añade a un ``asyncio.Lock``:

* Reentrada para la tarea que ya lo tiene (una expulsión dentro de una carga
  llama a ``unload()``, que vuelve a pedirlo).
* Espera acotada que se puede abandonar (``abort_if``), sondeada mientras se
  espera.
* Registro de lo que el dueño está esperando *pasivamente* (``waiting_on``):
  el drenado de las peticiones de un modelo o la gracia de un servicio. Un
  camino que espera el árbitro mientras tiene una petición en ese modelo (o
  es ese servicio) se rinde en vez de formar un ciclo con el dueño.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Iterator, Mapping
from contextlib import asynccontextmanager, contextmanager, suppress
from datetime import UTC, datetime


class LoadGateTimeoutError(TimeoutError):
    """Waiting for the global load gate (or a model's own load lock) ran out.

    Retryable: nothing failed, another model's load/eviction is still holding
    the GPU (a load may legitimately drain busy neighbours for minutes). The
    waiting load leaves its model as it was instead of marking it ERROR, and
    the API answers 503 with ``Retry-After`` like any other failed load.
    Subclasses ``TimeoutError`` so existing ``except TimeoutError`` keep
    catching it.
    """


async def acquire_lock_unless(
    lock: asyncio.Lock, abort_if: Callable[[], bool], timeout_s: float
) -> bool | None:
    """Acquire ``lock`` unless ``abort_if()`` turns true first.

    Returns True (held), False (aborted, not held) or None (timed out, not
    held). Not ``asyncio.wait_for``: on 3.11 a timeout racing the acquisition
    can drop a lock that was in fact taken; here it is handed back.
    """
    if abort_if():
        return False
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    acquire = asyncio.ensure_future(lock.acquire())
    aborted = False
    try:
        while True:
            done, _ = await asyncio.wait({acquire}, timeout=0.25)
            if done:
                return True
            if abort_if():
                aborted = True
                break
            if loop.time() >= deadline:
                break
    except BaseException:
        acquire.cancel()
        with suppress(asyncio.CancelledError):
            await acquire
        if not acquire.cancelled() and acquire.exception() is None:
            lock.release()
        raise
    acquire.cancel()
    with suppress(asyncio.CancelledError):
        await acquire
    if not acquire.cancelled() and acquire.exception() is None:
        # Got the lock in the same tick we gave up: hand it back.
        lock.release()
    return False if aborted else None


async def wait_for_reading(
    read: Callable[[], Awaitable[int | None]],
    *,
    target: int,
    max_s: float,
    poll_s: float,
    settle_s: float | None = None,
    initial: int | None = None,
) -> int | None:
    """Poll ``read()`` until it reaches ``target``; returns the last reading.

    The one shape of every "wait until the VRAM is released" in the arbiter:
    it ends as soon as the reading shows the memory (never on a fixed sleep),
    when the reading has stopped moving for ``settle_s`` (the release is over
    and was not enough: the caller moves on), or at ``max_s`` (bounded,
    whatever the GPU does). ``read`` returning ``None`` (no reading) ends it.
    ``initial`` is a reading the caller just took: used as the first one.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + max(0.0, float(max_s))
    last: int | None = None
    last_change = loop.time()
    pending = initial
    while True:
        if pending is not None:
            value, pending = pending, None
        else:
            value = await read()
        if value is None or value >= target:
            return value
        now = loop.time()
        if last is None or value != last:
            last, last_change = value, now
        elif settle_s is not None and now - last_change >= settle_s:
            return value
        if now >= deadline:
            return value
        await asyncio.sleep(poll_s)


# ------------------------------------------------------------------ planner
#
# Pure decisions over a snapshot of the residents: which models an eviction
# may take and in what order. No awaits, no side effects: the executor
# (ModelManager, under the arbiter) applies them. States are duck-typed
# (``ModelState``): this module must not import model_manager.

_POLICY_ORDER = {"on_demand": 0, "warm": 1, "pin": 2}
_EPOCH = datetime.min.replace(tzinfo=UTC)


def _policy(state) -> str:
    policy = getattr(state, "load_policy", None)
    return str(getattr(policy, "value", policy))


def _status(state) -> str:
    status = getattr(state, "status", None)
    return str(getattr(status, "value", status))


def is_hard_pin(state) -> bool:
    """``pin_permanent`` only means something for PIN models."""
    return bool(getattr(state, "pin_permanent", False)) and _policy(state) == "pin"


def _last_used(state) -> datetime:
    return state.last_request_at or state.loaded_at or _EPOCH


def rank_lru_candidates(
    states: Iterable,
    gpu_index: int,
    *,
    is_busy: Callable[[str], bool],
    is_session_held: Callable[[str], bool],
) -> list[str]:
    """The polite eviction list for one GPU: LOADED warm/on-demand models on
    it (never a PIN), idle, not held by a Realtime session, least recently
    used first."""
    candidates = [
        state
        for state in states
        if _status(state) == "loaded"
        and _policy(state) != "pin"
        and gpu_index in (state.current_gpu or [])
        and not is_busy(state.model_id)
        and not is_session_held(state.model_id)
    ]
    candidates.sort(key=_last_used)
    return [state.model_id for state in candidates]


def rank_pressure_candidates(
    states: Mapping[str, object],
    workers: Mapping[str, object],
    requested_model_id: str,
    gpu_indices: list[int] | None,
    *,
    is_session_held: Callable[[str], bool],
) -> list[str]:
    """The last-resort eviction list (see
    ``ModelManager._get_pressure_eviction_candidates`` for the reasons):
    LOADED models on one of ``gpu_indices`` (any GPU when ``None``; never a
    CPU-only one), on-demand < warm < soft pin, least recently used first
    within a policy. Never the requester, a hard pin or a Realtime session's
    worker. With none, falls back to the worker pool's GPU workers (not
    Ollama's), still never a hard pin."""

    def on_target(gpus) -> bool:
        if not gpus:
            return False
        return gpu_indices is None or any(g in gpu_indices for g in gpus)

    candidates = [
        state
        for state in states.values()
        if state.model_id != requested_model_id
        and _status(state) == "loaded"
        and not is_hard_pin(state)
        and not is_session_held(state.model_id)
        and on_target(state.current_gpu)
    ]
    candidates.sort(key=lambda state: (_POLICY_ORDER.get(_policy(state), 99), _last_used(state)))
    if candidates:
        return [state.model_id for state in candidates]

    # Fallback: workers the states above do not offer (a worker whose state
    # went ERROR with its process alive, or lost its GPUs). Never a hard pin:
    # this fallback used to return every worker on the card.
    def fallback_ok(model_id: str) -> bool:
        state = states.get(model_id)
        return state is None or not is_hard_pin(state)

    return [
        model_id
        for model_id, worker in workers.items()
        if model_id != requested_model_id
        and worker.backend_type != "ollama"
        and not is_session_held(model_id)
        and on_target(worker.gpu_indices)
        and fallback_ok(model_id)
    ]


def _never() -> bool:
    return False


class VramArbiter:
    """Serialises every operation that changes VRAM (see module docstring)."""

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self._owner: asyncio.Task | None = None
        self._purpose: str | None = None
        # Keys (model ids, ``service:<id>``) the owner is passively waiting
        # for, with a count so nested registrations unwind correctly.
        self._waiting_on: dict[str, int] = {}

    # ---------------------------------------------------------------- state

    def held_by_current(self) -> bool:
        current = asyncio.current_task()
        return current is not None and self._owner is current

    def locked(self) -> bool:
        return self.lock.locked()

    def busy(self) -> bool:
        """Some *other* task holds the arbiter."""
        return self.lock.locked() and not self.held_by_current()

    @property
    def purpose(self) -> str | None:
        """What the current owner is doing (``load``, ``unload``, ...)."""
        return self._purpose if self.lock.locked() else None

    # -------------------------------------------------------------- holding

    @asynccontextmanager
    async def hold(
        self,
        *,
        timeout_s: float,
        purpose: str,
        abort_if: Callable[[], bool] | None = None,
        timeout_yields_false: bool = False,
    ):
        """Hold the arbiter; yields True, or False when ``abort_if`` won.

        Reentrant for the task that holds it (yields True without waiting).

        Raises:
            LoadGateTimeoutError: the wait ran out (retryable), unless
                ``timeout_yields_false`` (then it yields False instead).
        """
        if self.held_by_current():
            yield True
            return
        acquired = await acquire_lock_unless(self.lock, abort_if or _never, timeout_s)
        if acquired is None and timeout_yields_false:
            acquired = False
        if acquired is None:
            raise LoadGateTimeoutError(
                "Timed out waiting for the GPU load lock — another model's "
                "load/eviction is still holding it"
            )
        if not acquired:
            yield False
            return
        async with self._owned(purpose):
            yield True

    @asynccontextmanager
    async def try_hold(self, *, purpose: str):
        """Hold the arbiter only if nobody does; yields whether it is held."""
        if self.held_by_current():
            yield True
            return
        if self.lock.locked():
            yield False
            return
        if not await acquire_lock_unless(self.lock, _never, 1.0):
            yield False
            return
        async with self._owned(purpose):
            yield True

    @asynccontextmanager
    async def _owned(self, purpose: str):
        """Body shared by ``hold``/``try_hold`` once ``self.lock`` is taken."""
        self._owner = asyncio.current_task()
        self._purpose = purpose
        try:
            yield
        finally:
            self._owner = None
            self._purpose = None
            # A waiting_on left behind by a cancelled owner must not make the
            # next owner's waiters give up.
            self._waiting_on.clear()
            self.lock.release()

    # ------------------------------------------------------- passive waits

    @contextmanager
    def waiting_on(self, key: str) -> Iterator[None]:
        """Mark that the owner is waiting for ``key`` (a model's requests to
        drain, a service's generation to end). No-op for a non-owner: only
        the owner's waits can close a cycle through the arbiter."""
        if not self.held_by_current():
            yield
            return
        self._waiting_on[key] = self._waiting_on.get(key, 0) + 1
        try:
            yield
        finally:
            remaining = self._waiting_on.get(key, 0) - 1
            if remaining > 0:
                self._waiting_on[key] = remaining
            else:
                self._waiting_on.pop(key, None)

    def is_waiting_on(self, key: str) -> bool:
        return self.lock.locked() and key in self._waiting_on

    def is_waiting(self) -> bool:
        """The owner is in a passive wait (draining, a service's grace):
        it measures nothing until that ends and re-reads afterwards."""
        return self.lock.locked() and bool(self._waiting_on)
