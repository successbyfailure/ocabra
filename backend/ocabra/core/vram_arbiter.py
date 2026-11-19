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
from collections.abc import Callable, Iterator
from contextlib import asynccontextmanager, contextmanager, suppress


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
