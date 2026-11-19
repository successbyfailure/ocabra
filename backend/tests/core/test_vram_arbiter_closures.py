"""Lo que la primera versión del árbitro dejó fuera (REF-03): resize_context
desde la API / la sonda / _return_to_initial, grow_context_back sin regla de
rendición y el crecimiento de whisper dentro de la petición sin reserva.

Cada carrera se fuerza con eventos; los tiempos solo acotan cuánto se mira.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from ocabra.config import settings
from ocabra.core.model_manager import LoadGateTimeoutError, ModelStatus
from tests.test_live_context import _manager

_LOOK_S = 0.4


@pytest.fixture(autouse=True)
def _quiet():
    with (
        patch("ocabra.core.model_manager.publish", new=AsyncMock()),
        patch("ocabra.core.model_manager.set_key", new=AsyncMock()),
    ):
        yield


async def _holder(mm, held: asyncio.Event, release: asyncio.Event, *, waiting_on=None):
    async with mm._acquire_load_gate("test-load"):
        if waiting_on is None:
            held.set()
            await release.wait()
            return
        with mm._arbiter.waiting_on(waiting_on):
            held.set()
            await release.wait()


async def test_resize_from_the_api_waits_for_the_arbiter(monkeypatch):
    mm, state, backend, _ = _manager(monkeypatch)
    held, release = asyncio.Event(), asyncio.Event()
    h = asyncio.create_task(_holder(mm, held, release))
    await held.wait()
    resize = asyncio.create_task(mm.resize_context(state.model_id, context=16384))
    await asyncio.sleep(_LOOK_S)
    assert backend.reconfigure_context.await_count == 0, "resized under someone else's arbiter"
    release.set()
    await asyncio.wait_for(resize, timeout=3)
    await h
    assert backend.reconfigure_context.await_count == 1


async def test_resize_from_the_api_gives_up_when_the_wait_runs_out(monkeypatch):
    monkeypatch.setattr(settings, "pressure_eviction_drain_timeout_s", 1)
    mm, state, backend, _ = _manager(monkeypatch)
    held, release = asyncio.Event(), asyncio.Event()
    h = asyncio.create_task(_holder(mm, held, release))
    await held.wait()
    with pytest.raises(RuntimeError, match="retry"):
        await asyncio.wait_for(mm.resize_context(state.model_id, context=16384), timeout=3)
    release.set()
    await h
    assert backend.reconfigure_context.await_count == 0


async def test_return_to_initial_never_waits_behind_a_load(monkeypatch):
    mm, state, backend, _ = _manager(monkeypatch)
    state.live_context = {"n_ctx": 262144, "initial_ctx": 65536, "nominal": {"n_ctx": 262144}}
    held, release = asyncio.Event(), asyncio.Event()
    h = asyncio.create_task(_holder(mm, held, release))
    await held.wait()
    mm._load_in_progress.add("other")
    await asyncio.wait_for(mm._return_to_initial(state.model_id), timeout=2)
    assert backend.reconfigure_context.await_count == 0
    mm._load_in_progress.discard("other")
    release.set()
    await h


async def test_resize_drain_is_a_passive_wait_of_the_owner(monkeypatch):
    mm, state, backend, _ = _manager(monkeypatch, busy=True)
    monkeypatch.setattr(settings, "llama_cpp_live_context_drain_timeout_s", 5)
    resize = asyncio.create_task(mm.resize_context(state.model_id, context=16384))
    for _ in range(100):
        if mm._arbiter.is_waiting_on(state.model_id):
            break
        await asyncio.sleep(0.01)
    assert mm._arbiter.is_waiting_on(state.model_id)
    mm._in_flight[state.model_id] = 0
    await asyncio.wait_for(resize, timeout=3)
    assert backend.reconfigure_context.await_count == 1


async def test_resize_of_a_model_being_loaded_does_not_wait_for_its_lock(monkeypatch):
    """Arbiter -> model lock against a load's model lock -> arbiter: the
    resize gives up instead of both waiting until the timeouts."""
    monkeypatch.setattr(settings, "max_drain_timeout_s", 30)
    mm, state, backend, _ = _manager(monkeypatch)
    lock_taken = asyncio.Event()
    got_gate = asyncio.Event()

    async def load_like():
        async with mm._acquire_model_load_lock(state.model_id):
            state.status = ModelStatus.LOADING
            lock_taken.set()
            async with mm._acquire_load_gate():
                got_gate.set()
            state.status = ModelStatus.LOADED

    # The resize gets the arbiter first, then the load takes the model lock.
    entered = asyncio.Event()
    original = mm._resize_context_held

    async def held_then_wait(*args, **kwargs):
        entered.set()
        await lock_taken.wait()
        return await original(*args, **kwargs)

    mm._resize_context_held = held_then_wait
    resize = asyncio.create_task(mm.resize_context(state.model_id, context=16384))
    await entered.wait()
    load = asyncio.create_task(load_like())
    with pytest.raises(RuntimeError, match="not loaded"):
        await asyncio.wait_for(resize, timeout=3)
    await asyncio.wait_for(got_gate.wait(), timeout=3)
    await load
    assert backend.reconfigure_context.await_count == 0


async def test_grow_context_back_gives_up_when_the_owner_drains_that_worker(monkeypatch):
    monkeypatch.setattr(settings, "max_drain_timeout_s", 5)
    monkeypatch.setattr(settings, "pressure_eviction_drain_timeout_s", 5)
    mm, state, backend, _ = _manager(monkeypatch)
    state.live_context = {"n_ctx": 16384, "nominal": {"n_ctx": 262144}}
    held, release = asyncio.Event(), asyncio.Event()
    h = asyncio.create_task(_holder(mm, held, release, waiting_on=state.model_id))
    await held.wait()
    loop = asyncio.get_running_loop()
    started = loop.time()
    with pytest.raises(LoadGateTimeoutError):
        await mm.grow_context_back(state.model_id)
    assert loop.time() - started < 1.5
    release.set()
    await h
    assert backend.reconfigure_context.await_count == 0
