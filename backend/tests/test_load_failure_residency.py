"""BUG-04 and BUG-05: what a failed, cancelled or Redis-less load leaves behind.

BUG-04: load() and unload() set LOADING/UNLOADING and publish the event before
their ``try``. A Redis error there escaped and left the model stuck in that
state with its worker alive.

BUG-05: the VRAM lock is taken once the worker is up, and awaits that can
still fail or be cancelled come after it. Neither error path released it.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from ocabra.backends._mock import MockBackend
from ocabra.core.model_manager import LoadPolicy, ModelManager, ModelState, ModelStatus
from ocabra.core.worker_pool import WorkerPool

MODEL = "mock/model"


class _GPUs:
    """Just the reservation bookkeeping of GPUManager, which is what matters here."""

    def __init__(self) -> None:
        self.locks: dict[int, dict[str, int]] = {}

    async def lock_vram(self, gpu: int, mb: int, model_id: str) -> None:
        self.locks.setdefault(gpu, {})[model_id] = mb

    async def unlock_vram(self, gpu: int, model_id: str) -> None:
        self.locks.get(gpu, {}).pop(model_id, None)

    def locked_gpu_indices(self, model_id: str) -> list[int]:
        return [g for g, held in self.locks.items() if model_id in held]

    def reserved(self, model_id: str) -> int:
        return sum(held.get(model_id, 0) for held in self.locks.values())

    async def get_free_vram(self, _gpu: int) -> int:
        return 20_000

    def get_process_vram_mb(self, _gpu: int) -> dict:
        return {}

    async def get_all_states(self) -> list:
        return []

    def __getattr__(self, name):  # anything else the load path asks
        return AsyncMock(return_value=None)


def _manager():
    pool = WorkerPool()
    pool.register_backend("mock", MockBackend(vram_mb=4096))
    gpus = _GPUs()
    scheduler = AsyncMock()
    scheduler.find_gpu_for_model = AsyncMock(return_value=[0])
    mm = ModelManager(pool, gpu_manager=gpus, gpu_scheduler=scheduler)
    mm._states[MODEL] = ModelState(
        model_id=MODEL, display_name=MODEL, backend_type="mock", load_policy=LoadPolicy.ON_DEMAND
    )
    mm._load_locks[MODEL] = asyncio.Lock()
    return mm, gpus


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    monkeypatch.setattr(ModelManager, "_wait_for_free_vram", AsyncMock())
    monkeypatch.setattr(ModelManager, "_wait_for_vram_released", AsyncMock())
    monkeypatch.setattr("ocabra.core.model_manager.set_key", AsyncMock())
    monkeypatch.setattr("ocabra.core.model_manager.publish_system_alert", AsyncMock())


@pytest.mark.asyncio
async def test_a_load_that_fails_after_reserving_vram_releases_it():
    mm, gpus = _manager()

    with (
        patch("ocabra.core.model_manager.publish", new=AsyncMock()),
        patch.object(mm, "_record_model_load_stat", new=AsyncMock(side_effect=RuntimeError("db"))),
        pytest.raises(RuntimeError),
    ):
        await mm.load(MODEL)

    state = await mm.get_state(MODEL)
    assert state.status == ModelStatus.ERROR
    assert gpus.reserved(MODEL) == 0
    assert state.current_gpu == [] and state.vram_used_mb == 0


@pytest.mark.asyncio
async def test_a_load_cancelled_after_reserving_vram_releases_it():
    mm, gpus = _manager()
    reached = asyncio.Event()

    async def _slow_stat(**_kwargs):
        reached.set()
        await asyncio.sleep(3600)

    with (
        patch("ocabra.core.model_manager.publish", new=AsyncMock()),
        patch.object(mm, "_record_model_load_stat", new=_slow_stat),
    ):
        task = asyncio.create_task(mm.load(MODEL))
        await asyncio.wait_for(reached.wait(), timeout=5)
        assert gpus.reserved(MODEL) > 0  # the lock really was taken
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    state = await mm.get_state(MODEL)
    assert state.status == ModelStatus.UNLOADED
    assert gpus.reserved(MODEL) == 0
    assert state.current_gpu == []


@pytest.mark.asyncio
async def test_a_redis_outage_does_not_leave_the_model_stuck():
    """Loading and unloading complete even when every event publish fails."""
    mm, gpus = _manager()

    with (
        patch("ocabra.core.model_manager.publish", new=AsyncMock(side_effect=ConnectionError)),
        patch.object(mm, "_record_model_load_stat", new=AsyncMock()),
    ):
        loaded = await mm.load(MODEL)
        assert loaded.status == ModelStatus.LOADED

        await mm.unload(MODEL)

    state = await mm.get_state(MODEL)
    assert state.status == ModelStatus.UNLOADED
    assert gpus.reserved(MODEL) == 0
