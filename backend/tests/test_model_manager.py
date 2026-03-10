import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ocabra.backends._mock import MockBackend
from ocabra.core.model_manager import LoadPolicy, ModelManager, ModelStatus
from ocabra.core.worker_pool import WorkerPool


def make_worker_pool_with_mock(vram_mb: int = 4096) -> WorkerPool:
    wp = WorkerPool()
    wp.register_backend("mock", MockBackend(vram_mb=vram_mb))
    return wp


@pytest.fixture
def worker_pool():
    return make_worker_pool_with_mock()


@pytest.fixture
def model_manager(worker_pool):
    return ModelManager(worker_pool)


async def add_test_model(mm, model_id="test/model", load_policy="on_demand"):
    """Helper: add a model bypassing DB."""
    from ocabra.core.model_manager import ModelState
    mm._states[model_id] = ModelState(
        model_id=model_id,
        display_name=model_id,
        backend_type="mock",
        load_policy=LoadPolicy(load_policy),
    )
    mm._load_locks[model_id] = asyncio.Lock()
    return mm._states[model_id]


@pytest.mark.asyncio
async def test_load_model_full_cycle(model_manager):
    """CONFIGURED → LOADED → UNLOADED cycle."""
    with patch("ocabra.core.model_manager.publish", new=AsyncMock()), \
         patch("ocabra.core.model_manager.set_key", new=AsyncMock()):
        await add_test_model(model_manager)

        state = await model_manager.load("test/model")
        assert state.status == ModelStatus.LOADED
        assert state.vram_used_mb == 4096
        assert state.loaded_at is not None

        await model_manager.unload("test/model")
        state = await model_manager.get_state("test/model")
        assert state.status == ModelStatus.UNLOADED
        assert state.vram_used_mb == 0
        assert state.loaded_at is None


@pytest.mark.asyncio
async def test_pin_policy_loads_on_start(worker_pool):
    """Models with load_policy=pin are loaded automatically on start()."""
    mm = ModelManager(worker_pool)

    with patch("ocabra.core.model_manager.publish", new=AsyncMock()), \
         patch("ocabra.core.model_manager.set_key", new=AsyncMock()), \
         patch.object(mm, "_load_configs_from_db", new=AsyncMock()):
        await add_test_model(mm, load_policy="pin")
        await mm.start()
        await asyncio.sleep(0.1)  # let the task run

        state = await mm.get_state("test/model")
        assert state.status == ModelStatus.LOADED


@pytest.mark.asyncio
async def test_on_request_loads_unloaded_model(model_manager):
    """on_request() auto-loads an UNLOADED model."""
    with patch("ocabra.core.model_manager.publish", new=AsyncMock()), \
         patch("ocabra.core.model_manager.set_key", new=AsyncMock()):
        state = await add_test_model(model_manager)
        state.status = ModelStatus.UNLOADED

        await model_manager.on_request("test/model")
        assert state.status == ModelStatus.LOADED


@pytest.mark.asyncio
async def test_idle_eviction(model_manager):
    """on_demand model idle > timeout is evicted."""
    from datetime import datetime, timedelta, timezone

    with patch("ocabra.core.model_manager.publish", new=AsyncMock()), \
         patch("ocabra.core.model_manager.set_key", new=AsyncMock()), \
         patch.object(model_manager._worker_pool._backends["mock"], "unload", new=AsyncMock()):
        state = await add_test_model(model_manager, load_policy="on_demand")
        state.status = ModelStatus.LOADED
        state.last_request_at = datetime.now(timezone.utc) - timedelta(seconds=999)

        with patch.object(model_manager._worker_pool._backends["mock"], "unload", new=AsyncMock()):
            await model_manager.check_idle_evictions()
            await asyncio.sleep(0.1)

        final = await model_manager.get_state("test/model")
        assert final.status == ModelStatus.UNLOADED


@pytest.mark.asyncio
async def test_concurrent_load_only_loads_once(model_manager):
    """Concurrent calls to load() only trigger one actual load."""
    load_count = 0
    original_load = model_manager._worker_pool._backends["mock"].load

    async def counting_load(*args, **kwargs):
        nonlocal load_count
        load_count += 1
        await asyncio.sleep(0.05)
        return await original_load(*args, **kwargs)

    model_manager._worker_pool._backends["mock"].load = counting_load

    with patch("ocabra.core.model_manager.publish", new=AsyncMock()), \
         patch("ocabra.core.model_manager.set_key", new=AsyncMock()):
        await add_test_model(model_manager)
        await asyncio.gather(
            model_manager.load("test/model"),
            model_manager.load("test/model"),
            model_manager.load("test/model"),
        )

    assert load_count == 1
