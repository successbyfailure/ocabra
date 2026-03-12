import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ocabra.backends._mock import MockBackend
from ocabra.backends.base import BackendCapabilities, BackendInterface, WorkerInfo
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


@pytest.mark.asyncio
async def test_load_evicts_on_demand_model_on_vram_pressure(worker_pool):
    mm = ModelManager(worker_pool, gpu_scheduler=AsyncMock())

    with patch("ocabra.core.model_manager.publish", new=AsyncMock()), \
         patch("ocabra.core.model_manager.set_key", new=AsyncMock()), \
         patch.object(mm, "unload", new=AsyncMock()) as unload_mock:
        requested = await add_test_model(mm, model_id="test/requested")
        candidate = await add_test_model(mm, model_id="test/candidate", load_policy="on_demand")
        candidate.status = ModelStatus.LOADED
        candidate.current_gpu = [0]
        candidate.loaded_at = datetime.now(timezone.utc) - timedelta(minutes=10)
        candidate.last_request_at = datetime.now(timezone.utc) - timedelta(minutes=5)

        from ocabra.core.scheduler import InsufficientVRAMError

        mm._gpu_scheduler.find_gpu_for_model = AsyncMock(
            side_effect=[
                InsufficientVRAMError("full"),
                [1],
            ]
        )

        state = await mm.load("test/requested")

    assert state.status == ModelStatus.LOADED
    unload_mock.assert_awaited_once_with("test/candidate", reason="pressure")
    assert requested.current_gpu == [1]


@pytest.mark.asyncio
async def test_pressure_eviction_prefers_on_demand_then_warm_then_pin(worker_pool):
    mm = ModelManager(worker_pool)

    now = datetime.now(timezone.utc)
    on_demand = await add_test_model(mm, model_id="test/on-demand", load_policy="on_demand")
    warm = await add_test_model(mm, model_id="test/warm", load_policy="warm")
    pin = await add_test_model(mm, model_id="test/pin", load_policy="pin")

    on_demand.status = ModelStatus.LOADED
    on_demand.current_gpu = [0]
    on_demand.last_request_at = now - timedelta(minutes=3)

    warm.status = ModelStatus.LOADED
    warm.current_gpu = [1]
    warm.last_request_at = now - timedelta(minutes=30)

    pin.status = ModelStatus.LOADED
    pin.current_gpu = [0]
    pin.last_request_at = now - timedelta(hours=1)

    assert mm._get_pressure_eviction_candidates("test/requested") == [
        "test/on-demand",
        "test/warm",
        "test/pin",
    ]


class _PortRequiredBackend(BackendInterface):
    def __init__(self) -> None:
        self.received_port = None

    async def load(self, model_id: str, gpu_indices: list[int], **kwargs) -> WorkerInfo:
        port = kwargs.get("port", 0)
        if not port:
            raise ValueError("port is required")
        self.received_port = int(port)
        return WorkerInfo(
            backend_type="portreq",
            model_id=model_id,
            gpu_indices=gpu_indices,
            port=self.received_port,
            pid=1234,
            vram_used_mb=1024,
        )

    async def unload(self, model_id: str) -> None:
        return None

    async def health_check(self, model_id: str) -> bool:
        return True

    async def get_capabilities(self, model_id: str) -> BackendCapabilities:
        return BackendCapabilities(chat=True, completion=True, streaming=True)

    async def get_vram_estimate_mb(self, model_id: str) -> int:
        return 1024

    async def forward_request(self, model_id: str, path: str, body: dict):
        return {}

    async def forward_stream(self, model_id: str, path: str, body: dict):
        if False:
            yield b""


@pytest.mark.asyncio
async def test_load_assigns_port_for_backend_that_requires_it():
    wp = WorkerPool()
    backend = _PortRequiredBackend()
    wp.register_backend("portreq", backend)
    mm = ModelManager(wp)

    with patch("ocabra.core.model_manager.publish", new=AsyncMock()), \
         patch("ocabra.core.model_manager.set_key", new=AsyncMock()):
        from ocabra.core.model_manager import ModelState

        mm._states["test/port-required"] = ModelState(
            model_id="test/port-required",
            display_name="test/port-required",
            backend_type="portreq",
            load_policy=LoadPolicy.ON_DEMAND,
        )
        mm._load_locks["test/port-required"] = asyncio.Lock()

        state = await mm.load("test/port-required")

    assert state.status == ModelStatus.LOADED
    assert backend.received_port is not None
    worker = wp.get_worker("test/port-required")
    assert worker is not None
    assert worker.port == backend.received_port
