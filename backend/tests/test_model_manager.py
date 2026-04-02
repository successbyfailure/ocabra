import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import sqlalchemy as sa

from ocabra.backends._mock import MockBackend
from ocabra.backends.base import BackendCapabilities, BackendInterface, WorkerInfo
from ocabra.core.model_manager import LoadPolicy, ModelManager, ModelStatus
from ocabra.core.worker_pool import WorkerPool
from ocabra.database import AsyncSessionLocal
from ocabra.db.stats import ModelLoadStat


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
async def test_load_records_model_load_stat(model_manager):
    with patch("ocabra.core.model_manager.publish", new=AsyncMock()), \
         patch("ocabra.core.model_manager.set_key", new=AsyncMock()):
        await add_test_model(model_manager)

        state = await model_manager.load("test/model")
        assert state.status == ModelStatus.LOADED

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            sa.select(ModelLoadStat).where(ModelLoadStat.model_id == "test/model")
        )
        rows = result.scalars().all()

    assert rows
    assert rows[-1].duration_ms is not None
    assert rows[-1].duration_ms >= 0


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
async def test_in_flight_counters_are_thread_safe(model_manager):
    model_id = "test/model"
    await add_test_model(model_manager, model_id=model_id)

    await asyncio.gather(
        *[asyncio.to_thread(model_manager.begin_request, model_id) for _ in range(200)]
    )
    assert model_manager._in_flight[model_id] == 200

    await asyncio.gather(
        *[asyncio.to_thread(model_manager.end_request, model_id) for _ in range(200)]
    )
    assert model_manager._in_flight.get(model_id, 0) == 0


@pytest.mark.asyncio
async def test_background_task_failure_is_logged(model_manager, monkeypatch):
    from ocabra.core import model_manager as model_manager_module

    fake_logger = MagicMock()
    monkeypatch.setattr(model_manager_module, "logger", fake_logger)

    async def boom():
        raise RuntimeError("boom")

    task = model_manager._create_background_task(
        boom(),
        task_name="boom-task",
        model_id="test/model",
    )

    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert task.done()
    fake_logger.warning.assert_called_once()


@pytest.mark.asyncio
async def test_touch_last_request_at_persists_timestamp(model_manager):
    now = datetime.now(timezone.utc)

    with patch("ocabra.core.model_manager.publish", new=AsyncMock()), \
         patch("ocabra.core.model_manager.set_key", new=AsyncMock()) as set_key_mock:
        state = await add_test_model(model_manager)
        await model_manager.touch_last_request_at("test/model", now)

    assert state.last_request_at == now
    set_key_mock.assert_awaited_once()
    key = set_key_mock.await_args.args[0]
    payload = set_key_mock.await_args.args[1]
    assert key == "model:state:test/model"
    assert payload["last_request_at"] == now.isoformat()


@pytest.mark.asyncio
async def test_hydrate_last_request_at_from_redis_restores_timestamp(model_manager):
    now = datetime.now(timezone.utc).replace(microsecond=0)
    state = await add_test_model(model_manager)
    state.last_request_at = None

    with patch("ocabra.core.model_manager.get_key", new=AsyncMock(return_value={"last_request_at": now.isoformat()})):
        await model_manager._hydrate_last_request_at_from_redis()

    assert state.last_request_at == now


@pytest.mark.asyncio
async def test_load_passes_model_extra_config_to_backend():
    wp = WorkerPool()
    backend = _PortRequiredBackend()
    backend.load = AsyncMock(
        return_value=WorkerInfo(
            backend_type="portreq",
            model_id="test/overrides",
            gpu_indices=[1],
            port=18001,
            pid=1234,
            vram_used_mb=1024,
        )
    )
    wp.register_backend("portreq", backend)
    mm = ModelManager(wp)

    with patch("ocabra.core.model_manager.publish", new=AsyncMock()), \
         patch("ocabra.core.model_manager.set_key", new=AsyncMock()):
        from ocabra.core.model_manager import ModelState

        mm._states["test/overrides"] = ModelState(
            model_id="test/overrides",
            display_name="test/overrides",
            backend_type="portreq",
            load_policy=LoadPolicy.ON_DEMAND,
            extra_config={"vllm": {"max_num_seqs": 4}},
        )
        mm._load_locks["test/overrides"] = asyncio.Lock()

        await mm.load("test/overrides")

    assert backend.load.await_args.kwargs["extra_config"] == {"vllm": {"max_num_seqs": 4}}


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
async def test_update_config_rejects_unsupported_fields(model_manager):
    await add_test_model(model_manager)

    with pytest.raises(ValueError, match="Unsupported model config fields"):
        await model_manager.update_config("test/model", {"status": "loaded"})


@pytest.mark.asyncio
async def test_update_config_applies_whitelisted_fields(model_manager):
    state = await add_test_model(model_manager)

    updated = await model_manager.update_config(
        "test/model",
        {
            "display_name": "renamed",
            "load_policy": "warm",
            "auto_reload": True,
            "preferred_gpu": 1,
            "extra_config": {"foo": "bar"},
        },
    )

    assert updated is state
    assert state.display_name == "renamed"
    assert state.load_policy == LoadPolicy.WARM
    assert state.auto_reload is True
    assert state.preferred_gpu == 1
    assert state.extra_config == {"foo": "bar"}


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

class _BitnetPortBackend(BackendInterface):
    def __init__(self) -> None:
        self.received_port = None

    async def load(self, model_id: str, gpu_indices: list[int], **kwargs) -> WorkerInfo:
        port = kwargs.get("port", 0)
        if not port:
            raise ValueError("port is required")
        self.received_port = int(port)
        return WorkerInfo(
            backend_type="bitnet",
            model_id=model_id,
            gpu_indices=gpu_indices,
            port=self.received_port,
            pid=4321,
            vram_used_mb=0,
        )

    async def unload(self, model_id: str) -> None:
        return None

    async def health_check(self, model_id: str) -> bool:
        return True

    async def get_capabilities(self, model_id: str) -> BackendCapabilities:
        return BackendCapabilities(chat=True, completion=True, streaming=True)

    async def get_vram_estimate_mb(self, model_id: str) -> int:
        return 0

    async def forward_request(self, model_id: str, path: str, body: dict):
        return {}

    async def forward_stream(self, model_id: str, path: str, body: dict):
        if False:
            yield b""


@pytest.mark.asyncio
async def test_bitnet_cpu_only_skips_gpu_assignment_but_keeps_port():
    wp = WorkerPool()
    backend = _BitnetPortBackend()
    wp.register_backend("bitnet", backend)
    scheduler = AsyncMock()
    scheduler.find_gpu_for_model = AsyncMock(return_value=[1])
    mm = ModelManager(wp, gpu_scheduler=scheduler)

    with patch("ocabra.core.model_manager.publish", new=AsyncMock()), \
         patch("ocabra.core.model_manager.set_key", new=AsyncMock()):
        from ocabra.core.model_manager import ModelState

        mm._states["test/bitnet-cpu"] = ModelState(
            model_id="test/bitnet-cpu",
            display_name="test/bitnet-cpu",
            backend_type="bitnet",
            load_policy=LoadPolicy.ON_DEMAND,
            extra_config={"gpu_layers": 0},
        )
        mm._load_locks["test/bitnet-cpu"] = asyncio.Lock()

        state = await mm.load("test/bitnet-cpu")

    assert state.status == ModelStatus.LOADED
    assert state.current_gpu == []
    assert backend.received_port is not None
    scheduler.find_gpu_for_model.assert_not_called()


@pytest.mark.asyncio
async def test_bitnet_gpu_layers_uses_extra_config_for_scheduling():
    wp = WorkerPool()
    backend = _BitnetPortBackend()
    wp.register_backend("bitnet", backend)
    scheduler = AsyncMock()
    scheduler.find_gpu_for_model = AsyncMock(return_value=[1])
    mm = ModelManager(wp, gpu_scheduler=scheduler)

    with patch("ocabra.core.model_manager.publish", new=AsyncMock()), \
         patch("ocabra.core.model_manager.set_key", new=AsyncMock()):
        from ocabra.core.model_manager import ModelState

        mm._states["test/bitnet-gpu"] = ModelState(
            model_id="test/bitnet-gpu",
            display_name="test/bitnet-gpu",
            backend_type="bitnet",
            load_policy=LoadPolicy.ON_DEMAND,
            extra_config={"gpu_layers": 16, "total_layers": 32, "model_vram_mb": 400},
        )
        mm._load_locks["test/bitnet-gpu"] = asyncio.Lock()

        state = await mm.load("test/bitnet-gpu")

    assert state.status == ModelStatus.LOADED
    assert state.current_gpu == [1]
    scheduler.find_gpu_for_model.assert_awaited_once_with(
        200,
        None,
        enforce_vllm_headroom=False,
    )


def test_diarized_variant_helpers_for_whisper_models() -> None:
    from ocabra.core.model_manager import ModelState
    from ocabra.core.model_manager_helpers import (
        build_diarized_extra_config,
        diarized_variant_model_id,
        is_diarized_model_id,
        should_auto_create_diarized_variant,
    )

    base = ModelState(
        model_id="nvidia/parakeet-tdt-0.6b-v3",
        display_name="parakeet",
        backend_type="whisper",
    )
    assert should_auto_create_diarized_variant(base) is True
    assert diarized_variant_model_id(base.model_id) == "nvidia/parakeet-tdt-0.6b-v3::diarize"

    diarized_id_state = ModelState(
        model_id="openai/whisper-medium::diarize",
        display_name="wm",
        backend_type="whisper",
    )
    assert should_auto_create_diarized_variant(diarized_id_state) is False
    assert is_diarized_model_id(diarized_id_state.model_id, {}) is True

    diarized_cfg_state = ModelState(
        model_id="openai/whisper-medium",
        display_name="wm",
        backend_type="whisper",
        extra_config={"diarization_enabled": True},
    )
    assert should_auto_create_diarized_variant(diarized_cfg_state) is False

    non_whisper = ModelState(
        model_id="gpt-oss:20b",
        display_name="gpt",
        backend_type="ollama",
    )
    assert should_auto_create_diarized_variant(non_whisper) is False

    merged = build_diarized_extra_config({"base_model_id": "/path/model.nemo"})
    assert merged["diarization_enabled"] is True
    assert merged["base_model_id"] == "/path/model.nemo"
