import asyncio
import contextlib
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
    with (
        patch("ocabra.core.model_manager.publish", new=AsyncMock()),
        patch("ocabra.core.model_manager.set_key", new=AsyncMock()),
    ):
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
    with (
        patch("ocabra.core.model_manager.publish", new=AsyncMock()),
        patch("ocabra.core.model_manager.set_key", new=AsyncMock()),
    ):
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

    with (
        patch("ocabra.core.model_manager.publish", new=AsyncMock()),
        patch("ocabra.core.model_manager.set_key", new=AsyncMock()),
        patch.object(mm, "_load_configs_from_db", new=AsyncMock()),
    ):
        await add_test_model(mm, load_policy="pin")
        await mm.start()
        await asyncio.sleep(0.1)  # let the task run

        state = await mm.get_state("test/model")
        assert state.status == ModelStatus.LOADED


@pytest.mark.asyncio
async def test_warm_policy_stays_lazy_on_start(worker_pool):
    """Warm models load on demand; only pin preloads during start()."""
    mm = ModelManager(worker_pool)

    with (
        patch("ocabra.core.model_manager.publish", new=AsyncMock()),
        patch("ocabra.core.model_manager.set_key", new=AsyncMock()),
        patch.object(mm, "_load_configs_from_db", new=AsyncMock()),
    ):
        await add_test_model(mm, load_policy="warm")
        await mm.start()
        await asyncio.sleep(0.1)

        state = await mm.get_state("test/model")
        assert state.status == ModelStatus.CONFIGURED


@pytest.mark.asyncio
async def test_idle_eviction(model_manager):
    """on_demand model idle > timeout is evicted."""
    with (
        patch("ocabra.core.model_manager.publish", new=AsyncMock()),
        patch("ocabra.core.model_manager.set_key", new=AsyncMock()),
        patch.object(model_manager._worker_pool._backends["mock"], "unload", new=AsyncMock()),
    ):
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

    with (
        patch("ocabra.core.model_manager.publish", new=AsyncMock()),
        patch("ocabra.core.model_manager.set_key", new=AsyncMock()),
    ):
        await add_test_model(model_manager)
        await asyncio.gather(
            model_manager.load("test/model"),
            model_manager.load("test/model"),
            model_manager.load("test/model"),
        )

    assert load_count == 1


@pytest.mark.asyncio
async def test_gpu_admission_waits_for_prior_backend_load_to_finish():
    """A queued vLLM load must inspect VRAM after an Ollama load completes."""
    from ocabra.core.gpu_manager import GPUState
    from ocabra.core.model_manager import ModelState

    ollama_started = asyncio.Event()
    finish_ollama = asyncio.Event()

    ollama_backend = AsyncMock(spec=BackendInterface)
    ollama_backend.get_vram_estimate_mb = AsyncMock(return_value=12_000)

    async def _load_ollama(*args, **kwargs):
        ollama_started.set()
        await finish_ollama.wait()
        return WorkerInfo(
            backend_type="ollama",
            model_id="gemma4:26b",
            gpu_indices=[],
            port=0,
            pid=100,
            vram_used_mb=12_000,
        )

    ollama_backend.load = AsyncMock(side_effect=_load_ollama)
    ollama_backend.get_capabilities = AsyncMock(return_value=BackendCapabilities(chat=True))

    vllm_backend = AsyncMock(spec=BackendInterface)
    vllm_backend.get_vram_estimate_mb = AsyncMock(return_value=1024)
    vllm_backend.load = AsyncMock(
        return_value=WorkerInfo(
            backend_type="vllm",
            model_id="gemma-4-12B-AWQ",
            gpu_indices=[1],
            port=18001,
            pid=200,
            vram_used_mb=1024,
        )
    )
    vllm_backend.get_capabilities = AsyncMock(return_value=BackendCapabilities(chat=True))

    worker_pool = WorkerPool()
    worker_pool.register_backend("ollama", ollama_backend)
    worker_pool.register_backend("vllm", vllm_backend)

    gpu_state = GPUState(
        index=1,
        name="RTX 3090",
        total_vram_mb=24_576,
        free_vram_mb=24_000,
        used_vram_mb=576,
        utilization_pct=0,
        temperature_c=40,
        power_draw_w=50,
        power_limit_w=350,
        locked_vram_mb=0,
    )
    gpu_manager = AsyncMock()
    gpu_manager.get_free_vram = AsyncMock(return_value=24_000)
    gpu_manager.get_state = AsyncMock(return_value=gpu_state)
    gpu_manager.get_all_states = AsyncMock(return_value=[gpu_state])
    gpu_manager.lock_vram = AsyncMock()

    mm = ModelManager(worker_pool, gpu_manager=gpu_manager)
    scheduler = AsyncMock()

    async def _admit_vllm(*args, **kwargs):
        assert mm._states["ollama/gemma4:26b"].status == ModelStatus.LOADED
        return [1]

    scheduler.find_gpu_for_model = AsyncMock(side_effect=_admit_vllm)
    mm._gpu_scheduler = scheduler

    mm._states["ollama/gemma4:26b"] = ModelState(
        model_id="ollama/gemma4:26b",
        display_name="gemma4:26b",
        backend_type="ollama",
    )
    mm._states["vllm/gemma-4-12B-AWQ"] = ModelState(
        model_id="vllm/gemma-4-12B-AWQ",
        display_name="gemma-4-12B-AWQ",
        backend_type="vllm",
        preferred_gpu=1,
        extra_config={"vllm": {"gpu_memory_utilization": 0.94}},
    )

    with (
        patch("ocabra.core.model_manager.publish", new=AsyncMock()),
        patch("ocabra.core.model_manager.set_key", new=AsyncMock()),
        patch.object(mm, "_record_model_load_stat", new=AsyncMock()),
    ):
        ollama_task = asyncio.create_task(mm.load("ollama/gemma4:26b"))
        await asyncio.wait_for(ollama_started.wait(), timeout=1)

        vllm_task = asyncio.create_task(mm.load("vllm/gemma-4-12B-AWQ"))
        for _ in range(100):
            if mm._load_queue_waiters == 1:
                break
            await asyncio.sleep(0)

        assert mm._load_queue_waiters == 1
        scheduler.find_gpu_for_model.assert_not_called()

        finish_ollama.set()
        ollama_state, vllm_state = await asyncio.gather(ollama_task, vllm_task)

    assert ollama_state.status == ModelStatus.LOADED
    assert vllm_state.status == ModelStatus.LOADED
    scheduler.find_gpu_for_model.assert_awaited_once()
    vllm_backend.load.assert_awaited_once()


@pytest.mark.asyncio
async def test_gpu_admission_times_out_instead_of_hanging_forever(monkeypatch):
    """A load stuck holding the backend gate (e.g. an eviction cascade wedged
    on a backend that never frees its VRAM) must not block every other
    model's load indefinitely. The waiting load should fail with a clear
    TimeoutError instead of sitting in-flight forever with no 429/503."""
    from ocabra.core.model_manager import ModelState

    monkeypatch.setattr("ocabra.core.model_manager.settings.max_drain_timeout_s", 1)

    stuck_started = asyncio.Event()
    never_finishes = asyncio.Event()

    stuck_backend = AsyncMock(spec=BackendInterface)
    stuck_backend.get_vram_estimate_mb = AsyncMock(return_value=12_000)

    async def _load_stuck(*args, **kwargs):
        stuck_started.set()
        await never_finishes.wait()  # pragma: no cover - never set in this test
        raise AssertionError("should never reach here")

    stuck_backend.load = AsyncMock(side_effect=_load_stuck)
    stuck_backend.get_capabilities = AsyncMock(return_value=BackendCapabilities(chat=True))

    other_backend = AsyncMock(spec=BackendInterface)
    other_backend.get_vram_estimate_mb = AsyncMock(return_value=1024)
    other_backend.load = AsyncMock(
        return_value=WorkerInfo(
            backend_type="vllm",
            model_id="other-model",
            gpu_indices=[1],
            port=18001,
            pid=200,
            vram_used_mb=1024,
        )
    )
    other_backend.get_capabilities = AsyncMock(return_value=BackendCapabilities(chat=True))

    worker_pool = WorkerPool()
    worker_pool.register_backend("ollama", stuck_backend)
    worker_pool.register_backend("vllm", other_backend)

    mm = ModelManager(worker_pool)
    mm._states["ollama/stuck-model"] = ModelState(
        model_id="ollama/stuck-model",
        display_name="stuck-model",
        backend_type="ollama",
    )
    mm._states["vllm/other-model"] = ModelState(
        model_id="vllm/other-model",
        display_name="other-model",
        backend_type="vllm",
        preferred_gpu=1,
    )

    with (
        patch("ocabra.core.model_manager.publish", new=AsyncMock()),
        patch("ocabra.core.model_manager.publish_system_alert", new=AsyncMock()),
        patch("ocabra.core.model_manager.set_key", new=AsyncMock()),
        patch.object(mm, "_record_model_load_stat", new=AsyncMock()),
    ):
        stuck_task = asyncio.create_task(mm.load("ollama/stuck-model"))
        await asyncio.wait_for(stuck_started.wait(), timeout=1)

        with pytest.raises(TimeoutError):
            await asyncio.wait_for(mm.load("vllm/other-model"), timeout=5)

        assert mm._states["vllm/other-model"].status == ModelStatus.ERROR
        assert mm._load_queue_waiters == 0

        never_finishes.set()
        stuck_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await stuck_task


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
async def test_try_begin_request_scales_cap_down_for_single_slot_worker(
    model_manager, monkeypatch
):
    """A model with no declared ``_runtime.parallelism`` (the common case for
    Ollama models) defaults to 1 slot. The flat 32-request ceiling must not
    apply as-is — a burst against a single-slot worker should get rejected
    with 429 well before 32 requests pile up behind the one being served."""
    monkeypatch.setattr("ocabra.core.model_manager.settings.max_queued_per_slot", 4)
    model_id = "test/model"
    await add_test_model(model_manager, model_id=model_id)

    accepted = [
        model_manager.try_begin_request(model_id, max_inflight=32) for _ in range(6)
    ]
    assert accepted.count(None) == 2  # 4 admitted (1 slot * cap 4), 2 rejected
    assert accepted[:4] != [None, None, None, None]


@pytest.mark.asyncio
async def test_try_begin_request_scales_cap_up_for_multi_slot_worker(
    model_manager, monkeypatch
):
    """A model that declares real concurrency via ``_runtime.parallelism``
    (e.g. a vLLM worker with max_num_seqs > 1) should get a proportionally
    larger cap instead of being clamped to the single-slot default."""
    monkeypatch.setattr("ocabra.core.model_manager.settings.max_queued_per_slot", 4)
    model_id = "test/model"
    state = await add_test_model(model_manager, model_id=model_id)
    state.extra_config = {"_runtime": {"parallelism": 8}}

    results = [
        model_manager.try_begin_request(model_id, max_inflight=32) for _ in range(32)
    ]
    assert results.count(None) == 0  # 8 slots * cap 4 = 32, none rejected

    assert model_manager.try_begin_request(model_id, max_inflight=32) is None


def test_effective_inflight_cap_disabled_when_max_inflight_is_zero(model_manager):
    assert model_manager.effective_inflight_cap("test/model", 0) == 0


def test_maintenance_toggle_covers_derived_worker_keys(model_manager):
    canonical = "ollama/qwen3.8:27b"
    derived = f"{canonical}::abc123"

    assert not model_manager.is_in_maintenance(canonical)
    assert not model_manager.is_in_maintenance(derived)

    model_manager.set_maintenance(canonical, True)
    assert model_manager.is_in_maintenance(canonical)
    assert model_manager.is_in_maintenance(derived)  # same canonical id

    model_manager.set_maintenance(canonical, False)
    assert not model_manager.is_in_maintenance(canonical)
    assert not model_manager.is_in_maintenance(derived)


def test_maintenance_toggle_via_derived_key_resolves_to_canonical(model_manager):
    canonical = "ollama/qwen3.8:27b"
    derived = f"{canonical}::abc123"

    model_manager.set_maintenance(derived, True)
    assert model_manager.is_in_maintenance(canonical)
    assert model_manager.is_in_maintenance(derived)


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

    with (
        patch("ocabra.core.model_manager.publish", new=AsyncMock()),
        patch("ocabra.core.model_manager.set_key", new=AsyncMock()) as set_key_mock,
    ):
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

    with patch(
        "ocabra.core.model_manager.get_key",
        new=AsyncMock(return_value={"last_request_at": now.isoformat()}),
    ):
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

    with (
        patch("ocabra.core.model_manager.publish", new=AsyncMock()),
        patch("ocabra.core.model_manager.set_key", new=AsyncMock()),
    ):
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

    with (
        patch("ocabra.core.model_manager.publish", new=AsyncMock()),
        patch("ocabra.core.model_manager.set_key", new=AsyncMock()),
        patch.object(mm, "unload", new=AsyncMock()) as unload_mock,
    ):
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

    async def get_vram_estimate_mb(self, model_id: str, **kwargs) -> int:
        return 1024

    async def forward_request(self, model_id: str, path: str, body: dict):
        return {}

    async def forward_stream(self, model_id: str, path: str, body: dict):
        if False:
            yield b""


@pytest.mark.asyncio
async def test_vllm_load_uses_model_gpu_memory_utilization_for_headroom():
    from ocabra.core.gpu_manager import GPUState
    from ocabra.core.model_manager import ModelState

    wp = WorkerPool()
    backend = _PortRequiredBackend()
    wp.register_backend("vllm", backend)

    scheduler = AsyncMock()
    scheduler.find_gpu_for_model = AsyncMock(return_value=[0])

    gpu_manager = AsyncMock()
    # Generic usable VRAM is below the vLLM fraction, while physical free VRAM
    # still satisfies it. The generic buffer must not be counted twice.
    gpu_manager.get_free_vram = AsyncMock(return_value=2500)
    gpu_manager.get_state = AsyncMock(
        return_value=GPUState(
            index=0,
            name="RTX 3060",
            total_vram_mb=12000,
            free_vram_mb=5000,
            used_vram_mb=7000,
            utilization_pct=0,
            temperature_c=40,
            power_draw_w=50,
            power_limit_w=170,
            locked_vram_mb=0,
        )
    )
    gpu_manager.lock_vram = AsyncMock()

    mm = ModelManager(wp, gpu_manager=gpu_manager, gpu_scheduler=scheduler)

    with (
        patch("ocabra.core.model_manager.publish", new=AsyncMock()),
        patch("ocabra.core.model_manager.set_key", new=AsyncMock()),
        patch.object(mm, "_record_model_load_stat", new=AsyncMock()),
        patch("ocabra.core.model_manager.settings.vllm_gpu_memory_utilization", 0.9),
    ):
        mm._states["vllm/BAAI/bge-reranker-v2-m3"] = ModelState(
            model_id="vllm/BAAI/bge-reranker-v2-m3",
            display_name="bge-reranker-v2-m3",
            backend_type="vllm",
            load_policy=LoadPolicy.ON_DEMAND,
            extra_config={"vllm": {"gpu_memory_utilization": 0.25}},
        )
        mm._load_locks["vllm/BAAI/bge-reranker-v2-m3"] = asyncio.Lock()

        state = await mm.load("vllm/BAAI/bge-reranker-v2-m3")

    assert state.status == ModelStatus.LOADED
    scheduler.find_gpu_for_model.assert_awaited_once_with(
        1024,
        None,
        enforce_vllm_headroom=True,
        vllm_gpu_memory_utilization=0.25,
    )
    gpu_manager.lock_vram.assert_awaited_once_with(
        0,
        1024,
        "vllm/BAAI/bge-reranker-v2-m3",
    )


@pytest.mark.asyncio
async def test_vllm_load_with_fixed_kv_uses_estimated_vram_instead_of_fraction():
    from ocabra.core.gpu_manager import GPUState
    from ocabra.core.model_manager import ModelState

    wp = WorkerPool()
    backend = _PortRequiredBackend()
    wp.register_backend("vllm", backend)

    scheduler = AsyncMock()
    scheduler.find_gpu_for_model = AsyncMock(return_value=[0])

    gpu_manager = AsyncMock()
    gpu_manager.get_free_vram = AsyncMock(return_value=1500)
    gpu_manager.get_state = AsyncMock(
        return_value=GPUState(
            index=0,
            name="RTX 3060",
            total_vram_mb=12000,
            free_vram_mb=1500,
            used_vram_mb=10500,
            utilization_pct=0,
            temperature_c=40,
            power_draw_w=50,
            power_limit_w=170,
            locked_vram_mb=0,
        )
    )
    gpu_manager.lock_vram = AsyncMock()

    mm = ModelManager(wp, gpu_manager=gpu_manager, gpu_scheduler=scheduler)
    model_id = "vllm/fixed-kv-model"

    with (
        patch("ocabra.core.model_manager.publish", new=AsyncMock()),
        patch("ocabra.core.model_manager.set_key", new=AsyncMock()),
        patch.object(mm, "_record_model_load_stat", new=AsyncMock()),
    ):
        mm._states[model_id] = ModelState(
            model_id=model_id,
            display_name="fixed-kv-model",
            backend_type="vllm",
            load_policy=LoadPolicy.ON_DEMAND,
            extra_config={
                "vllm": {
                    "gpu_memory_utilization": 0.98,
                    "extra_args": ["--kv-cache-memory-bytes", "536870912"],
                }
            },
        )
        mm._load_locks[model_id] = asyncio.Lock()

        state = await mm.load(model_id)

    assert state.status == ModelStatus.LOADED
    scheduler.find_gpu_for_model.assert_awaited_once_with(
        1024,
        None,
        enforce_vllm_headroom=False,
        vllm_gpu_memory_utilization=0.98,
    )


@pytest.mark.asyncio
async def test_use_case_resolution_sums_capacity_across_forced_multi_gpu():
    """Forced multi-GPU placement (llama.cpp/bitnet --split-mode layer,
    Ollama's shared pool) must size the use-case plan against the SUMMED
    capacity of every selected GPU, not just gpu_indices[0]. Using only the
    first GPU (often the smallest) used to compute a near-zero budget there,
    silently leaving the stale/base ctx_size in place instead of the
    use-case-resolved one — see gpu-bench/README.md 2026-10-26."""
    from ocabra.core.gpu_manager import GPUState
    from ocabra.core.model_manager import ModelState

    wp = WorkerPool()
    backend = _PortRequiredBackend()
    wp.register_backend("llama_cpp", backend)

    gpu_manager = AsyncMock()
    gpu_manager.get_all_states = AsyncMock(
        return_value=[
            GPUState(
                index=0, name="RTX 3060", total_vram_mb=12_288, free_vram_mb=12_000,
                used_vram_mb=288, utilization_pct=0, temperature_c=40,
                power_draw_w=50, power_limit_w=170, locked_vram_mb=0,
            ),
            GPUState(
                index=1, name="RTX 3090", total_vram_mb=24_576, free_vram_mb=24_000,
                used_vram_mb=576, utilization_pct=0, temperature_c=40,
                power_draw_w=50, power_limit_w=350, locked_vram_mb=0,
            ),
        ]
    )
    gpu_manager.lock_vram = AsyncMock()

    mm = ModelManager(wp, gpu_manager=gpu_manager)
    model_id = "llama_cpp/multi-gpu-model"

    captured_gpu_total = {}

    async def _fake_resolve_use_case(state, gpu_total_mb):
        captured_gpu_total["value"] = gpu_total_mb
        return {"applied": True, "effective_context": 131_072, "slots": 1}

    with (
        patch("ocabra.core.model_manager.publish", new=AsyncMock()),
        patch("ocabra.core.model_manager.set_key", new=AsyncMock()),
        patch.object(mm, "_record_model_load_stat", new=AsyncMock()),
        patch.object(mm, "_admit_gpu_load", new=AsyncMock(return_value=[0, 1])),
        patch(
            "ocabra.core.vram_capacity.resolve_use_case",
            new=_fake_resolve_use_case,
        ),
        patch.object(backend, "load", new=AsyncMock(wraps=backend.load)),
    ):
        mm._states[model_id] = ModelState(
            model_id=model_id,
            display_name="multi-gpu-model",
            backend_type="llama_cpp",
            load_policy=LoadPolicy.ON_DEMAND,
            extra_config={
                "llama_cpp": {"ctx_size": 65_536, "parallel_slots": 1, "gpu_layers": 99},
                "use_case": {"context": 131_072, "slots": 1, "kv_dtype": "q8"},
                "_runtime": {"gpu_indices": [0, 1]},
            },
        )
        mm._load_locks[model_id] = asyncio.Lock()

        state = await mm.load(model_id)
        load_kwargs = backend.load.await_args.kwargs

    assert state.status == ModelStatus.LOADED
    # 12_288 + 24_576, not just GPU0's 12_288.
    assert captured_gpu_total["value"] == 36_864
    assert load_kwargs["extra_config"]["llama_cpp"]["ctx_size"] == 131_072


class _FakeOllamaBackend(BackendInterface):
    """Minimal ollama-shaped backend exposing single_gpu_pool_candidates,
    for testing the preferred-GPU-busy fallback without the real
    OllamaBackend's pool/HTTP machinery."""

    def __init__(
        self, fallback_gpus: list[int], resident: list[dict] | None = None
    ) -> None:
        self._fallback_gpus = fallback_gpus
        self.load_calls: list[list[int]] = []
        self._resident = resident or []

    async def load(self, model_id: str, gpu_indices: list[int], **kwargs) -> WorkerInfo:
        self.load_calls.append(list(gpu_indices))
        return WorkerInfo(
            backend_type="ollama", model_id=model_id, gpu_indices=gpu_indices,
            port=0, pid=1, vram_used_mb=1024,
        )

    def single_gpu_pool_candidates(self, *, exclude_gpu, required_parallelism=1):
        return [g for g in self._fallback_gpus if g != exclude_gpu]

    async def discover_resident_models(self) -> list[dict]:
        return self._resident

    async def unload(self, model_id: str) -> None:
        return None

    async def health_check(self, model_id: str) -> bool:
        return True

    async def get_capabilities(self, model_id: str) -> BackendCapabilities:
        return BackendCapabilities(chat=True, completion=True, streaming=True)

    async def get_vram_estimate_mb(self, model_id: str, **kwargs) -> int:
        return 5_000

    async def forward_request(self, model_id: str, path: str, body: dict):
        raise NotImplementedError

    async def forward_stream(self, model_id: str, path: str, body: dict):
        raise NotImplementedError
        yield b""  # pragma: no cover - makes this an async generator


@pytest.mark.asyncio
async def test_ollama_falls_back_to_alternate_gpu_when_preferred_is_busy():
    """gemma4:12b incident (2026-10-26): a model statically pinned to one
    GPU used to have zero fallback — a busy preferred GPU meant 5-7 minutes
    through the pressure-eviction cascade and a hard failure, even when a
    completely idle single-GPU Ollama pool had plenty of room. The preferred
    GPU's real free VRAM must be checked first, and only then should it fall
    back to an alternate single-GPU pool."""
    from ocabra.core.gpu_manager import GPUState
    from ocabra.core.model_manager import ModelState

    wp = WorkerPool()
    backend = _FakeOllamaBackend(fallback_gpus=[0])
    wp.register_backend("ollama", backend)

    gpu_manager = AsyncMock()
    gpu_manager.get_all_states = AsyncMock(
        return_value=[
            GPUState(
                index=0, name="RTX 3060", total_vram_mb=12_288, free_vram_mb=12_000,
                used_vram_mb=288, utilization_pct=0, temperature_c=40,
                power_draw_w=50, power_limit_w=170, locked_vram_mb=0,
            ),
            GPUState(
                index=1, name="RTX 3090", total_vram_mb=24_576, free_vram_mb=500,
                used_vram_mb=24_076, utilization_pct=0, temperature_c=40,
                power_draw_w=50, power_limit_w=350, locked_vram_mb=0,
            ),
        ]
    )
    # GPU1 (preferred) is nearly full — not enough for the 5_000 MB need.
    # GPU0 (fallback) has plenty.
    gpu_manager.get_free_vram = AsyncMock(side_effect=lambda idx: {0: 20_000, 1: 500}[idx])
    gpu_manager.lock_vram = AsyncMock()

    mm = ModelManager(wp, gpu_manager=gpu_manager)
    model_id = "ollama/busy-preferred-model"

    with (
        patch("ocabra.core.model_manager.publish", new=AsyncMock()),
        patch("ocabra.core.model_manager.set_key", new=AsyncMock()),
        patch.object(mm, "_record_model_load_stat", new=AsyncMock()),
        patch.object(mm, "_admit_gpu_load", new=AsyncMock(return_value=[1])),
    ):
        mm._states[model_id] = ModelState(
            model_id=model_id,
            display_name="busy-preferred-model",
            backend_type="ollama",
            load_policy=LoadPolicy.ON_DEMAND,
            preferred_gpu=1,
        )
        mm._load_locks[model_id] = asyncio.Lock()

        state = await mm.load(model_id)

    assert state.status == ModelStatus.LOADED
    assert backend.load_calls[-1] == [0]


@pytest.mark.asyncio
async def test_ollama_keeps_preferred_gpu_when_it_has_room():
    from ocabra.core.gpu_manager import GPUState
    from ocabra.core.model_manager import ModelState

    wp = WorkerPool()
    backend = _FakeOllamaBackend(fallback_gpus=[0])
    wp.register_backend("ollama", backend)

    gpu_manager = AsyncMock()
    gpu_manager.get_all_states = AsyncMock(
        return_value=[
            GPUState(
                index=1, name="RTX 3090", total_vram_mb=24_576, free_vram_mb=20_000,
                used_vram_mb=4_576, utilization_pct=0, temperature_c=40,
                power_draw_w=50, power_limit_w=350, locked_vram_mb=0,
            ),
        ]
    )
    gpu_manager.get_free_vram = AsyncMock(return_value=20_000)
    gpu_manager.lock_vram = AsyncMock()

    mm = ModelManager(wp, gpu_manager=gpu_manager)
    model_id = "ollama/happy-model"

    with (
        patch("ocabra.core.model_manager.publish", new=AsyncMock()),
        patch("ocabra.core.model_manager.set_key", new=AsyncMock()),
        patch.object(mm, "_record_model_load_stat", new=AsyncMock()),
        patch.object(mm, "_admit_gpu_load", new=AsyncMock(return_value=[1])),
    ):
        mm._states[model_id] = ModelState(
            model_id=model_id,
            display_name="happy-model",
            backend_type="ollama",
            load_policy=LoadPolicy.ON_DEMAND,
            preferred_gpu=1,
        )
        mm._load_locks[model_id] = asyncio.Lock()

        state = await mm.load(model_id)

    assert state.status == ModelStatus.LOADED
    assert backend.load_calls[-1] == [1]


@pytest.mark.asyncio
async def test_load_assigns_port_for_backend_that_requires_it():
    wp = WorkerPool()
    backend = _PortRequiredBackend()
    wp.register_backend("portreq", backend)
    mm = ModelManager(wp)

    with (
        patch("ocabra.core.model_manager.publish", new=AsyncMock()),
        patch("ocabra.core.model_manager.set_key", new=AsyncMock()),
    ):
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


@pytest.mark.asyncio
async def test_reconcile_ollama_state_adopts_untracked_resident_model():
    """gemma4:12b incident (2026-10-25): after an API restart, a model that
    was still resident in a persistent Ollama container had zero record in
    ModelManager's in-memory state — oCabra reported it unloaded/0 VRAM
    while NVML correctly (but silently) showed the real GPU as unavailable.
    Reconciliation must adopt it as LOADED using the real pool GPUs/VRAM
    reported by the backend."""
    from ocabra.core.model_manager import ModelState

    wp = WorkerPool()
    backend = _FakeOllamaBackend(
        fallback_gpus=[0],
        resident=[
            {
                "name": "gemma4:12b",
                "vram_mb": 10_300,
                "pool_url": "http://ocabra-ollama-3060-1:11434",
                "pool_gpus": [0],
            }
        ],
    )
    wp.register_backend("ollama", backend)
    gpu_manager = AsyncMock()
    gpu_manager.lock_vram = AsyncMock()

    mm = ModelManager(wp, gpu_manager=gpu_manager)
    model_id = "ollama/gemma4-12b"
    mm._states[model_id] = ModelState(
        model_id=model_id,
        display_name="gemma4:12b",
        backend_type="ollama",
        backend_model_id="gemma4:12b",
        load_policy=LoadPolicy.ON_DEMAND,
        preferred_gpu=0,
    )

    await mm._reconcile_ollama_state()

    state = mm._states[model_id]
    assert state.status == ModelStatus.LOADED
    assert state.current_gpu == [0]
    assert state.vram_used_mb == 10_300
    assert state.loaded_at is not None
    assert wp.get_worker(model_id) is not None
    gpu_manager.lock_vram.assert_awaited_once_with(0, 10_300, model_id)


@pytest.mark.asyncio
async def test_reconcile_ollama_state_leaves_already_loaded_model_alone():
    """A model oCabra already correctly tracks as LOADED must not be
    re-adopted/re-locked — reconciliation only fills in gaps."""
    from ocabra.core.model_manager import ModelState

    wp = WorkerPool()
    backend = _FakeOllamaBackend(
        fallback_gpus=[0],
        resident=[
            {
                "name": "qwen3.8:27b",
                "vram_mb": 24_000,
                "pool_url": "http://ocabra-ollama-1:11434",
                "pool_gpus": [1],
            }
        ],
    )
    wp.register_backend("ollama", backend)
    gpu_manager = AsyncMock()
    gpu_manager.lock_vram = AsyncMock()

    mm = ModelManager(wp, gpu_manager=gpu_manager)
    model_id = "ollama/qwen3.8-27b"
    mm._states[model_id] = ModelState(
        model_id=model_id,
        display_name="qwen3.8:27b",
        backend_type="ollama",
        backend_model_id="qwen3.8:27b",
        load_policy=LoadPolicy.ON_DEMAND,
        preferred_gpu=1,
        status=ModelStatus.LOADED,
        current_gpu=[1],
        vram_used_mb=24_000,
    )

    await mm._reconcile_ollama_state()

    gpu_manager.lock_vram.assert_not_awaited()
    assert wp.get_worker(model_id) is None


@pytest.mark.asyncio
async def test_reconcile_ollama_state_warns_on_unknown_resident_model(caplog):
    """A resident model with no matching model_config must not crash
    startup — it should just be flagged for an admin to investigate."""
    wp = WorkerPool()
    backend = _FakeOllamaBackend(
        fallback_gpus=[0],
        resident=[
            {
                "name": "some-stray-model:latest",
                "vram_mb": 2_000,
                "pool_url": "http://ocabra-ollama-1:11434",
                "pool_gpus": [1],
            }
        ],
    )
    wp.register_backend("ollama", backend)
    gpu_manager = AsyncMock()
    gpu_manager.lock_vram = AsyncMock()

    mm = ModelManager(wp, gpu_manager=gpu_manager)

    await mm._reconcile_ollama_state()

    gpu_manager.lock_vram.assert_not_awaited()
    assert wp.get_worker("some-stray-model:latest") is None


@pytest.mark.asyncio
async def test_reconcile_ollama_state_is_a_noop_without_an_ollama_backend():
    """Most test/dev setups (and any deployment with Ollama disabled)
    register no 'ollama' backend at all — reconciliation must no-op, not
    raise."""
    wp = WorkerPool()
    mm = ModelManager(wp)

    await mm._reconcile_ollama_state()


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

    async def get_vram_estimate_mb(self, model_id: str, **kwargs) -> int:
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

    with (
        patch("ocabra.core.model_manager.publish", new=AsyncMock()),
        patch("ocabra.core.model_manager.set_key", new=AsyncMock()),
    ):
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
async def test_add_model_concurrent_calls_for_same_derived_key_share_one_state():
    """qwen3.8:27b incident (2026-10-26): two concurrent requests creating
    the same derived worker key (e.g. llama_cpp/qwen3.8-27b::<hash>) both
    passed add_model's early 'already registered' guard — it only runs
    before the GGUF fingerprint await, not after — so both went on to
    insert a *new* ModelState + Lock, with whichever finished last
    clobbering the other's objects. A load already running against the
    first (now orphaned) object kept mutating it — including setting it to
    LOADED once the real worker came up healthy — invisibly to
    get_state()/_load_model, which only ever saw the second object. Net
    effect in production: the real worker loaded and reserved its own VRAM
    correctly, but ModelManager's visible state for it stayed stuck at
    error/unloaded, so the model started rejecting its own future requests
    with insufficient_vram forever, even though it was healthy and serving."""
    wp = make_worker_pool_with_mock()
    mm = ModelManager(wp)
    model_id = "llama_cpp/race-model::abc123"

    release = asyncio.Event()

    async def _slow_fingerprint(*args, **kwargs):
        await release.wait()
        return (None, None, None)

    with patch.object(
        mm, "_extract_llama_cpp_fingerprint", new=AsyncMock(side_effect=_slow_fingerprint)
    ):
        task_a = asyncio.create_task(
            mm.add_model(model_id=model_id, backend_type="llama_cpp", persist=False)
        )
        task_b = asyncio.create_task(
            mm.add_model(model_id=model_id, backend_type="llama_cpp", persist=False)
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        release.set()
        state_a, state_b = await asyncio.gather(task_a, task_b)

    assert state_a is state_b
    assert mm._states[model_id] is state_a
    assert mm._load_locks[model_id] is not None


@pytest.mark.asyncio
async def test_bitnet_gpu_layers_uses_extra_config_for_scheduling():
    wp = WorkerPool()
    backend = _BitnetPortBackend()
    wp.register_backend("bitnet", backend)
    scheduler = AsyncMock()
    scheduler.find_gpu_for_model = AsyncMock(return_value=[1])
    mm = ModelManager(wp, gpu_scheduler=scheduler)

    with (
        patch("ocabra.core.model_manager.publish", new=AsyncMock()),
        patch("ocabra.core.model_manager.set_key", new=AsyncMock()),
    ):
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
        vllm_gpu_memory_utilization=None,
    )


def test_apply_capability_fallbacks_prefers_tensorrt_extra_config_context_length():
    from ocabra.core.model_manager import ModelState

    state = ModelState(
        model_id="tensorrt_llm/Qwen3-8B-fp16",
        display_name="Qwen3-8B-fp16",
        backend_type="tensorrt_llm",
        backend_model_id="Qwen3-8B-fp16",
        load_policy=LoadPolicy.ON_DEMAND,
        extra_config={"context_length": 8192},
    )

    caps = BackendCapabilities(chat=True, completion=True, streaming=True, context_length=512)
    merged = ModelManager._apply_capability_fallbacks(state, caps)

    assert merged.context_length == 8192


def test_build_diarized_extra_config() -> None:
    from ocabra.core.model_manager_helpers import build_diarized_extra_config

    merged = build_diarized_extra_config({"base_model_id": "/path/model.nemo"})
    assert merged["diarization_enabled"] is True
    assert merged["base_model_id"] == "/path/model.nemo"
    assert merged["whisper"]["diarizationEnabled"] is True
