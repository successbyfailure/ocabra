import asyncio
import contextlib
import os
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import sqlalchemy as sa

from ocabra.backends._mock import MockBackend
from ocabra.backends.base import BackendCapabilities, BackendInterface, WorkerInfo
from ocabra.core.model_manager import LoadPolicy, ModelManager, ModelStatus
from ocabra.core.worker_pool import WorkerPool
from ocabra.database import AsyncSessionLocal
from ocabra.db.stats import ModelLoadStat
from tests.conftest import requires_db


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
async def test_manual_unload_waits_for_fresh_vram_sample(worker_pool, monkeypatch):
    """An immediate reload must not see hw-monitor's pre-unload VRAM sample."""
    from ocabra.core.gpu_manager import GPUState
    from ocabra.core.model_manager import ModelState

    occupied = GPUState(
        index=1,
        name="RTX 3090",
        total_vram_mb=24_576,
        free_vram_mb=3_000,
        used_vram_mb=21_576,
        utilization_pct=0,
        temperature_c=40,
        power_draw_w=50,
        power_limit_w=350,
        locked_vram_mb=22_000,
    )
    released = GPUState(
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
    gpu_manager.get_all_states = AsyncMock(
        side_effect=[[occupied], [occupied], [released]]
    )
    gpu_manager.unlock_vram = AsyncMock()

    manager = ModelManager(worker_pool, gpu_manager=gpu_manager)
    manager._states["mock/gemma"] = ModelState(
        model_id="mock/gemma",
        backend_model_id="gemma",
        display_name="Gemma",
        backend_type="mock",
        status=ModelStatus.LOADED,
        current_gpu=[1],
        vram_used_mb=22_000,
    )
    manager._load_locks["mock/gemma"] = asyncio.Lock()
    sleep_mock = AsyncMock()
    monkeypatch.setattr("ocabra.core.model_manager.asyncio.sleep", sleep_mock)

    with (
        patch("ocabra.core.model_manager.publish", new=AsyncMock()),
        patch("ocabra.core.model_manager.set_key", new=AsyncMock()),
    ):
        await manager.unload("mock/gemma", reason="manual")

    assert manager._states["mock/gemma"].status == ModelStatus.UNLOADED
    assert gpu_manager.get_all_states.await_count == 3
    sleep_mock.assert_any_await(0.75)


@pytest.mark.asyncio
@requires_db
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
async def test_start_does_not_reload_reconciled_pin(worker_pool):
    """A resident Ollama PIN stays evictable while the API starts."""
    mm = ModelManager(worker_pool)

    with (
        patch.object(mm, "_load_configs_from_db", new=AsyncMock()),
        patch.object(mm, "_hydrate_last_request_at_from_redis", new=AsyncMock()),
        patch.object(mm, "_reconcile_ollama_state", new=AsyncMock()),
        patch.object(mm, "_load_model", new=AsyncMock()) as load_model,
    ):
        state = await add_test_model(mm, load_policy="pin")
        state.status = ModelStatus.LOADED
        await mm.start()
        await asyncio.sleep(0)

    load_model.assert_not_awaited()


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
        state.last_request_at = datetime.now(UTC) - timedelta(seconds=999)

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


class TestLiveRequestProgress:
    """2026-10-29 — model/GPU card now shows per-request progress (input
    tokens estimate, output tokens streamed so far, last-activity age) so an
    operator can tell a genuinely slow request from a hung one instead of
    just watching elapsed time climb."""

    @pytest.mark.asyncio
    async def test_input_tokens_estimate_is_stored_at_admission(self, model_manager):
        model_id = "test/model"
        await add_test_model(model_manager, model_id=model_id)

        request_id = model_manager.try_begin_request(
            model_id, max_inflight=32, input_tokens_estimate=1234
        )

        [active] = model_manager.active_requests_snapshot()
        assert active.request_id == request_id
        assert active.input_tokens_estimate == 1234
        assert active.output_tokens_estimate is None
        assert active.last_activity_at > 0

    @pytest.mark.asyncio
    async def test_update_request_progress_sets_output_tokens_and_bumps_activity(
        self, model_manager
    ):
        model_id = "test/model"
        await add_test_model(model_manager, model_id=model_id)
        request_id = model_manager.try_begin_request(model_id, max_inflight=32)
        [before] = model_manager.active_requests_snapshot()
        first_activity = before.last_activity_at

        await asyncio.sleep(0.01)
        model_manager.update_request_progress(request_id, output_tokens_estimate=42)

        [after] = model_manager.active_requests_snapshot()
        assert after.output_tokens_estimate == 42
        assert after.last_activity_at > first_activity

    @pytest.mark.asyncio
    async def test_update_request_progress_is_a_noop_for_unknown_request_id(
        self, model_manager
    ):
        """The request may have already finished (cancelled/evicted) between
        the tee's throttle check and the update call — must not raise."""
        model_manager.update_request_progress("does-not-exist", output_tokens_estimate=10)


class TestGetLoadPhase:
    """2026-10-29 — distinguishes cold-load / eviction-wait / already-loaded
    so the 'esperando salida' indicator can say *why* instead of an
    undifferentiated wait that looks identical whether the worker is
    starting, blocked on a busy neighbour, or just processing a huge
    prompt. Added after repeatedly having to grep docker logs by hand to
    tell these apart during gemma4<->qwen3.8 contention investigations."""

    @pytest.mark.asyncio
    async def test_loaded_model_is_ready(self, model_manager):
        model_id = "test/model"
        await add_test_model(model_manager, model_id=model_id)
        model_manager._states[model_id].status = ModelStatus.LOADED

        assert model_manager.get_load_phase(model_id) == ("ready", None)

    @pytest.mark.asyncio
    async def test_loading_model_reports_loading_phase(self, model_manager):
        model_id = "test/model"
        await add_test_model(model_manager, model_id=model_id)
        model_manager._states[model_id].status = ModelStatus.LOADING

        assert model_manager.get_load_phase(model_id) == ("loading", None)

    @pytest.mark.asyncio
    async def test_eviction_wait_reports_the_model_being_evicted(self, model_manager):
        model_id = "test/model"
        await add_test_model(model_manager, model_id=model_id)
        model_manager._states[model_id].status = ModelStatus.LOADING
        model_manager._eviction_wait_for[model_id] = "other/busy-model"

        phase, waiting_on = model_manager.get_load_phase(model_id)
        assert phase == "waiting_for_gpu"
        assert waiting_on == "other/busy-model"

    @pytest.mark.asyncio
    async def test_eviction_wait_takes_priority_over_plain_loading(self, model_manager):
        """A model mid-drain-wait is technically still LOADING too — the
        more specific 'waiting_for_gpu' phase must win so the UI doesn't
        under-report what's actually blocking it."""
        model_id = "test/model"
        await add_test_model(model_manager, model_id=model_id)
        model_manager._states[model_id].status = ModelStatus.LOADING
        model_manager._eviction_wait_for[model_id] = "other/busy-model"

        phase, _ = model_manager.get_load_phase(model_id)
        assert phase == "waiting_for_gpu"

    def test_unknown_model_id_is_ready_not_an_error(self, model_manager):
        """No state tracked at all (e.g. a stale/derived key that already
        unloaded) must not raise — defaults to 'ready' so the caller falls
        through to its own prompt-eval/generating distinction."""
        assert model_manager.get_load_phase("nonexistent/model") == ("ready", None)


class TestMonitorLoadVramProgress:
    """2026-10-29 — concurrent watchdog for `backend.load()`, added after a
    live incident where a cold load sat at 256MB VRAM / 0% GPU util for
    minutes under host memory pressure with nothing in the UI distinguishing
    it from a normal slow load. `asyncio.sleep` is monkeypatched to resolve
    instantly so these tests don't take real wall-clock seconds; the monitor
    is always cancelled externally (mirrors how `_load_model` uses it)."""

    @staticmethod
    def _patch_fast_sleep(monkeypatch, model_manager_module):
        """Replaces asyncio.sleep(5) with a real asyncio.sleep(0) so the
        monitor's loop runs many iterations within the test's wall-clock
        budget instead of the real 5s per sample — while still genuinely
        yielding to the event loop each iteration (an AsyncMock that
        resolves without a real await point would spin the loop into a
        busy-loop that starves the test's own timeout)."""
        real_sleep = asyncio.sleep

        async def _fast_sleep(_seconds):
            await real_sleep(0)

        monkeypatch.setattr(model_manager_module.asyncio, "sleep", _fast_sleep)

    async def _run_briefly(self, model_manager, model_id, gpu_indices):
        task = asyncio.create_task(
            model_manager._monitor_load_vram_progress(model_id, gpu_indices)
        )
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(task, timeout=0.2)
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    @pytest.mark.asyncio
    async def test_warns_when_vram_and_utilization_stay_flat(
        self, model_manager, monkeypatch
    ):
        import ocabra.core.model_manager as model_manager_module
        from ocabra.core.gpu_manager import GPUState

        self._patch_fast_sleep(monkeypatch, model_manager_module)
        flat_state = GPUState(
            index=0,
            name="RTX 3060",
            total_vram_mb=12_000,
            free_vram_mb=11_700,
            used_vram_mb=256,
            utilization_pct=0,
            temperature_c=40,
            power_draw_w=50,
            power_limit_w=170,
            locked_vram_mb=0,
        )
        gpu_manager = AsyncMock()
        gpu_manager.get_state = AsyncMock(return_value=flat_state)
        model_manager._gpu_manager = gpu_manager
        log_mock = MagicMock()
        monkeypatch.setattr(model_manager_module, "logger", log_mock)

        await self._run_briefly(model_manager, "test/stalled-model", [0])

        warnings = [
            c for c in log_mock.warning.call_args_list if c.args[0] == "model_load_appears_stalled"
        ]
        assert warnings, "expected a model_load_appears_stalled warning"
        assert warnings[0].kwargs["model_id"] == "test/stalled-model"

    @pytest.mark.asyncio
    async def test_does_not_warn_when_vram_keeps_growing(
        self, model_manager, monkeypatch
    ):
        import ocabra.core.model_manager as model_manager_module
        from ocabra.core.gpu_manager import GPUState

        self._patch_fast_sleep(monkeypatch, model_manager_module)
        counter = {"used_mb": 0}

        async def _growing_state(_idx):
            counter["used_mb"] += 500
            return GPUState(
                index=0,
                name="RTX 3090",
                total_vram_mb=24_576,
                free_vram_mb=24_000 - counter["used_mb"],
                used_vram_mb=counter["used_mb"],
                utilization_pct=0,
                temperature_c=40,
                power_draw_w=50,
                power_limit_w=350,
                locked_vram_mb=0,
            )

        gpu_manager = AsyncMock()
        gpu_manager.get_state = AsyncMock(side_effect=_growing_state)
        model_manager._gpu_manager = gpu_manager
        log_mock = MagicMock()
        monkeypatch.setattr(model_manager_module, "logger", log_mock)

        await self._run_briefly(model_manager, "test/healthy-model", [0])

        warnings = [
            c for c in log_mock.warning.call_args_list if c.args[0] == "model_load_appears_stalled"
        ]
        assert not warnings

    @pytest.mark.asyncio
    async def test_per_sample_logging_follows_debug_flag(
        self, model_manager, monkeypatch
    ):
        import ocabra.core.model_manager as model_manager_module
        from ocabra.config import settings
        from ocabra.core.gpu_manager import GPUState

        self._patch_fast_sleep(monkeypatch, model_manager_module)
        monkeypatch.setattr(settings, "debug_log_load_vram_samples", False)
        state = GPUState(
            index=0,
            name="RTX 3060",
            total_vram_mb=12_000,
            free_vram_mb=11_700,
            used_vram_mb=256,
            utilization_pct=0,
            temperature_c=40,
            power_draw_w=50,
            power_limit_w=170,
            locked_vram_mb=0,
        )
        gpu_manager = AsyncMock()
        gpu_manager.get_state = AsyncMock(return_value=state)
        model_manager._gpu_manager = gpu_manager
        log_mock = MagicMock()
        monkeypatch.setattr(model_manager_module, "logger", log_mock)

        await self._run_briefly(model_manager, "test/quiet-model", [0])

        samples = [
            c for c in log_mock.info.call_args_list if c.args[0] == "llama_cpp_load_vram_sample"
        ]
        assert not samples, "per-sample logging must be off when the debug flag is False"


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


@pytest.mark.asyncio
async def test_effective_inflight_cap_skips_slot_scaling_for_embedding_models(
    model_manager, monkeypatch
):
    """2026-10-29 incident: an embedding model with no ``_runtime`` metadata
    fell back to the generative-model default of 1 slot, capping it at just
    4 concurrent requests (1 slot * max_queued_per_slot) — a burst of ~15
    parallel embedding calls got mostly rejected with 429. Embeddings/
    pooling/rerank/classification/score are single-pass encoders with no
    per-slot KV cache to exhaust, so they should get the full flat
    max_inflight ceiling instead of the generative-model fallback."""
    monkeypatch.setattr("ocabra.core.model_manager.settings.max_queued_per_slot", 4)
    model_id = "test/embedding-model"
    state = await add_test_model(model_manager, model_id=model_id)
    state.capabilities = BackendCapabilities(embeddings=True)
    # No _runtime metadata recorded — would fall back to slots=1 for a
    # generative model, capping at 1 * 4 = 4.

    assert model_manager.effective_inflight_cap(model_id, 32) == 32

    results = [
        model_manager.try_begin_request(model_id, max_inflight=32) for _ in range(32)
    ]
    assert results.count(None) == 0

    assert model_manager.try_begin_request(model_id, max_inflight=32) is None


@pytest.mark.asyncio
async def test_effective_inflight_cap_exempts_encoder_endpoints_without_capabilities(
    model_manager, monkeypatch
):
    """2026-11-06 incident: ``bge-reranker-v2-m3`` only ever executes on a
    federation peer, so its local ModelState never reaches LOADED and its
    ``capabilities`` stay at the all-false default. The capability-based
    exemption above therefore never fired and admission fell back to the
    generative 1-slot assumption, capping /v1/rerank at 4 concurrent
    requests (measured live: 8 parallel calls -> 4x200 + 4x429) while the
    remote worker had capacity to spare. The endpoint being called is known
    with certainty at admission time, so it is the signal we trust."""
    monkeypatch.setattr("ocabra.core.model_manager.settings.max_queued_per_slot", 4)
    model_id = "vllm/BAAI/bge-reranker-v2-m3"
    state = await add_test_model(model_manager, model_id=model_id)
    # Exactly the broken situation: never loaded locally, nothing to infer from.
    state.capabilities = BackendCapabilities()
    assert state.status != ModelStatus.LOADED

    assert model_manager.effective_inflight_cap(model_id, 32) == 4
    assert model_manager.effective_inflight_cap(model_id, 32, request_kind="rerank") == 32

    results = [
        model_manager.try_begin_request(model_id, max_inflight=32, request_kind="rerank")
        for _ in range(32)
    ]
    assert results.count(None) == 0
    assert model_manager.try_begin_request(model_id, max_inflight=32, request_kind="rerank") is None


@pytest.mark.asyncio
async def test_effective_inflight_cap_skips_slot_scaling_when_bound_for_a_peer(
    model_manager, monkeypatch
):
    """A model that only ever runs on a federation peer has no local worker,
    so the slot-derived cap describes a KV cache the request never touches.
    This is the gap that would bite a generative model hosted remotely (e.g.
    on a unified-memory box joined to the cluster): it would inherit the
    1-slot assumption and be throttled to 4 concurrent requests regardless of
    the peer's real capacity, which applies its own admission control."""
    monkeypatch.setattr("ocabra.core.model_manager.settings.max_queued_per_slot", 4)
    model_id = "remote/big-generative-model"
    state = await add_test_model(model_manager, model_id=model_id)
    state.capabilities = BackendCapabilities(chat=True)

    # Generative + not remote: the local slot cap still protects a real KV cache.
    assert model_manager.effective_inflight_cap(model_id, 32, request_kind="chat") == 4
    # Same model, same kind, but headed for a peer.
    assert (
        model_manager.effective_inflight_cap(
            model_id, 32, request_kind="chat", remote_bound=True
        )
        == 32
    )

    results = [
        model_manager.try_begin_request(
            model_id, max_inflight=32, request_kind="chat", remote_bound=True
        )
        for _ in range(32)
    ]
    assert results.count(None) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "request_kind", ["embedding", "ollama_embedding", "rerank", "pooling", "score", "classify"]
)
async def test_effective_inflight_cap_exempts_every_encoder_kind(
    model_manager, monkeypatch, request_kind
):
    """Every single-pass encoder endpoint gets the flat ceiling. /v1/pooling,
    /v1/score and /v1/classify used to classify as "other" and so could not
    be exempted at all."""
    monkeypatch.setattr("ocabra.core.model_manager.settings.max_queued_per_slot", 4)
    model_id = f"test/encoder-{request_kind}"
    state = await add_test_model(model_manager, model_id=model_id)
    state.capabilities = BackendCapabilities()

    assert model_manager.effective_inflight_cap(model_id, 32, request_kind=request_kind) == 32


@pytest.mark.asyncio
async def test_effective_inflight_cap_still_scales_down_generative_kinds(
    model_manager, monkeypatch
):
    """The exemption must not leak to generative endpoints: a single-slot
    worker still has a real KV cache to exhaust, so chat/completion keep the
    slot-scaled cap."""
    monkeypatch.setattr("ocabra.core.model_manager.settings.max_queued_per_slot", 4)
    model_id = "test/generative-model"
    state = await add_test_model(model_manager, model_id=model_id)
    state.capabilities = BackendCapabilities(chat=True)

    assert model_manager.effective_inflight_cap(model_id, 32, request_kind="chat") == 4
    assert model_manager.effective_inflight_cap(model_id, 32, request_kind="other") == 4
    assert model_manager.effective_inflight_cap(model_id, 32) == 4


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
    now = datetime.now(UTC)

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
    now = datetime.now(UTC).replace(microsecond=0)
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
        candidate.loaded_at = datetime.now(UTC) - timedelta(minutes=10)
        candidate.last_request_at = datetime.now(UTC) - timedelta(minutes=5)

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
async def test_pressure_eviction_skips_impossible_aggregate_plan(worker_pool):
    """Do not evict a resident model when all reclaimable VRAM is still short."""
    from ocabra.core.scheduler import InsufficientVRAMError

    gpu_manager = AsyncMock()
    gpu_manager.get_all_states.return_value = [
        SimpleNamespace(index=0),
        SimpleNamespace(index=1),
    ]
    gpu_manager.get_free_vram.side_effect = [2800, 1800]
    gpu_manager.get_free_vram_if_released.return_value = 1800 + 23_632
    gpu_manager.locked_gpu_indices = MagicMock(return_value=[1])
    scheduler = AsyncMock()
    scheduler.find_gpu_for_model.side_effect = InsufficientVRAMError("full")
    mm = ModelManager(worker_pool, gpu_manager=gpu_manager, gpu_scheduler=scheduler)

    requested = await add_test_model(mm, model_id="test/requested")
    requested.preferred_gpu = 1
    candidate = await add_test_model(mm, model_id="test/candidate", load_policy="pin")
    candidate.status = ModelStatus.LOADED
    candidate.current_gpu = [1]
    candidate.vram_used_mb = 23_632

    with (
        patch.object(mm, "unload", new=AsyncMock()) as unload_mock,
        pytest.raises(InsufficientVRAMError, match="full"),
    ):
        await mm._assign_gpus_for_load(
            "test/requested",
            vram_needed=29_656,
            preferred_gpu=1,
            enforce_vllm_headroom=False,
        )

    unload_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_pressure_eviction_counts_real_footprint_not_estimate(worker_pool):
    """2026-11-09 live incident: gemma4:26b (PIN, 23,600 MB) got a 409 on a 3090
    holding only ravenx. ravenx was estimated at 23,237 MB but really held
    ~23,594, so "53 free + 23,237 reclaimable" looked short and eviction was
    skipped, although an empty 3090 offers 23,647. The plan must be judged on
    the projected free VRAM after release, which counts the real footprint."""
    from ocabra.core.scheduler import InsufficientVRAMError

    gpu_manager = AsyncMock()
    gpu_manager.get_all_states.return_value = [
        SimpleNamespace(index=0),
        SimpleNamespace(index=1),
    ]
    gpu_manager.get_free_vram.return_value = 53
    gpu_manager.get_free_vram_if_released.return_value = 23_647
    gpu_manager.locked_gpu_indices = MagicMock(return_value=[1])
    scheduler = AsyncMock()
    scheduler.find_gpu_for_model.side_effect = [InsufficientVRAMError("full"), [1]]
    mm = ModelManager(worker_pool, gpu_manager=gpu_manager, gpu_scheduler=scheduler)

    await add_test_model(mm, model_id="test/gemma")
    candidate = await add_test_model(mm, model_id="test/ravenx")
    candidate.status = ModelStatus.LOADED
    candidate.current_gpu = [1]
    candidate.vram_used_mb = 23_237
    candidate.worker_info = WorkerInfo(
        backend_type="mock",
        model_id="test/ravenx",
        gpu_indices=[1],
        port=18001,
        pid=4321,
        vram_used_mb=23_237,
    )

    with patch.object(mm, "unload", new=AsyncMock()) as unload_mock:
        await mm._assign_gpus_for_load(
            "test/gemma",
            vram_needed=23_600,
            preferred_gpu=1,
            enforce_vllm_headroom=False,
        )

    unload_mock.assert_awaited_once_with("test/ravenx", reason="pressure")
    gpu_manager.get_free_vram_if_released.assert_any_await(1, {"test/ravenx": 4321})


@pytest.mark.asyncio
async def test_pressure_eviction_counts_unlocked_candidate_by_estimate(worker_pool):
    """Ollama runners hold no lock and no PID of ours: keep their estimate."""
    from ocabra.core.scheduler import InsufficientVRAMError

    gpu_manager = AsyncMock()
    gpu_manager.get_free_vram.return_value = 1_000
    gpu_manager.get_free_vram_if_released.return_value = 1_000
    gpu_manager.get_all_states.return_value = [SimpleNamespace(index=0)]
    gpu_manager.locked_gpu_indices = MagicMock(return_value=[])
    scheduler = AsyncMock()
    scheduler.find_gpu_for_model.side_effect = [InsufficientVRAMError("full"), [0]]
    mm = ModelManager(worker_pool, gpu_manager=gpu_manager, gpu_scheduler=scheduler)

    await add_test_model(mm, model_id="test/requested")
    candidate = await add_test_model(mm, model_id="ollama/runner")
    candidate.status = ModelStatus.LOADED
    candidate.current_gpu = [0]
    candidate.vram_used_mb = 6_000

    with patch.object(mm, "unload", new=AsyncMock()) as unload_mock:
        await mm._assign_gpus_for_load(
            "test/requested",
            vram_needed=5_000,
            preferred_gpu=0,
            enforce_vllm_headroom=False,
        )

    unload_mock.assert_awaited_once_with("ollama/runner", reason="pressure")


@pytest.mark.asyncio
async def test_unload_releases_lock_when_backend_lost_gpu_attribution(worker_pool):
    """A stale lock must not survive an Ollama worker reporting ``gpu=[]``."""
    gpu_manager = AsyncMock()
    gpu_manager.locked_gpu_indices.return_value = [0]
    mm = ModelManager(worker_pool, gpu_manager=gpu_manager)
    state = await add_test_model(mm, model_id="test/stale-lock")
    state.status = ModelStatus.LOADED
    state.current_gpu = []
    state.vram_used_mb = 5903
    state.worker_info = WorkerInfo(
        backend_type="mock",
        model_id=state.model_id,
        gpu_indices=[],
        port=0,
        pid=0,
        vram_used_mb=5903,
    )
    worker_pool.set_worker(state.model_id, state.worker_info)

    with (
        patch("ocabra.core.model_manager.publish", new=AsyncMock()),
        patch("ocabra.core.model_manager.set_key", new=AsyncMock()),
    ):
        await mm.unload(state.model_id, reason="pressure")

    gpu_manager.unlock_vram.assert_awaited_once_with(0, state.model_id)


@pytest.mark.asyncio
async def test_unload_releases_lock_for_model_stuck_in_error_status(worker_pool):
    """2026-11-04 live incident: gemma4:26b sat in ERROR status after a
    generation timeout, with its backend process still alive and its VRAM
    lock still held — unload() used to no-op on anything but LOADED, so
    there was no way to recover short of killing the process by hand. Every
    other model's load on that GPU failed with "insufficient VRAM" until
    then. unload() must now also act on ERROR status."""
    gpu_manager = AsyncMock()
    gpu_manager.locked_gpu_indices.return_value = [1]
    mm = ModelManager(worker_pool, gpu_manager=gpu_manager)
    state = await add_test_model(mm, model_id="llama_cpp/stuck-model", load_policy="pin")
    state.status = ModelStatus.ERROR
    state.current_gpu = [1]
    state.vram_used_mb = 23632
    state.worker_info = WorkerInfo(
        backend_type="mock",
        model_id=state.model_id,
        gpu_indices=[1],
        port=0,
        pid=0,
        vram_used_mb=23632,
    )
    worker_pool.set_worker(state.model_id, state.worker_info)

    with (
        patch("ocabra.core.model_manager.publish", new=AsyncMock()),
        patch("ocabra.core.model_manager.set_key", new=AsyncMock()),
    ):
        await mm.unload(state.model_id, reason="manual")

    gpu_manager.unlock_vram.assert_awaited_once_with(1, state.model_id)
    assert state.status == ModelStatus.UNLOADED
    assert state.current_gpu == []
    assert state.vram_used_mb == 0


@pytest.mark.asyncio
async def test_pressure_eviction_skips_wrong_gpu_and_cpu_only_candidates(worker_pool):
    """2026-10-31 live incident: a GPU1 shortfall evicted piper (no GPU at
    all) and a GPU0 model before ever reaching the GPU1 neighbour that
    actually had room to give. Neither of those could ever have freed GPU1
    VRAM — _get_pressure_eviction_candidates must skip them when the
    requested model has an explicit preferred_gpu.
    """
    mm = ModelManager(worker_pool, gpu_scheduler=AsyncMock())

    with (
        patch("ocabra.core.model_manager.publish", new=AsyncMock()),
        patch("ocabra.core.model_manager.set_key", new=AsyncMock()),
        patch.object(mm, "unload", new=AsyncMock()) as unload_mock,
    ):
        requested = await add_test_model(mm, model_id="test/requested")
        requested.preferred_gpu = 1

        cpu_only = await add_test_model(mm, model_id="test/cpu-only", load_policy="on_demand")
        cpu_only.status = ModelStatus.LOADED
        cpu_only.current_gpu = []
        cpu_only.loaded_at = datetime.now(UTC) - timedelta(minutes=20)

        wrong_gpu = await add_test_model(mm, model_id="test/wrong-gpu", load_policy="on_demand")
        wrong_gpu.status = ModelStatus.LOADED
        wrong_gpu.current_gpu = [0]
        wrong_gpu.loaded_at = datetime.now(UTC) - timedelta(minutes=15)

        right_gpu = await add_test_model(mm, model_id="test/right-gpu", load_policy="warm")
        right_gpu.status = ModelStatus.LOADED
        right_gpu.current_gpu = [1]
        right_gpu.loaded_at = datetime.now(UTC) - timedelta(minutes=10)

        from ocabra.core.scheduler import InsufficientVRAMError

        mm._gpu_scheduler.find_gpu_for_model = AsyncMock(
            side_effect=[
                InsufficientVRAMError("full"),
                [1],
            ]
        )

        state = await mm.load("test/requested", force_gpu=1)

    assert state.status == ModelStatus.LOADED
    unload_mock.assert_awaited_once_with("test/right-gpu", reason="pressure")


@pytest.mark.asyncio
async def test_pressure_eviction_falls_back_to_other_gpu_pool(worker_pool):
    """2026-11-02 live report: a model merely *preferring* GPU0 (not
    restricted to it — find_gpu_for_model already falls back across GPUs
    when one has free room) failed outright because GPU0 had nothing
    evictable while GPU1 held a soft-pin neighbour that would have freed
    plenty of room. preferred_gpu must only order which GPU's resident
    models are tried first for eviction, not strand the load on a GPU
    with no candidates — it still never evicts on GPU A hoping to free
    GPU B (each pool pass stays scoped to one GPU at a time)."""
    from ocabra.core.scheduler import InsufficientVRAMError

    gpu_manager = AsyncMock()
    gpu_manager.get_all_states.return_value = [
        SimpleNamespace(index=0),
        SimpleNamespace(index=1),
    ]
    gpu_manager.get_free_vram = AsyncMock(return_value=400)
    gpu_manager.get_free_vram_if_released = AsyncMock(return_value=400 + 22_000)
    gpu_manager.locked_gpu_indices = MagicMock(return_value=[1])
    scheduler = AsyncMock()
    scheduler.find_gpu_for_model = AsyncMock(side_effect=[InsufficientVRAMError("full"), [1]])
    mm = ModelManager(worker_pool, gpu_manager=gpu_manager, gpu_scheduler=scheduler)

    requested = await add_test_model(mm, model_id="test/embedding")
    requested.preferred_gpu = 0

    other_gpu_pin = await add_test_model(mm, model_id="test/pinned-llm", load_policy="pin")
    other_gpu_pin.status = ModelStatus.LOADED
    other_gpu_pin.current_gpu = [1]
    other_gpu_pin.vram_used_mb = 22_000
    other_gpu_pin.loaded_at = datetime.now(UTC) - timedelta(hours=2)

    with patch.object(mm, "unload", new=AsyncMock()) as unload_mock:
        gpu_indices = await mm._assign_gpus_for_load(
            "test/embedding",
            vram_needed=7740,
            preferred_gpu=0,
            enforce_vllm_headroom=False,
        )

    assert gpu_indices == [1]
    unload_mock.assert_awaited_once_with("test/pinned-llm", reason="pressure")


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
            "preferred_gpu": 1,
            "extra_config": {"foo": "bar"},
        },
    )

    assert updated is state
    assert state.display_name == "renamed"
    assert state.load_policy == LoadPolicy.WARM
    assert state.preferred_gpu == 1
    assert state.extra_config == {"foo": "bar"}


@pytest.mark.asyncio
async def test_update_config_deep_merges_extra_config_preserving_unmentioned_fields(
    model_manager,
):
    """2026-10-29 incident: the Models UI only knows about its own form
    fields (gpuLayers, ctxSize, ...) and has no notion of backend-managed
    fields like llama_cpp.model_path. A PATCH that only touches the known
    fields must not wipe model_path — reproduced live against
    qwen3.8:27b, which lost model_path this way and broke context-capacity
    planning for every subsequent request."""
    state = await add_test_model(model_manager)
    state.extra_config = {
        "llama_cpp": {
            "model_path": "/data/models/real-weights.gguf",
            "gpu_layers": 99,
            "ctx_size": 32768,
        },
        "registry_metadata": {"source": "huggingface"},
    }

    await model_manager.update_config(
        "test/model",
        {
            "extra_config": {
                "llama_cpp": {
                    "gpuLayers": 99,
                    "ctxSize": 65536,
                    "flashAttn": True,
                },
            },
        },
    )

    assert state.extra_config["llama_cpp"]["model_path"] == "/data/models/real-weights.gguf"
    assert state.extra_config["llama_cpp"]["gpu_layers"] == 99
    # The UI's camelCase edit is normalised onto the stored snake_case key, so
    # it takes effect. Before migration 0034 it was kept next to ``ctx_size``
    # and the reader (snake first) went on applying 32768.
    assert state.extra_config["llama_cpp"]["ctx_size"] == 65536
    assert state.extra_config["llama_cpp"]["flash_attn"] is True
    assert "ctxSize" not in state.extra_config["llama_cpp"]
    assert state.extra_config["registry_metadata"] == {"source": "huggingface"}



@pytest.mark.asyncio
async def test_update_config_null_clears_a_field_back_to_its_default(model_manager):
    """Clearing a field in the Models form sends ``null``: the key must go, so
    the option reader falls back to the backend default instead of keeping the
    old value (key omitted) or returning ``None`` (key stored as null)."""
    state = await add_test_model(model_manager)
    state.extra_config = {"llama_cpp": {"ctx_size": 32768, "gpu_layers": 99}}

    await model_manager.update_config(
        "test/model", {"extra_config": {"llama_cpp": {"ctx_size": None}}}
    )

    assert "ctx_size" not in state.extra_config["llama_cpp"]
    assert state.extra_config["llama_cpp"]["gpu_layers"] == 99

@pytest.mark.asyncio
async def test_pressure_eviction_prefers_on_demand_then_warm_then_pin(worker_pool):
    mm = ModelManager(worker_pool)

    now = datetime.now(UTC)
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


@pytest.mark.asyncio
async def test_runtime_single_gpu_placement_bypasses_scheduler_spill():
    """A forced one-GPU runtime remains on that GPU even when a generic
    estimate would make the scheduler prefer a spill allocation."""
    from ocabra.core.model_manager import ModelState

    wp = WorkerPool()
    backend = _PortRequiredBackend()
    wp.register_backend("llama_cpp", backend)
    mm = ModelManager(wp)
    model_id = "llama_cpp/forced-single-gpu"

    with (
        patch("ocabra.core.model_manager.publish", new=AsyncMock()),
        patch("ocabra.core.model_manager.set_key", new=AsyncMock()),
        patch.object(mm, "_record_model_load_stat", new=AsyncMock()),
        patch.object(mm, "_admit_gpu_load", new=AsyncMock(return_value=[1])) as admit,
    ):
        mm._states[model_id] = ModelState(
            model_id=model_id,
            display_name="forced-single-gpu",
            backend_type="llama_cpp",
            load_policy=LoadPolicy.ON_DEMAND,
            extra_config={
                "llama_cpp": {"ctx_size": 131_072, "parallel_slots": 4},
                "_runtime": {"gpu_indices": [1]},
            },
        )
        mm._load_locks[model_id] = asyncio.Lock()

        state = await mm.load(model_id)

    assert state.current_gpu == [1]
    assert admit.await_args.kwargs["required_gpu_indices"] == [1]


@pytest.mark.asyncio
async def test_multi_gpu_admission_evicts_for_each_tensor_shard() -> None:
    gpu_manager = AsyncMock()
    gpu_manager.get_free_vram = AsyncMock(
        # GPU0 before eviction, GPU0 after eviction, then GPU1.
        side_effect=[5_000, 9_100, 24_000]
    )
    mm = ModelManager(WorkerPool(), gpu_manager=gpu_manager)

    with (
        patch.object(mm, "_reconcile_ollama_state", new=AsyncMock()) as reconcile,
        patch.object(mm, "_evict_for_space", new=AsyncMock(return_value=0)),
        patch.object(
            mm,
            "_evict_last_resort_for_space",
            new=AsyncMock(return_value=4_000),
        ) as last_resort,
    ):
        selected = await mm._admit_gpu_load(
            model_id="llama_cpp/dual",
            vram_needed=27_000,
            preferred_gpu=None,
            gpu_managed=True,
            enforce_vllm_headroom=False,
            vllm_gpu_memory_utilization=None,
            required_gpu_indices=[0, 1],
            required_gpu_vram_mb={0: 9_000, 1: 18_000},
        )

    assert selected == [0, 1]
    assert reconcile.await_count == 2
    last_resort.assert_awaited_once_with("llama_cpp/dual", 0, 4_000)


def _llama_cpp_pool() -> WorkerPool:
    """A worker pool whose mock backend is registered as llama.cpp."""
    wp = WorkerPool()
    wp.register_backend("llamacpp_mks", MockBackend(vram_mb=4096))
    return wp


async def _load_multi_gpu(backend_type: str, real_usage_mb: dict[int, int]):
    from ocabra.core.model_manager import ModelState

    gpu_manager = AsyncMock()
    gpu_manager.get_process_vram_mb = MagicMock(
        side_effect=lambda idx: {12345: real_usage_mb[idx]} if idx in real_usage_mb else {}
    )
    wp = _llama_cpp_pool() if backend_type == "llamacpp_mks" else make_worker_pool_with_mock()
    mm = ModelManager(wp, gpu_manager=gpu_manager)
    model_id = f"{backend_type}/multi-gpu"
    mm._states[model_id] = ModelState(
        model_id=model_id,
        display_name=model_id,
        backend_type=backend_type,
        load_policy=LoadPolicy.ON_DEMAND,
        extra_config={"llama_cpp": {"gpu_layers": 99}},
    )
    mm._load_locks[model_id] = asyncio.Lock()

    with (
        patch("ocabra.core.model_manager.publish", new=AsyncMock()),
        patch("ocabra.core.model_manager.set_key", new=AsyncMock()),
        patch.object(mm, "_admit_gpu_load", new=AsyncMock(return_value=[0, 1])),
    ):
        await mm.load(model_id)
    return gpu_manager, model_id


@pytest.mark.asyncio
async def test_multi_gpu_load_locks_real_per_gpu_vram_not_even_split() -> None:
    """2026-11-04 live incident: a tensor-split worker (qwen3.8:27b) landed
    unevenly across two GPUs (8.3 GB / 23.2 GB in production) but got
    locked for an even split of the total on EACH GPU — making the lighter
    GPU register as fully reserved (free computed negative, clamped to 0)
    even though it had several GB physically free, starving an unrelated
    load that would have fit there. lock_vram must use the real per-process
    reading for this worker's PID on every GPU the worker touched.

    2026-11-09: that reading must be taken live from the API's own NVML. The
    polled snapshot comes from hw-monitor, whose PID namespace lists no
    processes, so the original fix never applied in production."""
    gpu_manager, model_id = await _load_multi_gpu("llamacpp_mks", {0: 8_292, 1: 23_214})

    gpu_manager.lock_vram.assert_any_call(0, 8_292, model_id)
    gpu_manager.lock_vram.assert_any_call(1, 23_214, model_id)


@pytest.mark.asyncio
async def test_multi_gpu_lock_falls_back_to_even_split_on_partial_reading() -> None:
    """A GPU without a reading for the PID must not be locked at 0."""
    gpu_manager, model_id = await _load_multi_gpu("llamacpp_mks", {1: 23_214})

    locked = {c.args[0]: c.args[1] for c in gpu_manager.lock_vram.await_args_list}
    assert locked[0] == locked[1]
    assert locked[0] > 0


@pytest.mark.asyncio
async def test_multi_gpu_lock_ignores_pid_reading_outside_llama_cpp() -> None:
    """vLLM allocates in a child process, so its recorded PID can show a
    small CUDA context on every GPU; locking that would under-reserve both."""
    gpu_manager, model_id = await _load_multi_gpu("mock", {0: 400, 1: 400})

    locked = {c.args[0]: c.args[1] for c in gpu_manager.lock_vram.await_args_list}
    assert locked[0] == locked[1]
    assert locked[0] > 400


@pytest.mark.asyncio
async def test_validated_preset_uses_physical_capacity_at_measured_edge() -> None:
    from ocabra.core.model_manager import ModelState

    gpu_manager = AsyncMock()
    gpu_manager.get_free_vram = AsyncMock(return_value=23_343)
    gpu_manager.get_state = AsyncMock(
        return_value=SimpleNamespace(free_vram_mb=24_159)
    )
    mm = ModelManager(WorkerPool(), gpu_manager=gpu_manager)
    model_id = "llama_cpp/measured-edge"
    mm._states[model_id] = ModelState(
        model_id=model_id,
        display_name="Measured edge",
        backend_type="llama_cpp",
        extra_config={"_runtime": {"capacity_validated": True}},
    )

    selected = await mm._admit_gpu_load(
        model_id=model_id,
        vram_needed=23_360,
        preferred_gpu=1,
        gpu_managed=True,
        enforce_vllm_headroom=False,
        vllm_gpu_memory_utilization=None,
        required_gpu_indices=[1],
    )

    assert selected == [1]
    gpu_manager.get_state.assert_awaited_once_with(1)
    gpu_manager.get_free_vram.assert_not_awaited()


@pytest.mark.asyncio
async def test_last_resort_waits_for_busy_soft_pin_to_drain(monkeypatch) -> None:
    from ocabra.core.model_manager import ModelState

    mm = ModelManager(WorkerPool())
    candidate_id = "ollama/qwen3-embedding:4b"
    mm._states[candidate_id] = ModelState(
        model_id=candidate_id,
        backend_model_id="qwen3-embedding:4b",
        display_name="Embedding",
        backend_type="ollama",
        load_policy=LoadPolicy.PIN,
        pin_permanent=False,
        status=ModelStatus.LOADED,
        current_gpu=[0],
        vram_used_mb=5903,
    )
    mm._in_flight[candidate_id] = 1

    async def finish_request(_delay: float) -> None:
        mm._in_flight.pop(candidate_id, None)

    monkeypatch.setattr("ocabra.core.model_manager.asyncio.sleep", finish_request)
    mm.unload = AsyncMock()  # type: ignore[method-assign]
    mm._wait_for_vram_released = AsyncMock()  # type: ignore[method-assign]

    freed = await mm._evict_last_resort_for_space("llama_cpp/dual", 0, 5_000)

    assert freed == 5903
    mm.unload.assert_awaited_once_with(candidate_id, reason="vram_pressure")


@pytest.mark.asyncio
async def test_runtime_downgrade_recreates_missing_warm_default() -> None:
    from ocabra.core.model_manager import ModelState

    mm = ModelManager(WorkerPool())
    base_id = "llama_cpp/gemma"
    max_id = f"{base_id}::max"
    default_id = f"{base_id}::normal"
    mm._states[base_id] = ModelState(
        model_id=base_id,
        display_name="Gemma",
        backend_type="llama_cpp",
        backend_model_id="gemma",
        load_policy=LoadPolicy.WARM,
        extra_config={"llama_cpp": {"model_path": "/models/gemma.gguf"}},
    )
    mm._states[max_id] = ModelState(
        model_id=max_id,
        display_name="Gemma max",
        backend_type="llama_cpp",
        backend_model_id="gemma",
        load_policy=LoadPolicy.ON_DEMAND,
        status=ModelStatus.LOADED,
        extra_config={
            "_runtime": {
                "profile_id": "gemma",
                "base_model_id": base_id,
                "effective_context": 262_144,
                "default_context": 32_768,
                "default_worker_key": default_id,
                "default_load_overrides": {
                    "use_case": {"context": 32_768, "slots": 4, "kv_dtype": "q8"},
                    "_runtime": {
                        "profile_id": "gemma",
                        "base_model_id": base_id,
                        "effective_context": 32_768,
                        "gpu_indices": [1],
                    },
                },
            }
        },
    )

    async def _unload(model_id: str, reason: str) -> None:
        assert mm.runtime_transition(base_id) == "downsizing"
        mm._states[model_id].status = ModelStatus.UNLOADED

    with (
        patch.object(mm, "unload", new=_unload),
        patch.object(mm, "_load_model", new=AsyncMock()) as load_model,
        patch("ocabra.core.model_manager.publish", new=AsyncMock()),
        patch("ocabra.core.model_manager.set_key", new=AsyncMock()),
    ):
        await mm._downgrade_runtime_variant(max_id)

    default_state = mm._states[default_id]
    assert default_state.load_policy == LoadPolicy.WARM
    assert default_state.preferred_gpu == 1
    assert default_state.extra_config["use_case"]["slots"] == 4
    load_model.assert_awaited_once_with(default_id)
    assert mm.runtime_transition(base_id) is None


@pytest.mark.asyncio
async def test_cancel_load_cancels_derived_owner_task() -> None:
    from ocabra.core.model_manager import ModelState

    mm = ModelManager(WorkerPool())
    base_id = "llama_cpp/deadbydawn/chaos-gguf::Chaos-Q4_K_M"
    worker_id = f"{base_id}::large"
    mm._states[base_id] = ModelState(
        model_id=base_id,
        display_name="Chaos",
        backend_type="llama_cpp",
    )
    mm._states[worker_id] = ModelState(
        model_id=worker_id,
        display_name="Chaos large",
        backend_type="llama_cpp",
        status=ModelStatus.LOADING,
        extra_config={"_runtime": {"base_model_id": base_id}},
    )
    started = asyncio.Event()

    async def _owner() -> None:
        started.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(_owner())
    await started.wait()
    mm._load_tasks[worker_id] = task

    cancelled = await mm.cancel_load(base_id)

    assert cancelled == [worker_id]
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_downsize_runtime_rejects_busy_multi_gpu_worker() -> None:
    from ocabra.core.model_manager import ModelState

    mm = ModelManager(WorkerPool())
    worker_id = "llama_cpp/chaos::large"
    mm._states[worker_id] = ModelState(
        model_id=worker_id,
        display_name="Chaos large",
        backend_type="llama_cpp",
        status=ModelStatus.LOADED,
        current_gpu=[0, 1],
        extra_config={
            "_runtime": {
                "effective_context": 131_072,
                "default_context": 65_536,
                "gpu_indices": [0, 1],
                "default_load_overrides": {
                    "_runtime": {"effective_context": 65_536, "gpu_indices": [1]}
                },
            }
        },
    )
    mm._in_flight[worker_id] = 1

    with pytest.raises(RuntimeError, match="active requests"):
        await mm.downsize_runtime("llama_cpp/chaos")


@pytest.mark.asyncio
async def test_canonical_gguf_unload_reaches_loaded_derived_worker() -> None:
    from ocabra.core.model_manager import ModelState

    mm = ModelManager(WorkerPool())
    base_id = "llama_cpp/deadbydawn/chaos-gguf::Chaos-Q4_K_M"
    worker_id = f"{base_id}::runtime-hash"
    mm._states[base_id] = ModelState(
        model_id=base_id,
        display_name="Chaos",
        backend_type="llama_cpp",
        status=ModelStatus.CONFIGURED,
    )
    mm._states[worker_id] = ModelState(
        model_id=worker_id,
        display_name="Chaos runtime",
        backend_type="llama_cpp",
        status=ModelStatus.LOADED,
        extra_config={"_runtime": {"base_model_id": base_id}},
    )
    unloaded: list[str] = []

    async def _unload(model_id: str, reason: str = "manual") -> None:
        if model_id == base_id:
            await ModelManager.unload(mm, model_id, reason)
            return
        unloaded.append(model_id)
        mm._states[model_id].status = ModelStatus.UNLOADED

    with patch.object(mm, "unload", new=_unload):
        await _unload(base_id)

    assert unloaded == [worker_id]


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
async def test_reconcile_ollama_state_refreshes_loaded_model_without_relocking():
    """Reconciliation repairs placement drift without double-locking VRAM."""
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
        current_gpu=[],
        vram_used_mb=24_000,
    )

    await mm._reconcile_ollama_state()

    gpu_manager.lock_vram.assert_not_awaited()
    assert mm._states[model_id].current_gpu == [1]
    assert wp.get_worker(model_id) is not None


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


class _PiperStyleBackend(BackendInterface):
    """Mimics PiperBackend: a local HTTP server (needs a port) that is
    unconditionally CPU-only, regardless of the gpu_indices it's called
    with — the real backend ignores its gpu_indices argument entirely."""

    def __init__(self) -> None:
        self.received_gpu_indices: list[int] | None = None

    async def load(self, model_id: str, gpu_indices: list[int], **kwargs) -> WorkerInfo:
        self.received_gpu_indices = gpu_indices
        return WorkerInfo(
            backend_type="piper",
            model_id=model_id,
            gpu_indices=[],
            # The test process's own PID — guaranteed to exist with a real,
            # non-zero RSS, so tests can assert ram_used_mb actually reflects
            # a psutil measurement rather than a dead/fake PID silently
            # resolving to the "unknown" (0) fallback.
            port=kwargs.get("port", 0),
            pid=os.getpid(),
            vram_used_mb=0,
        )

    async def unload(self, model_id: str) -> None:
        return None

    async def health_check(self, model_id: str) -> bool:
        return True

    async def get_capabilities(self, model_id: str) -> BackendCapabilities:
        return BackendCapabilities(tts=True, streaming=True)

    async def get_vram_estimate_mb(self, model_id: str, **kwargs) -> int:
        return 0

    async def forward_request(self, model_id: str, path: str, body: dict):
        return {}

    async def forward_stream(self, model_id: str, path: str, body: dict):
        if False:
            yield b""


@pytest.mark.asyncio
async def test_piper_backend_is_cpu_only_and_skips_gpu_admission():
    """Regression test (2026-10-28): Piper models showed a GPU badge and
    were attributed to a GPU card on the Dashboard even though Piper is
    CPU-only. Root cause #1: piper wasn't added to the gpu_managed
    exclusion list (unlike bitnet/llama_cpp's CPU-only special cases), so
    the generic scheduler assigned it a real GPU before PiperBackend.load()
    ignored that and returned gpu_indices=[]. This test locks in that the
    scheduler is never even consulted for piper."""
    wp = WorkerPool()
    backend = _PiperStyleBackend()
    wp.register_backend("piper", backend)
    scheduler = AsyncMock()
    scheduler.find_gpu_for_model = AsyncMock(return_value=[1])
    mm = ModelManager(wp, gpu_scheduler=scheduler)

    with (
        patch("ocabra.core.model_manager.publish", new=AsyncMock()),
        patch("ocabra.core.model_manager.set_key", new=AsyncMock()),
    ):
        from ocabra.core.model_manager import ModelState

        mm._states["test/piper-voice"] = ModelState(
            model_id="test/piper-voice",
            display_name="test/piper-voice",
            backend_type="piper",
            load_policy=LoadPolicy.ON_DEMAND,
        )
        mm._load_locks["test/piper-voice"] = asyncio.Lock()

        state = await mm.load("test/piper-voice")

    assert state.status == ModelStatus.LOADED
    assert state.current_gpu == []
    scheduler.find_gpu_for_model.assert_not_called()
    # CPU-only: oCabra should measure the worker process's real RAM (RSS)
    # instead of leaving memory usage unreported like the vram_used_mb=0
    # it already correctly shows.
    assert state.ram_used_mb > 0
    assert state.vram_used_mb == 0


@pytest.mark.asyncio
async def test_load_never_substitutes_scheduler_gpu_for_an_empty_worker_report():
    """Root cause #2 of the same incident: even when a backend DOES go
    through GPU admission (gpu_managed=True) but its WorkerInfo honestly
    reports gpu_indices=[] (e.g. it fell back to CPU internally), the old
    `worker_info.gpu_indices or gpu_indices` silently replaced that empty
    list with whatever GPU the scheduler had pre-assigned — Python treats
    [] as falsy, so the "or" always preferred the scheduler's guess over
    the backend's actual, authoritative report. state.current_gpu must
    reflect what the worker actually reported, never a scheduler guess."""
    wp = WorkerPool()
    backend = _PiperStyleBackend()
    wp.register_backend("mock-gpu-managed", backend)
    scheduler = AsyncMock()
    scheduler.find_gpu_for_model = AsyncMock(return_value=[1])
    mm = ModelManager(wp, gpu_scheduler=scheduler)

    with (
        patch("ocabra.core.model_manager.publish", new=AsyncMock()),
        patch("ocabra.core.model_manager.set_key", new=AsyncMock()),
    ):
        from ocabra.core.model_manager import ModelState

        mm._states["test/gpu-managed-but-cpu-fallback"] = ModelState(
            model_id="test/gpu-managed-but-cpu-fallback",
            display_name="test/gpu-managed-but-cpu-fallback",
            backend_type="mock-gpu-managed",
            load_policy=LoadPolicy.ON_DEMAND,
        )
        mm._load_locks["test/gpu-managed-but-cpu-fallback"] = asyncio.Lock()

        state = await mm.load("test/gpu-managed-but-cpu-fallback")

    # The scheduler WAS consulted (this backend isn't in the gpu_managed
    # exclusion list) and assigned GPU 1 — but the worker's own report of
    # gpu_indices=[] must win.
    scheduler.find_gpu_for_model.assert_awaited()
    assert backend.received_gpu_indices == [1]
    assert state.current_gpu == []
    assert state.ram_used_mb > 0


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
    assert state.ram_used_mb == 0
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
    assert merged["whisper"]["diarization_enabled"] is True


def test_build_diarized_extra_config_wins_over_a_snake_case_base() -> None:
    """Regression: the base whisper section is snake_case since migration 0034;
    a camelCase override lost against ``diarization_enabled: false`` because the
    reader checks snake_case first, so diarized profiles stopped diarizing."""
    from ocabra.backends.whisper_backend import _get_whisper_option
    from ocabra.core.model_manager_helpers import build_diarized_extra_config

    merged = build_diarized_extra_config({"whisper": {"diarization_enabled": False, "diarizationEnabled": False}})

    assert _get_whisper_option(merged, "diarization_enabled") is True
    assert "diarizationEnabled" not in merged["whisper"]


@pytest.mark.asyncio
async def test_cuda_oom_load_failure_logs_vram_snapshot(worker_pool) -> None:
    """A CUDA OOM after admission said "fits" must leave a per-GPU record."""
    from ocabra.core.gpu_manager import GPUProcessInfo

    gpu_manager = AsyncMock()
    gpu_manager.get_all_states.return_value = [
        SimpleNamespace(
            index=0,
            free_vram_mb=900,
            processes=[GPUProcessInfo(pid=55, process_name="x", process_type="compute", used_vram_mb=0)],
        )
    ]
    gpu_manager.locks_snapshot = MagicMock(return_value={"test/resident": 2_800})
    gpu_manager.get_process_vram_mb = MagicMock(return_value={55: 3_100, 99: 400})
    mm = ModelManager(worker_pool, gpu_manager=gpu_manager)
    resident = await add_test_model(mm, model_id="test/resident")
    resident.worker_info = WorkerInfo(
        backend_type="mock",
        model_id="test/resident",
        gpu_indices=[0],
        port=18001,
        pid=55,
        vram_used_mb=2_800,
    )

    with patch("ocabra.core.model_manager.logger") as log:
        await mm._log_cuda_oom_snapshot("test/new")

    event, kwargs = log.error.call_args.args[0], log.error.call_args.kwargs
    assert event == "model_load_cuda_oom_snapshot"
    assert kwargs["gpus"] == [
        {
            "gpu": 0,
            "nvml_free_mb": 900,
            "locks_mb": {"test/resident": 2_800},
            "processes_mb": {"test/resident": 3_100, "pid:99": 400},
        }
    ]
