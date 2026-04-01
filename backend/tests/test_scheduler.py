from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ocabra.core.scheduler import GPUScheduler, InsufficientVRAMError


def make_gpu_manager(free_by_gpu: dict[int, int]):
    """Create a mock GPUManager with given free VRAM per GPU."""
    from ocabra.core.gpu_manager import GPUState

    states = [
        GPUState(
            index=i,
            name=f"GPU-{i}",
            total_vram_mb=24000,
            free_vram_mb=free,
            used_vram_mb=0,
            utilization_pct=0,
            temperature_c=40,
            power_draw_w=50,
            power_limit_w=370,
            locked_vram_mb=0,
        )
        for i, free in free_by_gpu.items()
    ]

    gm = AsyncMock()
    gm.get_all_states = AsyncMock(return_value=states)
    gm.get_free_vram = AsyncMock(side_effect=lambda i: free_by_gpu[i])
    return gm


def make_gpu_manager_with_totals(free_by_gpu: dict[int, int], total_by_gpu: dict[int, int]):
    from ocabra.core.gpu_manager import GPUState

    states = [
        GPUState(
            index=i,
            name=f"GPU-{i}",
            total_vram_mb=total_by_gpu[i],
            free_vram_mb=free,
            used_vram_mb=0,
            utilization_pct=0,
            temperature_c=40,
            power_draw_w=50,
            power_limit_w=370,
            locked_vram_mb=0,
        )
        for i, free in free_by_gpu.items()
    ]

    gm = AsyncMock()
    gm.get_all_states = AsyncMock(return_value=states)
    gm.get_free_vram = AsyncMock(side_effect=lambda i: free_by_gpu[i])
    return gm


@pytest.mark.asyncio
async def test_assign_preferred_gpu():
    """Model fits on preferred GPU (1) → assigns [1]."""
    gm = make_gpu_manager({0: 3000, 1: 20000})
    scheduler = GPUScheduler(gm)
    result = await scheduler.find_gpu_for_model(8000, preferred_gpu=1)
    assert result == [1]


@pytest.mark.asyncio
async def test_fallback_to_other_gpu():
    """Preferred GPU (1) doesn't have enough VRAM, fallback to GPU 0."""
    gm = make_gpu_manager({0: 10000, 1: 2000})
    scheduler = GPUScheduler(gm)
    result = await scheduler.find_gpu_for_model(8000, preferred_gpu=1)
    assert result == [0]


@pytest.mark.asyncio
async def test_tensor_parallel_when_no_single_gpu_fits():
    """Model requires 30GB, no single GPU fits → tensor parallel [0, 1]."""
    gm = make_gpu_manager({0: 10000, 1: 22000})
    scheduler = GPUScheduler(gm)
    result = await scheduler.find_gpu_for_model(30000)
    assert result == [0, 1]


@pytest.mark.asyncio
async def test_insufficient_vram_raises():
    """Not enough VRAM anywhere → InsufficientVRAMError."""
    gm = make_gpu_manager({0: 5000, 1: 8000})
    scheduler = GPUScheduler(gm)
    with pytest.raises(InsufficientVRAMError):
        await scheduler.find_gpu_for_model(50000)


@pytest.mark.asyncio
async def test_uses_default_gpu_when_no_preferred():
    """No preferred GPU → uses settings.default_gpu_index."""
    from unittest.mock import patch
    from ocabra import config

    gm = make_gpu_manager({0: 5000, 1: 20000})
    scheduler = GPUScheduler(gm)

    with patch.object(config.settings, "default_gpu_index", 1):
        result = await scheduler.find_gpu_for_model(8000, preferred_gpu=None)
    assert result == [1]


@pytest.mark.asyncio
async def test_tensor_parallel_skips_gpu_without_vllm_headroom():
    """With vLLM headroom policy, low-free GPU is excluded from TP candidates."""
    # GPU0 (12GB total, 6GB free) fails vLLM 0.6 threshold (~7.2GB).
    # GPU1 (24GB total, 16GB free) passes threshold (~14.4GB).
    gm = make_gpu_manager_with_totals(
        free_by_gpu={0: 6000, 1: 16000},
        total_by_gpu={0: 12000, 1: 24000},
    )
    scheduler = GPUScheduler(gm)
    with patch("ocabra.core.scheduler.settings.vllm_gpu_memory_utilization", 0.6):
        result = await scheduler.find_gpu_for_model(
            12000,
            preferred_gpu=None,
            enforce_vllm_headroom=True,
        )
    assert result == [1]


@pytest.mark.asyncio
async def test_pressure_threshold_prefers_healthier_gpu_over_pressured_preferred_gpu():
    gm = make_gpu_manager_with_totals(
        free_by_gpu={0: 10000, 1: 9000},
        total_by_gpu={0: 12000, 1: 24000},
    )
    scheduler = GPUScheduler(gm)

    with patch("ocabra.core.scheduler.settings.vram_pressure_threshold_pct", 80.0):
        result = await scheduler.find_gpu_for_model(8000, preferred_gpu=1)

    assert result == [0]


@pytest.mark.asyncio
async def test_pressure_threshold_falls_back_to_pressured_gpu_when_needed():
    gm = make_gpu_manager_with_totals(
        free_by_gpu={0: 9800, 1: 15000},
        total_by_gpu={0: 12000, 1: 24000},
    )
    scheduler = GPUScheduler(gm)

    with patch("ocabra.core.scheduler.settings.vram_pressure_threshold_pct", 80.0):
        result = await scheduler.find_gpu_for_model(10000, preferred_gpu=0)

    assert result == [1]


@pytest.mark.asyncio
async def test_vllm_headroom_can_trigger_insufficient_even_if_raw_sum_fits():
    """
    Raw free sum can be enough, but if no eligible GPUs satisfy vLLM headroom,
    scheduler must fail early.
    """
    gm = make_gpu_manager_with_totals(
        free_by_gpu={0: 6000, 1: 9000},
        total_by_gpu={0: 12000, 1: 24000},
    )
    scheduler = GPUScheduler(gm)
    with patch("ocabra.core.scheduler.settings.vllm_gpu_memory_utilization", 0.6):
        with pytest.raises(InsufficientVRAMError):
            await scheduler.find_gpu_for_model(
                12000,
                preferred_gpu=None,
                enforce_vllm_headroom=True,
            )


@pytest.mark.asyncio
async def test_schedule_evictions_unload_due_warm_models():
    from types import SimpleNamespace

    from ocabra.core.model_manager import LoadPolicy, ModelStatus, ModelState

    gm = make_gpu_manager({0: 24000})
    model_manager = AsyncMock()
    scheduler = GPUScheduler(gm, model_manager=model_manager)
    state = ModelState(
        model_id="vllm/warm-model",
        display_name="Warm Model",
        backend_type="vllm",
        status=ModelStatus.LOADED,
        load_policy=LoadPolicy.WARM,
    )
    model_manager.list_states = AsyncMock(return_value=[state])
    scheduler._load_enabled_schedules = AsyncMock(
        return_value=[
            SimpleNamespace(
                id="sched-1",
                model_id=None,
                action="evict_warm",
                cron_expr="* * * * *",
            )
        ]
    )
    scheduler._is_schedule_due = MagicMock(return_value=True)

    await scheduler.check_schedule_evictions()

    model_manager.unload.assert_awaited_once_with(
        "vllm/warm-model",
        reason="schedule:evict_warm",
    )


@pytest.mark.asyncio
async def test_schedule_reloads_load_due_pin_models():
    from types import SimpleNamespace

    from ocabra.core.model_manager import LoadPolicy, ModelStatus, ModelState

    gm = make_gpu_manager({0: 24000})
    model_manager = AsyncMock()
    scheduler = GPUScheduler(gm, model_manager=model_manager)
    state = ModelState(
        model_id="vllm/pin-model",
        display_name="Pinned Model",
        backend_type="vllm",
        status=ModelStatus.UNLOADED,
        load_policy=LoadPolicy.PIN,
        auto_reload=True,
        preferred_gpu=1,
    )
    model_manager.list_states = AsyncMock(return_value=[state])
    scheduler._load_enabled_schedules = AsyncMock(
        return_value=[
            SimpleNamespace(
                id="sched-2",
                model_id=None,
                action="reload",
                cron_expr="* * * * *",
            )
        ]
    )
    scheduler._is_schedule_due = MagicMock(return_value=True)

    await scheduler.check_schedule_reloads()

    model_manager.load.assert_awaited_once_with(
        "vllm/pin-model",
        force_gpu=1,
    )


@pytest.mark.asyncio
async def test_schedule_reloads_skip_models_without_auto_reload():
    from types import SimpleNamespace

    from ocabra.core.model_manager import LoadPolicy, ModelStatus, ModelState

    gm = make_gpu_manager({0: 24000})
    model_manager = AsyncMock()
    scheduler = GPUScheduler(gm, model_manager=model_manager)
    state = ModelState(
        model_id="vllm/pin-model",
        display_name="Pinned Model",
        backend_type="vllm",
        status=ModelStatus.UNLOADED,
        load_policy=LoadPolicy.PIN,
        auto_reload=False,
        preferred_gpu=1,
    )
    model_manager.list_states = AsyncMock(return_value=[state])
    scheduler._load_enabled_schedules = AsyncMock(
        return_value=[
            SimpleNamespace(
                id="sched-3",
                model_id=None,
                action="reload",
                cron_expr="* * * * *",
            )
        ]
    )
    scheduler._is_schedule_due = MagicMock(return_value=True)

    await scheduler.check_schedule_reloads()

    model_manager.load.assert_not_awaited()


@pytest.mark.asyncio
async def test_load_enabled_schedules_reads_db_rows():
    from ocabra.db.model_config import global_schedule_payload_to_rows

    rows = global_schedule_payload_to_rows("weekday-window", [1, 2, 3], "02:00", "06:00", True)
    rows.append(
        global_schedule_payload_to_rows("disabled-window", [4], "03:00", "05:00", False)[0]
    )

    class FakeResult:
        def __init__(self, rows):
            self._rows = rows

        def scalars(self):
            return self

        def all(self):
            return list(self._rows)

    class FakeSession:
        def __init__(self, rows):
            self.rows = list(rows)

        async def execute(self, query):
            sql = str(query.compile(compile_kwargs={"literal_binds": True})).lower()
            filtered = [row for row in self.rows if row.enabled]
            if "action in ('reload')" in sql:
                filtered = [row for row in filtered if row.action == "reload"]
            elif "action in ('evict_all')" in sql:
                filtered = [row for row in filtered if row.action == "evict_all"]
            return FakeResult(filtered)

    class FakeSessionFactory:
        def __init__(self, rows):
            self.rows = rows

        def __call__(self):
            return self

        async def __aenter__(self):
            return FakeSession(self.rows)

        async def __aexit__(self, exc_type, exc, tb):
            return False

    gm = make_gpu_manager({0: 24000})
    scheduler = GPUScheduler(gm)

    with patch("ocabra.core.scheduler.AsyncSessionLocal", new=FakeSessionFactory(rows)):
        evictions = await scheduler._load_enabled_schedules(actions={"evict_all"})
        reloads = await scheduler._load_enabled_schedules(actions={"reload"})

    assert [row.action for row in evictions] == ["evict_all"]
    assert [row.action for row in reloads] == ["reload"]
