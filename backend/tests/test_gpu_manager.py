import asyncio
from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ocabra.core.gpu_manager import GPUManager


def _make_mock_nvml(gpu_count: int = 2, free_mb: int = 10000, used_mb: int = 2000):
    """Build a minimal pynvml mock."""
    mem = MagicMock()
    mem.total = (free_mb + used_mb) * 1024 * 1024
    mem.free = free_mb * 1024 * 1024
    mem.used = used_mb * 1024 * 1024

    util = MagicMock()
    util.gpu = 45

    handle = MagicMock()

    patches = {
        "pynvml.nvmlInit": MagicMock(),
        "pynvml.nvmlShutdown": MagicMock(),
        "pynvml.nvmlDeviceGetCount": MagicMock(return_value=gpu_count),
        "pynvml.nvmlDeviceGetHandleByIndex": MagicMock(return_value=handle),
        "pynvml.nvmlDeviceGetMemoryInfo": MagicMock(return_value=mem),
        "pynvml.nvmlDeviceGetUtilizationRates": MagicMock(return_value=util),
        "pynvml.nvmlDeviceGetTemperature": MagicMock(return_value=65),
        "pynvml.nvmlDeviceGetPowerUsage": MagicMock(return_value=150_000),
        "pynvml.nvmlDeviceGetPowerManagementLimit": MagicMock(return_value=370_000),
        "pynvml.nvmlDeviceGetName": MagicMock(return_value=b"RTX 3090"),
    }
    return patches


@pytest.fixture
def gpu_manager():
    return GPUManager()


@pytest.mark.asyncio
async def test_start_detects_gpus():
    patches = _make_mock_nvml(gpu_count=2)
    with ExitStack() as stack:
        stack.enter_context(patch("ocabra.core.gpu_manager.publish", new=AsyncMock()))
        stack.enter_context(patch("ocabra.core.gpu_manager.set_key", new=AsyncMock()))
        for k, v in patches.items():
            stack.enter_context(patch(k, v))

        gm = GPUManager()
        gm._poll_task = MagicMock()  # prevent real task
        # Start manually without the poll loop
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            gm._locks[i] = {}
            gm._states[i] = gm._read_gpu(i)

        states = await gm.get_all_states()
        assert len(states) == 2
        assert states[0].index == 0
        assert states[1].index == 1


@pytest.mark.asyncio
async def test_lock_vram_reduces_free():
    # Idle GPU: the locked model's memory is not visible in NVML yet.
    patches = _make_mock_nvml(free_mb=24000, used_mb=0)
    with ExitStack() as stack:
        stack.enter_context(patch("ocabra.core.gpu_manager.publish", new=AsyncMock()))
        stack.enter_context(patch("ocabra.core.gpu_manager.set_key", new=AsyncMock()))
        for k, v in patches.items():
            stack.enter_context(patch(k, v))

        gm = GPUManager()
        gm._locks = {0: {}}
        gm._states[0] = gm._read_gpu(0)

        free_before = await gm.get_free_vram(0)
        await gm.lock_vram(0, 5000, "my-model")
        free_after = await gm.get_free_vram(0)

        assert free_after == free_before - 5000


@pytest.mark.asyncio
async def test_unlock_vram_restores_free():
    patches = _make_mock_nvml(free_mb=24000, used_mb=0)
    with ExitStack() as stack:
        stack.enter_context(patch("ocabra.core.gpu_manager.publish", new=AsyncMock()))
        stack.enter_context(patch("ocabra.core.gpu_manager.set_key", new=AsyncMock()))
        for k, v in patches.items():
            stack.enter_context(patch(k, v))

        gm = GPUManager()
        gm._locks = {0: {}}
        gm._states[0] = gm._read_gpu(0)

        await gm.lock_vram(0, 5000, "my-model")
        free_locked = await gm.get_free_vram(0)
        await gm.unlock_vram(0, "my-model")
        free_unlocked = await gm.get_free_vram(0)

        assert free_unlocked == free_locked + 5000


@pytest.mark.asyncio
async def test_resident_model_lock_not_double_counted():
    """A loaded model's lock must not be subtracted again from NVML free memory."""
    from ocabra.config import settings

    # 6 GB card: 700 MB desktop + 2150 MB of an already-resident vLLM worker.
    patches = _make_mock_nvml(free_mb=3294, used_mb=2850)
    with ExitStack() as stack:
        stack.enter_context(patch("ocabra.core.gpu_manager.publish", new=AsyncMock()))
        stack.enter_context(patch("ocabra.core.gpu_manager.set_key", new=AsyncMock()))
        for k, v in patches.items():
            stack.enter_context(patch(k, v))

        gm = GPUManager()
        gm._locks = {0: {}}
        gm._states[0] = gm._read_gpu(0)
        await gm.lock_vram(0, 2150, "vllm/Qwen/Qwen3-Embedding-0.6B")

        assert await gm.get_free_vram(0) == 3294 - settings.vram_buffer_mb


@pytest.mark.asyncio
async def test_service_lock_always_subtracted():
    """Service reservations are taken before the service allocates, so they count."""
    from ocabra.config import settings

    patches = _make_mock_nvml(free_mb=10000, used_mb=2000)
    with ExitStack() as stack:
        stack.enter_context(patch("ocabra.core.gpu_manager.publish", new=AsyncMock()))
        stack.enter_context(patch("ocabra.core.gpu_manager.set_key", new=AsyncMock()))
        for k, v in patches.items():
            stack.enter_context(patch(k, v))

        gm = GPUManager()
        gm._locks = {0: {}}
        gm._states[0] = gm._read_gpu(0)
        await gm.lock_vram(0, 4000, "service:hunyuan")

        assert await gm.get_free_vram(0) == 10000 - 4000 - settings.vram_buffer_mb


@pytest.mark.asyncio
async def test_vram_buffer_always_reserved():
    """get_free_vram should subtract the configured buffer."""
    from ocabra.config import settings

    patches = _make_mock_nvml(free_mb=10000, used_mb=0)
    with ExitStack() as stack:
        stack.enter_context(patch("ocabra.core.gpu_manager.publish", new=AsyncMock()))
        stack.enter_context(patch("ocabra.core.gpu_manager.set_key", new=AsyncMock()))
        for k, v in patches.items():
            stack.enter_context(patch(k, v))

        gm = GPUManager()
        gm._locks = {0: {}}
        gm._states[0] = gm._read_gpu(0)

        free = await gm.get_free_vram(0)
        assert free == 10000 - settings.vram_buffer_mb

@pytest.mark.asyncio
async def test_read_gpu_includes_running_processes() -> None:
    patches = _make_mock_nvml(gpu_count=1)

    proc_compute = MagicMock()
    proc_compute.pid = 4321
    proc_compute.usedGpuMemory = 1024 * 1024 * 1536

    proc_graphics = MagicMock()
    proc_graphics.pid = 9876
    proc_graphics.usedGpuMemory = 1024 * 1024 * 256

    patches.update(
        {
            "pynvml.nvmlDeviceGetComputeRunningProcesses_v3": MagicMock(return_value=[proc_compute]),
            "pynvml.nvmlDeviceGetGraphicsRunningProcesses_v3": MagicMock(return_value=[proc_graphics]),
        }
    )

    with ExitStack() as stack:
        stack.enter_context(patch("ocabra.core.gpu_manager.publish", new=AsyncMock()))
        stack.enter_context(patch("ocabra.core.gpu_manager.set_key", new=AsyncMock()))
        for k, v in patches.items():
            stack.enter_context(patch(k, v))

        gm = GPUManager()
        with patch.object(gm, "_read_process_name", side_effect=["python", "Xorg"]):
            state = gm._read_gpu(0)

    assert len(state.processes) == 2
    assert state.processes[0].pid == 4321
    assert state.processes[0].process_name == "python"
    assert state.processes[0].process_type == "compute"
    assert state.processes[0].used_vram_mb == 1536
    assert state.processes[1].pid == 9876
    assert state.processes[1].process_type == "graphics"


def _process(pid: int, mb: int) -> MagicMock:
    proc = MagicMock()
    proc.pid = pid
    proc.usedGpuMemory = mb * 1024 * 1024
    return proc


def _gm_with_processes(stack: ExitStack, *, free_mb: int, used_mb: int, procs: list) -> GPUManager:
    patches = _make_mock_nvml(gpu_count=1, free_mb=free_mb, used_mb=used_mb)
    patches.update(
        {
            "pynvml.nvmlDeviceGetComputeRunningProcesses_v3": MagicMock(return_value=procs),
            "pynvml.nvmlDeviceGetGraphicsRunningProcesses_v3": MagicMock(return_value=[]),
        }
    )
    stack.enter_context(patch("ocabra.core.gpu_manager.publish", new=AsyncMock()))
    stack.enter_context(patch("ocabra.core.gpu_manager.set_key", new=AsyncMock()))
    for k, v in patches.items():
        stack.enter_context(patch(k, v))
    gm = GPUManager()
    gm._nvml_available = True
    gm._locks = {0: {}}
    stack.enter_context(patch.object(gm, "_read_process_name", return_value="llama-server"))
    gm._states[0] = gm._read_gpu(0)
    return gm


@pytest.mark.asyncio
async def test_free_vram_if_released_counts_real_footprint_over_lock() -> None:
    """2026-11-09: ravenx locked at its 23,237 MB estimate but really held
    23,594 MB. Releasing it must give back what the process holds."""
    from ocabra.config import settings

    with ExitStack() as stack:
        gm = _gm_with_processes(
            stack, free_mb=53, used_mb=24_523, procs=[_process(4321, 23_594)]
        )
        await gm.lock_vram(0, 23_237, "llamacpp_mks/ravenx")

        projected = await gm.get_free_vram_if_released(0, {"llamacpp_mks/ravenx": 4321})

    assert projected == 53 + 23_594 - settings.vram_buffer_mb


@pytest.mark.asyncio
async def test_free_vram_if_released_uses_lock_when_pid_holds_less() -> None:
    """vLLM allocates in a child process: a small reading for the recorded PID
    must not shrink the bound below the model's lock."""
    from ocabra.config import settings

    with ExitStack() as stack:
        gm = _gm_with_processes(stack, free_mb=1_000, used_mb=11_000, procs=[_process(77, 400)])
        await gm.lock_vram(0, 10_000, "vllm/model")

        projected = await gm.get_free_vram_if_released(0, {"vllm/model": 77})

    assert projected == 1_000 + 10_000 - settings.vram_buffer_mb


@pytest.mark.asyncio
async def test_free_vram_if_released_keeps_other_models_and_services() -> None:
    from ocabra.config import settings

    with ExitStack() as stack:
        gm = _gm_with_processes(
            stack,
            free_mb=2_000,
            used_mb=10_000,
            procs=[_process(1, 4_000), _process(2, 6_000)],
        )
        await gm.lock_vram(0, 4_000, "a")
        await gm.lock_vram(0, 6_000, "b")
        await gm.lock_vram(0, 1_500, "service:trellis2")

        projected = await gm.get_free_vram_if_released(0, {"a": 1})

    assert projected == 2_000 + 4_000 - 1_500 - settings.vram_buffer_mb


@pytest.mark.asyncio
async def test_poll_replaces_hw_monitor_processes_with_local_reading() -> None:
    """hw-monitor's container sees no processes (NVML filters by PID
    namespace); the API's own reading lists its workers under their PIDs."""
    from ocabra.core.gpu_manager import GPUState

    with ExitStack() as stack:
        gm = _gm_with_processes(stack, free_mb=5_000, used_mb=7_000, procs=[_process(19647, 6_800)])
        redis_state = GPUState(
            index=0,
            name="RTX 3090",
            total_vram_mb=12_000,
            free_vram_mb=5_000,
            used_vram_mb=7_000,
            utilization_pct=0.0,
            temperature_c=40.0,
            power_draw_w=100.0,
            power_limit_w=300.0,
            locked_vram_mb=0,
            processes=[],
        )
        stack.enter_context(
            patch.object(gm, "_read_gpu_from_redis", new=AsyncMock(return_value=redis_state))
        )
        stack.enter_context(patch.object(gm, "_check_thermal_throttle", new=AsyncMock()))
        sleep = AsyncMock(side_effect=asyncio.CancelledError)
        stack.enter_context(patch("ocabra.core.gpu_manager.asyncio.sleep", sleep))

        with pytest.raises(asyncio.CancelledError):
            await gm._poll_loop()

        assert [(p.pid, p.used_vram_mb) for p in gm._states[0].processes] == [(19647, 6_800)]
        assert gm.get_process_vram_mb(0) == {19647: 6_800}


@pytest.mark.asyncio
async def test_physical_free_vram_ignores_inflated_model_locks() -> None:
    """2026-11-10: growing qwen3.8 back evicted whisper for 446 MB because
    whisper locks 3,500 MB while holding ~390. Model locks must not count."""
    from ocabra.config import settings

    with ExitStack() as stack:
        gm = _gm_with_processes(stack, free_mb=9_000, used_mb=3_000, procs=[_process(5, 390)])
        await gm.lock_vram(0, 3_500, "whisper/large")
        await gm.lock_vram(0, 1_000, "service:trellis2")

        physical = await gm.get_physical_free_vram(0)
        scheduled = await gm.get_free_vram(0)

    assert physical == 9_000 - 1_000 - settings.vram_buffer_mb
    assert scheduled < physical
