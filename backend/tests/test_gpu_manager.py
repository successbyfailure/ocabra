from contextlib import ExitStack
from dataclasses import asdict
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
            gm._poll_history[i] = []
            gm._states[i] = gm._read_gpu(i)

        states = await gm.get_all_states()
        assert len(states) == 2
        assert states[0].index == 0
        assert states[1].index == 1


@pytest.mark.asyncio
async def test_lock_vram_reduces_free():
    patches = _make_mock_nvml(free_mb=20000, used_mb=4000)
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
    patches = _make_mock_nvml(free_mb=20000, used_mb=4000)
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
