from unittest.mock import AsyncMock, patch

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
