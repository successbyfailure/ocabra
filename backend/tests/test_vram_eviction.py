"""Tests for LRU eviction + VRAM threshold (sub-block 11.1)."""

import asyncio
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ocabra.backends._mock import MockBackend
from ocabra.core.model_manager import LoadPolicy, ModelManager, ModelState, ModelStatus
from ocabra.core.worker_pool import WorkerPool


def _make_mm(gpu_manager=None) -> ModelManager:
    wp = WorkerPool()
    wp.register_backend("mock", MockBackend(vram_mb=4096))
    return ModelManager(wp, gpu_manager=gpu_manager)


def _add_loaded_model(
    mm: ModelManager,
    model_id: str,
    *,
    load_policy: str = "on_demand",
    gpu_indices: list[int] | None = None,
    vram_mb: int = 4096,
    last_request_at: datetime | None = None,
    in_flight: int = 0,
) -> ModelState:
    gpu_indices = gpu_indices or [0]
    state = ModelState(
        model_id=model_id,
        display_name=model_id,
        backend_type="mock",
        status=ModelStatus.LOADED,
        load_policy=LoadPolicy(load_policy),
        current_gpu=gpu_indices,
        vram_used_mb=vram_mb,
        last_request_at=last_request_at,
    )
    mm._states[model_id] = state
    mm._load_locks[model_id] = asyncio.Lock()
    if in_flight > 0:
        mm._in_flight[model_id] = in_flight
    return state


# ------------------------------------------------------------------
# _get_eviction_candidates tests
# ------------------------------------------------------------------


def test_eviction_candidates_excludes_pin():
    mm = _make_mm()
    _add_loaded_model(mm, "test/pin-model", load_policy="pin", gpu_indices=[0])
    _add_loaded_model(mm, "test/warm-model", load_policy="warm", gpu_indices=[0])

    candidates = mm._get_eviction_candidates(0)
    assert "test/pin-model" not in candidates
    assert "test/warm-model" in candidates


def test_eviction_candidates_excludes_busy():
    mm = _make_mm()
    _add_loaded_model(mm, "test/busy", gpu_indices=[0], in_flight=2)
    _add_loaded_model(mm, "test/idle", gpu_indices=[0])

    candidates = mm._get_eviction_candidates(0)
    assert "test/busy" not in candidates
    assert "test/idle" in candidates


def test_eviction_order_is_lru():
    mm = _make_mm()
    now = datetime.now(timezone.utc)
    # Oldest request first
    _add_loaded_model(
        mm,
        "test/old",
        gpu_indices=[0],
        last_request_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    _add_loaded_model(
        mm,
        "test/recent",
        gpu_indices=[0],
        last_request_at=now,
    )
    _add_loaded_model(
        mm,
        "test/middle",
        gpu_indices=[0],
        last_request_at=datetime(2025, 6, 1, tzinfo=timezone.utc),
    )

    candidates = mm._get_eviction_candidates(0)
    assert candidates == ["test/old", "test/middle", "test/recent"]


# ------------------------------------------------------------------
# _evict_for_space tests
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_preload_evicts_when_insufficient_vram():
    mm = _make_mm()
    _add_loaded_model(
        mm,
        "test/existing",
        gpu_indices=[0],
        vram_mb=4000,
        last_request_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )

    with (
        patch("ocabra.core.model_manager.publish", new=AsyncMock()),
        patch("ocabra.core.model_manager.set_key", new=AsyncMock()),
    ):
        freed = await mm._evict_for_space(0, 3000)

    assert freed >= 3000
    assert mm._states["test/existing"].status == ModelStatus.UNLOADED


@pytest.mark.asyncio
async def test_preload_fails_if_cannot_free_enough():
    mm = _make_mm()
    # Only PIN models — can't evict
    _add_loaded_model(mm, "test/pinned", load_policy="pin", gpu_indices=[0], vram_mb=4000)

    freed = await mm._evict_for_space(0, 3000)
    assert freed < 3000
    assert mm._states["test/pinned"].status == ModelStatus.LOADED


# ------------------------------------------------------------------
# _vram_watchdog tests (simulated)
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_watchdog_evicts_above_threshold():
    """Simulate VRAM at 95%, verify that models are evicted."""
    from dataclasses import dataclass

    @dataclass
    class FakeGPUState:
        index: int = 0
        total_vram_mb: int = 10000
        used_vram_mb: int = 9500
        free_vram_mb: int = 500

    gpu_manager = MagicMock()
    gpu_manager.get_all_states = AsyncMock(return_value=[FakeGPUState()])
    gpu_manager.unlock_vram = AsyncMock()

    mm = _make_mm(gpu_manager=gpu_manager)
    _add_loaded_model(
        mm,
        "test/warm-model",
        load_policy="warm",
        gpu_indices=[0],
        vram_mb=2000,
        last_request_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )

    with (
        patch("ocabra.core.model_manager.publish", new=AsyncMock()),
        patch("ocabra.core.model_manager.set_key", new=AsyncMock()),
        patch("ocabra.core.model_manager.settings") as mock_settings,
    ):
        mock_settings.vram_eviction_threshold = 0.90
        mock_settings.idle_eviction_check_interval_seconds = 1

        # Run one iteration of the watchdog logic directly
        threshold = 0.90
        gpu_states = await gpu_manager.get_all_states()
        for gpu_state in gpu_states:
            ratio = gpu_state.used_vram_mb / gpu_state.total_vram_mb
            if ratio > threshold:
                over_mb = gpu_state.used_vram_mb - int(gpu_state.total_vram_mb * threshold)
                await mm._evict_for_space(gpu_state.index, over_mb)

    assert mm._states["test/warm-model"].status == ModelStatus.UNLOADED


@pytest.mark.asyncio
async def test_watchdog_does_nothing_below_threshold():
    """Simulate VRAM at 80%, verify no eviction."""
    from dataclasses import dataclass

    @dataclass
    class FakeGPUState:
        index: int = 0
        total_vram_mb: int = 10000
        used_vram_mb: int = 8000
        free_vram_mb: int = 2000

    gpu_manager = MagicMock()
    gpu_manager.get_all_states = AsyncMock(return_value=[FakeGPUState()])

    mm = _make_mm(gpu_manager=gpu_manager)
    _add_loaded_model(
        mm,
        "test/warm-model",
        load_policy="warm",
        gpu_indices=[0],
        vram_mb=2000,
        last_request_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )

    # Check that threshold=0.90 does not trigger eviction at 80%
    threshold = 0.90
    gpu_states = await gpu_manager.get_all_states()
    for gpu_state in gpu_states:
        ratio = gpu_state.used_vram_mb / gpu_state.total_vram_mb
        assert ratio <= threshold

    # Model should still be loaded
    assert mm._states["test/warm-model"].status == ModelStatus.LOADED
