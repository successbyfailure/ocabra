"""Returning a model to its preferred GPU once that GPU frees up (2026-11-06).

Placement is decided from whatever is free at load time, and nothing revisited
it: a model whose preferred GPU was briefly full landed elsewhere and stayed.
Observed in production — whisper (preferred_gpu=0, PIN) ran 17 minutes on GPU 1
while GPU 0 sat empty.

A move costs an unload plus a cold start, so almost every test here is about a
condition that must *prevent* one.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from ocabra.core.model_manager import LoadPolicy, ModelManager, ModelState, ModelStatus
from ocabra.core.worker_pool import WorkerPool


def _gpu_manager(free_by_gpu: dict[int, int], locked: set[int] | None = None) -> MagicMock:
    gm = MagicMock()
    gm.get_free_vram = AsyncMock(side_effect=lambda idx: free_by_gpu.get(idx, 0))
    gm.has_service_lock = MagicMock(side_effect=lambda idx: idx in (locked or set()))
    return gm


def _manager(free_by_gpu=None, **state_kwargs) -> tuple[ModelManager, ModelState]:
    mm = ModelManager(WorkerPool(), gpu_manager=_gpu_manager(free_by_gpu or {0: 8000, 1: 8000}))
    defaults = dict(
        model_id="whisper/turbo",
        display_name="whisper",
        backend_type="whisper",
        load_policy=LoadPolicy.PIN,
        status=ModelStatus.LOADED,
        preferred_gpu=0,
        current_gpu=[1],
        vram_estimate_mb=3500,
        loaded_at=datetime.now(UTC) - timedelta(hours=1),
    )
    defaults.update(state_kwargs)
    state = ModelState(**defaults)
    mm._states[state.model_id] = state
    mm.unload = AsyncMock()
    mm.load = AsyncMock()
    return mm, state


@pytest.mark.asyncio
async def test_moves_back_when_the_preferred_gpu_has_room() -> None:
    mm, state = _manager({0: 8000, 1: 8000})

    await mm.check_preferred_gpu_replacements()

    mm.unload.assert_awaited_once()
    mm.load.assert_awaited_once()
    assert mm.load.await_args.kwargs["force_gpu"] == 0


@pytest.mark.asyncio
async def test_does_not_move_when_the_preferred_gpu_cannot_hold_it() -> None:
    """Only moves when the GPU already fits it — never by evicting something,
    which is exactly the cascade the scheduler's pressure heuristic avoids."""
    mm, _ = _manager({0: 1000, 1: 8000})

    await mm.check_preferred_gpu_replacements()

    mm.unload.assert_not_awaited()


@pytest.mark.asyncio
async def test_does_not_move_a_busy_model() -> None:
    mm, state = _manager()
    mm.try_begin_request(state.model_id, max_inflight=32)

    await mm.check_preferred_gpu_replacements()

    mm.unload.assert_not_awaited()


@pytest.mark.asyncio
async def test_does_not_move_while_another_load_is_running() -> None:
    """Never compete with real traffic for a GPU."""
    mm, _ = _manager()
    mm._load_in_progress.add("some/other-model")

    await mm.check_preferred_gpu_replacements()

    mm.unload.assert_not_awaited()


@pytest.mark.asyncio
async def test_ignores_tensor_split_models() -> None:
    """A model spread over several GPUs has no single GPU to return to."""
    mm, _ = _manager(current_gpu=[0, 1])

    await mm.check_preferred_gpu_replacements()

    mm.unload.assert_not_awaited()


@pytest.mark.asyncio
async def test_ignores_a_forced_runtime_preset() -> None:
    """Forced placement was chosen explicitly and is already honoured."""
    mm, _ = _manager(extra_config={"_runtime": {"gpu_indices": [1]}})

    await mm.check_preferred_gpu_replacements()

    mm.unload.assert_not_awaited()


@pytest.mark.asyncio
async def test_respects_minimum_residency() -> None:
    mm, _ = _manager(loaded_at=datetime.now(UTC))

    await mm.check_preferred_gpu_replacements()

    mm.unload.assert_not_awaited()


@pytest.mark.asyncio
async def test_cooldown_prevents_ping_pong() -> None:
    """Two GPUs taking turns at being free must not bounce a model forever."""
    mm, state = _manager()
    await mm.check_preferred_gpu_replacements()
    assert mm.unload.await_count == 1

    state.current_gpu = [1]  # pretend it ended up off-preference again
    await mm.check_preferred_gpu_replacements()
    assert mm.unload.await_count == 1, "second move should be blocked by the cooldown"


@pytest.mark.asyncio
async def test_kill_switch(monkeypatch) -> None:
    monkeypatch.setattr(
        "ocabra.core.model_manager.settings.preferred_gpu_replacement_enabled", False
    )
    mm, _ = _manager()

    await mm.check_preferred_gpu_replacements()

    mm.unload.assert_not_awaited()


@pytest.mark.asyncio
async def test_skips_a_gpu_reserved_by_an_external_service() -> None:
    mm = ModelManager(WorkerPool(), gpu_manager=_gpu_manager({0: 8000, 1: 8000}, locked={0}))
    mm._states["whisper/turbo"] = ModelState(
        model_id="whisper/turbo",
        display_name="whisper",
        backend_type="whisper",
        load_policy=LoadPolicy.PIN,
        status=ModelStatus.LOADED,
        preferred_gpu=0,
        current_gpu=[1],
        vram_estimate_mb=3500,
        loaded_at=datetime.now(UTC) - timedelta(hours=1),
    )
    mm.unload = AsyncMock()

    await mm.check_preferred_gpu_replacements()

    mm.unload.assert_not_awaited()


@pytest.mark.asyncio
async def test_only_one_move_per_pass() -> None:
    """The next tick re-reads real free VRAM instead of acting on figures the
    first move just invalidated."""
    mm, _ = _manager()
    for i in (2, 3):
        mm._states[f"whisper/other-{i}"] = ModelState(
            model_id=f"whisper/other-{i}",
            display_name=f"other-{i}",
            backend_type="whisper",
            load_policy=LoadPolicy.PIN,
            status=ModelStatus.LOADED,
            preferred_gpu=0,
            current_gpu=[1],
            vram_estimate_mb=3500,
            loaded_at=datetime.now(UTC) - timedelta(hours=1),
        )

    await mm.check_preferred_gpu_replacements()

    assert mm.unload.await_count == 1
