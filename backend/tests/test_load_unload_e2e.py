"""E2E load/unload flow tests for ModelManager.

Tests the full lifecycle: configure → load → verify LOADED → unload → verify UNLOADED,
plus error cases, pressure eviction, and policy-specific behaviors.
All external dependencies (GPU, DB, Redis) are mocked.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ocabra.backends._mock import MockBackend
from ocabra.backends.base import BackendCapabilities, BackendInterface, WorkerInfo
from ocabra.core.model_manager import LoadPolicy, ModelManager, ModelState, ModelStatus
from ocabra.core.worker_pool import WorkerPool


# ── Helpers ──────────────────────────────────────────────────────


def _make_worker_pool(vram_mb: int = 4096) -> WorkerPool:
    wp = WorkerPool()
    wp.register_backend("mock", MockBackend(vram_mb=vram_mb))
    return wp


def _make_manager(
    worker_pool: WorkerPool | None = None,
    gpu_manager=None,
    gpu_scheduler=None,
) -> ModelManager:
    wp = worker_pool or _make_worker_pool()
    return ModelManager(wp, gpu_manager=gpu_manager, gpu_scheduler=gpu_scheduler)


async def _add_model(
    mm: ModelManager,
    model_id: str = "mock/test-model",
    load_policy: str = "on_demand",
    auto_reload: bool = False,
    preferred_gpu: int | None = None,
    extra_config: dict | None = None,
) -> ModelState:
    """Add a model directly into the manager's state (bypasses DB)."""
    state = ModelState(
        model_id=model_id,
        display_name=model_id,
        backend_type="mock",
        load_policy=LoadPolicy(load_policy),
        auto_reload=auto_reload,
        preferred_gpu=preferred_gpu,
        extra_config=extra_config or {},
    )
    mm._states[model_id] = state
    mm._load_locks[model_id] = asyncio.Lock()
    return state


def _patch_externals():
    """Patch Redis publish/set_key and DB writes used during load/unload."""
    return (
        patch("ocabra.core.model_manager.publish", new=AsyncMock()),
        patch("ocabra.core.model_manager.set_key", new=AsyncMock()),
        patch("ocabra.core.model_manager.get_key", new=AsyncMock(return_value=None)),
        patch("ocabra.core.model_manager.publish_system_alert", new=AsyncMock()),
    )


# ── Full cycle ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_full_load_unload_cycle():
    """CONFIGURED → LOADING → LOADED → UNLOADING → UNLOADED."""
    mm = _make_manager()
    p1, p2, p3, p4 = _patch_externals()
    with p1, p2, p3, p4:
        await _add_model(mm)
        state = mm._states["mock/test-model"]
        assert state.status == ModelStatus.CONFIGURED

        loaded = await mm.load("mock/test-model")
        assert loaded.status == ModelStatus.LOADED
        assert loaded.vram_used_mb == 4096
        assert loaded.loaded_at is not None
        assert loaded.worker_info is not None
        assert loaded.capabilities.chat is True

        await mm.unload("mock/test-model")
        state = await mm.get_state("mock/test-model")
        assert state.status == ModelStatus.UNLOADED
        assert state.vram_used_mb == 0
        assert state.loaded_at is None
        assert state.worker_info is None


# ── Error cases ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_load_failure_sets_error_state():
    """When the backend.load() raises, the model state becomes ERROR."""
    wp = WorkerPool()
    failing_backend = AsyncMock(spec=BackendInterface)
    failing_backend.get_vram_estimate_mb = AsyncMock(return_value=2048)
    failing_backend.load = AsyncMock(side_effect=RuntimeError("GPU exploded"))
    wp.register_backend("mock", failing_backend)

    mm = _make_manager(worker_pool=wp)
    p1, p2, p3, p4 = _patch_externals()
    with p1, p2, p3, p4:
        await _add_model(mm)

        with pytest.raises(RuntimeError, match="GPU exploded"):
            await mm.load("mock/test-model")

        state = await mm.get_state("mock/test-model")
        assert state.status == ModelStatus.ERROR
        assert "GPU exploded" in state.error_message


@pytest.mark.asyncio
async def test_double_load_is_idempotent():
    """Loading an already-loaded model returns immediately without error."""
    mm = _make_manager()
    p1, p2, p3, p4 = _patch_externals()
    with p1, p2, p3, p4:
        await _add_model(mm)

        state1 = await mm.load("mock/test-model")
        assert state1.status == ModelStatus.LOADED

        state2 = await mm.load("mock/test-model")
        assert state2.status == ModelStatus.LOADED
        # Should be the same object — no second load occurred
        assert state1 is state2


@pytest.mark.asyncio
async def test_load_unknown_model_raises():
    """Attempting to load a model that was never configured raises KeyError."""
    mm = _make_manager()
    p1, p2, p3, p4 = _patch_externals()
    with p1, p2, p3, p4:
        with pytest.raises(KeyError, match="not configured"):
            await mm.load("mock/nonexistent")


@pytest.mark.asyncio
async def test_unload_not_loaded_is_noop():
    """Unloading a model that isn't loaded does nothing (no error)."""
    mm = _make_manager()
    p1, p2, p3, p4 = _patch_externals()
    with p1, p2, p3, p4:
        await _add_model(mm)
        # Model is CONFIGURED, not LOADED — unload should be a no-op
        await mm.unload("mock/test-model")
        state = await mm.get_state("mock/test-model")
        assert state.status == ModelStatus.CONFIGURED


@pytest.mark.asyncio
async def test_unload_failure_sets_error_state():
    """When backend.unload() raises, the model transitions to ERROR."""
    wp = WorkerPool()
    bad_backend = AsyncMock(spec=BackendInterface)
    bad_backend.get_vram_estimate_mb = AsyncMock(return_value=2048)
    bad_backend.load = AsyncMock(
        return_value=WorkerInfo(
            backend_type="mock", model_id="test-model",
            gpu_indices=[], port=9999, pid=1, vram_used_mb=2048,
        )
    )
    bad_backend.get_capabilities = AsyncMock(return_value=BackendCapabilities(chat=True))
    bad_backend.unload = AsyncMock(side_effect=RuntimeError("Cannot kill process"))
    wp.register_backend("mock", bad_backend)

    mm = _make_manager(worker_pool=wp)
    p1, p2, p3, p4 = _patch_externals()
    with p1, p2, p3, p4:
        await _add_model(mm)
        await mm.load("mock/test-model")

        with pytest.raises(RuntimeError, match="Cannot kill process"):
            await mm.unload("mock/test-model")

        state = await mm.get_state("mock/test-model")
        assert state.status == ModelStatus.ERROR


# ── Pressure eviction ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_pressure_eviction_evicts_on_demand_first():
    """When VRAM is insufficient, on_demand models are evicted before warm/pin."""
    wp = _make_worker_pool(vram_mb=2048)

    # GPU scheduler: first call fails (not enough VRAM), second succeeds (after eviction)
    gpu_scheduler = AsyncMock()
    from ocabra.core.scheduler import InsufficientVRAMError
    call_count = 0

    async def _find_gpu(vram_needed, preferred_gpu=None, enforce_vllm_headroom=False):
        nonlocal call_count
        call_count += 1
        if call_count <= 1:
            raise InsufficientVRAMError("No VRAM")
        return [0]

    gpu_scheduler.find_gpu_for_model = _find_gpu

    mm = _make_manager(worker_pool=wp, gpu_scheduler=gpu_scheduler)
    p1, p2, p3, p4 = _patch_externals()
    with p1, p2, p3, p4:
        # Load model A (on_demand) — manually put in LOADED state
        await _add_model(mm, model_id="mock/model-a", load_policy="on_demand")
        state_a = mm._states["mock/model-a"]
        state_a.status = ModelStatus.LOADED
        state_a.vram_used_mb = 2048
        state_a.worker_info = WorkerInfo(
            backend_type="mock", model_id="model-a",
            gpu_indices=[0], port=9000, pid=1, vram_used_mb=2048,
        )
        wp.set_worker("mock/model-a", state_a.worker_info)

        # Load model B — should trigger eviction of model A
        await _add_model(mm, model_id="mock/model-b", load_policy="on_demand")

        await mm.load("mock/model-b")

        state_a = await mm.get_state("mock/model-a")
        assert state_a.status == ModelStatus.UNLOADED

        state_b = await mm.get_state("mock/model-b")
        assert state_b.status == ModelStatus.LOADED


@pytest.mark.asyncio
async def test_eviction_order_on_demand_before_warm_before_pin():
    """Eviction candidates are sorted: on_demand < warm < pin."""
    mm = _make_manager()
    p1, p2, p3, p4 = _patch_externals()
    with p1, p2, p3, p4:
        # Add three models with different policies, all "loaded"
        for mid, policy in [
            ("mock/pinned", "pin"),
            ("mock/warm", "warm"),
            ("mock/on-demand", "on_demand"),
        ]:
            await _add_model(mm, model_id=mid, load_policy=policy)
            mm._states[mid].status = ModelStatus.LOADED
            mm._states[mid].vram_used_mb = 1024
            wi = WorkerInfo(
                backend_type="mock", model_id=mid.split("/")[1],
                gpu_indices=[0], port=9000, pid=1, vram_used_mb=1024,
            )
            mm._states[mid].worker_info = wi
            mm._worker_pool.set_worker(mid, wi)

        candidates = mm._get_pressure_eviction_candidates("mock/new-model")
        assert candidates[0] == "mock/on-demand"
        assert candidates[1] == "mock/warm"
        assert candidates[2] == "mock/pinned"


# ── Policy behaviors ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_pin_policy_auto_loads_on_start():
    """Pinned models are loaded automatically when start() is called."""
    mm = _make_manager()
    p1, p2, p3, p4 = _patch_externals()
    with p1, p2, p3, p4, \
         patch.object(mm, "_load_configs_from_db", new=AsyncMock()):
        await _add_model(mm, model_id="mock/pinned", load_policy="pin")
        await mm.start()
        # Let background task run
        await asyncio.sleep(0.15)

        state = await mm.get_state("mock/pinned")
        assert state.status == ModelStatus.LOADED


@pytest.mark.asyncio
async def test_warm_policy_does_not_auto_load():
    """Warm models are NOT auto-loaded on start (they wait for first request)."""
    mm = _make_manager()
    p1, p2, p3, p4 = _patch_externals()
    with p1, p2, p3, p4, \
         patch.object(mm, "_load_configs_from_db", new=AsyncMock()):
        await _add_model(mm, model_id="mock/warm", load_policy="warm")
        await mm.start()
        await asyncio.sleep(0.05)

        state = await mm.get_state("mock/warm")
        # warm is NOT auto-loaded; stays in CONFIGURED
        assert state.status in (ModelStatus.CONFIGURED, ModelStatus.UNLOADED)


@pytest.mark.asyncio
async def test_on_demand_does_not_auto_load():
    """on_demand models are NOT auto-loaded on start."""
    mm = _make_manager()
    p1, p2, p3, p4 = _patch_externals()
    with p1, p2, p3, p4, \
         patch.object(mm, "_load_configs_from_db", new=AsyncMock()):
        await _add_model(mm, model_id="mock/od", load_policy="on_demand")
        await mm.start()
        await asyncio.sleep(0.05)

        state = await mm.get_state("mock/od")
        assert state.status == ModelStatus.CONFIGURED


@pytest.mark.asyncio
async def test_auto_reload_triggers_watch_after_pressure_eviction():
    """When a model with auto_reload=True is pressure-evicted,
    _watch_and_reload is kicked off."""
    mm = _make_manager()
    p1, p2, p3, p4 = _patch_externals()
    with p1, p2, p3, p4:
        await _add_model(mm, model_id="mock/reloadable", load_policy="pin", auto_reload=True)
        state = mm._states["mock/reloadable"]
        state.status = ModelStatus.LOADED
        state.vram_used_mb = 2048
        wi = WorkerInfo(
            backend_type="mock", model_id="reloadable",
            gpu_indices=[0], port=9000, pid=1, vram_used_mb=2048,
        )
        state.worker_info = wi
        mm._worker_pool.set_worker("mock/reloadable", wi)

        # Track whether _watch_and_reload was triggered
        watch_called = False
        original_watch = mm._watch_and_reload

        async def _spy_watch(model_id):
            nonlocal watch_called
            watch_called = True
            # Don't actually run the watch loop in tests

        with patch.object(mm, "_watch_and_reload", side_effect=_spy_watch):
            await mm.unload("mock/reloadable", reason="pressure")
            await asyncio.sleep(0.05)

        assert watch_called


@pytest.mark.asyncio
async def test_auto_reload_not_triggered_on_manual_unload():
    """Manual unloads do NOT trigger _watch_and_reload even with auto_reload=True."""
    mm = _make_manager()
    p1, p2, p3, p4 = _patch_externals()
    with p1, p2, p3, p4:
        await _add_model(mm, model_id="mock/reloadable", load_policy="pin", auto_reload=True)
        state = mm._states["mock/reloadable"]
        state.status = ModelStatus.LOADED
        state.vram_used_mb = 2048
        wi = WorkerInfo(
            backend_type="mock", model_id="reloadable",
            gpu_indices=[0], port=9000, pid=1, vram_used_mb=2048,
        )
        state.worker_info = wi
        mm._worker_pool.set_worker("mock/reloadable", wi)

        watch_called = False

        async def _spy_watch(model_id):
            nonlocal watch_called
            watch_called = True

        with patch.object(mm, "_watch_and_reload", side_effect=_spy_watch):
            await mm.unload("mock/reloadable", reason="manual")
            await asyncio.sleep(0.05)

        assert not watch_called


# ── State transitions ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_state_transitions_during_load():
    """Verify the state passes through LOADING before reaching LOADED."""
    mm = _make_manager()
    observed_statuses = []

    original_publish = AsyncMock()

    async def _capture_publish(channel, payload):
        if isinstance(payload, dict) and "new_status" in payload:
            observed_statuses.append(payload["new_status"])

    p1, p2, p3, p4 = _patch_externals()
    with p1, p2, p3, p4:
        # Replace the publish mock with our capturing version
        with patch("ocabra.core.model_manager.publish", side_effect=_capture_publish):
            await _add_model(mm)
            await mm.load("mock/test-model")

    assert "loading" in observed_statuses
    assert "loaded" in observed_statuses
    # LOADING comes before LOADED
    assert observed_statuses.index("loading") < observed_statuses.index("loaded")


@pytest.mark.asyncio
async def test_worker_pool_tracks_worker_after_load():
    """After load, the worker pool should have the worker registered."""
    mm = _make_manager()
    p1, p2, p3, p4 = _patch_externals()
    with p1, p2, p3, p4:
        await _add_model(mm)
        await mm.load("mock/test-model")

        worker = mm._worker_pool.get_worker("mock/test-model")
        assert worker is not None
        assert worker.backend_type == "mock"

        await mm.unload("mock/test-model")
        worker = mm._worker_pool.get_worker("mock/test-model")
        assert worker is None


@pytest.mark.asyncio
async def test_load_after_error_can_retry():
    """A model in ERROR state can be loaded again successfully."""
    wp = WorkerPool()
    call_count = 0
    backend = AsyncMock(spec=BackendInterface)
    backend.get_vram_estimate_mb = AsyncMock(return_value=2048)

    async def _load_with_retry(model_id, gpu_indices, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("Temporary failure")
        return WorkerInfo(
            backend_type="mock", model_id=model_id,
            gpu_indices=gpu_indices, port=9999, pid=1, vram_used_mb=2048,
        )

    backend.load = _load_with_retry
    backend.get_capabilities = AsyncMock(return_value=BackendCapabilities(chat=True))
    wp.register_backend("mock", backend)

    mm = _make_manager(worker_pool=wp)
    p1, p2, p3, p4 = _patch_externals()
    with p1, p2, p3, p4:
        await _add_model(mm)

        # First load fails
        with pytest.raises(RuntimeError, match="Temporary failure"):
            await mm.load("mock/test-model")
        assert mm._states["mock/test-model"].status == ModelStatus.ERROR

        # Second load succeeds
        state = await mm.load("mock/test-model")
        assert state.status == ModelStatus.LOADED


@pytest.mark.asyncio
async def test_idle_eviction_only_affects_on_demand():
    """check_idle_evictions only evicts on_demand models, not warm or pin."""
    mm = _make_manager()
    p1, p2, p3, p4 = _patch_externals()
    with p1, p2, p3, p4:
        for mid, policy in [
            ("mock/od", "on_demand"),
            ("mock/warm", "warm"),
            ("mock/pinned", "pin"),
        ]:
            await _add_model(mm, model_id=mid, load_policy=policy)
            state = mm._states[mid]
            state.status = ModelStatus.LOADED
            # Set last_request_at to way in the past
            state.last_request_at = datetime(2020, 1, 1, tzinfo=timezone.utc)
            state.vram_used_mb = 1024
            wi = WorkerInfo(
                backend_type="mock", model_id=mid.split("/")[1],
                gpu_indices=[0], port=9000, pid=1, vram_used_mb=1024,
            )
            state.worker_info = wi
            mm._worker_pool.set_worker(mid, wi)

        with patch("ocabra.core.model_manager.settings") as mock_settings:
            mock_settings.idle_timeout_seconds = 60
            await mm.check_idle_evictions()
            await asyncio.sleep(0.1)

        # on_demand should have been scheduled for eviction (via background task)
        # warm and pin should remain LOADED
        warm_state = await mm.get_state("mock/warm")
        pin_state = await mm.get_state("mock/pinned")
        assert warm_state.status == ModelStatus.LOADED
        assert pin_state.status == ModelStatus.LOADED
