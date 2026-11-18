"""Arreglos del núcleo del ModelManager (revisión del 16/11).

BUG-06 vigilante de peticiones colgadas, BUG-07 unload bajo el lock del
modelo, BUG-23 evicciones en segundo plano frente a una carga, BUG-24
presupuesto de drenado y espera de la puerta reintentable, BUG-26 atributos
declarados, BUG-27 VRAM por GPU al expulsar, BUG-29 update_config igual en BD
y memoria, BUG-33 recuperación de arranque muerta.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import ocabra.core.model_manager as mm_module
from ocabra.backends._mock import MockBackend
from ocabra.config import settings
from ocabra.core.model_manager import (
    LoadPolicy,
    ModelManager,
    ModelState,
    ModelStatus,
)
from ocabra.core.worker_pool import WorkerPool


class _BlockingBackend(MockBackend):
    """Mock backend whose unload blocks until released."""

    def __init__(self) -> None:
        super().__init__(load_delay=0)
        self.unload_started = asyncio.Event()
        self.release_unload = asyncio.Event()
        self.loads = 0
        self.unloads = 0

    async def load(self, model_id, gpu_indices, **kwargs):
        self.loads += 1
        return await super().load(model_id, gpu_indices, **kwargs)

    async def unload(self, model_id: str) -> None:
        self.unloads += 1
        self.unload_started.set()
        await self.release_unload.wait()


def _mm(backend=None, gpu_manager=None) -> tuple[ModelManager, MockBackend]:
    backend = backend or MockBackend(load_delay=0)
    wp = WorkerPool()
    wp.register_backend("mock", backend)
    mm = ModelManager(wp, gpu_manager=gpu_manager)
    mm._record_model_load_stat = AsyncMock()  # no database here
    return mm, backend


def _loaded(
    mm: ModelManager,
    model_id: str,
    *,
    gpus: list[int] | None = None,
    vram_mb: int = 4096,
    policy: LoadPolicy = LoadPolicy.ON_DEMAND,
) -> ModelState:
    state = ModelState(
        model_id=model_id,
        display_name=model_id,
        backend_type="mock",
        status=ModelStatus.LOADED,
        load_policy=policy,
        current_gpu=list(gpus or [0]),
        vram_used_mb=vram_mb,
    )
    mm._states[model_id] = state
    mm._load_locks[model_id] = asyncio.Lock()
    return state


@pytest.fixture(autouse=True)
def _quiet_redis():
    with (
        patch("ocabra.core.model_manager.publish", new=AsyncMock()),
        patch("ocabra.core.model_manager.set_key", new=AsyncMock()),
        patch("ocabra.core.model_manager.publish_system_alert", new=AsyncMock()),
    ):
        yield


async def _hold_gate(mm: ModelManager, release: asyncio.Event, held: asyncio.Event) -> None:
    async with mm._acquire_load_gate():
        held.set()
        await release.wait()


# ---------------------------------------------------------------- BUG-06


def _stale_request(mm: ModelManager, model_id: str, age_s: float = 5000) -> str:
    rid = mm.begin_request(model_id)
    req = mm._active_requests[rid]
    req.started_at = time.time() - age_s
    req.last_activity_at = time.time() - age_s
    return rid


@pytest.mark.asyncio
async def test_busy_timeout_mark_error_stops_the_worker(monkeypatch):
    """mark_error used to leave ERROR with the process alive (VRAM leaked:
    nothing evicts an ERROR model). Now the worker is unloaded first."""
    monkeypatch.setattr(settings, "busy_timeout_action", "mark_error", raising=False)
    gpu_manager = MagicMock()
    gpu_manager.locked_gpu_indices = MagicMock(return_value=[0])
    gpu_manager.unlock_vram = AsyncMock()
    mm, backend = _mm(gpu_manager=gpu_manager)
    backend.unload = AsyncMock()
    state = _loaded(mm, "m", vram_mb=8000)
    _stale_request(mm, "m")

    await mm._check_busy_timeouts()

    backend.unload.assert_awaited_once()
    gpu_manager.unlock_vram.assert_awaited_once_with(0, "m")
    assert state.status == ModelStatus.ERROR
    assert state.vram_used_mb == 0
    assert state.current_gpu == []
    assert "stuck" in (state.error_message or "")


@pytest.mark.asyncio
async def test_busy_timeout_leaves_a_worker_that_is_still_serving(monkeypatch):
    """A leaked entry next to requests that are progressing: drop the entry,
    don't kill the worker serving them."""
    monkeypatch.setattr(settings, "busy_timeout_action", "mark_error", raising=False)
    mm, backend = _mm()
    backend.unload = AsyncMock()
    state = _loaded(mm, "m")
    stale = _stale_request(mm, "m")
    fresh = mm.begin_request("m")

    await mm._check_busy_timeouts()

    assert stale not in mm._active_requests
    assert fresh in mm._active_requests
    backend.unload.assert_not_awaited()
    assert state.status == ModelStatus.LOADED


# ---------------------------------------------------------------- BUG-07


@pytest.mark.asyncio
async def test_load_waits_for_an_unload_in_progress():
    """A load arriving while the model is UNLOADING used to start a second
    worker at once; now it waits for the unload under the model's lock."""
    backend = _BlockingBackend()
    mm, _ = _mm(backend=backend)
    state = _loaded(mm, "m")

    unload_task = asyncio.create_task(mm.unload("m", reason="idle"))
    await asyncio.wait_for(backend.unload_started.wait(), timeout=2)
    assert state.status == ModelStatus.UNLOADING

    load_task = asyncio.create_task(mm.load("m"))
    await asyncio.sleep(0.2)
    assert backend.loads == 0, "the load must wait for the unload to finish"
    assert not load_task.done()

    backend.release_unload.set()
    await asyncio.wait_for(unload_task, timeout=2)
    await asyncio.wait_for(load_task, timeout=2)
    assert backend.loads == 1
    assert state.status == ModelStatus.LOADED


@pytest.mark.asyncio
async def test_unload_inside_the_gate_does_not_wait_for_a_load_queued_behind_it():
    """An eviction holding the gate unloads V while V's own load holds V's
    lock (about to wait for that gate): unload must give up once V is
    LOADING, not deadlock until the timeouts fire — and not kill the worker
    that load is starting."""
    mm, backend = _mm()
    backend.unload = AsyncMock()
    state = _loaded(mm, "v")
    state.status = ModelStatus.ERROR  # unloadable when the eviction looks
    release = asyncio.Event()

    async def load_holding_v_lock() -> None:
        async with mm._acquire_model_load_lock("v"):
            await asyncio.sleep(0.3)
            state.status = ModelStatus.LOADING
            await release.wait()  # would be: waiting for the gate

    loader = asyncio.create_task(load_holding_v_lock())
    await asyncio.sleep(0.05)

    async def evict_inside_gate() -> None:
        async with mm._acquire_load_gate():
            await mm.unload("v", reason="pressure")

    started = time.monotonic()
    await asyncio.wait_for(evict_inside_gate(), timeout=3)
    assert time.monotonic() - started < 2
    backend.unload.assert_not_awaited()
    assert state.status == ModelStatus.LOADING
    release.set()
    await loader


@pytest.mark.asyncio
async def test_unload_is_reentrant_for_the_lock_holder():
    mm, _ = _mm()
    state = _loaded(mm, "m")
    async with mm._acquire_model_load_lock("m"):
        await asyncio.wait_for(mm.unload("m", reason="manual_test"), timeout=2)
    assert state.status == ModelStatus.UNLOADED


# ---------------------------------------------------------------- BUG-23


@pytest.mark.asyncio
async def test_vram_watchdog_skips_while_a_load_holds_the_gate(monkeypatch):
    monkeypatch.setattr(settings, "idle_eviction_check_interval_seconds", 1)
    monkeypatch.setattr(settings, "vram_eviction_threshold", 0.5, raising=False)
    gpu_manager = MagicMock()
    gpu_manager.get_all_states = AsyncMock(
        return_value=[SimpleNamespace(index=0, total_vram_mb=1000, used_vram_mb=990)]
    )
    mm, _ = _mm(gpu_manager=gpu_manager)
    _loaded(mm, "a")
    _loaded(mm, "b")
    evict = AsyncMock(return_value=0)
    monkeypatch.setattr(mm, "_evict_for_space", evict)

    release = asyncio.Event()
    held = asyncio.Event()
    gate = asyncio.create_task(_hold_gate(mm, release, held))
    await held.wait()
    watchdog = asyncio.create_task(mm._vram_watchdog())
    await asyncio.sleep(1.3)
    evict.assert_not_awaited()

    release.set()
    await gate
    await asyncio.sleep(1.1)
    watchdog.cancel()
    await asyncio.gather(watchdog, return_exceptions=True)
    evict.assert_awaited()


@pytest.mark.asyncio
async def test_idle_eviction_waits_for_the_load_in_progress(monkeypatch):
    monkeypatch.setattr(settings, "idle_timeout_seconds", 1)
    mm, _ = _mm()
    state = _loaded(mm, "idle")
    state.loaded_at = mm_module.datetime.now(mm_module.UTC) - mm_module.timedelta(hours=1)
    created: list[str] = []

    def fake_create(coro, *, task_name, model_id=None):
        coro.close()
        created.append(task_name)

    monkeypatch.setattr(mm, "_create_background_task", fake_create)

    release = asyncio.Event()
    held = asyncio.Event()
    gate = asyncio.create_task(_hold_gate(mm, release, held))
    await held.wait()
    await mm.check_idle_evictions()
    assert created == []

    release.set()
    await gate
    await mm.check_idle_evictions()
    assert created == ["idle-unload:idle"]


@pytest.mark.asyncio
async def test_evict_for_whisper_does_not_evict_under_a_running_load():
    gpu_manager = MagicMock()
    mm, _ = _mm(gpu_manager=gpu_manager)
    _loaded(mm, "whisper", policy=LoadPolicy.PIN)
    _loaded(mm, "victim")
    mm.unload = AsyncMock()

    release = asyncio.Event()
    held = asyncio.Event()
    gate = asyncio.create_task(_hold_gate(mm, release, held))
    await held.wait()
    assert await mm.evict_for_whisper("whisper", needed_mb=1000) is False
    mm.unload.assert_not_awaited()
    release.set()
    await gate


# ---------------------------------------------------------------- BUG-24


@pytest.mark.asyncio
async def test_drain_waits_share_one_budget_per_load(monkeypatch):
    """Two busy candidates used to get up to the full drain timeout each."""
    monkeypatch.setattr(settings, "pressure_eviction_drain_timeout_s", 2)
    monkeypatch.setattr(settings, "max_drain_timeout_s", 2)
    mm, _ = _mm()
    _loaded(mm, "busy-a")
    _loaded(mm, "busy-b")
    mm._in_flight["busy-a"] = 1
    mm._in_flight["busy-b"] = 1
    monkeypatch.setattr(
        mm, "_get_pressure_eviction_candidates", lambda *a, **k: ["busy-a", "busy-b"]
    )
    vars(mm).setdefault("_drain_deadlines", {})["req"] = asyncio.get_running_loop().time() + 2

    started = time.monotonic()
    freed = await mm._evict_last_resort_for_space("req", 0, 1000)
    elapsed = time.monotonic() - started

    assert freed == 0
    assert elapsed < 3.0, f"waited {elapsed:.1f}s: one budget, not one per candidate"


@pytest.mark.asyncio
async def test_model_lock_timeout_is_retryable(monkeypatch):
    monkeypatch.setattr(settings, "max_drain_timeout_s", 1)
    mm, _ = _mm()
    _loaded(mm, "m").status = ModelStatus.UNLOADED
    lock = mm._load_locks["m"]
    await lock.acquire()
    try:
        with pytest.raises(mm_module.LoadGateTimeoutError):
            await mm.load("m")
    finally:
        lock.release()
    assert mm._states["m"].status == ModelStatus.UNLOADED


# ---------------------------------------------------------------- BUG-26


def test_late_bound_collaborators_are_declared():
    mm = ModelManager(WorkerPool())
    for name in ("_session_registry", "_duration_estimator", "_batch_priority"):
        assert name in vars(mm), name
        assert getattr(mm, name) is None


# ---------------------------------------------------------------- BUG-27


@pytest.mark.asyncio
async def test_eviction_counts_what_the_model_holds_on_that_gpu():
    """A 12 GB tensor-split model holding 9 GB on GPU 1 frees 9 GB there,
    not 12 / 2 = 6."""
    gpu_manager = MagicMock()
    gpu_manager.locks_snapshot = MagicMock(
        side_effect=lambda gpu: {"split": 3000 if gpu == 0 else 9000}
    )
    mm, _ = _mm(gpu_manager=gpu_manager)
    _loaded(mm, "split", gpus=[0, 1], vram_mb=12000)
    mm.unload = AsyncMock()
    mm._try_shrink_for_space = AsyncMock(return_value=0)

    assert await mm._evict_for_space(1, 8000) == 9000

    _loaded(mm, "split", gpus=[0, 1], vram_mb=12000)
    mm._get_vram_free_snapshot = AsyncMock(return_value=None)
    mm._wait_for_vram_released = AsyncMock()
    state = mm._states["split"]
    assert await mm._last_resort_evict_claimed("split", state, 0, 1000) == 3000


# ---------------------------------------------------------------- BUG-29


class _FakeSession:
    def __init__(self, cfg) -> None:
        self._cfg = cfg
        self.commit = AsyncMock()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def execute(self, _stmt):
        return SimpleNamespace(scalar_one_or_none=lambda: self._cfg)


@pytest.mark.asyncio
async def test_update_config_merges_the_same_patch_into_db_and_memory(monkeypatch):
    artifact = {"kind": "file", "path": "/models/x.gguf"}
    cfg = SimpleNamespace(
        extra_config={"llama_cpp": {"ctx_size": 4096}},
        artifact=artifact,
    )
    monkeypatch.setattr(mm_module.database, "AsyncSessionLocal", lambda: _FakeSession(cfg))
    # Paths come from the artifact; keep it out of the filesystem here.
    monkeypatch.setattr(
        mm_module.model_artifacts,
        "materialize",
        lambda backend, art, extra, root: {
            **extra,
            "llama_cpp": {**(extra.get("llama_cpp") or {}), "model_path": "/models/x.gguf"},
        },
    )
    monkeypatch.setattr(
        mm_module.model_artifacts, "apply_path_edits", lambda backend, art, inc, cur: art
    )

    mm, _ = _mm()
    state = ModelState(
        model_id="llama_cpp/x",
        display_name="x",
        backend_type="llama_cpp",
        artifact=artifact,
        extra_config={
            "llama_cpp": {"ctx_size": 4096, "model_path": "/models/x.gguf"},
            "registry_metadata": {"source": "hf"},
        },
    )
    mm._states[state.model_id] = state
    mm._persisted_model_ids.add(state.model_id)

    # What the UI sends back: GET's payload with one field edited, including
    # a stale registry_metadata and the derived path.
    await mm.update_config(
        state.model_id,
        {
            "extra_config": {
                "llama_cpp": {"ctx_size": 8192, "model_path": "/models/x.gguf"},
                "registry_metadata": {"source": "stale"},
            }
        },
    )

    assert cfg.extra_config == {"llama_cpp": {"ctx_size": 8192}}
    assert state.extra_config["llama_cpp"] == {"ctx_size": 8192, "model_path": "/models/x.gguf"}
    # Memory keeps the row's registry metadata, as the database does.
    assert state.extra_config["registry_metadata"] == {"source": "hf"}


# ---------------------------------------------------------------- BUG-33


@pytest.mark.asyncio
async def test_pin_autoload_does_not_unload_error_models(monkeypatch):
    monkeypatch.setattr(settings, "pin_autoload_delay_s", 0)
    monkeypatch.setattr(settings, "pin_autoload_max_wait_s", 0)
    mm = ModelManager(worker_pool=None)
    mm._states["pin"] = ModelState(
        model_id="pin",
        display_name="pin",
        backend_type="mock",
        load_policy=LoadPolicy.PIN,
        status=ModelStatus.ERROR,
    )
    unload = AsyncMock()
    monkeypatch.setattr(mm, "unload", unload)
    monkeypatch.setattr(mm, "_load_model", AsyncMock())
    await mm._autoload_pinned()
    await asyncio.sleep(0)
    unload.assert_not_awaited()
