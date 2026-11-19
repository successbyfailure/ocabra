"""Árbitro de VRAM (BUG-23 / REF-03): las operaciones que cambian la VRAM
fuera de una carga pasan por la misma puerta que la admisión de las cargas.

Cada carrera se fuerza con eventos (sin esperas por tiempo para provocar el
orden): una tarea toma la puerta, o una carga real queda parada en su
admisión, y se comprueba que la otra operación no toca la VRAM hasta que la
puerta se suelta. Los tiempos que aparecen solo acotan cuánto se mira.
Diseño: docs/tasks/vram-arbiter-plan.md.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ocabra.backends._mock import MockBackend
from ocabra.config import settings
from ocabra.core.model_manager import (
    LoadPolicy,
    ModelManager,
    ModelState,
    ModelStatus,
)
from ocabra.core.scheduler import InsufficientVRAMError
from ocabra.core.vram_arbiter import LoadGateTimeoutError, VramArbiter
from ocabra.core.worker_pool import VisionNoRoomError, WorkerPool

# How long a test looks for something that must NOT happen while the gate is
# held. The outcome never depends on it: with the arbiter it cannot happen at
# all, without it it happens on the first ticks.
_LOOK_S = 0.4


class _RecordingBackend(MockBackend):
    """Mock backend whose unload records (and can block) the call."""

    def __init__(self) -> None:
        super().__init__(load_delay=0)
        self.unloaded: list[str] = []
        self.unload_started = asyncio.Event()

    async def unload(self, model_id: str) -> None:
        self.unloaded.append(model_id)
        self.unload_started.set()


def _mm(backend=None, gpu_manager=None) -> tuple[ModelManager, MockBackend]:
    backend = backend or _RecordingBackend()
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
    status: ModelStatus = ModelStatus.LOADED,
) -> ModelState:
    state = ModelState(
        model_id=model_id,
        display_name=model_id,
        backend_type="mock",
        status=status,
        load_policy=policy,
        current_gpu=list(gpus or [0]) if status == ModelStatus.LOADED else [],
        vram_used_mb=vram_mb if status == ModelStatus.LOADED else 0,
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


async def _hold_gate(
    mm: ModelManager,
    release: asyncio.Event,
    held: asyncio.Event,
    *,
    waiting_on: str | None = None,
) -> None:
    """Stand-in for a load inside its critical section; ``waiting_on`` makes
    it a load in a passive wait (draining that model / that service)."""
    async with mm._acquire_load_gate():
        if waiting_on is None:
            held.set()
            await release.wait()
            return
        mm._eviction_wait_for["loader"] = waiting_on
        try:
            with mm._arbiter.waiting_on(waiting_on):
                held.set()
                await release.wait()
        finally:
            mm._eviction_wait_for.pop("loader", None)


async def _not_within(event: asyncio.Event, seconds: float = _LOOK_S) -> bool:
    try:
        await asyncio.wait_for(event.wait(), timeout=seconds)
    except TimeoutError:
        return True
    return False


# ------------------------------------------------------------ VramArbiter


@pytest.mark.asyncio
async def test_arbiter_is_reentrant_and_tracks_waits_only_for_its_owner():
    arbiter = VramArbiter()
    with arbiter.waiting_on("x"):  # not the owner: ignored
        assert not arbiter.is_waiting_on("x")
    async with arbiter.hold(timeout_s=1, purpose="t") as held:
        assert held and arbiter.held_by_current() and arbiter.purpose == "t"
        async with arbiter.hold(timeout_s=1, purpose="inner") as again:
            assert again
        with arbiter.waiting_on("x"):
            assert arbiter.is_waiting_on("x") and arbiter.is_waiting()
        assert not arbiter.is_waiting()
    assert not arbiter.locked()


@pytest.mark.asyncio
async def test_arbiter_wait_can_be_abandoned_or_time_out():
    arbiter = VramArbiter()
    release = asyncio.Event()
    held = asyncio.Event()

    async def owner() -> None:
        async with arbiter.hold(timeout_s=1, purpose="load"):
            with arbiter.waiting_on("m"):
                held.set()
                await release.wait()

    task = asyncio.create_task(owner())
    await asyncio.wait_for(held.wait(), timeout=2)
    async with arbiter.hold(
        timeout_s=5, purpose="req", abort_if=lambda: arbiter.is_waiting_on("m")
    ) as got:
        assert got is False
    with pytest.raises(LoadGateTimeoutError):
        async with arbiter.hold(timeout_s=0.3, purpose="req"):
            pass
    async with arbiter.hold(timeout_s=0.3, purpose="req", timeout_yields_false=True) as got:
        assert got is False
    release.set()
    await task
    assert not arbiter.locked()


# ------------------------------------------- manual unload vs a load's admission


@pytest.mark.asyncio
async def test_manual_unload_does_not_free_vram_inside_a_loads_admission():
    """A load reading the cards to admit itself (here: parked inside the
    scheduler's find_gpu_for_model) used to see a manual unload change them
    under its feet. Now the unload waits for the load's critical section."""
    mm, backend = _mm()
    admitting = asyncio.Event()
    finish_admission = asyncio.Event()

    async def find_gpu_for_model(*args, **kwargs):
        admitting.set()
        await finish_admission.wait()
        return [0]

    mm._gpu_scheduler = SimpleNamespace(find_gpu_for_model=find_gpu_for_model)
    _loaded(mm, "new", status=ModelStatus.CONFIGURED)
    _loaded(mm, "other")

    load = asyncio.create_task(mm.load("new"))
    await asyncio.wait_for(admitting.wait(), timeout=2)

    unload = asyncio.create_task(mm.unload("other", reason="manual"))
    assert await _not_within(backend.unload_started), (
        "the manual unload changed VRAM while the load was admitting"
    )
    assert mm._states["other"].status == ModelStatus.LOADED

    finish_admission.set()
    await asyncio.wait_for(load, timeout=3)
    await asyncio.wait_for(unload, timeout=3)
    assert backend.unloaded == ["other"]
    assert mm._states["other"].status == ModelStatus.UNLOADED
    assert mm._states["new"].status == ModelStatus.LOADED


@pytest.mark.asyncio
async def test_manual_unload_goes_ahead_while_the_load_only_drains():
    """While the gate's owner is passively draining a busy candidate it reads
    nothing; the request it drains may be the one asking for this unload
    (a worker transport error, the busy-timeout watchdog). The unload goes
    ahead as before instead of waiting for the drain."""
    mm, backend = _mm()
    _loaded(mm, "stuck")
    release = asyncio.Event()
    held = asyncio.Event()
    gate = asyncio.create_task(_hold_gate(mm, release, held, waiting_on="stuck"))
    await asyncio.wait_for(held.wait(), timeout=2)

    await asyncio.wait_for(mm.unload("stuck", reason="busy_timeout"), timeout=2)
    assert backend.unloaded == ["stuck"]
    release.set()
    await gate


@pytest.mark.asyncio
async def test_idle_unload_skips_under_a_load_instead_of_queueing():
    """The idle sweep's background unload skips while a load runs (the model
    goes on a later tick): it used to check the gate and then unload with a
    load free to start in between."""
    mm, backend = _mm()
    _loaded(mm, "idle")
    release = asyncio.Event()
    held = asyncio.Event()
    gate = asyncio.create_task(_hold_gate(mm, release, held))
    await asyncio.wait_for(held.wait(), timeout=2)
    mm._load_in_progress.add("loading-model")

    await asyncio.wait_for(mm.unload("idle", reason="idle"), timeout=2)
    assert backend.unloaded == []
    assert mm._states["idle"].status == ModelStatus.LOADED
    mm._load_in_progress.discard("loading-model")
    release.set()
    await gate


@pytest.mark.asyncio
async def test_unload_by_a_task_holding_a_model_lock_does_not_wait_for_the_gate():
    """Lock order is model lock -> gate only for loads; an unload from a task
    that holds some model's lock goes ahead as before rather than wait."""
    mm, backend = _mm()
    _loaded(mm, "m")
    release = asyncio.Event()
    held = asyncio.Event()
    gate = asyncio.create_task(_hold_gate(mm, release, held))
    await asyncio.wait_for(held.wait(), timeout=2)

    async with mm._acquire_model_load_lock("m"):
        await asyncio.wait_for(mm.unload("m", reason="manual"), timeout=2)
    assert backend.unloaded == ["m"]
    release.set()
    await gate


# ------------------------------------------- make_room_for_vision vs an eviction


class _VisionBackend(_RecordingBackend):
    SUPPORTS_LIVE_CONTEXT = True

    def __init__(self) -> None:
        super().__init__()
        self.active = False
        self.activating = asyncio.Event()

    def _contains_image(self, body):
        return True

    def vision_is_active(self, model_id):
        return self.active

    def vision_projector_path(self, model_id):
        return "/models/mmproj.gguf"

    async def activate_vision(self, model_id):
        self.activating.set()
        self.active = True


def _vision_mm(free_mb: int) -> tuple[ModelManager, _VisionBackend]:
    gpu_manager = MagicMock()
    mm, backend = _mm(backend=_VisionBackend(), gpu_manager=gpu_manager)
    mm._worker_pool._backends["mock"] = backend
    mm._registered_backend = lambda backend_type: backend
    mm._vision_cost_estimate = lambda state, path: {0: 1000}
    mm._fresh_free_vram = AsyncMock(return_value=free_mb)
    return mm, backend


@pytest.mark.asyncio
async def test_vision_projector_waits_for_a_load_holding_the_gate():
    """The projector used to load (and the context to shrink for it) while a
    load was admitting on the same card: the load saw room the projector was
    about to take, then one of the two died with cudaMalloc."""
    mm, backend = _vision_mm(free_mb=5000)
    _loaded(mm, "vlm")
    release = asyncio.Event()
    held = asyncio.Event()
    gate = asyncio.create_task(_hold_gate(mm, release, held))
    await asyncio.wait_for(held.wait(), timeout=2)

    room = asyncio.create_task(mm.make_room_for_vision("vlm", {"messages": []}))
    assert await _not_within(backend.activating), "projector loaded under a load"

    release.set()
    await gate
    await asyncio.wait_for(room, timeout=2)
    assert backend.active


@pytest.mark.asyncio
async def test_vision_shrink_waits_for_an_eviction_holding_the_gate():
    """Same, with no room: the shrink of the model's own context waited only
    for the model's lock, not for the eviction measuring that card."""
    mm, backend = _vision_mm(free_mb=0)
    _loaded(mm, "vlm")
    shrinking = asyncio.Event()

    async def shrink_exactly(model_id, gpu, needed):
        shrinking.set()
        return needed

    mm._can_change_context_live = lambda model_id: True
    mm._shrink_exactly = shrink_exactly
    release = asyncio.Event()
    held = asyncio.Event()
    gate = asyncio.create_task(_hold_gate(mm, release, held))
    await asyncio.wait_for(held.wait(), timeout=2)

    room = asyncio.create_task(mm.make_room_for_vision("vlm", {"messages": []}))
    assert await _not_within(shrinking), "shrank the context under an eviction"

    release.set()
    await gate
    await asyncio.wait_for(room, timeout=2)
    assert shrinking.is_set() and backend.active


@pytest.mark.asyncio
async def test_vision_gives_up_when_the_gate_owner_is_draining_this_model():
    """This request is in flight on the model the owner drains: waiting for
    the gate would hold both until the timeouts. It fails at once with the
    retryable vision_no_room (a router retries elsewhere)."""
    mm, backend = _vision_mm(free_mb=5000)
    _loaded(mm, "vlm")
    release = asyncio.Event()
    held = asyncio.Event()
    gate = asyncio.create_task(_hold_gate(mm, release, held, waiting_on="vlm"))
    await asyncio.wait_for(held.wait(), timeout=2)

    with pytest.raises(VisionNoRoomError):
        await asyncio.wait_for(mm.make_room_for_vision("vlm", {"messages": []}), timeout=2)
    assert not backend.active
    release.set()
    await gate


# --------------------------------------------------- make_room_for_whisper


@pytest.mark.asyncio
async def test_whisper_room_shrink_waits_for_a_load_holding_the_gate(monkeypatch):
    monkeypatch.setattr(settings, "whisper_diarization_extra_mb", 3000)
    mm, _ = _mm(gpu_manager=MagicMock())
    _loaded(mm, "whisper", policy=LoadPolicy.PIN)
    _loaded(mm, "llm")
    mm._fresh_free_vram = AsyncMock(return_value=0)
    mm._can_change_context_live = lambda model_id: model_id == "llm"
    mm._get_pressure_eviction_candidates = lambda *a, **k: ["llm"]
    shrinking = asyncio.Event()

    async def try_shrink(candidate, gpu, needed, **kwargs):
        shrinking.set()
        return needed

    mm._try_shrink_for_space = try_shrink
    release = asyncio.Event()
    held = asyncio.Event()
    gate = asyncio.create_task(_hold_gate(mm, release, held))
    await asyncio.wait_for(held.wait(), timeout=2)

    room = asyncio.create_task(mm.make_room_for_whisper("whisper", diarize=True))
    assert await _not_within(shrinking), "shrank a neighbour under a load"

    release.set()
    await gate
    await asyncio.wait_for(room, timeout=2)
    assert shrinking.is_set()


# --------------------------------------------------------- ensure_vram_free


@pytest.mark.asyncio
async def test_ensure_vram_free_waits_for_a_load_holding_the_gate():
    """A service evicting for itself used to run its eviction cascade in
    parallel with a load's, both on the same figures."""
    mm, _ = _mm()
    assign = AsyncMock(return_value=[0])
    mm._assign_gpus_for_load = assign
    release = asyncio.Event()
    held = asyncio.Event()
    gate = asyncio.create_task(_hold_gate(mm, release, held))
    await asyncio.wait_for(held.wait(), timeout=2)

    service = asyncio.create_task(mm.ensure_vram_free("service:trellis", 8000, 0))
    await asyncio.sleep(_LOOK_S)
    assign.assert_not_awaited()

    release.set()
    await gate
    assert await asyncio.wait_for(service, timeout=2) == [0]
    assign.assert_awaited_once()


@pytest.mark.asyncio
async def test_ensure_vram_free_reserves_before_releasing_the_gate():
    """The reservation used to be taken by the endpoint after the evictions
    returned: a load could take the freed room in between."""
    mm, _ = _mm()
    mm._assign_gpus_for_load = AsyncMock(return_value=[0])
    seen: list[bool] = []

    async def reserve(service_id, vram_mb):
        seen.append(mm._holds_load_gate())

    mm._service_manager = SimpleNamespace(reserve_gpu_vram=reserve)
    await mm.ensure_vram_free("service:trellis", 8000, 0)
    assert seen == [True]

    # A partial failure reserves too (the endpoint's contract), still inside.
    mm._assign_gpus_for_load = AsyncMock(side_effect=InsufficientVRAMError("short"))
    with pytest.raises(InsufficientVRAMError):
        await mm.ensure_vram_free("service:trellis", 8000, 0)
    assert seen == [True, True]


@pytest.mark.asyncio
async def test_ensure_vram_free_inside_a_load_evicting_that_service_does_not_wait():
    """A load under the gate asks the service to unload (pressure_evict waits
    out its generation) while that same service's request is in
    ensure_vram_free. The service gives up at once instead of waiting for a
    gate whose owner waits for it; the load then finishes normally."""
    mm, backend = _mm()
    service_asked = asyncio.Event()
    service_answered = asyncio.Event()
    service_result: list[object] = []
    calls: list[str] = []

    async def pressure_evict(service_id):
        service_asked.set()
        # The service's generation is "running" until its own request ends.
        await asyncio.wait_for(service_answered.wait(), timeout=5)
        return True

    find_calls = {"n": 0}

    async def find_gpu_for_model(*args, **kwargs):
        find_calls["n"] += 1
        calls.append(asyncio.current_task().get_name())
        if find_calls["n"] == 1:
            raise InsufficientVRAMError("full")
        return [0]

    mm._gpu_scheduler = SimpleNamespace(find_gpu_for_model=find_gpu_for_model)
    mm._service_manager = SimpleNamespace(
        get_pressure_eviction_candidates=lambda preferred_gpu=None: ["trellis"],
        pressure_evict=pressure_evict,
        reserve_gpu_vram=AsyncMock(),
    )
    mm._wait_for_vram_released = AsyncMock()
    _loaded(mm, "new", status=ModelStatus.CONFIGURED)

    load = asyncio.create_task(mm.load("new"), name="load")
    await asyncio.wait_for(service_asked.wait(), timeout=2)

    async def service_request() -> None:
        try:
            service_result.append(await mm.ensure_vram_free("service:trellis", 8000, 0))
        except InsufficientVRAMError as exc:
            service_result.append(exc)
        finally:
            service_answered.set()

    await asyncio.wait_for(service_request(), timeout=2)
    assert isinstance(service_result[0], InsufficientVRAMError)
    assert calls and set(calls) == {"load"}, "the service evicted inside the load's eviction"
    await asyncio.wait_for(load, timeout=3)
    assert mm._states["new"].status == ModelStatus.LOADED


# ------------------------------------------------------------ VRAM watchdog


@pytest.mark.asyncio
async def test_vram_watchdog_eviction_runs_under_the_gate(monkeypatch):
    """The watchdog checked that the gate was free and then evicted: a load
    could take the gate in between. Its eviction now holds it."""
    monkeypatch.setattr(settings, "idle_eviction_check_interval_seconds", 1)
    monkeypatch.setattr(settings, "vram_eviction_threshold", 0.5, raising=False)
    gpu_manager = MagicMock()
    gpu_manager.get_all_states = AsyncMock(
        return_value=[SimpleNamespace(index=0, total_vram_mb=1000, used_vram_mb=990)]
    )
    mm, _ = _mm(gpu_manager=gpu_manager)
    _loaded(mm, "a")
    _loaded(mm, "b")
    held_during: list[bool] = []
    evicted = asyncio.Event()

    async def evict(gpu_index, needed_mb):
        held_during.append(mm._holds_load_gate())
        evicted.set()
        return 0

    mm._evict_for_space = evict
    watchdog = asyncio.create_task(mm._vram_watchdog())
    await asyncio.wait_for(evicted.wait(), timeout=3)
    watchdog.cancel()
    await asyncio.gather(watchdog, return_exceptions=True)
    assert held_during and all(held_during)
