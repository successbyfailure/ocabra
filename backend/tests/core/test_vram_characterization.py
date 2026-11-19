"""Caracterización de la máquina de estados de VRAM (prerrequisito de REF-03).

Fijan el comportamiento observable de hoy —qué GPU se elige, a quién se
expulsa y en qué orden, qué transiciones de estado y qué errores salen— con un
mundo falso que ocupa y libera memoria de verdad (``vram_world``): dos
tarjetas como las de producción (0 = 3060 de 12 GB, 1 = 3090 de 24 GB), el
``GPUManager`` y el ``GPUScheduler`` reales, Redis y backend falsos.

La última sección inyecta un fallo (excepción o cancelación) en cada ``await``
de una carga que expulsa, y comprueba que no queda nada a medias: ni el
árbitro tomado, ni modelos en LOADING/UNLOADING, ni reservas o workers
huérfanos, y que la siguiente carga funciona.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from ocabra.config import settings
from ocabra.core.model_manager import LoadPolicy, ModelStatus
from ocabra.core.scheduler import InsufficientVRAMError
from ocabra.core.worker_pool import VisionNoRoomError
from tests.core.vram_world import GPU_3060, GPU_3090, Env, assert_quiescent

# hw-monitor publishes ~2 s late in production; a short lag keeps the release
# waits honest (they must see the figure move) without slowing the suite.
_LAG_S = 0.2


@pytest.fixture(autouse=True)
def _fast_drains(monkeypatch):
    monkeypatch.setattr(settings, "pressure_eviction_drain_timeout_s", 1)
    monkeypatch.setattr(settings, "max_drain_timeout_s", 5)


@pytest.fixture
def env():
    with Env(publish_lag_s=_LAG_S) as e:
        yield e


async def _full_3060(env: Env) -> None:
    """A hard pin filling the 3060, so every choice happens on the 3090."""
    await env.resident("whisper", 11_500, [GPU_3060], policy=LoadPolicy.PIN, pin_permanent=True)


# ------------------------------------------------------------------ placement


async def test_unplaced_model_goes_to_the_default_gpu(env):
    env.configure("m", 8_000)
    await env.mm.load("m")
    assert env.mm._states["m"].current_gpu == [settings.default_gpu_index]
    assert env.gpu.locks_snapshot(settings.default_gpu_index) == {"m": 8_000}
    assert_quiescent(env)


async def test_preferred_gpu_wins_when_it_has_room(env):
    env.configure("m", 8_000, preferred_gpu=GPU_3060)
    await env.mm.load("m")
    assert env.mm._states["m"].current_gpu == [GPU_3060]


async def test_preferred_gpu_without_room_falls_to_the_other_card(env):
    await _full_3060(env)
    env.configure("m", 8_000, preferred_gpu=GPU_3060)
    await env.mm.load("m")
    assert env.mm._states["m"].current_gpu == [GPU_3090]
    assert env.backend.unloads == []


async def test_model_bigger_than_any_card_is_split_across_both(env):
    env.configure("big", 30_000)
    await env.mm.load("big")
    assert env.mm._states["big"].current_gpu == [GPU_3060, GPU_3090]
    assert set(env.gpu.locked_gpu_indices("big")) == {GPU_3060, GPU_3090}
    assert_quiescent(env)


async def test_preset_placement_admits_each_shard_and_evicts_per_card(env):
    """A runtime preset on both cards checks each shard on its own card and
    evicts only where its shard lacks room."""
    await env.resident("od0", 6_000, [GPU_3060], last_used_s_ago=50)
    await env.resident("od1", 6_000, [GPU_3090], last_used_s_ago=50)
    env.configure(
        "tp",
        28_000,
        extra_config={
            "_runtime": {
                "gpu_indices": [GPU_3060, GPU_3090],
                "capacity_vram_by_gpu_mb": {"0": 8_000, "1": 16_000},
            }
        },
        split={GPU_3060: 8_000, GPU_3090: 16_000},
    )
    await env.mm.load("tp")
    # 3060: 12,288 - 6,000 - 512 < 8,000 → od0 goes; 3090 has room for 16,000.
    assert env.backend.unloads == ["od0"]
    assert env.status("od1") == ModelStatus.LOADED
    assert env.gpu.locks_snapshot(GPU_3060) == {"tp": 8_000}
    assert_quiescent(env)


# --------------------------------------------------- eviction order (scheduler)


async def _three_on_the_3090(env: Env) -> None:
    await _full_3060(env)
    await env.resident("od", 6_000, [GPU_3090], last_used_s_ago=10)
    await env.resident("warm", 6_000, [GPU_3090], policy=LoadPolicy.WARM, last_used_s_ago=1_000)
    await env.resident("pin", 10_000, [GPU_3090], policy=LoadPolicy.PIN, last_used_s_ago=5_000)


async def test_pressure_evicts_on_demand_before_warm_before_soft_pin(env):
    await _three_on_the_3090(env)
    env.configure("new", 8_000, preferred_gpu=GPU_3090)
    await env.mm.load("new")
    assert env.backend.unloads == ["od"]
    assert env.mm._states["new"].current_gpu == [GPU_3090]
    assert env.status("warm") == env.status("pin") == ModelStatus.LOADED
    assert_quiescent(env)


async def test_pressure_goes_down_the_order_until_it_fits(env):
    await _three_on_the_3090(env)
    env.configure("new", 14_000, preferred_gpu=GPU_3090)
    await env.mm.load("new")
    assert env.backend.unloads == ["od", "warm"]
    assert env.status("pin") == ModelStatus.LOADED


async def test_soft_pin_is_evicted_last_but_is_evicted(env):
    await _three_on_the_3090(env)
    env.configure("new", 20_000, preferred_gpu=GPU_3090)
    await env.mm.load("new")
    assert env.backend.unloads == ["od", "warm", "pin"]
    assert_quiescent(env)


async def test_busy_candidate_is_drained_then_skipped_when_it_stays_busy(env):
    await _three_on_the_3090(env)
    env.mm.begin_request("od")
    env.configure("new", 8_000, preferred_gpu=GPU_3090)
    await env.mm.load("new")
    assert env.backend.unloads == ["warm"]
    assert env.status("od") == ModelStatus.LOADED


async def test_busy_candidate_that_drains_in_time_is_evicted(env, monkeypatch):
    monkeypatch.setattr(settings, "pressure_eviction_drain_timeout_s", 5)
    await _three_on_the_3090(env)
    rid = env.mm.begin_request("od")
    env.configure("new", 8_000, preferred_gpu=GPU_3090)
    load = asyncio.create_task(env.mm.load("new"))
    for _ in range(100):
        if env.mm._eviction_wait_for.get("new") == "od":
            break
        await asyncio.sleep(0.02)
    assert env.mm._eviction_wait_for.get("new") == "od"
    assert env.mm._arbiter.is_waiting_on("od")
    env.mm.end_request("od", rid)
    await asyncio.wait_for(load, timeout=5)
    assert env.backend.unloads == ["od"]
    assert_quiescent(env)


async def test_hard_pin_is_never_evicted(env):
    await _full_3060(env)
    await env.resident("hard", 16_000, [GPU_3090], policy=LoadPolicy.PIN, pin_permanent=True)
    env.configure("new", 12_000, preferred_gpu=GPU_3090)
    with pytest.raises(InsufficientVRAMError):
        await env.mm.load("new")
    assert env.backend.unloads == []
    assert env.status("new") == ModelStatus.ERROR
    assert_quiescent(env)


async def test_realtime_session_worker_is_never_evicted(env):
    await _full_3060(env)
    await env.resident("rt", 16_000, [GPU_3090], last_used_s_ago=5_000)
    await env.resident("od", 6_000, [GPU_3090], last_used_s_ago=10)

    class Sessions:
        def has_active_session_holding(self, model_id):
            return model_id == "rt"

    env.mm.set_session_registry(Sessions())
    env.configure("new", 8_000, preferred_gpu=GPU_3090)
    await env.mm.load("new")
    assert env.backend.unloads == ["od"]


async def test_preferred_card_that_cannot_be_freed_frees_the_other(env):
    """2026-11-02: preferring the 3060 must not stop the cascade from freeing
    the 3090 when only that one can give room."""
    await _full_3060(env)
    await env.resident("od", 20_000, [GPU_3090], last_used_s_ago=10)
    env.configure("new", 8_000, preferred_gpu=GPU_3060)
    await env.mm.load("new")
    assert env.backend.unloads == ["od"]
    assert env.mm._states["new"].current_gpu == [GPU_3090]


async def test_impossible_plan_evicts_nothing(env):
    await _full_3060(env)
    await env.resident("hard", 16_000, [GPU_3090], policy=LoadPolicy.PIN, pin_permanent=True)
    await env.resident("od", 6_000, [GPU_3090], last_used_s_ago=10)
    env.configure("new", 13_000, preferred_gpu=GPU_3090)
    with pytest.raises(InsufficientVRAMError):
        await env.mm.load("new")
    assert env.backend.unloads == []
    assert env.status("od") == ModelStatus.LOADED


async def test_peer_rules_without_evict_never_push_anything_out(env):
    from ocabra.core.federation_rules import current_peer_rules

    await _three_on_the_3090(env)
    env.configure("new", 8_000, preferred_gpu=GPU_3090)

    class Rules:
        allow_evict = False

    token = current_peer_rules.set(Rules())
    try:
        with pytest.raises(InsufficientVRAMError):
            await env.mm.load("new")
    finally:
        current_peer_rules.reset(token)
    assert env.backend.unloads == []


# --------------------------------------------- eviction order (forced placement)


async def test_forced_gpu_evicts_by_lru_among_warm_and_on_demand_before_any_pin(env):
    """The forced-placement cascade is not the scheduler's: its polite pass
    is pure LRU over warm/on-demand (warm used longer ago goes first), and
    only then the soft pin."""
    await _three_on_the_3090(env)
    env.configure("new", 12_000, extra_config={"_runtime": {"gpu_indices": [GPU_3090]}})
    await env.mm.load("new")
    assert env.backend.unloads == ["warm", "od"]
    assert env.status("pin") == ModelStatus.LOADED
    assert_quiescent(env)


async def test_forced_gpu_reaches_the_soft_pin_as_last_resort(env):
    await _three_on_the_3090(env)
    env.configure("new", 20_000, extra_config={"_runtime": {"gpu_indices": [GPU_3090]}})
    await env.mm.load("new")
    assert env.backend.unloads == ["warm", "od", "pin"]


async def test_forced_gpu_skips_a_busy_model_after_the_drain(env):
    await _three_on_the_3090(env)
    env.mm.begin_request("od")
    env.configure("new", 12_000, extra_config={"_runtime": {"gpu_indices": [GPU_3090]}})
    await env.mm.load("new")
    assert "od" not in env.backend.unloads
    assert env.backend.unloads == ["warm", "pin"]


def test_lru_candidates_exclude_pins_busy_and_other_gpus():
    with Env() as env:
        mm = env.mm
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_three_on_the_3090(env))
            loop.run_until_complete(env.resident("other", 1_000, [GPU_3060], last_used_s_ago=9))
            loop.run_until_complete(env.resident("busy", 1_000, [GPU_3090], last_used_s_ago=99_999))
        finally:
            loop.close()
        mm.begin_request("busy")
        assert mm._get_eviction_candidates(GPU_3090) == ["warm", "od"]
        assert mm._get_pressure_eviction_candidates("x", gpu_indices=[GPU_3090]) == [
            "busy",
            "od",
            "warm",
            "pin",
        ]
        # Hard pins never appear; the requester never appears.
        assert "whisper" not in mm._get_pressure_eviction_candidates("x")
        assert "od" not in mm._get_pressure_eviction_candidates("od")


# ------------------------------------------------------------------ watchdog


async def _one_watchdog_tick(env: Env, monkeypatch) -> None:
    monkeypatch.setattr(settings, "idle_eviction_check_interval_seconds", 1)
    task = asyncio.create_task(env.mm._vram_watchdog())
    await asyncio.sleep(1.3)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


async def test_watchdog_evicts_the_lru_model_over_the_threshold(env, monkeypatch):
    monkeypatch.setattr(settings, "vram_eviction_threshold", 0.9, raising=False)
    await env.resident("old", 12_000, [GPU_3090], last_used_s_ago=1_000)
    await env.resident("new", 11_000, [GPU_3090], last_used_s_ago=1)
    await _one_watchdog_tick(env, monkeypatch)
    assert env.backend.unloads == ["old"]
    assert_quiescent(env)


async def test_watchdog_leaves_a_single_model_alone(env, monkeypatch):
    monkeypatch.setattr(settings, "vram_eviction_threshold", 0.9, raising=False)
    await env.resident("only", 23_000, [GPU_3090], last_used_s_ago=1_000)
    await _one_watchdog_tick(env, monkeypatch)
    assert env.backend.unloads == []


async def test_watchdog_skips_while_someone_holds_the_arbiter(env, monkeypatch):
    monkeypatch.setattr(settings, "vram_eviction_threshold", 0.9, raising=False)
    await env.resident("old", 12_000, [GPU_3090], last_used_s_ago=1_000)
    await env.resident("new", 11_000, [GPU_3090], last_used_s_ago=1)
    release = asyncio.Event()
    held = asyncio.Event()

    async def holder():
        async with env.mm._acquire_load_gate("test"):
            held.set()
            await release.wait()

    h = asyncio.create_task(holder())
    await held.wait()
    await _one_watchdog_tick(env, monkeypatch)
    release.set()
    await h
    assert env.backend.unloads == []


# ---------------------------------------------------------- ensure_vram_free


class _Services:
    def __init__(self, env: Env, candidates: list[str] | None = None) -> None:
        self.env = env
        self.candidates = list(candidates or [])
        self.evicted: list[str] = []
        self.reserved: list[tuple[str, int]] = []

    def get_pressure_eviction_candidates(self, preferred_gpu=None):
        return [c for c in self.candidates if c not in self.evicted]

    async def pressure_evict(self, service_id):
        self.evicted.append(service_id)
        return True

    async def reserve_gpu_vram(self, service_id, vram_mb):
        self.reserved.append((service_id, vram_mb))
        await self.env.gpu.lock_vram(GPU_3090, vram_mb, f"service:{service_id}")


async def test_ensure_vram_free_evicts_by_pressure_order_and_reserves(env):
    await _three_on_the_3090(env)
    services = _Services(env)
    env.mm.set_service_manager(services)
    gpus = await env.mm.ensure_vram_free("service:trellis", 8_000, GPU_3090)
    assert gpus == [GPU_3090]
    assert env.backend.unloads == ["od"]
    assert services.reserved == [("trellis", 8_000)]
    assert env.gpu.locks_snapshot(GPU_3090)["service:trellis"] == 8_000
    assert_quiescent(env)


async def test_ensure_vram_free_partial_failure_still_reserves(env):
    await _full_3060(env)
    await env.resident("hard", 16_000, [GPU_3090], policy=LoadPolicy.PIN, pin_permanent=True)
    services = _Services(env)
    env.mm.set_service_manager(services)
    with pytest.raises(InsufficientVRAMError):
        await env.mm.ensure_vram_free("service:trellis", 12_000, GPU_3090)
    assert services.reserved == [("trellis", 12_000)]
    assert_quiescent(env)


async def test_ensure_vram_free_evicts_other_services_before_models(env):
    await _three_on_the_3090(env)
    services = _Services(env, candidates=["comfy"])

    async def pressure_evict(service_id):
        services.evicted.append(service_id)
        # comfy held 9 GB on the 3090 (as a foreign process).
        env.world.release(-1)
        return True

    env.world.allocate(-1, {GPU_3090: 0})
    services.pressure_evict = pressure_evict
    env.mm.set_service_manager(services)
    with pytest.raises(InsufficientVRAMError):
        await env.mm.ensure_vram_free("service:trellis", 30_000, GPU_3090)
    assert services.evicted == ["comfy"]


# --------------------------------------------------------- whisper and vision


async def test_whisper_diarization_evicts_an_idle_unshrinkable_neighbour(env):
    await env.resident("whisper", 2_000, [GPU_3060], policy=LoadPolicy.PIN)
    await env.resident("emb", 7_000, [GPU_3060], last_used_s_ago=100)
    env.world.publish()
    env.mm.begin_request("whisper")
    await env.mm.make_room_for_whisper("whisper", diarize=True)
    assert env.backend.unloads == ["emb"]
    assert env.status("whisper") == ModelStatus.LOADED


async def test_whisper_does_nothing_when_the_card_has_room(env):
    await env.resident("whisper", 2_000, [GPU_3060], policy=LoadPolicy.PIN)
    await env.resident("emb", 4_000, [GPU_3060], last_used_s_ago=100)
    env.mm.begin_request("whisper")
    await env.mm.make_room_for_whisper("whisper", diarize=True)
    assert env.backend.unloads == []


async def test_whisper_never_evicts_a_busy_neighbour(env):
    await env.resident("whisper", 2_000, [GPU_3060], policy=LoadPolicy.PIN)
    await env.resident("emb", 7_000, [GPU_3060], last_used_s_ago=100)
    env.mm.begin_request("whisper")
    env.mm.begin_request("emb")
    await env.mm.make_room_for_whisper("whisper", diarize=True)
    assert env.backend.unloads == []


class _VisionBackend:
    """Vision hooks on top of WorldBackend for one model."""

    def __init__(self, env: Env, projector_mb: int) -> None:
        self.env = env
        self.projector_mb = projector_mb
        self.active: set[str] = set()

    def install(self) -> None:
        backend = self.env.backend
        backend._contains_image = lambda body: bool(body.get("image"))
        backend.vision_is_active = lambda mid: mid in self.active
        backend.vision_projector_path = lambda mid: "/nonexistent/mmproj.gguf"

        async def activate_vision(mid):
            self.active.add(mid)
            gpu = self.env.mm._states[mid].current_gpu[0]
            self.env.world.allocate(backend.pids[mid], {gpu: self.projector_mb})

        backend.activate_vision = activate_vision


async def test_vision_projector_loads_when_the_card_has_room(env):
    await env.resident("vl", 8_000, [GPU_3090])
    vision = _VisionBackend(env, 1_000)
    vision.install()
    await env.mm.make_room_for_vision("vl", {"image": True})
    assert vision.active == {"vl"}
    assert env.world.live[GPU_3090][env.backend.pids["vl"]] == 9_000
    assert not env.mm._arbiter.locked()


async def test_vision_without_room_on_an_unshrinkable_model_is_vision_no_room(env):
    await env.resident("vl", 23_500, [GPU_3090])
    _VisionBackend(env, 1_000).install()
    with pytest.raises(VisionNoRoomError):
        await env.mm.make_room_for_vision("vl", {"image": True})
    assert not env.backend.vision_is_active("vl")
    assert env.backend.unloads == []
    assert not env.mm._arbiter.locked()


# ------------------------------------------- failures injected at every await


async def _eviction_scenario(env: Env) -> None:
    await _full_3060(env)
    await env.resident("od", 16_000, [GPU_3090], last_used_s_ago=10)
    env.configure("big", 12_000, preferred_gpu=GPU_3090)


async def _run_load(env: Env, fail_at: int | None, exc: BaseException | None):
    async def run():
        env.faults.arm(fail_at, exc)
        try:
            return await env.mm.load("big")
        finally:
            env.faults.task = None

    task = asyncio.create_task(run())
    results = await asyncio.gather(task, return_exceptions=True)
    return results[0]


async def _count_points() -> int:
    with Env(publish_lag_s=_LAG_S) as env:
        await _eviction_scenario(env)
        result = await _run_load(env, None, None)
        assert not isinstance(result, BaseException), result
        assert env.backend.unloads == ["od"]
        return env.faults.count


@pytest.mark.parametrize("kind", ["error", "cancel"])
async def test_a_failure_at_any_await_of_an_evicting_load_leaves_nothing_behind(kind):
    total = await _count_points()
    assert total > 10
    outcomes: dict[str, set[str]] = {}
    for index in range(total):
        exc: BaseException = (
            RuntimeError(f"injected@{index}") if kind == "error" else asyncio.CancelledError()
        )
        with Env(publish_lag_s=_LAG_S) as env:
            await _eviction_scenario(env)
            result = await _run_load(env, index, exc)
            point = env.faults.fired or "-"
            big = env.status("big")
            outcomes.setdefault(point, set()).add(big.value)
            if isinstance(result, BaseException):
                assert big != ModelStatus.LOADED, (index, point)
                if kind == "cancel":
                    assert big == ModelStatus.UNLOADED, (index, point, big)
                else:
                    assert big in (ModelStatus.ERROR, ModelStatus.CONFIGURED), (index, point, big)
            else:
                assert big == ModelStatus.LOADED, (index, point)
            assert_quiescent(env)
            # Physical truth matches the books: every LOADED model has its
            # process, nothing else is resident on the 3090.
            loaded_pids = {
                env.backend.pids.get(m)
                for m, s in env.mm._states.items()
                if s.status == ModelStatus.LOADED
            }
            alive = set(env.world.live[GPU_3090]) | set(env.world.live[GPU_3060])
            stray = alive - loaded_pids
            erred = {
                env.backend.pids.get(m)
                for m, s in env.mm._states.items()
                if s.status == ModelStatus.ERROR
            }
            assert stray <= erred, (index, point, stray)

            # And the next load works from whatever was left.
            env.faults.arm(None)
            env.faults.task = None
            if env.status("big") != ModelStatus.LOADED:
                await env.mm.load("big")
            assert env.status("big") == ModelStatus.LOADED, (index, point)
            assert env.mm._states["big"].current_gpu == [GPU_3090]
            assert_quiescent(env)
    assert outcomes


async def test_cancelling_a_load_during_its_drain_wait_cleans_up(env, monkeypatch):
    monkeypatch.setattr(settings, "pressure_eviction_drain_timeout_s", 30)
    await _three_on_the_3090(env)
    env.mm.begin_request("od")
    env.configure("new", 8_000, preferred_gpu=GPU_3090)
    load = asyncio.create_task(env.mm.load("new"))
    for _ in range(100):
        if env.mm._eviction_wait_for.get("new") == "od":
            break
        await asyncio.sleep(0.02)
    assert env.mm._arbiter.is_waiting_on("od")
    load.cancel()
    await asyncio.gather(load, return_exceptions=True)
    assert env.status("new") == ModelStatus.UNLOADED
    assert env.status("od") == ModelStatus.LOADED
    # The cancel cleanup always tells the backend to stop "new" (a no-op here).
    assert "od" not in env.backend.unloads
    assert_quiescent(env)


async def test_two_loads_racing_for_the_same_card_never_overcommit_it(env):
    await _full_3060(env)
    await env.resident("od", 10_000, [GPU_3090], last_used_s_ago=10)
    env.configure("a", 12_000, preferred_gpu=GPU_3090)
    env.configure("b", 12_000, preferred_gpu=GPU_3090)
    started = time.monotonic()
    results = await asyncio.gather(env.mm.load("a"), env.mm.load("b"), return_exceptions=True)
    assert time.monotonic() - started < 30
    loaded = [m for m in ("a", "b") if env.status(m) == ModelStatus.LOADED]
    used = sum(env.world.live[GPU_3090].values())
    assert used <= env.world.totals[GPU_3090], (results, env.world.live)
    assert loaded
    assert_quiescent(env)


async def test_an_eviction_already_visible_is_not_waited_out_again():
    with Env(publish_lag_s=0.0) as env:
        await _three_on_the_3090(env)
        env.configure("new", 8_000, preferred_gpu=GPU_3090)
        started = time.monotonic()
        await env.mm.load("new")
        assert env.backend.unloads == ["od"]
        assert time.monotonic() - started < 3.0


async def test_a_cancelled_unload_can_be_finished_by_the_next_one(env):
    """Cancelled while the worker is stopping: ERROR (not UNLOADING for good),
    and a second unload releases the process and its locks."""
    await env.resident("od", 6_000, [GPU_3090])
    stopping = asyncio.Event()
    release = asyncio.Event()
    original = env.backend.unload

    async def slow_unload(model_id):
        stopping.set()
        await release.wait()
        await original(model_id)

    env.backend.unload = slow_unload
    task = asyncio.create_task(env.mm.unload("od"))
    await asyncio.wait_for(stopping.wait(), timeout=2)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert env.status("od") == ModelStatus.ERROR
    env.backend.unload = original
    await env.mm.unload("od")
    assert env.status("od") == ModelStatus.UNLOADED
    assert env.world.live[GPU_3090] == {}
    assert_quiescent(env)


async def test_a_release_hw_monitor_has_not_published_yet_is_read_live():
    """The poll lags (hw-monitor ~2 s, here far longer): the waits read NVML
    themselves, so one eviction is enough and is not waited out twice. With
    the polled figure only, the retry judged the card still full and evicted
    the next candidate too."""
    with Env(publish_lag_s=60.0) as env:
        env.world.publish()
        await _three_on_the_3090(env)
        env.world.publish()
        env.configure("new", 8_000, preferred_gpu=GPU_3090)
        started = time.monotonic()
        await env.mm.load("new")
        assert env.backend.unloads == ["od"]
        assert time.monotonic() - started < 3.0
        assert_quiescent(env)


async def test_without_nvml_the_waits_still_follow_the_poll():
    """No live reading: the waits end when hw-monitor publishes the release."""
    with Env(publish_lag_s=0.5) as env:
        env.world.nvml = False
        await _three_on_the_3090(env)
        await asyncio.sleep(0.6)
        env.configure("new", 8_000, preferred_gpu=GPU_3090)
        started = time.monotonic()
        await env.mm.load("new")
        assert env.backend.unloads == ["od"]
        assert time.monotonic() - started < 5.0
