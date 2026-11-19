"""Mundo de VRAM falso para los tests de caracterización del árbitro (REF-03).

Dos tarjetas como las de producción (índice 0 = RTX 3060 de 12 GB, índice 1 =
RTX 3090 de 24 GB), un ``GPUManager`` real al que solo se le cambia de dónde
lee (el mundo en vez de NVML/hw-monitor), el ``GPUScheduler`` real, un
backend falso que ocupa y libera memoria de verdad en el mundo y Redis falso.

``Faults`` es el inyector: cada ``await`` de los colaboradores falsos (GPU,
backend, Redis, estadísticas) pasa por ``Faults.point``; con ``fail_at=N`` el
N-ésimo de la tarea vigilada lanza la excepción pedida. Así un test recorre
todos los puntos de suspensión de una carga sin conocer el código por dentro.
"""

from __future__ import annotations

import asyncio
import itertools
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

from ocabra.backends._mock import MockBackend
from ocabra.backends.base import WorkerInfo
from ocabra.core.gpu_manager import GPUManager, GPUProcessInfo, GPUState
from ocabra.core.model_manager import LoadPolicy, ModelManager, ModelState, ModelStatus
from ocabra.core.scheduler import GPUScheduler
from ocabra.core.worker_pool import WorkerPool

GPU_3060 = 0
GPU_3090 = 1
TOTALS = {GPU_3060: 12_288, GPU_3090: 24_576}


class Faults:
    """Counts the awaits of the watched task and fails the ``fail_at``-th."""

    def __init__(self) -> None:
        self.count = 0
        self.trace: list[str] = []
        self.fail_at: int | None = None
        self.exc: BaseException | None = None
        self.task: asyncio.Task | None = None
        self.fired: str | None = None

    def arm(self, fail_at: int | None, exc: BaseException | None = None) -> None:
        self.count = 0
        self.trace = []
        self.fail_at = fail_at
        self.exc = exc
        self.fired = None
        self.task = asyncio.current_task()

    async def point(self, name: str) -> None:
        if self.task is None or asyncio.current_task() is not self.task:
            await asyncio.sleep(0)
            return
        index = self.count
        self.count += 1
        self.trace.append(name)
        if self.fail_at is not None and index == self.fail_at and self.exc is not None:
            self.fired = name
            raise self.exc
        await asyncio.sleep(0)


@dataclass
class World:
    """Physical truth (``live``) and what hw-monitor last published."""

    totals: dict[int, int] = field(default_factory=lambda: dict(TOTALS))
    live: dict[int, dict[int, int]] = field(default_factory=dict)
    published: dict[int, dict[int, int]] = field(default_factory=dict)
    # Seconds a change takes to reach the polled figure (hw-monitor ~2 s).
    publish_lag_s: float = 0.0
    _pids: Any = field(default_factory=lambda: itertools.count(1000))

    def __post_init__(self) -> None:
        for gpu in self.totals:
            self.live.setdefault(gpu, {})
            self.published.setdefault(gpu, {})

    def new_pid(self) -> int:
        return next(self._pids)

    def _changed(self) -> None:
        if self.publish_lag_s <= 0:
            self.publish()
            return
        snapshot = {gpu: dict(procs) for gpu, procs in self.live.items()}
        loop = asyncio.get_running_loop()
        loop.call_later(self.publish_lag_s, self._publish_snapshot, snapshot)

    def _publish_snapshot(self, snapshot: dict[int, dict[int, int]]) -> None:
        self.published = snapshot

    def publish(self) -> None:
        self.published = {gpu: dict(procs) for gpu, procs in self.live.items()}

    def allocate(self, pid: int, per_gpu: dict[int, int]) -> None:
        for gpu, mb in per_gpu.items():
            self.live[gpu][pid] = self.live[gpu].get(pid, 0) + int(mb)
        self._changed()

    def release(self, pid: int) -> None:
        for procs in self.live.values():
            procs.pop(pid, None)
        self._changed()

    def live_free(self, gpu: int) -> int:
        return self.totals[gpu] - sum(self.live[gpu].values())

    def published_free(self, gpu: int) -> int:
        return self.totals[gpu] - sum(self.published[gpu].values())


class FakeGPUManager(GPUManager):
    """The real GPUManager reading the world instead of NVML / hw-monitor."""

    def __init__(self, world: World, faults: Faults) -> None:
        super().__init__()
        self.world = world
        self.faults = faults
        for gpu in world.totals:
            self._locks[gpu] = {}
        self._sync()

    def _sync(self) -> None:
        for gpu, total in self.world.totals.items():
            used = sum(self.world.published[gpu].values())
            self._states[gpu] = GPUState(
                index=gpu,
                name=f"gpu{gpu}",
                total_vram_mb=total,
                free_vram_mb=total - used,
                used_vram_mb=used,
                utilization_pct=0.0,
                temperature_c=40.0,
                power_draw_w=50.0,
                power_limit_w=300.0,
                processes=[
                    GPUProcessInfo(
                        pid=pid, process_name=None, process_type="compute", used_vram_mb=mb
                    )
                    for pid, mb in self.world.published[gpu].items()
                ],
            )

    # Readings ---------------------------------------------------------------

    def _read_reserved_mb(self, index: int) -> int | None:
        return 0

    def _read_local_processes(self, index: int) -> list[GPUProcessInfo] | None:
        return [
            GPUProcessInfo(pid=pid, process_name=None, process_type="compute", used_vram_mb=mb)
            for pid, mb in self.world.live[index].items()
        ]

    def live_free_vram_mb(self, gpu_index: int) -> int | None:
        return self.world.live_free(gpu_index)

    def raw_compute_pids(self, gpu_indices: list[int]) -> set[int]:
        return {pid for gpu in gpu_indices for pid in self.world.live[gpu]}

    async def get_all_states(self) -> list[GPUState]:
        await self.faults.point("gpu.get_all_states")
        self._sync()
        return await super().get_all_states()

    async def get_state(self, index: int) -> GPUState:
        await self.faults.point("gpu.get_state")
        self._sync()
        return await super().get_state(index)

    async def get_free_vram(self, gpu_index: int) -> int:
        await self.faults.point("gpu.get_free_vram")
        self._sync()
        return await super().get_free_vram(gpu_index)

    async def get_physical_free_vram(self, gpu_index: int) -> int:
        await self.faults.point("gpu.get_physical_free_vram")
        self._sync()
        return await super().get_physical_free_vram(gpu_index)

    async def get_free_vram_if_released(self, gpu_index: int, released) -> int:
        await self.faults.point("gpu.get_free_vram_if_released")
        self._sync()
        return await super().get_free_vram_if_released(gpu_index, released)

    async def lock_vram(self, gpu_index: int, amount_mb: int, model_id: str) -> None:
        await self.faults.point("gpu.lock_vram")
        await super().lock_vram(gpu_index, amount_mb, model_id)

    async def unlock_vram(self, gpu_index: int, model_id: str) -> None:
        await self.faults.point("gpu.unlock_vram")
        await super().unlock_vram(gpu_index, model_id)


class WorldBackend(MockBackend):
    """Backend whose workers really take (and give back) memory in the world.

    ``sizes`` maps backend_model_id to MB; a multi-GPU load splits it evenly
    unless ``splits`` gives the per-GPU share.
    """

    def __init__(self, world: World, faults: Faults) -> None:
        super().__init__(load_delay=0)
        self.world = world
        self.faults = faults
        self.sizes: dict[str, int] = {}
        self.splits: dict[str, dict[int, int]] = {}
        self.pids: dict[str, int] = {}
        self.loads: list[tuple[str, list[int]]] = []
        self.unloads: list[str] = []
        # Optional gates to hold a load or an unload mid-way.
        self.load_started: asyncio.Event = asyncio.Event()
        self.hold_load: asyncio.Event | None = None

    async def get_vram_estimate_mb(self, model_id: str, extra_config=None) -> int:
        await self.faults.point("backend.get_vram_estimate_mb")
        return self.sizes.get(model_id, 4096)

    async def load(self, model_id: str, gpu_indices: list[int], **kwargs) -> WorkerInfo:
        await self.faults.point("backend.load")
        self.load_started.set()
        if self.hold_load is not None:
            await self.hold_load.wait()
        size = self.sizes.get(model_id, 4096)
        if gpu_indices:
            per_gpu = self.splits.get(model_id) or {
                gpu: size // len(gpu_indices) for gpu in gpu_indices
            }
        else:
            per_gpu = {}
        pid = self.world.new_pid()
        self.world.allocate(pid, per_gpu)
        self.pids[model_id] = pid
        self.loads.append((model_id, list(gpu_indices)))
        return WorkerInfo(
            backend_type="mock",
            model_id=model_id,
            gpu_indices=list(gpu_indices),
            port=kwargs.get("port") or 0,
            pid=pid,
            vram_used_mb=size,
        )

    async def unload(self, model_id: str) -> None:
        await self.faults.point("backend.unload")
        self.unloads.append(model_id)
        pid = self.pids.pop(model_id, None)
        if pid is not None:
            self.world.release(pid)

    async def get_capabilities(self, model_id: str):
        await self.faults.point("backend.get_capabilities")
        return await super().get_capabilities(model_id)


class Env:
    """One ModelManager wired to the world. Use as ``with Env() as env``."""

    def __init__(self, *, publish_lag_s: float = 0.0) -> None:
        self.faults = Faults()
        self.world = World(publish_lag_s=publish_lag_s)
        self.gpu = FakeGPUManager(self.world, self.faults)
        self.backend = WorldBackend(self.world, self.faults)
        self.pool = WorkerPool()
        self.pool.register_backend("mock", self.backend)
        self.scheduler = GPUScheduler(self.gpu)
        self.mm = ModelManager(self.pool, gpu_manager=self.gpu, gpu_scheduler=self.scheduler)
        self.scheduler.set_model_manager(self.mm)
        self.events: list[tuple[str, str]] = []
        self._patches: list[Any] = []

        async def _publish(channel, payload):
            await self.faults.point("redis.publish")
            if isinstance(payload, dict) and "event" in payload:
                self.events.append((payload.get("model_id"), payload["event"]))

        async def _set_key(key, value, *args, **kwargs):
            await self.faults.point("redis.set_key")

        async def _alert(*args, **kwargs):
            await self.faults.point("redis.alert")

        async def _load_stat(**kwargs):
            await self.faults.point("db.load_stat")

        self._publish, self._set_key, self._alert = _publish, _set_key, _alert
        self.mm._record_model_load_stat = _load_stat  # type: ignore[method-assign]

    def __enter__(self) -> Env:
        self._patches = [
            patch("ocabra.core.model_manager.publish", new=self._publish),
            patch("ocabra.core.model_manager.set_key", new=self._set_key),
            patch("ocabra.core.model_manager.publish_system_alert", new=self._alert),
        ]
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc) -> None:
        for p in reversed(self._patches):
            p.stop()

    # Model helpers ------------------------------------------------------------

    def configure(
        self,
        model_id: str,
        size_mb: int,
        *,
        policy: LoadPolicy = LoadPolicy.ON_DEMAND,
        preferred_gpu: int | None = None,
        pin_permanent: bool = False,
        split: dict[int, int] | None = None,
        extra_config: dict | None = None,
    ) -> ModelState:
        state = ModelState(
            model_id=model_id,
            display_name=model_id,
            backend_type="mock",
            load_policy=policy,
            preferred_gpu=preferred_gpu,
            extra_config=dict(extra_config or {}),
        )
        state.pin_permanent = pin_permanent
        self.mm._states[model_id] = state
        self.backend.sizes[model_id] = size_mb
        if split:
            self.backend.splits[model_id] = dict(split)
        return state

    async def resident(
        self,
        model_id: str,
        size_mb: int,
        gpus: list[int],
        *,
        policy: LoadPolicy = LoadPolicy.ON_DEMAND,
        last_used_s_ago: float | None = None,
        pin_permanent: bool = False,
        split: dict[int, int] | None = None,
    ) -> ModelState:
        """A model already LOADED as a load would have left it (lock included)."""
        from datetime import UTC, datetime, timedelta

        state = self.configure(
            model_id, size_mb, policy=policy, pin_permanent=pin_permanent, split=split
        )
        info = await self.backend.load(model_id, list(gpus))
        state.worker_info = info
        state.current_gpu = list(gpus)
        state.vram_used_mb = size_mb
        state.vram_estimate_mb = size_mb
        state.status = ModelStatus.LOADED
        now = datetime.now(UTC)
        state.loaded_at = now - timedelta(hours=1)
        if last_used_s_ago is not None:
            state.last_request_at = now - timedelta(seconds=last_used_s_ago)
        per_gpu = self.backend.splits.get(model_id) or {g: size_mb // len(gpus) for g in gpus}
        for gpu, mb in per_gpu.items():
            await self.gpu.lock_vram(gpu, mb, model_id)
        self.pool.set_worker(model_id, info)
        self.backend.loads.clear()
        return state

    def status(self, model_id: str) -> ModelStatus:
        return self.mm._states[model_id].status


def assert_quiescent(env: Env) -> None:
    """Nothing of an operation may outlive it, however it ended."""
    mm = env.mm
    assert not mm._arbiter.locked(), "arbiter still held"
    assert not mm._load_in_progress, mm._load_in_progress
    assert mm._load_queue_waiters == 0
    assert not mm._drain_deadlines, mm._drain_deadlines
    assert not mm._eviction_claims, mm._eviction_claims
    assert not mm._eviction_wait_for, mm._eviction_wait_for
    assert not mm._load_tasks, mm._load_tasks
    assert not mm._load_lock_owners, mm._load_lock_owners
    for lock in mm._load_locks.values():
        assert not lock.locked()
    for model_id, state in mm._states.items():
        assert state.status not in (ModelStatus.LOADING, ModelStatus.UNLOADING), (
            model_id,
            state.status,
        )
        locked = env.gpu.locked_gpu_indices(model_id)
        if state.status == ModelStatus.LOADED:
            assert env.pool.get_worker(model_id) is not None, model_id
            assert set(locked) == set(state.current_gpu), (model_id, locked, state.current_gpu)
        elif state.status in (ModelStatus.UNLOADED, ModelStatus.CONFIGURED):
            assert not locked, (model_id, locked)
            assert env.pool.get_worker(model_id) is None, model_id
            assert state.current_gpu == [], (model_id, state.current_gpu)
