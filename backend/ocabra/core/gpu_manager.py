import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import pynvml
import structlog

from ocabra.config import settings
from ocabra.redis_client import publish, set_key

logger = structlog.get_logger(__name__)


@dataclass
class GPUState:
    index: int
    name: str
    total_vram_mb: int
    free_vram_mb: int
    used_vram_mb: int
    utilization_pct: float
    temperature_c: float
    power_draw_w: float
    power_limit_w: float
    locked_vram_mb: int = 0


class GPUManager:
    def __init__(self) -> None:
        self._states: dict[int, GPUState] = {}
        self._locks: dict[int, dict[str, int]] = {}  # gpu_index → {model_id: vram_mb}
        self._poll_task: asyncio.Task | None = None
        self._poll_history: dict[int, list[dict]] = {}

    async def start(self) -> None:
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            self._locks[i] = {}
            self._poll_history[i] = []
            self._states[i] = self._read_gpu(i)
        self._poll_task = asyncio.create_task(self._poll_loop())
        logger.info("gpu_manager_started", gpu_count=count)

    async def stop(self) -> None:
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass

    def _read_gpu(self, index: int) -> GPUState:
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        try:
            power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
            power_limit_mw = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
            power_w = power_mw / 1000.0
            power_limit_w = power_limit_mw / 1000.0
        except pynvml.NVMLError:
            power_w = 0.0
            power_limit_w = 0.0
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode()
        locked = sum(self._locks.get(index, {}).values())
        return GPUState(
            index=index,
            name=name,
            total_vram_mb=mem.total // (1024 * 1024),
            free_vram_mb=mem.free // (1024 * 1024),
            used_vram_mb=mem.used // (1024 * 1024),
            utilization_pct=float(util.gpu),
            temperature_c=float(temp),
            power_draw_w=power_w,
            power_limit_w=power_limit_w,
            locked_vram_mb=locked,
        )

    async def _poll_loop(self) -> None:
        while True:
            try:
                states = []
                for i in self._states:
                    state = self._read_gpu(i)
                    self._states[i] = state
                    states.append(asdict(state))
                    self._poll_history[i].append(asdict(state))
                    await set_key(f"gpu:state:{i}", asdict(state), ttl=5)
                await publish("gpu:stats", states)
                # Aggregate to DB every 30 polls (~60s)
                if self._poll_history and len(self._poll_history[0]) >= 30:
                    await self._persist_stats()
            except Exception as e:
                logger.error("gpu_poll_error", error=str(e))
            await asyncio.sleep(2)

    async def _persist_stats(self) -> None:
        from ocabra.database import AsyncSessionLocal
        from ocabra.db.stats import GpuStat

        now = datetime.now(timezone.utc)
        async with AsyncSessionLocal() as session:
            for i, history in self._poll_history.items():
                if not history:
                    continue
                n = len(history)
                session.add(
                    GpuStat(
                        recorded_at=now,
                        gpu_index=i,
                        utilization_pct=sum(h["utilization_pct"] for h in history) / n,
                        vram_used_mb=int(sum(h["used_vram_mb"] for h in history) / n),
                        power_draw_w=sum(h["power_draw_w"] for h in history) / n,
                        temperature_c=sum(h["temperature_c"] for h in history) / n,
                    )
                )
                self._poll_history[i] = []
            await session.commit()

    async def get_all_states(self) -> list[GPUState]:
        return list(self._states.values())

    async def get_state(self, index: int) -> GPUState:
        if index not in self._states:
            raise KeyError(f"GPU {index} not found")
        return self._states[index]

    async def lock_vram(self, gpu_index: int, amount_mb: int, model_id: str) -> None:
        self._locks.setdefault(gpu_index, {})[model_id] = amount_mb
        logger.debug("vram_locked", gpu=gpu_index, model=model_id, mb=amount_mb)

    async def unlock_vram(self, gpu_index: int, model_id: str) -> None:
        self._locks.get(gpu_index, {}).pop(model_id, None)
        logger.debug("vram_unlocked", gpu=gpu_index, model=model_id)

    async def get_free_vram(self, gpu_index: int) -> int:
        state = self._states[gpu_index]
        locked = sum(self._locks.get(gpu_index, {}).values())
        return max(0, state.free_vram_mb - locked - settings.vram_buffer_mb)
