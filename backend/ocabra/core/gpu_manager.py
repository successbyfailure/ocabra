import asyncio
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pynvml
import structlog

from ocabra.config import settings
from ocabra.redis_client import publish, set_key

logger = structlog.get_logger(__name__)


@dataclass
class GPUProcessInfo:
    pid: int
    process_name: str | None
    process_type: str
    used_vram_mb: int


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
    processes: list[GPUProcessInfo] = field(default_factory=list)


class GPUManager:
    def __init__(self) -> None:
        self._states: dict[int, GPUState] = {}
        self._locks: dict[int, dict[str, int]] = {}  # gpu_index → {model_id: vram_mb}
        self._poll_task: asyncio.Task | None = None
        self._poll_history: dict[int, list[dict]] = {}
        self._running: bool = False

    async def start(self) -> None:
        try:
            pynvml.nvmlInit()
        except pynvml.NVMLError as e:
            logger.warning("nvml_unavailable", error=str(e), detail="GPU monitoring disabled")
            self._running = True
            return
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            self._locks[i] = {}
            self._poll_history[i] = []
            self._states[i] = self._read_gpu(i)
        self._poll_task = asyncio.create_task(self._poll_loop())
        self._running = True
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
        processes = self._read_gpu_processes(handle)
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
            processes=processes,
        )

    def _read_gpu_processes(self, handle) -> list[GPUProcessInfo]:
        process_map: dict[tuple[str, int], GPUProcessInfo] = {}

        for process in self._query_nvml_processes(
            handle,
            query_names=(
                "nvmlDeviceGetComputeRunningProcesses_v3",
                "nvmlDeviceGetComputeRunningProcesses_v2",
                "nvmlDeviceGetComputeRunningProcesses",
            ),
            process_type="compute",
        ):
            process_map[(process.process_type, process.pid)] = process

        for process in self._query_nvml_processes(
            handle,
            query_names=(
                "nvmlDeviceGetGraphicsRunningProcesses_v3",
                "nvmlDeviceGetGraphicsRunningProcesses_v2",
                "nvmlDeviceGetGraphicsRunningProcesses",
            ),
            process_type="graphics",
        ):
            process_map[(process.process_type, process.pid)] = process

        processes = list(process_map.values())
        processes.sort(key=lambda item: (-item.used_vram_mb, item.pid))
        return processes

    def _query_nvml_processes(
        self,
        handle,
        query_names: tuple[str, ...],
        process_type: str,
    ) -> list[GPUProcessInfo]:
        for query_name in query_names:
            query = getattr(pynvml, query_name, None)
            if not callable(query):
                continue
            try:
                nvml_processes = query(handle)
            except pynvml.NVMLError:
                continue
            return [self._to_process_info(process, process_type) for process in nvml_processes]
        return []

    def _to_process_info(self, nvml_process, process_type: str) -> GPUProcessInfo:
        pid = int(getattr(nvml_process, "pid", 0) or 0)
        used_vram_mb = self._normalize_vram_mb(getattr(nvml_process, "usedGpuMemory", 0))
        return GPUProcessInfo(
            pid=pid,
            process_name=self._read_process_name(pid),
            process_type=process_type,
            used_vram_mb=used_vram_mb,
        )

    def _normalize_vram_mb(self, used_gpu_memory: int | None) -> int:
        if used_gpu_memory in {None, 0}:
            return 0

        unavailable = getattr(pynvml, "NVML_VALUE_NOT_AVAILABLE", None)
        if unavailable is not None and used_gpu_memory == unavailable:
            return 0

        try:
            used = int(used_gpu_memory)
        except (TypeError, ValueError):
            return 0

        if used < 0:
            return 0
        return used // (1024 * 1024)

    def _read_process_name(self, pid: int) -> str | None:
        if pid <= 0:
            return None

        cmdline = Path(f"/proc/{pid}/cmdline")
        try:
            raw = cmdline.read_bytes()
            if raw:
                first = raw.split(b"\x00", 1)[0].decode(errors="ignore").strip()
                if first:
                    return Path(first).name
        except OSError:
            pass

        comm = Path(f"/proc/{pid}/comm")
        try:
            value = comm.read_text(encoding="utf-8", errors="ignore").strip()
            if value:
                return value
        except OSError:
            pass

        return None

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

    def get_state_nowait(self, index: int) -> GPUState | None:
        """Return the most recently cached GPU state without awaiting the poll loop."""
        return self._states.get(index)

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
