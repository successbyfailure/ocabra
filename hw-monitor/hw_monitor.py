"""hw-monitor — Standalone hardware monitoring service for oCabra.

Polls GPU (pynvml) and CPU (RAPL / hwmon) sensors, publishes live state
to Redis, and persists per-minute averages to PostgreSQL.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pynvml
import redis.asyncio as aioredis
import structlog
from sqlalchemy import Column, DateTime, Float, Index, Integer, String, Text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql+asyncpg://ocabra:ocabra@postgres:5432/ocabra")
POLL_INTERVAL_S = float(os.environ.get("POLL_INTERVAL_S", "2"))
PERSIST_INTERVAL_POLLS = int(os.environ.get("PERSIST_INTERVAL_POLLS", "30"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)
logger = structlog.get_logger("hw-monitor")

# ---------------------------------------------------------------------------
# SQLAlchemy models (standalone — mirrors backend ORM)
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    pass


class GpuStat(Base):
    __tablename__ = "gpu_stats"

    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, primary_key=True)
    gpu_index: Mapped[int] = mapped_column(Integer, nullable=False, primary_key=True)
    utilization_pct: Mapped[float | None] = mapped_column(Float)
    vram_used_mb: Mapped[int | None] = mapped_column(Integer)
    power_draw_w: Mapped[float | None] = mapped_column(Float)
    temperature_c: Mapped[float | None] = mapped_column(Float)

    __table_args__ = (
        Index("ix_gpu_stats_recorded_at_gpu", "recorded_at", "gpu_index"),
    )


class ServerStat(Base):
    __tablename__ = "server_stats"

    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, primary_key=True)
    cpu_power_w: Mapped[float | None] = mapped_column(Float)
    cpu_temp_c: Mapped[float | None] = mapped_column(Float)
    total_gpu_power_w: Mapped[float | None] = mapped_column(Float)
    total_power_w: Mapped[float | None] = mapped_column(Float)

    __table_args__ = (
        Index("ix_server_stats_recorded_at", "recorded_at"),
    )


# ---------------------------------------------------------------------------
# DB engine (lazy init)
# ---------------------------------------------------------------------------

_engine = None
_session_factory = None


def _get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _engine, _session_factory
    if _session_factory is None:
        _engine = create_async_engine(DATABASE_URL, pool_pre_ping=True, pool_size=2, max_overflow=2)
        _session_factory = async_sessionmaker(_engine, class_=AsyncSession, expire_on_commit=False)
    return _session_factory


# ---------------------------------------------------------------------------
# Redis client
# ---------------------------------------------------------------------------

_redis: aioredis.Redis | None = None


async def _get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)
    return _redis


# ---------------------------------------------------------------------------
# CPU sensor helpers
# ---------------------------------------------------------------------------

RAPL_ENERGY_PATH = Path("/sys/class/powercap/intel-rapl:0/energy_uj")
RAPL_MAX_ENERGY_PATH = Path("/sys/class/powercap/intel-rapl:0/max_energy_range_uj")


def _rapl_available() -> bool:
    return RAPL_ENERGY_PATH.exists()


def _read_rapl_energy_uj() -> int | None:
    try:
        return int(RAPL_ENERGY_PATH.read_text().strip())
    except (OSError, ValueError):
        return None


def _read_rapl_max_energy_uj() -> int:
    try:
        return int(RAPL_MAX_ENERGY_PATH.read_text().strip())
    except (OSError, ValueError):
        return 2**63  # effectively no wrap-around


def _find_k10temp_dir() -> Path | None:
    """Locate the hwmon directory whose ``name`` file contains ``k10temp``."""
    hwmon_base = Path("/sys/class/hwmon")
    if not hwmon_base.exists():
        return None
    for entry in hwmon_base.iterdir():
        name_file = entry / "name"
        try:
            if name_file.read_text().strip() == "k10temp":
                return entry
        except OSError:
            continue
    return None


def _read_cpu_temp_c(hwmon_dir: Path | None) -> float | None:
    if hwmon_dir is None:
        return None
    try:
        raw = (hwmon_dir / "temp1_input").read_text().strip()
        return int(raw) / 1000.0
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------


def _read_process_name(pid: int) -> str | None:
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


def _normalize_vram_mb(used_gpu_memory: int | None) -> int:
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


def _query_nvml_processes(handle, query_names: tuple[str, ...], process_type: str) -> list[dict]:
    for query_name in query_names:
        query = getattr(pynvml, query_name, None)
        if not callable(query):
            continue
        try:
            nvml_processes = query(handle)
        except pynvml.NVMLError:
            continue
        results = []
        for p in nvml_processes:
            pid = int(getattr(p, "pid", 0) or 0)
            used_vram_mb = _normalize_vram_mb(getattr(p, "usedGpuMemory", 0))
            results.append({
                "pid": pid,
                "process_name": _read_process_name(pid),
                "process_type": process_type,
                "used_vram_mb": used_vram_mb,
            })
        return results
    return []


def _read_gpu_processes(handle) -> list[dict]:
    process_map: dict[tuple[str, int], dict] = {}

    for proc in _query_nvml_processes(
        handle,
        query_names=(
            "nvmlDeviceGetComputeRunningProcesses_v3",
            "nvmlDeviceGetComputeRunningProcesses_v2",
            "nvmlDeviceGetComputeRunningProcesses",
        ),
        process_type="compute",
    ):
        process_map[(proc["process_type"], proc["pid"])] = proc

    for proc in _query_nvml_processes(
        handle,
        query_names=(
            "nvmlDeviceGetGraphicsRunningProcesses_v3",
            "nvmlDeviceGetGraphicsRunningProcesses_v2",
            "nvmlDeviceGetGraphicsRunningProcesses",
        ),
        process_type="graphics",
    ):
        process_map[(proc["process_type"], proc["pid"])] = proc

    processes = list(process_map.values())
    processes.sort(key=lambda item: (-item["used_vram_mb"], item["pid"]))
    return processes


def _read_gpu(index: int) -> dict:
    """Read GPU state and return a dict matching GPUState dataclass fields."""
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
    processes = _read_gpu_processes(handle)
    return {
        "index": index,
        "name": name,
        "total_vram_mb": mem.total // (1024 * 1024),
        "free_vram_mb": mem.free // (1024 * 1024),
        "used_vram_mb": mem.used // (1024 * 1024),
        "utilization_pct": float(util.gpu),
        "temperature_c": float(temp),
        "power_draw_w": power_w,
        "power_limit_w": power_limit_w,
        "locked_vram_mb": 0,
        "processes": processes,
    }


# ---------------------------------------------------------------------------
# Power limit management (Redis subscriber)
# ---------------------------------------------------------------------------


async def _power_limit_listener() -> None:
    """Subscribe to ``gpu:set_power_limit`` and apply power limits via NVML."""
    r = await _get_redis()
    pubsub = r.pubsub()
    await pubsub.subscribe("gpu:set_power_limit")
    logger.info("power_limit_listener_started")
    try:
        async for message in pubsub.listen():
            if message["type"] != "message":
                continue
            try:
                data = json.loads(message["data"])
                gpu_index = int(data["gpu_index"])
                limit_w = float(data["limit_w"])
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                limit_mw = int(limit_w * 1000)
                pynvml.nvmlDeviceSetPowerManagementLimit(handle, limit_mw)
                logger.info("power_limit_set", gpu=gpu_index, limit_w=limit_w)
            except Exception as exc:
                logger.error("power_limit_error", error=str(exc))
    finally:
        await pubsub.unsubscribe("gpu:set_power_limit")
        await pubsub.aclose()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


async def main() -> None:
    # Init pynvml — GPUs are essential
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError as exc:
        logger.error("nvml_init_failed", error=str(exc))
        sys.exit(1)

    gpu_count = pynvml.nvmlDeviceGetCount()
    logger.info("nvml_initialized", gpu_count=gpu_count)

    # CPU sensor probing
    rapl_ok = _rapl_available()
    if not rapl_ok:
        logger.warning("rapl_not_available", detail="CPU power will be None")

    k10temp_dir = _find_k10temp_dir()
    if k10temp_dir is None:
        logger.warning("k10temp_not_found", detail="CPU temperature will be None")
    else:
        logger.info("k10temp_found", path=str(k10temp_dir))

    rapl_max_uj = _read_rapl_max_energy_uj() if rapl_ok else 0

    r = await _get_redis()

    # Start power-limit listener as background task
    power_limit_task = asyncio.create_task(_power_limit_listener())

    # State for RAPL delta calculation
    prev_energy_uj: int | None = _read_rapl_energy_uj() if rapl_ok else None
    prev_energy_time: float = time.monotonic()

    # Accumulation buffers for persistence
    gpu_history: dict[int, list[dict]] = {i: [] for i in range(gpu_count)}
    server_history: list[dict] = []
    poll_count = 0

    shutdown = asyncio.Event()

    def _signal_handler() -> None:
        shutdown.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    logger.info("poll_loop_starting", interval_s=POLL_INTERVAL_S, persist_every=PERSIST_INTERVAL_POLLS)

    while not shutdown.is_set():
        try:
            # --- GPU polling ---
            gpu_states: list[dict] = []
            total_gpu_power_w = 0.0
            for i in range(gpu_count):
                state = _read_gpu(i)
                gpu_states.append(state)
                total_gpu_power_w += state["power_draw_w"]
                gpu_history[i].append(state)
                await r.setex(f"gpu:state:{i}", 5, json.dumps(state))
            await r.publish("gpu:stats", json.dumps(gpu_states))

            # --- CPU polling ---
            cpu_power_w: float | None = None
            now_mono = time.monotonic()
            if rapl_ok and prev_energy_uj is not None:
                cur_energy_uj = _read_rapl_energy_uj()
                if cur_energy_uj is not None:
                    delta_uj = cur_energy_uj - prev_energy_uj
                    if delta_uj < 0:
                        # Counter wrapped
                        delta_uj += rapl_max_uj
                    delta_s = now_mono - prev_energy_time
                    if delta_s > 0:
                        cpu_power_w = delta_uj / (delta_s * 1_000_000)
                    prev_energy_uj = cur_energy_uj
                    prev_energy_time = now_mono
            elif rapl_ok:
                prev_energy_uj = _read_rapl_energy_uj()
                prev_energy_time = now_mono

            cpu_temp_c = _read_cpu_temp_c(k10temp_dir)

            total_power_w: float | None = None
            if cpu_power_w is not None:
                total_power_w = cpu_power_w + total_gpu_power_w
            elif total_gpu_power_w > 0:
                total_power_w = total_gpu_power_w

            server_power = {
                "cpu_power_w": cpu_power_w,
                "cpu_temp_c": cpu_temp_c,
                "total_gpu_power_w": total_gpu_power_w,
                "total_power_w": total_power_w,
            }
            await r.setex("server:power", 5, json.dumps(server_power))
            server_history.append(server_power)

            # --- Heartbeat ---
            await r.setex("hw-monitor:heartbeat", 10, "1")

            poll_count += 1

            # --- Persistence ---
            if poll_count >= PERSIST_INTERVAL_POLLS:
                await _persist(gpu_history, server_history)
                for i in range(gpu_count):
                    gpu_history[i] = []
                server_history = []
                poll_count = 0

        except Exception as exc:
            logger.error("poll_error", error=str(exc))

        try:
            await asyncio.wait_for(shutdown.wait(), timeout=POLL_INTERVAL_S)
            break  # shutdown signalled
        except asyncio.TimeoutError:
            pass  # normal timeout, continue polling

    # Cleanup
    power_limit_task.cancel()
    try:
        await power_limit_task
    except asyncio.CancelledError:
        pass

    # Persist any remaining data
    if any(gpu_history[i] for i in range(gpu_count)) or server_history:
        await _persist(gpu_history, server_history)

    try:
        pynvml.nvmlShutdown()
    except pynvml.NVMLError:
        pass

    if _redis is not None:
        await _redis.aclose()
    if _engine is not None:
        await _engine.dispose()

    logger.info("hw_monitor_stopped")


async def _persist(gpu_history: dict[int, list[dict]], server_history: list[dict]) -> None:
    """Write averaged stats to the database."""
    now = datetime.now(timezone.utc)
    factory = _get_session_factory()
    try:
        async with factory() as session:
            for i, history in gpu_history.items():
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

            if server_history:
                n = len(server_history)

                def _avg(key: str) -> float | None:
                    vals = [h[key] for h in server_history if h[key] is not None]
                    return sum(vals) / len(vals) if vals else None

                session.add(
                    ServerStat(
                        recorded_at=now,
                        cpu_power_w=_avg("cpu_power_w"),
                        cpu_temp_c=_avg("cpu_temp_c"),
                        total_gpu_power_w=_avg("total_gpu_power_w"),
                        total_power_w=_avg("total_power_w"),
                    )
                )

            await session.commit()
            logger.info("stats_persisted", gpu_count=len(gpu_history), server_samples=len(server_history))
    except Exception as exc:
        logger.error("persist_error", error=str(exc))


if __name__ == "__main__":
    asyncio.run(main())
