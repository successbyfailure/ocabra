"""Host system stats endpoint (CPU, RAM)."""
from __future__ import annotations

import asyncio

import psutil
import structlog
from fastapi import APIRouter, Depends

from ocabra.api._deps_auth import UserContext, require_role

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["host"])

# TDP de CPUs de servidor/HEDT habituales, por si el kernel no publica el
# límite (el RAPL de AMD no expone constraint_*). Clave: regex sobre el
# "model name" de /proc/cpuinfo. Ampliar aquí o fijar CPU_POWER_MAX_W.
_KNOWN_TDP_W: list[tuple[str, float]] = [
    # Threadripper 3000 (3960X/3970X/3990X, PRO 39x5WX) y PRO 5000 WX: 280 W.
    (r"Threadripper (PRO )?3\d{3}W?X", 280.0),
    (r"Threadripper (PRO )?5\d{3}WX", 280.0),
    # Threadripper 7000 (7960X/7970X/7980X, PRO 79x5WX): 350 W.
    (r"Threadripper (PRO )?7\d{3}W?X", 350.0),
]


def _cpu_model() -> str | None:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        return None
    return None


def _rapl_limit_w() -> float | None:
    """Límite de potencia del paquete si el kernel lo publica (Intel suele)."""
    from pathlib import Path

    base = Path("/sys/class/powercap/intel-rapl:0")
    for name in ("constraint_0_power_limit_uw", "constraint_0_max_power_uw"):
        try:
            value = int((base / name).read_text().strip())
        except (OSError, ValueError):
            continue
        if value > 0:
            return round(value / 1_000_000, 1)
    return None


def cpu_power_max(model: str | None) -> tuple[float | None, str | None]:
    """(W, fuente) de la potencia máxima de la CPU: ajuste, RAPL o TDP."""
    import re

    from ocabra.config import settings

    if settings.cpu_power_max_w:
        return float(settings.cpu_power_max_w), "config"
    rapl = _rapl_limit_w()
    if rapl:
        return rapl, "rapl"
    for pattern, tdp in _KNOWN_TDP_W:
        if model and re.search(pattern, model):
            return tdp, "tdp"
    return None, None


# Cache to avoid hammering /proc on every dashboard refresh
_cache: dict = {}
_cache_ts: float = 0.0
_CACHE_TTL_S: float = 2.0


async def _collect() -> dict:
    """Collect host CPU and memory stats in a thread pool (psutil can block briefly)."""
    loop = asyncio.get_event_loop()

    def _read() -> dict:
        cpu_pct = psutil.cpu_percent(interval=0.2)
        cpu_count = psutil.cpu_count(logical=True)
        cpu_count_physical = psutil.cpu_count(logical=False)
        load_avg = psutil.getloadavg()
        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()
        model = _cpu_model()
        power_max, power_source = cpu_power_max(model)
        return {
            "cpu_model": model,
            # Escala de la barra de consumo del dashboard; ``None`` si no se
            # conoce (entonces la UI escala contra el pico observado y lo dice).
            "cpu_power_max_w": power_max,
            "cpu_power_max_source": power_source,
            "cpu_pct": cpu_pct,
            "cpu_count": cpu_count,
            "cpu_count_physical": cpu_count_physical,
            "load_avg_1m": round(load_avg[0], 2),
            "load_avg_5m": round(load_avg[1], 2),
            "load_avg_15m": round(load_avg[2], 2),
            "mem_total_mb": vm.total // (1024 * 1024),
            "mem_used_mb": vm.used // (1024 * 1024),
            "mem_free_mb": vm.available // (1024 * 1024),
            "mem_pct": vm.percent,
            "swap_total_mb": swap.total // (1024 * 1024),
            "swap_used_mb": swap.used // (1024 * 1024),
            "swap_pct": swap.percent,
        }

    return await loop.run_in_executor(None, _read)


@router.get("/host/stats")
async def get_host_stats(
    _user: UserContext = Depends(require_role("user")),
) -> dict:
    """Return host CPU and RAM stats. Cached for 2 seconds.

    Includes ``cpu_model`` and ``cpu_power_max_w`` (with
    ``cpu_power_max_source``: ``config``, ``rapl`` or ``tdp``) so the dashboard
    can scale the CPU power bar against a real maximum.
    """
    import time

    global _cache, _cache_ts
    now = time.monotonic()
    if now - _cache_ts < _CACHE_TTL_S and _cache:
        return _cache
    try:
        _cache = await _collect()
        _cache_ts = now
    except Exception as exc:
        logger.warning("host_stats_error", error=str(exc))
        if _cache:
            return _cache
        return {"error": str(exc)}
    return _cache
