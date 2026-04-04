"""Host system stats endpoint (CPU, RAM)."""
from __future__ import annotations

import asyncio

import psutil
import structlog
from fastapi import APIRouter, Depends

from ocabra.api._deps_auth import UserContext, require_role

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["host"])

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
        return {
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
    """Return host CPU and RAM stats. Cached for 2 seconds."""
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
