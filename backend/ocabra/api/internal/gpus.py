from datetime import datetime, timedelta, timezone
from dataclasses import asdict

import sqlalchemy as sa
from fastapi import APIRouter, Depends, HTTPException, Query, Request

from ocabra.api._deps_auth import UserContext, require_role
from ocabra.db.stats import GpuStat

router = APIRouter(tags=["gpus"])

WINDOW_MAP = {
    "5m": timedelta(minutes=5),
    "1h": timedelta(hours=1),
    "24h": timedelta(hours=24),
}


@router.get("/gpus")
async def list_gpus(
    request: Request,
    _user: UserContext = Depends(require_role("user")),
) -> list[dict]:
    """List current state of all GPUs."""
    gpu_manager = request.app.state.gpu_manager
    states = await gpu_manager.get_all_states()
    return [asdict(s) for s in states]


@router.get("/gpus/{index}")
async def get_gpu(
    index: int,
    request: Request,
    _user: UserContext = Depends(require_role("user")),
) -> dict:
    """Get current state of a single GPU."""
    gpu_manager = request.app.state.gpu_manager
    try:
        state = await gpu_manager.get_state(index)
        return asdict(state)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"GPU {index} not found")


@router.get("/gpus/{index}/stats")
async def get_gpu_stats(
    index: int,
    request: Request,
    window: str = Query(default="1h", pattern="^(5m|1h|24h)$"),
    _user: UserContext = Depends(require_role("user")),
) -> list[dict]:
    """Get historical GPU stats for a given time window."""
    from ocabra.database import AsyncSessionLocal

    delta = WINDOW_MAP[window]
    since = datetime.now(timezone.utc) - delta

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            sa.select(GpuStat)
            .where(GpuStat.gpu_index == index)
            .where(GpuStat.recorded_at >= since)
            .order_by(GpuStat.recorded_at)
        )
        rows = result.scalars().all()

    return [
        {
            "recorded_at": r.recorded_at.isoformat(),
            "gpu_index": r.gpu_index,
            "utilization_pct": r.utilization_pct,
            "vram_used_mb": r.vram_used_mb,
            "power_draw_w": r.power_draw_w,
            "temperature_c": r.temperature_c,
        }
        for r in rows
    ]
