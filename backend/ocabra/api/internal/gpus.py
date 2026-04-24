from datetime import datetime, timedelta, timezone
from dataclasses import asdict

import sqlalchemy as sa
import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from ocabra.api._deps_auth import UserContext, require_role
from ocabra.db.stats import GpuStat

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["gpus"])

WINDOW_MAP = {
    "5m": timedelta(minutes=5),
    "1h": timedelta(hours=1),
    "24h": timedelta(hours=24),
}


@router.get(
    "/gpus",
    summary="List all GPUs",
    description="Return the live state of every GPU: VRAM usage, utilization, temperature, power draw, and running processes.",
)
async def list_gpus(
    request: Request,
    _user: UserContext = Depends(require_role("user")),
) -> list[dict]:
    """List current state of all GPUs."""
    gpu_manager = request.app.state.gpu_manager
    states = await gpu_manager.get_all_states()
    return [asdict(s) for s in states]


@router.get(
    "/gpus/{index}",
    summary="Get GPU state",
    description="Return the live state of a single GPU by its index.",
    responses={404: {"description": "GPU index not found"}},
)
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


@router.get(
    "/gpus/{index}/stats",
    summary="Get GPU historical stats",
    description=(
        "Return time-series GPU statistics (utilization, VRAM, power, temperature) "
        "for the given window. Supported windows: 5m, 1h, 24h."
    ),
)
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


# ── GPU Control Endpoints ────────────────────────────────────────────────


@router.get(
    "/gpus/{index}/power-limits",
    summary="Get GPU power limit constraints",
    description="Return the current, default, min, and max power limits for a GPU.",
)
async def get_gpu_power_limits(
    index: int,
    _user: UserContext = Depends(require_role("user")),
) -> dict:
    """Return power limit info for a single GPU."""
    import pynvml

    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        current = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
        default = pynvml.nvmlDeviceGetPowerManagementDefaultLimit(handle)
        min_w, max_w = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)
        persistence = pynvml.nvmlDeviceGetPersistenceMode(handle)
        return {
            "gpu_index": index,
            "current_w": current // 1000,
            "default_w": default // 1000,
            "min_w": min_w // 1000,
            "max_w": max_w // 1000,
            "persistence_mode": bool(persistence),
        }
    except pynvml.NVMLError as exc:
        raise HTTPException(status_code=500, detail=f"NVML error: {exc}")


class PowerLimitPatch(BaseModel):
    power_limit_w: int | None = Field(None, description="Power limit in watts (0 = reset to default)")
    persistence_mode: bool | None = Field(None, description="Enable/disable persistence mode")


@router.patch(
    "/gpus/{index}/power",
    summary="Set GPU power limit and persistence mode",
    description="Adjust the power limit (TDP) and/or persistence mode for a GPU. Requires system_admin role.",
)
async def set_gpu_power(
    index: int,
    body: PowerLimitPatch,
    _user: UserContext = Depends(require_role("system_admin")),
) -> dict:
    """Set power limit and/or persistence mode for a GPU."""
    import pynvml

    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)

        if body.persistence_mode is not None:
            mode = pynvml.NVML_FEATURE_ENABLED if body.persistence_mode else pynvml.NVML_FEATURE_DISABLED
            pynvml.nvmlDeviceSetPersistenceMode(handle, mode)
            logger.info("gpu_persistence_mode_set", gpu=index, enabled=body.persistence_mode)

        if body.power_limit_w is not None:
            target_w = body.power_limit_w
            if target_w == 0:
                # Reset to default
                default = pynvml.nvmlDeviceGetPowerManagementDefaultLimit(handle)
                target_w = default // 1000

            # Try via hw-monitor (privileged) first, fall back to direct pynvml
            from ocabra.redis_client import publish as redis_publish
            import json as _json

            try:
                await redis_publish(
                    "gpu:set_power_limit",
                    _json.dumps({"gpu_index": index, "limit_w": target_w}),
                )
                logger.info("gpu_power_limit_requested_via_hw_monitor", gpu=index, watts=target_w)
                # Give hw-monitor a moment to apply
                import asyncio
                await asyncio.sleep(0.5)
            except Exception:
                # Fallback: try direct pynvml (will fail if not privileged)
                try:
                    pynvml.nvmlDeviceSetPowerManagementLimit(handle, target_w * 1000)
                    logger.info("gpu_power_limit_set_direct", gpu=index, watts=target_w)
                except pynvml.NVMLError as nvml_exc:
                    raise HTTPException(
                        status_code=503,
                        detail=f"Cannot set power limit: {nvml_exc}. Ensure hw-monitor is running.",
                    )

        # Return updated state
        current = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
        default = pynvml.nvmlDeviceGetPowerManagementDefaultLimit(handle)
        min_w, max_w = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)
        persistence = pynvml.nvmlDeviceGetPersistenceMode(handle)
        return {
            "gpu_index": index,
            "current_w": current // 1000,
            "default_w": default // 1000,
            "min_w": min_w // 1000,
            "max_w": max_w // 1000,
            "persistence_mode": bool(persistence),
        }
    except pynvml.NVMLError as exc:
        raise HTTPException(status_code=500, detail=f"NVML error: {exc}")
