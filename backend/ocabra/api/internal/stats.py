"""
GET /ocabra/stats/* — Statistics API endpoints.
"""
from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Query, Request

router = APIRouter(tags=["stats"])


@router.get("/stats/requests", summary="Request statistics")
async def request_stats(
    from_dt: datetime | None = Query(None, alias="from"),
    to_dt: datetime | None = Query(None, alias="to"),
    model_id: str | None = Query(None, alias="modelId"),
) -> dict:
    """Return aggregated request statistics for the given time window."""
    from ocabra.stats.aggregator import get_request_stats
    return await get_request_stats(from_dt, to_dt, model_id)


@router.get("/stats/energy", summary="Energy statistics")
async def energy_stats(
    from_dt: datetime | None = Query(None, alias="from"),
    to_dt: datetime | None = Query(None, alias="to"),
) -> dict:
    """Return energy consumption statistics per GPU."""
    from ocabra.stats.aggregator import get_energy_stats
    return await get_energy_stats(from_dt, to_dt)


@router.get("/stats/performance", summary="Performance statistics")
async def performance_stats(
    model_id: str | None = Query(None, alias="modelId"),
) -> dict:
    """Return per-model performance statistics."""
    from ocabra.stats.aggregator import get_performance_stats
    return await get_performance_stats(model_id)
