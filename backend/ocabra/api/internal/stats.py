"""
GET /ocabra/stats/* — Statistics API endpoints.
"""
from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, Query

from ocabra.api._deps_auth import UserContext, require_role

router = APIRouter(tags=["stats"])


@router.get("/stats/requests", summary="Request statistics")
async def request_stats(
    from_dt: datetime | None = Query(None, alias="from"),
    to_dt: datetime | None = Query(None, alias="to"),
    model_id: str | None = Query(None, alias="model_id"),
    model_id_camel: str | None = Query(None, alias="modelId"),
    _user: UserContext = Depends(require_role("model_manager")),
) -> dict:
    """Return aggregated request statistics for the given time window."""
    from ocabra.stats.aggregator import get_request_stats

    return await get_request_stats(from_dt, to_dt, model_id or model_id_camel)


@router.get("/stats/energy", summary="Energy statistics")
async def energy_stats(
    from_dt: datetime | None = Query(None, alias="from"),
    to_dt: datetime | None = Query(None, alias="to"),
    _user: UserContext = Depends(require_role("user")),
) -> dict:
    """Return energy consumption statistics per GPU."""
    from ocabra.stats.aggregator import get_energy_stats

    return await get_energy_stats(from_dt, to_dt)


@router.get("/stats/performance", summary="Performance statistics")
async def performance_stats(
    from_dt: datetime | None = Query(None, alias="from"),
    to_dt: datetime | None = Query(None, alias="to"),
    model_id: str | None = Query(None, alias="model_id"),
    model_id_camel: str | None = Query(None, alias="modelId"),
    _user: UserContext = Depends(require_role("model_manager")),
) -> dict:
    """Return per-model performance statistics."""
    from ocabra.stats.aggregator import get_performance_stats

    return await get_performance_stats(model_id or model_id_camel, from_dt, to_dt)


@router.get("/stats/tokens", summary="Token statistics")
async def token_stats(
    from_dt: datetime | None = Query(None, alias="from"),
    to_dt: datetime | None = Query(None, alias="to"),
    model_id: str | None = Query(None, alias="model_id"),
    model_id_camel: str | None = Query(None, alias="modelId"),
    _user: UserContext = Depends(require_role("model_manager")),
) -> dict:
    """Return token usage totals and timeline."""
    from ocabra.stats.aggregator import get_token_stats

    return await get_token_stats(from_dt, to_dt, model_id or model_id_camel)


@router.get("/stats/overview", summary="Overview statistics")
async def overview_stats(
    from_dt: datetime | None = Query(None, alias="from"),
    to_dt: datetime | None = Query(None, alias="to"),
    model_id: str | None = Query(None, alias="model_id"),
    model_id_camel: str | None = Query(None, alias="modelId"),
    _user: UserContext = Depends(require_role("model_manager")),
) -> dict:
    """Return high-level statistics segmented by backend and request kind."""
    from ocabra.stats.aggregator import get_overview_stats

    return await get_overview_stats(from_dt, to_dt, model_id or model_id_camel)
