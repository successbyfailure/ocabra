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


@router.get("/stats/recent", summary="Recent requests log")
async def recent_requests(
    limit: int = Query(20, ge=1, le=200),
    user: UserContext = Depends(require_role("model_manager")),
) -> dict:
    """Return the most recent N inference requests with user and group info.

    Args:
        limit: Max number of records to return (default 20, max 200).

    Returns:
        { requests: [{ id, modelId, backendType, requestKind, statusCode, startedAt,
                       durationMs, inputTokens, outputTokens, error,
                       userId, username, groupId, groupName }] }
    """
    from ocabra.stats.aggregator import get_recent_requests
    return await get_recent_requests(limit)


@router.get("/stats/by-user", summary="Stats aggregated by user")
async def stats_by_user(
    from_dt: datetime | None = Query(None, alias="from"),
    to_dt: datetime | None = Query(None, alias="to"),
    _user: UserContext = Depends(require_role("model_manager")),
) -> dict:
    """Return request statistics grouped by user.

    Returns:
        { byUser: [{ userId, username, totalRequests, totalErrors,
                     avgDurationMs, totalInputTokens, totalOutputTokens }] }
    """
    from ocabra.stats.aggregator import get_stats_by_user
    return await get_stats_by_user(from_dt, to_dt)


@router.get("/stats/by-group", summary="Stats aggregated by group")
async def stats_by_group(
    from_dt: datetime | None = Query(None, alias="from"),
    to_dt: datetime | None = Query(None, alias="to"),
    _user: UserContext = Depends(require_role("model_manager")),
) -> dict:
    """Return request statistics grouped by group.

    Returns:
        { byGroup: [{ groupId, groupName, totalRequests, totalErrors,
                      avgDurationMs, totalInputTokens, totalOutputTokens }] }
    """
    from ocabra.stats.aggregator import get_stats_by_group
    return await get_stats_by_group(from_dt, to_dt)


@router.get("/stats/my", summary="Own user stats")
async def my_stats(
    from_dt: datetime | None = Query(None, alias="from"),
    to_dt: datetime | None = Query(None, alias="to"),
    model_id: str | None = Query(None, alias="modelId"),
    user: UserContext = Depends(require_role("user")),
) -> dict:
    """Return overview statistics for the currently authenticated user.

    Returns:
        OverviewStats shape filtered to the current user's requests.
    """
    from ocabra.stats.aggregator import get_my_stats
    return await get_my_stats(user.user_id, from_dt, to_dt, model_id)


@router.get("/stats/my-group", summary="Own group stats")
async def my_group_stats(
    from_dt: datetime | None = Query(None, alias="from"),
    to_dt: datetime | None = Query(None, alias="to"),
    user: UserContext = Depends(require_role("user")),
) -> dict:
    """Return overview statistics for the groups the current user belongs to.

    Returns:
        { groupId, groupName, stats: OverviewStats }
    """
    from ocabra.stats.aggregator import get_my_group_stats
    return await get_my_group_stats(user.group_ids, from_dt, to_dt)
