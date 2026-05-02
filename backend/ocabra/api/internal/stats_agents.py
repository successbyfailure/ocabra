"""Agent-scoped statistics endpoints (closes the frontend mock deuda).

Exposes:

* ``GET /ocabra/stats/by-agent`` — top agents in a window plus top tools
  with p50/p95/error rate.  Frontend consumer:
  ``frontend/src/components/stats/AgentsPanel.tsx`` and
  ``frontend/src/pages/Dashboard.tsx`` (Active agents card).
* ``GET /ocabra/stats/tool-calls`` — recent ``tool_call_stats`` rows for
  drill-down by agent.

Filtering by group is enforced for non-admin callers (mirrors the pattern
used by ``/ocabra/stats/recent``).

Plan: docs/tasks/agents-mcp-plan.md — Fase 5 / Stats endpoints.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import sqlalchemy as sa
from fastapi import APIRouter, Depends, Query

from ocabra.api._deps_auth import UserContext, require_role
from ocabra.database import AsyncSessionLocal
from ocabra.db.agents import Agent
from ocabra.db.stats import RequestStat, ToolCallStat

router = APIRouter(tags=["stats"])

_RANGE_TO_TIMEDELTA = {
    "1h": timedelta(hours=1),
    "24h": timedelta(hours=24),
    "7d": timedelta(days=7),
    "30d": timedelta(days=30),
}


def _resolve_window(
    range_value: str | None,
    from_dt: datetime | None,
    to_dt: datetime | None,
) -> tuple[datetime, datetime]:
    """Return ``(from, to)`` datetimes from either ``range`` or explicit ``from``/``to``."""
    now = datetime.now(UTC)
    if from_dt is not None and to_dt is not None:
        return from_dt, to_dt
    delta = _RANGE_TO_TIMEDELTA.get((range_value or "24h").lower(), timedelta(hours=24))
    if to_dt is None:
        to_dt = now
    if from_dt is None:
        from_dt = to_dt - delta
    return from_dt, to_dt


def _percentile_from_sorted(values: list[int], pct: float) -> int | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    k = max(0, min(len(values) - 1, int(round((pct / 100.0) * (len(values) - 1)))))
    return values[k]


@router.get(
    "/stats/by-agent",
    summary="Agent-aggregated statistics",
    description=(
        "Aggregate request_stats and tool_call_stats by agent for the window. "
        "Returns top agents with totals (requests, tool calls, errors, "
        "p50/p95 duration, total tokens) plus a ranking of tools by "
        "invocation count with p50/p95 latency and error rate."
    ),
)
async def by_agent(
    range_value: str | None = Query(None, alias="range"),
    from_dt: datetime | None = Query(None, alias="from"),
    to_dt: datetime | None = Query(None, alias="to"),
    user: UserContext = Depends(require_role("user")),
) -> dict[str, Any]:
    """Return per-agent aggregations + per-tool aggregations for a window."""
    window_from, window_to = _resolve_window(range_value, from_dt, to_dt)

    async with AsyncSessionLocal() as session:
        # Resolve the visible agent set first (group-scoped for non-admins).
        agent_q = sa.select(Agent.id, Agent.slug, Agent.display_name, Agent.group_id)
        agent_rows = (await session.execute(agent_q)).all()
        visible_ids: set[UUID] = set()
        agent_meta: dict[UUID, tuple[str, str]] = {}
        group_set = set(user.group_ids or [])
        for row in agent_rows:
            agent_id, slug, name, group_id = row
            if not user.is_admin:
                if group_id is not None and str(group_id) not in group_set:
                    continue
            visible_ids.add(agent_id)
            agent_meta[agent_id] = (slug, name)

        if not visible_ids:
            return {"by_agent": [], "by_tool": []}

        # Per-agent aggregation across both root and hop request_stats.
        # Both rows carry agent_id so we just group by it.
        request_q = sa.select(
            RequestStat.agent_id,
            RequestStat.duration_ms,
            RequestStat.input_tokens,
            RequestStat.output_tokens,
            RequestStat.status_code,
            RequestStat.parent_request_id,
        ).where(
            RequestStat.agent_id.in_(visible_ids),
            RequestStat.started_at >= window_from,
            RequestStat.started_at <= window_to,
        )
        per_agent_rows = (await session.execute(request_q)).all()

        # Per-agent accumulators.
        agg: dict[UUID, dict[str, Any]] = {
            aid: {
                "request_count": 0,  # only root rows count for "requests"
                "error_count": 0,
                "tokens_total": 0,
                "durations": [],  # for p50/p95 over root rows
            }
            for aid in visible_ids
        }
        for r in per_agent_rows:
            aid, dur, in_tok, out_tok, code, parent = r
            if aid not in agg:
                continue
            tokens = (in_tok or 0) + (out_tok or 0)
            agg[aid]["tokens_total"] += int(tokens)
            if parent is None:
                agg[aid]["request_count"] += 1
                if dur is not None:
                    agg[aid]["durations"].append(int(dur))
                if code is not None and int(code) >= 400:
                    agg[aid]["error_count"] += 1

        # tool_call counts per agent.
        tc_count_q = (
            sa.select(ToolCallStat.agent_id, sa.func.count())
            .where(
                ToolCallStat.agent_id.in_(visible_ids),
                ToolCallStat.created_at >= window_from,
                ToolCallStat.created_at <= window_to,
            )
            .group_by(ToolCallStat.agent_id)
        )
        tc_counts: dict[UUID, int] = {
            r[0]: int(r[1] or 0) for r in (await session.execute(tc_count_q)).all()
        }

        by_agent_list: list[dict[str, Any]] = []
        for aid, data in agg.items():
            durations = sorted(data["durations"])
            slug, name = agent_meta.get(aid, ("", ""))
            by_agent_list.append(
                {
                    "agent_id": str(aid),
                    "slug": slug,
                    "display_name": name,
                    "request_count": data["request_count"],
                    "tool_call_count": tc_counts.get(aid, 0),
                    "error_count": data["error_count"],
                    "p50_duration_ms": _percentile_from_sorted(durations, 50),
                    "p95_duration_ms": _percentile_from_sorted(durations, 95),
                    "total_tokens": data["tokens_total"],
                }
            )
        by_agent_list.sort(key=lambda d: d["request_count"], reverse=True)

        # Per-tool aggregation across visible agents in window.
        tool_q = sa.select(
            ToolCallStat.mcp_server_alias,
            ToolCallStat.tool_name,
            ToolCallStat.duration_ms,
            ToolCallStat.status,
        ).where(
            ToolCallStat.agent_id.in_(visible_ids),
            ToolCallStat.created_at >= window_from,
            ToolCallStat.created_at <= window_to,
        )
        tool_rows = (await session.execute(tool_q)).all()
        tools_acc: dict[tuple[str, str], dict[str, Any]] = {}
        for alias, tool_name, dur, status in tool_rows:
            key = (alias or "", tool_name or "")
            entry = tools_acc.setdefault(
                key,
                {"invocations": 0, "errors": 0, "durations": []},
            )
            entry["invocations"] += 1
            if status and status != "ok":
                entry["errors"] += 1
            if dur is not None:
                entry["durations"].append(int(dur))
        by_tool_list: list[dict[str, Any]] = []
        for (alias, tool_name), entry in tools_acc.items():
            durations = sorted(entry["durations"])
            invocations = entry["invocations"]
            errors = entry["errors"]
            by_tool_list.append(
                {
                    "mcp_server_alias": alias,
                    "tool_name": tool_name,
                    "invocations": invocations,
                    "errors": errors,
                    "p50_duration_ms": _percentile_from_sorted(durations, 50),
                    "p95_duration_ms": _percentile_from_sorted(durations, 95),
                    "error_rate": (errors / invocations) if invocations else 0.0,
                }
            )
        by_tool_list.sort(key=lambda d: d["invocations"], reverse=True)

    return {"by_agent": by_agent_list, "by_tool": by_tool_list}


@router.get(
    "/stats/tool-calls",
    summary="Recent tool_call_stats rows",
    description="Return recent tool_call_stats rows, optionally filtered by agent_id.",
)
async def tool_calls(
    agent_id: str | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
    since: datetime | None = Query(None),
    user: UserContext = Depends(require_role("user")),
) -> dict[str, Any]:
    """Return the most recent tool_call_stats rows for drill-down."""
    async with AsyncSessionLocal() as session:
        # Build the visible agent set first to enforce group ACL.
        agent_meta_q = sa.select(Agent.id, Agent.slug, Agent.group_id)
        agent_rows = (await session.execute(agent_meta_q)).all()
        group_set = set(user.group_ids or [])
        visible: dict[UUID, str] = {}
        for aid, slug, group_id in agent_rows:
            if not user.is_admin:
                if group_id is not None and str(group_id) not in group_set:
                    continue
            visible[aid] = slug
        if not visible:
            return {"tool_calls": []}

        target_ids: set[UUID] | None
        if agent_id:
            try:
                req_aid = UUID(agent_id)
            except ValueError:
                return {"tool_calls": []}
            if req_aid not in visible:
                return {"tool_calls": []}
            target_ids = {req_aid}
        else:
            target_ids = set(visible.keys())

        q = (
            sa.select(
                ToolCallStat.id,
                ToolCallStat.created_at,
                ToolCallStat.agent_id,
                ToolCallStat.mcp_server_alias,
                ToolCallStat.tool_name,
                ToolCallStat.status,
                ToolCallStat.duration_ms,
                ToolCallStat.hop_index,
                ToolCallStat.error,
                ToolCallStat.tool_args_redacted,
            )
            .where(ToolCallStat.agent_id.in_(target_ids))
            .order_by(ToolCallStat.created_at.desc())
            .limit(limit)
        )
        if since is not None:
            q = q.where(ToolCallStat.created_at >= since)
        rows = (await session.execute(q)).all()
        out: list[dict[str, Any]] = []
        for r in rows:
            (
                tc_id,
                created_at,
                aid,
                alias,
                tool_name,
                status,
                duration_ms,
                hop_index,
                error,
                args_redacted,
            ) = r
            out.append(
                {
                    "id": str(tc_id),
                    "created_at": created_at.isoformat() if created_at else None,
                    "agent_id": str(aid) if aid else None,
                    "agent_slug": visible.get(aid),
                    "mcp_server_alias": alias,
                    "tool_name": tool_name,
                    "status": status,
                    "duration_ms": duration_ms,
                    "hop_index": hop_index,
                    "error": error,
                    "args_redacted": args_redacted or {},
                }
            )
    return {"tool_calls": out}
