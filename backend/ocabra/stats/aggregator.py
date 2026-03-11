"""
Stats aggregator — reads request_stats and gpu_stats from DB
and returns summary structures for the REST API.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import sqlalchemy as sa
import structlog

from ocabra.database import AsyncSessionLocal
from ocabra.db.stats import GpuStat, RequestStat

logger = structlog.get_logger(__name__)


async def get_request_stats(
    from_dt: datetime | None = None,
    to_dt: datetime | None = None,
    model_id: str | None = None,
) -> dict:
    """
    Return aggregated request statistics.

    Args:
        from_dt: Start of time window. Defaults to 24h ago.
        to_dt: End of time window. Defaults to now.
        model_id: Filter to a specific model, or None for all.

    Returns:
        Dict with total_requests, error_rate, avg_duration_ms,
        p50_duration_ms, p95_duration_ms, and a time series.
    """
    if to_dt is None:
        to_dt = datetime.now(timezone.utc)
    if from_dt is None:
        from_dt = to_dt - timedelta(hours=24)

    async with AsyncSessionLocal() as session:
        q = sa.select(RequestStat).where(
            RequestStat.started_at >= from_dt,
            RequestStat.started_at <= to_dt,
        )
        if model_id:
            q = q.where(RequestStat.model_id == model_id)

        result = await session.execute(q)
        rows = result.scalars().all()

    if not rows:
        return {
            "totalRequests": 0,
            "errorRate": 0.0,
            "avgDurationMs": 0,
            "p50DurationMs": 0,
            "p95DurationMs": 0,
            "series": [],
        }

    durations = sorted(r.duration_ms or 0 for r in rows)
    errors = sum(1 for r in rows if r.error)
    n = len(rows)

    def percentile(data: list[int], p: float) -> int:
        idx = int(len(data) * p / 100)
        return data[min(idx, len(data) - 1)]

    return {
        "totalRequests": n,
        "errorRate": round(errors / n, 4) if n else 0.0,
        "avgDurationMs": int(sum(durations) / n),
        "p50DurationMs": percentile(durations, 50),
        "p95DurationMs": percentile(durations, 95),
        "series": [],  # Filled by frontend from WS events
    }


async def get_energy_stats(
    from_dt: datetime | None = None,
    to_dt: datetime | None = None,
) -> dict:
    """
    Return energy statistics per GPU.

    Args:
        from_dt: Start of time window.
        to_dt: End of time window.

    Returns:
        Dict with total_kwh, estimated_cost_eur, and per-GPU breakdown.
    """
    if to_dt is None:
        to_dt = datetime.now(timezone.utc)
    if from_dt is None:
        from_dt = to_dt - timedelta(hours=24)

    async with AsyncSessionLocal() as session:
        q = sa.select(GpuStat).where(
            GpuStat.recorded_at >= from_dt,
            GpuStat.recorded_at <= to_dt,
        )
        result = await session.execute(q)
        rows = result.scalars().all()

    from ocabra.config import settings

    by_gpu: dict[int, list[float]] = {}
    for row in rows:
        by_gpu.setdefault(row.gpu_index, []).append(row.power_draw_w or 0.0)

    gpu_summaries = []
    total_kwh = 0.0
    for gpu_index, powers in by_gpu.items():
        # Each row = 1 minute average → kWh = W * h / 1000 = W * (1/60) / 1000
        kwh = sum(powers) / 60 / 1000
        total_kwh += kwh
        gpu_summaries.append({
            "gpuIndex": gpu_index,
            "totalKwh": round(kwh, 4),
            "powerDrawW": round(sum(powers) / len(powers), 1) if powers else 0.0,
        })

    return {
        "totalKwh": round(total_kwh, 4),
        "estimatedCostEur": round(total_kwh * settings.energy_cost_eur_kwh, 4),
        "byGpu": gpu_summaries,
    }


async def get_performance_stats(model_id: str | None = None) -> dict:
    """
    Return per-model performance statistics.

    Args:
        model_id: Filter to a specific model, or None for all.

    Returns:
        Dict with per-model breakdown of requests, latency, tokens/s, errors, uptime.
    """
    async with AsyncSessionLocal() as session:
        q = sa.select(RequestStat)
        if model_id:
            q = q.where(RequestStat.model_id == model_id)
        result = await session.execute(q)
        rows = result.scalars().all()

    by_model: dict[str, list] = {}
    for row in rows:
        by_model.setdefault(row.model_id, []).append(row)

    summaries = []
    for mid, model_rows in by_model.items():
        durations = [r.duration_ms or 0 for r in model_rows]
        out_tokens = [r.output_tokens or 0 for r in model_rows]
        errors = sum(1 for r in model_rows if r.error)
        n = len(model_rows)
        avg_dur_s = (sum(durations) / n / 1000) if n else 0
        tokens_ps = (sum(out_tokens) / sum(durations) * 1000) if sum(durations) else 0

        summaries.append({
            "modelId": mid,
            "totalRequests": n,
            "avgLatencyMs": int(sum(durations) / n) if n else 0,
            "tokensPerSecond": round(tokens_ps, 1),
            "errorCount": errors,
            "uptimePct": round((n - errors) / n * 100, 1) if n else 0.0,
        })

    return {"byModel": summaries}


async def get_token_stats(
    from_dt: datetime | None = None,
    to_dt: datetime | None = None,
    model_id: str | None = None,
) -> dict:
    """
    Return token usage totals and a simple per-request time series.
    """
    if to_dt is None:
        to_dt = datetime.now(timezone.utc)
    if from_dt is None:
        from_dt = to_dt - timedelta(hours=24)

    async with AsyncSessionLocal() as session:
        q = sa.select(RequestStat).where(
            RequestStat.started_at >= from_dt,
            RequestStat.started_at <= to_dt,
        )
        if model_id:
            q = q.where(RequestStat.model_id == model_id)
        q = q.order_by(RequestStat.started_at.asc())
        result = await session.execute(q)
        rows = result.scalars().all()

    total_input = sum(r.input_tokens or 0 for r in rows)
    total_output = sum(r.output_tokens or 0 for r in rows)
    series = [
        {
            "timestamp": r.started_at.isoformat() if r.started_at else "",
            "inputTokens": int(r.input_tokens or 0),
            "outputTokens": int(r.output_tokens or 0),
        }
        for r in rows
    ]

    return {
        "totalInputTokens": int(total_input),
        "totalOutputTokens": int(total_output),
        "series": series,
    }
