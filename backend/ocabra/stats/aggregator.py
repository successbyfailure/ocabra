"""
Stats aggregator — reads request_stats and gpu_stats from DB
and returns summary structures for the REST API.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from math import ceil

import sqlalchemy as sa

from ocabra.database import AsyncSessionLocal
from ocabra.db.stats import GpuStat, ModelLoadStat, RequestStat


def _normalize_window(
    from_dt: datetime | None,
    to_dt: datetime | None,
) -> tuple[datetime, datetime]:
    if to_dt is None:
        to_dt = datetime.now(timezone.utc)
    if from_dt is None:
        from_dt = to_dt - timedelta(hours=24)
    return from_dt, to_dt


def _percentile(data: list[int], p: float) -> int:
    if not data:
        return 0
    idx = max(0, ceil(len(data) * p / 100.0) - 1)
    return data[min(idx, len(data) - 1)]


def _truncate_minute(dt: datetime) -> datetime:
    return dt.replace(second=0, microsecond=0)


async def get_request_stats(
    from_dt: datetime | None = None,
    to_dt: datetime | None = None,
    model_id: str | None = None,
) -> dict:
    """
    Return aggregated request statistics.

    Returns:
        Dict with total_requests, error_rate, avg_duration_ms,
        p50_duration_ms, p95_duration_ms, and minute-level series.
    """
    from_dt, to_dt = _normalize_window(from_dt, to_dt)

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

    durations = sorted(max(0, int(r.duration_ms or 0)) for r in rows)
    errors = sum(1 for r in rows if r.error or (r.status_code is not None and r.status_code >= 400))
    n = len(rows)

    per_minute: dict[datetime, int] = defaultdict(int)
    for row in rows:
        if row.started_at:
            per_minute[_truncate_minute(row.started_at)] += 1

    series = [
        {"timestamp": ts.isoformat(), "count": count}
        for ts, count in sorted(per_minute.items(), key=lambda item: item[0])
    ]

    return {
        "totalRequests": n,
        "errorRate": round(errors / n, 4) if n else 0.0,
        "avgDurationMs": int(sum(durations) / n),
        "p50DurationMs": _percentile(durations, 50),
        "p95DurationMs": _percentile(durations, 95),
        "series": series,
    }


async def get_energy_stats(
    from_dt: datetime | None = None,
    to_dt: datetime | None = None,
) -> dict:
    """
    Return energy statistics per GPU.
    """
    from_dt, to_dt = _normalize_window(from_dt, to_dt)

    async with AsyncSessionLocal() as session:
        q = sa.select(GpuStat).where(
            GpuStat.recorded_at >= from_dt,
            GpuStat.recorded_at <= to_dt,
        )
        q = q.order_by(GpuStat.gpu_index, GpuStat.recorded_at)
        result = await session.execute(q)
        rows = result.scalars().all()

    from ocabra.config import settings

    by_gpu: dict[int, list[GpuStat]] = {}
    for row in rows:
        by_gpu.setdefault(row.gpu_index, []).append(row)

    gpu_summaries = []
    total_kwh = 0.0
    for gpu_index, powers in by_gpu.items():
        ordered_rows = sorted(powers, key=lambda row: row.recorded_at)
        energy_wh = 0.0
        total_seconds = 0.0
        for i, row in enumerate(ordered_rows):
            start = row.recorded_at
            end = ordered_rows[i + 1].recorded_at if i + 1 < len(ordered_rows) else to_dt
            if end <= start:
                continue
            interval_seconds = (end - start).total_seconds()
            total_seconds += interval_seconds
            energy_wh += float(row.power_draw_w or 0.0) * interval_seconds / 3600.0

        kwh = energy_wh / 1000.0
        total_kwh += kwh
        avg_power_w = energy_wh / (total_seconds / 3600.0) if total_seconds else 0.0
        gpu_summaries.append(
            {
                "gpuIndex": gpu_index,
                "totalKwh": round(kwh, 4),
                "powerDrawW": round(avg_power_w, 1),
            }
        )

    return {
        "totalKwh": round(total_kwh, 4),
        "estimatedCostEur": round(total_kwh * settings.energy_cost_eur_kwh, 4),
        "byGpu": gpu_summaries,
    }


async def get_performance_stats(
    model_id: str | None = None,
    from_dt: datetime | None = None,
    to_dt: datetime | None = None,
) -> dict:
    """
    Return per-model performance statistics.
    """
    from_dt, to_dt = _normalize_window(from_dt, to_dt)

    async with AsyncSessionLocal() as session:
        q = sa.select(RequestStat).where(
            RequestStat.started_at >= from_dt,
            RequestStat.started_at <= to_dt,
        )
        if model_id:
            q = q.where(RequestStat.model_id == model_id)
        result = await session.execute(q)
        rows = result.scalars().all()

        load_q = sa.select(ModelLoadStat).where(
            ModelLoadStat.started_at >= from_dt,
            ModelLoadStat.started_at <= to_dt,
        )
        if model_id:
            load_q = load_q.where(ModelLoadStat.model_id == model_id)
        load_result = await session.execute(load_q)
        load_rows = load_result.scalars().all()

    by_model: dict[str, list[RequestStat]] = {}
    for row in rows:
        by_model.setdefault(row.model_id, []).append(row)

    load_by_model: dict[str, list[ModelLoadStat]] = {}
    for row in load_rows:
        load_by_model.setdefault(row.model_id, []).append(row)

    summaries = []
    for mid in sorted(set(by_model) | set(load_by_model)):
        model_rows = by_model.get(mid, [])
        model_load_rows = load_by_model.get(mid, [])
        durations = [max(0, int(r.duration_ms or 0)) for r in model_rows]
        in_tokens = [max(0, int(r.input_tokens or 0)) for r in model_rows]
        out_tokens = [max(0, int(r.output_tokens or 0)) for r in model_rows]
        errors = sum(1 for r in model_rows if r.error or (r.status_code is not None and r.status_code >= 400))
        n = len(model_rows)
        total_duration_ms = sum(durations)
        tokenized_requests = sum(1 for r in model_rows if (r.input_tokens or 0) > 0 or (r.output_tokens or 0) > 0)

        tokens_ps = (sum(out_tokens) / total_duration_ms * 1000) if total_duration_ms else 0.0
        req_per_min = (n * 60000 / total_duration_ms) if total_duration_ms else 0.0

        request_kinds = sorted({(r.request_kind or "other") for r in model_rows})
        backend_type = next(
            (r.backend_type for r in model_rows if r.backend_type),
            next((r.backend_type for r in model_load_rows if r.backend_type), "unknown"),
        )
        load_durations = sorted(max(0, int(r.duration_ms or 0)) for r in model_load_rows)
        avg_load_ms = int(sum(load_durations) / len(load_durations)) if load_durations else 0
        p95_load_ms = _percentile(load_durations, 95) if load_durations else 0
        last_load_ms = max(load_durations) if load_durations else 0

        summaries.append(
            {
                "modelId": mid,
                "backendType": backend_type,
                "requestKinds": request_kinds,
                "totalRequests": n,
                "avgLatencyMs": int(total_duration_ms / n) if n else 0,
                "p95LatencyMs": _percentile(sorted(durations), 95),
                "requestsPerMinute": round(req_per_min, 2),
                "tokensPerSecond": round(tokens_ps, 2),
                "totalInputTokens": int(sum(in_tokens)),
                "totalOutputTokens": int(sum(out_tokens)),
                "tokenizedRequests": int(tokenized_requests),
                "errorCount": errors,
                "uptimePct": round((n - errors) / n * 100, 1) if n else 0.0,
                "loadCount": len(load_durations),
                "avgLoadMs": avg_load_ms,
                "p95LoadMs": p95_load_ms,
                "lastLoadMs": last_load_ms,
            }
        )

    return {"byModel": summaries}


async def get_token_stats(
    from_dt: datetime | None = None,
    to_dt: datetime | None = None,
    model_id: str | None = None,
) -> dict:
    """
    Return token usage totals and per-request time series.
    """
    from_dt, to_dt = _normalize_window(from_dt, to_dt)

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

    total_input = sum(max(0, int(r.input_tokens or 0)) for r in rows)
    total_output = sum(max(0, int(r.output_tokens or 0)) for r in rows)

    by_backend: dict[str, dict[str, int]] = defaultdict(lambda: {"inputTokens": 0, "outputTokens": 0})
    for row in rows:
        backend = row.backend_type or "unknown"
        by_backend[backend]["inputTokens"] += max(0, int(row.input_tokens or 0))
        by_backend[backend]["outputTokens"] += max(0, int(row.output_tokens or 0))

    per_minute: dict[datetime, dict[str, int]] = defaultdict(lambda: {"inputTokens": 0, "outputTokens": 0})
    for row in rows:
        if row.started_at:
            bucket = _truncate_minute(row.started_at)
            per_minute[bucket]["inputTokens"] += max(0, int(row.input_tokens or 0))
            per_minute[bucket]["outputTokens"] += max(0, int(row.output_tokens or 0))

    series = [
        {"timestamp": ts.isoformat(), **counts}
        for ts, counts in sorted(per_minute.items())
    ]

    return {
        "totalInputTokens": int(total_input),
        "totalOutputTokens": int(total_output),
        "byBackend": [
            {"backendType": backend, **values}
            for backend, values in sorted(by_backend.items(), key=lambda item: item[0])
        ],
        "series": series,
    }


async def get_overview_stats(
    from_dt: datetime | None = None,
    to_dt: datetime | None = None,
    model_id: str | None = None,
) -> dict:
    """
    Return high-level metrics segmented by backend and request kind.
    """
    from_dt, to_dt = _normalize_window(from_dt, to_dt)

    async with AsyncSessionLocal() as session:
        q = sa.select(RequestStat).where(
            RequestStat.started_at >= from_dt,
            RequestStat.started_at <= to_dt,
        )
        if model_id:
            q = q.where(RequestStat.model_id == model_id)
        result = await session.execute(q)
        rows = result.scalars().all()

    total_requests = len(rows)
    if total_requests == 0:
        return {
            "totalRequests": 0,
            "totalErrors": 0,
            "avgDurationMs": 0,
            "tokenizedRequests": 0,
            "totalInputTokens": 0,
            "totalOutputTokens": 0,
            "byBackend": [],
            "byRequestKind": [],
        }

    durations = [max(0, int(r.duration_ms or 0)) for r in rows]
    total_errors = sum(1 for r in rows if r.error or (r.status_code is not None and r.status_code >= 400))
    tokenized_requests = sum(1 for r in rows if (r.input_tokens or 0) > 0 or (r.output_tokens or 0) > 0)
    total_input_tokens = sum(max(0, int(r.input_tokens or 0)) for r in rows)
    total_output_tokens = sum(max(0, int(r.output_tokens or 0)) for r in rows)

    by_backend: dict[str, list[RequestStat]] = defaultdict(list)
    by_kind: dict[str, list[RequestStat]] = defaultdict(list)
    for row in rows:
        by_backend[row.backend_type or "unknown"].append(row)
        by_kind[row.request_kind or "other"].append(row)

    def _summarize(group_rows: list[RequestStat], key_name: str, key_value: str) -> dict:
        g_n = len(group_rows)
        g_durations = [max(0, int(r.duration_ms or 0)) for r in group_rows]
        g_errors = sum(1 for r in group_rows if r.error or (r.status_code is not None and r.status_code >= 400))
        return {
            key_name: key_value,
            "totalRequests": g_n,
            "errorRate": round(g_errors / g_n, 4) if g_n else 0.0,
            "avgLatencyMs": int(sum(g_durations) / g_n) if g_n else 0,
            "p95LatencyMs": _percentile(sorted(g_durations), 95),
        }

    return {
        "totalRequests": total_requests,
        "totalErrors": total_errors,
        "avgDurationMs": int(sum(durations) / total_requests) if total_requests else 0,
        "tokenizedRequests": tokenized_requests,
        "totalInputTokens": int(total_input_tokens),
        "totalOutputTokens": int(total_output_tokens),
        "byBackend": [
            _summarize(group_rows, "backendType", backend)
            for backend, group_rows in sorted(by_backend.items(), key=lambda item: item[0])
        ],
        "byRequestKind": [
            _summarize(group_rows, "requestKind", kind)
            for kind, group_rows in sorted(by_kind.items(), key=lambda item: item[0])
        ],
    }
