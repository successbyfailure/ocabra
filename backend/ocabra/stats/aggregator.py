"""
Stats aggregator — reads request_stats and gpu_stats from DB
and returns summary structures for the REST API.
"""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from math import ceil

import sqlalchemy as sa

from ocabra.config import settings
from ocabra.database import AsyncSessionLocal
from ocabra.db.stats import RequestStat
from ocabra.stats.cost_calculator import classify_model_tier, estimate_request_cost

_WORKER_HASH_RE = re.compile(r"::[0-9a-f]{12}$")


def _audio_seconds(meta: object) -> float | None:
    """Segundos de audio de ``work_size_meta`` (transcripciones), si los hay."""
    if isinstance(meta, dict):
        value = meta.get("audio_seconds")
        if isinstance(value, (int, float)) and value > 0:
            return round(float(value), 1)
    return None


def video_speed(meta: object) -> tuple[float | None, int | None]:
    """``(fps generados, fotogramas)`` de un vídeo generado, si los hay.

    La velocidad de un vídeo son los fotogramas que produce por segundo de
    generación (``generation_fps``, sin cola ni carga del modelo).
    """
    if not isinstance(meta, dict):
        return None, None
    fps = meta.get("generation_fps")
    frames = meta.get("input_frames")
    fps_out = round(float(fps), 2) if isinstance(fps, (int, float)) and fps > 0 else None
    frames_out = int(frames) if isinstance(frames, (int, float)) and frames > 0 else None
    return fps_out, frames_out


# Speed of the requests that are not measured in generated tokens, in the
# unit of what they do. Each entry: request kinds, unit, how to read the work
# of one row (from work_size_meta / tokens) and whether the value is work per
# second (rate) or seconds per unit of work (cost).
_WORK_SPEEDS: tuple[tuple[tuple[str, ...], str, str, bool], ...] = (
    (("audio_transcription", "realtime_transcription"), "x_realtime", "audio_seconds", True),
    (("tts", "realtime_tts", "music"), "x_realtime", "audio_seconds", True),
    (("tts", "realtime_tts"), "chars_per_s", "text_chars", True),
    (("embedding", "ollama_embedding"), "inputs_per_s", "n_inputs", True),
    (("embedding", "ollama_embedding"), "tokens_per_s", "input_tokens", True),
    (("rerank", "score", "classify", "pooling"), "documents_per_s", "n_documents", True),
    (("image_generation",), "s_per_image", "n_images", False),
    (("video_upscale",), "frames_per_s", "input_frames", True),
    # Vídeo generado: fotogramas producidos por segundo (sin cola ni carga).
    (("video_generation",), "frames_per_s", "input_frames", True),
    (("image_upscale",), "s_per_image", "n_images", False),
    # Megapíxeles que produce la red por segundo: lo que separa una foto de
    # 1080p (~10 s) de una 4K (~45 s), que s/imagen mezcla.
    (("image_upscale",), "mp_per_s", "model_mp", True),
    (("realtime_chat",), "tokens_per_s", "output_tokens", True),
)


def _work_amount(row: object, field: str) -> float:
    if field in ("input_tokens", "output_tokens"):
        return float(max(0, int(getattr(row, field, None) or 0)))
    meta = getattr(row, "work_size_meta", None)
    value = meta.get(field) if isinstance(meta, dict) else None
    return float(value) if isinstance(value, (int, float)) and value > 0 else 0.0


def work_speeds(rows: list) -> list[dict]:
    """Speed per request kind in its own unit (see ``_WORK_SPEEDS``).

    Over the rows of one model that succeeded and carry that work: total work
    / total time (``x_realtime`` = seconds of audio per second of wall time),
    or time per unit of work for ``s_per_image``. Before, the performance
    table only had tokens per second: transcription, speech, embeddings,
    rerank and images showed no speed at all.
    """
    out: list[dict] = []
    for kinds, unit, field, rate in _WORK_SPEEDS:
        work = 0.0
        seconds = 0.0
        samples = 0
        for r in rows:
            if (r.request_kind or "") not in kinds or r.error:
                continue
            if r.status_code is not None and r.status_code >= 400:
                continue
            amount = _work_amount(r, field)
            # Sin la espera a que cargase el modelo ni la cola: una carga en
            # frío de 8 s convertía un escalado de 10 s en uno de 18 s.
            ms = max(
                0,
                int(r.duration_ms or 0)
                - (_telemetry_int(r, "load_ms") or 0)
                - (_telemetry_int(r, "queue_ms") or 0),
            )
            if amount <= 0 or ms <= 0:
                continue
            work += amount
            seconds += ms / 1000.0
            samples += 1
        if not samples or seconds <= 0 or work <= 0:
            continue
        value = work / seconds if rate else seconds / work
        out.append(
            {
                "kinds": [k for k in kinds if any((r.request_kind or "") == k for r in rows)],
                "unit": unit,
                "value": round(value, 2),
                "samples": samples,
            }
        )
    return out


def _model_key_of_worker(worker_id: str) -> str:
    """``llamacpp_mks/gemma4:26b::0123456789ab`` → ``gemma4:26b``."""
    return _WORKER_HASH_RE.sub("", worker_id).split("/", 1)[-1]


def _normalize_window(
    from_dt: datetime | None,
    to_dt: datetime | None,
) -> tuple[datetime, datetime]:
    if to_dt is None:
        to_dt = datetime.now(UTC)
    if from_dt is None:
        from_dt = to_dt - timedelta(hours=24)
    # Callers (including plain `curl` against the public API, not just the
    # frontend) may pass a bare ISO datetime with no offset, e.g.
    # ``datetime.utcnow().isoformat()``. FastAPI parses that into a naive
    # datetime, which then can't be compared against the tz-aware
    # ``recorded_at``/``started_at`` columns (TypeError: can't compare
    # offset-naive and offset-aware datetimes) — found live 2026-10-30 via
    # /ocabra/stats/energy. Treat a naive input as UTC rather than rejecting
    # it or crashing downstream.
    if from_dt.tzinfo is None:
        from_dt = from_dt.replace(tzinfo=UTC)
    if to_dt.tzinfo is None:
        to_dt = to_dt.replace(tzinfo=UTC)
    return from_dt, to_dt


def _percentile(data: list[int], p: float) -> int:
    if not data:
        return 0
    idx = max(0, ceil(len(data) * p / 100.0) - 1)
    return data[min(idx, len(data) - 1)]


# --- SQL aggregation helpers (REF-12) ------------------------------------
# These let the aggregator compute totals in PostgreSQL instead of pulling
# every ORM row into Python (161k request_stats rows and growing). The
# expressions reproduce, result for result, the semantics the old row-by-row
# code had:
#   * ``GREATEST(COALESCE(x, 0), 0)`` for the ``max(0, int(x or 0))`` guards;
#   * an error predicate that treats ``NULL`` and the empty string as "no
#     error" (Python truthiness of ``r.error``);
#   * ``percentile_disc(p)`` which is the exact SQL equivalent of
#     ``_percentile`` — both pick the nearest rank with ``ceil(p*n)``.
# ``date_trunc('minute', started_at)`` keeps the column's ``timestamptz`` type
# so asyncpg returns a UTC-aware datetime whose ``.isoformat()`` carries the
# ``+00:00`` offset, matching the old ``_truncate_minute(...).isoformat()``.


def _dur() -> sa.ColumnElement:
    """``max(0, int(duration_ms or 0))`` in SQL."""
    return sa.func.greatest(sa.func.coalesce(RequestStat.duration_ms, 0), 0)


def _pos(col: sa.ColumnElement) -> sa.ColumnElement:
    """``max(0, int(col or 0))`` in SQL (non-negative integer column)."""
    return sa.func.greatest(sa.func.coalesce(col, 0), 0)


def _is_error_sql(*, exclude_429: bool = False) -> sa.ColumnElement:
    """Mirror ``r.error or (status_code is not None and status_code >= 400)``.

    ``r.error`` is Python-truthy only for a non-empty string, so an empty
    string counts as no error. With ``exclude_429`` the admission back-pressure
    status is kept out of the *status* branch (but an explicit error message on
    a 429 still counts, exactly as the old code did).
    """
    status = sa.and_(RequestStat.status_code.is_not(None), RequestStat.status_code >= 400)
    if exclude_429:
        status = sa.and_(status, RequestStat.status_code != 429)
    return sa.or_(sa.and_(RequestStat.error.is_not(None), RequestStat.error != ""), status)


def _tokenized_sql() -> sa.ColumnElement:
    """Mirror ``(input_tokens or 0) > 0 or (output_tokens or 0) > 0``."""
    return sa.or_(
        sa.func.coalesce(RequestStat.input_tokens, 0) > 0,
        sa.func.coalesce(RequestStat.output_tokens, 0) > 0,
    )


def _cost_from_token_groups(groups) -> float:
    """Reference USD cost from token sums grouped by what sets the price tier.

    ``classify_model_tier`` depends only on ``(model_id, backend_type,
    request_kind)`` and ``estimate_request_cost`` is linear in the token
    counts, so summing tokens per that triple in SQL and costing each group
    gives the same total as costing every row (``estimate_cost_for_rows``) —
    without loading the rows. ``groups`` yields
    ``(model_id, backend_type, request_kind, input_sum, output_sum)``.
    """
    total = 0.0
    for model_id, backend_type, request_kind, in_sum, out_sum in groups:
        tier = classify_model_tier(model_id, backend_type, request_kind)
        total += estimate_request_cost(int(in_sum or 0), int(out_sum or 0), tier)
    return total


async def _cost_by_entity(session, entity_col, window) -> dict:
    """USD cost per entity, grouped in SQL by the cost-tier triple.

    Returns ``{entity_value: cost}``. ``window`` is the list of WHERE clauses
    (time range plus any ``isnot(None)`` guard the caller already applies).
    """
    rows = (
        await session.execute(
            sa.select(
                entity_col,
                RequestStat.model_id,
                RequestStat.backend_type,
                RequestStat.request_kind,
                sa.func.sum(_pos(RequestStat.input_tokens)),
                sa.func.sum(_pos(RequestStat.output_tokens)),
            )
            .where(*window)
            .group_by(
                entity_col,
                RequestStat.model_id,
                RequestStat.backend_type,
                RequestStat.request_kind,
            )
        )
    ).all()
    by_entity: dict = defaultdict(list)
    for entity, model_id, backend_type, request_kind, in_sum, out_sum in rows:
        by_entity[entity].append((model_id, backend_type, request_kind, in_sum, out_sum))
    return {entity: _cost_from_token_groups(groups) for entity, groups in by_entity.items()}


async def _cost_total(session, window) -> float:
    """Total reference USD cost over a window, grouped by the cost-tier triple."""
    rows = (
        await session.execute(
            sa.select(
                RequestStat.model_id,
                RequestStat.backend_type,
                RequestStat.request_kind,
                sa.func.sum(_pos(RequestStat.input_tokens)),
                sa.func.sum(_pos(RequestStat.output_tokens)),
            )
            .where(*window)
            .group_by(RequestStat.model_id, RequestStat.backend_type, RequestStat.request_kind)
        )
    ).all()
    return _cost_from_token_groups(rows)


# Raw-SQL fragments used by the text() queries in get_performance_stats. Error
# predicate includes 429 (that function's error count does, unlike
# get_request_stats). ``error <> ''`` reproduces Python truthiness.
_IS_ERROR_SQL = (
    "((error IS NOT NULL AND error <> '') OR (status_code IS NOT NULL AND status_code >= 400))"
)


def _telemetry_sql(key: str) -> str:
    """One integer out of ``runtime_telemetry`` JSONB, like ``_telemetry_int``.

    Only accepts JSON numbers and truncates toward zero (``int(float)``),
    yielding NULL for a missing key or a non-numeric value.
    """
    return (
        f"CASE WHEN jsonb_typeof(runtime_telemetry -> '{key}') = 'number' "
        f"THEN trunc((runtime_telemetry ->> '{key}')::double precision)::bigint END"
    )


def _json_num(key: str) -> str:
    """Positive numeric ``work_size_meta`` field, else 0 — like ``_work_amount``."""
    return (
        f"CASE WHEN jsonb_typeof(work_size_meta -> '{key}') = 'number' "
        f"AND (work_size_meta ->> '{key}')::double precision > 0 "
        f"THEN (work_size_meta ->> '{key}')::double precision ELSE 0 END"
    )


# (field name, per-row amount SQL) for every ``field`` referenced by
# _WORK_SPEEDS. Token fields read a column; the rest a JSONB numeric.
_WORK_SPEED_FIELDS: tuple[tuple[str, str], ...] = (
    ("audio_seconds", _json_num("audio_seconds")),
    ("text_chars", _json_num("text_chars")),
    ("n_inputs", _json_num("n_inputs")),
    ("input_tokens", "GREATEST(COALESCE(input_tokens, 0), 0)"),
    ("n_documents", _json_num("n_documents")),
    ("n_images", _json_num("n_images")),
    ("input_frames", _json_num("input_frames")),
    ("model_mp", _json_num("model_mp")),
    ("output_tokens", "GREATEST(COALESCE(output_tokens, 0), 0)"),
)


def _work_speeds_sql(where: str) -> str:
    """Per ``(model_id, request_kind)`` work/time/count for each work field.

    ``ms`` and ``success`` match ``work_speeds``: wall time minus the load and
    queue telemetry, and a non-error, non-4xx response.
    """
    ms = (
        f"GREATEST(0, COALESCE(duration_ms, 0) - COALESCE({_telemetry_sql('load_ms')}, 0) "
        f"- COALESCE({_telemetry_sql('queue_ms')}, 0))"
    )
    base_cols = [
        "model_id",
        "request_kind",
        "((error IS NULL OR error = '') AND (status_code IS NULL OR status_code < 400)) AS success",
        f"{ms} AS ms",
    ]
    base_cols += [f"{amount} AS amt_{name}" for name, amount in _WORK_SPEED_FIELDS]
    agg_cols: list[str] = []
    for name, _amount in _WORK_SPEED_FIELDS:
        cond = f"success AND amt_{name} > 0 AND ms > 0"
        agg_cols.append(f"COALESCE(SUM(amt_{name}) FILTER (WHERE {cond}), 0) AS w_{name}")
        agg_cols.append(f"COALESCE(SUM(ms) FILTER (WHERE {cond}), 0) AS ms_{name}")
        agg_cols.append(f"COUNT(*) FILTER (WHERE {cond}) AS n_{name}")
    return f"""
        WITH base AS (
            SELECT {", ".join(base_cols)}
            FROM request_stats
            WHERE {where}
        )
        SELECT model_id, request_kind, {", ".join(agg_cols)}
        FROM base
        GROUP BY model_id, request_kind
    """


async def _summary_by(session, window, key_col, key_name: str, default: str) -> list[dict]:
    """``_summarize`` groups (totalRequests/errorRate/avgLatency/p95) in SQL.

    Grouped by ``COALESCE(key_col, default)``; the list is sorted in Python by
    the key (codepoint, like the old ``sorted(by_backend.items())`` / ``by_kind``)
    rather than by a DB ``ORDER BY`` whose locale collation could reorder keys.
    The error count includes 429 (the overview/my-stats summaries always did).
    """
    label = sa.func.coalesce(key_col, default)
    rows = (
        await session.execute(
            sa.select(
                label,
                sa.func.count(),
                sa.func.count().filter(_is_error_sql()),
                sa.func.coalesce(sa.func.sum(_dur()), 0),
                sa.func.percentile_disc(0.95).within_group(_dur()),
            )
            .where(*window)
            .group_by(label)
        )
    ).all()
    out = []
    for key_value, g_n, g_errors, g_dur, p95 in rows:
        g_n = int(g_n)
        out.append(
            {
                key_name: key_value,
                "totalRequests": g_n,
                "errorRate": round(int(g_errors) / g_n, 4) if g_n else 0.0,
                "avgLatencyMs": int(int(g_dur) / g_n) if g_n else 0,
                "p95LatencyMs": int(p95 or 0),
            }
        )
    # Sort in Python (codepoint) to match the old ``sorted(dict.items())``
    # instead of the DB's locale collation.
    out.sort(key=lambda d: d[key_name])
    return out


def _work_speeds_from_groups(
    kind_map: dict[str, dict[str, tuple[float, float, int]]],
    present_kinds: set[str],
) -> list[dict]:
    """Recombine the per-kind SQL sums into ``work_speeds``' output shape."""
    out: list[dict] = []
    for kinds, unit, field, rate in _WORK_SPEEDS:
        work = 0.0
        ms_total = 0.0
        samples = 0
        for k in kinds:
            per_field = kind_map.get(k)
            if not per_field:
                continue
            w, ms, n = per_field.get(field, (0.0, 0.0, 0))
            work += w
            ms_total += ms
            samples += n
        seconds = ms_total / 1000.0
        if not samples or seconds <= 0 or work <= 0:
            continue
        value = work / seconds if rate else seconds / work
        out.append(
            {
                "kinds": [k for k in kinds if k in present_kinds],
                "unit": unit,
                "value": round(value, 2),
                "samples": samples,
            }
        )
    return out


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

    window = [RequestStat.started_at >= from_dt, RequestStat.started_at <= to_dt]
    if model_id:
        window.append(RequestStat.model_id == model_id)

    minute = sa.func.date_trunc("minute", RequestStat.started_at)
    async with AsyncSessionLocal() as session:
        # Admission rejections (429) are intentional back-pressure, not
        # failures — count them separately and keep them out of the error rate.
        n, errors, rejections, total_dur, p50, p95 = (
            await session.execute(
                sa.select(
                    sa.func.count(),
                    sa.func.count().filter(_is_error_sql(exclude_429=True)),
                    sa.func.count().filter(RequestStat.status_code == 429),
                    sa.func.coalesce(sa.func.sum(_dur()), 0),
                    sa.func.percentile_disc(0.5).within_group(_dur()),
                    sa.func.percentile_disc(0.95).within_group(_dur()),
                ).where(*window)
            )
        ).one()
        series_rows = (
            await session.execute(
                sa.select(minute.label("t"), sa.func.count())
                .where(*window)
                .group_by(minute)
                .order_by(minute)
            )
        ).all()

    n = int(n or 0)
    if not n:
        return {
            "totalRequests": 0,
            "errorRate": 0.0,
            "rejections": 0,
            "avgDurationMs": 0,
            "p50DurationMs": 0,
            "p95DurationMs": 0,
            "series": [],
        }

    series = [{"timestamp": ts.isoformat(), "count": int(count)} for ts, count in series_rows]

    return {
        "totalRequests": n,
        "errorRate": round(int(errors) / n, 4),
        "rejections": int(rejections),
        "avgDurationMs": int(int(total_dur) / n),
        "p50DurationMs": int(p50 or 0),
        "p95DurationMs": int(p95 or 0),
        "series": series,
    }


async def get_gpu_usage_stats(
    from_dt: datetime | None = None,
    to_dt: datetime | None = None,
    active_threshold_pct: float = 5.0,
) -> dict:
    """Aggregated GPU usage over a time window.

    Returns per-GPU totals (``hours_total``, ``hours_active`` where
    ``utilization_pct >= active_threshold_pct``, average/peak VRAM, average
    temperature, estimated kWh) plus a bucketed time series ready for a
    chart. Bucketing switches from hourly to daily when the window exceeds
    72 h so the client never gets more than ~200 points per GPU.

    Interval accounting caps per-sample gaps at
    ``settings.gpu_stats_expected_interval_s`` (default 90s — a bit above
    hw-monitor's real ~60s persist cadence) so a server restart or monitor
    pause doesn't inflate the "covered time" totals, without the cap itself
    truncating every normal interval (2026-10-26: it used to hardcode 30s
    against a real 60s cadence, silently halving every coverage/activity/kWh
    figure in this endpoint, always, regardless of window size).
    """
    from_dt, to_dt = _normalize_window(from_dt, to_dt)
    window_hours = max(0.0, (to_dt - from_dt).total_seconds() / 3600.0)
    bucket = "day" if window_hours > 72 else "hour"

    totals_q = sa.text(
        """
        WITH ordered AS (
            SELECT
                gpu_index,
                recorded_at,
                utilization_pct,
                vram_used_mb,
                power_draw_w,
                temperature_c,
                LEAD(recorded_at) OVER (
                    PARTITION BY gpu_index ORDER BY recorded_at
                ) AS next_at
            FROM gpu_stats
            WHERE recorded_at BETWEEN :from_dt AND :to_dt
        ),
        with_gap AS (
            SELECT
                gpu_index,
                utilization_pct,
                vram_used_mb,
                power_draw_w,
                temperature_c,
                LEAST(
                    EXTRACT(EPOCH FROM (COALESCE(next_at, :to_dt) - recorded_at)),
                    :gap_cap_s
                ) AS gap_s
            FROM ordered
        )
        SELECT
            gpu_index,
            COUNT(*) AS samples,
            COALESCE(SUM(gap_s), 0) / 3600.0 AS hours_total,
            COALESCE(SUM(gap_s) FILTER (
                WHERE utilization_pct >= :threshold
            ), 0) / 3600.0 AS hours_active,
            AVG(utilization_pct) AS avg_util,
            AVG(vram_used_mb) AS avg_vram,
            MAX(vram_used_mb) AS peak_vram,
            AVG(temperature_c) AS avg_temp,
            COALESCE(SUM(power_draw_w * gap_s), 0) / 3600.0 / 1000.0 AS kwh
        FROM with_gap
        GROUP BY gpu_index
        ORDER BY gpu_index
        """
    )

    series_q = sa.text(
        """
        SELECT
            date_trunc(:bucket, recorded_at) AS t,
            gpu_index,
            AVG(utilization_pct) AS util,
            AVG(vram_used_mb) AS vram,
            AVG(power_draw_w) AS power,
            AVG(temperature_c) AS temp
        FROM gpu_stats
        WHERE recorded_at BETWEEN :from_dt AND :to_dt
        GROUP BY t, gpu_index
        ORDER BY t, gpu_index
        """
    )

    async with AsyncSessionLocal() as session:
        totals_result = await session.execute(
            totals_q,
            {
                "from_dt": from_dt,
                "to_dt": to_dt,
                "threshold": active_threshold_pct,
                "gap_cap_s": settings.gpu_stats_expected_interval_s,
            },
        )
        totals_rows = totals_result.all()
        series_result = await session.execute(
            series_q,
            {"bucket": bucket, "from_dt": from_dt, "to_dt": to_dt},
        )
        series_rows = series_result.all()

    totals = []
    for row in totals_rows:
        m = row._mapping
        hours_total = float(m["hours_total"] or 0.0)
        hours_active = float(m["hours_active"] or 0.0)
        totals.append(
            {
                "gpuIndex": int(m["gpu_index"]),
                "samples": int(m["samples"] or 0),
                "hoursTotal": round(hours_total, 3),
                "hoursActive": round(hours_active, 3),
                "activeRatio": round(hours_active / hours_total if hours_total else 0.0, 4),
                "avgUtilizationPct": round(float(m["avg_util"] or 0.0), 2),
                "avgVramMb": int(round(float(m["avg_vram"] or 0.0))),
                "peakVramMb": int(m["peak_vram"] or 0),
                "avgTempC": round(float(m["avg_temp"] or 0.0), 1),
                "kwh": round(float(m["kwh"] or 0.0), 4),
            }
        )

    series: list[dict] = []
    for row in series_rows:
        m = row._mapping
        t = m["t"]
        series.append(
            {
                "t": t.isoformat() if t is not None else None,
                "gpuIndex": int(m["gpu_index"]),
                "utilizationPct": round(float(m["util"] or 0.0), 2),
                "vramMb": int(round(float(m["vram"] or 0.0))),
                "powerW": round(float(m["power"] or 0.0), 1),
                "tempC": round(float(m["temp"] or 0.0), 1),
            }
        )

    return {
        "from": from_dt.isoformat(),
        "to": to_dt.isoformat(),
        "bucket": bucket,
        "activeThresholdPct": active_threshold_pct,
        "totals": totals,
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

    from ocabra.config import settings

    # Integrate power over each sample's interval (its timestamp to the next
    # one, or to the window end for the last). Unlike get_gpu_usage_stats this
    # applies NO gap cap — the old Python code didn't either. An interval whose
    # end is not after its start contributes nothing (gap_s > 0 filter), which
    # matches the old ``if end <= start: continue``.
    gpu_q = sa.text(
        """
        WITH ordered AS (
            SELECT
                gpu_index,
                recorded_at,
                power_draw_w,
                LEAD(recorded_at) OVER (
                    PARTITION BY gpu_index ORDER BY recorded_at
                ) AS next_at
            FROM gpu_stats
            WHERE recorded_at BETWEEN :from_dt AND :to_dt
        ),
        gaps AS (
            SELECT
                gpu_index,
                COALESCE(power_draw_w, 0) AS power_draw_w,
                EXTRACT(EPOCH FROM (COALESCE(next_at, :to_dt) - recorded_at)) AS gap_s
            FROM ordered
        )
        SELECT
            gpu_index,
            COALESCE(SUM(power_draw_w * gap_s) FILTER (WHERE gap_s > 0), 0) / 3600.0 AS energy_wh,
            COALESCE(SUM(gap_s) FILTER (WHERE gap_s > 0), 0) AS total_seconds
        FROM gaps
        GROUP BY gpu_index
        ORDER BY gpu_index
        """
    )
    cpu_q = sa.text(
        """
        WITH ordered AS (
            SELECT
                recorded_at,
                cpu_power_w,
                LEAD(recorded_at) OVER (ORDER BY recorded_at) AS next_at
            FROM server_stats
            WHERE recorded_at BETWEEN :from_dt AND :to_dt
        ),
        gaps AS (
            SELECT
                COALESCE(cpu_power_w, 0) AS cpu_power_w,
                EXTRACT(EPOCH FROM (COALESCE(next_at, :to_dt) - recorded_at)) AS gap_s
            FROM ordered
        )
        SELECT
            COALESCE(SUM(cpu_power_w * gap_s) FILTER (WHERE gap_s > 0), 0) / 3600.0 AS cpu_wh
        FROM gaps
        """
    )
    params = {"from_dt": from_dt, "to_dt": to_dt}

    async with AsyncSessionLocal() as session:
        gpu_rows = (await session.execute(gpu_q, params)).all()
        cpu_wh = (await session.execute(cpu_q, params)).scalar() or 0.0

    gpu_summaries = []
    total_kwh = 0.0
    for row in gpu_rows:
        m = row._mapping
        energy_wh = float(m["energy_wh"] or 0.0)
        total_seconds = float(m["total_seconds"] or 0.0)
        kwh = energy_wh / 1000.0
        total_kwh += kwh
        avg_power_w = energy_wh / (total_seconds / 3600.0) if total_seconds else 0.0
        gpu_summaries.append(
            {
                "gpuIndex": int(m["gpu_index"]),
                "totalKwh": round(kwh, 4),
                "powerDrawW": round(avg_power_w, 1),
            }
        )

    cpu_kwh = float(cpu_wh) / 1000.0
    total_server_kwh = total_kwh + cpu_kwh

    return {
        "totalKwh": round(total_kwh, 4),
        "estimatedCostEur": round(total_kwh * settings.energy_cost_eur_kwh, 4),
        "byGpu": gpu_summaries,
        "cpuKwh": round(cpu_kwh, 4),
        "totalServerKwh": round(total_server_kwh, 4),
    }


def _telemetry_int(row, key: str) -> int | None:
    """Read one integer out of a request's runtime telemetry, tolerating rows
    written before the field existed (older DB rows, fixtures)."""
    telemetry = getattr(row, "runtime_telemetry", None)
    if not isinstance(telemetry, dict):
        return None
    try:
        value = telemetry.get(key)
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


async def get_performance_stats(
    model_id: str | None = None,
    from_dt: datetime | None = None,
    to_dt: datetime | None = None,
) -> dict:
    """
    Return per-model performance statistics.
    """
    from_dt, to_dt = _normalize_window(from_dt, to_dt)

    # All of the per-model aggregates below are computed in PostgreSQL instead
    # of loading every request_stats row into Python (REF-12). The telemetry
    # figures (ttft/queue/load) come out of the ``runtime_telemetry`` JSONB
    # with the same truncate-towards-zero semantics as ``_telemetry_int`` via
    # ``jsonb_typeof(...)='number'`` + ``trunc(...)``; ``percentile_disc``
    # reproduces ``_percentile`` (nearest rank).
    where = "started_at BETWEEN :from_dt AND :to_dt"
    params: dict = {"from_dt": from_dt, "to_dt": to_dt}
    if model_id:
        where += " AND model_id = :model_id"
        params["model_id"] = model_id

    d = "GREATEST(COALESCE(duration_ms, 0), 0)"
    tin = "GREATEST(COALESCE(input_tokens, 0), 0)"
    tout = "GREATEST(COALESCE(output_tokens, 0), 0)"
    eff = "CASE WHEN COALESCE(generation_ms, 0) <> 0 THEN generation_ms ELSE duration_ms END"
    ttft = _telemetry_sql("ttft_ms")
    queue = _telemetry_sql("queue_ms")
    load = _telemetry_sql("load_ms")
    engine = f"GREATEST(0, {ttft} - COALESCE({load}, 0) - COALESCE({queue}, 0))"
    # percentile_disc ignores NULLs in its ordering input, so a CASE that maps
    # excluded rows to NULL is the portable equivalent of FILTER on an
    # ordered-set aggregate (FILTER + WITHIN GROUP is not universally allowed).
    ttft_val = f"CASE WHEN {ttft} IS NOT NULL AND {ttft} >= 0 THEN {ttft} END"
    engine_val = f"CASE WHEN {ttft} IS NOT NULL THEN {engine} END"
    queue_val = f"CASE WHEN {queue} IS NOT NULL AND {queue} >= 0 THEN {queue} END"
    load_val = f"CASE WHEN {load} IS NOT NULL AND {load} > 0 THEN {load} END"

    perf_q = sa.text(
        f"""
        SELECT
            model_id,
            COUNT(*) AS n,
            COUNT(*) FILTER (WHERE {_IS_ERROR_SQL}) AS errors,
            COALESCE(SUM({d}), 0) AS total_dur,
            COALESCE(SUM({tin}), 0) AS sum_in,
            COALESCE(SUM({tout}), 0) AS sum_out,
            COUNT(*) FILTER (
                WHERE COALESCE(input_tokens, 0) > 0 OR COALESCE(output_tokens, 0) > 0
            ) AS tokenized,
            percentile_disc(0.95) WITHIN GROUP (ORDER BY {d}) AS p95_dur,
            COALESCE(SUM({tout}) FILTER (WHERE ({eff}) IS NOT NULL AND ({eff}) <> 0), 0)
                AS tokens_for_rate,
            COALESCE(
                SUM(GREATEST({eff}, 0)) FILTER (WHERE ({eff}) IS NOT NULL AND ({eff}) <> 0), 0
            ) AS ms_for_rate,
            COALESCE(SUM(COALESCE(draft_n, 0)), 0) AS draft_n_total,
            COALESCE(SUM(COALESCE(draft_n_accepted, 0)), 0) AS draft_acc_total,
            SUM(predicted_tokens_per_second)
                FILTER (WHERE COALESCE(predicted_tokens_per_second, 0) <> 0) AS pred_sum,
            COUNT(predicted_tokens_per_second)
                FILTER (WHERE COALESCE(predicted_tokens_per_second, 0) <> 0) AS pred_n,
            SUM(prompt_tokens_per_second)
                FILTER (WHERE COALESCE(prompt_tokens_per_second, 0) <> 0) AS prompt_sum,
            COUNT(prompt_tokens_per_second)
                FILTER (WHERE COALESCE(prompt_tokens_per_second, 0) <> 0) AS prompt_n,
            MIN(backend_type) FILTER (WHERE backend_type IS NOT NULL) AS backend_type,
            array_remove(array_agg(DISTINCT request_kind), NULL) AS kinds_present,
            bool_or(request_kind IS NULL) AS has_null_kind,
            COUNT(*) FILTER (WHERE {ttft} IS NOT NULL AND {ttft} >= 0) AS ttft_n,
            percentile_disc(0.5) WITHIN GROUP (ORDER BY {ttft_val}) AS ttft_p50,
            percentile_disc(0.95) WITHIN GROUP (ORDER BY {ttft_val}) AS ttft_p95,
            COUNT(*) FILTER (WHERE {ttft} IS NOT NULL) AS engine_n,
            percentile_disc(0.5) WITHIN GROUP (ORDER BY {engine_val}) AS engine_p50,
            percentile_disc(0.95) WITHIN GROUP (ORDER BY {engine_val}) AS engine_p95,
            COUNT(*) FILTER (WHERE {queue} IS NOT NULL AND {queue} > 1000) AS queued_requests,
            COALESCE(MAX({queue}) FILTER (WHERE {queue} IS NOT NULL AND {queue} >= 0), 0)
                AS max_queue,
            COUNT(*) FILTER (WHERE {queue} IS NOT NULL AND {queue} >= 0) AS queue_n,
            percentile_disc(0.95) WITHIN GROUP (ORDER BY {queue_val}) AS queue_p95,
            COUNT(*) FILTER (WHERE {load} IS NOT NULL AND {load} > 0) AS load_n,
            percentile_disc(0.5) WITHIN GROUP (ORDER BY {load_val}) AS load_p50,
            COALESCE(MAX({load}) FILTER (WHERE {load} IS NOT NULL AND {load} > 0), 0) AS max_load
        FROM request_stats
        WHERE {where}
        GROUP BY model_id
        """
    )

    ws_q = sa.text(_work_speeds_sql(where))

    # Loads are recorded per deployment/worker (``<backend>/<key>[::hash]``)
    # while requests are recorded per logical model (``<key>``): fold the loads
    # onto their model with the same regex ``_model_key_of_worker`` applies.
    load_q = sa.text(
        """
        WITH stripped AS (
            SELECT
                regexp_replace(model_id, '::[0-9a-f]{12}$', '') AS s,
                backend_type,
                GREATEST(COALESCE(duration_ms, 0), 0) AS dur
            FROM model_load_stats
            WHERE started_at BETWEEN :from_dt AND :to_dt
        ),
        keyed AS (
            SELECT
                CASE WHEN position('/' IN s) > 0
                     THEN substring(s FROM position('/' IN s) + 1)
                     ELSE s END AS mkey,
                backend_type,
                dur
            FROM stripped
        )
        SELECT
            mkey,
            COUNT(*) AS load_count,
            COALESCE(SUM(dur), 0) AS load_dur_sum,
            percentile_disc(0.95) WITHIN GROUP (ORDER BY dur) AS load_p95,
            COALESCE(MAX(dur), 0) AS last_load,
            MIN(backend_type) FILTER (WHERE backend_type IS NOT NULL) AS load_backend
        FROM keyed
        GROUP BY mkey
        """
    )

    async with AsyncSessionLocal() as session:
        perf_rows = (await session.execute(perf_q, params)).all()
        ws_rows = (await session.execute(ws_q, params)).all()
        load_rows = (await session.execute(load_q, params)).all()

    # Work speeds: per (model, request_kind) sums per field, recombined in
    # Python exactly as ``work_speeds`` would over the rows.
    ws_by_model: dict[str, dict[str, dict[str, tuple[float, float, int]]]] = defaultdict(dict)
    for row in ws_rows:
        m = row._mapping
        mid = m["model_id"]
        kind = m["request_kind"]
        per_field: dict[str, tuple[float, float, int]] = {}
        for field_name, _col in _WORK_SPEED_FIELDS:
            per_field[field_name] = (
                float(m[f"w_{field_name}"] or 0.0),
                float(m[f"ms_{field_name}"] or 0.0),
                int(m[f"n_{field_name}"] or 0),
            )
        ws_by_model[mid][kind] = per_field

    load_by_model: dict[str, dict] = {}
    for row in load_rows:
        m = row._mapping
        mkey = m["mkey"]
        if model_id and mkey != model_id:
            continue
        load_by_model[mkey] = dict(m)

    perf_by_model: dict[str, dict] = {}
    for row in perf_rows:
        m = row._mapping
        perf_by_model[m["model_id"]] = dict(m)

    summaries = []
    for mid in sorted(set(perf_by_model) | set(load_by_model)):
        p = perf_by_model.get(mid)
        lr = load_by_model.get(mid)
        n = int(p["n"]) if p else 0
        total_duration_ms = int(p["total_dur"]) if p else 0

        if p:
            draft_n_total = int(p["draft_n_total"])
            draft_acceptance_rate = (
                round(int(p["draft_acc_total"]) / draft_n_total, 3) if draft_n_total else None
            )
            pred_n = int(p["pred_n"] or 0)
            avg_predicted_tps = round(float(p["pred_sum"]) / pred_n, 2) if pred_n else None
            prompt_n = int(p["prompt_n"] or 0)
            avg_prompt_tps = round(float(p["prompt_sum"]) / prompt_n, 2) if prompt_n else None
            tokens_for_rate = int(p["tokens_for_rate"])
            ms_for_rate = int(p["ms_for_rate"])
            tokens_ps = (tokens_for_rate / ms_for_rate * 1000) if ms_for_rate else 0.0
            req_per_min = (n * 60000 / total_duration_ms) if total_duration_ms else 0.0
            errors = int(p["errors"])
            kinds_present = list(p["kinds_present"] or [])
            # ``sorted({(r.request_kind or "other") ...})``: array_agg(DISTINCT)
            # drops NULL request_kinds, so add "other" back when has_null_kind.
            request_kinds = sorted(
                set(kinds_present) | ({"other"} if p["has_null_kind"] else set())
            )
            ttft_n = int(p["ttft_n"] or 0)
            engine_n = int(p["engine_n"] or 0)
            queue_n = int(p["queue_n"] or 0)
            load_n = int(p["load_n"] or 0)
        else:
            draft_acceptance_rate = None
            avg_predicted_tps = None
            avg_prompt_tps = None
            tokens_ps = 0.0
            req_per_min = 0.0
            errors = 0
            kinds_present = []
            request_kinds = []
            ttft_n = engine_n = queue_n = load_n = 0

        kinds_set = set(kinds_present)
        work_speeds_out = _work_speeds_from_groups(ws_by_model.get(mid, {}), kinds_set)

        if lr:
            load_count = int(lr["load_count"])
            avg_load_ms = int(int(lr["load_dur_sum"]) / load_count) if load_count else 0
            p95_load_ms = int(lr["load_p95"] or 0) if load_count else 0
            last_load_ms = int(lr["last_load"] or 0)
        else:
            load_count = 0
            avg_load_ms = 0
            p95_load_ms = 0
            last_load_ms = 0

        backend_type = (
            (p["backend_type"] if p and p["backend_type"] else None)
            or (lr["load_backend"] if lr and lr["load_backend"] else None)
            or "unknown"
        )

        summaries.append(
            {
                "modelId": mid,
                "backendType": backend_type,
                "requestKinds": request_kinds,
                "totalRequests": n,
                "avgLatencyMs": int(total_duration_ms / n) if n else 0,
                "p95LatencyMs": int(p["p95_dur"] or 0) if p else 0,
                "requestsPerMinute": round(req_per_min, 2),
                "tokensPerSecond": round(tokens_ps, 2),
                "draftAcceptanceRate": draft_acceptance_rate,
                "workSpeeds": work_speeds_out,
                "avgPredictedTokensPerSecond": avg_predicted_tps,
                "avgPromptTokensPerSecond": avg_prompt_tps,
                "totalInputTokens": int(p["sum_in"]) if p else 0,
                "totalOutputTokens": int(p["sum_out"]) if p else 0,
                "tokenizedRequests": int(p["tokenized"]) if p else 0,
                "errorCount": errors,
                "uptimePct": round((n - errors) / n * 100, 1) if n else 0.0,
                "loadCount": load_count,
                "avgLoadMs": avg_load_ms,
                "p95LoadMs": p95_load_ms,
                "lastLoadMs": last_load_ms,
                "ttftSamples": ttft_n,
                "ttftP50Ms": int(p["ttft_p50"]) if p and ttft_n else None,
                "ttftP95Ms": int(p["ttft_p95"]) if p and ttft_n else None,
                "ttftEngineP50Ms": int(p["engine_p50"]) if p and engine_n else None,
                "ttftEngineP95Ms": int(p["engine_p95"]) if p and engine_n else None,
                "queuedRequests": int(p["queued_requests"]) if p else 0,
                "maxQueueMs": int(p["max_queue"]) if p else 0,
                "queueP95Ms": int(p["queue_p95"]) if p and queue_n else None,
                "coldStartRequests": load_n,
                "loadP50Ms": int(p["load_p50"]) if p and load_n else None,
                "maxLoadMs": int(p["max_load"]) if p else 0,
            }
        )

    return {"byModel": summaries}


async def get_stats_by_user(
    from_dt: datetime | None = None,
    to_dt: datetime | None = None,
) -> dict:
    """Return request stats aggregated by user."""
    from_dt, to_dt = _normalize_window(from_dt, to_dt)
    from ocabra.db.auth import User

    window = [
        RequestStat.started_at >= from_dt,
        RequestStat.started_at <= to_dt,
        RequestStat.user_id.isnot(None),
    ]
    async with AsyncSessionLocal() as session:
        agg_rows = (
            await session.execute(
                sa.select(
                    RequestStat.user_id,
                    sa.func.count(),
                    sa.func.count().filter(_is_error_sql()),
                    sa.func.coalesce(sa.func.sum(_dur()), 0),
                    sa.func.coalesce(sa.func.sum(_pos(RequestStat.input_tokens)), 0),
                    sa.func.coalesce(sa.func.sum(_pos(RequestStat.output_tokens)), 0),
                    sa.func.sum(RequestStat.energy_wh),
                )
                .where(*window)
                .group_by(RequestStat.user_id)
            )
        ).all()
        cost_by_uid = await _cost_by_entity(session, RequestStat.user_id, window)

        user_ids = {r[0] for r in agg_rows}
        username_map: dict = {}
        if user_ids:
            u_result = await session.execute(sa.select(User).where(User.id.in_(user_ids)))
            for u in u_result.scalars().all():
                username_map[u.id] = u.username

    summaries = []
    for uid, n, errors, total_dur, sum_in, sum_out, energy in sorted(
        agg_rows, key=lambda r: (-int(r[1]), str(r[0]))
    ):
        n = int(n)
        summaries.append(
            {
                "userId": str(uid),
                "username": username_map.get(uid, str(uid)),
                "totalRequests": n,
                "totalErrors": int(errors),
                "avgDurationMs": int(int(total_dur) / n) if n else 0,
                "totalInputTokens": int(sum_in),
                "totalOutputTokens": int(sum_out),
                "totalEnergyWh": round(float(energy or 0), 2),
                "estimatedCostUsd": round(cost_by_uid.get(uid, 0.0), 4),
            }
        )
    return {"byUser": summaries}


async def get_stats_by_group(
    from_dt: datetime | None = None,
    to_dt: datetime | None = None,
) -> dict:
    """Return request stats aggregated by group."""
    from_dt, to_dt = _normalize_window(from_dt, to_dt)
    from ocabra.db.auth import Group

    window = [
        RequestStat.started_at >= from_dt,
        RequestStat.started_at <= to_dt,
        RequestStat.group_id.isnot(None),
    ]
    async with AsyncSessionLocal() as session:
        agg_rows = (
            await session.execute(
                sa.select(
                    RequestStat.group_id,
                    sa.func.count(),
                    sa.func.count().filter(_is_error_sql()),
                    sa.func.coalesce(sa.func.sum(_dur()), 0),
                    sa.func.coalesce(sa.func.sum(_pos(RequestStat.input_tokens)), 0),
                    sa.func.coalesce(sa.func.sum(_pos(RequestStat.output_tokens)), 0),
                    sa.func.sum(RequestStat.energy_wh),
                )
                .where(*window)
                .group_by(RequestStat.group_id)
            )
        ).all()
        cost_by_gid = await _cost_by_entity(session, RequestStat.group_id, window)

        group_ids = {r[0] for r in agg_rows}
        group_name_map: dict = {}
        if group_ids:
            g_result = await session.execute(sa.select(Group).where(Group.id.in_(group_ids)))
            for g in g_result.scalars().all():
                group_name_map[g.id] = g.name

    summaries = []
    for gid, n, errors, total_dur, sum_in, sum_out, energy in sorted(
        agg_rows, key=lambda r: (-int(r[1]), str(r[0]))
    ):
        n = int(n)
        summaries.append(
            {
                "groupId": str(gid),
                "groupName": group_name_map.get(gid, str(gid)),
                "totalRequests": n,
                "totalErrors": int(errors),
                "avgDurationMs": int(int(total_dur) / n) if n else 0,
                "totalInputTokens": int(sum_in),
                "totalOutputTokens": int(sum_out),
                "totalEnergyWh": round(float(energy or 0), 2),
                "estimatedCostUsd": round(cost_by_gid.get(gid, 0.0), 4),
            }
        )
    return {"byGroup": summaries}


async def get_batch_vs_interactive_stats(
    from_dt: datetime | None = None,
    to_dt: datetime | None = None,
) -> dict:
    """Request volume split between Batches API traffic and direct
    (interactive) calls — validates whether the "move non-interactive
    workloads to batches" recommendation (2026-10-30) is actually being
    adopted, and gives the batch-vs-interactive priority mechanism's
    cost/throughput trade-off a number instead of a guess.
    """
    from_dt, to_dt = _normalize_window(from_dt, to_dt)

    window = [RequestStat.started_at >= from_dt, RequestStat.started_at <= to_dt]
    # UTC calendar day, independent of the DB session timezone, to match the
    # old ``started_at.date()`` on the UTC-aware column.
    utc_day = sa.cast(sa.text("started_at AT TIME ZONE 'UTC'"), sa.Date)
    async with AsyncSessionLocal() as session:
        agg_rows = (
            await session.execute(
                sa.select(
                    RequestStat.is_batch,
                    sa.func.count(),
                    sa.func.count().filter(_is_error_sql()),
                    sa.func.coalesce(sa.func.sum(_dur()), 0),
                    sa.func.coalesce(sa.func.sum(_pos(RequestStat.input_tokens)), 0),
                    sa.func.coalesce(sa.func.sum(_pos(RequestStat.output_tokens)), 0),
                    sa.func.sum(RequestStat.energy_wh),
                )
                .where(*window)
                .group_by(RequestStat.is_batch)
            )
        ).all()
        cost_by_batch = await _cost_by_entity(session, RequestStat.is_batch, window)
        daily_rows = (
            await session.execute(
                sa.select(utc_day.label("d"), RequestStat.is_batch, sa.func.count())
                .where(*window)
                .group_by(utc_day, RequestStat.is_batch)
                .order_by(utc_day)
            )
        ).all()

    agg_by_batch: dict[bool, tuple] = {bool(r[0]): r for r in agg_rows}

    def _summary(is_batch: bool) -> dict:
        row = agg_by_batch.get(is_batch)
        if row is None:
            return {
                "totalRequests": 0,
                "totalErrors": 0,
                "avgDurationMs": 0,
                "totalInputTokens": 0,
                "totalOutputTokens": 0,
                "totalTokens": 0,
                "totalEnergyWh": 0.0,
                "estimatedCostUsd": 0.0,
            }
        _b, n, errors, total_dur, sum_in, sum_out, energy = row
        n = int(n)
        total_input = int(sum_in)
        total_output = int(sum_out)
        return {
            "totalRequests": n,
            "totalErrors": int(errors),
            "avgDurationMs": int(int(total_dur) / n) if n else 0,
            "totalInputTokens": total_input,
            "totalOutputTokens": total_output,
            "totalTokens": total_input + total_output,
            "totalEnergyWh": round(float(energy or 0), 2),
            "estimatedCostUsd": round(cost_by_batch.get(is_batch, 0.0), 4),
        }

    daily: dict[str, dict[str, int]] = defaultdict(lambda: {"batch": 0, "interactive": 0})
    for day, is_batch, count in daily_rows:
        daily[day.isoformat()]["batch" if is_batch else "interactive"] += int(count)
    series = [
        {"date": day, "batch": counts["batch"], "interactive": counts["interactive"]}
        for day, counts in sorted(daily.items())
    ]

    return {
        "batch": _summary(True),
        "interactive": _summary(False),
        "dailySeries": series,
    }


async def get_recent_requests(limit: int = 20) -> dict:
    """Return the most recent N requests with user/group info."""
    from ocabra.db.auth import Group, User

    async with AsyncSessionLocal() as session:
        q = sa.select(RequestStat).order_by(RequestStat.started_at.desc()).limit(limit)
        result = await session.execute(q)
        rows = result.scalars().all()

        user_ids = {r.user_id for r in rows if r.user_id}
        group_ids = {r.group_id for r in rows if r.group_id}

        username_map: dict = {}
        if user_ids:
            u_result = await session.execute(sa.select(User).where(User.id.in_(user_ids)))
            for u in u_result.scalars().all():
                username_map[u.id] = u.username

        group_name_map: dict = {}
        if group_ids:
            g_result = await session.execute(sa.select(Group).where(Group.id.in_(group_ids)))
            for g in g_result.scalars().all():
                group_name_map[g.id] = g.name

    requests_out = []
    for r in rows:
        requests_out.append(
            {
                "id": str(r.id),
                "modelId": r.model_id,
                # The name the client used (profile, alias): tells a
                # "-diarized" transcription from a plain one of the same model.
                "requestedModel": getattr(r, "requested_model", None),
                "backendType": r.backend_type,
                "requestKind": r.request_kind,
                "endpointPath": r.endpoint_path,
                "statusCode": r.status_code,
                "startedAt": r.started_at.isoformat() if r.started_at else None,
                "durationMs": r.duration_ms,
                "inputTokens": r.input_tokens,
                "outputTokens": r.output_tokens,
                "error": r.error,
                "userId": str(r.user_id) if r.user_id else None,
                "username": username_map.get(r.user_id) if r.user_id else None,
                "groupId": str(r.group_id) if r.group_id else None,
                "groupName": group_name_map.get(r.group_id) if r.group_id else None,
                "apiKeyName": getattr(r, "api_key_name", None),
                # Federation: who served it, who sent it, and why.
                "node": getattr(r, "remote_node_name", None)
                or (r.remote_node_id if getattr(r, "remote_node_id", None) else None),
                "inboundFrom": getattr(r, "federation_inbound_from", None),
                "route": getattr(r, "federation_route", None),
                "clientAddr": getattr(r, "client_addr", None),
                "userAgent": getattr(r, "user_agent", None),
                "runtimeTelemetry": getattr(r, "runtime_telemetry", None),
                **_rendimiento_de_la_peticion(r),
            }
        )
    return {"requests": requests_out}


def _rendimiento_de_la_peticion(r) -> dict:
    """Lo que hace falta para diagnosticar UNA peticion, no un promedio.

    Estaba todo guardado pero no se publicaba por peticion: habia que ir al
    agregado por modelo (``/ocabra/stats/performance``) o leer la base de
    datos a mano. Y son tres cosas distintas que se diagnostican por separado:
    una cola larga es contencion, un prefill lento es un prompt enorme, y una
    generacion lenta es el modelo o la tarjeta.
    """

    telemetria = getattr(r, "runtime_telemetry", None)
    espera = None
    if isinstance(telemetria, dict):
        valor = telemetria.get("queue_ms")
        if valor is not None:
            try:
                espera = int(valor)
            except (TypeError, ValueError):
                espera = None
    return {
        "queueMs": espera,
        "promptTokensPerSecond": getattr(r, "prompt_tokens_per_second", None),
        "predictedTokensPerSecond": getattr(r, "predicted_tokens_per_second", None),
        "generationMs": getattr(r, "generation_ms", None),
        "cachedTokens": getattr(r, "cached_tokens", None),
        "effectiveContext": getattr(r, "effective_context", None),
        "runtimePreset": getattr(r, "runtime_preset", None),
        "gpuIndex": getattr(r, "gpu_index", None),
        # Duración del audio de una transcripción: con duration_ms da el
        # factor de tiempo real (×N) en el dashboard.
        "audioSeconds": _audio_seconds(getattr(r, "work_size_meta", None)),
        # Vídeo generado: fotogramas por segundo de generación y fotogramas.
        "videoFps": video_speed(getattr(r, "work_size_meta", None))[0],
        "videoFrames": video_speed(getattr(r, "work_size_meta", None))[1],
    }


async def get_recent_requests_for_user(user_id: str, limit: int = 20) -> dict:
    """Return the most recent N requests for a specific user."""
    import uuid as _uuid

    parsed_uid = _uuid.UUID(str(user_id))

    async with AsyncSessionLocal() as session:
        q = (
            sa.select(RequestStat)
            .where(RequestStat.user_id == parsed_uid)
            .order_by(RequestStat.started_at.desc())
            .limit(limit)
        )
        result = await session.execute(q)
        rows = result.scalars().all()

    requests_out = []
    for r in rows:
        requests_out.append(
            {
                "id": str(r.id),
                "modelId": r.model_id,
                # The name the client used (profile, alias): tells a
                # "-diarized" transcription from a plain one of the same model.
                "requestedModel": getattr(r, "requested_model", None),
                "backendType": r.backend_type,
                "requestKind": r.request_kind,
                "endpointPath": r.endpoint_path,
                "statusCode": r.status_code,
                "startedAt": r.started_at.isoformat() if r.started_at else None,
                "durationMs": r.duration_ms,
                "inputTokens": r.input_tokens,
                "outputTokens": r.output_tokens,
                "error": r.error,
                "apiKeyName": getattr(r, "api_key_name", None),
                # Federation: who served it, who sent it, and why.
                "node": getattr(r, "remote_node_name", None)
                or (r.remote_node_id if getattr(r, "remote_node_id", None) else None),
                "inboundFrom": getattr(r, "federation_inbound_from", None),
                "route": getattr(r, "federation_route", None),
                "runtimeTelemetry": getattr(r, "runtime_telemetry", None),
                **_rendimiento_de_la_peticion(r),
            }
        )
    return {"requests": requests_out}


async def get_my_stats(
    user_id_str: str,
    from_dt: datetime | None = None,
    to_dt: datetime | None = None,
    model_id: str | None = None,
) -> dict:
    """Return overview stats filtered to a specific user."""
    from_dt, to_dt = _normalize_window(from_dt, to_dt)
    import uuid as _uuid

    try:
        uid = _uuid.UUID(user_id_str)
    except ValueError:
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

    window = [
        RequestStat.started_at >= from_dt,
        RequestStat.started_at <= to_dt,
        RequestStat.user_id == uid,
    ]
    if model_id:
        window.append(RequestStat.model_id == model_id)

    async with AsyncSessionLocal() as session:
        total_requests, total_errors, tokenized_requests, total_dur, sum_in, sum_out = (
            await session.execute(
                sa.select(
                    sa.func.count(),
                    sa.func.count().filter(_is_error_sql()),
                    sa.func.count().filter(_tokenized_sql()),
                    sa.func.coalesce(sa.func.sum(_dur()), 0),
                    sa.func.coalesce(sa.func.sum(_pos(RequestStat.input_tokens)), 0),
                    sa.func.coalesce(sa.func.sum(_pos(RequestStat.output_tokens)), 0),
                ).where(*window)
            )
        ).one()
        total_requests = int(total_requests or 0)
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
        by_backend = await _summary_by(
            session, window, RequestStat.backend_type, "backendType", "unknown"
        )
        by_kind = await _summary_by(
            session, window, RequestStat.request_kind, "requestKind", "other"
        )

    return {
        "totalRequests": total_requests,
        "totalErrors": int(total_errors),
        "avgDurationMs": int(int(total_dur) / total_requests),
        "tokenizedRequests": int(tokenized_requests),
        "totalInputTokens": int(sum_in),
        "totalOutputTokens": int(sum_out),
        "byBackend": by_backend,
        "byRequestKind": by_kind,
    }


async def get_my_group_stats(
    group_ids: list[str],
    from_dt: datetime | None = None,
    to_dt: datetime | None = None,
) -> dict:
    """Return overview stats for a user's groups, plus group name."""
    from_dt, to_dt = _normalize_window(from_dt, to_dt)
    import uuid as _uuid

    from ocabra.db.auth import Group

    if not group_ids:
        return {
            "groupId": None,
            "groupName": None,
            "stats": {
                "totalRequests": 0,
                "totalErrors": 0,
                "avgDurationMs": 0,
                "tokenizedRequests": 0,
                "totalInputTokens": 0,
                "totalOutputTokens": 0,
                "byBackend": [],
                "byRequestKind": [],
            },
        }

    parsed_ids = []
    for gid in group_ids:
        try:
            parsed_ids.append(_uuid.UUID(gid))
        except ValueError:
            pass

    async with AsyncSessionLocal() as session:
        total, errors, tokenized, total_dur, sum_in, sum_out = (
            await session.execute(
                sa.select(
                    sa.func.count(),
                    sa.func.count().filter(_is_error_sql()),
                    sa.func.count().filter(_tokenized_sql()),
                    sa.func.coalesce(sa.func.sum(_dur()), 0),
                    sa.func.coalesce(sa.func.sum(_pos(RequestStat.input_tokens)), 0),
                    sa.func.coalesce(sa.func.sum(_pos(RequestStat.output_tokens)), 0),
                ).where(
                    RequestStat.started_at >= from_dt,
                    RequestStat.started_at <= to_dt,
                    RequestStat.group_id.in_(parsed_ids),
                )
            )
        ).one()

        group_name = None
        group_id_out = group_ids[0] if group_ids else None
        if parsed_ids:
            g = await session.execute(sa.select(Group).where(Group.id == parsed_ids[0]))
            grp = g.scalar_one_or_none()
            if grp:
                group_name = grp.name
                group_id_out = str(grp.id)

    total = int(total or 0)
    if total == 0:
        return {
            "groupId": group_id_out,
            "groupName": group_name,
            "stats": {
                "totalRequests": 0,
                "totalErrors": 0,
                "avgDurationMs": 0,
                "tokenizedRequests": 0,
                "totalInputTokens": 0,
                "totalOutputTokens": 0,
                "byBackend": [],
                "byRequestKind": [],
            },
        }

    stats = {
        "totalRequests": total,
        "totalErrors": int(errors),
        "avgDurationMs": int(int(total_dur) / total),
        "tokenizedRequests": int(tokenized),
        "totalInputTokens": int(sum_in),
        "totalOutputTokens": int(sum_out),
        "byBackend": [],
        "byRequestKind": [],
    }
    return {"groupId": group_id_out, "groupName": group_name, "stats": stats}


async def get_token_stats(
    from_dt: datetime | None = None,
    to_dt: datetime | None = None,
    model_id: str | None = None,
    *,
    all_time: bool = False,
    include_series: bool = True,
) -> dict:
    """
    Return token usage totals and per-request time series.
    """
    if not all_time:
        from_dt, to_dt = _normalize_window(from_dt, to_dt)

    filters = []
    if from_dt is not None:
        filters.append(RequestStat.started_at >= from_dt)
    if to_dt is not None:
        filters.append(RequestStat.started_at <= to_dt)
    if model_id:
        filters.append(RequestStat.model_id == model_id)

    async with AsyncSessionLocal() as session:
        input_sum = sa.func.coalesce(sa.func.sum(RequestStat.input_tokens), 0)
        output_sum = sa.func.coalesce(sa.func.sum(RequestStat.output_tokens), 0)

        total_result = await session.execute(sa.select(input_sum, output_sum).where(*filters))
        total_input_raw, total_output_raw = total_result.one()
        total_input = int(total_input_raw or 0)
        total_output = int(total_output_raw or 0)

        backend_expr = sa.func.coalesce(RequestStat.backend_type, "unknown")
        backend_result = await session.execute(
            sa.select(backend_expr, input_sum, output_sum)
            .where(*filters)
            .group_by(backend_expr)
            .order_by(backend_expr)
        )
        by_backend = [
            {
                "backendType": str(backend or "unknown"),
                "inputTokens": int(input_tokens or 0),
                "outputTokens": int(output_tokens or 0),
            }
            for backend, input_tokens, output_tokens in backend_result.all()
        ]

        gpu_result = await session.execute(
            sa.select(RequestStat.gpu_index, input_sum, output_sum)
            .where(*filters)
            .group_by(RequestStat.gpu_index)
        )
        by_gpu = [
            {
                "gpuIndex": gpu_index,
                "inputTokens": int(input_tokens or 0),
                "outputTokens": int(output_tokens or 0),
            }
            for gpu_index, input_tokens, output_tokens in gpu_result.all()
        ]

        series = []
        if include_series:
            minute = sa.func.date_trunc("minute", RequestStat.started_at)
            series_rows = (
                await session.execute(
                    sa.select(
                        minute.label("t"),
                        sa.func.coalesce(sa.func.sum(_pos(RequestStat.input_tokens)), 0),
                        sa.func.coalesce(sa.func.sum(_pos(RequestStat.output_tokens)), 0),
                    )
                    .where(*filters)
                    .group_by(minute)
                    .order_by(minute)
                )
            ).all()
            series = [
                {"timestamp": ts.isoformat(), "inputTokens": int(i), "outputTokens": int(o)}
                for ts, i, o in series_rows
            ]

    by_gpu.sort(
        key=lambda row: (
            row["gpuIndex"] is None,
            row["gpuIndex"] if row["gpuIndex"] is not None else 0,
        )
    )

    return {
        "totalInputTokens": total_input,
        "totalOutputTokens": total_output,
        "byBackend": by_backend,
        "byGpu": by_gpu,
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

    window = [RequestStat.started_at >= from_dt, RequestStat.started_at <= to_dt]
    if model_id:
        window.append(RequestStat.model_id == model_id)

    async with AsyncSessionLocal() as session:
        total_requests, total_errors, tokenized, total_dur, sum_in, sum_out, energy = (
            await session.execute(
                sa.select(
                    sa.func.count(),
                    sa.func.count().filter(_is_error_sql()),
                    sa.func.count().filter(_tokenized_sql()),
                    sa.func.coalesce(sa.func.sum(_dur()), 0),
                    sa.func.coalesce(sa.func.sum(_pos(RequestStat.input_tokens)), 0),
                    sa.func.coalesce(sa.func.sum(_pos(RequestStat.output_tokens)), 0),
                    sa.func.sum(RequestStat.energy_wh),
                ).where(*window)
            )
        ).one()
        total_requests = int(total_requests or 0)
        if total_requests == 0:
            return {
                "totalRequests": 0,
                "totalErrors": 0,
                "avgDurationMs": 0,
                "tokenizedRequests": 0,
                "totalInputTokens": 0,
                "totalOutputTokens": 0,
                "totalEnergyWh": 0.0,
                "estimatedCostUsd": 0.0,
                "byBackend": [],
                "byRequestKind": [],
            }
        cost = await _cost_total(session, window)
        by_backend = await _summary_by(
            session, window, RequestStat.backend_type, "backendType", "unknown"
        )
        by_kind = await _summary_by(
            session, window, RequestStat.request_kind, "requestKind", "other"
        )

    return {
        "totalRequests": total_requests,
        "totalErrors": int(total_errors),
        "avgDurationMs": int(int(total_dur) / total_requests),
        "tokenizedRequests": int(tokenized),
        "totalInputTokens": int(sum_in),
        "totalOutputTokens": int(sum_out),
        "totalEnergyWh": round(float(energy or 0), 2),
        "estimatedCostUsd": round(cost, 4),
        "byBackend": by_backend,
        "byRequestKind": by_kind,
    }


async def get_stats_by_api_key(
    from_dt: datetime | None = None,
    to_dt: datetime | None = None,
) -> dict:
    """Return request stats aggregated by API key name."""
    from_dt, to_dt = _normalize_window(from_dt, to_dt)
    from ocabra.db.auth import User

    window = [
        RequestStat.started_at >= from_dt,
        RequestStat.started_at <= to_dt,
        RequestStat.api_key_name.isnot(None),
    ]
    async with AsyncSessionLocal() as session:
        # One representative user per key (a key belongs to a single user); the
        # old code took the first row's non-null user_id, MIN is deterministic.
        agg_rows = (
            await session.execute(
                sa.select(
                    RequestStat.api_key_name,
                    sa.func.count(),
                    sa.func.count().filter(_is_error_sql()),
                    sa.func.coalesce(sa.func.sum(_dur()), 0),
                    sa.func.coalesce(sa.func.sum(_pos(RequestStat.input_tokens)), 0),
                    sa.func.coalesce(sa.func.sum(_pos(RequestStat.output_tokens)), 0),
                    sa.func.sum(RequestStat.energy_wh),
                    sa.func.min(RequestStat.user_id).filter(RequestStat.user_id.isnot(None)),
                )
                .where(*window)
                .group_by(RequestStat.api_key_name)
            )
        ).all()
        cost_by_key = await _cost_by_entity(session, RequestStat.api_key_name, window)

        user_ids = {r[7] for r in agg_rows if r[7]}
        username_map: dict = {}
        if user_ids:
            u_result = await session.execute(sa.select(User).where(User.id.in_(user_ids)))
            for u in u_result.scalars().all():
                username_map[u.id] = u.username

    summaries = []
    for key_name, n, errors, total_dur, sum_in, sum_out, energy, uid in sorted(
        agg_rows, key=lambda r: (-int(r[1]), str(r[0]))
    ):
        n = int(n)
        summaries.append(
            {
                "apiKeyName": key_name,
                "userId": str(uid) if uid else None,
                "username": username_map.get(uid, str(uid)) if uid else None,
                "totalRequests": n,
                "totalErrors": int(errors),
                "avgDurationMs": int(int(total_dur) / n) if n else 0,
                "totalInputTokens": int(sum_in),
                "totalOutputTokens": int(sum_out),
                "totalEnergyWh": round(float(energy or 0), 2),
                "estimatedCostUsd": round(cost_by_key.get(key_name, 0.0), 4),
            }
        )
    return {"byApiKey": summaries}


async def get_user_detail(
    user_id: str,
    from_dt: datetime | None = None,
    to_dt: datetime | None = None,
) -> dict:
    """Return detailed stats for a single user.

    `user_id` may be a UUID, a username, or an email address — they are
    resolved against the ``users`` table before querying ``request_stats``.
    """
    from_dt, to_dt = _normalize_window(from_dt, to_dt)
    import uuid as _uuid

    from ocabra.db.auth import User

    uid: _uuid.UUID | None = None
    try:
        uid = _uuid.UUID(str(user_id))
    except (ValueError, AttributeError):
        uid = None

    async with AsyncSessionLocal() as session:
        u_obj: User | None = None
        if uid is not None:
            u_obj = (
                await session.execute(sa.select(User).where(User.id == uid))
            ).scalar_one_or_none()
        else:
            # Resolve by username first, then email — both columns are unique.
            u_obj = (
                await session.execute(sa.select(User).where(User.username == user_id))
            ).scalar_one_or_none()
            if u_obj is None:
                u_obj = (
                    await session.execute(sa.select(User).where(User.email == user_id))
                ).scalar_one_or_none()
            if u_obj is not None:
                uid = u_obj.id

        if uid is None or u_obj is None:
            return {
                "userId": user_id,
                "username": user_id,
                "totalRequests": 0,
                "totalErrors": 0,
                "totalInputTokens": 0,
                "totalOutputTokens": 0,
                "totalEnergyWh": 0.0,
                "estimatedCostUsd": 0.0,
                "topModels": [],
                "byRequestKind": [],
                "tokenSeries": [],
            }

        username = u_obj.username

        window = [
            RequestStat.started_at >= from_dt,
            RequestStat.started_at <= to_dt,
            RequestStat.user_id == uid,
        ]
        n, errors, total_input, total_output, energy = (
            await session.execute(
                sa.select(
                    sa.func.count(),
                    sa.func.count().filter(_is_error_sql()),
                    sa.func.coalesce(sa.func.sum(_pos(RequestStat.input_tokens)), 0),
                    sa.func.coalesce(sa.func.sum(_pos(RequestStat.output_tokens)), 0),
                    sa.func.sum(RequestStat.energy_wh),
                ).where(*window)
            )
        ).one()
        n = int(n or 0)
        if n == 0:
            return {
                "userId": str(uid),
                "username": username,
                "totalRequests": 0,
                "totalErrors": 0,
                "totalInputTokens": 0,
                "totalOutputTokens": 0,
                "totalEnergyWh": 0.0,
                "estimatedCostUsd": 0.0,
                "topModels": [],
                "byRequestKind": [],
                "tokenSeries": [],
            }

        cost = await _cost_total(session, window)

        model_rows = (
            await session.execute(
                sa.select(
                    RequestStat.model_id,
                    sa.func.count().label("requests"),
                    sa.func.coalesce(sa.func.sum(_pos(RequestStat.input_tokens)), 0),
                    sa.func.coalesce(sa.func.sum(_pos(RequestStat.output_tokens)), 0),
                )
                .where(*window)
                .group_by(RequestStat.model_id)
                .order_by(sa.func.count().desc(), RequestStat.model_id.asc())
                .limit(10)
            )
        ).all()
        top_models = [
            {
                "modelId": mid,
                "requests": int(reqs),
                "inputTokens": int(sum_in),
                "outputTokens": int(sum_out),
            }
            for mid, reqs, sum_in, sum_out in model_rows
        ]

        kind_label = sa.func.coalesce(RequestStat.request_kind, "other")
        kind_rows = (
            await session.execute(
                sa.select(kind_label, sa.func.count()).where(*window).group_by(kind_label)
            )
        ).all()
        # Sort in Python (codepoint) like the old ``sorted(by_kind.items())``.
        by_request_kind = [
            {"requestKind": k, "count": int(c)} for k, c in sorted(kind_rows, key=lambda r: r[0])
        ]

        minute = sa.func.date_trunc("minute", RequestStat.started_at)
        series_rows = (
            await session.execute(
                sa.select(
                    minute.label("t"),
                    sa.func.coalesce(sa.func.sum(_pos(RequestStat.input_tokens)), 0),
                    sa.func.coalesce(sa.func.sum(_pos(RequestStat.output_tokens)), 0),
                )
                .where(*window)
                .group_by(minute)
                .order_by(minute)
            )
        ).all()
        token_series = [
            {"timestamp": ts.isoformat(), "inputTokens": int(i), "outputTokens": int(o)}
            for ts, i, o in series_rows
        ]

    return {
        "userId": str(uid),
        "username": username,
        "totalRequests": n,
        "totalErrors": int(errors),
        "totalInputTokens": int(total_input),
        "totalOutputTokens": int(total_output),
        "totalEnergyWh": round(float(energy or 0), 2),
        "estimatedCostUsd": round(cost, 4),
        "topModels": top_models,
        "byRequestKind": by_request_kind,
        "tokenSeries": token_series,
    }


async def get_federation_stats(
    from_dt: datetime | None = None,
    to_dt: datetime | None = None,
) -> dict:
    """Federated traffic in a window, both ways, aggregated in SQL.

    * ``outbound``: requests we sent to each node — count, errors, tokens,
      p50/p95 duration, top models and, for cloud nodes, what they cost in the
      window (``costUsd``, per request and per 1k tokens).
    * ``inbound``: requests other nodes sent us (served here).
    * ``routes``: why requests went where they did (decision reasons, forced
      nodes, ``fallback_local`` when a chosen peer failed).
    * ``byNode``: the old shape (id, name, count, average), kept for clients.

    Nodes are keyed by name when known (kept after a peer is deleted), else id.
    """
    from ocabra.core.cloud.manager import node_cost
    from ocabra.db.cloud import CloudNode

    from_dt, to_dt = _normalize_window(from_dt, to_dt)
    R = RequestStat
    window = sa.and_(R.started_at >= from_dt, R.started_at <= to_dt)
    is_error = sa.or_(R.status_code >= 400, R.error.is_not(None))

    def _summary_cols():
        return (
            sa.func.count().label("n"),
            sa.func.count().filter(is_error).label("errors"),
            sa.func.coalesce(sa.func.sum(R.input_tokens), 0).label("tin"),
            sa.func.coalesce(sa.func.sum(R.output_tokens), 0).label("tout"),
            sa.func.avg(R.duration_ms).label("avg"),
            sa.func.percentile_cont(0.5).within_group(R.duration_ms).label("p50"),
            sa.func.percentile_cont(0.95).within_group(R.duration_ms).label("p95"),
        )

    out_key = sa.func.coalesce(R.remote_node_name, R.remote_node_id)
    async with AsyncSessionLocal() as session:
        totals = (
            await session.execute(
                sa.select(
                    sa.func.count().filter(R.remote_node_id.is_(None)),
                    sa.func.count().filter(R.remote_node_id.is_not(None)),
                    sa.func.count().filter(R.federation_inbound_from.is_not(None)),
                    sa.func.count().filter(R.federation_route == "fallback_local"),
                ).where(window)
            )
        ).one()
        outbound_rows = (
            await session.execute(
                sa.select(out_key.label("node"), sa.func.max(R.remote_node_id), *_summary_cols())
                .where(window, R.remote_node_id.is_not(None))
                .group_by(out_key)
            )
        ).all()
        inbound_rows = (
            await session.execute(
                sa.select(R.federation_inbound_from.label("node"), *_summary_cols())
                .where(window, R.federation_inbound_from.is_not(None))
                .group_by(R.federation_inbound_from)
            )
        ).all()
        model_rows = (
            await session.execute(
                sa.select(
                    sa.case(
                        (R.remote_node_id.is_not(None), sa.literal("out")), else_=sa.literal("in")
                    ).label("dir"),
                    sa.func.coalesce(out_key, R.federation_inbound_from).label("node"),
                    R.model_id,
                    sa.func.count().label("n"),
                )
                .where(
                    window,
                    sa.or_(R.remote_node_id.is_not(None), R.federation_inbound_from.is_not(None)),
                )
                .group_by("dir", "node", R.model_id)
            )
        ).all()
        route_rows = (
            await session.execute(
                sa.select(R.federation_route, sa.func.count())
                .where(window, R.federation_route.is_not(None))
                .group_by(R.federation_route)
            )
        ).all()
        fallback_rows = (
            await session.execute(
                sa.select(R.model_id, sa.func.count())
                .where(window, R.federation_route == "fallback_local")
                .group_by(R.model_id)
            )
        ).all()
        cloud_rows = (
            (
                await session.execute(
                    sa.select(CloudNode).where(
                        sa.func.coalesce(CloudNode.terminated_at, to_dt) >= from_dt,
                        sa.func.coalesce(CloudNode.started_at, CloudNode.created_at) <= to_dt,
                    )
                )
            )
            .scalars()
            .all()
        )

    models: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for direction, node, model_id, n in model_rows:
        models[(direction, str(node))].append({"model": model_id, "count": int(n)})
    for lst in models.values():
        lst.sort(key=lambda m: -m["count"])
        del lst[5:]

    # By peer id while the peer exists, and by name after: deleting a pod's
    # peer (it terminated) nulls ``cloud_nodes.peer_id``, and the pod's peer is
    # named after the pod.
    cost_by_peer: dict[str, float] = defaultdict(float)
    cost_by_name: dict[str, float] = defaultdict(float)
    for row in cloud_rows:
        cost = node_cost(row, since=from_dt, now=min(to_dt, datetime.now(UTC)))
        if row.peer_id is not None:
            cost_by_peer[str(row.peer_id)] += cost
        cost_by_name[row.name] += cost

    def _row(r, direction: str, node_id: str | None = None) -> dict:
        tokens = int(r.tin or 0) + int(r.tout or 0)
        out = {
            "node": str(r.node),
            "count": int(r.n),
            "errors": int(r.errors),
            "inputTokens": int(r.tin or 0),
            "outputTokens": int(r.tout or 0),
            "avgDurationMs": int(r.avg or 0),
            "p50DurationMs": int(r.p50 or 0),
            "p95DurationMs": int(r.p95 or 0),
            "topModels": models.get((direction, str(r.node)), []),
        }
        if direction == "out":
            cost = (cost_by_peer.get(str(node_id)) if node_id else None) or cost_by_name.get(
                str(r.node)
            )
            out["nodeId"] = node_id
            out["costUsd"] = round(cost, 4) if cost else None
            out["usdPerRequest"] = round(cost / r.n, 5) if cost and r.n else None
            out["usdPer1kTokens"] = round(cost * 1000 / tokens, 5) if cost and tokens else None
        return out

    outbound = sorted((_row(r, "out", r[1]) for r in outbound_rows), key=lambda x: -x["count"])
    inbound = sorted((_row(r, "in") for r in inbound_rows), key=lambda x: -x["count"])
    return {
        "localCount": int(totals[0]),
        "remoteCount": int(totals[1]),
        "inboundCount": int(totals[2]),
        "fallbackCount": int(totals[3]),
        "outbound": outbound,
        "inbound": inbound,
        "routes": sorted(
            ({"route": route, "count": int(n)} for route, n in route_rows),
            key=lambda x: -x["count"],
        ),
        "fallbacksByModel": sorted(
            ({"model": m, "count": int(n)} for m, n in fallback_rows),
            key=lambda x: -x["count"],
        ),
        "byNode": [
            {
                "nodeId": o["nodeId"],
                "nodeName": o["node"],
                "count": o["count"],
                "avgDurationMs": o["avgDurationMs"],
            }
            for o in outbound
        ],
    }
