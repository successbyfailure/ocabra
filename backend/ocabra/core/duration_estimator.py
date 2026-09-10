"""Duration estimator — predicts how long an inference request takes.

Bloque 20 baseline (Nivel 1).

The estimator answers two consumer questions:

    estimate(model_id, body) -> Estimate    # "how long will this take?"
    in_flight_remaining(worker) -> float?    # "how much left of what's running?"

Consumers today:
    * ``core.router_resolver`` — decides between primary and fallback.
    * ``core.model_manager`` (pressure_eviction) — dynamic drain grace.
    * ``api.internal.estimate`` — /ocabra/estimate for clients + Playground.

Design pillars:

    * **Honesty over precision.** Every estimate carries a ``confidence`` in
      [0, 1]. Consumers that need to be safe (drain grace, router) respect the
      floor from ``settings.router_confidence_floor``; when confidence dips
      below it, they should treat the answer as "unknown" and fall back to
      conservative defaults (p95, family_default, refuse to redirect).
    * **Warmup without panic.** The very first requests to a new model come
      out of ``FAMILY_DEFAULT`` with confidence≈0 — a wide range so no
      consumer risks a wrong bet. As samples accrue, source graduates from
      ``family_default`` → ``percentile`` → ``regression`` (added in Etapa 2)
      and confidence climbs.
    * **Cheap per call.** All hot-path lookups hit the in-memory cache with
      TTL ``estimator_cache_ttl_s``; the calibration loop is what runs SQL
      and only every ``estimator_refresh_interval_s``.
    * **Sanity clamp.** Nothing returned to a consumer exceeds
      ``settings.estimator_max_estimate_s`` — protects against runaway
      regression on tiny/dirty data.

The Nivel 1 baseline this file ships uses raw percentiles per ``model_id`` on
the last 7 days of ``request_stats``. It doesn't yet split by input size —
that's Etapa 2 (linear regression) and Etapa 5 (family-specific formulas).
Even so it's already useful for the router and drain grace: they mostly need
"about how long?" not "give me the exact 34.7 seconds".
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import sqlalchemy as sa
import structlog

from ocabra.config import settings
from ocabra.database import AsyncSessionLocal
from ocabra.db.stats import ModelLoadStat, RequestStat

logger = structlog.get_logger(__name__)


# Broad time-cost brackets by backend/kind used when we have no percentile
# data for the model at all. Values are pessimistic (upper bound of what's
# normal) so first-request consumers err on the side of "assume it'll take a
# while" instead of "assume it's instant".
#
# Numbers came from the 30-day stats survey on 2026-09-10 — they should be
# refreshed occasionally but don't need to be exact. They exist as an
# always-available floor while real calibration warms up.
FAMILY_DEFAULT_S: dict[str, dict[str, float]] = {
    "vllm":         {"p50":  8.0, "p95":  90.0},
    "sglang":       {"p50":  8.0, "p95":  90.0},
    "tensorrt_llm": {"p50":  6.0, "p95":  60.0},
    "llama_cpp":    {"p50": 15.0, "p95": 250.0},
    "bitnet":       {"p50": 10.0, "p95": 120.0},
    "ollama":       {"p50": 14.0, "p95": 250.0},
    "whisper":      {"p50":  5.0, "p95":  60.0},
    "vibeasr":      {"p50":  3.0, "p95":  30.0},
    "tts":          {"p50":  2.0, "p95":  10.0},
    "voxtral":      {"p50":  4.0, "p95":  20.0},
    "chatterbox":   {"p50":  2.0, "p95":  10.0},
    "diffusers":    {"p50": 15.0, "p95":  60.0},
    "mage":         {"p50": 15.0, "p95":  60.0},
    "acestep":      {"p50": 240.0, "p95": 900.0},
    "upscaler":     {"p50":  5.0, "p95":  30.0},
    "flashvsr":     {"p50": 300.0, "p95": 2700.0},
    # Realtime sessions are intentionally not here — they're not estimable
    # and consumers must go through SessionRegistry instead.
}


@dataclass
class Estimate:
    """The estimator's answer for a specific (model, work_size) pair.

    Fields are in seconds unless noted. All consumers should look at
    ``confidence`` before trusting ``expected_s`` for irreversible decisions
    (e.g. redirecting a client) — below the ``router_confidence_floor`` the
    number is a rough hint, not a promise.
    """

    expected_s: float
    p50_s: float
    p95_s: float
    confidence: float          # 0..1
    source: str                # "regression" | "percentile" | "family_default"
    sample_count: int
    cold_start_s: float = 0.0  # added when the target model is currently UNLOADED

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "expected_seconds": self.expected_s,
            "p50_seconds": self.p50_s,
            "p95_seconds": self.p95_s,
            "confidence": self.confidence,
            "source": self.source,
            "sample_count": self.sample_count,
            "cold_start_seconds": self.cold_start_s,
        }


@dataclass
class _ChatRegression:
    """Fitted ``duration_ms ≈ a·input_tokens + b·output_tokens + c`` model.

    Only populated for chat/completion families where we have both
    ``input_tokens`` and ``output_tokens`` on enough historical rows to fit
    a line. When absent, callers fall back to the raw percentiles.
    """

    a_in: float   # ms per input token
    b_out: float  # ms per output token
    c: float      # fixed overhead in ms
    residual_std_s: float  # 1-sigma residual in seconds, drives confidence
    p95_s: float           # unconditional p95 fallback (safety net)


# Chat-shaped backends whose durations are well described by
# a·input_tokens + b·output_tokens + c. Everything else stays on the
# percentile path in Etapa 1 / gets its own family estimator in Etapa 5.
_CHAT_FAMILIES = frozenset(
    {"vllm", "sglang", "tensorrt_llm", "llama_cpp", "bitnet", "ollama"}
)


@dataclass
class _ModelCalibration:
    """Per-model cached calibration filled by the refresh loop."""

    model_id: str
    backend_type: str | None
    p50_s: float
    p95_s: float
    sample_count: int
    residual_ratio: float       # relative spread, drives percentile confidence
    cold_start_p95_s: float     # from model_load_stats, 0 if unknown
    chat_regression: _ChatRegression | None = None
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def confidence(self) -> float:
        """Confidence in this calibration.

        Two factors:
          * sample_count: sigmoid centred at ~15 samples so a very fresh
            model sits below the router floor.
          * residual_ratio: (p95 - p50) / max(p50, 1) — how spread-out the
            durations are; wider = less predictable.

        The two are multiplied so tiny-sample OR wildly-varied models both
        end up with low confidence. Clamped to [0, 1] for the consumers'
        threshold comparisons.
        """
        n_factor = 1.0 / (1.0 + math.exp(-(self.sample_count - 15) / 8.0))
        spread = max(0.0, self.residual_ratio)
        # ``residual_ratio`` of 0 (constant) -> 1.0; of 1.0 (p95 twice p50) ->
        # ~0.5; of 3+ -> ~0.
        spread_factor = 1.0 / (1.0 + spread)
        return max(0.0, min(1.0, n_factor * spread_factor))


_MIN_REGRESSION_SAMPLES = 30
# If the residual standard deviation is larger than the mean prediction,
# the "line" is basically noise — no better than the raw percentile with
# its own confidence. 1.0 says "one sigma of noise is comparable to the
# average signal we're trying to predict"; anything above that has no
# business being reported as source=regression.
_MAX_RESIDUAL_RATIO_FOR_FIT = 1.0


def _fit_chat_regression(
    points: list[tuple[float, float, float]],
) -> _ChatRegression | None:
    """Fit ``duration_ms ≈ a·input + b·output + c`` by ordinary least
    squares with the closed-form normal equations. Returns ``None`` when
    the fit is unreliable — too few points, singular design matrix, or the
    residual is so wild that the raw percentile is a better tool.

    We stay inside stdlib to avoid a numpy dependency in the hot loop —
    the design matrix is 3×3 so a hand-rolled solve costs nothing.
    """
    if len(points) < _MIN_REGRESSION_SAMPLES:
        return None

    # Winsorize durations at 1%/99% to keep a single 900s hanger from
    # dominating the fit.
    durs = sorted(p[2] for p in points)
    lo = durs[max(0, int(len(durs) * 0.01))]
    hi = durs[min(len(durs) - 1, int(len(durs) * 0.99))]

    # Build the sums of the normal equations for the 3×3 system:
    #   [Σx²   Σxy   Σx ] [a]   [Σxd]
    #   [Σxy   Σy²   Σy ] [b] = [Σyd]
    #   [Σx    Σy    n  ] [c]   [Σd ]
    n = 0
    sx = sy = sd = 0.0
    sxx = syy = sxy = 0.0
    sxd = syd = 0.0
    for x, y, d in points:
        if d < lo or d > hi:
            continue
        n += 1
        sx += x; sy += y; sd += d
        sxx += x * x; syy += y * y; sxy += x * y
        sxd += x * d; syd += y * d
    if n < _MIN_REGRESSION_SAMPLES:
        return None

    # Solve by Cramer's rule (3×3 — small determinants, no numpy needed).
    def _det3(m: list[list[float]]) -> float:
        return (
            m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
        )

    A = [[sxx, sxy, sx], [sxy, syy, sy], [sx, sy, float(n)]]
    rhs = [sxd, syd, sd]
    det = _det3(A)
    if abs(det) < 1e-6:
        return None  # near-singular — inputs all clustered

    def _with_col(col: int) -> list[list[float]]:
        return [
            [rhs[i] if j == col else A[i][j] for j in range(3)]
            for i in range(3)
        ]

    a = _det3(_with_col(0)) / det
    b = _det3(_with_col(1)) / det
    c = _det3(_with_col(2)) / det

    # Residual std of prediction (ms).
    sse = 0.0
    used = 0
    mean_pred_ms = 0.0
    for x, y, d in points:
        if d < lo or d > hi:
            continue
        pred = a * x + b * y + c
        sse += (d - pred) ** 2
        mean_pred_ms += pred
        used += 1
    if used <= 3:
        return None
    residual_std_ms = math.sqrt(sse / max(1, used - 3))
    mean_pred_ms /= used
    if mean_pred_ms <= 0:
        return None
    # If the residual is larger than a couple of prediction means, the fit
    # is basically noise — the raw percentile with its own confidence is a
    # more useful tool for the router.
    if residual_std_ms / max(mean_pred_ms, 1.0) > _MAX_RESIDUAL_RATIO_FOR_FIT:
        return None

    # Also derive an unconditional p95 in seconds as a safety net when the
    # regression prediction ends up negative on adversarial inputs.
    all_durs_s = sorted(d / 1000.0 for _, _, d in points if lo <= d <= hi)
    p95_s = all_durs_s[min(len(all_durs_s) - 1, int(len(all_durs_s) * 0.95))]

    return _ChatRegression(
        a_in=a,
        b_out=b,
        c=c,
        residual_std_s=residual_std_ms / 1000.0,
        p95_s=p95_s,
    )


class DurationEstimator:
    """Baseline percentile-based estimator (Nivel 1)."""

    _CALIBRATION_LOOKBACK = timedelta(days=7)
    _COLD_START_LOOKBACK = timedelta(days=14)

    def __init__(self) -> None:
        # Per-model calibrations from request_stats. Empty until the refresh
        # loop runs; get_state() serves FAMILY_DEFAULT until then.
        self._calibrations: dict[str, _ModelCalibration] = {}
        # In-flight registry: worker_key -> started_at_monotonic
        # Populated by the model_manager / worker_pool via ``mark_start`` and
        # ``mark_end``. Reads are lock-free — a stale read means at most a
        # slightly wrong "remaining" number.
        self._in_flight: dict[str, float] = {}
        self._refresh_lock = asyncio.Lock()
        self._task: asyncio.Task | None = None

    # ── Public API ────────────────────────────────────────────

    async def estimate(
        self,
        model_id: str,
        *,
        backend_type: str | None = None,
        currently_loaded: bool = True,
        input_tokens: int | None = None,
        max_tokens: int | None = None,
    ) -> Estimate:
        """Return an estimate for ``model_id``.

        ``backend_type`` is used to pick the ``FAMILY_DEFAULT`` when there's
        no per-model calibration yet. It's optional because the calibration
        table already stores it after the first refresh, but the very first
        estimate for a brand new model needs the hint.

        ``currently_loaded=False`` adds the cold-start estimate to
        ``expected_s`` — consumers use this to compare fair "time to first
        token" between a warm primary and a cold fallback.

        ``input_tokens`` and ``max_tokens`` enable the regression path for
        chat-shaped families: when the model has a fitted
        ``a·input + b·output + c`` calibration and the caller provides at
        least ``input_tokens``, the returned estimate uses the regression
        prediction with much higher confidence than the raw percentile.
        """
        cal = self._calibrations.get(model_id)
        cold_start = 0.0
        if cal is not None and not currently_loaded:
            cold_start = cal.cold_start_p95_s

        if cal is None or cal.sample_count == 0:
            # Warmup path — nothing in request_stats yet.
            fam = (cal.backend_type if cal else backend_type) or "unknown"
            defaults = FAMILY_DEFAULT_S.get(fam) or {"p50": 30.0, "p95": 300.0}
            p50 = defaults["p50"]
            p95 = defaults["p95"]
            expected = min(settings.estimator_max_estimate_s, p50 + cold_start)
            return Estimate(
                expected_s=expected,
                p50_s=p50,
                p95_s=min(settings.estimator_max_estimate_s, p95 + cold_start),
                confidence=0.0,
                source="family_default",
                sample_count=0,
                cold_start_s=cold_start,
            )

        # Regression path: available only when the model has a fitted
        # chat_regression AND the caller passed a concrete input size. The
        # per-request prediction is much sharper than the raw p50 because
        # it splits the huge duration variance (chat 500-tok vs chat 200k-tok
        # on the same model) into two separate axes.
        if (
            cal.chat_regression is not None
            and input_tokens is not None
            and input_tokens >= 0
        ):
            reg = cal.chat_regression
            # Assume the client uses the full max_tokens budget by default —
            # gives a pessimistic but useful upper bound. If max_tokens isn't
            # given, fall back to the historical average output size, proxied
            # by (p50 - c - a·input) / b.
            if max_tokens is not None and max_tokens > 0:
                out_tokens = float(max_tokens)
            else:
                out_tokens = max(
                    1.0,
                    (cal.p50_s * 1000.0 - reg.c - reg.a_in * input_tokens) / max(reg.b_out, 0.001),
                )
            pred_ms = reg.a_in * input_tokens + reg.b_out * out_tokens + reg.c
            pred_s = max(0.0, pred_ms / 1000.0)
            # p95 = prediction + 1.645·residual_std (one-sided normal
            # approximation — good enough for a bound, not for a distribution
            # tail).
            p95 = max(pred_s + 1.645 * reg.residual_std_s, cal.p50_s)
            expected = min(settings.estimator_max_estimate_s, pred_s + cold_start)
            return Estimate(
                expected_s=expected,
                p50_s=pred_s,
                p95_s=min(settings.estimator_max_estimate_s, p95 + cold_start),
                # Regression confidence is the base cal.confidence() scaled up
                # by how tight the residuals are relative to the prediction.
                # Tight residuals + enough samples ≈ 1.0.
                confidence=min(
                    1.0,
                    cal.confidence() * (
                        1.0 / (1.0 + reg.residual_std_s / max(pred_s, 1.0))
                    ) * 2.0,
                ),
                source="regression",
                sample_count=cal.sample_count,
                cold_start_s=cold_start,
            )

        p50 = cal.p50_s
        p95 = cal.p95_s
        expected = min(settings.estimator_max_estimate_s, p50 + cold_start)
        return Estimate(
            expected_s=expected,
            p50_s=p50,
            p95_s=min(settings.estimator_max_estimate_s, p95 + cold_start),
            confidence=cal.confidence(),
            source="percentile",
            sample_count=cal.sample_count,
            cold_start_s=cold_start,
        )

    def in_flight_remaining(self, worker_key: str, *, model_id: str | None = None) -> float | None:
        """Estimate how many seconds are left on the in-flight request for
        ``worker_key``, or ``None`` if nothing is in flight.

        Uses ``estimate.p95_s - elapsed`` clamped to [0, p95]. Consumers
        (drain grace) should add their own margin.
        """
        started = self._in_flight.get(worker_key)
        if started is None:
            return None
        elapsed = max(0.0, time.monotonic() - started)
        cal = self._calibrations.get(model_id or worker_key)
        if cal is None or cal.sample_count == 0:
            # No calibration → conservative default: assume worst-case for
            # the whole family.
            fam = (cal.backend_type if cal else None) or "unknown"
            defaults = FAMILY_DEFAULT_S.get(fam) or {"p95": 300.0}
            budget = defaults["p95"]
        else:
            budget = cal.p95_s
        remaining = max(0.0, budget - elapsed)
        return min(settings.estimator_max_estimate_s, remaining)

    def mark_start(self, worker_key: str) -> None:
        self._in_flight[worker_key] = time.monotonic()

    def mark_end(self, worker_key: str) -> None:
        self._in_flight.pop(worker_key, None)

    # ── Calibration loop ─────────────────────────────────────

    async def refresh(self) -> int:
        """Recompute per-model calibrations from request_stats.

        Returns the number of models covered. Safe to call concurrently —
        an internal lock serialises actual work.
        """
        async with self._refresh_lock:
            cutoff = datetime.now(UTC) - self._CALIBRATION_LOOKBACK
            load_cutoff = datetime.now(UTC) - self._COLD_START_LOOKBACK
            async with AsyncSessionLocal() as session:
                # Percentiles per model over the last 7 days of SUCCESSFUL
                # requests. Errors would poison the average with tiny 400ms
                # 404s or timeouts at ``inference_request_timeout_seconds``.
                rows = (await session.execute(
                    sa.text(
                        """
                        SELECT
                            model_id,
                            backend_type,
                            COUNT(*)                                                                  AS n,
                            PERCENTILE_DISC(0.5)  WITHIN GROUP (ORDER BY duration_ms)::float / 1000.0 AS p50_s,
                            PERCENTILE_DISC(0.95) WITHIN GROUP (ORDER BY duration_ms)::float / 1000.0 AS p95_s
                        FROM request_stats
                        WHERE started_at > :cutoff
                          AND status_code < 400
                          AND error IS NULL
                          AND duration_ms IS NOT NULL
                        GROUP BY model_id, backend_type
                        """
                    ),
                    {"cutoff": cutoff},
                )).all()

                # Cold-start p95 comes from model_load_stats — a much smaller
                # table so we can afford the same window and a fresh query.
                loads = (await session.execute(
                    sa.text(
                        """
                        SELECT
                            model_id,
                            PERCENTILE_DISC(0.95) WITHIN GROUP (ORDER BY duration_ms)::float / 1000.0 AS p95_s
                        FROM model_load_stats
                        WHERE started_at > :cutoff
                          AND duration_ms IS NOT NULL
                        GROUP BY model_id
                        """
                    ),
                    {"cutoff": load_cutoff},
                )).all()

            cold_by_model: dict[str, float] = {row.model_id: float(row.p95_s or 0.0) for row in loads}

            # Regression path — only chat-shaped families and only for
            # models with enough rows carrying BOTH input_tokens and
            # output_tokens. Kept as a separate query so we don't drag the
            # token columns through the percentile aggregation.
            chat_families_tuple = tuple(_CHAT_FAMILIES)
            fit_by_model: dict[str, _ChatRegression] = {}
            async with AsyncSessionLocal() as session:
                fit_rows = (await session.execute(
                    sa.text(
                        """
                        SELECT model_id,
                               input_tokens,
                               output_tokens,
                               duration_ms::float AS duration_ms
                        FROM request_stats
                        WHERE started_at > :cutoff
                          AND status_code < 400
                          AND error IS NULL
                          AND duration_ms IS NOT NULL
                          AND input_tokens IS NOT NULL
                          AND output_tokens IS NOT NULL
                          AND backend_type = ANY(:families)
                        """
                    ),
                    {"cutoff": cutoff, "families": list(chat_families_tuple)},
                )).all()

            # Group by model_id in Python (a lot cheaper than shipping
            # OLS/ridge to Postgres, and the volumes are modest — thousands
            # of rows per model at most).
            by_model: dict[str, list[tuple[float, float, float]]] = {}
            for row in fit_rows:
                by_model.setdefault(row.model_id, []).append(
                    (float(row.input_tokens), float(row.output_tokens), float(row.duration_ms))
                )
            for model_id, points in by_model.items():
                fit = _fit_chat_regression(points)
                if fit is not None:
                    fit_by_model[model_id] = fit

            new_map: dict[str, _ModelCalibration] = {}
            for row in rows:
                p50 = float(row.p50_s or 0.0)
                p95 = float(row.p95_s or 0.0)
                # Relative spread — protected against p50=0.
                residual = (p95 - p50) / max(p50, 0.5)
                new_map[row.model_id] = _ModelCalibration(
                    model_id=row.model_id,
                    backend_type=row.backend_type,
                    p50_s=p50,
                    p95_s=p95,
                    sample_count=int(row.n),
                    residual_ratio=residual,
                    cold_start_p95_s=cold_by_model.get(row.model_id, 0.0),
                    chat_regression=fit_by_model.get(row.model_id),
                )

            # Atomic swap — a concurrent reader sees either the old map or
            # the new one in full, never a torn state.
            self._calibrations = new_map
            logger.info(
                "duration_estimator_refreshed",
                models=len(new_map),
                with_regression=sum(1 for c in new_map.values() if c.chat_regression),
            )
            return len(new_map)

    async def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._refresh_loop(), name="duration-estimator-refresh")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except (asyncio.CancelledError, Exception):
            pass
        self._task = None

    async def _refresh_loop(self) -> None:
        # Kick off an initial refresh so consumers don't spend the first
        # 30 minutes serving FAMILY_DEFAULT.
        try:
            await self.refresh()
        except Exception as exc:  # noqa: BLE001 — cold DB, unusual but survivable
            logger.warning("duration_estimator_initial_refresh_failed", error=str(exc))
        interval = max(60, int(settings.estimator_refresh_interval_s))
        while True:
            try:
                await asyncio.sleep(interval)
                await self.refresh()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — never break the loop
                logger.warning("duration_estimator_refresh_failed", error=str(exc))
