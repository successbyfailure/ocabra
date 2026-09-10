"""Unit tests for the DurationEstimator baseline (Nivel 1).

Bloque 20, Etapa 1.

Covers:
    * FAMILY_DEFAULT path when there's no calibration yet — must return
      confidence=0 so consumers know not to bet on it.
    * Percentile path when calibration is populated — confidence rises with
      sample count and falls with residual spread.
    * cold_start_seconds addition when the model is currently UNLOADED.
    * ``estimator_max_estimate_s`` clamp for pathological calibrations.
    * ``in_flight_remaining`` gives sensible values (elapsed-aware, clamped).
    * mark_start / mark_end life-cycle.

The calibration refresh loop hits Postgres; that path is exercised end-to-end
by ``test_routing_e2e.py`` in Etapa 6. Here we bypass it by injecting
calibrations directly into ``_calibrations`` — enough to exercise the
consumer contract in isolation.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ocabra.core.duration_estimator import (
    FAMILY_DEFAULT_S,
    DurationEstimator,
    _ChatRegression,
    _fit_chat_regression,
    _ModelCalibration,
)


def _put(estimator: DurationEstimator, model_id: str, **kwargs) -> None:
    """Helper: inject a calibration bypassing the SQL refresh."""
    estimator._calibrations[model_id] = _ModelCalibration(
        model_id=model_id,
        backend_type=kwargs.pop("backend_type", "vllm"),
        p50_s=kwargs.pop("p50_s", 5.0),
        p95_s=kwargs.pop("p95_s", 20.0),
        sample_count=kwargs.pop("sample_count", 200),
        residual_ratio=kwargs.pop("residual_ratio", 0.5),
        cold_start_p95_s=kwargs.pop("cold_start_p95_s", 0.0),
        updated_at=kwargs.pop("updated_at", datetime.now(UTC)),
    )


class TestFamilyDefaultFallback:
    @pytest.mark.asyncio
    async def test_unknown_model_returns_family_default(self):
        estimator = DurationEstimator()
        result = await estimator.estimate("vllm/something-new", backend_type="vllm")
        assert result.source == "family_default"
        assert result.confidence == 0.0
        assert result.p50_s == FAMILY_DEFAULT_S["vllm"]["p50"]
        assert result.p95_s == FAMILY_DEFAULT_S["vllm"]["p95"]

    @pytest.mark.asyncio
    async def test_unknown_family_returns_generic_default(self):
        """A family we don't have in FAMILY_DEFAULT (typo, new backend
        added after this file was written) still returns something usable
        — not a crash."""
        estimator = DurationEstimator()
        result = await estimator.estimate("bogus/x", backend_type="martian_backend")
        assert result.source == "family_default"
        assert result.expected_s > 0

    @pytest.mark.asyncio
    async def test_family_default_respects_max_clamp(self, monkeypatch):
        """Even if some future family default is stupid-large, ``expected_s``
        never returns more than ``estimator_max_estimate_s``. Guarantees a
        consumer never gets a nonsense number from us."""
        from ocabra.config import settings

        monkeypatch.setattr(settings, "estimator_max_estimate_s", 60)
        estimator = DurationEstimator()
        result = await estimator.estimate("some/flashvsr", backend_type="flashvsr")
        assert result.expected_s <= 60
        assert result.p95_s <= 60


class TestPercentilePath:
    @pytest.mark.asyncio
    async def test_populated_calibration_beats_family_default(self):
        estimator = DurationEstimator()
        _put(estimator, "vllm/prod-model", p50_s=8.0, p95_s=30.0, sample_count=500)
        result = await estimator.estimate("vllm/prod-model", backend_type="vllm")
        assert result.source == "percentile"
        assert result.p50_s == 8.0
        assert result.p95_s == 30.0
        assert result.sample_count == 500
        # Well above the confidence floor (0.3) with 500 samples + moderate spread.
        assert result.confidence > 0.5

    @pytest.mark.asyncio
    async def test_low_sample_count_has_low_confidence(self):
        """A model with only 3 requests can't be trusted yet. The confidence
        must drop below the router's floor so consumers fall back."""
        estimator = DurationEstimator()
        _put(estimator, "vllm/new-model", sample_count=3, residual_ratio=0.2)
        result = await estimator.estimate("vllm/new-model")
        assert result.source == "percentile"
        assert result.confidence < 0.3

    @pytest.mark.asyncio
    async def test_high_variance_lowers_confidence(self):
        """Two models with identical sample counts but different spreads
        must NOT have the same confidence. A p95 that's 5x the p50 is much
        less predictable than one that's 1.2x."""
        estimator = DurationEstimator()
        _put(estimator, "vllm/tight", sample_count=100, residual_ratio=0.2)
        _put(estimator, "vllm/wild", sample_count=100, residual_ratio=3.0)
        tight = await estimator.estimate("vllm/tight")
        wild = await estimator.estimate("vllm/wild")
        assert tight.confidence > wild.confidence


class TestColdStart:
    @pytest.mark.asyncio
    async def test_currently_loaded_ignores_cold_start(self):
        estimator = DurationEstimator()
        _put(estimator, "vllm/x", p50_s=5, p95_s=20, cold_start_p95_s=42)
        result = await estimator.estimate("vllm/x", currently_loaded=True)
        assert result.cold_start_s == 0
        assert result.expected_s == 5.0

    @pytest.mark.asyncio
    async def test_currently_unloaded_adds_cold_start(self):
        """Fair time-to-first-token: an idle model has to pay its cold-start
        before generating anything. Router uses ``expected_s`` to compare a
        warm primary vs a cold fallback — this is the number that matters."""
        estimator = DurationEstimator()
        _put(estimator, "vllm/x", p50_s=5, p95_s=20, cold_start_p95_s=42)
        result = await estimator.estimate("vllm/x", currently_loaded=False)
        assert result.cold_start_s == 42
        assert result.expected_s == 47.0
        # p95 is also inflated so drain-grace factors the cold-start.
        assert result.p95_s == 62.0


class TestInFlightRemaining:
    def test_no_in_flight_returns_none(self):
        estimator = DurationEstimator()
        assert estimator.in_flight_remaining("nothing") is None

    def test_mark_start_then_end_removes_entry(self):
        estimator = DurationEstimator()
        estimator.mark_start("worker-a")
        assert estimator.in_flight_remaining("worker-a") is not None
        estimator.mark_end("worker-a")
        assert estimator.in_flight_remaining("worker-a") is None

    def test_remaining_uses_p95_budget_minus_elapsed(self, monkeypatch):
        """The remaining budget is (p95 - elapsed) clamped to [0, p95]. Uses
        monotonic time so tests can inject a start moment."""
        import time

        estimator = DurationEstimator()
        _put(estimator, "worker-a", p50_s=5.0, p95_s=30.0, sample_count=100)
        fake_now = [1000.0]
        monkeypatch.setattr(time, "monotonic", lambda: fake_now[0])
        estimator.mark_start("worker-a")
        # 10 seconds later: 30 - 10 = 20 remaining.
        fake_now[0] = 1010.0
        remaining = estimator.in_flight_remaining("worker-a", model_id="worker-a")
        assert 19.9 < (remaining or 0) < 20.1

    def test_remaining_clamped_to_zero_when_past_budget(self, monkeypatch):
        """Live requests routinely exceed p95 — the estimator says '0 left'
        instead of negative to avoid consumers doing weird math."""
        import time

        estimator = DurationEstimator()
        _put(estimator, "worker-a", p95_s=10.0, sample_count=100)
        fake_now = [500.0]
        monkeypatch.setattr(time, "monotonic", lambda: fake_now[0])
        estimator.mark_start("worker-a")
        fake_now[0] = 600.0  # 100s elapsed, way past 10s p95
        remaining = estimator.in_flight_remaining("worker-a", model_id="worker-a")
        assert remaining == 0.0


# ── Etapa 2: chat/completion regression ──────────────────────


def _synthetic_chat_points(
    a: float, b: float, c: float, n: int = 200, noise_ms: float = 500.0
) -> list[tuple[float, float, float]]:
    """Generate a synthetic dataset that follows ``duration_ms ≈ a·in + b·out + c``
    with Gaussian noise, so the fitter has a known ground truth to recover."""
    import random

    random.seed(42)
    points: list[tuple[float, float, float]] = []
    for _ in range(n):
        in_tokens = float(random.randint(50, 20_000))
        out_tokens = float(random.randint(50, 4_000))
        dur = a * in_tokens + b * out_tokens + c + random.gauss(0, noise_ms)
        points.append((in_tokens, out_tokens, max(1.0, dur)))
    return points


class TestChatRegressionFit:
    def test_recovers_known_coefficients(self):
        """Give the fitter a known linear signal — recovered coefficients
        should land within a few % of ground truth. Regression not worth
        anything if it can't reproduce toy data."""
        # duration_ms = 0.5 ms/in + 30 ms/out + 200 ms overhead
        points = _synthetic_chat_points(0.5, 30.0, 200.0, n=300, noise_ms=100.0)
        fit = _fit_chat_regression(points)
        assert fit is not None
        assert 0.4 < fit.a_in < 0.6
        assert 25.0 < fit.b_out < 35.0
        assert fit.residual_std_s < 1.0  # 100ms noise → < 1s residual

    def test_rejects_too_few_samples(self):
        """Below the sample threshold the fit is unreliable — the fitter
        must decline instead of returning garbage the router would trust."""
        points = _synthetic_chat_points(0.5, 30.0, 200.0, n=10)
        assert _fit_chat_regression(points) is None

    def test_rejects_wildly_varying_data(self):
        """Data where the residual dwarfs the prediction is not a linear
        signal — better to fall back to raw percentiles than pretend a fit."""
        # Ground truth would give predictions around 15-30 s; adding 200 s
        # of Gaussian noise wipes out the signal so residual_std / mean_pred
        # blows past ``_MAX_RESIDUAL_RATIO_FOR_FIT``.
        points = _synthetic_chat_points(0.5, 30.0, 200.0, n=200, noise_ms=200_000.0)
        assert _fit_chat_regression(points) is None


class TestRegressionEstimatePath:
    @pytest.mark.asyncio
    async def test_regression_used_when_input_tokens_passed(self):
        """With a fitted calibration AND concrete input_tokens, we return
        the sharp regression prediction, not the raw p50."""
        estimator = DurationEstimator()
        _put(estimator, "vllm/chat-a", p50_s=8.0, p95_s=40.0, sample_count=500)
        estimator._calibrations["vllm/chat-a"].chat_regression = _ChatRegression(
            a_in=0.5, b_out=30.0, c=200.0, residual_std_s=0.5, p95_s=42.0,
        )
        # 1000 in + 500 out = 500 + 15000 + 200 = 15700 ms ≈ 15.7s
        result = await estimator.estimate(
            "vllm/chat-a",
            input_tokens=1000,
            max_tokens=500,
        )
        assert result.source == "regression"
        assert 15.0 < result.p50_s < 16.5
        # p95 = pred + 1.645·residual_std_s → 15.7 + 0.8 ≈ 16.5
        assert 16.0 < result.p95_s < 17.5

    @pytest.mark.asyncio
    async def test_regression_bypassed_without_input_tokens(self):
        """No input hint → we can't project the linear model → fall back
        to the raw percentile with no attempt to guess the axes."""
        estimator = DurationEstimator()
        _put(estimator, "vllm/chat-b", p50_s=8.0, p95_s=40.0, sample_count=500)
        estimator._calibrations["vllm/chat-b"].chat_regression = _ChatRegression(
            a_in=0.5, b_out=30.0, c=200.0, residual_std_s=0.5, p95_s=42.0,
        )
        result = await estimator.estimate("vllm/chat-b")  # no tokens passed
        assert result.source == "percentile"

    @pytest.mark.asyncio
    async def test_regression_confidence_is_higher_than_percentile(self):
        """Same model, same call otherwise — the regression path must give
        strictly more confidence than the raw percentile when its residuals
        are tight. Otherwise there's no incentive for consumers to prefer it."""
        estimator = DurationEstimator()
        _put(estimator, "vllm/chat-c", p50_s=8.0, p95_s=40.0, sample_count=500)
        # tight fit — residual small relative to prediction
        estimator._calibrations["vllm/chat-c"].chat_regression = _ChatRegression(
            a_in=0.5, b_out=30.0, c=200.0, residual_std_s=0.3, p95_s=42.0,
        )
        reg = await estimator.estimate("vllm/chat-c", input_tokens=1000, max_tokens=500)
        # Force the percentile path by asking the same model without hints.
        pct = await estimator.estimate("vllm/chat-c")
        assert reg.confidence > pct.confidence
