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
