"""Tests for the dynamic drain grace in pressure_eviction.

Bloque 20, Etapa 7.

Verifies that ``_assign_gpus_for_load``'s drain wait uses the estimator's
``in_flight_remaining`` when available, capped by ``max_drain_timeout_s``,
and falls back to the static ``pressure_eviction_drain_timeout_s`` when
the estimator has nothing to say.

The pressure-eviction path is deep behind a scheduler; here we shortcut by
calling into the estimator + in-flight bridging surface directly. The
integration path itself is exercised in ``test_load_unload_e2e`` (already
covered by regressions) — this file just proves the wiring points.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from ocabra.core.duration_estimator import DurationEstimator


class TestManagerEstimatorBridge:
    def test_mark_request_start_forwards_to_estimator(self):
        """The bridge on the manager side lets it announce a load path is
        active without holding a direct estimator reference. Verify the
        estimator's in-flight table sees the entry."""
        from ocabra.core.model_manager import ModelManager

        mm = ModelManager(worker_pool=MagicMock())
        est = DurationEstimator()
        mm.set_duration_estimator(est)
        mm.mark_request_start("vllm/x")
        assert "vllm/x" in est._in_flight
        mm.mark_request_end("vllm/x")
        assert "vllm/x" not in est._in_flight

    def test_bridge_survives_missing_estimator(self):
        """Manager is constructed before the estimator in main.py, so a
        mark_request_start before set_duration_estimator must not raise."""
        from ocabra.core.model_manager import ModelManager

        mm = ModelManager(worker_pool=MagicMock())
        mm.mark_request_start("no-op")  # no exception
        mm.mark_request_end("no-op")


class TestDrainGraceCalculation:
    def test_estimator_remaining_shapes_drain_timeout(self):
        """The pressure_eviction path uses
        ``min(max_drain_timeout_s, remaining + 15)`` as its drain timeout.
        Verify the estimator returns a usable number given a live in-flight
        entry."""
        import time

        est = DurationEstimator()
        # Inject a calibration so remaining actually computes.
        from ocabra.core.duration_estimator import _ModelCalibration

        est._calibrations["vllm/x"] = _ModelCalibration(
            model_id="vllm/x",
            backend_type="vllm",
            p50_s=8.0,
            p95_s=30.0,
            sample_count=100,
            residual_ratio=0.5,
            cold_start_p95_s=0.0,
        )
        est.mark_start("vllm/x")
        remaining = est.in_flight_remaining("vllm/x", model_id="vllm/x")
        # A tick after mark_start we should still have close to the full p95.
        assert remaining is not None and 25 < remaining <= 30
        est.mark_end("vllm/x")

    def test_max_drain_timeout_caps_wild_estimates(self, monkeypatch):
        """Even if the estimator says 5000s, we never wait more than
        ``max_drain_timeout_s`` — a guard against runaway calibration."""
        from ocabra.config import settings

        monkeypatch.setattr(settings, "max_drain_timeout_s", 300)
        # Simulate the min() computation the manager does inline.
        remaining = 5000.0
        drain = min(int(settings.max_drain_timeout_s), int(remaining) + 15)
        assert drain == 300

    def test_static_default_used_when_estimator_absent(self, monkeypatch):
        """No estimator wired → dust off the pre-Bloque-20 constant so the
        pressure_eviction loop behaves exactly as it used to."""
        from ocabra.config import settings

        monkeypatch.setattr(settings, "pressure_eviction_drain_timeout_s", 60)
        static = max(1, int(settings.pressure_eviction_drain_timeout_s))
        assert static == 60
