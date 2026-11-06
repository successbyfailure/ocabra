"""TTFT and queue wait in the performance aggregate (2026-11-06).

Both were already captured per request (``RequestRuntimeTelemetry``) and
persisted in ``request_stats.runtime_telemetry``, but nothing aggregated them,
so they were invisible outside a hand-written SQL query — 100 of 357 chat
requests carried a ttft_ms nobody could see. They answer two questions
``avgLatencyMs`` cannot: how long until the user sees anything, and how much
of the wait was queueing rather than work.
"""

from __future__ import annotations

from types import SimpleNamespace

from ocabra.stats.aggregator import _percentile, _telemetry_int


class TestTelemetryInt:
    def test_reads_a_value(self) -> None:
        row = SimpleNamespace(runtime_telemetry={"ttft_ms": 120})
        assert _telemetry_int(row, "ttft_ms") == 120

    def test_missing_key_is_none(self) -> None:
        assert _telemetry_int(SimpleNamespace(runtime_telemetry={}), "ttft_ms") is None

    def test_rows_written_before_the_field_existed(self) -> None:
        """Older DB rows and fixtures carry no telemetry at all."""
        assert _telemetry_int(SimpleNamespace(runtime_telemetry=None), "ttft_ms") is None
        assert _telemetry_int(SimpleNamespace(), "ttft_ms") is None

    def test_garbage_does_not_raise(self) -> None:
        row = SimpleNamespace(runtime_telemetry={"ttft_ms": "no-soy-un-numero"})
        assert _telemetry_int(row, "ttft_ms") is None


class TestPercentile:
    """Reuses the aggregator's existing nearest-rank helper (percentage-based);
    the empty-sample distinction is made by the caller, since that one returns
    0 and "no streaming requests" must not read as "0 ms"."""

    def test_median_and_tail(self) -> None:
        values = sorted([100, 120, 130, 200, 900])
        assert _percentile(values, 50) == 130
        assert _percentile(values, 95) == 900

    def test_single_value(self) -> None:
        assert _percentile([42], 95) == 42

    def test_never_indexes_out_of_range(self) -> None:
        assert _percentile(sorted([1, 2]), 100) == 2


class TestTtftDecomposition:
    """``ttft_ms`` is measured from HTTP arrival, so it bundles the admission
    wait, the model load and the engine's own latency. Reporting only the total
    makes a cold start read as a slow model: measured in production on
    2026-11-06, a 68.7s "TTFT" on qwen3:8b that was 68.2s of load, 0s of queue
    and ~0.5s of engine.
    """

    @staticmethod
    def _engine(ttft: int | None, load: int | None, queue: int | None) -> int:
        """Mirrors the aggregator's arithmetic."""
        return max(0, (ttft or 0) - (load or 0) - (queue or 0))

    def test_strips_a_cold_start(self) -> None:
        assert self._engine(68691, 68206, 0) == 485

    def test_strips_queue_wait(self) -> None:
        assert self._engine(13500, 0, 13000) == 500

    def test_strips_both(self) -> None:
        assert self._engine(80000, 60000, 19000) == 1000

    def test_a_warm_uncontended_request_is_unchanged(self) -> None:
        assert self._engine(120, None, 0) == 120

    def test_never_goes_negative(self) -> None:
        """Load and queue are recorded by different components, so clock skew
        or overlap must not produce a negative latency."""
        assert self._engine(100, 200, 50) == 0
