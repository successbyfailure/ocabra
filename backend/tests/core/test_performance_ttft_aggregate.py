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
