from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
import sqlalchemy as sa

from ocabra.config import settings
from ocabra.database import AsyncSessionLocal
from ocabra.db.stats import GpuStat
from ocabra.stats import aggregator


def test_percentile_uses_nearest_rank() -> None:
    data = list(range(1, 101))

    assert aggregator._percentile(data, 95) == 95


@pytest.mark.asyncio
async def test_gpu_usage_coverage_matches_real_window_at_normal_cadence() -> None:
    """Regression: the gap cap used to be hardcoded at 30s while
    hw-monitor's real persist interval is ~60s, so EVERY normal inter-sample
    gap got truncated in half — hoursTotal/hoursActive/kwh were silently
    halved, always, independent of window size (reported by the user
    2026-10-26 as 'horas cubiertas' showing ~50% of the requested 24h/7d
    window). Use an unused gpu_index (999) and clean up unconditionally so
    this doesn't pollute the real dashboard — AsyncSessionLocal here points
    at the same Postgres the live stats pages read from."""
    gpu_index = 999
    t0 = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    # 5 samples, 60s apart — a real 4-minute window at hw-monitor's actual
    # cadence, no genuine gaps.
    samples = [
        GpuStat(
            recorded_at=t0 + timedelta(seconds=60 * i),
            gpu_index=gpu_index,
            utilization_pct=50.0,
            vram_used_mb=1000,
            power_draw_w=100.0,
            temperature_c=60.0,
        )
        for i in range(5)
    ]
    try:
        async with AsyncSessionLocal() as session:
            for row in samples:
                session.add(row)
            await session.commit()

        payload = await aggregator.get_gpu_usage_stats(
            t0, t0 + timedelta(seconds=60 * 4)
        )
        totals = {row["gpuIndex"]: row for row in payload["totals"]}

        # 4 real minutes of coverage, not 2 (half) — the old hardcoded 30s
        # cap would have reported ~0.0333h (2 min) instead of ~0.0667h.
        assert totals[gpu_index]["hoursTotal"] == pytest.approx(4 / 60, abs=1e-3)
        assert totals[gpu_index]["hoursActive"] == pytest.approx(4 / 60, abs=1e-3)
    finally:
        async with AsyncSessionLocal() as session:
            await session.execute(
                sa.delete(GpuStat).where(GpuStat.gpu_index == gpu_index)
            )
            await session.commit()


@pytest.mark.asyncio
async def test_gpu_usage_coverage_still_caps_a_genuine_outage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real multi-hour monitor outage must still be capped, not counted
    as if it were covered — the fix for the 30s-vs-60s mismatch must not
    remove this protection entirely."""
    monkeypatch.setattr(settings, "gpu_stats_expected_interval_s", 90)
    gpu_index = 998
    t0 = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    samples = [
        GpuStat(
            recorded_at=t0, gpu_index=gpu_index, utilization_pct=50.0,
            vram_used_mb=1000, power_draw_w=100.0, temperature_c=60.0,
        ),
        # 3-hour gap simulating a monitor outage.
        GpuStat(
            recorded_at=t0 + timedelta(hours=3), gpu_index=gpu_index,
            utilization_pct=50.0, vram_used_mb=1000, power_draw_w=100.0,
            temperature_c=60.0,
        ),
    ]
    try:
        async with AsyncSessionLocal() as session:
            for row in samples:
                session.add(row)
            await session.commit()

        payload = await aggregator.get_gpu_usage_stats(
            t0, t0 + timedelta(hours=3)
        )
        totals = {row["gpuIndex"]: row for row in payload["totals"]}

        # Capped at 90s for the first gap; the second sample is the last one
        # so its own trailing interval is capped against :to_dt the same way.
        assert totals[gpu_index]["hoursTotal"] == pytest.approx(180 / 3600, abs=1e-3)
    finally:
        async with AsyncSessionLocal() as session:
            await session.execute(
                sa.delete(GpuStat).where(GpuStat.gpu_index == gpu_index)
            )
            await session.commit()


@pytest.mark.asyncio
async def test_energy_stats_use_sample_intervals(monkeypatch: pytest.MonkeyPatch) -> None:
    t0 = datetime(2026, 4, 2, 12, 0, tzinfo=timezone.utc)
    rows = [
        SimpleNamespace(
            gpu_index=0,
            recorded_at=t0 + timedelta(minutes=20),
            power_draw_w=200.0,
        ),
        SimpleNamespace(
            gpu_index=0,
            recorded_at=t0,
            power_draw_w=100.0,
        ),
        SimpleNamespace(
            gpu_index=0,
            recorded_at=t0 + timedelta(minutes=60),
            power_draw_w=300.0,
        ),
    ]

    class _FakeResult:
        def __init__(self, rows: list[SimpleNamespace]) -> None:
            self._rows = rows

        def scalars(self) -> "_FakeResult":
            return self

        def all(self) -> list[SimpleNamespace]:
            return self._rows

    class _FakeSession:
        def __init__(self) -> None:
            self.calls = 0

        async def execute(self, _query):
            self.calls += 1
            return _FakeResult(rows if self.calls == 1 else [])

    fake_session = _FakeSession()

    class _FakeSessionFactory:
        async def __aenter__(self) -> _FakeSession:
            return fake_session

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

    monkeypatch.setattr(aggregator, "AsyncSessionLocal", lambda: _FakeSessionFactory())
    monkeypatch.setattr(settings, "energy_cost_eur_kwh", 0.2)

    payload = await aggregator.get_energy_stats(t0, t0 + timedelta(hours=1))

    assert payload["totalKwh"] == pytest.approx(0.1667, rel=1e-4)
    assert payload["estimatedCostEur"] == pytest.approx(0.0333, rel=1e-4)
    assert payload["byGpu"][0]["powerDrawW"] == pytest.approx(166.7, rel=1e-3)


@pytest.mark.asyncio
async def test_token_stats_all_time_can_skip_series(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeResult:
        def __init__(self, rows: list[tuple]) -> None:
            self._rows = rows

        def one(self) -> tuple:
            return self._rows[0]

        def all(self) -> list[tuple]:
            return self._rows

    class _FakeSession:
        def __init__(self) -> None:
            self.calls = 0
            self.results = [
                [(1500, 2500)],
                [("llama_cpp", 500, 700), ("vllm", 1000, 1800)],
                [(1, 1000, 1800), (None, 500, 700)],
            ]

        async def execute(self, _query):
            result = _FakeResult(self.results[self.calls])
            self.calls += 1
            return result

    fake_session = _FakeSession()

    class _FakeSessionFactory:
        async def __aenter__(self) -> _FakeSession:
            return fake_session

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

    monkeypatch.setattr(aggregator, "AsyncSessionLocal", lambda: _FakeSessionFactory())

    payload = await aggregator.get_token_stats(all_time=True, include_series=False)

    assert payload == {
        "totalInputTokens": 1500,
        "totalOutputTokens": 2500,
        "byBackend": [
            {"backendType": "llama_cpp", "inputTokens": 500, "outputTokens": 700},
            {"backendType": "vllm", "inputTokens": 1000, "outputTokens": 1800},
        ],
        "byGpu": [
            {"gpuIndex": 1, "inputTokens": 1000, "outputTokens": 1800},
            {"gpuIndex": None, "inputTokens": 500, "outputTokens": 700},
        ],
        "series": [],
    }
    assert fake_session.calls == 3


@pytest.mark.asyncio
async def test_performance_stats_tokens_per_second_prefers_generation_ms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """2026-10-27 bug: duration_ms is wall clock from HTTP arrival, so a
    request that waited behind a full queue shows a misleadingly low
    tok/s (reported: ~5 tok/s during real contention for an engine that
    does 50-80+ in isolation). generation_ms is the engine's own pure
    decode time; tokensPerSecond must use it when present instead of
    duration_ms, and only fall back to duration_ms for rows that lack it
    (older data, or backends that don't report it)."""
    # Request A: 10s wall clock (mostly queue wait) but only 1s of real
    # generation for 60 tokens -> true rate is 60 tok/s, not 6.
    row_with_generation_ms = SimpleNamespace(
        model_id="llama_cpp/qwen3.8-27b", backend_type="llama_cpp",
        request_kind="chat", duration_ms=10000, generation_ms=1000,
        input_tokens=10, output_tokens=60, error=None, status_code=200,
    )
    # Request B: no generation_ms (older row / non-reporting backend) ->
    # falls back to duration_ms, as before this fix.
    row_without_generation_ms = SimpleNamespace(
        model_id="llama_cpp/qwen3.8-27b", backend_type="llama_cpp",
        request_kind="chat", duration_ms=2000, generation_ms=None,
        input_tokens=5, output_tokens=20, error=None, status_code=200,
    )

    class _FakeResult:
        def __init__(self, rows: list) -> None:
            self._rows = rows

        def scalars(self) -> "_FakeResult":
            return self

        def all(self) -> list:
            return self._rows

    class _FakeSession:
        def __init__(self) -> None:
            self.calls = 0

        async def execute(self, _query):
            self.calls += 1
            rows = [row_with_generation_ms, row_without_generation_ms] if self.calls == 1 else []
            return _FakeResult(rows)

    fake_session = _FakeSession()

    class _FakeSessionFactory:
        async def __aenter__(self) -> _FakeSession:
            return fake_session

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

    monkeypatch.setattr(aggregator, "AsyncSessionLocal", lambda: _FakeSessionFactory())

    payload = await aggregator.get_performance_stats(model_id="llama_cpp/qwen3.8-27b")

    summary = payload["byModel"][0]
    # (60 + 20) tokens / (1000 + 2000) ms * 1000 = 26.67 tok/s — NOT
    # (60 + 20) / (10000 + 2000) * 1000 = 6.67 tok/s, the old (wrong) number.
    assert summary["tokensPerSecond"] == pytest.approx(26.67, abs=0.01)


@pytest.mark.asyncio
async def test_performance_stats_exposes_draft_acceptance_and_engine_speed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """2026-10-29 — added after being unable to confirm live whether
    qwen3.8:27b's self-MTP acceptance rate had regressed from an earlier
    ~80 tok/s baseline (measured ~33-35 tok/s that day, but real GPU1
    contention made it impossible to catch a clean, uncontended moment to
    verify why). draftAcceptanceRate = sum(draft_n_accepted)/sum(draft_n)
    across all rows — the single number that most directly explains a
    self-MTP model's real-world tok/s."""
    row_a = SimpleNamespace(
        model_id="llama_cpp/qwen3.8-27b", backend_type="llama_cpp",
        request_kind="chat", duration_ms=1000, generation_ms=1000,
        input_tokens=10, output_tokens=60, error=None, status_code=200,
        draft_n=16, draft_n_accepted=13,
        predicted_tokens_per_second=80.0, prompt_tokens_per_second=400.0,
    )
    row_b = SimpleNamespace(
        model_id="llama_cpp/qwen3.8-27b", backend_type="llama_cpp",
        request_kind="chat", duration_ms=1000, generation_ms=1000,
        input_tokens=10, output_tokens=40, error=None, status_code=200,
        draft_n=20, draft_n_accepted=7,
        predicted_tokens_per_second=34.0, prompt_tokens_per_second=380.0,
    )
    # Non-speculative row (plain llama.cpp load, no draft_n at all) — must
    # not pollute the acceptance-rate denominator with a phantom 0/0.
    row_c = SimpleNamespace(
        model_id="llama_cpp/qwen3.8-27b", backend_type="llama_cpp",
        request_kind="chat", duration_ms=1000, generation_ms=1000,
        input_tokens=10, output_tokens=20, error=None, status_code=200,
        draft_n=None, draft_n_accepted=None,
        predicted_tokens_per_second=None, prompt_tokens_per_second=None,
    )

    class _FakeResult:
        def __init__(self, rows: list) -> None:
            self._rows = rows

        def scalars(self) -> "_FakeResult":
            return self

        def all(self) -> list:
            return self._rows

    class _FakeSession:
        def __init__(self) -> None:
            self.calls = 0

        async def execute(self, _query):
            self.calls += 1
            rows = [row_a, row_b, row_c] if self.calls == 1 else []
            return _FakeResult(rows)

    fake_session = _FakeSession()

    class _FakeSessionFactory:
        async def __aenter__(self) -> _FakeSession:
            return fake_session

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

    monkeypatch.setattr(aggregator, "AsyncSessionLocal", lambda: _FakeSessionFactory())

    payload = await aggregator.get_performance_stats(model_id="llama_cpp/qwen3.8-27b")

    summary = payload["byModel"][0]
    # (13 + 7) accepted / (16 + 20) drafted = 0.556
    assert summary["draftAcceptanceRate"] == pytest.approx(0.556, abs=0.001)
    assert summary["avgPredictedTokensPerSecond"] == pytest.approx(57.0, abs=0.01)
    assert summary["avgPromptTokensPerSecond"] == pytest.approx(390.0, abs=0.01)


@pytest.mark.asyncio
async def test_performance_stats_speculative_fields_are_none_without_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-llama.cpp model (or rows predating this field) must show None,
    not 0 or a crash — 0 would misleadingly read as '0% acceptance'."""
    row = SimpleNamespace(
        model_id="ollama/mistral:7b", backend_type="ollama",
        request_kind="chat", duration_ms=1000, generation_ms=None,
        input_tokens=10, output_tokens=20, error=None, status_code=200,
        draft_n=None, draft_n_accepted=None,
        predicted_tokens_per_second=None, prompt_tokens_per_second=None,
    )

    class _FakeResult:
        def __init__(self, rows: list) -> None:
            self._rows = rows

        def scalars(self) -> "_FakeResult":
            return self

        def all(self) -> list:
            return self._rows

    class _FakeSession:
        def __init__(self) -> None:
            self.calls = 0

        async def execute(self, _query):
            self.calls += 1
            rows = [row] if self.calls == 1 else []
            return _FakeResult(rows)

    fake_session = _FakeSession()

    class _FakeSessionFactory:
        async def __aenter__(self) -> _FakeSession:
            return fake_session

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

    monkeypatch.setattr(aggregator, "AsyncSessionLocal", lambda: _FakeSessionFactory())

    payload = await aggregator.get_performance_stats(model_id="ollama/mistral:7b")

    summary = payload["byModel"][0]
    assert summary["draftAcceptanceRate"] is None
    assert summary["avgPredictedTokensPerSecond"] is None
    assert summary["avgPromptTokensPerSecond"] is None
