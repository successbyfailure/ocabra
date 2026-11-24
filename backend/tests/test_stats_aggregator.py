from datetime import UTC, datetime, timedelta

import pytest
import sqlalchemy as sa

from ocabra.config import settings
from ocabra.database import AsyncSessionLocal
from ocabra.db.stats import GpuStat
from ocabra.stats import aggregator
from tests.conftest import requires_db


def test_percentile_uses_nearest_rank() -> None:
    data = list(range(1, 101))

    assert aggregator._percentile(data, 95) == 95


@pytest.mark.asyncio
@requires_db
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
    t0 = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
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
@requires_db
async def test_gpu_usage_coverage_still_caps_a_genuine_outage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real multi-hour monitor outage must still be capped, not counted
    as if it were covered — the fix for the 30s-vs-60s mismatch must not
    remove this protection entirely."""
    monkeypatch.setattr(settings, "gpu_stats_expected_interval_s", 90)
    gpu_index = 998
    t0 = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
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

        # El hueco de 3 h del primer sample se capa a 90 s. El segundo sample
        # es el ultimo y cae exactamente en el final de la ventana, asi que su
        # intervalo de cola es CERO (`COALESCE(next_at, :to_dt) - recorded_at`),
        # no 90 s: el total correcto son 90 s, no 180. El test esperaba 180
        # dando por hecho que el ultimo sample suma un intervalo completo.
        assert totals[gpu_index]["hoursTotal"] == pytest.approx(90 / 3600, abs=1e-3)
        # Lo que de verdad protege este test: el apagon no se cuenta como
        # cubierto. Sin el cap serian las 3 h enteras.
        assert totals[gpu_index]["hoursTotal"] < 0.1
    finally:
        async with AsyncSessionLocal() as session:
            await session.execute(
                sa.delete(GpuStat).where(GpuStat.gpu_index == gpu_index)
            )
            await session.commit()


@pytest.mark.asyncio
@requires_db
async def test_energy_stats_use_sample_intervals(monkeypatch: pytest.MonkeyPatch) -> None:
    """Energy integrates power over each sample's interval (now in SQL via a
    LEAD window, no per-sample gap cap). Three GPU samples 100/200/300 W at
    0/20/60 min over a 1 h window: 100 W * 20 min + 200 W * 40 min = 166.67 Wh
    (the last sample sits on the window end, so its tail interval is zero)."""
    from ocabra.db.stats import GpuStat

    monkeypatch.setattr(settings, "energy_cost_eur_kwh", 0.2)
    gpu_index = 991
    t0 = datetime(2026, 4, 2, 12, 0, tzinfo=UTC)
    samples = [
        GpuStat(recorded_at=t0, gpu_index=gpu_index, power_draw_w=100.0),
        GpuStat(recorded_at=t0 + timedelta(minutes=20), gpu_index=gpu_index, power_draw_w=200.0),
        GpuStat(recorded_at=t0 + timedelta(minutes=60), gpu_index=gpu_index, power_draw_w=300.0),
    ]
    try:
        async with AsyncSessionLocal() as session:
            for row in samples:
                session.add(row)
            await session.commit()

        payload = await aggregator.get_energy_stats(t0, t0 + timedelta(hours=1))
        by_gpu = {row["gpuIndex"]: row for row in payload["byGpu"]}

        assert by_gpu[gpu_index]["totalKwh"] == pytest.approx(0.1667, rel=1e-4)
        assert by_gpu[gpu_index]["powerDrawW"] == pytest.approx(166.7, rel=1e-3)
        assert payload["totalKwh"] == pytest.approx(0.1667, rel=1e-4)
        assert payload["estimatedCostEur"] == pytest.approx(0.0333, rel=1e-4)
    finally:
        async with AsyncSessionLocal() as session:
            await session.execute(sa.delete(GpuStat).where(GpuStat.gpu_index == gpu_index))
            await session.commit()


async def _seed_requests(rows: list) -> None:
    from ocabra.db.stats import RequestStat

    async with AsyncSessionLocal() as session:
        for kw in rows:
            session.add(RequestStat(**kw))
        await session.commit()


async def _clear_model(model_id: str) -> None:
    from ocabra.db.stats import RequestStat

    async with AsyncSessionLocal() as session:
        await session.execute(sa.delete(RequestStat).where(RequestStat.model_id == model_id))
        await session.commit()


@pytest.mark.asyncio
@requires_db
async def test_performance_stats_tokens_per_second_prefers_generation_ms() -> None:
    """duration_ms is wall clock (queue + cold start included); tokensPerSecond
    must use generation_ms when present and fall back to duration_ms only when
    it is missing. (60+20) tokens / (1000+2000) ms * 1000 = 26.67 tok/s, not
    the 6.67 you get dividing by the 10s+2s wall clock."""
    mid = "llama_cpp/qwen3.8-27b-testperf-tps"
    t0 = datetime(2026, 4, 3, 9, 0, tzinfo=UTC)
    try:
        await _seed_requests(
            [
                dict(
                    model_id=mid, backend_type="llama_cpp", request_kind="chat",
                    started_at=t0, status_code=200, duration_ms=10000, generation_ms=1000,
                    input_tokens=10, output_tokens=60,
                ),
                dict(
                    model_id=mid, backend_type="llama_cpp", request_kind="chat",
                    started_at=t0, status_code=200, duration_ms=2000, generation_ms=None,
                    input_tokens=5, output_tokens=20,
                ),
            ]
        )
        payload = await aggregator.get_performance_stats(
            model_id=mid, from_dt=t0 - timedelta(hours=1), to_dt=t0 + timedelta(hours=1)
        )
        summary = payload["byModel"][0]
        assert summary["tokensPerSecond"] == pytest.approx(26.67, abs=0.01)
    finally:
        await _clear_model(mid)


@pytest.mark.asyncio
@requires_db
async def test_performance_stats_exposes_draft_acceptance_and_engine_speed() -> None:
    """draftAcceptanceRate = sum(accepted)/sum(drafted) across rows; a
    non-speculative row (no draft_n) must not pollute the denominator with a
    phantom 0/0, and the tok/s averages ignore rows that don't report them."""
    mid = "llama_cpp/qwen3.8-27b-testperf-draft"
    t0 = datetime(2026, 4, 3, 10, 0, tzinfo=UTC)
    try:
        await _seed_requests(
            [
                dict(
                    model_id=mid, backend_type="llama_cpp", request_kind="chat",
                    started_at=t0, status_code=200, duration_ms=1000, generation_ms=1000,
                    input_tokens=10, output_tokens=60, draft_n=16, draft_n_accepted=13,
                    predicted_tokens_per_second=80.0, prompt_tokens_per_second=400.0,
                ),
                dict(
                    model_id=mid, backend_type="llama_cpp", request_kind="chat",
                    started_at=t0, status_code=200, duration_ms=1000, generation_ms=1000,
                    input_tokens=10, output_tokens=40, draft_n=20, draft_n_accepted=7,
                    predicted_tokens_per_second=34.0, prompt_tokens_per_second=380.0,
                ),
                dict(
                    model_id=mid, backend_type="llama_cpp", request_kind="chat",
                    started_at=t0, status_code=200, duration_ms=1000, generation_ms=1000,
                    input_tokens=10, output_tokens=20, draft_n=None, draft_n_accepted=None,
                    predicted_tokens_per_second=None, prompt_tokens_per_second=None,
                ),
            ]
        )
        payload = await aggregator.get_performance_stats(
            model_id=mid, from_dt=t0 - timedelta(hours=1), to_dt=t0 + timedelta(hours=1)
        )
        summary = payload["byModel"][0]
        assert summary["draftAcceptanceRate"] == pytest.approx(0.556, abs=0.001)
        assert summary["avgPredictedTokensPerSecond"] == pytest.approx(57.0, abs=0.01)
        assert summary["avgPromptTokensPerSecond"] == pytest.approx(390.0, abs=0.01)
    finally:
        await _clear_model(mid)


@pytest.mark.asyncio
@requires_db
async def test_performance_stats_speculative_fields_are_none_without_data() -> None:
    """A non-llama.cpp model (no draft/tps data) shows None, not 0 — 0 would
    misleadingly read as '0% acceptance'."""
    mid = "ollama/mistral-7b-testperf-none"
    t0 = datetime(2026, 4, 3, 11, 0, tzinfo=UTC)
    try:
        await _seed_requests(
            [
                dict(
                    model_id=mid, backend_type="ollama", request_kind="chat",
                    started_at=t0, status_code=200, duration_ms=1000, generation_ms=None,
                    input_tokens=10, output_tokens=20,
                ),
            ]
        )
        payload = await aggregator.get_performance_stats(
            model_id=mid, from_dt=t0 - timedelta(hours=1), to_dt=t0 + timedelta(hours=1)
        )
        summary = payload["byModel"][0]
        assert summary["draftAcceptanceRate"] is None
        assert summary["avgPredictedTokensPerSecond"] is None
        assert summary["avgPromptTokensPerSecond"] is None
    finally:
        await _clear_model(mid)


@pytest.mark.asyncio
@requires_db
async def test_performance_stats_work_speeds_in_sql() -> None:
    """workSpeeds (x_realtime for transcription) is now computed in SQL from
    work_size_meta. Two 10s-audio transcriptions at 5s wall clock each:
    20 s audio / 10 s wall = 2.0x realtime over 2 samples."""
    mid = "whisper/large-v3-testperf-ws"
    t0 = datetime(2026, 4, 3, 12, 0, tzinfo=UTC)
    try:
        await _seed_requests(
            [
                dict(
                    model_id=mid, backend_type="whisper",
                    request_kind="audio_transcription", started_at=t0, status_code=200,
                    duration_ms=5000, work_size_meta={"audio_seconds": 10.0},
                ),
                dict(
                    model_id=mid, backend_type="whisper",
                    request_kind="audio_transcription", started_at=t0, status_code=200,
                    duration_ms=5000, work_size_meta={"audio_seconds": 10.0},
                ),
            ]
        )
        payload = await aggregator.get_performance_stats(
            model_id=mid, from_dt=t0 - timedelta(hours=1), to_dt=t0 + timedelta(hours=1)
        )
        summary = payload["byModel"][0]
        speeds = {(tuple(s["kinds"]), s["unit"]): s for s in summary["workSpeeds"]}
        key = (("audio_transcription",), "x_realtime")
        assert key in speeds
        assert speeds[key]["value"] == pytest.approx(2.0, abs=0.01)
        assert speeds[key]["samples"] == 2
    finally:
        await _clear_model(mid)


@pytest.mark.asyncio
@requires_db
async def test_request_stats_aggregated_in_sql() -> None:
    """get_request_stats counts, error rate (429 excluded as rejections),
    percentiles and the per-minute series, all in SQL."""
    mid = "llama_cpp/testreq"
    t0 = datetime(2026, 4, 4, 8, 0, tzinfo=UTC)
    try:
        await _seed_requests(
            [
                dict(model_id=mid, started_at=t0, status_code=200, duration_ms=100),
                dict(model_id=mid, started_at=t0, status_code=200, duration_ms=200),
                dict(model_id=mid, started_at=t0, status_code=500, duration_ms=300),
                dict(model_id=mid, started_at=t0, status_code=429, duration_ms=50),
            ]
        )
        payload = await aggregator.get_request_stats(
            t0 - timedelta(hours=1), t0 + timedelta(hours=1), mid
        )
        assert payload["totalRequests"] == 4
        assert payload["rejections"] == 1
        assert payload["errorRate"] == pytest.approx(round(1 / 4, 4))
        assert payload["p95DurationMs"] == 300
        # Nearest-rank p50 of [50, 100, 200, 300] is the 2nd value (ceil(0.5*4)=2).
        assert payload["p50DurationMs"] == 100
        assert sum(pt["count"] for pt in payload["series"]) == 4
    finally:
        await _clear_model(mid)


@pytest.mark.asyncio
@requires_db
async def test_overview_stats_aggregated_in_sql() -> None:
    """get_overview_stats totals, by-backend summaries and the reference cost
    are computed in SQL; cost groups by the tier triple."""
    mid = "llama_cpp/testoverview-7b"
    t0 = datetime(2026, 4, 4, 9, 0, tzinfo=UTC)
    try:
        await _seed_requests(
            [
                dict(
                    model_id=mid, backend_type="llama_cpp", request_kind="chat",
                    started_at=t0, status_code=200, duration_ms=100,
                    input_tokens=1000, output_tokens=2000,
                ),
                dict(
                    model_id=mid, backend_type="llama_cpp", request_kind="chat",
                    started_at=t0, status_code=500, duration_ms=300,
                    input_tokens=0, output_tokens=0,
                ),
            ]
        )
        payload = await aggregator.get_overview_stats(
            t0 - timedelta(hours=1), t0 + timedelta(hours=1), mid
        )
        assert payload["totalRequests"] == 2
        assert payload["totalErrors"] == 1
        assert payload["totalInputTokens"] == 1000
        assert payload["totalOutputTokens"] == 2000
        assert payload["tokenizedRequests"] == 1
        assert payload["byBackend"][0]["backendType"] == "llama_cpp"
        assert payload["byBackend"][0]["totalRequests"] == 2
        # small tier (7b): input 1000 * small_input/1M + output 2000 * small_output/1M
        from ocabra.stats.cost_calculator import estimate_request_cost
        expected = round(estimate_request_cost(1000, 2000, "small"), 4)
        assert payload["estimatedCostUsd"] == pytest.approx(expected)
    finally:
        await _clear_model(mid)
