from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from ocabra.config import settings
from ocabra.stats import aggregator


def test_percentile_uses_nearest_rank() -> None:
    data = list(range(1, 101))

    assert aggregator._percentile(data, 95) == 95


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
