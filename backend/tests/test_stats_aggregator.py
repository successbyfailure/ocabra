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
        async def execute(self, _query):
            return _FakeResult(rows)

    class _FakeSessionFactory:
        async def __aenter__(self) -> _FakeSession:
            return _FakeSession()

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

    monkeypatch.setattr(aggregator, "AsyncSessionLocal", lambda: _FakeSessionFactory())
    monkeypatch.setattr(settings, "energy_cost_eur_kwh", 0.2)

    payload = await aggregator.get_energy_stats(t0, t0 + timedelta(hours=1))

    assert payload["totalKwh"] == pytest.approx(0.1667, rel=1e-4)
    assert payload["estimatedCostEur"] == pytest.approx(0.0333, rel=1e-4)
    assert payload["byGpu"][0]["powerDrawW"] == pytest.approx(166.7, rel=1e-3)
