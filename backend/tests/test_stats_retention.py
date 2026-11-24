"""Tests for the stats retention prune (REF-12)."""

from datetime import UTC, datetime, timedelta

import pytest
import sqlalchemy as sa

from ocabra.config import settings
from ocabra.database import AsyncSessionLocal
from ocabra.db.stats import GpuStat, RequestStat
from ocabra.stats import retention
from tests.conftest import requires_db


@pytest.mark.asyncio
async def test_purge_is_a_noop_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disabled by default: deploying must never delete anything. The session
    factory must not even be touched."""
    monkeypatch.setattr(settings, "stats_retention_enabled", False)

    def _boom() -> None:
        raise AssertionError("session factory must not be used when retention is disabled")

    result = await retention.purge_old_stats(_boom)
    assert result == {}


@pytest.mark.asyncio
@requires_db
async def test_purge_deletes_only_rows_past_the_horizon(monkeypatch: pytest.MonkeyPatch) -> None:
    """With retention enabled, rows older than the per-table horizon are
    deleted in batches and fresh rows are kept."""
    monkeypatch.setattr(settings, "stats_retention_enabled", True)
    monkeypatch.setattr(settings, "stats_retention_request_stats_days", 30)
    monkeypatch.setattr(settings, "stats_retention_gpu_stats_days", 0)  # keep forever
    monkeypatch.setattr(settings, "stats_retention_server_stats_days", 0)
    monkeypatch.setattr(settings, "stats_retention_batch_size", 2)

    mid = "llama_cpp/testretention"
    gpu_index = 981
    now = datetime(2026, 5, 1, 12, 0, tzinfo=UTC)
    old = now - timedelta(days=60)
    fresh = now - timedelta(days=5)
    try:
        async with AsyncSessionLocal() as session:
            # 3 old request_stats (forces >1 batch at batch_size=2) + 1 fresh.
            for _ in range(3):
                session.add(RequestStat(model_id=mid, started_at=old))
            session.add(RequestStat(model_id=mid, started_at=fresh))
            # A gpu_stats row older than any horizon; days=0 means keep it.
            session.add(GpuStat(recorded_at=old, gpu_index=gpu_index, power_draw_w=10.0))
            await session.commit()

        deleted = await retention.purge_old_stats(AsyncSessionLocal, now=now)
        assert deleted.get("request_stats") == 3
        assert "gpu_stats" not in deleted  # days=0 -> kept

        async with AsyncSessionLocal() as session:
            remaining = (
                await session.execute(
                    sa.select(sa.func.count()).where(RequestStat.model_id == mid)
                )
            ).scalar()
            assert remaining == 1  # only the fresh row survived
            gpu_remaining = (
                await session.execute(
                    sa.select(sa.func.count()).where(GpuStat.gpu_index == gpu_index)
                )
            ).scalar()
            assert gpu_remaining == 1  # kept (days=0)
    finally:
        async with AsyncSessionLocal() as session:
            await session.execute(sa.delete(RequestStat).where(RequestStat.model_id == mid))
            await session.execute(sa.delete(GpuStat).where(GpuStat.gpu_index == gpu_index))
            await session.commit()
