"""Prune old statistics rows so the stats tables stop growing unbounded.

No stats table had any retention: on the production server when this was
written ``gpu_stats`` held ~708k rows, ``server_stats`` ~304k and
``request_stats`` ~161k, all the way back to 2026-03-12. This prunes the three
high-volume time-series tables past a configurable per-table horizon.

Safe by design on first boot: the whole thing is **off** unless
``settings.stats_retention_enabled`` is true, so merely deploying this never
deletes anything. The Alembic migration that ships alongside only adds an
index — it deletes no data — so an operator can watch the tables, tune the
horizons in Settings and then enable pruning, instead of discovering months of
history gone after an upgrade.

Deletes are **batched** (``stats_retention_batch_size`` rows per statement, a
fresh short transaction each time) and yield between batches, so a first prune
over a large backlog never holds a long lock or blocks the event loop. Deleting
a ``request_stats`` row cascades to its ``tool_call_stats`` children (the FK is
``ON DELETE CASCADE``) and nulls ``parent_request_id`` on any hops that pointed
at it (``ON DELETE SET NULL``).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import sqlalchemy as sa
import structlog

logger = structlog.get_logger(__name__)

# (table, timestamp column, settings attribute with the max age in days). Table
# and column names are internal constants, never user input, so interpolating
# them into the DELETE is safe; the cutoff and batch size are bound params.
_TABLES: tuple[tuple[str, str, str], ...] = (
    ("request_stats", "started_at", "stats_retention_request_stats_days"),
    ("gpu_stats", "recorded_at", "stats_retention_gpu_stats_days"),
    ("server_stats", "recorded_at", "stats_retention_server_stats_days"),
)


async def _prune_table(
    session_factory: Any, table: str, column: str, cutoff: datetime, batch: int
) -> int:
    """Delete rows older than *cutoff* from *table*, ``batch`` at a time.

    ``ctid`` keeps this independent of each table's primary key (the
    time-series tables use a composite PK, request_stats a uuid).
    """
    stmt = sa.text(
        f"DELETE FROM {table} WHERE ctid IN "  # noqa: S608 - table/column are fixed constants
        f"(SELECT ctid FROM {table} WHERE {column} < :cutoff LIMIT :batch)"
    )
    total = 0
    while True:
        async with session_factory() as session:
            result = await session.execute(stmt, {"cutoff": cutoff, "batch": batch})
            await session.commit()
            removed = result.rowcount or 0
        total += removed
        if removed < batch:
            break
        await asyncio.sleep(0)  # yield between batches; never hold the loop
    return total


async def purge_old_stats(session_factory: Any, now: datetime | None = None) -> dict[str, int]:
    """Prune every configured stats table. Returns ``{table: rows_deleted}``.

    A no-op (returns ``{}``) when retention is disabled. A per-table age of 0
    means "keep forever" and skips that table.
    """
    from ocabra.config import settings

    if not settings.stats_retention_enabled:
        return {}
    now = now or datetime.now(UTC)
    batch = max(1, int(settings.stats_retention_batch_size))
    deleted: dict[str, int] = {}
    for table, column, days_attr in _TABLES:
        days = int(getattr(settings, days_attr))
        if days <= 0:
            continue
        cutoff = now - timedelta(days=days)
        removed = await _prune_table(session_factory, table, column, cutoff, batch)
        if removed:
            deleted[table] = removed
    if deleted:
        logger.info("stats_retention_purged", **deleted)
    return deleted


async def retention_loop(stop_event: asyncio.Event, session_factory: Any) -> None:
    """Run :func:`purge_old_stats` every ``stats_retention_interval_s``."""
    from ocabra.config import settings

    while not stop_event.is_set():
        try:
            await purge_old_stats(session_factory)
        except Exception as exc:  # noqa: BLE001 - keep the loop alive
            logger.warning("stats_retention_loop_error", error=str(exc))
        try:
            await asyncio.wait_for(
                stop_event.wait(), timeout=max(60, int(settings.stats_retention_interval_s))
            )
        except TimeoutError:
            continue
