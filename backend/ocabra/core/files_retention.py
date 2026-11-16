"""Delete batch files and uploads once they are no longer needed.

The "optimal" privacy mode promises that what oCabra writes temporarily gets
cleaned up. Batch input/output/error files were never deleted: 1,964 of them
(89 MB, since 2026-07-26) on the production server when this was written.

Rule, with ``openai_files_retention_hours`` (0 = keep everything):

* a **finished** batch (completed, failed, expired, cancelled) older than the
  retention is deleted together with its output, error and input files (rows
  and files on disk) — the batch row has to go first: its input file is
  ``ON DELETE RESTRICT``;
* a file **no batch refers to**, uploaded longer ago than the retention, is
  deleted;
* batches still running, and their files, are never touched.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import sqlalchemy as sa
import structlog

logger = structlog.get_logger(__name__)

FINISHED = ("completed", "failed", "expired", "cancelled")


def _unlink(path: str | None) -> None:
    if path:
        try:
            Path(path).unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("files_retention_unlink_failed", path=path, error=str(exc))


async def purge_expired(
    session_factory: Any, retention_hours: int, now: datetime | None = None
) -> dict[str, int]:
    """Delete what the retention lets go. Returns ``{batches, files}`` deleted."""
    from ocabra.db.openai_batches import OpenAIBatch, OpenAIFile

    if retention_hours <= 0:
        return {"batches": 0, "files": 0}
    now = now or datetime.now(UTC)
    cutoff = now - timedelta(hours=retention_hours)
    ended_at = sa.func.coalesce(
        OpenAIBatch.completed_at,
        OpenAIBatch.failed_at,
        OpenAIBatch.expired_at,
        OpenAIBatch.cancelled_at,
        OpenAIBatch.created_at,
    )
    deleted_batches = 0
    to_unlink: list[str] = []
    async with session_factory() as session:
        batches = (
            (
                await session.execute(
                    sa.select(OpenAIBatch).where(
                        OpenAIBatch.status.in_(FINISHED), ended_at < cutoff
                    )
                )
            )
            .scalars()
            .all()
        )
        file_ids: set[Any] = set()
        for b in batches:
            file_ids.update(f for f in (b.input_file_id, b.output_file_id, b.error_file_id) if f)
            await session.delete(b)
            deleted_batches += 1
        await session.flush()

        # Files of those batches no other (live) batch still uses, plus loose
        # uploads past the retention that no batch refers to.
        referenced = sa.select(OpenAIBatch.input_file_id).union(
            sa.select(OpenAIBatch.output_file_id).where(OpenAIBatch.output_file_id.is_not(None)),
            sa.select(OpenAIBatch.error_file_id).where(OpenAIBatch.error_file_id.is_not(None)),
        )
        candidates = sa.or_(
            OpenAIFile.id.in_(file_ids) if file_ids else sa.false(),
            OpenAIFile.created_at < cutoff,
        )
        files = (
            (
                await session.execute(
                    sa.select(OpenAIFile).where(candidates, OpenAIFile.id.not_in(referenced))
                )
            )
            .scalars()
            .all()
        )
        for f in files:
            to_unlink.append(f.storage_path)
            await session.delete(f)
        await session.commit()
    for path in to_unlink:
        await asyncio.to_thread(_unlink, path)
    if deleted_batches or to_unlink:
        logger.info("files_retention_purged", batches=deleted_batches, files=len(to_unlink))
    return {"batches": deleted_batches, "files": len(to_unlink)}


async def retention_loop(stop_event: asyncio.Event, session_factory: Any) -> None:
    """Run :func:`purge_expired` every hour (and once at start)."""
    from ocabra.config import settings

    while not stop_event.is_set():
        try:
            await purge_expired(session_factory, int(settings.openai_files_retention_hours))
        except Exception as exc:  # noqa: BLE001 — keep the loop alive
            logger.warning("files_retention_loop_error", error=str(exc))
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=3600)
        except TimeoutError:
            continue
