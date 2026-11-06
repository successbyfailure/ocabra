"""Fire-and-forget asyncio tasks that actually survive to completion.

``asyncio.create_task`` returns a task the event loop only holds a *weak*
reference to. If the caller drops the handle — the usual shape for
"record this and move on" work — the task can be garbage collected mid-flight
and its coroutine silently never finishes. The CPython docs call this out
explicitly, and it is not theoretical: request stats went missing in
production on 2026-11-06 (a ~55 minute gap in ``request_stats`` covering
requests observed in flight at the time), which is exactly the shape of the
bug — non-deterministic loss under load, no error anywhere.

Keeping a strong reference until the task finishes is the documented fix.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Strong references to in-flight background work. Entries remove themselves on
# completion, so this stays bounded by how much is genuinely running.
_PENDING: set[asyncio.Task[Any]] = set()


def spawn(coro: Coroutine[Any, Any, Any], *, name: str | None = None) -> asyncio.Task[Any]:
    """Schedule ``coro`` and keep it alive until it completes.

    Use instead of a bare ``asyncio.create_task`` whenever nobody awaits the
    result. Exceptions are logged rather than raised into a dangling task,
    where they would otherwise surface only as "Task exception was never
    retrieved" at interpreter shutdown.
    """
    task = asyncio.create_task(coro, name=name)
    _PENDING.add(task)

    def _done(finished: asyncio.Task[Any]) -> None:
        _PENDING.discard(finished)
        if finished.cancelled():
            return
        exc = finished.exception()
        if exc is not None:
            logger.warning(
                "background_task_failed", task=name or finished.get_name(), error=str(exc)
            )

    task.add_done_callback(_done)
    return task


def pending_count() -> int:
    """How many background tasks are currently in flight (for tests/diagnostics)."""
    return len(_PENDING)
