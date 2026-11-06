"""Fire-and-forget tasks must survive to completion (2026-11-06).

``asyncio.create_task`` leaves the event loop holding only a weak reference,
so a task nobody keeps a handle on can be garbage collected mid-flight. The
stats middleware did exactly that in eight places, and production lost request
rows: a ~55 minute gap in ``request_stats`` covering requests that were
observed in flight at the time, with no error logged anywhere.
"""

from __future__ import annotations

import asyncio
import gc

import pytest

from ocabra.core.background_tasks import pending_count, spawn


@pytest.mark.asyncio
async def test_keeps_a_strong_reference_while_running() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def work() -> None:
        started.set()
        await release.wait()

    before = pending_count()
    spawn(work())  # handle deliberately dropped, as every caller does
    await started.wait()
    assert pending_count() == before + 1

    release.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert pending_count() == before, "the reference must be released once done"


@pytest.mark.asyncio
async def test_completes_even_when_the_caller_keeps_nothing_and_gc_runs() -> None:
    """The actual failure mode: nobody holds the task and a collection cycle
    happens before it finishes."""
    done = asyncio.Event()

    async def work() -> None:
        await asyncio.sleep(0.01)
        done.set()

    spawn(work())
    gc.collect()
    await asyncio.wait_for(done.wait(), timeout=2)


@pytest.mark.asyncio
async def test_an_exception_is_logged_not_raised_into_the_void() -> None:
    """Without this the only sign of trouble is "Task exception was never
    retrieved" at interpreter shutdown, long after the useful context is gone."""

    async def boom() -> None:
        raise RuntimeError("fallo de prueba")

    before = pending_count()
    task = spawn(boom(), name="prueba")
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert task.done()
    assert isinstance(task.exception(), RuntimeError)
    assert pending_count() == before


@pytest.mark.asyncio
async def test_cancellation_is_not_reported_as_a_failure() -> None:
    async def work() -> None:
        await asyncio.sleep(10)

    before = pending_count()
    task = spawn(work())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)
    assert pending_count() == before
