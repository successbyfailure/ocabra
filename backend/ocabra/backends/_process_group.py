"""Stop a worker together with everything it launched.

Workers that start processes of their own (voxtral's ``vllm serve``, vibeasr's
``asr_stream_server``, the ffmpeg that ``whisper-server --convert`` spawns…)
used to be stopped with ``terminate()``/``kill()`` on the worker alone. When
that reached SIGKILL the worker died without a chance to stop its children, and
they kept running — holding GPU memory — with nobody left to stop them.

The fix is the one vllm/strata/tensorrt already use: launch the worker with
``start_new_session=True`` (its pid becomes the process-group id) and signal
the whole group. :func:`terminate_process_group` does the SIGTERM → grace →
SIGKILL dance on the group, and also sweeps leftovers when the worker itself
is already gone (the group outlives its leader as long as any member lives,
and the kernel does not hand that id out again while it is in use).

Only call it on processes launched with ``start_new_session=True``: it assumes
pgid == pid.
"""

from __future__ import annotations

import asyncio
import os
import signal
from contextlib import suppress
from typing import Any

_POLL_S = 0.1


def _signal_group(pgid: int, sig: int) -> bool:
    """Send ``sig`` to the group; False when the group no longer exists."""
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Not ours (should not happen: we only signal groups we created).
        return False
    return True


def _group_alive(pgid: int) -> bool:
    return _signal_group(pgid, 0)


async def _wait_group_gone(process: Any, pgid: int, timeout_s: float) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + max(0.0, timeout_s)
    if process.returncode is None:
        try:
            await asyncio.wait_for(process.wait(), timeout=max(0.0, timeout_s))
        except TimeoutError:
            return False
    while _group_alive(pgid):
        if loop.time() >= deadline:
            return False
        await asyncio.sleep(_POLL_S)
    return True


async def terminate_process_group(
    process: Any,
    *,
    grace_s: float,
    graceful: bool = True,
) -> None:
    """Stop ``process`` and every process in its group.

    ``graceful``: SIGTERM the group and give it ``grace_s`` seconds (leader
    and stragglers) before SIGKILL. Without it, SIGKILL straight away.
    Safe to call when the leader already exited: leftover group members
    are still swept.
    """
    pid = getattr(process, "pid", None)
    if not isinstance(pid, int) or pid <= 1:
        # Defensive: no usable pgid, fall back to the single process.
        if process.returncode is None:
            with suppress(ProcessLookupError):
                process.kill()
            await process.wait()
        return

    pgid = pid
    if graceful:
        if not _signal_group(pgid, signal.SIGTERM):
            if process.returncode is None:
                await process.wait()
            return
        if await _wait_group_gone(process, pgid, grace_s):
            return

    _signal_group(pgid, signal.SIGKILL)
    if process.returncode is None:
        with suppress(ProcessLookupError):
            await process.wait()
