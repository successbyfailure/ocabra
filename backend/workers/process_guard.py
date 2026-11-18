"""Make a worker's own child processes die with it.

The oCabra backend launches each worker in its own process group and stops
the whole group (see ``ocabra/backends/_process_group.py``). That covers an
orderly unload, but not a worker that dies on its own — OOM killer, segfault,
a stray SIGKILL — while its child (``vllm serve``, ``asr_stream_server``…)
keeps running and holding GPU memory with nobody to stop it.

``parent_death_preexec()`` returns a ``preexec_fn`` that asks the Linux kernel
(``prctl(PR_SET_PDEATHSIG)``) to send the child SIGTERM when the worker dies,
whatever the cause. SIGTERM rather than SIGKILL so the child can still stop
its own children (vllm's engine processes). No-op outside Linux. Needs no
privileges: it only concerns the calling process.
"""

from __future__ import annotations

import ctypes
import os
import signal
import sys
from collections.abc import Callable

_PR_SET_PDEATHSIG = 1

_prctl: Callable[..., int] | None = None
if sys.platform.startswith("linux"):
    try:
        # Resolved now, not after fork: the preexec_fn should do as little
        # as possible in the child.
        _prctl = ctypes.CDLL(None, use_errno=True).prctl
    except (OSError, AttributeError):
        _prctl = None


def parent_death_preexec(sig: int = signal.SIGTERM) -> Callable[[], None] | None:
    """``preexec_fn`` that ties the child's life to this process.

    Returns ``None`` where it is not supported, so it can be passed straight
    to ``subprocess``/``asyncio.create_subprocess_exec``.
    """
    if _prctl is None:
        return None
    prctl = _prctl
    parent_pid = os.getpid()

    def _preexec() -> None:
        prctl(_PR_SET_PDEATHSIG, int(sig), 0, 0, 0)
        # The parent may have died between fork() and prctl(): then the
        # signal would never come.
        if os.getppid() != parent_pid:
            os.kill(os.getpid(), int(sig))

    return _preexec
