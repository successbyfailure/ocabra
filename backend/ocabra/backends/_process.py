"""Shared process lifecycle for backends that launch an HTTP worker (REF-02).

Several backends (diffusers, upscaler, flashvsr, mage, comfyui_video…) each
carried their own byte-for-byte copy of the same three steps: poll ``/health``
until the worker answers (bailing out early if the process died), check one
port, and stop the process. They also opened their log under ``/tmp`` — lost
when the container is recreated, and diffusers truncated it on every load.

This module factors that out into composable pieces:

- :func:`open_worker_log` — a persistent append log under
  ``settings.worker_log_dir`` (``/data/logs/workers`` by default), falling back
  to ``/tmp`` with a warning when that path can't be created or written.
- :func:`health_check_port` — one GET to ``127.0.0.1:<port>/health`` over the
  shared health client (see ``_health_http``).
- :func:`wait_until_healthy` — the health poll with early-exit on process death.
- :func:`stop_process` — SIGTERM → grace → SIGKILL on the worker's *group*
  (via ``_process_group``), so children the worker spawned (ffmpeg, …) die with
  it instead of being orphaned. Launch the worker with
  ``start_new_session=True`` for this to reach the whole group.

Backends keep their own ``create_subprocess_exec`` call (the command and env
are backend-specific); they pass ``stderr`` from :func:`open_worker_log`,
``start_new_session=True``, and use these helpers for the wait and the stop.
The per-backend health timeout and stop grace stay explicit arguments, so each
backend preserves its own effective values instead of a hidden default.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import BinaryIO

import structlog

from ocabra.backends._health_http import health_client
from ocabra.backends._process_group import terminate_process_group
from ocabra.config import settings

logger = structlog.get_logger(__name__)

_FALLBACK_LOG_DIR = Path("/tmp")


def worker_log_dir() -> Path:
    """Directory for persistent worker logs.

    ``settings.worker_log_dir`` when it can be created and written; otherwise
    ``/tmp`` with a one-line warning, so a read-only ``/data`` never stops a
    worker from launching.
    """
    configured = Path(settings.worker_log_dir)
    try:
        configured.mkdir(parents=True, exist_ok=True)
        if os.access(configured, os.W_OK):
            return configured
        reason = "not writable"
    except OSError as exc:
        reason = str(exc)
    logger.warning(
        "worker_log_dir_unavailable",
        path=str(configured),
        reason=reason,
        fallback=str(_FALLBACK_LOG_DIR),
    )
    return _FALLBACK_LOG_DIR


def open_worker_log(name: str) -> tuple[BinaryIO, str]:
    """Open (append) the persistent log file for a worker.

    ``name`` is a logical worker name (e.g. ``"diffusers_worker_<model_id>"``);
    slashes are flattened so a canonical ``<backend>/<model>`` id is safe.
    Returns ``(file object, absolute path)``; the caller hands the file to
    ``create_subprocess_exec(stderr=...)`` which owns and closes the fd.
    """
    safe = name.replace("/", "_")
    path = worker_log_dir() / f"{safe}.log"
    handle = open(path, "ab")  # noqa: ASYNC230,SIM115 — long-lived fd owned by the child process
    return handle, str(path)


async def health_check_port(
    port: int, *, path: str = "/health", ok_key: str | None = "ok"
) -> bool:
    """True when ``127.0.0.1:<port><path>`` answers 200 (and ``ok_key`` is truthy).

    ``ok_key=None`` accepts any 200 without reading the body. Any error (the
    worker not up yet, a connection refused) is False, never an exception.
    """
    try:
        client = health_client()
        response = await client.get(f"http://127.0.0.1:{port}{path}", timeout=2.0)
        if response.status_code != 200:
            return False
        if ok_key is None:
            return True
        return bool(response.json().get(ok_key, False))
    except Exception:  # noqa: BLE001 — a health probe never raises
        return False


async def wait_until_healthy(
    *,
    port: int,
    process: asyncio.subprocess.Process,
    timeout_s: float,
    label: str,
    path: str = "/health",
    ok_key: str | None = "ok",
) -> None:
    """Poll ``/health`` until the worker answers, bailing out if it dies first.

    Raises ``RuntimeError`` as soon as the process exits (so a crash costs no
    more than the next poll, not the whole timeout), and ``TimeoutError`` when
    ``timeout_s`` elapses with no healthy answer.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.returncode is not None:
            raise RuntimeError(f"{label} worker exited with code {process.returncode}")
        if await health_check_port(port, path=path, ok_key=ok_key):
            return
        await asyncio.sleep(1)
    raise TimeoutError(f"{label} worker health check timed out after {timeout_s}s")


async def stop_process(process: asyncio.subprocess.Process, *, grace_s: float = 30.0) -> None:
    """Stop a worker and everything in its process group.

    SIGTERM the group, wait up to ``grace_s`` for it to exit, then SIGKILL.
    Only correct for processes launched with ``start_new_session=True`` (so the
    pid is the group id); see ``_process_group``. A no-op if already exited.
    """
    await terminate_process_group(process, grace_s=grace_s)
