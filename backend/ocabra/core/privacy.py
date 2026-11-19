"""Privacy modes: ``functional`` (the default) and ``zero_data``.

* **functional** — as private as possible without getting in the way of how
  things work: it may write things temporarily
  (batch input/output files, images served by URL) and cleans them up once they
  are no longer needed (``openai_files_retention_hours``, ``image_url_ttl_seconds``).
  Per-request metadata (who, which model, how long — never the content) goes to
  the statistics.
* **zero_data** — the node keeps nothing about the content of requests or who
  made them. Meant for nodes you don't fully control (a rented pod): no rows in
  the statistics, no user names in the live feed, no batches or files (they need
  storage), images inline instead of by URL, no external tracing, no content in
  logs. What it does *not* promise: confidentiality against the machine's owner,
  who can read memory and disk — that is what Secure Cloud is for.

See docs/tasks/cloud-storage-privacy-plan.md for the full inventory.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

import structlog

from ocabra.config import settings

logger = structlog.get_logger(__name__)

FUNCTIONAL = "functional"
ZERO_DATA = "zero_data"
MODES = (FUNCTIONAL, ZERO_DATA)
HEADER = "X-Ocabra-Privacy"


def mode() -> str:
    value = str(settings.privacy_mode or FUNCTIONAL).strip().lower()
    return value if value in MODES else FUNCTIONAL


def zero_data() -> bool:
    return mode() == ZERO_DATA


def preview(text: Any, limit: int = 200) -> str | None:
    """A bit of a response body for a log line — nothing in zero data."""
    if zero_data() or text is None:
        return None
    return str(text)[:limit]


class PrivacyHeaderMiddleware:
    """Pure ASGI: tell clients which privacy mode served them."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        header = (HEADER.lower().encode(), mode().replace("_", "-").encode())

        async def _send(message: dict) -> None:
            if message.get("type") == "http.response.start":
                message = {**message, "headers": [*message.get("headers", []), header]}
            await send(message)

        await self.app(scope, receive, _send)


async def require_storage_allowed() -> None:
    """FastAPI dependency: 403 on endpoints that need to store request data.

    Batches and ``/v1/files`` keep the input and output on disk until fetched;
    a zero-data node refuses them instead of quietly storing them.
    """
    if zero_data():
        from fastapi import HTTPException

        raise HTTPException(
            status_code=403,
            detail={
                "error": {
                    "message": (
                        "This node runs in zero-data mode: it stores nothing about requests, "
                        "so batches and file uploads are not available here."
                    ),
                    "type": "invalid_request_error",
                    "code": "zero_data_mode",
                }
            },
        )


# ── temporary files in memory (zero data) ───────────────────────

MEMORY_TEMP_DIR = "/dev/shm/ocabra-tmp"
_ORIGINAL_TMPDIR = os.environ.get("TMPDIR")


def temp_dir_status() -> dict[str, Any]:
    """Where temporary files go now, and how much room there is."""
    current = tempfile.gettempdir()
    try:
        st = os.statvfs(current)
        free_mb = st.f_bavail * st.f_frsize // (1024 * 1024)
    except OSError:
        free_mb = None
    return {
        "dir": current,
        "in_memory": current.startswith("/dev/shm"),
        "free_mb": free_mb,
    }


def apply_temp_dir() -> dict[str, Any]:
    """Point temporary files at RAM in zero data, back to the default otherwise.

    Uploads above 1 MB (FastAPI spools them), audio and video the workers
    process (VibeASR, FlashVSR, the upscaler) and anything else using
    ``tempfile`` go through ``TMPDIR``: on a zero-data node that is
    ``/dev/shm`` so request media never touches the disk. Workers are
    subprocesses and inherit it. If ``/dev/shm`` is small, a large upload
    fails instead of falling back to the disk — on purpose.
    """
    if zero_data() and os.path.isdir("/dev/shm"):
        try:
            os.makedirs(MEMORY_TEMP_DIR, mode=0o700, exist_ok=True)
            os.environ["TMPDIR"] = MEMORY_TEMP_DIR
            tempfile.tempdir = MEMORY_TEMP_DIR
        except OSError as exc:
            logger.warning("privacy_memory_tempdir_failed", error=str(exc))
    else:
        if _ORIGINAL_TMPDIR is None:
            os.environ.pop("TMPDIR", None)
        else:
            os.environ["TMPDIR"] = _ORIGINAL_TMPDIR
        tempfile.tempdir = None
    status = temp_dir_status()
    logger.info("privacy_temp_dir", mode=mode(), **status)
    if zero_data() and not status["in_memory"]:
        logger.warning("privacy_zero_data_temp_on_disk", **status)
    return status
