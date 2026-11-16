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

from typing import Any

from ocabra.config import settings

FUNCTIONAL = "functional"
ZERO_DATA = "zero_data"
MODES = (FUNCTIONAL, ZERO_DATA)
HEADER = "X-Ocabra-Privacy"


def mode() -> str:
    value = str(getattr(settings, "privacy_mode", FUNCTIONAL) or FUNCTIONAL).strip().lower()
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
