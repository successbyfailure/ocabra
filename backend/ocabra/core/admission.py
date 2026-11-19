"""Admission helpers shared by the stats middleware (BUG-37, BUG-38).

The in-flight cap and maintenance mode still run inside
``ocabra.stats.collector.StatsMiddleware`` (REF-08 is the move out); this
module holds the pieces that do not depend on the stats code:

* the 503/429 responses, so HTTP, multipart and WebSocket rejections say the
  same thing;
* reading the ``model`` field of a multipart upload without reading the file
  (the body is replayed untouched to the endpoint);
* turning a response into a WebSocket handshake denial.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi.responses import JSONResponse

Message = dict[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]

# Media types whose body is never JSON: the middleware does not read them.
_BINARY_PREFIXES = ("multipart/", "audio/", "video/", "image/", "application/octet-stream")

# How much of a multipart body may be read looking for the ``model`` field
# before giving up. The text fields come first in what the usual clients send
# (httpx/openai-python put ``data`` before ``files``); a field that only shows
# up after a file is not looked for, the file is never read here.
MULTIPART_PEEK_LIMIT = 64 * 1024
_MAX_FIELD_VALUE = 512

_BOUNDARY_RE = re.compile(r'boundary=(?:"([^"]+)"|([^;\s]+))', re.IGNORECASE)
_NAME_RE = re.compile(rb'\bname="([^"]*)"', re.IGNORECASE)
_FILENAME_RE = re.compile(rb"\bfilename\*?=", re.IGNORECASE)


def body_is_binary(content_type: str) -> bool:
    """True for bodies the middleware must not buffer (uploads, media)."""
    ctype = (content_type or "").strip().lower()
    return ctype.startswith(_BINARY_PREFIXES)


def is_multipart(content_type: str) -> bool:
    return (content_type or "").strip().lower().startswith("multipart/form-data")


def maintenance_response(model_id: str) -> JSONResponse:
    """503 for a model under maintenance (``ModelManager.set_maintenance``)."""
    return JSONResponse(
        status_code=503,
        headers={"Retry-After": "30"},
        content={
            "error": {
                "message": f"Model '{model_id}' is under maintenance. Please retry shortly.",
                "type": "server_error",
                "code": "model_maintenance",
            }
        },
    )


def capacity_retry_after(effective_cap: int) -> int:
    """A low-concurrency worker (e.g. a single-slot Ollama model) drains its
    queue far slower than the flat 2 s this used to suggest unconditionally —
    tell low-slot callers to back off longer."""
    return 2 if effective_cap > 4 else 10


def capacity_response(model_id: str, effective_cap: int) -> JSONResponse:
    """429 for a worker already at its in-flight cap."""
    return JSONResponse(
        status_code=429,
        headers={"Retry-After": str(capacity_retry_after(effective_cap))},
        content={
            "error": {
                "message": (
                    f"Model '{model_id}' is at capacity "
                    f"({effective_cap} concurrent requests for this "
                    "worker). Please retry shortly."
                ),
                "type": "rate_limit_error",
                "code": "model_at_capacity",
            }
        },
    )


def _multipart_boundary(content_type: str) -> bytes | None:
    match = _BOUNDARY_RE.search(content_type or "")
    if not match:
        return None
    value = match.group(1) or match.group(2)
    return value.encode("latin-1", errors="ignore") or None


def _scan_multipart(data: bytes, boundary: bytes, field: str) -> tuple[str | None, bool]:
    """Look for ``field`` in the parts already received.

    Returns ``(value, done)``: ``done`` once the value is found, a file part
    starts before it, or the body is malformed; ``(None, False)`` means "need
    more bytes".
    """
    delimiter = b"--" + boundary
    wanted = field.encode()
    pos = data.find(delimiter)
    if pos < 0:
        return None, False
    while True:
        pos += len(delimiter)
        if data[pos : pos + 2] == b"--":
            return None, True  # closing delimiter: no such field
        headers_end = data.find(b"\r\n\r\n", pos)
        if headers_end < 0:
            return None, False
        headers = data[pos:headers_end]
        name = _NAME_RE.search(headers)
        if _FILENAME_RE.search(headers):
            # A file part: what follows is the upload itself.
            return None, True
        value_start = headers_end + 4
        next_delim = data.find(b"\r\n" + delimiter, value_start)
        if name is not None and name.group(1) == wanted:
            if next_delim < 0:
                if len(data) - value_start > _MAX_FIELD_VALUE:
                    return None, True
                return None, False
            raw = data[value_start:next_delim]
            if len(raw) > _MAX_FIELD_VALUE:
                return None, True
            value = raw.decode("utf-8", errors="ignore").strip()
            return (value or None), True
        if next_delim < 0:
            return None, False
        pos = next_delim + 2


async def peek_multipart_field(
    receive: Receive,
    content_type: str,
    field: str = "model",
    *,
    limit: int = MULTIPART_PEEK_LIMIT,
) -> tuple[str | None, Receive]:
    """Read just enough of a multipart body to find one text field.

    Returns the value (or None) and a ``receive`` that replays the messages
    already read, unchanged, before handing over to the original one — the
    endpoint parses the full upload as if nothing had looked at it. Reads at
    most ``limit`` bytes plus one ASGI message, and stops at the first file
    part, so an audio or image upload is never buffered here.
    """
    boundary = _multipart_boundary(content_type)
    if boundary is None:
        return None, receive

    buffered: list[Message] = []
    data = bytearray()
    value: str | None = None
    while True:
        message = await receive()
        buffered.append(message)
        if message.get("type") != "http.request":
            break  # disconnect: replay it and let the endpoint see it
        data += message.get("body", b"") or b""
        found, done = _scan_multipart(bytes(data), boundary, field)
        if done:
            value = found
            break
        if not message.get("more_body", False) or len(data) >= limit:
            break
    del data

    async def replay() -> Message:
        if buffered:
            return buffered.pop(0)
        return await receive()

    return value, replay


async def send_websocket_denial(
    response: JSONResponse, scope: dict, receive: Receive, send: Send
) -> None:
    """Reject a WebSocket handshake with ``response`` (status, headers, body).

    Uses the ASGI WebSocket Denial Response extension (uvicorn has it), so the
    client gets the same 503/429 and ``Retry-After`` an HTTP request would.
    Without the extension the handshake is closed before ``accept``, which
    the server answers with a 403.
    """
    if "websocket.http.response" in (scope.get("extensions") or {}):
        await response(scope, receive, send)
        return
    await send({"type": "websocket.close", "code": 1013, "reason": "try again later"})
