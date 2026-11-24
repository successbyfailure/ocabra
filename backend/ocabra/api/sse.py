"""Server-Sent Events helpers shared by the streaming OpenAI/Ollama endpoints.

Keepalive framing during cold-start loads and the oCabra load-status *comment*
frames. Kept in the ``api`` layer (handlers emit these); no ``core`` code uses
them.
"""

from __future__ import annotations

import asyncio
import json

SSE_KEEPALIVE_INTERVAL_S = 10.0


async def keepalive_until_done(task: asyncio.Task, interval: float = SSE_KEEPALIVE_INTERVAL_S):
    """Yield ``b": keepalive\\n\\n"`` every *interval* seconds until *task* finishes.

    Mirrors the pattern from :mod:`ocabra.agents.executor`: shields the task
    so the wait_for timeout doesn't cancel it, emits a comment-line keepalive
    on each timeout to keep proxies (NPM, Cloudflare, ...) from closing the
    connection during long model loads, and exits when the task completes —
    leaving the caller to inspect ``task.result()`` / ``task.exception()``.
    """
    while not task.done():
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=interval)
        except TimeoutError:
            yield b": keepalive\n\n"
        except Exception:
            # The caller owns error serialization.  Do not let a failed load
            # escape from the keepalive iterator after the SSE response has
            # already started; leave it to inspect task.exception().
            return


def sse_ocabra_event(event: str, **fields: object) -> bytes:
    """Encode an oCabra load-status hint as an SSE *comment* frame.

    Previously emitted a named ``event: ocabra.<event>`` frame with a JSON
    ``data:`` payload. Strict OpenAI clients don't filter by event name — they
    parse every ``data:`` line as a ``ChatCompletionChunk`` and choke on the
    non-chunk JSON. A comment line (starting with ``:``) is ignored by every
    spec-compliant SSE parser (the OpenAI SDK included), so it can't be
    mis-parsed, while still forcing a header flush and keeping the connection
    alive during cold starts. oCabra-aware clients may read the JSON after the
    ``:``; status is also available out-of-band via the ``X-Ocabra-*`` headers,
    ``GET /v1/models/{id}`` (``ocabra`` extension) and
    ``GET /ocabra/models/{id}/status``.
    """
    payload = {"event": f"ocabra.{event}", **fields}
    encoded = json.dumps(payload, ensure_ascii=False, default=str)
    return f": {encoded}\n\n".encode()
