"""Regression test: StatsMiddleware must NOT buffer audio/video streams.

Before 2026-10-15, ``is_streaming`` only checked for SSE / NDJSON /
``isinstance(response, StreamingResponse)``. The last check failed inside
``BaseHTTPMiddleware`` because Starlette wraps every downstream response in
a private ``_StreamingResponse`` type — ``isinstance`` returned False, the
middleware fell through to the non-streaming branch and read
``response.body``, which drained the entire upstream iterator in one go.

Visible symptom: ``/v1/audio/speech`` returned ``Content-Length`` + delivered
every byte at the end (TTFB == total duration). This test boots a FastAPI
app with the real StatsMiddleware and verifies that:

1. The response headers use ``Transfer-Encoding: chunked`` (not Content-Length).
2. Chunks arrive at the client as the generator yields them (TTFB « total).
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from ocabra.stats.collector import StatsMiddleware


def _make_app(media_type: str) -> FastAPI:
    app = FastAPI()
    # StatsMiddleware reads request.app.state.model_manager; a Mock is fine
    # — the middleware only calls ``try_begin_request`` / ``end_request``
    # for the fast path, and both return None/no-op on a Mock.
    app.state.model_manager = MagicMock()
    app.state.model_manager.try_begin_request = MagicMock(return_value=None)
    app.add_middleware(StatsMiddleware)

    @app.post("/v1/audio/speech")
    async def speech():
        async def gen():
            for i in range(4):
                await asyncio.sleep(0.1)
                yield f"chunk{i}\n".encode()
        return StreamingResponse(gen(), media_type=media_type)

    return app


async def _serve_and_hit(media_type: str, port: int) -> tuple[dict[str, str], list[float]]:
    app = _make_app(media_type)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    task = asyncio.create_task(server.serve())
    try:
        await asyncio.sleep(0.6)  # let uvicorn bind
        async with httpx.AsyncClient(timeout=10.0) as client:
            t0 = time.monotonic()
            arrivals: list[float] = []
            async with client.stream("POST", f"http://127.0.0.1:{port}/v1/audio/speech") as r:
                headers = dict(r.headers)
                async for _chunk in r.aiter_bytes():
                    arrivals.append(time.monotonic() - t0)
        return headers, arrivals
    finally:
        server.should_exit = True
        await task


@pytest.mark.asyncio
@pytest.mark.parametrize("media_type", ["audio/mpeg", "audio/pcm", "audio/wav", "video/mp4"])
async def test_middleware_does_not_buffer_streaming_media(media_type: str) -> None:
    """The 4 arrivals should be spread over ~400 ms, not all land at the end."""
    # Different port per parametrisation so a timing-flaky teardown doesn't
    # cross-contaminate subsequent iterations.
    port = 19200 + abs(hash(media_type)) % 100
    headers, arrivals = await _serve_and_hit(media_type, port)

    assert headers.get("transfer-encoding") == "chunked", (
        f"{media_type}: expected chunked, got {headers}"
    )
    assert "content-length" not in headers, (
        f"{media_type}: non-chunked content-length means response was buffered: {headers}"
    )
    assert len(arrivals) >= 2, f"{media_type}: expected multiple chunks, got {arrivals}"
    # First chunk should arrive well before the last — classical "stream in,
    # stream out" shape. If the middleware buffered, every arrival would land
    # at ~0.4 s (all 4 chunks delivered after the generator finished).
    assert arrivals[0] < arrivals[-1] - 0.1, (
        f"{media_type}: all chunks arrived within <100ms, buffering suspected: {arrivals}"
    )
