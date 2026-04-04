"""Transparent HTTP streaming proxy and WebSocket bridge."""
from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import AsyncIterator

import httpx
import websockets
import websockets.exceptions
from fastapi import Request, WebSocket
from fastapi.responses import Response, StreamingResponse
from starlette.datastructures import Headers

from config import OCABRA_API_URL, TOUCH_INTERVAL_S

logger = logging.getLogger("gateway.proxy")

# Headers that must not be forwarded (hop-by-hop)
_HOP_BY_HOP = frozenset({
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade",
    "proxy-connection", "content-length",  # httpx re-calculates these
})

# Per-service throttle: service_id → last touch timestamp
_last_touch: dict[str, float] = {}

# Collapse repeated leading slashes to one (e.g. // → /)
_MULTI_SLASH_RE = re.compile(r"^/+")


def _normalize_path(path: str) -> str:
    return _MULTI_SLASH_RE.sub("/", path) if path else "/"


def _filter_headers(headers: Headers, *, extra_remove: set[str] | None = None) -> dict[str, str]:
    remove = _HOP_BY_HOP | (extra_remove or set())
    return {k: v for k, v in headers.items() if k.lower() not in remove}


async def touch_service(service_id: str) -> None:
    """Notify ocabra that the service is being used (throttled)."""
    now = time.monotonic()
    if now - _last_touch.get(service_id, 0) < TOUCH_INTERVAL_S:
        return
    _last_touch[service_id] = now
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            await client.post(f"{OCABRA_API_URL}/ocabra/services/{service_id}/touch")
    except Exception:
        pass  # touch is best-effort, never block the proxy


async def proxy_http(
    request: Request,
    upstream: str,
    service_id: str,
    *,
    public_url: str = "",
) -> StreamingResponse:
    """Proxy an HTTP request to upstream, streaming the response body."""
    path = _normalize_path(request.url.path)
    url = f"{upstream}{path}"
    if request.url.query:
        url = f"{url}?{request.url.query}"

    headers = _filter_headers(request.headers, extra_remove={"host"})
    # Tell the upstream who it really is
    headers["host"] = upstream.split("//", 1)[-1].split("/")[0]

    body = await request.body()

    asyncio.create_task(touch_service(service_id))

    upstream_base = upstream.rstrip("/")

    client = httpx.AsyncClient(
        # read=None: no timeout between chunks — required for long SSE streams
        # (Gradio 5 queue/data can be silent for minutes during 3D/video generation)
        timeout=httpx.Timeout(connect=10.0, read=None, write=30.0, pool=5.0),
        follow_redirects=False,
    )
    try:
        method = request.method
        current_url = url
        current_body = body

        # Follow redirects that point back to the same upstream internally.
        # Gradio/Uvicorn emits  307 Location: http://svc:port//  on GET /
        # which would loop forever if forwarded to the browser.
        for _ in range(10):
            req = client.build_request(
                method=method,
                url=current_url,
                headers=headers,
                content=current_body or None,
            )
            response = await client.send(req, stream=True)

            if response.status_code in (301, 302, 303, 307, 308):
                loc = response.headers.get("location", "")
                if loc.startswith(upstream_base):
                    # Internal redirect — follow silently
                    await response.aclose()
                    current_url = loc
                    if response.status_code == 303:
                        method = "GET"
                        current_body = b""
                    continue
            break
        else:
            await client.aclose()
            return Response(status_code=508, content=b"Too many internal redirects")

        resp_headers = _filter_headers(response.headers)

        # Rewrite Location headers: map internal upstream → public URL
        if public_url and "location" in resp_headers:
            loc = resp_headers["location"]
            if loc.startswith(upstream_base):
                # Absolute URL pointing to the internal upstream
                resp_headers["location"] = public_url.rstrip("/") + loc[len(upstream_base):]
            elif loc.startswith("//"):
                # Protocol-relative double-slash (Gradio redirect artefact) → collapse
                resp_headers["location"] = "/" + loc.lstrip("/")

        # For long-lived streaming responses (SSE, chunked) the gateway only
        # calls touch_service once at request start.  Spawn a background loop
        # that re-touches the service every TOUCH_INTERVAL_S so the idle timer
        # never fires while the client is still receiving data.
        _stream_done = asyncio.Event()

        async def _keep_alive_touch() -> None:
            while not _stream_done.is_set():
                try:
                    await asyncio.wait_for(_stream_done.wait(), timeout=TOUCH_INTERVAL_S)
                except (TimeoutError, asyncio.TimeoutError):
                    pass
                if not _stream_done.is_set():
                    await touch_service(service_id)

        _touch_task = asyncio.create_task(_keep_alive_touch())

        async def _body_then_close() -> AsyncIterator[bytes]:
            try:
                # aiter_raw() streams the raw (possibly compressed) bytes without
                # decompressing — required so content-encoding headers stay accurate.
                async for chunk in response.aiter_raw():
                    yield chunk
            finally:
                _stream_done.set()
                _touch_task.cancel()
                await response.aclose()
                await client.aclose()

        return StreamingResponse(
            _body_then_close(),
            status_code=response.status_code,
            headers=resp_headers,
            media_type=response.headers.get("content-type"),
        )
    except Exception:
        await client.aclose()
        raise


async def proxy_websocket(websocket: WebSocket, upstream: str, service_id: str) -> None:
    """Bridge a WebSocket connection bidirectionally to the upstream service."""
    path = _normalize_path(websocket.url.path)
    query = websocket.url.query
    ws_upstream = upstream.replace("http://", "ws://").replace("https://", "wss://")
    ws_url = f"{ws_upstream}{path}"
    if query:
        ws_url = f"{ws_url}?{query}"

    # Subprotocols requested by client
    requested = websocket.headers.get("sec-websocket-protocol", "")
    subprotocols = [s.strip() for s in requested.split(",") if s.strip()]

    # Accept the client WebSocket IMMEDIATELY so the browser handshake doesn't time out
    # while we connect to the upstream. Pick the first subprotocol if any were requested.
    await websocket.accept(subprotocol=subprotocols[0] if subprotocols else None)

    asyncio.create_task(touch_service(service_id))

    # Keep touching while the WebSocket is open (same idle-timer fix as HTTP SSE)
    _ws_done = asyncio.Event()

    async def _ws_keep_alive_touch() -> None:
        while not _ws_done.is_set():
            try:
                await asyncio.wait_for(_ws_done.wait(), timeout=TOUCH_INTERVAL_S)
            except (TimeoutError, asyncio.TimeoutError):
                pass
            if not _ws_done.is_set():
                await touch_service(service_id)

    _ws_touch_task = asyncio.create_task(_ws_keep_alive_touch())

    # Forward relevant headers (not hop-by-hop, not WS handshake headers)
    forward_headers = {
        k: v for k, v in websocket.headers.items()
        if k.lower() not in _HOP_BY_HOP
        and k.lower() not in {
            "host", "sec-websocket-key", "sec-websocket-version", "sec-websocket-extensions",
        }
    }

    try:
        async with websockets.connect(
            ws_url,
            additional_headers=forward_headers,
            subprotocols=subprotocols if subprotocols else None,  # type: ignore[arg-type]
            open_timeout=15,
            ping_interval=20,
            ping_timeout=30,
        ) as upstream_ws:

            async def client_to_upstream() -> None:
                try:
                    while True:
                        data = await websocket.receive()
                        if "bytes" in data:
                            await upstream_ws.send(data["bytes"])
                        elif "text" in data:
                            await upstream_ws.send(data["text"])
                        else:
                            break
                except Exception:
                    pass

            async def upstream_to_client() -> None:
                try:
                    async for message in upstream_ws:
                        if isinstance(message, bytes):
                            await websocket.send_bytes(message)
                        else:
                            await websocket.send_text(message)
                except Exception:
                    pass

            tasks = [
                asyncio.create_task(client_to_upstream()),
                asyncio.create_task(upstream_to_client()),
            ]
            _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass

    except websockets.exceptions.WebSocketException as exc:
        logger.debug("ws_upstream_error service=%s error=%s", service_id, exc)
        try:
            await websocket.close(code=1011, reason=str(exc)[:123])
        except Exception:
            pass
    except Exception as exc:
        logger.debug("ws_proxy_error service=%s error=%s", service_id, exc)
        try:
            await websocket.close(code=1011, reason=str(exc)[:123])
        except Exception:
            pass
    finally:
        _ws_done.set()
        _ws_touch_task.cancel()
