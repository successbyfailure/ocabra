"""Reverse tunnel for federation: one WebSocket, requests in both directions.

Only one side of a federation link needs a reachable URL. The node *without*
one (a RunPod pod, a home box behind NAT) dials the node that has one and keeps
the socket open. From then on the link is **symmetric**: either side can send
requests to the other over that socket, so a mutual ("swarm") federation needs
no URL, port or API key exchange on the dialing side.

* :class:`TunnelConnection` owns one live socket. As a *client* it multiplexes
  our outgoing requests (exposed as an ``httpx`` transport,
  :class:`TunnelTransport`, so :class:`~ocabra.core.federation.FederationManager`
  keeps using a plain ``httpx.AsyncClient`` for the peer: heartbeat, proxy_*,
  remote ops unchanged). As a *server* it executes the other side's requests
  against **our own local API** with a local service API key (see
  :class:`LocalServer`), so our normal auth/ACL decides what the peer can do.
* :class:`TunnelDialer` is the reconnecting loop of the side that dials.
* :func:`accept_tunnel` is the handshake of the side that accepts.

Stream ids: the dialer allocates **even** ids, the acceptor **odd** ones, so a
frame's id says whether it belongs to a request we sent (response frames) or
one we are serving (request frames).

Wire format — every WebSocket message is one binary frame::

    [type: u8][stream_id: u32 big-endian][payload]

Control frames (HELLO/WELCOME/REJECT) use stream 0 and a JSON payload. A
request is ``REQ_HEAD`` (JSON) → ``REQ_BODY``* → ``REQ_END``; its response is
``RESP_HEAD`` (JSON) → ``RESP_BODY``* → ``RESP_END``, or ``ERROR`` (JSON) at any
point. ``CANCEL`` (requester → server) aborts a stream whose client went away.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import random
import struct
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx
import structlog

logger = structlog.get_logger(__name__)

PROTOCOL_VERSION = 1

# Frame types
HELLO = 0x01
WELCOME = 0x02
REJECT = 0x03
REQ_HEAD = 0x10
REQ_BODY = 0x11
REQ_END = 0x12
RESP_HEAD = 0x20
RESP_BODY = 0x21
RESP_END = 0x22
ERROR = 0x30
CANCEL = 0x31

_HEADER = struct.Struct(">BI")
# Bodies are split so one big upload/stream can't monopolise the socket.
CHUNK_SIZE = 64 * 1024
# Max WebSocket message: one chunk plus the frame header, with headroom for the
# JSON of a REQ_HEAD with many headers.
MAX_MESSAGE_SIZE = 1024 * 1024
# Per-stream buffered frames before the reader starts applying backpressure
# (which, being a single socket, pauses every stream: acceptable for v1).
_STREAM_QUEUE_MAX = 256

# Headers that describe *this* hop and must not be replayed on the next one.
_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-connection",
        "transfer-encoding",
        "te",
        "trailer",
        "upgrade",
        "host",
    }
)


class TunnelClosed(Exception):
    """The tunnel's WebSocket closed (or was never open)."""


def encode_frame(frame_type: int, stream_id: int, payload: bytes = b"") -> bytes:
    """Serialise one tunnel frame."""
    return _HEADER.pack(frame_type, stream_id) + payload


def decode_frame(data: bytes) -> tuple[int, int, bytes]:
    """Parse one tunnel frame into ``(type, stream_id, payload)``."""
    if len(data) < _HEADER.size:
        raise ValueError("tunnel frame too short")
    frame_type, stream_id = _HEADER.unpack_from(data)
    return frame_type, stream_id, bytes(data[_HEADER.size :])


def _json(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, separators=(",", ":")).encode()


def _split(data: bytes) -> list[bytes]:
    return [data[i : i + CHUNK_SIZE] for i in range(0, len(data), CHUNK_SIZE)] or []


# ── Channel abstraction ──────────────────────────────────────────


class Channel(Protocol):
    """Minimal bidirectional binary message channel (a WebSocket)."""

    async def send(self, data: bytes) -> None: ...

    async def recv(self) -> bytes:
        """Return the next message; raise :class:`TunnelClosed` when closed."""
        ...

    async def close(self, code: int = 1000, reason: str = "") -> None: ...


class StarletteChannel:
    """Adapter for an accepted ``starlette.websockets.WebSocket`` (hub side)."""

    def __init__(self, websocket: Any) -> None:
        self._ws = websocket

    async def send(self, data: bytes) -> None:
        try:
            await self._ws.send_bytes(data)
        except Exception as exc:  # noqa: BLE001 — any send failure means the socket is gone
            raise TunnelClosed(str(exc) or type(exc).__name__) from exc

    async def recv(self) -> bytes:
        from starlette.websockets import WebSocketDisconnect

        try:
            message = await self._ws.receive()
        except WebSocketDisconnect as exc:
            raise TunnelClosed(f"disconnect {exc.code}") from exc
        except RuntimeError as exc:
            raise TunnelClosed(str(exc)) from exc
        if message.get("type") == "websocket.disconnect":
            raise TunnelClosed(f"disconnect {message.get('code')}")
        data = message.get("bytes")
        if data is None:
            text = message.get("text")
            if text is None:
                raise TunnelClosed("empty websocket message")
            data = text.encode()
        return data

    async def close(self, code: int = 1000, reason: str = "") -> None:
        with contextlib.suppress(Exception):
            await self._ws.close(code=code, reason=reason)


class WebsocketsChannel:
    """Adapter for a ``websockets`` client connection (agent side)."""

    def __init__(self, connection: Any) -> None:
        self._conn = connection

    async def send(self, data: bytes) -> None:
        from websockets.exceptions import ConnectionClosed

        try:
            await self._conn.send(data)
        except ConnectionClosed as exc:
            raise TunnelClosed(str(exc)) from exc

    async def recv(self) -> bytes:
        from websockets.exceptions import ConnectionClosed

        try:
            data = await self._conn.recv()
        except ConnectionClosed as exc:
            raise TunnelClosed(str(exc)) from exc
        return data.encode() if isinstance(data, str) else data

    async def close(self, code: int = 1000, reason: str = "") -> None:
        with contextlib.suppress(Exception):
            await self._conn.close(code=code, reason=reason)


class MemoryChannel:
    """In-process channel pair, for tests."""

    def __init__(self) -> None:
        self._inbox: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._peer: MemoryChannel | None = None
        self.closed = False

    @classmethod
    def pair(cls) -> tuple[MemoryChannel, MemoryChannel]:
        a, b = cls(), cls()
        a._peer, b._peer = b, a
        return a, b

    async def send(self, data: bytes) -> None:
        if self.closed or self._peer is None or self._peer.closed:
            raise TunnelClosed("memory channel closed")
        await self._peer._inbox.put(bytes(data))

    async def recv(self) -> bytes:
        if self.closed:
            raise TunnelClosed("memory channel closed")
        data = await self._inbox.get()
        if data is None:
            raise TunnelClosed("memory channel closed")
        return data

    async def close(self, code: int = 1000, reason: str = "") -> None:
        if self.closed:
            return
        self.closed = True
        self._inbox.put_nowait(None)
        if self._peer is not None and not self._peer.closed:
            self._peer.closed = True
            self._peer._inbox.put_nowait(None)


# ── Connection ───────────────────────────────────────────────────

_CLOSED = object()

# Header added to every request we serve for a peer, so the local API knows it
# came through the federation and must not be forwarded to yet another peer
# (see ``federation_hop`` in core/federation.py).
FEDERATION_HOP_HEADER = "x-ocabra-federation-hop"


@dataclass
class _ClientStream:
    queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(_STREAM_QUEUE_MAX))
    finished: bool = False


@dataclass
class _ServedStream:
    head: dict[str, Any]
    body: bytearray = field(default_factory=bytearray)
    task: asyncio.Task | None = None


# What a peer can ask our local API for. The service key's role is the real
# access control; this is defence in depth so a peer can never reach who-can-
# access-what (users, groups, keys, federation) whatever the key's role.
_PATHS_INFERENCE = ("/v1/", "/api/", "/ocabra/federation/heartbeat")
# Stripped before replaying the peer's request on our API: its credentials mean
# nothing here, the service key is injected instead.
_SERVER_STRIP = _HOP_BY_HOP | {
    "authorization",
    "cookie",
    "content-length",
    "x-api-key",
    "x-gateway-token",
    "x-ocabra-internal-token",
    "x-internal-user-id",
    "x-internal-key-name",
}


# "full" (swarm: full control) reaches all of our API — the other node's admins
# can manage this one from their own UI — except who-can-access-what and the
# federation itself, which stay local decisions.
_FULL_DENY = (
    "/ocabra/auth",
    "/ocabra/users",
    "/ocabra/groups",
    "/ocabra/federation",
    "/ocabra/cloud",
    "/ocabra/host",
    "/ocabra/ws",
)
_FULL_READ_ONLY = ("/ocabra/config",)


def _under(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith(prefix + "/")


def path_allowed(path: str, access_level: str, method: str = "GET") -> bool:
    """Whether a tunnelled request may reach *path* on our local API."""
    clean = path.split("?", 1)[0]
    if ".." in clean or "//" in clean:
        return False
    if access_level == "full":
        if clean == "/ocabra/federation/heartbeat":
            return True
        if any(_under(clean, p) for p in _FULL_DENY):
            return False
        if any(_under(clean, p) for p in _FULL_READ_ONLY):
            return method.upper() == "GET"
        return any(clean.startswith(p) for p in ("/v1/", "/api/", "/ocabra/")) or clean in (
            "/health",
            "/ready",
            "/version",
        )
    return any(clean == p.rstrip("/") or clean.startswith(p) for p in _PATHS_INFERENCE)


@dataclass
class LocalServer:
    """How we serve a peer's requests: our API, a service key, an access level.

    ``api_key`` is read per request so rotating it applies to a live tunnel;
    ``None`` means we don't serve this peer (one-way link) → 403.
    """

    client: httpx.AsyncClient
    api_key: Callable[[], str | None]
    access_level: Callable[[], str]
    node_id: str = ""
    # «Mi servidor, mis reglas» (core/federation_rules.py): (method, path, body)
    # → (status, body) to refuse the request, or None to serve it.
    gate: Callable[[str, str, bytes], Awaitable[tuple[int, dict[str, Any]] | None]] | None = None


class TunnelConnection:
    """One live tunnel: our requests out, the peer's requests in."""

    def __init__(
        self,
        channel: Channel,
        *,
        dialer: bool,
        remote_info: dict[str, Any] | None = None,
        server: LocalServer | None = None,
    ) -> None:
        self._channel = channel
        self.dialer = dialer
        self.remote_info: dict[str, Any] = dict(remote_info or {})
        self._server = server
        self._parity = 0 if dialer else 1
        self._next_stream_id = 2 if dialer else 1
        self._client_streams: dict[int, _ClientStream] = {}
        self._served: dict[int, _ServedStream] = {}
        self._send_lock = asyncio.Lock()
        self._closed = asyncio.Event()
        self.connected_at = time.time()
        self.requests_served = 0
        self.requests_sent = 0

    # -- state

    @property
    def closed(self) -> bool:
        return self._closed.is_set()

    @property
    def active_streams(self) -> int:
        return len(self._client_streams) + len(self._served)

    async def wait_closed(self) -> None:
        await self._closed.wait()

    async def send_frame(self, frame_type: int, stream_id: int, payload: bytes = b"") -> None:
        if self.closed:
            raise TunnelClosed("tunnel closed")
        async with self._send_lock:
            await self._channel.send(encode_frame(frame_type, stream_id, payload))

    # -- client role

    def open_stream(self) -> tuple[int, _ClientStream]:
        if self.closed:
            raise TunnelClosed("tunnel closed")
        stream_id = self._next_stream_id
        self._next_stream_id += 2
        if self._next_stream_id > 0xFFFFFFF0:
            self._next_stream_id = 2 if self.dialer else 1
        stream = _ClientStream()
        self._client_streams[stream_id] = stream
        self.requests_sent += 1
        return stream_id, stream

    def release_stream(self, stream_id: int) -> None:
        self._client_streams.pop(stream_id, None)

    # -- main loop

    async def run(self) -> None:
        """Read frames until the channel closes, dispatching them by stream id."""
        try:
            while True:
                data = await self._channel.recv()
                try:
                    frame_type, stream_id, payload = decode_frame(data)
                except ValueError:
                    logger.warning("federation_tunnel_bad_frame", size=len(data))
                    continue
                if stream_id == 0:
                    continue  # no in-session control frames yet
                if stream_id % 2 == self._parity:
                    stream = self._client_streams.get(stream_id)
                    if stream is not None:
                        await stream.queue.put((frame_type, payload))
                    # else: late frames for a cancelled/finished stream — drop
                else:
                    self._on_served_frame(frame_type, stream_id, payload)
        except TunnelClosed:
            pass
        finally:
            await self.close()

    async def close(self, code: int = 1000, reason: str = "") -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        for stream in list(self._client_streams.values()):
            with contextlib.suppress(asyncio.QueueFull):
                stream.queue.put_nowait((_CLOSED, b""))
        for served in list(self._served.values()):
            if served.task is not None:
                served.task.cancel()
        self._served.clear()
        await self._channel.close(code=code, reason=reason)

    # -- server role

    def _on_served_frame(self, frame_type: int, stream_id: int, payload: bytes) -> None:
        if frame_type == REQ_HEAD:
            try:
                head = json.loads(payload)
            except Exception:  # noqa: BLE001
                head = {}
            self._served[stream_id] = _ServedStream(head=head if isinstance(head, dict) else {})
        elif frame_type == REQ_BODY:
            st = self._served.get(stream_id)
            if st is not None:
                st.body.extend(payload)
        elif frame_type == REQ_END:
            st = self._served.get(stream_id)
            if st is not None and st.task is None:
                st.task = asyncio.create_task(self._serve(stream_id, st))
        elif frame_type == CANCEL:
            st = self._served.pop(stream_id, None)
            if st is not None and st.task is not None:
                st.task.cancel()

    async def _reply(self, stream_id: int, status: int, body: dict[str, Any]) -> None:
        await self.send_frame(
            RESP_HEAD,
            stream_id,
            _json({"status": status, "headers": [["content-type", "application/json"]]}),
        )
        await self.send_frame(RESP_BODY, stream_id, _json(body))
        await self.send_frame(RESP_END, stream_id)

    async def _serve(self, stream_id: int, st: _ServedStream) -> None:
        method = str(st.head.get("method") or "GET").upper()
        path = str(st.head.get("path") or "/")
        server = self._server
        try:
            api_key = server.api_key() if server is not None else None
            if server is None or not api_key:
                await self._reply(
                    stream_id, 403, {"detail": "this node does not serve requests from you"}
                )
                return
            if not path_allowed(path, server.access_level(), method):
                await self._reply(
                    stream_id, 403, {"detail": f"path not offered to the federation: {path}"}
                )
                return
            if server.gate is not None:
                refusal = await server.gate(method, path, bytes(st.body))
                if refusal is not None:
                    await self._reply(stream_id, refusal[0], refusal[1])
                    return
            headers = [
                (str(k), str(v))
                for k, v in st.head.get("headers", [])
                if str(k).lower() not in _SERVER_STRIP and str(k).lower() != FEDERATION_HOP_HEADER
            ]
            headers.append(("authorization", f"Bearer {api_key}"))
            headers.append((FEDERATION_HOP_HEADER, str(self.remote_info.get("node_id") or "peer")))
            request = server.client.build_request(
                method, path, headers=headers, content=bytes(st.body)
            )
            response = await server.client.send(request, stream=True)
            try:
                await self.send_frame(
                    RESP_HEAD,
                    stream_id,
                    _json(
                        {
                            "status": response.status_code,
                            "headers": [
                                [k, v]
                                for k, v in response.headers.multi_items()
                                if k.lower() not in _HOP_BY_HOP
                            ],
                        }
                    ),
                )
                async for chunk in _raw_chunks(response):
                    for piece in _split(chunk):
                        await self.send_frame(RESP_BODY, stream_id, piece)
                await self.send_frame(RESP_END, stream_id)
            finally:
                await response.aclose()
            self.requests_served += 1
        except asyncio.CancelledError:
            raise
        except TunnelClosed:
            pass
        except Exception as exc:  # noqa: BLE001
            logger.warning("federation_tunnel_local_error", path=path, error=str(exc))
            with contextlib.suppress(TunnelClosed):
                await self.send_frame(
                    ERROR, stream_id, _json({"message": str(exc) or type(exc).__name__})
                )
        finally:
            self._served.pop(stream_id, None)


async def _raw_chunks(response: httpx.Response) -> AsyncIterator[bytes]:
    """Body bytes exactly as sent (still content-encoded, matching the headers)."""
    if response.is_stream_consumed:
        # Already read by the transport (in-process/mock transports do this).
        yield response.content
        return
    async for chunk in response.aiter_raw():
        yield chunk


# ── httpx transport over a connection ────────────────────────────


def _read_timeout(request: httpx.Request) -> float | None:
    timeout = request.extensions.get("timeout") or {}
    return timeout.get("read")


async def _next_frame(stream: _ClientStream, timeout: float | None, request: httpx.Request):  # noqa: ASYNC109
    try:
        if timeout is None:
            item = await stream.queue.get()
        else:
            item = await asyncio.wait_for(stream.queue.get(), timeout)
    except TimeoutError as exc:
        raise httpx.ReadTimeout("tunnel read timeout", request=request) from exc
    if item[0] is _CLOSED:
        raise httpx.ReadError("federation tunnel closed", request=request)
    return item


def _error_message(payload: bytes) -> str:
    try:
        return str(json.loads(payload).get("message") or "unknown error")
    except Exception:  # noqa: BLE001
        return payload.decode(errors="replace") or "unknown error"


class _TunnelResponseStream(httpx.AsyncByteStream):
    def __init__(
        self,
        conn: TunnelConnection,
        stream_id: int,
        stream: _ClientStream,
        request: httpx.Request,
    ) -> None:
        self._conn = conn
        self._stream_id = stream_id
        self._stream = stream
        self._request = request

    async def __aiter__(self) -> AsyncIterator[bytes]:
        timeout = _read_timeout(self._request)
        while not self._stream.finished:
            frame_type, payload = await _next_frame(self._stream, timeout, self._request)
            if frame_type == RESP_BODY:
                if payload:
                    yield payload
            elif frame_type == RESP_END:
                self._stream.finished = True
            elif frame_type == ERROR:
                self._stream.finished = True
                message = _error_message(payload)
                raise httpx.ReadError(f"peer tunnel error: {message}", request=self._request)

    async def aclose(self) -> None:
        if not self._stream.finished and not self._conn.closed:
            # The consumer stopped early (client disconnected): tell the peer.
            with contextlib.suppress(TunnelClosed):
                await self._conn.send_frame(CANCEL, self._stream_id)
        self._stream.finished = True
        self._conn.release_stream(self._stream_id)


class TunnelTransport(httpx.AsyncBaseTransport):
    """httpx transport that sends requests through the peer's current tunnel.

    ``conn_getter`` is called per request so a reconnected tunnel is picked up
    transparently by the same ``httpx.AsyncClient``.
    """

    def __init__(self, conn_getter: Callable[[], TunnelConnection | None]) -> None:
        self._conn_getter = conn_getter

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        conn = self._conn_getter()
        if conn is None or conn.closed:
            raise httpx.ConnectError("federation tunnel not connected", request=request)

        target = request.url.raw_path.decode("ascii", errors="replace")
        headers = [
            [k.decode("latin-1"), v.decode("latin-1")]
            for k, v in request.headers.raw
            if k.decode("latin-1").lower() not in _HOP_BY_HOP
        ]
        try:
            stream_id, stream = conn.open_stream()
        except TunnelClosed as exc:
            raise httpx.ConnectError(str(exc), request=request) from exc

        try:
            await conn.send_frame(
                REQ_HEAD,
                stream_id,
                _json({"method": request.method, "path": target, "headers": headers}),
            )
            async for chunk in request.stream:  # type: ignore[union-attr]
                for piece in _split(bytes(chunk)):
                    await conn.send_frame(REQ_BODY, stream_id, piece)
            await conn.send_frame(REQ_END, stream_id)

            frame_type, payload = await _next_frame(stream, _read_timeout(request), request)
        except TunnelClosed as exc:
            conn.release_stream(stream_id)
            raise httpx.WriteError(f"federation tunnel closed: {exc}", request=request) from exc
        except BaseException:
            conn.release_stream(stream_id)
            raise

        if frame_type == ERROR:
            conn.release_stream(stream_id)
            raise httpx.RemoteProtocolError(
                f"peer tunnel error: {_error_message(payload)}", request=request
            )
        if frame_type != RESP_HEAD:
            conn.release_stream(stream_id)
            raise httpx.RemoteProtocolError(
                f"unexpected tunnel frame 0x{frame_type:02x}", request=request
            )

        head = json.loads(payload)
        response_headers = [
            (str(k), str(v))
            for k, v in head.get("headers", [])
            if str(k).lower() not in _HOP_BY_HOP
        ]
        return httpx.Response(
            status_code=int(head.get("status", 502)),
            headers=response_headers,
            stream=_TunnelResponseStream(conn, stream_id, stream, request),
            request=request,
        )


# ── Handshakes ───────────────────────────────────────────────────


class TunnelRejected(Exception):
    """The other side refused our credentials; retrying fast is pointless."""


async def dial_handshake(
    channel: Channel,
    hello: dict[str, Any],
    *,
    timeout: float = 15.0,  # noqa: ASYNC109
) -> dict[str, Any]:
    """Dialer side: send HELLO, return the WELCOME payload or raise."""
    await channel.send(encode_frame(HELLO, 0, _json({"protocol": PROTOCOL_VERSION, **hello})))
    try:
        frame_type, _, payload = decode_frame(await asyncio.wait_for(channel.recv(), timeout))
    except TimeoutError as exc:
        raise TunnelClosed("handshake timeout") from exc
    if frame_type == REJECT:
        raise TunnelRejected(_error_message(payload))
    if frame_type != WELCOME:
        raise TunnelClosed(f"unexpected handshake frame 0x{frame_type:02x}")
    welcome = json.loads(payload or b"{}")
    return welcome if isinstance(welcome, dict) else {}


async def accept_tunnel(
    channel: Channel,
    *,
    authenticate: Callable[[dict[str, Any]], Awaitable[tuple[Any, dict[str, Any]] | None]],
    handshake_timeout: float = 15.0,
) -> tuple[Any, dict[str, Any]] | None:
    """Acceptor side of the handshake.

    ``authenticate(hello)`` returns ``(peer, welcome_payload)`` for a valid
    token or ``None``. On success sends WELCOME and returns ``(peer,
    remote_info)`` (the HELLO minus the token); the caller builds the
    :class:`TunnelConnection`. On failure sends REJECT and closes.
    """
    try:
        frame_type, _, payload = decode_frame(
            await asyncio.wait_for(channel.recv(), handshake_timeout)
        )
    except (TimeoutError, TunnelClosed, ValueError):
        await channel.close(code=1002, reason="handshake failed")
        return None
    if frame_type != HELLO:
        await channel.close(code=1002, reason="expected HELLO")
        return None
    try:
        hello = json.loads(payload)
    except Exception:  # noqa: BLE001
        hello = {}
    result = await authenticate(hello) if isinstance(hello, dict) else None
    if result is None:
        with contextlib.suppress(TunnelClosed):
            await channel.send(encode_frame(REJECT, 0, _json({"message": "invalid tunnel token"})))
        await channel.close(code=4401, reason="unauthorized")
        return None
    peer, welcome = result
    info = {k: v for k, v in hello.items() if k not in ("token", "protocol")}
    await channel.send(encode_frame(WELCOME, 0, _json(welcome)))
    return peer, info


def ws_url_for(base_url: str) -> str:
    """``https://host`` → ``wss://host/ocabra/federation/tunnel``."""
    base = base_url.rstrip("/")
    if base.startswith("https://"):
        base = "wss://" + base[len("https://") :]
    elif base.startswith("http://"):
        base = "ws://" + base[len("http://") :]
    return f"{base}/ocabra/federation/tunnel"


async def websockets_connect(url: str, *, verify_ssl: bool = True) -> Channel:
    """Open a client WebSocket with the tunnel's limits."""
    import ssl

    from websockets.asyncio.client import connect

    ssl_ctx: ssl.SSLContext | None = None
    if url.startswith("wss://"):
        ssl_ctx = ssl.create_default_context()
        if not verify_ssl:
            ssl_ctx.check_hostname = False
            ssl_ctx.verify_mode = ssl.CERT_NONE
    conn = await connect(
        url,
        max_size=MAX_MESSAGE_SIZE,
        ssl=ssl_ctx,
        ping_interval=20,
        ping_timeout=20,
        open_timeout=15,
    )
    return WebsocketsChannel(conn)


class TunnelDialer:
    """Keeps a dialed tunnel open: connect, handshake, run, reconnect.

    ``hello()`` builds the HELLO (must include ``token``). ``on_connected`` is
    awaited with each new :class:`TunnelConnection` before it starts running;
    ``on_disconnected`` is called when it ends.
    """

    def __init__(
        self,
        *,
        url: str,
        hello: Callable[[], dict[str, Any]],
        server: Callable[[], LocalServer | None],
        on_connected: Callable[[TunnelConnection], Awaitable[None]],
        on_disconnected: Callable[[TunnelConnection], None],
        connect: Callable[[str], Awaitable[Channel]] | None = None,
        verify_ssl: bool = True,
        name: str = "federation-tunnel-dialer",
    ) -> None:
        self.url = url.rstrip("/")
        self._hello = hello
        self._server = server
        self._on_connected = on_connected
        self._on_disconnected = on_disconnected
        self._connect = connect or (lambda u: websockets_connect(u, verify_ssl=verify_ssl))
        self._name = name
        self._task: asyncio.Task | None = None
        self._stopping = False
        # Status, for the UI.
        self.state = "idle"  # idle | connecting | connected | backoff | rejected | stopped
        self.last_error: str | None = None
        self.connection: TunnelConnection | None = None

    @property
    def ws_url(self) -> str:
        return ws_url_for(self.url)

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._stopping = False
            self._task = asyncio.create_task(self.run_forever(), name=self._name)

    async def stop(self) -> None:
        self._stopping = True
        if self.connection is not None:
            await self.connection.close(code=1001, reason="stopping")
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task
            self._task = None
        self.state = "stopped"

    async def run_forever(self) -> None:
        """Reconnect loop with jittered exponential backoff."""
        attempt = 0
        while not self._stopping:
            self.state = "connecting"
            try:
                if await self.serve_once():
                    attempt = 0  # only an established session resets the backoff
            except asyncio.CancelledError:
                raise
            except TunnelRejected as exc:
                self.state = "rejected"
                self.last_error = str(exc)
                logger.warning("federation_tunnel_rejected", hub=self.url, error=str(exc))
                await asyncio.sleep(300)
                continue
            except Exception as exc:  # noqa: BLE001
                self.last_error = str(exc) or type(exc).__name__
                logger.warning("federation_tunnel_error", hub=self.url, error=self.last_error)
            if self._stopping:
                break
            attempt += 1
            delay = min(60.0, 2 ** min(attempt, 6)) * (0.5 + random.random() / 2)
            self.state = "backoff"
            await asyncio.sleep(delay)

    async def serve_once(self, channel: Channel | None = None) -> bool:
        """Connect (unless given a channel), handshake and run until closed.

        Returns whether the handshake succeeded.
        """
        if channel is None:
            channel = await self._connect(self.ws_url)
        try:
            welcome = await dial_handshake(channel, self._hello())
        except BaseException:
            await channel.close()
            raise
        conn = TunnelConnection(channel, dialer=True, remote_info=welcome, server=self._server())
        self.connection = conn
        self.state = "connected"
        self.last_error = None
        logger.info("federation_tunnel_connected", hub=self.url, hub_node=welcome.get("node_name"))
        try:
            await self._on_connected(conn)
            await conn.run()
        finally:
            await conn.close()
            self.connection = None
            self._on_disconnected(conn)
            logger.info("federation_tunnel_disconnected", hub=self.url)
        return True
