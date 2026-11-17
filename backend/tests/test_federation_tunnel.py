"""Tests for the federation reverse tunnel (core/federation_tunnel.py).

The two ends talk over an in-memory channel; each side's "local API" (what it
runs the other side's requests against) is an httpx MockTransport.
"""

from __future__ import annotations

import asyncio
import json
import uuid

import httpx
import pytest

from ocabra.core.federation import (
    FederationHopMiddleware,
    FederationManager,
    PeerState,
    federation_hop,
    hash_tunnel_token,
)
from ocabra.core.federation_tunnel import (
    CHUNK_SIZE,
    FEDERATION_HOP_HEADER,
    HELLO,
    REJECT,
    WELCOME,
    LocalServer,
    MemoryChannel,
    TunnelConnection,
    TunnelDialer,
    TunnelRejected,
    TunnelTransport,
    accept_tunnel,
    decode_frame,
    encode_frame,
    path_allowed,
    ws_url_for,
)
from tests.test_federation import _make_session_factory, _make_settings

TOKEN = "oct_test-token"


# ── Helpers ──────────────────────────────────────────────────────


class _Local:
    """Fake local API of one node. Records what it received."""

    def __init__(self, name: str = "agente") -> None:
        self.name = name
        self.seen: list[httpx.Request] = []
        self.stream_cancelled = asyncio.Event()

    async def handler(self, request: httpx.Request) -> httpx.Response:
        self.seen.append(request)
        path = request.url.path
        if path == "/ocabra/federation/heartbeat":
            return httpx.Response(
                200,
                json={
                    "node_id": f"{self.name}-id",
                    "node_name": self.name,
                    "version": "test",
                    "uptime_seconds": 1.0,
                    "gpus": [
                        {"index": 0, "name": "RTX 3090", "total_vram_mb": 24576, "free_vram_mb": 1}
                    ],
                    "models": [{"model_id": f"llama_cpp/{self.name}", "status": "loaded"}],
                    "load": {"active_requests": 0, "gpu_utilization_avg_pct": 0.0},
                },
            )
        if path == "/v1/echo":
            body = await request.aread()
            return httpx.Response(
                201,
                content=body,
                headers={
                    "content-type": request.headers.get("content-type", ""),
                    "x-node": self.name,
                },
            )
        if path == "/v1/slow":
            delay = float(request.url.params.get("d", "0"))
            await asyncio.sleep(delay)
            return httpx.Response(200, json={"d": delay})
        if path == "/v1/stream":
            local = self

            async def gen():
                try:
                    for i in range(5):
                        yield f"data: {i}\n\n".encode()
                        await asyncio.sleep(0.01)
                    if request.url.params.get("forever"):
                        while True:
                            yield b"data: x\n\n"
                            await asyncio.sleep(0.01)
                finally:
                    local.stream_cancelled.set()

            return httpx.Response(200, content=gen(), headers={"content-type": "text/event-stream"})
        if path.startswith("/ocabra/models"):
            return httpx.Response(200, json={"ok": True})
        return httpx.Response(404, json={"detail": "nope"})


def _server(local: _Local, key: str | None, level: str = "inference") -> LocalServer:
    return LocalServer(
        client=httpx.AsyncClient(
            base_url="http://local", transport=httpx.MockTransport(local.handler)
        ),
        api_key=lambda: key,
        access_level=lambda: level,
    )


async def _authenticate(hello):
    if hello.get("token") == TOKEN:
        return "peer-object", {"node_id": "hub-id", "node_name": "hub"}
    return None


async def _pair(
    *,
    dialer_level: str = "inference",
    acceptor_level: str = "inference",
    dialer_key: str | None = "sk-key-on-dialer",
    acceptor_key: str | None = "sk-key-on-acceptor",
):
    """Wire acceptor (hub) and dialer over memory.

    Returns ``(hub_conn, dial_conn, hub_local, dial_local, tasks)``: requests
    sent on ``hub_conn`` are served by the dialer's local API, and vice versa.
    """
    hub_local, dial_local = _Local("hub"), _Local("agente")
    hub_ch, dial_ch = MemoryChannel.pair()
    connected: list[TunnelConnection] = []

    async def _on_connected(conn):
        connected.append(conn)

    dialer = TunnelDialer(
        url="https://hub.example",
        hello=lambda: {
            "token": TOKEN,
            "node_id": "agente-id",
            "node_name": "agente",
            "version": "t",
        },
        server=lambda: _server(dial_local, dialer_key, dialer_level),
        on_connected=_on_connected,
        on_disconnected=lambda conn: None,
    )
    dial_task = asyncio.create_task(dialer.serve_once(dial_ch))
    accepted = await accept_tunnel(hub_ch, authenticate=_authenticate)
    assert accepted is not None
    _, info = accepted
    hub_conn = TunnelConnection(
        hub_ch,
        dialer=False,
        remote_info=info,
        server=_server(hub_local, acceptor_key, acceptor_level),
    )
    hub_task = asyncio.create_task(hub_conn.run())
    for _ in range(100):
        if connected:
            break
        await asyncio.sleep(0.01)
    assert dialer.state == "connected"
    return hub_conn, connected[0], hub_local, dial_local, [dial_task, hub_task]


def _client(conn: TunnelConnection) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=TunnelTransport(lambda: conn), base_url="http://federation-tunnel.invalid"
    )


async def _shutdown(conn: TunnelConnection, tasks) -> None:
    await conn.close()
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


# ── Framing / policy ─────────────────────────────────────────────


def test_frame_round_trip():
    data = encode_frame(0x21, 123456, b"hola")
    assert decode_frame(data) == (0x21, 123456, b"hola")


def test_frame_too_short():
    with pytest.raises(ValueError):
        decode_frame(b"\x01\x00")


@pytest.mark.parametrize(
    ("path", "level", "method", "allowed"),
    [
        ("/v1/chat/completions", "inference", "POST", True),
        ("/api/chat", "inference", "POST", True),
        ("/ocabra/federation/heartbeat", "inference", "GET", True),
        ("/ocabra/models/x/load", "inference", "POST", False),
        ("/ocabra/stats/overview", "inference", "GET", False),
        # full (swarm): the other node's admins can manage this one remotely...
        ("/ocabra/models/x/load", "full", "POST", True),
        ("/ocabra/downloads", "full", "POST", True),
        ("/ocabra/stats/overview", "full", "GET", True),
        ("/ocabra/profiles", "full", "GET", True),
        ("/ocabra/status", "full", "GET", True),
        ("/ocabra/federation/heartbeat", "full", "GET", True),
        # ...but who-can-access-what and the federation stay local.
        ("/ocabra/users", "full", "GET", False),
        ("/ocabra/groups/g1/models", "full", "PUT", False),
        ("/ocabra/auth/keys", "full", "POST", False),
        ("/ocabra/federation/invites", "full", "POST", False),
        ("/ocabra/federation/peers", "full", "GET", False),
        ("/ocabra/cloud/nodes", "full", "POST", False),
        ("/ocabra/host/power", "full", "POST", False),
        ("/ocabra/ws", "full", "GET", False),
        ("/ocabra/config", "full", "GET", True),
        ("/ocabra/config", "full", "PATCH", False),
        ("/ocabra/usersx", "full", "GET", True),  # prefix match is per path segment
        ("/v1/../ocabra/users", "full", "GET", False),
        ("/v1//x", "full", "GET", False),
        ("/etc/passwd", "full", "GET", False),
    ],
)
def test_path_allowed(path, level, method, allowed):
    assert path_allowed(path, level, method) is allowed


def test_ws_url_for():
    assert ws_url_for("https://a.b/") == "wss://a.b/ocabra/federation/tunnel"
    assert ws_url_for("http://a:8484") == "ws://a:8484/ocabra/federation/tunnel"


# ── Both directions over one connection ──────────────────────────


async def test_hub_to_dialer_replaces_credentials_and_marks_hop():
    hub, _, _, dial_local, tasks = await _pair()
    try:
        async with _client(hub) as client:
            resp = await client.post(
                "/v1/echo?x=1",
                json={"hola": "mundo"},
                headers={
                    "Authorization": "Bearer hub-key",
                    "Cookie": "ocabra_session=abc",
                    "X-Gateway-Token": "secret",
                },
            )
        assert resp.status_code == 201
        assert resp.json() == {"hola": "mundo"}
        assert resp.headers["x-node"] == "agente"
        seen = dial_local.seen[-1]
        assert seen.url.path == "/v1/echo" and seen.url.params["x"] == "1"
        # The other side's credentials never reach our API: our service key does.
        assert seen.headers["authorization"] == "Bearer sk-key-on-dialer"
        assert "cookie" not in seen.headers and "x-gateway-token" not in seen.headers
        assert seen.headers[FEDERATION_HOP_HEADER] == "hub-id"
    finally:
        await _shutdown(hub, tasks)


async def test_dialer_to_hub_over_the_same_socket():
    hub, dial, hub_local, _, tasks = await _pair()
    try:
        async with _client(dial) as client:
            resp = await client.post("/v1/echo", json={"de": "b"})
        assert resp.status_code == 201 and resp.headers["x-node"] == "hub"
        assert hub_local.seen[-1].headers["authorization"] == "Bearer sk-key-on-acceptor"
        assert hub_local.seen[-1].headers[FEDERATION_HOP_HEADER] == "agente-id"
    finally:
        await _shutdown(hub, tasks)


async def test_concurrent_requests_both_directions():
    hub, dial, _, _, tasks = await _pair()
    try:
        delays = [0.04, 0.0, 0.03, 0.01]
        async with _client(hub) as c1, _client(dial) as c2:
            results = await asyncio.gather(
                *(c1.post(f"/v1/slow?d={d}", json={}) for d in delays),
                *(c2.post(f"/v1/slow?d={d}", json={}) for d in delays),
            )
        assert [r.json()["d"] for r in results] == delays + delays
        assert hub.active_streams == 0 and dial.active_streams == 0
        assert hub.requests_sent == 4 and hub.requests_served == 4
    finally:
        await _shutdown(hub, tasks)


async def test_side_without_service_key_refuses_to_serve():
    hub, _, _, dial_local, tasks = await _pair(dialer_key=None)
    try:
        async with _client(hub) as client:
            resp = await client.post("/v1/echo", json={})
        assert resp.status_code == 403
        assert dial_local.seen == []
    finally:
        await _shutdown(hub, tasks)


async def test_large_body_is_chunked_and_intact():
    hub, _, _, _, tasks = await _pair()
    try:
        payload = bytes(range(256)) * ((CHUNK_SIZE * 3) // 256 + 7)
        async with _client(hub) as client:
            resp = await client.post(
                "/v1/echo", content=payload, headers={"content-type": "application/octet-stream"}
            )
        assert resp.status_code == 201 and resp.content == payload
    finally:
        await _shutdown(hub, tasks)


async def test_multipart_upload_survives_the_tunnel():
    hub, _, _, dial_local, tasks = await _pair()
    try:
        async with _client(hub) as client:
            resp = await client.post(
                "/v1/echo",
                files={"file": ("a.wav", b"RIFF" + b"\x00" * 100_000, "audio/wav")},
                data={"model": "whisper"},
            )
        assert resp.status_code == 201
        assert (
            dial_local.seen[-1].headers["content-type"].startswith("multipart/form-data; boundary=")
        )
        assert b'name="model"' in resp.content and b"RIFF" in resp.content
    finally:
        await _shutdown(hub, tasks)


async def test_streaming_response_arrives_incrementally():
    hub, _, _, _, tasks = await _pair()
    try:
        chunks = []
        async with _client(hub) as client:
            async with client.stream("POST", "/v1/stream", json={}) as resp:
                assert resp.headers["content-type"] == "text/event-stream"
                async for chunk in resp.aiter_bytes():
                    chunks.append(chunk)
        assert b"".join(chunks) == b"".join(f"data: {i}\n\n".encode() for i in range(5))
        assert len(chunks) >= 2
    finally:
        await _shutdown(hub, tasks)


async def test_path_not_offered_is_403_and_never_reaches_local():
    hub, _, _, dial_local, tasks = await _pair(dialer_level="inference")
    try:
        async with _client(hub) as client:
            resp = await client.post("/ocabra/models/x/load", json={})
        assert resp.status_code == 403 and dial_local.seen == []
    finally:
        await _shutdown(hub, tasks)


async def test_full_access_reaches_model_ops():
    hub, _, _, _, tasks = await _pair(dialer_level="full")
    try:
        async with _client(hub) as client:
            resp = await client.post("/ocabra/models/x/load", json={})
        assert resp.status_code == 200
    finally:
        await _shutdown(hub, tasks)


async def test_closing_stream_early_cancels_remote_work():
    hub, _, _, dial_local, tasks = await _pair()
    try:
        async with _client(hub) as client:
            async with client.stream("POST", "/v1/stream?forever=1", json={}) as resp:
                async for _ in resp.aiter_bytes():
                    break  # client went away
        await asyncio.wait_for(dial_local.stream_cancelled.wait(), 2)
    finally:
        await _shutdown(hub, tasks)


async def test_tunnel_drop_mid_stream_raises_network_error():
    hub, _, _, _, tasks = await _pair()
    try:
        async with _client(hub) as client:
            with pytest.raises(httpx.ReadError):
                async with client.stream("POST", "/v1/stream?forever=1", json={}) as resp:
                    async for _ in resp.aiter_bytes():
                        await hub.close()
    finally:
        await _shutdown(hub, tasks)


async def test_no_connection_is_connect_error():
    async with httpx.AsyncClient(transport=TunnelTransport(lambda: None)) as client:
        with pytest.raises(httpx.ConnectError):
            await client.get("http://federation-tunnel.invalid/v1/models")


async def test_wrong_token_is_rejected():
    hub_ch, dial_ch = MemoryChannel.pair()
    dialer = TunnelDialer(
        url="https://hub.example",
        hello=lambda: {"token": "oct_wrong"},
        server=lambda: None,
        on_connected=lambda c: asyncio.sleep(0),
        on_disconnected=lambda c: None,
    )
    task = asyncio.create_task(dialer.serve_once(dial_ch))
    assert await accept_tunnel(hub_ch, authenticate=_authenticate) is None
    with pytest.raises(TunnelRejected):
        await task


# ── Loop guard ───────────────────────────────────────────────────


async def test_hop_middleware_sets_and_resets_contextvar():
    seen = []

    async def app(scope, receive, send):
        seen.append(federation_hop.get())

    mw = FederationHopMiddleware(app)
    await mw({"type": "http", "headers": [(b"x-ocabra-federation-hop", b"node-a")]}, None, None)
    await mw({"type": "http", "headers": []}, None, None)
    assert seen == ["node-a", None]
    assert federation_hop.get() is None


async def test_request_from_a_peer_never_routes_to_another_peer():
    fm = FederationManager(_make_settings(), _make_session_factory())
    peer = PeerState(
        peer_id="p",
        name="x",
        url="http://x",
        api_key="k",
        access_level="inference",
        online=True,
        models=[{"model_id": "m", "status": "loaded"}],
    )
    fm._peers["p"] = peer
    assert "m" in fm.get_remote_models()
    token = federation_hop.set("node-a")
    try:
        assert fm.get_remote_models() == {}
        assert fm.find_best_peer("m") is None
    finally:
        federation_hop.reset(token)


# ── FederationManager integration ────────────────────────────────


def _tunnel_in_peer(enabled: bool = True) -> PeerState:
    return PeerState(
        peer_id=str(uuid.uuid4()),
        name="pod-1",
        url="",
        api_key="",
        access_level="full",
        enabled=enabled,
        connection_mode="tunnel_in",
        tunnel_token_hash=hash_tunnel_token(TOKEN),
        local_api_key="sk-for-pod",
        local_access_level="full",
    )


async def _started_fm() -> FederationManager:
    fm = FederationManager(_make_settings(), _make_session_factory())
    await fm.start()
    return fm


async def test_fm_authenticates_only_tunnel_in_tokens():
    fm = await _started_fm()
    try:
        peer = _tunnel_in_peer()
        fm._peers[peer.peer_id] = peer
        result = await fm.authenticate_tunnel({"token": TOKEN})
        assert result is not None and result[0] is peer
        assert result[1]["peer_name"] == "pod-1"
        assert await fm.authenticate_tunnel({"token": "oct_nope"}) is None
        assert await fm.authenticate_tunnel({}) is None
        peer.connection_mode = "tunnel_out"
        assert await fm.authenticate_tunnel({"token": TOKEN}) is None
        peer.connection_mode = "tunnel_in"
        peer.enabled = False
        assert await fm.authenticate_tunnel({"token": TOKEN}) is None
    finally:
        await fm.stop()


async def test_fm_local_server_follows_peer_state():
    fm = await _started_fm()
    try:
        peer = _tunnel_in_peer()
        server = fm.local_server_for(peer)
        assert server.api_key() == "sk-for-pod" and server.access_level() == "full"
        peer.enabled = False
        assert server.api_key() is None
    finally:
        await fm.stop()


async def test_fm_routes_heartbeat_and_proxy_through_tunnel():
    fm = await _started_fm()
    peer = _tunnel_in_peer()
    fm._peers[peer.peer_id] = peer
    hub, _, _, dial_local, tasks = await _pair()
    try:
        await fm._poll_all_peers()
        assert peer.online is False and peer.consecutive_failures == 0

        await fm.attach_tunnel(peer, hub)
        for _ in range(100):
            if peer.online:
                break
            await asyncio.sleep(0.01)
        assert peer.online is True and peer.gpus[0]["name"] == "RTX 3090"
        assert "llama_cpp/agente" in fm.get_remote_models()

        resp = await fm.proxy_request(peer, "/v1/echo", {"a": 1}, {"Authorization": "Bearer user"})
        assert resp.status_code == 201 and resp.json() == {"a": 1}
        assert dial_local.seen[-1].headers["authorization"] == "Bearer sk-key-on-dialer"

        chunks = [c async for c in fm.proxy_stream(peer, "/v1/stream", {}, {})]
        assert b"".join(chunks).count(b"data:") == 5

        test = await fm.test_peer_connection(peer.peer_id)
        assert test["success"] is True and test["node_name"] == "agente"

        fm.detach_tunnel(peer, hub)
        assert peer.online is False
        with pytest.raises(httpx.ConnectError):
            await fm.proxy_request(peer, "/v1/echo", {}, {})
    finally:
        await _shutdown(hub, tasks)
        await fm.stop()


async def test_fm_new_tunnel_replaces_old_one():
    fm = await _started_fm()
    peer = _tunnel_in_peer()
    fm._peers[peer.peer_id] = peer
    h1, _, _, _, t1 = await _pair()
    h2, _, _, _, t2 = await _pair()
    try:
        await fm.attach_tunnel(peer, h1)
        await fm.attach_tunnel(peer, h2)
        assert h1.closed and peer.tunnel_session is h2
        fm.detach_tunnel(peer, h1)  # stale detach must not drop the new one
        assert peer.tunnel_session is h2
    finally:
        await _shutdown(h1, t1)
        await _shutdown(h2, t2)
        await fm.stop()


async def test_proxy_headers_carry_hop_marker():
    fm = FederationManager(_make_settings(), _make_session_factory())
    peer = PeerState(peer_id="p", name="x", url="http://x", api_key="k", access_level="inference")
    headers = fm._build_proxy_headers(peer, {"Authorization": "Bearer user"})
    assert headers["Authorization"] == "Bearer k"
    assert headers[FEDERATION_HOP_HEADER] == "test-node-id"


async def test_proxy_headers_never_forward_this_nodes_credentials():
    """SEG-10: a Batches request carries the service token and the impersonated
    user; the HTTP proxy forwarded them to the peer as is (the tunnel already
    stripped them on the way in)."""
    fm = FederationManager(_make_settings(), _make_session_factory())
    peer = PeerState(peer_id="p", name="x", url="http://x", api_key="k", access_level="inference")
    headers = fm._build_proxy_headers(
        peer,
        {
            "X-Gateway-Token": "service-secret",
            "X-Internal-User-Id": "11111111-1111-1111-1111-111111111111",
            "X-Internal-Key-Name": "batch",
            "X-Api-Key": "user-key",
            "Accept": "application/json",
        },
    )
    lowered = {k.lower() for k in headers}
    assert not lowered & {"x-gateway-token", "x-internal-user-id", "x-internal-key-name", "x-api-key"}
    assert headers["Accept"] == "application/json"


# ── WebSocket endpoint (Starlette adapter + handshake) ───────────


def _ws_app(fm):
    from fastapi import FastAPI

    from ocabra.api.internal.federation import router

    app = FastAPI()
    app.include_router(router, prefix="/ocabra")
    app.state.federation_manager = fm
    return app


def test_ws_endpoint_handshake_accepts_and_rejects():
    from starlette.testclient import TestClient

    class _FM:
        def __init__(self):
            self.attached = []

        async def authenticate_tunnel(self, hello):
            return ("peer", {"node_name": "hub"}) if hello.get("token") == TOKEN else None

        def local_server_for(self, peer):
            return None

        async def attach_tunnel(self, peer, session):
            self.attached.append(session.remote_info)

        def detach_tunnel(self, peer, session):
            pass

    fm = _FM()
    with TestClient(_ws_app(fm)) as client:
        with client.websocket_connect("/ocabra/federation/tunnel") as ws:
            ws.send_bytes(
                encode_frame(HELLO, 0, json.dumps({"token": TOKEN, "node_name": "b"}).encode())
            )
            frame_type, _, payload = decode_frame(ws.receive_bytes())
            assert frame_type == WELCOME and json.loads(payload)["node_name"] == "hub"
        assert fm.attached == [{"node_name": "b"}]  # token never kept in remote_info

        with client.websocket_connect("/ocabra/federation/tunnel") as ws:
            ws.send_bytes(encode_frame(HELLO, 0, json.dumps({"token": "bad"}).encode()))
            frame_type, _, _ = decode_frame(ws.receive_bytes())
            assert frame_type == REJECT


def test_ws_endpoint_closes_when_federation_disabled():
    from starlette.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

    with TestClient(_ws_app(None)) as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ocabra/federation/tunnel") as ws:
                ws.receive_bytes()


# ── Schemas ──────────────────────────────────────────────────────


def test_peer_create_http_requires_url_and_key():
    from pydantic import ValidationError

    from ocabra.schemas.federation import PeerCreate

    with pytest.raises(ValidationError):
        PeerCreate(name="x", connection_mode="http")
    with pytest.raises(ValidationError):
        PeerCreate(name="x", connection_mode="tunnel_out")
    PeerCreate(name="x", url="http://a", api_key="k")
    tunnel = PeerCreate(name="pod", connection_mode="tunnel_in")
    assert tunnel.url == "" and tunnel.api_key == ""


# ── Real stack: uvicorn + Starlette WS ↔ websockets client ───────


async def test_real_websocket_stack_end_to_end():
    import socket

    import uvicorn

    fm = await _started_fm()
    peer = _tunnel_in_peer()
    fm._peers[peer.peer_id] = peer
    hub_local = _Local("hub")
    fm._local_client = httpx.AsyncClient(
        base_url="http://local", transport=httpx.MockTransport(hub_local.handler)
    )
    app = _ws_app(fm)

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", ws="websockets"))
    server_task = asyncio.create_task(server.serve(sockets=[sock]))
    dial_local = _Local("agente")
    conns: list[TunnelConnection] = []

    async def _connected(conn):
        conns.append(conn)

    dialer = TunnelDialer(
        url=f"http://127.0.0.1:{port}",
        hello=lambda: {"token": TOKEN, "node_id": "agente-id", "node_name": "agente"},
        server=lambda: _server(dial_local, "sk-dialer"),
        on_connected=_connected,
        on_disconnected=lambda c: None,
    )
    try:
        for _ in range(200):
            if server.started:
                break
            await asyncio.sleep(0.01)
        dialer.start()
        for _ in range(300):
            if peer.online:
                break
            await asyncio.sleep(0.01)
        assert dialer.state == "connected" and peer.online and peer.tunnel_connected

        big = b"z" * (CHUNK_SIZE * 5 + 3)
        resp = await fm.proxy_multipart(
            peer, "/v1/echo", files={"file": ("a.bin", big, "application/octet-stream")}, data={}
        )
        assert resp.status_code == 201 and big in resp.content
        chunks = [c async for c in fm.proxy_stream(peer, "/v1/stream", {}, {})]
        assert b"".join(chunks).count(b"data:") == 5

        # And the other way: the dialer uses the hub over the same socket.
        async with _client(conns[-1]) as client:
            back = await client.post("/v1/echo", json={"vuelta": 1})
        assert back.status_code == 201 and back.headers["x-node"] == "hub"
        assert hub_local.seen[-1].headers["authorization"] == "Bearer sk-for-pod"

        # Hub drops the socket → dialer reconnects on its own.
        await peer.tunnel_session.close()
        for _ in range(500):
            if peer.tunnel_connected and peer.online:
                break
            await asyncio.sleep(0.01)
        assert peer.tunnel_connected
    finally:
        await dialer.stop()
        await fm.stop()
        server.should_exit = True
        await server_task


async def test_node_name_is_read_live():
    """Renaming the node in config must show up without restarting the API."""
    settings = _make_settings(federation_node_name="viejo")
    fm = FederationManager(settings, _make_session_factory())
    assert fm.node_name == "viejo"
    settings.federation_node_name = "nuevo"
    assert fm.node_name == "nuevo"
    assert (
        fm.tunnel_hello(PeerState(peer_id="p", name="x", url="", api_key="t", access_level="full"))[
            "node_name"
        ]
        == "nuevo"
    )
    settings.federation_node_name = ""
    assert fm.node_name == "node-test-nod"


def test_join_client_ip_prefers_x_real_ip_and_ignores_forged_xff():
    from starlette.requests import Request

    from ocabra.api.internal.federation import _client_ip

    def _req(headers):
        return Request(
            {
                "type": "http",
                "headers": [(k.encode(), v.encode()) for k, v in headers.items()],
                "client": ("172.22.0.5", 1234),
            }
        )

    assert (
        _client_ip(_req({"x-real-ip": "85.1.2.3", "x-forwarded-for": "6.6.6.6, 85.1.2.3"}))
        == "85.1.2.3"
    )
    # Without Caddy's header, a forged X-Forwarded-For must not win.
    assert _client_ip(_req({"x-forwarded-for": "6.6.6.6"})) == "172.22.0.5"


async def test_service_manager_survives_missing_docker_cli(monkeypatch):
    """The ocabra-node image has no docker CLI: startup used to crash on it."""
    import asyncio as _asyncio

    from ocabra.core.service_manager import ServiceManager

    async def _no_docker(*a, **kw):
        raise FileNotFoundError("docker")

    monkeypatch.setattr(_asyncio, "create_subprocess_exec", _no_docker)
    monkeypatch.setattr("ocabra.core.docker_ops.settings.docker_agent_url", "")
    sm = ServiceManager()
    container = next(s.docker_container_name for s in sm._states.values() if s.docker_container_name)
    code, _out, err = await sm._docker("container.inspect", container=container, field="running")
    assert (code, err) == (127, "docker CLI not available")


def test_remote_load_and_unload_accept_canonical_ids_with_slash():
    """Canonical ids are ``backend/name``: without ``:path`` the routes 404'd on all of them."""
    from starlette.routing import Match

    from ocabra.api.internal.federation import router

    for action in ("load", "unload"):
        path = f"/federation/peers/p1/models/llamacpp_mks/qwen2.5-0.5b/{action}"
        scope = {"type": "http", "path": path, "method": "POST"}
        matches = [r for r in router.routes if r.matches(scope)[0] == Match.FULL]
        assert matches, path
        assert matches[0].matches(scope)[1]["path_params"]["model_id"] == "llamacpp_mks/qwen2.5-0.5b"
