"""Join flow end to end: two FederationManagers over the ``<db>_test`` Postgres.

Node A is reachable (a real uvicorn serving the federation router); node B has
no URL. B asks to join, A accepts (or B uses an invite), B dials A and both use
each other over the one tunnel. Each side's local API is a MockTransport.
"""

from __future__ import annotations

import asyncio
import socket

import httpx
import pytest
import sqlalchemy as sa

from ocabra.core.federation import FederationManager
from ocabra.core.federation_join import JoinError
from ocabra.core.federation_tunnel import FEDERATION_HOP_HEADER
from ocabra.database import AsyncSessionLocal
from tests.conftest import requires_db
from tests.test_federation import _make_settings
from tests.test_federation_tunnel import _Local, _ws_app


async def _clean() -> None:
    from ocabra.db.auth import User
    from ocabra.db.federation import FederationInvite, FederationJoinRequest, FederationPeer

    async with AsyncSessionLocal() as session:
        await session.execute(sa.delete(FederationJoinRequest))
        await session.execute(sa.delete(FederationInvite))
        await session.execute(sa.delete(FederationPeer))
        await session.execute(sa.delete(User).where(User.username.like("federation:%")))
        await session.commit()


@pytest.fixture
async def two_nodes():
    """Yield ``(fm_a, fm_b, url_a, local_a, local_b)`` and tear everything down."""
    import uvicorn

    await _clean()
    settings_a = _make_settings(
        federation_node_id="node-a",
        federation_node_name="nodo-a",
        federation_accept_join_requests=True,
    )
    settings_b = _make_settings(federation_node_id="node-b", federation_node_name="nodo-b")
    fm_a = FederationManager(settings_a, AsyncSessionLocal)
    fm_b = FederationManager(settings_b, AsyncSessionLocal)
    await fm_a.start()
    await fm_b.start()
    local_a, local_b = _Local("nodo-a"), _Local("nodo-b")
    fm_a._local_client = httpx.AsyncClient(
        base_url="http://a", transport=httpx.MockTransport(local_a.handler)
    )
    fm_b._local_client = httpx.AsyncClient(
        base_url="http://b", transport=httpx.MockTransport(local_b.handler)
    )

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(_ws_app(fm_a), log_level="warning", ws="websockets"))
    server_task = asyncio.create_task(server.serve(sockets=[sock]))
    for _ in range(200):
        if server.started:
            break
        await asyncio.sleep(0.01)
    try:
        yield fm_a, fm_b, f"http://127.0.0.1:{port}", local_a, local_b
    finally:
        await fm_b.stop()
        await fm_a.stop()
        server.should_exit = True
        await server_task
        await _clean()


async def _wait(cond, timeout: float = 5.0) -> bool:  # noqa: ASYNC109 — polling helper, not an I/O timeout
    for _ in range(int(timeout / 0.02)):
        if cond():
            return True
        await asyncio.sleep(0.02)
    return cond()


def _only_peer(fm: FederationManager):
    peers = fm.get_all_peers()
    assert len(peers) == 1, peers
    return peers[0]


@requires_db
async def test_request_accept_then_mutual_tunnel(two_nodes):
    fm_a, fm_b, url_a, local_a, local_b = two_nodes

    peer_on_b = await fm_b.join_service.join(url_a, message="hola, soy B", gpus=[{"name": "4090"}])
    assert peer_on_b.join_state == "pending" and peer_on_b.connection_mode == "tunnel_out"
    code = peer_on_b.join_verification_code
    assert code and len(code) == 8

    pending = await fm_a.join_service.list_requests()
    assert len(pending) == 1
    req = pending[0]
    assert req["remote_node_name"] == "nodo-b" and req["message"] == "hola, soy B"
    assert req["verification_code"] == code and req["gpus"] == [{"name": "4090"}]
    assert fm_a.get_all_peers() == []

    peer_on_a = await fm_a.join_service.accept(req["id"], notes="de Fulano")
    assert peer_on_a.connection_mode == "tunnel_in" and peer_on_a.mode == "swarm"
    assert peer_on_a.access_level == "full" and peer_on_a.local_access_level == "full"
    assert peer_on_a.remote_node_id == "node-b"

    assert await fm_b.join_service.check_outgoing(peer_on_b) == "accepted"
    assert peer_on_b.join_state is None and peer_on_b.api_key.startswith("oct_")
    assert await _wait(lambda: peer_on_a.online and peer_on_b.online)

    # A → B: served by B's API with B's service key for A, marked as a hop.
    resp = await fm_a.proxy_request(peer_on_a, "/v1/echo", {"de": "a"}, {})
    assert resp.status_code == 201 and resp.headers["x-node"] == "nodo-b"
    seen_b = local_b.seen[-1]
    assert seen_b.headers["authorization"].startswith("Bearer sk-ocabra-")
    assert seen_b.headers[FEDERATION_HOP_HEADER] == "node-a"
    # B → A over the same socket.
    resp = await fm_b.proxy_request(peer_on_b, "/v1/echo", {"de": "b"}, {})
    assert resp.status_code == 201 and resp.headers["x-node"] == "nodo-a"
    assert local_a.seen[-1].headers[FEDERATION_HOP_HEADER] == "node-b"
    # Swarm = full control: model ops go through too.
    resp = await fm_a.proxy_request(peer_on_a, "/ocabra/models/x/load", {}, {})
    assert resp.status_code == 200

    # Each side created a service user for the other.
    from ocabra.db.auth import User

    async with AsyncSessionLocal() as session:
        names = set(
            (
                await session.execute(
                    sa.select(User.username).where(User.username.like("federation:%"))
                )
            ).scalars()
        )
    assert names == {f"federation:{peer_on_a.name}", f"federation:{peer_on_b.name}"}

    # The token was handed over once the tunnel came up: no longer retrievable.
    from ocabra.db.federation import FederationJoinRequest

    async with AsyncSessionLocal() as session:
        row = (await session.execute(sa.select(FederationJoinRequest))).scalar_one()
        assert row.status == "accepted" and row.token_encrypted is None

    # Reloading from the DB (restart) restores the peer with its secrets. (Both
    # test nodes share one database, so A also sees B's row: pick its own.)
    await fm_a._load_peers_from_db()
    reloaded = {p.peer_id: p for p in fm_a.get_all_peers()}
    again = reloaded[peer_on_a.peer_id]
    assert again.connection_mode == "tunnel_in"
    assert again.local_api_key == peer_on_a.local_api_key
    assert again.tunnel_token_hash == peer_on_a.tunnel_token_hash
    again_b = reloaded[peer_on_b.peer_id]
    assert again_b.api_key == peer_on_b.api_key and again_b.join_state is None


@requires_db
async def test_invite_skips_manual_approval(two_nodes):
    fm_a, fm_b, url_a, _, _ = two_nodes
    fm_a.settings.federation_accept_join_requests = False  # invites work regardless

    invite = await fm_a.join_service.create_invite(label="runpod-1", max_uses=1)
    peer_on_b = await fm_b.join_service.join(url_a, invite_code=invite["code"])
    assert peer_on_b.join_state is None
    peer_on_a = _only_peer(fm_a)
    assert await _wait(lambda: peer_on_a.online and peer_on_b.online)

    invites = await fm_a.join_service.list_invites()
    assert invites[0]["uses"] == 1 and invites[0]["usable"] is False
    # Used up: a second node can't reuse it.
    with pytest.raises(JoinError) as exc:
        await fm_a.join_service.create_request(
            {"node_id": "node-c", "node_name": "c", "invite_code": invite["code"]},
            source_ip="1.2.3.4",
        )
    assert exc.value.status == 403


@requires_db
async def test_disabled_requests_and_denial(two_nodes):
    fm_a, fm_b, url_a, _, _ = two_nodes

    fm_a.settings.federation_accept_join_requests = False
    with pytest.raises(JoinError) as exc:
        await fm_b.join_service.join(url_a)
    assert exc.value.status == 403
    assert fm_b.get_all_peers() == []

    fm_a.settings.federation_accept_join_requests = True
    peer_on_b = await fm_b.join_service.join(url_a)
    req = (await fm_a.join_service.list_requests())[0]
    await fm_a.join_service.deny(req["id"])
    assert await fm_b.join_service.check_outgoing(peer_on_b) == "denied"
    assert peer_on_b.join_state == "denied" and peer_on_b.dialer is None
    assert fm_a.get_all_peers() == []


@requires_db
async def test_hub_removing_peer_cuts_it_off(two_nodes):
    fm_a, fm_b, url_a, _, _ = two_nodes
    invite = await fm_a.join_service.create_invite(label="x")
    peer_on_b = await fm_b.join_service.join(url_a, invite_code=invite["code"])
    peer_on_a = _only_peer(fm_a)
    assert await _wait(lambda: peer_on_a.online and peer_on_b.online)

    await fm_a.remove_peer(peer_on_a.peer_id)
    assert await _wait(lambda: not peer_on_b.tunnel_connected)
    # B keeps trying; A now rejects its token.
    assert await _wait(lambda: peer_on_b.dialer.state == "rejected", timeout=8)

    from ocabra.db.auth import User

    async with AsyncSessionLocal() as session:
        left = (
            await session.execute(
                sa.select(User.username).where(User.username == f"federation:{peer_on_a.name}")
            )
        ).scalar_one_or_none()
    assert left is None  # A's service user for B is gone


@requires_db
async def test_public_request_guards(two_nodes):
    fm_a, _, _, _, _ = two_nodes
    svc = fm_a.join_service
    with pytest.raises(JoinError) as exc:
        await svc.create_request({"node_id": "node-a", "node_name": "yo"}, source_ip="1.1.1.1")
    assert exc.value.status == 400  # joining yourself
    for i in range(3):
        await svc.create_request({"node_id": f"n{i}", "node_name": f"n{i}"}, source_ip="9.9.9.9")
    with pytest.raises(JoinError) as exc:
        await svc.create_request({"node_id": "n9", "node_name": "n9"}, source_ip="9.9.9.9")
    assert exc.value.status == 429  # per-IP pending cap
    # Re-asking replaces the node's own pending request instead of piling up.
    await svc.create_request({"node_id": "n0", "node_name": "n0"}, source_ip="8.8.8.8")
    names = [r["remote_node_name"] for r in await svc.list_requests()]
    assert names.count("n0") == 1
