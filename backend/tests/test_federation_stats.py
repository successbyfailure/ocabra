"""Federation visibility: what each request records and the aggregate view."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
import sqlalchemy as sa

from ocabra.core.federation import (
    FederationHopMiddleware,
    federation_request_info,
    note_federation_fallback,
    resolve_federated,
)
from ocabra.stats.collector import _federation_fields
from tests.conftest import requires_db
from tests.test_federation_proxy import (
    _FakeModelManager,
    _FakeModelState,
    _make_fm_with_peers,
    _make_peer,
)

# ── per request ──────────────────────────────────────────────────


async def test_middleware_gives_each_request_its_own_record():
    seen = {}

    async def app(scope, receive, send):
        seen["state"] = scope["state"]["federation_info"]
        seen["ctx"] = federation_request_info.get()
        note_federation_fallback(SimpleNamespace(name="pod"), 502)

    mw = FederationHopMiddleware(app)
    scope = {
        "type": "http",
        "headers": [(b"x-ocabra-federation-hop", b"node-b"), (b"x-ocabra-node", b"Mapache")],
    }
    await mw(scope, None, None)
    info = seen["state"]
    assert info is seen["ctx"]  # handler writes are visible to the collector
    assert info["inbound_hop"] == "node-b" and info["force_node"] == "Mapache"
    assert info["fallback"] is True and info["fallback_peer"] == "pod"
    assert federation_request_info.get() is None  # reset after the request


@pytest.fixture
def _ctx():
    info: dict = {}
    token = federation_request_info.set(info)
    yield info
    federation_request_info.reset(token)


async def test_decision_and_forced_node_are_recorded(_ctx, monkeypatch):
    import ocabra.core.model_manager as mm_module
    from tests.test_federation_proxy import _FakeModelStatus

    monkeypatch.setattr(mm_module, "ModelStatus", _FakeModelStatus)
    pod = _make_peer(peer_id="p1", name="pod", models=[{"model_id": "m", "status": "loaded"}])
    mapache = _make_peer(peer_id="p2", name="Mapache", models=[])
    fm = _make_fm_with_peers(pod, mapache)
    mm = _FakeModelManager({"m": _FakeModelState(model_id="m", status="configured")})

    assert await resolve_federated("m", mm, fm) == ("remote", pod)
    assert _ctx["reason"] == "local_cold_peer_warm" and _ctx["peer_name"] == "pod"

    # X-Ocabra-Node: an explicit node for this request, whatever the rules say.
    _ctx.clear()
    _ctx["force_node"] = "mapache"
    assert await resolve_federated("m", mm, fm) == ("remote", mapache)
    assert _ctx["reason"] == "forced" and _ctx["peer_name"] == "Mapache"
    _ctx.clear()
    _ctx["force_node"] = "local"
    assert await resolve_federated("m", mm, fm) == ("local", None)
    assert _ctx["reason"] == "forced_local"
    # Unknown node: normal routing, and it says so.
    _ctx.clear()
    _ctx["force_node"] = "nope"
    assert await resolve_federated("m", mm, fm) == ("remote", pod)
    assert _ctx["forced_unavailable"] == "nope"
    # Por URL (/_node/<nodo>/, los benchmarks): nunca en otro sitio, 503 y esperar.
    _ctx.clear()
    _ctx.update(force_node="nope", force_strict=True)
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        await resolve_federated("m", mm, fm)
    assert exc.value.status_code == 503 and exc.value.headers["Retry-After"] == "15"


def test_collector_fields():
    pod = SimpleNamespace(peer_id="p1", name="pod", remote_node_id="pod-node")
    fm = SimpleNamespace(get_all_peers=lambda: [pod])
    app = SimpleNamespace(state=SimpleNamespace(federation_manager=fm))

    def req(info):
        return SimpleNamespace(app=app, state=SimpleNamespace(federation_info=info))

    # Sent to a peer: its name (kept even if the peer is later deleted) and why.
    out = _federation_fields(
        req({"peer_name": "pod", "reason": "local_cold_peer_warm"}), None, "p1"
    )
    assert out == ("pod", None, "local_cold_peer_warm")
    # A peer failed and we served it: the route says so.
    assert _federation_fields(
        req({"reason": "local_cold_peer_warm", "fallback": True}), None, None
    )[2] == ("fallback_local")
    # Received from a peer: by its hop header, or by its service user.
    assert _federation_fields(req({"inbound_hop": "pod-node"}), None, None)[1] == "pod"
    user = SimpleNamespace(username="federation:Mapache")
    assert _federation_fields(req({}), user, None)[1] == "Mapache"
    # Plain local traffic stays clean.
    assert _federation_fields(req({"reason": "local_loaded"}), None, None) == (None, None, None)


# ── aggregate ────────────────────────────────────────────────────


@requires_db
async def test_federation_stats_both_ways_with_cost():
    from ocabra.database import AsyncSessionLocal
    from ocabra.db.cloud import CloudNode
    from ocabra.db.stats import RequestStat
    from ocabra.stats.aggregator import get_federation_stats

    tag = f"zz-fedstats-{uuid.uuid4().hex[:6]}"
    peer_id = str(uuid.uuid4())
    now = datetime.now(UTC)

    def row(**kw):
        base = dict(
            model_id=f"{tag}-m1",
            started_at=now - timedelta(minutes=5),
            duration_ms=100,
            status_code=200,
            request_kind="chat",
            input_tokens=100,
            output_tokens=50,
        )
        return RequestStat(**{**base, **kw})

    async with AsyncSessionLocal() as s:
        s.add_all(
            [
                row(
                    remote_node_id=peer_id,
                    remote_node_name=f"{tag}-pod",
                    duration_ms=1000,
                    federation_route="local_cold_peer_warm",
                ),
                row(
                    remote_node_id=peer_id,
                    remote_node_name=f"{tag}-pod",
                    duration_ms=3000,
                    federation_route="local_cold_peer_warm",
                ),
                row(
                    remote_node_id=peer_id,
                    remote_node_name=f"{tag}-pod",
                    status_code=502,
                    model_id=f"{tag}-m2",
                    federation_route="forced",
                ),
                row(federation_inbound_from=f"{tag}-mapache"),
                row(federation_route="fallback_local"),
            ]
        )
        s.add(
            CloudNode(
                provider="runpod",
                name=f"{tag}-pod",
                gpu_type_id="4090",
                image="test",
                cost_per_hr=1.2,
                status="running",
                # Terminated pods lose their peer (FK SET NULL): matched by name.
                peer_id=None,
                started_at=now - timedelta(hours=1),
            )
        )
        await s.commit()
    try:
        out = await get_federation_stats(now - timedelta(hours=2), now)
        pod = next(o for o in out["outbound"] if o["node"] == f"{tag}-pod")
        assert pod["count"] == 3 and pod["errors"] == 1 and pod["nodeId"] == peer_id
        assert pod["inputTokens"] == 300 and pod["outputTokens"] == 150
        assert pod["p50DurationMs"] == 1000 and pod["p95DurationMs"] >= 2700
        assert pod["topModels"][0] == {"model": f"{tag}-m1", "count": 2}
        # One hour at $1.2 for 3 requests / 450 tokens.
        assert pod["costUsd"] == pytest.approx(1.2, rel=0.02)
        assert pod["usdPerRequest"] == pytest.approx(0.4, rel=0.02)
        assert pod["usdPer1kTokens"] == pytest.approx(1.2 * 1000 / 450, rel=0.02)
        inbound = next(i for i in out["inbound"] if i["node"] == f"{tag}-mapache")
        assert inbound["count"] == 1 and "costUsd" not in inbound
        routes = {r["route"]: r["count"] for r in out["routes"]}
        assert routes["local_cold_peer_warm"] >= 2 and routes["fallback_local"] >= 1
        assert out["fallbackCount"] >= 1 and out["inboundCount"] >= 1
        assert any(b["nodeName"] == f"{tag}-pod" for b in out["byNode"])  # old shape kept
    finally:
        async with AsyncSessionLocal() as s:
            await s.execute(sa.delete(RequestStat).where(RequestStat.model_id.like(f"{tag}%")))
            await s.execute(sa.delete(CloudNode).where(CloudNode.name.like(f"{tag}%")))
            await s.commit()


async def test_lo_fijado_por_url_nunca_se_sirve_aqui_si_el_nodo_falla(_ctx):
    from fastapi import HTTPException

    peer = SimpleNamespace(name="ocabra-bench-ifeval-df3a7d")
    _ctx.update(force_node=peer.name, force_strict=True)
    # Túnel reconectando: 503 para que el runner reintente.
    with pytest.raises(HTTPException) as exc:
        note_federation_fallback(peer, ConnectionError("federation tunnel closed"))
    assert exc.value.status_code == 503 and exc.value.headers["Retry-After"] == "15"
    # El nodo contestó con un error: se devuelve ese, no se sirve aquí.
    with pytest.raises(HTTPException) as exc:
        note_federation_fallback(peer, 409)
    assert exc.value.status_code == 409
    assert _ctx.get("fallback") is None
    # Sin fijar por URL (X-Ocabra-Node o reparto normal): como siempre.
    _ctx.clear()
    note_federation_fallback(peer, 502)
    assert _ctx["fallback"] is True
