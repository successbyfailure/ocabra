"""Per-model federation strategies and the summed catalog (core/federation_policy.py)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import ocabra.core.federation as fed
from ocabra.core import federation_policy as fp
from ocabra.core.federation import federation_hop, resolve_federated
from ocabra.core.model_manager import ModelStatus as _RealModelStatus
from tests.test_federation_proxy import (
    _FakeModelManager,
    _FakeModelState,
    _make_fm_with_peers,
    _make_peer,
)

EMB = "llamacpp_mks/qwen3-embedding:8b"
BIG = "llamacpp_mks/kimi-k3"


@pytest.fixture(autouse=True)
def _clean():
    fp.POLICIES.clear()
    yield
    fp.POLICIES.clear()


@pytest.fixture(autouse=True)
def _status(monkeypatch):
    import ocabra.core.model_manager as mm_module
    from tests.test_federation_proxy import _FakeModelStatus

    monkeypatch.setattr(mm_module, "ModelStatus", _FakeModelStatus)


def _mapache(status="configured", **kw):
    return _make_peer(
        peer_id="mapache-id",
        name="Mapache",
        models=[{"model_id": EMB, "status": status, "profiles": ["qwen3-embedding:8b"]}],
        **kw,
    )


# ── clean_policy ────────────────────────────────────────────────


def test_clean_policy():
    p = fp.clean_policy({"strategy": "prefer", "nodes": ["Mapache", " LOCAL ", "mapache", ""]})
    assert p.nodes == ("Mapache", "local")
    assert fp.clean_policy({"strategy": "local_only", "nodes": ["x"]}).nodes == ()
    with pytest.raises(ValueError):
        fp.clean_policy({"strategy": "prefer", "nodes": []})
    with pytest.raises(ValueError):
        fp.clean_policy({"strategy": "round_robin"})


# ── prefer ──────────────────────────────────────────────────────


async def test_prefer_goes_to_the_peer_even_cold_and_even_if_warm_here():
    """«I want the embeddings on Mapache»: cold there beats warm here."""
    fp.POLICIES[EMB] = fp.FederationPolicy("prefer", ("Mapache", "local"))
    mapache = _mapache("configured")
    mm = _FakeModelManager({EMB: _FakeModelState(EMB, status="loaded")})
    target, peer = await resolve_federated(EMB, mm, _make_fm_with_peers(mapache))
    assert (target, peer) == ("remote", mapache)


async def test_prefer_falls_to_the_next_node_when_the_peer_is_offline():
    fp.POLICIES[EMB] = fp.FederationPolicy("prefer", ("Mapache", "local"))
    mm = _FakeModelManager({EMB: _FakeModelState(EMB, status="configured")})
    fm = _make_fm_with_peers(_mapache(online=False))
    assert await resolve_federated(EMB, mm, fm) == ("local", None)


async def test_prefer_without_fallback_refuses_with_503():
    fp.POLICIES[EMB] = fp.FederationPolicy("prefer", ("Mapache",), fallback=False)
    mm = _FakeModelManager({EMB: _FakeModelState(EMB, status="loaded")})
    with pytest.raises(HTTPException) as exc:
        await resolve_federated(EMB, mm, _make_fm_with_peers(_mapache(online=False)))
    assert exc.value.status_code == 503
    assert exc.value.headers["Retry-After"]


async def test_prefer_with_fallback_goes_on_as_auto():
    fp.POLICIES[EMB] = fp.FederationPolicy("prefer", ("Mapache",))
    mm = _FakeModelManager({EMB: _FakeModelState(EMB, status="loaded")})
    fm = _make_fm_with_peers(_mapache(online=False))
    assert await resolve_federated(EMB, mm, fm) == ("local", None)


async def test_a_request_from_a_peer_ignores_our_strategy():
    """Mapache's own requests to us must be served here, never bounced back."""
    fp.POLICIES[EMB] = fp.FederationPolicy("prefer", ("Mapache",), fallback=False)
    mm = _FakeModelManager({EMB: _FakeModelState(EMB, status="configured")})
    token = federation_hop.set("mapache-id")
    try:
        assert await resolve_federated(EMB, mm, _make_fm_with_peers(_mapache())) == ("local", None)
    finally:
        federation_hop.reset(token)


# ── local_only ──────────────────────────────────────────────────


async def test_local_only_never_leaves_even_if_a_peer_has_it_warm():
    fp.POLICIES[EMB] = fp.FederationPolicy("local_only")
    mm = _FakeModelManager({EMB: _FakeModelState(EMB, status="configured")})
    fm = _make_fm_with_peers(_mapache("loaded"))
    assert await resolve_federated(EMB, mm, fm) == ("local", None)


# ── auto + the summed catalog ───────────────────────────────────


async def test_auto_does_not_send_a_local_model_to_a_cold_peer():
    mm = _FakeModelManager({EMB: _FakeModelState(EMB, status="configured")})
    assert await resolve_federated(EMB, mm, _make_fm_with_peers(_mapache())) == ("local", None)


async def test_a_model_only_a_peer_has_is_served_there_even_cold():
    pod = _make_peer(peer_id="pod", name="pod", models=[{"model_id": BIG, "status": "configured"}])
    target, peer = await resolve_federated(BIG, _FakeModelManager(), _make_fm_with_peers(pod))
    assert (target, peer) == ("remote", pod)


async def test_a_peer_only_model_resolves_by_the_public_name_it_advertises():
    """/v1/models lists it as ``kimi-k3``: asking for that must reach the pod."""
    pod = _make_peer(
        peer_id="pod",
        name="pod",
        models=[{"model_id": BIG, "status": "loaded", "profiles": ["kimi-k3"]}],
    )
    resolved: dict = {}
    target, peer = await resolve_federated(
        "kimi-k3", _FakeModelManager(), _make_fm_with_peers(pod), resolved=resolved
    )
    assert (target, peer) == ("remote", pod)
    assert resolved["model_id"] == BIG


def test_catalog_has_cold_models_but_remote_models_stays_warm_only():
    fm = _make_fm_with_peers(_mapache("configured"))
    assert EMB in fm.get_remote_catalog()
    assert fm.get_remote_models() == {}
    assert fm.remote_model_for("qwen3-embedding:8b") == EMB
    assert fm.remote_model_for("nope") is None


# ── routers with peer-only targets ──────────────────────────────


def _sota_registry():
    profiles = {
        "recomendado-sota": SimpleNamespace(
            profile_id="recomendado-sota",
            profile_kind="router",
            routing_targets=["kimi-k3", "gemma4:26b"],
            base_model_id=None,
            runtime_presets=[],
        ),
        "gemma4:26b": SimpleNamespace(
            profile_id="gemma4:26b",
            profile_kind="execution",
            base_model_id="llamacpp_mks/gemma4:26b",
            runtime_presets=[],
        ),
    }

    class _Registry:
        async def get(self, pid):
            return profiles.get(pid)

        async def resolve_alias(self, pid):
            return None

        async def list_by_model(self, mid):
            return []

    return _Registry()


async def test_router_target_on_a_pod_wins_when_it_is_first_and_warm():
    pod = _make_peer(
        peer_id="pod",
        name="pod",
        models=[{"model_id": BIG, "status": "loaded", "profiles": ["kimi-k3"]}],
    )
    mm = _FakeModelManager(
        {"llamacpp_mks/gemma4:26b": _FakeModelState("llamacpp_mks/gemma4:26b", status="loaded")}
    )
    resolved: dict = {}
    target, peer = await resolve_federated(
        "recomendado-sota", mm, _make_fm_with_peers(pod), _sota_registry(), resolved=resolved
    )
    assert (target, peer) == ("remote", pod)
    assert resolved["model_id"] == BIG
    assert resolved["profile"].profile_id == "kimi-k3"


async def test_router_skips_the_pod_target_while_the_pod_is_offline():
    mm = _FakeModelManager(
        {"llamacpp_mks/gemma4:26b": _FakeModelState("llamacpp_mks/gemma4:26b", status="loaded")}
    )
    resolved: dict = {}
    target, _ = await resolve_federated(
        "recomendado-sota", mm, _make_fm_with_peers(), _sota_registry(), resolved=resolved
    )
    assert target == "local"
    assert resolved["model_id"] == "llamacpp_mks/gemma4:26b"


async def test_router_pod_target_filtered_by_its_advertised_capabilities(monkeypatch):
    pod = _make_peer(
        peer_id="pod",
        name="pod",
        models=[{"model_id": BIG, "status": "loaded", "capabilities": {"chat": True}}],
    )
    mm = _FakeModelManager(
        {"llamacpp_mks/gemma4:26b": _FakeModelState("llamacpp_mks/gemma4:26b", status="loaded")}
    )

    async def _capable(_mm, profile, capability):
        return True

    monkeypatch.setattr(fed, "_profile_capable", _capable)
    resolved: dict = {}
    await resolve_federated(
        "recomendado-sota",
        mm,
        _make_fm_with_peers(pod),
        _sota_registry(),
        required_capability="vision",
        resolved=resolved,
    )
    assert resolved["model_id"] == "llamacpp_mks/gemma4:26b"


async def test_body_for_peer_names_the_chosen_target_not_the_router():
    from ocabra.api.openai._federation import body_for_peer

    pod = _make_peer(peer_id="pod", name="pod", models=[{"model_id": BIG, "status": "loaded"}])
    resolved = {"model_id": BIG, "profile": fed._RemoteTarget("kimi-k3", BIG)}
    body, profile = await body_for_peer(
        pod, "recomendado-sota", {"model": "recomendado-sota"}, resolved, _sota_registry()
    )
    assert body["model"] == BIG
    assert profile.base_model_id == BIG  # the canonical retry names it too


async def test_router_validation_accepts_a_peer_only_target():
    from ocabra.core.profile_registry import ProfileRegistry

    reg = ProfileRegistry()
    reg._profiles["gemma4:26b"] = SimpleNamespace()
    with pytest.raises(ValueError):
        await reg._validate_router_targets("r", ["kimi-k3", "gemma4:26b"])
    reg.remote_target_lookup = lambda name: BIG if name == "kimi-k3" else None
    await reg._validate_router_targets("r", ["kimi-k3", "gemma4:26b"])


# ── heartbeat: cold models are offered too ──────────────────────


def _hb_request(rules):
    from ocabra.core.model_manager import BackendCapabilities, ModelState, ModelStatus

    states = [
        ModelState(
            model_id=EMB, display_name="emb", backend_type="llama_cpp", status=ModelStatus.LOADED
        ),
        ModelState(
            model_id=BIG,
            display_name="big",
            backend_type="llama_cpp",
            status=ModelStatus.CONFIGURED,
        ),
        ModelState(
            model_id="x/broken", display_name="b", backend_type="vllm", status=ModelStatus.ERROR
        ),
        ModelState(
            model_id=f"{BIG}::128k",
            display_name="v",
            backend_type="llama_cpp",
            status=ModelStatus.CONFIGURED,
        ),
    ]
    # Real capabilities carry numbers too: 0.97.0 sent them as-is, the
    # heartbeat schema (bools) rejected them and the model list came out empty.
    states[0].capabilities = BackendCapabilities(embeddings=True, context_length=40960)
    states[1].capabilities = BackendCapabilities(chat=True, tools=True, context_length=262144)

    async def list_states(include_derived=False):
        return states

    class _Registry:
        async def list_by_model(self, mid):
            return [SimpleNamespace(profile_id="kimi-k3", enabled=True)] if mid == BIG else []

    fm = SimpleNamespace(
        rules_for_user=lambda uid: rules, node_id="n", node_name="hub", uptime_seconds=1.0
    )
    app = SimpleNamespace(
        state=SimpleNamespace(
            federation_manager=fm,
            gpu_manager=None,
            model_manager=SimpleNamespace(list_states=list_states, active_request_count=0),
            profile_registry=_Registry(),
        )
    )
    return SimpleNamespace(app=app)


async def test_heartbeat_offers_cold_models_with_capabilities(monkeypatch):
    from ocabra.api.internal import federation as api

    monkeypatch.setattr("ocabra.core.cuda_probe.probe_cuda", lambda: None)
    monkeypatch.setattr("ocabra.core.model_manager.ModelStatus", _RealModelStatus)
    hb = await api.heartbeat(_hb_request(None), _user=SimpleNamespace(user_id="u"))
    by_id = {m.model_id: m for m in hb.models}
    assert set(by_id) == {EMB, BIG}  # no ERROR, no derived variants
    assert by_id[BIG].status == "configured"
    assert by_id[BIG].profiles == ["kimi-k3"]
    assert by_id[BIG].capabilities["tools"] is True
    assert "context_length" not in by_id[BIG].capabilities
    assert by_id[EMB].capabilities["embeddings"] is True


async def test_heartbeat_only_warm_when_the_peer_may_not_load(monkeypatch):
    from ocabra.api.internal import federation as api
    from ocabra.core.federation_rules import PeerRules

    monkeypatch.setattr("ocabra.core.cuda_probe.probe_cuda", lambda: None)
    monkeypatch.setattr("ocabra.core.model_manager.ModelStatus", _RealModelStatus)
    hb = await api.heartbeat(
        _hb_request(PeerRules(allow_loads=False)), _user=SimpleNamespace(user_id="u")
    )
    assert [m.model_id for m in hb.models] == [EMB]
