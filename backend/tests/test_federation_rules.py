"""«Mi servidor, mis reglas» (core/federation_rules.py)."""

from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace

import pytest

from ocabra.core import federation_rules as fr


def _peer(mode="rules", rules=None, peer_id="p1", name="vecino"):
    return SimpleNamespace(mode=mode, rules=rules, peer_id=peer_id, name=name)


def _fm(defaults=None, loaded=(), canonical=None):
    async def get_state(mid):
        return SimpleNamespace(model_id=mid, status="loaded" if mid in loaded else "configured")

    async def list_states():
        return []

    async def registry_get(name):
        return SimpleNamespace(base_model_id=(canonical or {}).get(name)) if canonical else None

    async def resolve_alias(name):
        return None

    return SimpleNamespace(
        default_rules=defaults or {},
        model_manager=SimpleNamespace(get_state=get_state, list_states=list_states),
        profile_registry=SimpleNamespace(get=registry_get, resolve_alias=resolve_alias),
    )


@pytest.fixture(autouse=True)
def _fresh_usage():
    fr._usage.clear()
    yield
    fr._usage.clear()


def test_swarm_means_everything_allowed():
    assert fr.effective_rules(_peer(mode="swarm"), {"allow_loads": False}) == fr.ALL_ALLOWED
    assert fr.effective_rules(_peer(mode="manual"), {}) == fr.ALL_ALLOWED


def test_rules_mode_layers_base_node_and_peer():
    rules = fr.effective_rules(
        _peer(rules={"allow_admin": True}), {"max_requests_per_hour": 10, "local_first": False}
    )
    assert rules.allow_admin is True  # peer adjustment
    assert rules.max_requests_per_hour == 10 and rules.local_first is False  # node
    assert rules.allow_downloads is False and rules.allow_evict is False  # base


def test_clean_rules():
    out = fr.clean_rules(
        {"max_requests_per_hour": "0", "hours": [22, 32], "offered_models": ["a", " "], "x": 1}
    )
    assert out == {"max_requests_per_hour": None, "hours": [22, 8], "offered_models": ["a"]}


@pytest.mark.parametrize(
    "hours, hour, ok",
    [
        (None, 3, True),
        ([9, 18], 10, True),
        ([9, 18], 20, False),
        ([22, 8], 23, True),
        ([22, 8], 3, True),
        ([22, 8], 12, False),
    ],
)
def test_within_hours(hours, hour, ok):
    assert fr.within_hours(hours, datetime(2026, 11, 16, hour)) is ok


async def test_swarm_peer_is_never_gated():
    assert await fr.gate(_fm(), _peer(mode="swarm"), "POST", "/ocabra/config", b"") is None


async def test_admin_read_only_and_downloads():
    fm, peer = _fm(), _peer()
    assert await fr.gate(fm, peer, "GET", "/ocabra/models", b"") is None
    status, body = await fr.gate(fm, peer, "POST", "/ocabra/models/x/load", b"")
    assert status == 403 and body["error"]["code"] == "read_only"
    status, body = await fr.gate(fm, peer, "POST", "/ocabra/downloads", b"{}")
    assert status == 403 and body["error"]["code"] == "downloads_not_allowed"
    status, _ = await fr.gate(fm, peer, "POST", "/ocabra/backends/diffusers/install", b"")
    assert status == 403


async def test_hourly_quota_is_a_503_so_the_peer_falls_back():
    fm, peer = _fm(), _peer(rules={"max_requests_per_hour": 2})
    body = json.dumps({"model": "m"}).encode()
    assert await fr.gate(fm, peer, "POST", "/v1/chat/completions", body) is None
    assert await fr.gate(fm, peer, "POST", "/v1/chat/completions", body) is None
    status, out = await fr.gate(fm, peer, "POST", "/v1/chat/completions", body)
    assert status == 503 and out["error"]["code"] == "hourly_quota"
    assert fr.requests_last_hour("p1") == 2  # refused ones don't count


async def test_energy_quota(monkeypatch):
    async def spent(fm, peer, **kw):
        return 12.5

    monkeypatch.setattr(fr, "energy_this_month", spent)
    status, out = await fr.gate(
        _fm(), _peer(rules={"max_wh_per_month": 10}), "POST", "/v1/embeddings", b"{}"
    )
    assert status == 503 and out["error"]["code"] == "energy_quota"


async def test_offered_models_by_public_name_or_id():
    fm = _fm(canonical={"gemma4:26b": "llamacpp_mks/gemma4:26b"})
    peer = _peer(rules={"offered_models": ["llamacpp_mks/gemma4:26b"]})
    ok = json.dumps({"model": "gemma4:26b"}).encode()
    other = json.dumps({"model": "qwen3:8b"}).encode()
    assert await fr.gate(fm, peer, "POST", "/v1/chat/completions", ok) is None
    status, out = await fr.gate(fm, peer, "POST", "/v1/chat/completions", other)
    assert status == 404 and out["error"]["code"] == "model_not_offered"


async def test_only_what_is_warm():
    peer = _peer(rules={"allow_loads": False})
    body = json.dumps({"model": "m"}).encode()
    status, out = await fr.gate(_fm(loaded=()), peer, "POST", "/v1/chat/completions", body)
    assert status == 503 and out["error"]["code"] == "model_not_warm"
    assert await fr.gate(_fm(loaded=("m",)), peer, "POST", "/v1/chat/completions", body) is None


async def test_outside_hours(monkeypatch):
    monkeypatch.setattr(fr, "within_hours", lambda hours, now=None: False)
    status, out = await fr.gate(_fm(), _peer(rules={"hours": [22, 8]}), "POST", "/v1/x", b"{}")
    assert status == 503 and out["error"]["code"] == "outside_hours"


def test_me_first_gives_peer_requests_the_batch_class():
    from ocabra.api.openai._deps import is_batch_origin_request

    rules = fr.PeerRules(local_first=True)
    request = SimpleNamespace(state=SimpleNamespace(federation_rules=rules), headers={})
    assert is_batch_origin_request(request) is True
    plain = SimpleNamespace(state=SimpleNamespace(), headers={})
    assert is_batch_origin_request(plain) is False
