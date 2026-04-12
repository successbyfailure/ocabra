"""
Tests for Phase 3 — Federated inventory in model listing endpoints.

Verifies that /v1/models, /api/tags, and /ocabra/models correctly merge
remote models from federated peers with deduplication.
"""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# ── Helpers ──────────────────────────────────────────────────────


def _make_peer(
    name: str,
    peer_id: str = "peer-uuid-1",
    models: list[dict] | None = None,
    load: dict | None = None,
    online: bool = True,
):
    """Create a SimpleNamespace mimicking a PeerState."""
    return SimpleNamespace(
        peer_id=peer_id,
        name=name,
        url=f"https://{name}.local:8000",
        api_key="sk-test",
        access_level="inference",
        enabled=True,
        online=online,
        last_heartbeat=datetime.now(timezone.utc),
        gpus=[],
        models=models or [],
        load=load or {"active_requests": 0, "gpu_utilization_avg_pct": 0.0},
        consecutive_failures=0,
    )


def _make_federation_manager(peers: list | None = None, remote_models: dict | None = None):
    """Create a mock FederationManager that returns pre-configured remote models."""
    fm = MagicMock()
    fm.node_id = "local-node-uuid"
    fm.node_name = "nodo-local"

    _peers = peers or []

    def _get_remote_models():
        if remote_models is not None:
            return remote_models
        # Build from peers' model lists
        model_map: dict[str, list] = {}
        for peer in _peers:
            if not peer.online:
                continue
            for m in peer.models:
                mid = m.get("model_id", "")
                status = m.get("status", "")
                if mid and status == "LOADED":
                    model_map.setdefault(mid, []).append(peer)
        return model_map

    fm.get_remote_models = _get_remote_models
    fm.get_online_peers = MagicMock(return_value=[p for p in _peers if p.online])
    fm.get_all_peers = MagicMock(return_value=_peers)
    return fm


def _make_test_profile(profile_id, base_model_id, category="llm", **kwargs):
    return SimpleNamespace(
        profile_id=profile_id,
        base_model_id=base_model_id,
        display_name=kwargs.get("display_name", profile_id),
        description=None,
        category=category,
        load_overrides=kwargs.get("load_overrides", {}),
        request_defaults=kwargs.get("request_defaults", {}),
        assets=kwargs.get("assets", {}),
        enabled=True,
        is_default=True,
    )


class FakeProfileRegistry:
    def __init__(self, profiles=None):
        self._profiles = {p.profile_id: p for p in (profiles or [])}

    async def get(self, profile_id):
        return self._profiles.get(profile_id)

    async def list_enabled(self):
        return [p for p in self._profiles.values() if p.enabled]

    async def list_by_model(self, base_model_id):
        return [p for p in self._profiles.values() if p.base_model_id == base_model_id]

    async def list_all(self):
        return list(self._profiles.values())


def _make_model_state(model_id, status="loaded", backend_type="vllm", vram_used_mb=4096):
    from ocabra.backends.base import BackendCapabilities
    from ocabra.core.model_manager import LoadPolicy, ModelState, ModelStatus

    return ModelState(
        model_id=model_id,
        backend_type=backend_type,
        backend_model_id=model_id,
        display_name=model_id.split("/")[-1],
        status=ModelStatus(status),
        load_policy=LoadPolicy.ON_DEMAND,
        capabilities=BackendCapabilities(chat=True, streaming=True),
        vram_used_mb=vram_used_mb,
    )


# ── App factories ────────────────────────────────────────────────


def _make_openai_app(
    model_states: list | None = None,
    profiles: list | None = None,
    federation_manager=None,
):
    """Create a minimal FastAPI app with OpenAI router and optional federation."""
    from ocabra.api._deps_auth import UserContext, get_current_user
    from ocabra.api.openai import router as openai_router
    from ocabra.api.openai._deps import get_openai_user

    _admin_ctx = UserContext(
        user_id=None,
        username="__test__",
        role="system_admin",
        group_ids=[],
        accessible_model_ids=set(),
        is_anonymous=False,
    )

    app = FastAPI()
    app.include_router(openai_router, prefix="/v1")

    async def _fake_user():
        return _admin_ctx

    app.dependency_overrides[get_current_user] = _fake_user
    app.dependency_overrides[get_openai_user] = _fake_user

    mm = MagicMock()
    states_map = {s.model_id: s for s in (model_states or [])}

    async def _get_state(mid):
        return states_map.get(mid)

    async def _list_states():
        return list(states_map.values())

    mm.get_state = _get_state
    mm.list_states = _list_states

    app.state.model_manager = mm
    app.state.profile_registry = FakeProfileRegistry(profiles or [])
    app.state.federation_manager = federation_manager

    return app


def _make_ollama_app(
    model_states: list | None = None,
    profiles: list | None = None,
    federation_manager=None,
):
    """Create a minimal FastAPI app with Ollama tags router and optional federation."""
    from ocabra.api._deps_auth import UserContext, get_current_user
    from ocabra.api.ollama.tags import router as tags_router
    from ocabra.api.ollama._shared import get_ollama_user

    _admin_ctx = UserContext(
        user_id=None,
        username="__test__",
        role="system_admin",
        group_ids=[],
        accessible_model_ids=set(),
        is_anonymous=False,
    )

    app = FastAPI()
    app.include_router(tags_router, prefix="/api")

    async def _fake_user():
        return _admin_ctx

    app.dependency_overrides[get_current_user] = _fake_user
    app.dependency_overrides[get_ollama_user] = _fake_user

    mm = MagicMock()
    states_map = {s.model_id: s for s in (model_states or [])}

    async def _get_state(mid):
        return states_map.get(mid)

    async def _list_states():
        return list(states_map.values())

    mm.get_state = _get_state
    mm.list_states = _list_states

    app.state.model_manager = mm
    app.state.profile_registry = FakeProfileRegistry(profiles or [])
    app.state.federation_manager = federation_manager

    return app


def _make_internal_app(
    model_states: list | None = None,
    profiles: list | None = None,
    federation_manager=None,
):
    """Create a minimal FastAPI app with internal models router and optional federation."""
    from ocabra.api._deps_auth import UserContext, get_current_user, require_role
    from ocabra.api.internal.models import router as models_router

    _admin_ctx = UserContext(
        user_id=None,
        username="__test__",
        role="system_admin",
        group_ids=[],
        accessible_model_ids=set(),
        is_anonymous=False,
    )

    app = FastAPI()
    app.include_router(models_router, prefix="/ocabra")

    async def _fake_user():
        return _admin_ctx

    app.dependency_overrides[get_current_user] = _fake_user

    mm = MagicMock()
    states_map = {s.model_id: s for s in (model_states or [])}

    async def _get_state(mid):
        return states_map.get(mid)

    async def _list_states():
        return list(states_map.values())

    mm.get_state = _get_state
    mm.list_states = _list_states
    mm.sync_ollama_inventory = MagicMock(return_value=None)

    # Make sync_ollama_inventory async-safe
    import asyncio

    async def _sync_ollama(*a, **kw):
        return None

    mm.sync_ollama_inventory = _sync_ollama

    app.state.model_manager = mm
    app.state.profile_registry = FakeProfileRegistry(profiles or [])
    app.state.federation_manager = federation_manager
    app.state.worker_pool = None

    return app


# ── Tests: /v1/models ────────────────────────────────────────────


class TestV1ModelsIncludesRemoteModels:
    """test_v1_models_includes_remote_models"""

    def test_remote_model_appears_in_listing(self):
        """Remote-only models from federation peers appear in /v1/models."""
        peer = _make_peer(
            "nodo-B",
            peer_id="peer-b-uuid",
            models=[
                {
                    "model_id": "vllm/Qwen3-32B",
                    "status": "LOADED",
                    "profiles": ["qwen3-32b"],
                }
            ],
        )
        fm = _make_federation_manager(peers=[peer])

        # No local models
        app = _make_openai_app(
            model_states=[],
            profiles=[],
            federation_manager=fm,
        )
        client = TestClient(app)
        resp = client.get("/v1/models")
        assert resp.status_code == 200

        data = resp.json()["data"]
        assert len(data) == 1
        model = data[0]
        assert model["id"] == "qwen3-32b"
        assert model["owned_by"] == "nodo-B"
        assert model["federation"]["remote"] is True
        assert model["federation"]["node_name"] == "nodo-B"
        assert model["federation"]["node_id"] == "peer-b-uuid"


class TestV1ModelsDeduplicatesSameModel:
    """test_v1_models_deduplicates_same_model"""

    def test_local_and_remote_same_model_appears_once(self):
        """A model available both locally and remotely appears once (local entry)."""
        local_state = _make_model_state("vllm/Qwen3-32B")
        profile = _make_test_profile("qwen3-32b", "vllm/Qwen3-32B")

        peer = _make_peer(
            "nodo-B",
            peer_id="peer-b-uuid",
            models=[
                {
                    "model_id": "vllm/Qwen3-32B",
                    "status": "LOADED",
                    "profiles": ["qwen3-32b"],
                }
            ],
        )
        fm = _make_federation_manager(peers=[peer])

        app = _make_openai_app(
            model_states=[local_state],
            profiles=[profile],
            federation_manager=fm,
        )
        client = TestClient(app)
        resp = client.get("/v1/models")
        assert resp.status_code == 200

        data = resp.json()["data"]
        # Should appear exactly once
        assert len(data) == 1
        model = data[0]
        assert model["id"] == "qwen3-32b"
        assert model["owned_by"] == "ocabra"  # local model
        # Should have federation annotation showing remote availability
        assert model["federation"]["remote"] is False
        assert len(model["federation"]["also_available_on"]) == 1
        assert model["federation"]["also_available_on"][0]["node_name"] == "nodo-B"


class TestV1ModelsNoRemoteWhenFederationDisabled:
    """test_v1_models_no_remote_when_federation_disabled"""

    def test_no_federation_field_when_disabled(self):
        """When federation is disabled, no federation metadata is added."""
        local_state = _make_model_state("vllm/Qwen3-8B")
        profile = _make_test_profile("qwen3-8b", "vllm/Qwen3-8B")

        app = _make_openai_app(
            model_states=[local_state],
            profiles=[profile],
            federation_manager=None,  # disabled
        )
        client = TestClient(app)
        resp = client.get("/v1/models")
        assert resp.status_code == 200

        data = resp.json()["data"]
        assert len(data) == 1
        model = data[0]
        assert "federation" not in model
        assert model["id"] == "qwen3-8b"


# ── Tests: /api/tags ─────────────────────────────────────────────


class TestApiTagsIncludesRemoteModels:
    """test_api_tags_includes_remote_models"""

    def test_remote_model_appears_in_tags(self):
        """Remote-only models from federation peers appear in /api/tags."""
        peer = _make_peer(
            "nodo-B",
            peer_id="peer-b-uuid",
            models=[
                {
                    "model_id": "vllm/Qwen3-32B",
                    "status": "LOADED",
                    "profiles": ["qwen3-32b"],
                }
            ],
        )
        fm = _make_federation_manager(peers=[peer])

        app = _make_ollama_app(
            model_states=[],
            profiles=[],
            federation_manager=fm,
        )
        client = TestClient(app)
        resp = client.get("/api/tags")
        assert resp.status_code == 200

        models = resp.json()["models"]
        assert len(models) == 1
        model = models[0]
        assert model["name"] == "qwen3-32b"
        assert model["federation"]["remote"] is True
        assert model["federation"]["node_name"] == "nodo-B"
        assert model["loaded"] is True

    def test_tags_dedup_local_and_remote(self):
        """A model in both local and remote appears once with federation metadata."""
        local_state = _make_model_state("vllm/Qwen3-8B")
        profile = _make_test_profile("qwen3-8b", "vllm/Qwen3-8B")

        peer = _make_peer(
            "nodo-C",
            peer_id="peer-c-uuid",
            models=[
                {
                    "model_id": "vllm/Qwen3-8B",
                    "status": "LOADED",
                    "profiles": ["qwen3-8b"],
                }
            ],
        )
        fm = _make_federation_manager(peers=[peer])

        app = _make_ollama_app(
            model_states=[local_state],
            profiles=[profile],
            federation_manager=fm,
        )
        client = TestClient(app)
        resp = client.get("/api/tags")
        assert resp.status_code == 200

        models = resp.json()["models"]
        assert len(models) == 1
        assert models[0]["name"] == "qwen3-8b"
        assert models[0]["federation"]["remote"] is False
        assert len(models[0]["federation"]["also_available_on"]) == 1


# ── Tests: /ocabra/models (admin) ───────────────────────────────


class TestAdminModelsShowsNodeInfo:
    """test_admin_models_shows_node_info"""

    def test_local_model_shows_federation_nodes(self):
        """When federation is enabled, local models show node availability."""
        local_state = _make_model_state("vllm/Qwen3-8B")
        profile = _make_test_profile("qwen3-8b", "vllm/Qwen3-8B")

        peer = _make_peer(
            "nodo-B",
            peer_id="peer-b-uuid",
            models=[
                {
                    "model_id": "vllm/Qwen3-8B",
                    "status": "LOADED",
                    "profiles": ["qwen3-8b"],
                }
            ],
        )
        fm = _make_federation_manager(peers=[peer])

        app = _make_internal_app(
            model_states=[local_state],
            profiles=[profile],
            federation_manager=fm,
        )
        client = TestClient(app)
        resp = client.get("/ocabra/models")
        assert resp.status_code == 200

        payloads = resp.json()
        assert len(payloads) == 1
        item = payloads[0]
        assert "federation" in item
        assert item["federation"]["local"] is True
        nodes = item["federation"]["nodes"]
        assert len(nodes) == 2  # local + nodo-B
        assert nodes[0]["node_name"] == "local"
        assert nodes[1]["node_name"] == "nodo-B"


# ── Tests: Remote-only in admin ──────────────────────────────────


class TestRemoteOnlyModelAppearsInListing:
    """test_remote_only_model_appears_in_listing"""

    def test_remote_only_model_in_admin(self):
        """Remote-only models appear as read-only entries in /ocabra/models."""
        peer = _make_peer(
            "nodo-B",
            peer_id="peer-b-uuid",
            models=[
                {
                    "model_id": "vllm/Llama-4-70B",
                    "status": "LOADED",
                    "profiles": ["llama4-70b"],
                }
            ],
        )
        fm = _make_federation_manager(peers=[peer])

        app = _make_internal_app(
            model_states=[],
            profiles=[],
            federation_manager=fm,
        )
        client = TestClient(app)
        resp = client.get("/ocabra/models")
        assert resp.status_code == 200

        payloads = resp.json()
        assert len(payloads) == 1
        item = payloads[0]
        assert item["model_id"] == "vllm/Llama-4-70B"
        assert item["status"] == "remote"
        assert item["federation"]["local"] is False
        assert item["federation"]["read_only"] is True
        assert len(item["federation"]["nodes"]) == 1
        assert item["federation"]["nodes"][0]["node_name"] == "nodo-B"

    def test_no_remote_models_when_federation_disabled(self):
        """No remote models or federation metadata when federation is disabled."""
        local_state = _make_model_state("vllm/Qwen3-8B")
        profile = _make_test_profile("qwen3-8b", "vllm/Qwen3-8B")

        app = _make_internal_app(
            model_states=[local_state],
            profiles=[profile],
            federation_manager=None,
        )
        client = TestClient(app)
        resp = client.get("/ocabra/models")
        assert resp.status_code == 200

        payloads = resp.json()
        assert len(payloads) == 1
        assert "federation" not in payloads[0]
