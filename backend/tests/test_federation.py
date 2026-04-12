"""Tests for federation Phase 1: peer management, heartbeat, encryption."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ocabra.core.federation import FederationManager, PeerState, _derive_fernet_key


# ── Helpers ──────────────────────────────────────────────────────


def _make_settings(**overrides):
    """Create a minimal Settings-like object for testing."""
    defaults = {
        "federation_enabled": True,
        "federation_node_id": "test-node-id",
        "federation_node_name": "test-node",
        "federation_heartbeat_interval": 30,
        "federation_proxy_timeout_s": 300,
        "federation_verify_ssl": False,
        "jwt_secret": "test-secret-key-for-federation-tests",
        "app_version": "0.5.0-test",
    }
    defaults.update(overrides)
    return MagicMock(**defaults)


def _make_session_factory():
    """Create a mock async session factory."""
    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)

    # Mock query result (empty by default)
    result = MagicMock()
    result.scalars.return_value.all.return_value = []
    session.execute = AsyncMock(return_value=result)
    session.commit = AsyncMock()

    factory = MagicMock(return_value=session)
    factory._session = session
    return factory


def _make_peer_state(
    name: str = "peer-1",
    online: bool = True,
    active_requests: int = 0,
    gpu_util: float = 0.0,
    models: list | None = None,
) -> PeerState:
    """Create a PeerState for testing."""
    return PeerState(
        peer_id=str(uuid.uuid4()),
        name=name,
        url=f"https://{name}.local:8000",
        api_key="sk-test-key",
        access_level="inference",
        enabled=True,
        online=online,
        last_heartbeat=datetime.now(timezone.utc) if online else None,
        models=models or [],
        load={
            "active_requests": active_requests,
            "gpu_utilization_avg_pct": gpu_util,
        },
    )


# ── Tests ────────────────────────────────────────────────────────


class TestApiKeyEncryption:
    """Test that API keys are encrypted and decrypted correctly."""

    def test_encrypt_decrypt_round_trip(self):
        """Verify key is encrypted in DB and decrypted in PeerState."""
        settings = _make_settings()
        session_factory = _make_session_factory()
        fm = FederationManager(settings, session_factory)

        original_key = "sk-ocabra-test-api-key-12345"
        encrypted = fm.encrypt_api_key(original_key)

        # Encrypted value must differ from plaintext.
        assert encrypted != original_key
        assert len(encrypted) > len(original_key)

        # Decrypted value must match original.
        decrypted = fm.decrypt_api_key(encrypted)
        assert decrypted == original_key

    def test_different_secrets_produce_different_ciphertexts(self):
        """Different jwt_secret values produce different encryptions."""
        fm1 = FederationManager(_make_settings(jwt_secret="secret-a"), _make_session_factory())
        fm2 = FederationManager(_make_settings(jwt_secret="secret-b"), _make_session_factory())

        key = "sk-ocabra-test"
        enc1 = fm1.encrypt_api_key(key)
        enc2 = fm2.encrypt_api_key(key)

        assert enc1 != enc2

    def test_derive_fernet_key_deterministic(self):
        """Same secret always produces the same Fernet key."""
        key1 = _derive_fernet_key("test-secret")
        key2 = _derive_fernet_key("test-secret")
        assert key1 == key2


class TestFederationManagerInit:
    """Test FederationManager initialization."""

    def test_auto_generates_node_id_when_empty(self):
        """Node ID is auto-generated when not provided."""
        settings = _make_settings(federation_node_id="")
        fm = FederationManager(settings, _make_session_factory())
        assert fm.node_id
        assert len(fm.node_id) > 0

    def test_uses_provided_node_id(self):
        """Node ID uses the provided value."""
        settings = _make_settings(federation_node_id="custom-id")
        fm = FederationManager(settings, _make_session_factory())
        assert fm.node_id == "custom-id"

    def test_auto_generates_node_name(self):
        """Node name is auto-generated from node_id prefix."""
        settings = _make_settings(federation_node_name="", federation_node_id="abc-123")
        fm = FederationManager(settings, _make_session_factory())
        assert fm.node_name.startswith("node-")


class TestFindBestPeer:
    """Test load-based peer selection."""

    def test_find_best_peer_lowest_load(self):
        """Selects the peer with the lowest load score."""
        settings = _make_settings()
        fm = FederationManager(settings, _make_session_factory())

        model_id = "vllm/Qwen3-32B"
        peer_high = _make_peer_state(
            name="high-load",
            active_requests=10,
            gpu_util=80.0,
            models=[{"model_id": model_id, "status": "LOADED"}],
        )
        peer_low = _make_peer_state(
            name="low-load",
            active_requests=1,
            gpu_util=20.0,
            models=[{"model_id": model_id, "status": "LOADED"}],
        )

        fm._peers[peer_high.peer_id] = peer_high
        fm._peers[peer_low.peer_id] = peer_low

        best = fm.find_best_peer(model_id)
        assert best is not None
        assert best.name == "low-load"

    def test_find_best_peer_no_candidates(self):
        """Returns None when no peer has the model."""
        fm = FederationManager(_make_settings(), _make_session_factory())
        assert fm.find_best_peer("vllm/nonexistent") is None

    def test_select_target_prefers_local(self):
        """Local gets a bias, so it's preferred when load is similar."""
        fm = FederationManager(_make_settings(), _make_session_factory())

        model_id = "vllm/Qwen3-32B"
        peer = _make_peer_state(
            name="remote",
            active_requests=0,
            gpu_util=10.0,
            models=[{"model_id": model_id, "status": "LOADED"}],
        )
        fm._peers[peer.peer_id] = peer

        # Local available with similar load — should prefer local due to -5 bias
        target = fm.select_target(model_id, local_available=True)
        assert target == "local"

    def test_select_target_remote_only(self):
        """Returns remote peer when model is not available locally."""
        fm = FederationManager(_make_settings(), _make_session_factory())

        model_id = "vllm/Qwen3-32B"
        peer = _make_peer_state(
            name="remote",
            models=[{"model_id": model_id, "status": "LOADED"}],
        )
        fm._peers[peer.peer_id] = peer

        target = fm.select_target(model_id, local_available=False)
        assert isinstance(target, PeerState)
        assert target.name == "remote"


class TestOfflineDetection:
    """Test peer offline detection after consecutive failures."""

    @pytest.mark.asyncio
    async def test_offline_after_failures(self):
        """Peer is marked offline after 3 consecutive heartbeat failures."""
        settings = _make_settings()
        fm = FederationManager(settings, _make_session_factory())
        fm._http_client = AsyncMock()

        peer = _make_peer_state(name="flaky")
        peer.online = True
        fm._peers[peer.peer_id] = peer

        # Mock HTTP client to always fail
        fm._http_client.get = AsyncMock(side_effect=Exception("Connection refused"))

        # Poll 3 times
        for _ in range(3):
            await fm._poll_peer(peer)

        assert peer.online is False
        assert peer.consecutive_failures >= 3

    @pytest.mark.asyncio
    async def test_recovery_resets_failures(self):
        """Successful heartbeat resets failure counter and marks peer online."""
        settings = _make_settings()
        fm = FederationManager(settings, _make_session_factory())
        fm._http_client = AsyncMock()

        peer = _make_peer_state(name="recovering")
        peer.online = False
        peer.consecutive_failures = 5
        fm._peers[peer.peer_id] = peer

        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "node_id": "remote-id",
            "gpus": [],
            "models": [],
            "load": {},
        }
        fm._http_client.get = AsyncMock(return_value=mock_response)

        await fm._poll_peer(peer)

        assert peer.online is True
        assert peer.consecutive_failures == 0


class TestPeerCRUD:
    """Test peer add/remove/update operations."""

    @pytest.mark.asyncio
    async def test_add_peer_and_list(self):
        """Add a peer, verify it appears in the list."""
        fm = FederationManager(_make_settings(), _make_session_factory())

        state = await fm.add_peer(
            name="new-peer",
            url="https://new-peer.local:8000",
            api_key="sk-test",
            access_level="inference",
        )

        assert state.name == "new-peer"
        assert state.url == "https://new-peer.local:8000"
        assert state.access_level == "inference"

        all_peers = fm.get_all_peers()
        assert len(all_peers) == 1
        assert all_peers[0].name == "new-peer"

    @pytest.mark.asyncio
    async def test_remove_peer(self):
        """Add then remove a peer, verify it's gone."""
        fm = FederationManager(_make_settings(), _make_session_factory())

        # Mock delete to return rowcount=1
        session = fm._session_factory._session
        delete_result = MagicMock()
        delete_result.rowcount = 1
        session.execute = AsyncMock(return_value=delete_result)

        state = await fm.add_peer(
            name="temp-peer",
            url="https://temp.local:8000",
            api_key="sk-test",
        )
        assert len(fm.get_all_peers()) == 1

        removed = await fm.remove_peer(state.peer_id)
        assert removed is True
        assert len(fm.get_all_peers()) == 0

    @pytest.mark.asyncio
    async def test_update_peer(self):
        """Update a peer's name and access_level."""
        fm = FederationManager(_make_settings(), _make_session_factory())

        # Mock update to return rowcount=1
        session = fm._session_factory._session
        update_result = MagicMock()
        update_result.rowcount = 1
        session.execute = AsyncMock(return_value=update_result)

        state = await fm.add_peer(
            name="original",
            url="https://original.local:8000",
            api_key="sk-test",
        )

        updated = await fm.update_peer(
            state.peer_id,
            name="updated",
            access_level="full",
        )
        assert updated is not None
        assert updated.name == "updated"
        assert updated.access_level == "full"


class TestPeerConnectionTest:
    """Test the single-heartbeat connection test."""

    @pytest.mark.asyncio
    async def test_peer_connection_test_success(self):
        """Successful connection test returns node info and latency."""
        fm = FederationManager(_make_settings(), _make_session_factory())
        fm._http_client = AsyncMock()

        peer = _make_peer_state(name="testable")
        fm._peers[peer.peer_id] = peer

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "node_id": "remote-node",
            "node_name": "Remote Node",
        }
        fm._http_client.get = AsyncMock(return_value=mock_response)

        result = await fm.test_peer_connection(peer.peer_id)
        assert result["success"] is True
        assert result["node_id"] == "remote-node"
        assert result["node_name"] == "Remote Node"
        assert result["latency_ms"] is not None
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_peer_connection_test_failure(self):
        """Failed connection test returns error info."""
        fm = FederationManager(_make_settings(), _make_session_factory())
        fm._http_client = AsyncMock()

        peer = _make_peer_state(name="unreachable")
        fm._peers[peer.peer_id] = peer

        fm._http_client.get = AsyncMock(side_effect=Exception("Connection refused"))

        result = await fm.test_peer_connection(peer.peer_id)
        assert result["success"] is False
        assert result["error"] is not None
        assert "Connection refused" in result["error"]

    @pytest.mark.asyncio
    async def test_peer_connection_test_not_found(self):
        """Test for a non-existent peer returns error."""
        fm = FederationManager(_make_settings(), _make_session_factory())
        fm._http_client = AsyncMock()

        result = await fm.test_peer_connection("nonexistent-id")
        assert result["success"] is False
        assert result["error"] == "Peer not found"


class TestHeartbeatResponse:
    """Test heartbeat endpoint response structure."""

    @pytest.mark.asyncio
    async def test_heartbeat_response_structure(self):
        """Verify heartbeat returns proper structure with mocked managers."""
        from ocabra.schemas.federation import HeartbeatResponse

        # Build a mock heartbeat response manually to test schema validation
        response = HeartbeatResponse(
            node_id="test-node-id",
            node_name="test-node",
            version="0.5.0",
            uptime_seconds=3600.0,
            gpus=[
                {
                    "index": 0,
                    "name": "RTX 3060",
                    "total_vram_mb": 12288,
                    "free_vram_mb": 4096,
                }
            ],
            models=[
                {
                    "model_id": "vllm/Qwen3-32B",
                    "status": "LOADED",
                    "profiles": ["qwen3-32b"],
                }
            ],
            load={"active_requests": 3, "gpu_utilization_avg_pct": 45.0},
        )

        assert response.node_id == "test-node-id"
        assert response.node_name == "test-node"
        assert response.version == "0.5.0"
        assert response.uptime_seconds == 3600.0
        assert len(response.gpus) == 1
        assert response.gpus[0].name == "RTX 3060"
        assert len(response.models) == 1
        assert response.models[0].model_id == "vllm/Qwen3-32B"
        assert response.load.active_requests == 3
        assert response.load.gpu_utilization_avg_pct == 45.0


class TestGetRemoteModels:
    """Test remote model aggregation."""

    def test_get_remote_models(self):
        """Aggregates models from online peers."""
        fm = FederationManager(_make_settings(), _make_session_factory())

        model_id = "vllm/Qwen3-32B"
        peer1 = _make_peer_state(
            name="peer-1",
            models=[{"model_id": model_id, "status": "LOADED"}],
        )
        peer2 = _make_peer_state(
            name="peer-2",
            models=[
                {"model_id": model_id, "status": "LOADED"},
                {"model_id": "whisper/large-v3", "status": "LOADED"},
            ],
        )
        peer_offline = _make_peer_state(
            name="peer-offline",
            online=False,
            models=[{"model_id": model_id, "status": "LOADED"}],
        )

        fm._peers[peer1.peer_id] = peer1
        fm._peers[peer2.peer_id] = peer2
        fm._peers[peer_offline.peer_id] = peer_offline

        remote = fm.get_remote_models()
        assert model_id in remote
        assert len(remote[model_id]) == 2  # only online peers
        assert "whisper/large-v3" in remote

    def test_ignores_unloaded_models(self):
        """Does not include models with non-LOADED status."""
        fm = FederationManager(_make_settings(), _make_session_factory())

        peer = _make_peer_state(
            name="peer",
            models=[{"model_id": "vllm/test", "status": "UNLOADED"}],
        )
        fm._peers[peer.peer_id] = peer

        remote = fm.get_remote_models()
        assert len(remote) == 0
