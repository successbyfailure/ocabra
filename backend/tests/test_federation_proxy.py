"""
Tests for Phase 2 of the federation feature: transparent inference proxy
and federated model resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from ocabra.core.federation import (
    FederationManager,
    PeerState,
    resolve_federated,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeModelStatus:
    LOADED = "loaded"
    CONFIGURED = "configured"
    UNLOADED = "unloaded"
    LOADING = "loading"
    ERROR = "error"


@dataclass
class _FakeModelState:
    model_id: str
    status: str = "loaded"
    backend_type: str = "vllm"
    backend_model_id: str = ""

    def __post_init__(self):
        if not self.backend_model_id:
            self.backend_model_id = self.model_id


class _FakeModelManager:
    def __init__(self, states: dict[str, _FakeModelState] | None = None):
        self._states = states or {}

    async def get_state(self, model_id: str):
        return self._states.get(model_id)


def _make_peer(
    peer_id: str = "peer-1",
    name: str = "nodo-B",
    url: str = "https://nodo-b.local:8000",
    api_key: str = "sk-test-key",
    online: bool = True,
    models: list[dict] | None = None,
    load: dict | None = None,
) -> PeerState:
    return PeerState(
        peer_id=peer_id,
        name=name,
        url=url,
        api_key=api_key,
        access_level="inference",
        online=online,
        models=models or [],
        load=load or {"active_requests": 0, "gpu_utilization_avg_pct": 0.0},
    )


def _make_fm_with_peers(*peers: PeerState) -> FederationManager:
    """Create a FederationManager with pre-populated peers (no DB)."""
    fm = object.__new__(FederationManager)
    fm._peers = {}
    fm._heartbeat_task = None
    fm._http_client = None
    fm._started_at = 0.0
    fm._node_id = "test-node"
    fm._node_name = "test"
    fm._settings = MagicMock()
    fm._settings.federation_proxy_timeout_s = 300
    fm._settings.federation_verify_ssl = True
    fm._session_factory = MagicMock()
    fm._fernet = MagicMock()
    for p in peers:
        fm._peers[p.peer_id] = p
    return fm


@pytest.fixture(autouse=True)
def _patch_model_status(monkeypatch):
    import ocabra.core.model_manager as mm_module
    monkeypatch.setattr(mm_module, "ModelStatus", _FakeModelStatus)


# ---------------------------------------------------------------------------
# Tests: resolve_federated
# ---------------------------------------------------------------------------


class TestResolveFederated:

    @pytest.mark.asyncio
    async def test_local_loaded(self):
        mm = _FakeModelManager({"my-model": _FakeModelState("my-model", status="loaded")})
        target, peer = await resolve_federated("my-model", mm, None)
        assert target == "local"
        assert peer is None

    @pytest.mark.asyncio
    async def test_local_exists_unloaded(self):
        mm = _FakeModelManager({"my-model": _FakeModelState("my-model", status="unloaded")})
        target, peer = await resolve_federated("my-model", mm, None)
        assert target == "local"
        assert peer is None

    @pytest.mark.asyncio
    async def test_local_exists_configured(self):
        mm = _FakeModelManager({"my-model": _FakeModelState("my-model", status="configured")})
        target, peer = await resolve_federated("my-model", mm, None)
        assert target == "local"
        assert peer is None

    @pytest.mark.asyncio
    async def test_remote_only(self):
        mm = _FakeModelManager({})
        peer = _make_peer(
            models=[{"model_id": "remote-model", "status": "LOADED", "profiles": []}],
        )
        fm = _make_fm_with_peers(peer)
        target, returned_peer = await resolve_federated("remote-model", mm, fm)
        assert target == "remote"
        assert returned_peer is peer

    @pytest.mark.asyncio
    async def test_not_found(self):
        mm = _FakeModelManager({})
        fm = _make_fm_with_peers()
        with pytest.raises(Exception) as exc_info:
            await resolve_federated("missing-model", mm, fm)
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_not_found_no_federation(self):
        mm = _FakeModelManager({})
        with pytest.raises(Exception) as exc_info:
            await resolve_federated("missing-model", mm, None)
        assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# Tests: select_target
# ---------------------------------------------------------------------------


class TestSelectTarget:

    def test_prefers_local(self):
        peer = _make_peer(
            models=[{"model_id": "shared-model", "status": "LOADED", "profiles": []}],
            load={"active_requests": 0, "gpu_utilization_avg_pct": 3.0},
        )
        fm = _make_fm_with_peers(peer)
        result = fm.select_target("shared-model", local_available=True)
        assert result == "local"

    def test_picks_least_loaded_peer(self):
        peer_a = _make_peer(
            peer_id="a",
            models=[{"model_id": "model-x", "status": "LOADED", "profiles": []}],
            load={"active_requests": 5, "gpu_utilization_avg_pct": 80.0},
        )
        peer_b = _make_peer(
            peer_id="b",
            models=[{"model_id": "model-x", "status": "LOADED", "profiles": []}],
            load={"active_requests": 1, "gpu_utilization_avg_pct": 20.0},
        )
        fm = _make_fm_with_peers(peer_a, peer_b)
        result = fm.select_target("model-x", local_available=False)
        assert result is peer_b

    def test_only_local(self):
        fm = _make_fm_with_peers()
        result = fm.select_target("local-only", local_available=True)
        assert result == "local"

    def test_nowhere(self):
        fm = _make_fm_with_peers()
        result = fm.select_target("nonexistent", local_available=False)
        assert result is None


# ---------------------------------------------------------------------------
# Tests: proxy_request
# ---------------------------------------------------------------------------


class TestProxyRequest:

    @pytest.mark.asyncio
    async def test_forwards_correctly(self):
        peer = _make_peer(url="https://remote:8000", api_key="sk-federation-key")
        fm = _make_fm_with_peers(peer)

        mock_response = httpx.Response(
            200,
            json={"choices": [{"message": {"content": "hello"}}]},
            request=httpx.Request("POST", "https://remote:8000/v1/chat/completions"),
        )

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        fm._http_client = mock_client

        resp = await fm.proxy_request(
            peer, "/v1/chat/completions", {"model": "qwen3-32b"}, {},
        )

        mock_client.post.assert_awaited_once()
        call_kwargs = mock_client.post.call_args[1]
        assert call_kwargs["headers"]["Authorization"] == "Bearer sk-federation-key"
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_strips_auth_header(self):
        peer = _make_peer(url="https://remote:8000", api_key="sk-key")
        fm = _make_fm_with_peers(peer)

        mock_response = httpx.Response(
            200, json={},
            request=httpx.Request("POST", "https://remote:8000/v1/embeddings"),
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        fm._http_client = mock_client

        await fm.proxy_request(
            peer, "/v1/embeddings", {},
            headers={"authorization": "Bearer user-key", "x-custom": "keep"},
        )

        call_kwargs = mock_client.post.call_args[1]
        assert call_kwargs["headers"]["Authorization"] == "Bearer sk-key"
        assert call_kwargs["headers"]["x-custom"] == "keep"


# ---------------------------------------------------------------------------
# Tests: proxy_stream
# ---------------------------------------------------------------------------


class TestProxyStream:

    @pytest.mark.asyncio
    async def test_yields_chunks(self):
        peer = _make_peer(url="https://remote:8000", api_key="sk-key")
        fm = _make_fm_with_peers(peer)

        chunks = [b"data: chunk1\n\n", b"data: chunk2\n\n", b"data: [DONE]\n\n"]

        mock_resp = AsyncMock()
        mock_resp.raise_for_status = MagicMock()

        async def fake_aiter_bytes():
            for c in chunks:
                yield c

        mock_resp.aiter_bytes = fake_aiter_bytes

        mock_stream_cm = AsyncMock()
        mock_stream_cm.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_stream_cm.__aexit__ = AsyncMock(return_value=False)

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=mock_stream_cm)
        fm._http_client = mock_client

        received = []
        async for chunk in fm.proxy_stream(peer, "/v1/chat/completions", {}, {}):
            received.append(chunk)

        assert received == chunks


# ---------------------------------------------------------------------------
# Tests: federation disabled
# ---------------------------------------------------------------------------


class TestFederationDisabled:

    @pytest.mark.asyncio
    async def test_federation_disabled_skips_remote(self):
        mm = _FakeModelManager({"my-model": _FakeModelState("my-model", status="loaded")})
        target, peer = await resolve_federated("my-model", mm, None)
        assert target == "local"
        assert peer is None

    def test_get_federation_manager_returns_none_when_not_set(self):
        from ocabra.api.openai._deps import get_federation_manager

        mock_request = MagicMock()
        mock_request.app.state = MagicMock(spec=[])
        del mock_request.app.state.federation_manager
        result = get_federation_manager(mock_request)
        assert result is None


# ---------------------------------------------------------------------------
# Tests: FederationManager basics (no DB)
# ---------------------------------------------------------------------------


class TestFederationManagerBasics:

    def test_peer_management(self):
        peer = _make_peer(peer_id="p1")
        fm = _make_fm_with_peers(peer)
        assert len(fm.get_online_peers()) == 1

        fm._peers.pop("p1")
        assert len(fm.get_online_peers()) == 0

    def test_get_remote_models(self):
        peer = _make_peer(
            models=[
                {"model_id": "vllm/Qwen3-32B", "status": "LOADED"},
                {"model_id": "whisper/large-v3", "status": "UNLOADED"},
            ],
        )
        fm = _make_fm_with_peers(peer)
        remote = fm.get_remote_models()
        assert "vllm/Qwen3-32B" in remote
        assert "whisper/large-v3" not in remote

    def test_get_remote_models_offline_peer(self):
        peer = _make_peer(
            online=False,
            models=[{"model_id": "vllm/Qwen3-32B", "status": "LOADED"}],
        )
        fm = _make_fm_with_peers(peer)
        remote = fm.get_remote_models()
        assert len(remote) == 0
