"""Tests for Phase 5 of federation: remote operations for peers with access_level='full'.

Tests cover:
- Access level validation (inference peer gets 403)
- Remote model load/unload proxy
- Remote download proxy (SSE streaming)
- Remote GPU monitoring proxy
- Peer not found (404)
- Peer offline (502)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from ocabra.core.federation import FederationManager, PeerState


# ── Helpers ──────────────────────────────────────────────────────


def _make_peer(
    peer_id: str | None = None,
    name: str = "peer-full",
    access_level: str = "full",
    online: bool = True,
) -> PeerState:
    """Create a PeerState for testing."""
    return PeerState(
        peer_id=peer_id or str(uuid.uuid4()),
        name=name,
        url=f"https://{name}.local:8000",
        api_key="sk-test-key",
        access_level=access_level,
        enabled=True,
        online=online,
        last_heartbeat=datetime.now(timezone.utc) if online else None,
        models=[{"model_id": "vllm/test-model", "status": "LOADED"}],
        load={"active_requests": 0, "gpu_utilization_avg_pct": 10.0},
    )


def _make_fm_with_peers(*peers: PeerState) -> FederationManager:
    """Create a FederationManager with pre-populated peers (no DB)."""
    fm = object.__new__(FederationManager)
    fm._peers = {}
    fm._heartbeat_task = None
    fm._http_client = AsyncMock()
    fm._started_at = 1.0
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


def _mock_app(fm: FederationManager):
    """Create a mock app with federation_manager on state."""
    app = MagicMock()
    app.state.federation_manager = fm
    return app


def _mock_request(app):
    """Create a mock Request with the given app."""
    req = MagicMock()
    req.app = app
    req.headers = {"content-type": "application/json", "authorization": "Bearer user-key"}
    return req


# ── Import the router module under test ─────────────────────────


@pytest.fixture()
def peer_full():
    return _make_peer(access_level="full", online=True)


@pytest.fixture()
def peer_inference():
    return _make_peer(name="peer-inference", access_level="inference", online=True)


@pytest.fixture()
def peer_offline():
    return _make_peer(name="peer-offline", access_level="full", online=False)


# ── Tests: access level validation ──────────────────────────────


@pytest.mark.asyncio
async def test_remote_load_requires_full_access(peer_inference):
    """Peer with access_level='inference' should get 403 on remote load."""
    from fastapi import HTTPException

    from ocabra.api.internal.federation import _get_full_access_peer

    fm = _make_fm_with_peers(peer_inference)
    app = _mock_app(fm)
    request = _mock_request(app)

    with pytest.raises(HTTPException) as exc_info:
        await _get_full_access_peer(peer_inference.peer_id, request)
    assert exc_info.value.status_code == 403
    assert "full" in str(exc_info.value.detail).lower()


# ── Tests: remote load ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_remote_load_proxies_to_peer(peer_full):
    """Remote load should proxy POST /ocabra/models/{id}/load to the peer."""
    fm = _make_fm_with_peers(peer_full)

    mock_response = MagicMock()
    mock_response.json.return_value = {"status": "loading", "model_id": "vllm/test-model"}
    fm.proxy_request = AsyncMock(return_value=mock_response)

    app = _mock_app(fm)
    request = _mock_request(app)

    from ocabra.api.internal.federation import _get_full_access_peer, remote_load_model

    # Directly call the endpoint function
    with patch(
        "ocabra.api.internal.federation._get_full_access_peer",
        new=AsyncMock(return_value=peer_full),
    ), patch(
        "ocabra.api.internal.federation._get_federation_manager",
        return_value=fm,
    ):
        result = await remote_load_model(
            peer_id=peer_full.peer_id,
            model_id="vllm/test-model",
            request=request,
            body=None,
            _user=MagicMock(),
        )

    fm.proxy_request.assert_awaited_once()
    call_kwargs = fm.proxy_request.call_args
    assert call_kwargs.kwargs["path"] == "/ocabra/models/vllm/test-model/load"
    assert result == {"status": "loading", "model_id": "vllm/test-model"}


# ── Tests: remote unload ────────────────────────────────────────


@pytest.mark.asyncio
async def test_remote_unload_proxies_to_peer(peer_full):
    """Remote unload should proxy POST /ocabra/models/{id}/unload to the peer."""
    fm = _make_fm_with_peers(peer_full)

    mock_response = MagicMock()
    mock_response.json.return_value = {"status": "unloaded", "model_id": "vllm/test-model"}
    fm.proxy_request = AsyncMock(return_value=mock_response)

    app = _mock_app(fm)
    request = _mock_request(app)

    from ocabra.api.internal.federation import remote_unload_model

    with patch(
        "ocabra.api.internal.federation._get_full_access_peer",
        new=AsyncMock(return_value=peer_full),
    ), patch(
        "ocabra.api.internal.federation._get_federation_manager",
        return_value=fm,
    ):
        result = await remote_unload_model(
            peer_id=peer_full.peer_id,
            model_id="vllm/test-model",
            request=request,
            _user=MagicMock(),
        )

    fm.proxy_request.assert_awaited_once()
    call_kwargs = fm.proxy_request.call_args
    assert call_kwargs.kwargs["path"] == "/ocabra/models/vllm/test-model/unload"
    assert result == {"status": "unloaded", "model_id": "vllm/test-model"}


# ── Tests: remote download ──────────────────────────────────────


@pytest.mark.asyncio
async def test_remote_download_proxies_to_peer(peer_full):
    """Remote download should proxy POST /ocabra/downloads and return SSE stream."""
    fm = _make_fm_with_peers(peer_full)

    chunks = [b"data: {\"progress\": 50}\n\n", b"data: {\"progress\": 100}\n\n"]

    async def _fake_stream(*args, **kwargs):
        for c in chunks:
            yield c

    fm.proxy_stream = _fake_stream

    app = _mock_app(fm)
    request = _mock_request(app)

    from ocabra.schemas.federation import RemoteDownloadRequest
    from ocabra.api.internal.federation import remote_download

    body = RemoteDownloadRequest(
        source="huggingface",
        model_ref="org/test-model",
    )

    with patch(
        "ocabra.api.internal.federation._get_full_access_peer",
        new=AsyncMock(return_value=peer_full),
    ), patch(
        "ocabra.api.internal.federation._get_federation_manager",
        return_value=fm,
    ):
        response = await remote_download(
            peer_id=peer_full.peer_id,
            body=body,
            request=request,
            _user=MagicMock(),
        )

    assert response.media_type == "text/event-stream"
    # Collect the streamed chunks
    collected = []
    async for chunk in response.body_iterator:
        collected.append(chunk)
    assert len(collected) == 2
    assert b"progress" in collected[0]


# ── Tests: remote GPUs ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_remote_gpus_proxies_to_peer(peer_full):
    """Remote GPUs should proxy GET /ocabra/gpus to the peer."""
    fm = _make_fm_with_peers(peer_full)

    gpu_data = [
        {"index": 0, "name": "RTX 3090", "total_vram_mb": 24576, "free_vram_mb": 12000},
        {"index": 1, "name": "RTX 3060", "total_vram_mb": 12288, "free_vram_mb": 8000},
    ]
    mock_response = MagicMock()
    mock_response.json.return_value = gpu_data
    fm.proxy_request = AsyncMock(return_value=mock_response)

    app = _mock_app(fm)
    request = _mock_request(app)

    from ocabra.api.internal.federation import remote_gpus

    with patch(
        "ocabra.api.internal.federation._get_full_access_peer",
        new=AsyncMock(return_value=peer_full),
    ), patch(
        "ocabra.api.internal.federation._get_federation_manager",
        return_value=fm,
    ):
        result = await remote_gpus(
            peer_id=peer_full.peer_id,
            request=request,
            _user=MagicMock(),
        )

    fm.proxy_request.assert_awaited_once()
    call_kwargs = fm.proxy_request.call_args
    assert call_kwargs.kwargs["path"] == "/ocabra/gpus"
    assert result == gpu_data


# ── Tests: peer not found ───────────────────────────────────────


@pytest.mark.asyncio
async def test_remote_op_peer_not_found():
    """Non-existent peer_id should return 404."""
    from fastapi import HTTPException

    from ocabra.api.internal.federation import _get_full_access_peer

    fm = _make_fm_with_peers()  # no peers
    app = _mock_app(fm)
    request = _mock_request(app)

    with pytest.raises(HTTPException) as exc_info:
        await _get_full_access_peer("nonexistent-peer-id", request)
    assert exc_info.value.status_code == 404


# ── Tests: peer offline ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_remote_op_peer_offline(peer_offline):
    """Offline peer with full access should return 502."""
    from fastapi import HTTPException

    from ocabra.api.internal.federation import _get_full_access_peer

    fm = _make_fm_with_peers(peer_offline)
    app = _mock_app(fm)
    request = _mock_request(app)

    with pytest.raises(HTTPException) as exc_info:
        await _get_full_access_peer(peer_offline.peer_id, request)
    assert exc_info.value.status_code == 502
    assert "offline" in str(exc_info.value.detail).lower()
