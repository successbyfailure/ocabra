"""
Tests for Phase 2 of the federation feature: transparent inference proxy
and federated model resolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from ocabra.config import settings
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
    async def test_a_model_outside_the_users_groups_is_not_found_even_if_warm_on_a_peer(self):
        """SEG-08: federation ran before the group check, so a model warm on a
        peer was served to any key. Denied is a 404, like an unknown model."""
        from fastapi import HTTPException

        from ocabra.api._deps_auth import UserContext

        mm = _FakeModelManager({})
        peer = _make_peer(
            models=[{"model_id": "remote-model", "status": "LOADED", "profiles": []}],
        )
        fm = _make_fm_with_peers(peer)
        outsider = UserContext(user_id="u", username="u", role="user",
                               accessible_model_ids={"other-model"})
        member = UserContext(user_id="m", username="m", role="user",
                             accessible_model_ids={"remote-model"})

        with pytest.raises(HTTPException) as exc:
            await resolve_federated("remote-model", mm, fm, user=outsider)
        assert exc.value.status_code == 404
        assert exc.value.detail["error"]["code"] == "model_not_found"

        target, returned_peer = await resolve_federated("remote-model", mm, fm, user=member)
        assert (target, returned_peer) == ("remote", peer)

    @pytest.mark.asyncio
    async def test_a_profile_grant_covers_its_base_model(self):
        from types import SimpleNamespace

        from ocabra.api._deps_auth import UserContext

        mm = _FakeModelManager({"llamacpp_mks/chat:8b": _FakeModelState(
            "llamacpp_mks/chat:8b", status="loaded")})
        registry = MagicMock()
        registry.get = AsyncMock(return_value=SimpleNamespace(
            profile_id="chat:8b", base_model_id="llamacpp_mks/chat:8b",
            profile_kind="execution", enabled=True))
        registry.resolve_alias = AsyncMock(return_value=None)
        member = UserContext(user_id="m", username="m", role="user",
                             accessible_model_ids={"chat:8b"})

        target, _ = await resolve_federated("chat:8b", mm, None, registry, user=member)

        assert target == "local"

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

    @pytest.mark.asyncio
    async def test_local_unloaded_prefers_warm_peer(self):
        """Local model is registered but UNLOADED; a peer has it warm.

        Federation must proxy to the peer instead of paying a cold-start
        locally — that's the whole reason federation exists.
        """
        mm = _FakeModelManager({"whisper-large": _FakeModelState("whisper-large", status="unloaded")})
        peer = _make_peer(
            models=[{"model_id": "whisper-large", "status": "LOADED", "profiles": []}],
        )
        fm = _make_fm_with_peers(peer)
        target, returned_peer = await resolve_federated("whisper-large", mm, fm)
        assert target == "remote"
        assert returned_peer is peer

    @pytest.mark.asyncio
    async def test_local_loading_prefers_warm_peer(self):
        """A local LOADING state should not block proxying to a warm peer."""
        mm = _FakeModelManager({"whisper-large": _FakeModelState("whisper-large", status="loading")})
        peer = _make_peer(
            models=[{"model_id": "whisper-large", "status": "LOADED", "profiles": []}],
        )
        fm = _make_fm_with_peers(peer)
        target, returned_peer = await resolve_federated("whisper-large", mm, fm)
        assert target == "remote"
        assert returned_peer is peer

    @pytest.mark.asyncio
    async def test_local_unloaded_no_peer_falls_back_to_local(self):
        mm = _FakeModelManager({"whisper-large": _FakeModelState("whisper-large", status="unloaded")})
        fm = _make_fm_with_peers()
        target, peer = await resolve_federated("whisper-large", mm, fm)
        assert target == "local"
        assert peer is None


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

    @pytest.mark.asyncio
    async def test_strips_local_session_cookie(self):
        """Local session cookies must not be forwarded to the peer.

        The peer's auth dep tries cookies before bearer tokens; a JWT signed
        by our node's secret would fail validation on the peer and short-circuit
        to 401 ("Session expired or invalid") without ever checking the bearer.
        """
        peer = _make_peer(url="https://remote:8000", api_key="sk-key")
        fm = _make_fm_with_peers(peer)

        mock_response = httpx.Response(
            200, json={},
            request=httpx.Request("POST", "https://remote:8000/v1/chat/completions"),
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        fm._http_client = mock_client

        await fm.proxy_request(
            peer, "/v1/chat/completions", {},
            headers={
                "cookie": "ocabra_session=local-jwt; theme=dark",
                "host": "local.example.com",
                "content-length": "9999",
                "x-keep": "yes",
            },
        )

        forwarded = call_kwargs = mock_client.post.call_args[1]["headers"]
        assert forwarded["Authorization"] == "Bearer sk-key"
        assert "cookie" not in {k.lower() for k in forwarded}
        assert "host" not in {k.lower() for k in forwarded}
        assert "content-length" not in {k.lower() for k in forwarded}
        assert forwarded.get("x-keep") == "yes"
        _ = call_kwargs  # avoid unused alias warning


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

    def test_get_remote_models_status_is_case_insensitive(self):
        """Real peers send ``ModelStatus.LOADED.value`` which is the lowercase
        string ``"loaded"`` — make sure the consumer accepts that too. This
        regression silently broke federated proxying for every endpoint."""
        peer = _make_peer(
            models=[{"model_id": "whisper/large-v3", "status": "loaded"}],
        )
        fm = _make_fm_with_peers(peer)
        remote = fm.get_remote_models()
        assert "whisper/large-v3" in remote


# ---------------------------------------------------------------------------
# Canonical-id retry (defense-in-depth against stale peer profile cache)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_try_proxy_json_retries_with_canonical_id_on_404(monkeypatch) -> None:
    """A peer may advertise a profile_id in its heartbeat but then 404 that
    same id when hit directly (observed when the peer inserts a profile row
    without reloading its in-memory profile_registry). The federation helper
    must retry once with the canonical ``base_model_id`` before falling back
    to local, otherwise federation silently stops working for every freshly
    added profile until the peer restarts.
    """
    from types import SimpleNamespace

    from ocabra.api.openai._federation import try_proxy_json

    # try_proxy_json sale en la primera linea si la federacion esta apagada, y
    # esta apagada por defecto: sin esto el helper devolvia None sin llegar a
    # proxyar nada y el test no comprobaba el reintento canonico.
    monkeypatch.setattr(settings, "federation_enabled", True)

    base_id = "ollama/qwen3-embedding:0.6b"
    public_id = "qwen3-embedding:0.6b"

    peer = SimpleNamespace(
        name="mapache",
        peer_id="peer-1",
        url="http://peer",
        models=[{"model_id": base_id, "profiles": [public_id]}],
    )

    # proxy_request: first call (with public id) → 404, second (canonical) → 200.
    calls: list[dict[str, Any]] = []

    def _mk_resp(status: int, body: bytes) -> MagicMock:
        resp = MagicMock()
        resp.status_code = status
        resp.content = body
        resp.text = body.decode()
        resp.headers = {"content-type": "application/json"}
        return resp

    async def _proxy(peer, path, body, headers):
        calls.append({"model": body.get("model")})
        if len(calls) == 1:
            return _mk_resp(404, b'{"detail":{"error":{"message":"not found"}}}')
        return _mk_resp(200, b'{"data":[{"embedding":[0.1]}]}')

    fm = MagicMock()
    fm.get_remote_models.return_value = {base_id: [peer]}
    fm.find_best_peer.return_value = peer
    fm.select_target.return_value = "local"
    fm.proxy_request = _proxy

    # Model manager: local has the model CONFIGURED (not loaded), so federation
    # path kicks in with peer warm.
    mm = _FakeModelManager({base_id: _FakeModelState(model_id=base_id, status="configured")})

    class _FakeProfile:
        profile_id = public_id
        base_model_id = base_id
        runtime_presets: list = []

    class _FakeRegistry:
        async def get(self, pid):
            if pid in (public_id, base_id):
                return _FakeProfile()
            return None
        async def resolve_alias(self, pid):
            return None

    request = MagicMock()
    request.state = SimpleNamespace()
    request.url.path = "/v1/embeddings"
    request.headers = {}

    logger = MagicMock()

    # federation.py importa ModelStatus dentro de la funcion, no a nivel de
    # modulo, asi que hay que parchearlo en su origen: el modulo federation no
    # tiene el atributo y el patch antiguo fallaba con AttributeError.
    with patch("ocabra.core.model_manager.ModelStatus", _FakeModelStatus):
        resp = await try_proxy_json(
            request,
            model_id=public_id,
            body={"model": public_id, "input": "hi"},
            federation_manager=fm,
            model_manager=mm,
            profile_registry=_FakeRegistry(),
            logger=logger,
        )

    assert resp is not None, "helper should have returned the peer's canonical-retry response"
    assert resp.status_code == 200
    assert len(calls) == 2, f"expected two proxy_request calls (profile + canonical), got {calls}"
    assert calls[0]["model"] == public_id
    assert calls[1]["model"] == base_id


@pytest.mark.asyncio
async def test_try_proxy_json_does_not_retry_when_already_canonical(monkeypatch) -> None:
    """When the first proxy already used the canonical id (peer didn't
    advertise the public profile), a 404 is a real "model missing" — don't
    double-send, just fall back to local."""
    from types import SimpleNamespace

    from ocabra.api.openai._federation import try_proxy_json

    # try_proxy_json sale en la primera linea si la federacion esta apagada, y
    # esta apagada por defecto: sin esto el helper devolvia None sin llegar a
    # proxyar nada y el test no comprobaba el reintento canonico.
    monkeypatch.setattr(settings, "federation_enabled", True)

    base_id = "ollama/qwen3-embedding:0.6b"

    peer = SimpleNamespace(
        name="mapache",
        peer_id="peer-1",
        url="http://peer",
        models=[{"model_id": base_id, "profiles": []}],  # no public profile
    )

    calls: list[dict[str, Any]] = []

    def _mk_resp(status: int) -> MagicMock:
        resp = MagicMock()
        resp.status_code = status
        resp.content = b'{"err":"nope"}'
        resp.text = '{"err":"nope"}'
        resp.headers = {}
        return resp

    async def _proxy(peer, path, body, headers):
        calls.append({"model": body.get("model")})
        return _mk_resp(404)

    fm = MagicMock()
    fm.get_remote_models.return_value = {base_id: [peer]}
    fm.find_best_peer.return_value = peer
    fm.select_target.return_value = "local"
    fm.proxy_request = _proxy

    mm = _FakeModelManager({base_id: _FakeModelState(model_id=base_id, status="configured")})

    class _FakeProfile:
        profile_id = base_id
        base_model_id = base_id
        runtime_presets: list = []

    class _FakeRegistry:
        async def get(self, pid):
            return _FakeProfile() if pid == base_id else None
        async def resolve_alias(self, pid):
            return None

    request = MagicMock()
    request.state = SimpleNamespace()
    request.url.path = "/v1/embeddings"
    request.headers = {}

    # federation.py importa ModelStatus dentro de la funcion, no a nivel de
    # modulo, asi que hay que parchearlo en su origen: el modulo federation no
    # tiene el atributo y el patch antiguo fallaba con AttributeError.
    with patch("ocabra.core.model_manager.ModelStatus", _FakeModelStatus):
        resp = await try_proxy_json(
            request,
            model_id=base_id,
            body={"model": base_id, "input": "hi"},
            federation_manager=fm,
            model_manager=mm,
            profile_registry=_FakeRegistry(),
            logger=MagicMock(),
        )

    assert resp is None, "should fall back to local, not retry a canonical id with itself"
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# Routers across the federation: capability and the name sent to the peer
# ---------------------------------------------------------------------------


def _image_router_setup(monkeypatch):
    """``recomendado-imagen`` = [generate-only, edit-capable]; only the first
    is warm, and only on a cloud node that has no such router."""
    from types import SimpleNamespace

    import ocabra.core.federation as fed

    gen, edit = "diffusers/qwen-image-2.1", "diffusers/qwen-image-edit-plus"
    profiles = {
        "recomendado-imagen": SimpleNamespace(
            profile_id="recomendado-imagen",
            profile_kind="router",
            routing_targets=["qwen-image-2.1", "qwen-image-edit-plus"],
            base_model_id=None,
            runtime_presets=[],
        ),
        "qwen-image-2.1": SimpleNamespace(
            profile_id="qwen-image-2.1", profile_kind="execution", base_model_id=gen, runtime_presets=[]
        ),
        "qwen-image-edit-plus": SimpleNamespace(
            profile_id="qwen-image-edit-plus",
            profile_kind="execution",
            base_model_id=edit,
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

    async def _capable(_mm, profile, capability):
        return capability != "image_editing" or profile.base_model_id == edit

    monkeypatch.setattr(fed, "_profile_capable", _capable)
    monkeypatch.setattr(settings, "federation_enabled", True)
    pod = _make_peer(
        name="pod", models=[{"model_id": gen, "status": "loaded", "profiles": ["qwen-image-2.1"]}]
    )
    fm = _make_fm_with_peers(pod)
    mm = _FakeModelManager(
        {
            gen: _FakeModelState(model_id=gen, status="configured", backend_type="diffusers"),
            edit: _FakeModelState(model_id=edit, status="configured", backend_type="diffusers"),
        }
    )
    return fm, mm, _Registry(), pod


@pytest.mark.asyncio
async def test_router_edit_skips_the_warm_generate_only_target(monkeypatch):
    """Real case 2026-11-16: an edit to recomendado-imagen picked the pod's warm
    qwen-image-2.1 (can't edit), the pod 404'd the router name, and the local
    fallback evicted a chat model to load the edit model."""
    fm, mm, reg, pod = _image_router_setup(monkeypatch)

    resolved: dict = {}
    target, peer = await resolve_federated(
        "recomendado-imagen", mm, fm, reg, required_capability="image_editing", resolved=resolved
    )
    assert (target, peer) == ("local", None)
    assert resolved["model_id"] == "diffusers/qwen-image-edit-plus"

    resolved = {}
    target, peer = await resolve_federated(
        "recomendado-imagen", mm, fm, reg, required_capability="image_generation", resolved=resolved
    )
    assert target == "remote" and peer is pod
    assert resolved["profile"].profile_id == "qwen-image-2.1"


@pytest.mark.asyncio
async def test_router_is_sent_to_the_peer_as_the_chosen_target(monkeypatch):
    from types import SimpleNamespace

    from ocabra.api.openai._federation import try_proxy_multipart

    fm, mm, reg, pod = _image_router_setup(monkeypatch)
    sent = {}

    async def _multipart(peer, path, files, data, headers):
        sent.update(data)
        return httpx.Response(200, json={"data": []})

    fm.proxy_multipart = _multipart
    request = MagicMock()
    request.state = SimpleNamespace()
    request.url.path = "/v1/images/generations"
    request.headers = {}
    resp = await try_proxy_multipart(
        request,
        model_id="recomendado-imagen",
        files={},
        data={"model": "recomendado-imagen", "prompt": "x"},
        federation_manager=fm,
        model_manager=mm,
        profile_registry=reg,
        logger=MagicMock(),
        required_capability="image_generation",
    )
    assert resp is not None and resp.status_code == 200
    assert sent["model"] == "qwen-image-2.1"  # the pod has no recomendado-imagen
