"""/v1/pooling migrado al motor de flujo único (REF-01, fase A).

``/score``, ``/rerank`` y ``/classify`` siguen en ``_resolve_and_forward`` (fase
B+): aquí solo se cubre ``/pooling`` y, de paso, que ``/rerank`` sigue igual.
"""

from __future__ import annotations

import asyncio
import contextlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.responses import Response
from httpx import ASGITransport, AsyncClient

from ocabra.api._deps_auth import UserContext, get_current_user
from ocabra.core.inference.body import ClientGoneError

MODULE = "ocabra.api.openai.pooling"
PIPELINE = "ocabra.core.inference.pipeline"
WORKER_KEY = "bp::preset"


def _error_code(resp) -> str | None:
    body = resp.json()
    err = body.get("error")
    if err is None and isinstance(body.get("detail"), dict):
        err = body["detail"].get("error")
    return err.get("code") if isinstance(err, dict) else None


def _profile():
    return SimpleNamespace(
        profile_id="p", base_model_id="bp", load_overrides=None, request_defaults=None, assets=None
    )


def _state(pooling_cap: bool = True):
    return SimpleNamespace(
        model_id=WORKER_KEY,
        backend_type="llamacpp_mks",
        backend_model_id="p",
        base_model_id="bp",
        capabilities=SimpleNamespace(pooling=pooling_cap),
        extra_config={},
        live_context=None,
    )


def _build_app(pool):
    from ocabra.api.openai import router

    app = FastAPI()
    app.include_router(router, prefix="/v1")
    app.dependency_overrides[get_current_user] = lambda: UserContext(
        user_id="u1", username="tester", role="system_admin"
    )
    app.state.worker_pool = pool
    app.state.model_manager = MagicMock()
    app.state.profile_registry = MagicMock()
    app.state.router_resolver = MagicMock(is_router=MagicMock(return_value=False))
    return app


@pytest_asyncio.fixture
async def _client_factory():
    clients: list[AsyncClient] = []

    def make(app):
        c = AsyncClient(transport=ASGITransport(app=app), base_url="http://test")
        clients.append(c)
        return c

    yield make
    for c in clients:
        await c.aclose()


def _enter_happy_path(stack, *, state=None):
    state = state if state is not None else _state()
    stack.enter_context(patch(f"{MODULE}.try_proxy_json", AsyncMock(return_value=None)))
    stack.enter_context(
        patch(f"{PIPELINE}.resolve_profile", AsyncMock(return_value=(_profile(), state)))
    )
    stack.enter_context(patch(f"{PIPELINE}.to_backend_body", lambda _s, body: body))
    stack.enter_context(patch(f"{PIPELINE}.ocabra_response_headers", AsyncMock(return_value={})))


@pytest.mark.asyncio
async def test_pooling_burst_keeps_concurrency_no_fair_gate(_client_factory):
    n = 6

    async def _forward(*args, **kwargs):
        await asyncio.sleep(0.02)
        return {"data": [], "model": "p"}

    pool = MagicMock()
    pool.forward_request = AsyncMock(side_effect=_forward)
    client = _client_factory(_build_app(pool))

    with contextlib.ExitStack() as stack:
        _enter_happy_path(stack)
        results = await asyncio.gather(
            *[client.post("/v1/pooling", json={"model": "p", "input": ["x"]}) for _ in range(n)]
        )

    assert [r.status_code for r in results] == [200] * n
    assert pool.forward_request.await_count == n
    for call in pool.forward_request.await_args_list:
        assert "user_key" not in call.kwargs
        # Se despacha a la ruta interna /pooling con la clave del worker (preset).
        assert call.args[0] == WORKER_KEY
        assert call.args[1] == "/pooling"


@pytest.mark.asyncio
async def test_pooling_federation_short_circuits(_client_factory):
    peer = Response(content=b'{"ok":1}', status_code=200, media_type="application/json")
    pool = MagicMock()
    pool.forward_request = AsyncMock()
    client = _client_factory(_build_app(pool))

    with patch(f"{MODULE}.try_proxy_json", AsyncMock(return_value=peer)):
        resp = await client.post("/v1/pooling", json={"model": "p", "input": ["x"]})

    assert resp.status_code == 200
    pool.forward_request.assert_not_called()


@pytest.mark.asyncio
async def test_pooling_disconnect_returns_499(_client_factory):
    pool = MagicMock()
    pool.forward_request = AsyncMock(return_value={"data": []})
    client = _client_factory(_build_app(pool))

    with contextlib.ExitStack() as stack:
        _enter_happy_path(stack)
        stack.enter_context(
            patch(
                f"{PIPELINE}.await_unless_client_gone",
                AsyncMock(side_effect=ClientGoneError("gone")),
            )
        )
        resp = await client.post("/v1/pooling", json={"model": "p", "input": ["x"]})

    assert resp.status_code == 499
    assert _error_code(resp) == "client_disconnected"


@pytest.mark.asyncio
async def test_pooling_capability_rejected(_client_factory):
    pool = MagicMock()
    pool.forward_request = AsyncMock()
    client = _client_factory(_build_app(pool))

    with (
        patch(f"{MODULE}.try_proxy_json", AsyncMock(return_value=None)),
        patch(
            f"{PIPELINE}.resolve_profile",
            AsyncMock(return_value=(_profile(), _state(pooling_cap=False))),
        ),
    ):
        resp = await client.post("/v1/pooling", json={"model": "p", "input": ["x"]})

    assert resp.status_code == 400
    assert _error_code(resp) == "model_not_capable"
    pool.forward_request.assert_not_called()


@pytest.mark.asyncio
async def test_rerank_unchanged_still_forwards_locally(_client_factory):
    """`/rerank` NO se migró (fase B+): sigue sirviendo por _resolve_and_forward."""
    pool = MagicMock()
    pool.forward_request = AsyncMock(return_value={"results": []})
    client = _client_factory(_build_app(pool))

    state = SimpleNamespace(
        model_id=WORKER_KEY,
        backend_type="llamacpp_mks",
        backend_model_id="p",
        base_model_id="bp",
        capabilities=SimpleNamespace(rerank=True),
        extra_config={},
        live_context=None,
    )
    with (
        patch(f"{MODULE}.try_proxy_json", AsyncMock(return_value=None)),
        patch(f"{MODULE}.resolve_profile", AsyncMock(return_value=(_profile(), state))),
        patch(f"{MODULE}.to_backend_body", lambda _s, body: body),
        patch(f"{MODULE}.ocabra_response_headers", AsyncMock(return_value={})),
    ):
        resp = await client.post(
            "/v1/rerank", json={"model": "p", "query": "q", "documents": ["a", "b"]}
        )

    assert resp.status_code == 200
    call = pool.forward_request.await_args
    assert call.args[1] == "/rerank"
    # Sin reja de reparto justo, igual que antes.
    assert "user_key" not in call.kwargs
