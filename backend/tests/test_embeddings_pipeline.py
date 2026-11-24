"""/v1/embeddings migrado al motor de flujo único (REF-01, fase A).

Verifica el comportamiento sensible: la concurrencia de embeddings NO se
degrada (no se engancha la reja generativa de reparto justo), la cancelación por
desconexión (hueco rellenado), la federación corto-circuita, la ACL sigue dando
404 y la capacidad sigue rechazando un modelo sin embeddings.
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
from ocabra.core.inference.errors import _openai_error

MODULE = "ocabra.api.openai.embeddings"
PIPELINE = "ocabra.core.inference.pipeline"
BASE = "be"
WORKER_KEY = "be::preset"


def _error_code(resp) -> str | None:
    """Código de error, tolerante a la forma: la mini-app de test no monta los
    manejadores de familia (REF-10), así que el cuerpo va bajo ``detail``."""
    body = resp.json()
    err = body.get("error")
    if err is None and isinstance(body.get("detail"), dict):
        err = body["detail"].get("error")
    return err.get("code") if isinstance(err, dict) else None


def _profile():
    return SimpleNamespace(
        profile_id="e", base_model_id=BASE, load_overrides=None, request_defaults=None, assets=None
    )


def _state(embeddings_cap: bool = True):
    return SimpleNamespace(
        model_id=WORKER_KEY,
        backend_type="llamacpp_mks",
        backend_model_id="e",
        base_model_id=BASE,
        capabilities=SimpleNamespace(embeddings=embeddings_cap),
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


def _enter_happy_path(stack, pool, *, state=None, profile=None):
    """Patcha federación=none + resolución feliz + cuerpo/headers inertes."""
    state = state if state is not None else _state()
    profile = profile if profile is not None else _profile()
    stack.enter_context(patch(f"{MODULE}.try_proxy_json", AsyncMock(return_value=None)))
    stack.enter_context(
        patch(f"{PIPELINE}.resolve_profile", AsyncMock(return_value=(profile, state)))
    )
    stack.enter_context(patch(f"{PIPELINE}.to_backend_body", lambda _s, body: body))
    stack.enter_context(patch(f"{PIPELINE}.ocabra_response_headers", AsyncMock(return_value={})))


@pytest.mark.asyncio
async def test_embeddings_burst_keeps_concurrency_no_fair_gate(_client_factory):
    """Una ráfaga de 8 peticiones concurrentes: todas 200 y NINGUNA engancha la
    reja de reparto justo (``forward_request`` sin ``user_key``). Regresión del
    incidente ``project_embeddings_inflight_cap``: el fallback generativo de 1
    slot aplicado a embeddings causaba 429 en ráfaga."""
    n = 8

    async def _forward(*args, **kwargs):
        await asyncio.sleep(0.02)  # fuerza solapamiento real entre peticiones
        return {"data": [{"embedding": [0.1], "index": 0}], "model": "e"}

    pool = MagicMock()
    pool.forward_request = AsyncMock(side_effect=_forward)
    app = _build_app(pool)
    client = _client_factory(app)

    with contextlib.ExitStack() as stack:
        _enter_happy_path(stack, pool)
        results = await asyncio.gather(
            *[
                client.post("/v1/embeddings", json={"model": "e", "input": ["hola"]})
                for _ in range(n)
            ]
        )

    assert [r.status_code for r in results] == [200] * n
    assert pool.forward_request.await_count == n
    for call in pool.forward_request.await_args_list:
        assert "user_key" not in call.kwargs
        assert "slots" not in call.kwargs
        assert "is_batch" not in call.kwargs


@pytest.mark.asyncio
async def test_embeddings_federation_short_circuits(_client_factory):
    peer = Response(content=b'{"ok":1}', status_code=200, media_type="application/json")
    pool = MagicMock()
    pool.forward_request = AsyncMock()
    app = _build_app(pool)
    client = _client_factory(app)

    with patch(f"{MODULE}.try_proxy_json", AsyncMock(return_value=peer)):
        resp = await client.post("/v1/embeddings", json={"model": "e", "input": ["x"]})

    assert resp.status_code == 200
    pool.forward_request.assert_not_called()


@pytest.mark.asyncio
async def test_embeddings_acl_returns_404(_client_factory):
    pool = MagicMock()
    pool.forward_request = AsyncMock()
    app = _build_app(pool)
    client = _client_factory(app)

    acl_404 = _openai_error(
        "The model 'e' does not exist.",
        "invalid_request_error",
        param="model",
        code="model_not_found",
        status_code=404,
    )
    with (
        patch(f"{MODULE}.try_proxy_json", AsyncMock(return_value=None)),
        patch(f"{PIPELINE}.resolve_profile", AsyncMock(side_effect=acl_404)),
    ):
        resp = await client.post("/v1/embeddings", json={"model": "e", "input": ["x"]})

    assert resp.status_code == 404
    pool.forward_request.assert_not_called()


@pytest.mark.asyncio
async def test_embeddings_capability_rejected(_client_factory):
    """check_capability real: un modelo sin embeddings → 400 model_not_capable."""
    pool = MagicMock()
    pool.forward_request = AsyncMock()
    app = _build_app(pool)
    client = _client_factory(app)

    with (
        patch(f"{MODULE}.try_proxy_json", AsyncMock(return_value=None)),
        patch(
            f"{PIPELINE}.resolve_profile",
            AsyncMock(return_value=(_profile(), _state(embeddings_cap=False))),
        ),
    ):
        resp = await client.post("/v1/embeddings", json={"model": "e", "input": ["x"]})

    assert resp.status_code == 400
    assert _error_code(resp) == "model_not_capable"
    pool.forward_request.assert_not_called()


@pytest.mark.asyncio
async def test_embeddings_disconnect_returns_499(_client_factory):
    """Hueco rellenado: un cliente que se desconecta cancela el embedding y el
    endpoint responde 499 (client_disconnected)."""
    pool = MagicMock()
    pool.forward_request = AsyncMock(return_value={"data": []})
    app = _build_app(pool)
    client = _client_factory(app)

    with contextlib.ExitStack() as stack:
        _enter_happy_path(stack, pool)
        stack.enter_context(
            patch(
                f"{PIPELINE}.await_unless_client_gone",
                AsyncMock(side_effect=ClientGoneError("gone")),
            )
        )
        resp = await client.post("/v1/embeddings", json={"model": "e", "input": ["x"]})

    assert resp.status_code == 499
    assert _error_code(resp) == "client_disconnected"


@pytest.mark.asyncio
async def test_embeddings_invalid_dimensions_rejected_before_dispatch(_client_factory):
    pool = MagicMock()
    pool.forward_request = AsyncMock(return_value={"data": []})
    app = _build_app(pool)
    client = _client_factory(app)

    with contextlib.ExitStack() as stack:
        _enter_happy_path(stack, pool)
        resp = await client.post(
            "/v1/embeddings", json={"model": "e", "input": ["x"], "dimensions": 0}
        )

    assert resp.status_code == 400
    assert _error_code(resp) == "invalid_dimensions"
    pool.forward_request.assert_not_called()
