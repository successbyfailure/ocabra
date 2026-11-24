"""/v1/embeddings envía la petición al worker que resolve_profile ha cargado.

Regresión del 2026-11-20: un modelo de embeddings descargado de Hugging Face
recibe un preset de runtime ``normal`` y su worker se registra como
``<base>::<hash del preset>``. El endpoint recalculaba la clave con
``compute_worker_key(base, load_overrides)``, que no ve el preset, y la
petición fallaba siempre con 503 ``worker_unloaded`` aunque el modelo estuviera
cargado (harrier-oss-v1-0.6b, recién descargado).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ocabra.api._deps_auth import UserContext, get_current_user

MODULE = "ocabra.api.openai.embeddings"
# REF-01: la resolución/capacidad/despacho los ejecuta ahora el motor
# ``core/inference/pipeline``; la federación sigue inyectándose desde el endpoint.
PIPELINE = "ocabra.core.inference.pipeline"
BASE = "llamacpp_mks/harrier"
WORKER_KEY = f"{BASE}::ff9550da9dfb"


def test_usa_la_clave_del_worker_con_el_preset_de_runtime():
    from ocabra.api.openai import router

    app = FastAPI()
    app.include_router(router, prefix="/v1")
    app.dependency_overrides[get_current_user] = lambda: UserContext(
        user_id="u1", username="escriba", role="user"
    )
    pool = MagicMock()
    pool.forward_request = AsyncMock(
        return_value={"data": [{"embedding": [0.6, 0.8], "index": 0}], "model": "harrier"}
    )
    app.state.worker_pool = pool
    app.state.model_manager = MagicMock()
    app.state.profile_registry = MagicMock()
    app.state.router_resolver = MagicMock(is_router=MagicMock(return_value=False))

    profile = SimpleNamespace(
        profile_id="harrier",
        base_model_id=BASE,
        load_overrides=None,
        request_defaults=None,
        assets=None,
    )
    state = SimpleNamespace(
        model_id=WORKER_KEY,
        backend_type="llamacpp_mks",
        backend_model_id="harrier",
        capabilities=SimpleNamespace(embeddings=True),
    )
    with (
        patch(f"{MODULE}.try_proxy_json", AsyncMock(return_value=None)),
        patch(f"{PIPELINE}.resolve_profile", AsyncMock(return_value=(profile, state))),
        patch(f"{PIPELINE}.check_capability"),
        patch(f"{PIPELINE}.to_backend_body", lambda _state, body: body),
        patch(f"{PIPELINE}.ocabra_response_headers", AsyncMock(return_value={})),
    ):
        resp = TestClient(app, raise_server_exceptions=True).post(
            "/v1/embeddings", json={"model": "harrier", "input": ["hola"]}
        )

    assert resp.status_code == 200, resp.text
    assert pool.forward_request.await_args.args[0] == WORKER_KEY
