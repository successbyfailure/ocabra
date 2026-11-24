"""Benchmarks medidos en un nodo de la federación (el runner sigue en el hub)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from ocabra.benchmarks.manager import BenchmarkManager, node_base_url
from ocabra.core.federation import (
    FederationHopMiddleware,
    federation_request_info,
    strip_node_prefix,
)

# ── /_node/<nodo>/... ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("path", "target", "node"),
    [
        ("/_node/Mapache/v1/chat/completions", "/v1/chat/completions", "Mapache"),
        ("/_node/Pod%20GPU/v1/embeddings", "/v1/embeddings", "Pod GPU"),
        ("/_node/Mapache/api/chat", "/api/chat", "Mapache"),
        ("/_node/Mapache/version", "/version", "Mapache"),
    ],
)
def test_el_prefijo_de_nodo_se_quita_y_fuerza_el_nodo(path, target, node):
    scope, forced = strip_node_prefix({"type": "http", "path": path, "raw_path": path.encode()})
    assert scope["path"] == target and scope["raw_path"] == target.encode()
    assert forced == node


@pytest.mark.parametrize(
    "path",
    [
        "/_node/Mapache/ocabra/users",  # nada de la API de administración
        "/_node/Mapache",
        "/_node/<script>/v1/models",
        "/v1/chat/completions",
    ],
)
def test_fuera_de_la_api_de_inferencia_no_se_toca(path):
    scope = {"type": "http", "path": path}
    out, forced = strip_node_prefix(scope)
    assert out is scope and forced is None


@pytest.mark.asyncio
async def test_el_middleware_reescribe_la_ruta_y_manda_el_prefijo_sobre_la_cabecera():
    seen = {}

    async def app(scope, receive, send):
        seen["path"] = scope["path"]
        seen["info"] = dict(federation_request_info.get() or {})

    middleware = FederationHopMiddleware(app)
    await middleware(
        {
            "type": "http",
            "path": "/_node/Mapache/v1/models",
            "raw_path": b"/_node/Mapache/v1/models",
            "headers": [(b"x-ocabra-node", b"otro")],
        },
        None,
        None,
    )
    assert seen["path"] == "/v1/models"
    assert seen["info"]["force_node"] == "Mapache"


def test_la_url_base_del_runner_lleva_el_nodo():
    assert node_base_url("http://127.0.0.1:8000", None) == "http://127.0.0.1:8000"
    assert node_base_url("http://127.0.0.1:8000/", "Pod GPU") == (
        "http://127.0.0.1:8000/_node/Pod%20GPU"
    )


# ── Verificaciones del gestor ──────────────────────────────────────────


def test_una_medida_servida_en_otro_nodo_se_detecta():
    env = {"hardware": {"nodes": ["local", "Mapache"]}}
    assert BenchmarkManager._served_elsewhere(env, "mapache") == ["local"]
    assert BenchmarkManager._served_elsewhere({"hardware": {"nodes": ["Mapache"]}}, "Mapache") == []
    # Sin nodo pedido se mantiene el reparto normal de la federación.
    assert BenchmarkManager._served_elsewhere(env, None) == []


def _manager_with_peer(payload):
    peer = SimpleNamespace(name="Mapache", peer_id="p1", online=True, gpus=[])
    response = MagicMock(status_code=200)
    response.aread = AsyncMock(return_value=json.dumps(payload).encode())
    response.aclose = AsyncMock()
    federation = MagicMock()
    federation.get_all_peers.return_value = [peer]
    federation.proxy_raw = AsyncMock(return_value=response)
    manager = BenchmarkManager.__new__(BenchmarkManager)
    manager._app = SimpleNamespace(state=SimpleNamespace(federation_manager=federation))
    return manager


@pytest.mark.asyncio
async def test_el_trafico_propio_del_nodo_cuenta_y_el_que_le_mandamos_no():
    start = datetime(2026, 11, 24, 10, 0, tzinfo=UTC)
    rows = [
        # Del propio nodo, durante la medida: ajeno.
        {"id": "a", "modelId": "m", "startedAt": (start + timedelta(seconds=30)).isoformat(),
         "durationMs": 1000, "inboundFrom": None},
        # Llegó por la federación (desde el hub): no se cuenta aquí.
        {"id": "b", "modelId": "m", "startedAt": (start + timedelta(seconds=40)).isoformat(),
         "durationMs": 1000, "inboundFrom": "MakeSpace"},
        # Del nodo pero antes de empezar.
        {"id": "c", "modelId": "m", "startedAt": (start - timedelta(hours=1)).isoformat(),
         "durationMs": 1000, "inboundFrom": None},
    ]  # fmt: skip
    manager = _manager_with_peer({"requests": rows})
    external, verified = await manager._node_local_requests(
        "Mapache", start, start + timedelta(minutes=5)
    )
    assert verified is True
    assert [item["requestId"] for item in external] == ["Mapache:a"]


@pytest.mark.asyncio
async def test_si_no_se_puede_leer_el_nodo_queda_sin_comprobar():
    manager = _manager_with_peer({"detail": "Forbidden"})
    external, verified = await manager._node_local_requests(
        "Mapache", datetime.now(UTC), datetime.now(UTC)
    )
    assert external == [] and verified is False


# ── API ────────────────────────────────────────────────────────────────


def test_el_nodo_se_valida_contra_los_peers():
    from ocabra.api.internal.benchmarks import _canonical_node

    peers = [SimpleNamespace(name="Mapache", peer_id="p1", enabled=True)]
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(federation_manager=MagicMock(get_all_peers=lambda: peers))
        )
    )
    assert _canonical_node(request, "mapache") == "Mapache"
    assert _canonical_node(request, None) is None
    with pytest.raises(HTTPException) as err:
        _canonical_node(request, "nadie")
    assert err.value.status_code == 400
