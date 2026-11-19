"""Los endpoints de estimación respetan el ACL de modelos por grupo.

``POST /ocabra/estimate`` y ``POST /ocabra/models/{id}/memory-estimate``
pedían solo rol ``user``: cualquiera podía sondear un modelo que no tenía
concedido (y confirmar que existía). Ahora responden 404 igual que
``GET /models`` / ``load``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from ocabra.api._deps_auth import UserContext, get_current_user
from ocabra.api.internal import estimate as estimate_api
from ocabra.api.internal import models as models_api
from ocabra.api.internal import trtllm as trtllm_api


def _user(*granted: str) -> UserContext:
    return UserContext(user_id="u", username="u", role="user", accessible_model_ids=set(granted))


def _profile(profile_id: str, base: str) -> SimpleNamespace:
    return SimpleNamespace(
        profile_id=profile_id,
        base_model_id=base,
        profile_kind="execution",
        routing_targets=[],
        enabled=True,
    )


_STATE = SimpleNamespace(
    model_id="llama_cpp/secret",
    backend_type="llama_cpp",
    status=SimpleNamespace(value="configured"),
    extra_config={},
)


def _app(user: UserContext) -> FastAPI:
    app = FastAPI()
    app.include_router(estimate_api.router, prefix="/ocabra")
    app.include_router(models_api.router, prefix="/ocabra")
    app.include_router(trtllm_api.router, prefix="/ocabra")
    profiles = {"secret:8b": _profile("secret:8b", "llama_cpp/secret")}
    registry = MagicMock()
    registry.get = AsyncMock(side_effect=lambda pid: profiles.get(pid))
    registry.list_by_model = AsyncMock(
        side_effect=lambda mid: [p for p in profiles.values() if p.base_model_id == mid]
    )
    app.state.profile_registry = registry
    mm = MagicMock()
    mm.get_state = AsyncMock(side_effect=lambda mid: _STATE if mid == "llama_cpp/secret" else None)
    app.state.model_manager = mm
    estimator = MagicMock()
    estimator.estimate = AsyncMock(
        return_value=SimpleNamespace(
            as_dict=lambda: {
                "expected_seconds": 1.0,
                "p50_seconds": 1.0,
                "p95_seconds": 2.0,
                "confidence": 0.5,
                "source": "family_default",
                "sample_count": 0,
                "cold_start_seconds": 0.0,
            }
        )
    )
    app.state.duration_estimator = estimator
    app.dependency_overrides[get_current_user] = lambda: user
    return app


async def _post(app: FastAPI, url: str, payload: dict):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
        return await client.post(url, json=payload)


@pytest.mark.parametrize("model", ["secret:8b", "llama_cpp/secret"])
async def test_estimate_hides_ungranted_model(model: str) -> None:
    resp = await _post(_app(_user("other:4b")), "/ocabra/estimate", {"model": model})
    assert resp.status_code == 404


@pytest.mark.parametrize("model", ["secret:8b", "llama_cpp/secret"])
async def test_estimate_allows_granted_profile(model: str) -> None:
    # The profile grant also covers asking by the canonical id behind it.
    resp = await _post(_app(_user("secret:8b")), "/ocabra/estimate", {"model": model})
    assert resp.status_code == 200, resp.text
    assert resp.json()["source"] == "family_default"


async def test_estimate_admin_sees_everything() -> None:
    admin = UserContext(user_id=None, username="a", role="system_admin")
    resp = await _post(_app(admin), "/ocabra/estimate", {"model": "secret:8b"})
    assert resp.status_code == 200


async def test_memory_estimate_hides_ungranted_model(monkeypatch) -> None:
    built = AsyncMock(return_value={"ok": True})
    monkeypatch.setattr(models_api, "_build_model_memory_estimate", built)
    url = "/ocabra/models/llama_cpp/secret/memory-estimate"

    denied = await _post(_app(_user("other:4b")), url, {})
    allowed = await _post(_app(_user("secret:8b")), url, {})

    assert denied.status_code == 404
    assert allowed.status_code == 200
    built.assert_awaited_once()


async def test_trtllm_estimate_needs_model_manager() -> None:
    async with AsyncClient(
        transport=ASGITransport(app=_app(_user("secret:8b"))), base_url="http://t"
    ) as client:
        resp = await client.get("/ocabra/trtllm/estimate", params={"model_id": "vllm/x/y"})
    assert resp.status_code == 403
