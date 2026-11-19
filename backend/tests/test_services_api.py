"""``GET /ocabra/services/{id}/generations`` con un servicio desconocido daba
``[]`` (igual que un servicio sin generaciones); ahora 404."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ocabra.api._deps_auth import UserContext, get_current_user
from ocabra.api.internal.services import router


def _app() -> FastAPI:
    app = FastAPI()
    app.include_router(router, prefix="/ocabra")
    app.dependency_overrides[get_current_user] = lambda: UserContext(
        user_id="u", username="m", role="model_manager"
    )
    sm = MagicMock()
    sm.get_state = AsyncMock(
        side_effect=lambda sid: SimpleNamespace(service_id=sid) if sid == "comfyui" else None
    )
    app.state.service_manager = sm
    return app


def test_generations_of_unknown_service_is_404() -> None:
    session_factory = MagicMock()
    with patch("ocabra.database.AsyncSessionLocal", session_factory):
        resp = TestClient(app=_app()).get("/ocabra/services/nope/generations")
    assert resp.status_code == 404
    assert "nope" in resp.json()["detail"]
    session_factory.assert_not_called()


def test_generations_of_known_service_without_rows_is_empty() -> None:
    result = MagicMock()
    result.scalars.return_value.all.return_value = []
    session = MagicMock()
    session.execute = AsyncMock(return_value=result)
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    with patch("ocabra.database.AsyncSessionLocal", MagicMock(return_value=ctx)):
        resp = TestClient(app=_app()).get("/ocabra/services/comfyui/generations")
    assert resp.status_code == 200
    assert resp.json() == []
