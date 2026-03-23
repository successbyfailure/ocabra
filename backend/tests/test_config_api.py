from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ocabra.api.internal.config import router as config_router
from ocabra.config import settings


def _make_app() -> FastAPI:
    app = FastAPI()
    app.state.model_manager = object()
    app.include_router(config_router, prefix="/ocabra")
    return app


def test_get_config_includes_settings_fields() -> None:
    client = TestClient(_make_app())
    resp = client.get("/ocabra/config")
    assert resp.status_code == 200

    payload = resp.json()
    assert "defaultGpuIndex" in payload
    assert "modelsDir" in payload
    assert "downloadDir" in payload
    assert "maxTemperatureC" in payload
    assert "globalSchedules" in payload


def test_patch_config_applies_runtime_values() -> None:
    client = TestClient(_make_app())

    previous_models_dir = settings.models_dir
    previous_timeout = settings.idle_timeout_seconds

    try:
        resp = client.patch(
            "/ocabra/config",
            json={
                "modelsDir": "/tmp/test-models",
                "idleTimeoutSeconds": 123,
                "downloadDir": "/tmp/test-downloads",
                "maxTemperatureC": 91,
                "globalSchedules": [
                    {
                        "id": "night",
                        "days": [1, 2, 3],
                        "start": "01:00",
                        "end": "05:00",
                        "enabled": True,
                    }
                ],
            },
        )
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["modelsDir"] == "/tmp/test-models"
        assert payload["idleTimeoutSeconds"] == 123
        assert payload["downloadDir"] == "/tmp/test-downloads"
        assert payload["maxTemperatureC"] == 91
        assert payload["globalSchedules"][0]["id"] == "night"
    finally:
        settings.models_dir = previous_models_dir
        settings.idle_timeout_seconds = previous_timeout


def test_sync_litellm_contract() -> None:
    client = TestClient(_make_app())

    fake_result = SimpleNamespace(synced=4, errors=[])
    with patch("ocabra.integrations.litellm_sync.LiteLLMSync.sync_all", new=AsyncMock(return_value=fake_result)):
        resp = client.post("/ocabra/config/litellm/sync")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["synced_models"] == 4
    assert payload["errors"] == []
