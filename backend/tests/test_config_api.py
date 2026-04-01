import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from ocabra.api.internal import config as config_api
from ocabra.config import settings


@pytest.mark.asyncio
async def test_get_config_includes_settings_fields() -> None:
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))

    payload = await config_api.get_config(request)

    assert "defaultGpuIndex" in payload
    assert "modelsDir" in payload
    assert "downloadDir" in payload
    assert "maxTemperatureC" in payload
    assert "globalSchedules" in payload


@pytest.mark.asyncio
async def test_patch_config_applies_runtime_values() -> None:
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
    previous_timeout = settings.idle_timeout_seconds

    try:
        patch = config_api.ServerConfigPatch.model_validate(
            {
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
            }
        )

        payload = await config_api.patch_config(patch, request)

        assert payload["idleTimeoutSeconds"] == 123
        assert payload["downloadDir"] == "/tmp/test-downloads"
        assert payload["maxTemperatureC"] == 91
        assert payload["globalSchedules"][0]["id"] == "night"
    finally:
        settings.idle_timeout_seconds = previous_timeout


@pytest.mark.asyncio
async def test_patch_config_rejects_models_dir_runtime_mutation() -> None:
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
    previous_models_dir = settings.models_dir

    patch = config_api.ServerConfigPatch.model_validate({"modelsDir": "/tmp/test-models"})

    with pytest.raises(HTTPException) as exc_info:
        await config_api.patch_config(patch, request)

    assert exc_info.value.status_code == 400
    assert "cannot be changed at runtime" in str(exc_info.value.detail)
    assert settings.models_dir == previous_models_dir


@pytest.mark.asyncio
async def test_patch_config_rejects_legacy_snake_case_keys() -> None:
    with pytest.raises(ValidationError):
        config_api.ServerConfigPatch.model_validate({"models_dir": "/tmp/legacy-models"})


@pytest.mark.asyncio
async def test_sync_litellm_contract() -> None:
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(model_manager=object())))
    fake_result = SimpleNamespace(synced=4, errors=[])

    with patch("ocabra.integrations.litellm_sync.LiteLLMSync.sync_all", new=AsyncMock(return_value=fake_result)):
        response = await config_api.sync_litellm(request)

    payload = json.loads(response.body)
    assert payload["synced_models"] == 4
    assert payload["errors"] == []
