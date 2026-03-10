"""
Integration tests for LiteLLM sync.

Mocks the LiteLLM HTTP API and verifies payload structure.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ocabra.backends.base import BackendCapabilities
from ocabra.core.model_manager import LoadPolicy, ModelState, ModelStatus
from ocabra.integrations.litellm_sync import LiteLLMSync


def _loaded_state(model_id: str, display_name: str) -> ModelState:
    return ModelState(
        model_id=model_id,
        display_name=display_name,
        backend_type="vllm",
        status=ModelStatus.LOADED,
        load_policy=LoadPolicy.WARM,
        capabilities=BackendCapabilities(chat=True),
        current_gpu=[1],
    )


class TestLiteLLMSync:
    @pytest.mark.asyncio
    async def test_sync_sends_loaded_models(self):
        """sync_all() should POST the correct model_list to LiteLLM."""
        mm = MagicMock()
        mm.list_states = AsyncMock(return_value=[
            _loaded_state("meta-llama/llama-3-8b", "Llama 3 8B"),
            _loaded_state("mistralai/mistral-7b", "Mistral 7B"),
        ])

        captured = {}

        async def _fake_post(url, json=None, headers=None, **kwargs):
            captured["url"] = url
            captured["json"] = json
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            return resp

        with (
            patch("ocabra.integrations.litellm_sync.settings") as mock_settings,
            patch("httpx.AsyncClient") as MockClient,
        ):
            mock_settings.litellm_base_url = "http://litellm:4000"
            mock_settings.litellm_admin_key = "test-key"
            mock_settings.litellm_auto_sync = True

            client_instance = AsyncMock()
            client_instance.__aenter__ = AsyncMock(return_value=client_instance)
            client_instance.__aexit__ = AsyncMock(return_value=False)
            client_instance.post = AsyncMock(side_effect=_fake_post)
            MockClient.return_value = client_instance

            syncer = LiteLLMSync(mm)
            result = await syncer.sync_all()

        assert result.synced == 2
        assert result.errors == []
        assert "model_list" in captured["json"]
        model_names = [m["model_name"] for m in captured["json"]["model_list"]]
        assert "Llama 3 8B" in model_names
        assert "Mistral 7B" in model_names

        # Verify routing via oCabra
        for entry in captured["json"]["model_list"]:
            assert entry["litellm_params"]["api_base"] == "http://ocabra:8000/v1"

    @pytest.mark.asyncio
    async def test_sync_skipped_when_not_configured(self):
        """sync_all() should skip when litellm_base_url is empty."""
        mm = MagicMock()
        mm.list_states = AsyncMock(return_value=[])

        with patch("ocabra.integrations.litellm_sync.settings") as mock_settings:
            mock_settings.litellm_base_url = ""
            mock_settings.litellm_admin_key = ""
            mock_settings.litellm_auto_sync = False

            syncer = LiteLLMSync(mm)
            result = await syncer.sync_all()

        assert result.synced == 0

    @pytest.mark.asyncio
    async def test_sync_handles_http_error(self):
        """sync_all() should return error in result on HTTP failure."""
        import httpx

        mm = MagicMock()
        mm.list_states = AsyncMock(return_value=[
            _loaded_state("test-model", "Test"),
        ])

        with (
            patch("ocabra.integrations.litellm_sync.settings") as mock_settings,
            patch("httpx.AsyncClient") as MockClient,
        ):
            mock_settings.litellm_base_url = "http://litellm:4000"
            mock_settings.litellm_admin_key = "test-key"
            mock_settings.litellm_auto_sync = True

            client_instance = AsyncMock()
            client_instance.__aenter__ = AsyncMock(return_value=client_instance)
            client_instance.__aexit__ = AsyncMock(return_value=False)
            client_instance.post = AsyncMock(side_effect=Exception("Connection refused"))
            MockClient.return_value = client_instance

            syncer = LiteLLMSync(mm)
            result = await syncer.sync_all()

        assert result.synced == 0
        assert len(result.errors) == 1

    @pytest.mark.asyncio
    async def test_on_model_event_triggers_sync_when_auto_sync(self):
        mm = MagicMock()
        mm.list_states = AsyncMock(return_value=[])

        with patch("ocabra.integrations.litellm_sync.settings") as mock_settings:
            mock_settings.litellm_base_url = ""
            mock_settings.litellm_admin_key = ""
            mock_settings.litellm_auto_sync = True

            syncer = LiteLLMSync(mm)
            syncer.sync_all = AsyncMock(return_value=MagicMock(synced=0, errors=[]))

            await syncer.on_model_event("loaded", "some-model")
            syncer.sync_all.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_model_event_no_sync_when_disabled(self):
        mm = MagicMock()

        with patch("ocabra.integrations.litellm_sync.settings") as mock_settings:
            mock_settings.litellm_auto_sync = False

            syncer = LiteLLMSync(mm)
            syncer.sync_all = AsyncMock()

            await syncer.on_model_event("loaded", "some-model")
            syncer.sync_all.assert_not_called()
