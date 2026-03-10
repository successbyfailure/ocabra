"""
LiteLLM Proxy auto-sync.

Keeps the LiteLLM proxy config in sync with oCabra's loaded models,
so that LiteLLM always knows which models are available to route to.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import httpx
import structlog

from ocabra.config import settings

logger = structlog.get_logger(__name__)


@dataclass
class SyncResult:
    synced: int = 0
    errors: list[str] = field(default_factory=list)


class LiteLLMSync:
    """
    Manages synchronisation between oCabra's model registry and LiteLLM proxy.

    Each loaded model is registered as an OpenAI-compatible provider pointing
    back to oCabra's /v1 endpoint.
    """

    def __init__(self, model_manager) -> None:
        self._model_manager = model_manager

    async def sync_all(self) -> SyncResult:
        """
        Sync all loaded models to LiteLLM proxy.

        For each LOADED model, creates a LiteLLM model entry that routes
        through oCabra's OpenAI-compatible API.

        Returns:
            SyncResult with count of synced models and any errors.
        """
        if not settings.litellm_base_url or not settings.litellm_admin_key:
            logger.warning("litellm_sync_skipped", reason="not_configured")
            return SyncResult()

        from ocabra.core.model_manager import ModelStatus

        states = await self._model_manager.list_states()
        loaded = [s for s in states if s.status == ModelStatus.LOADED]

        model_list = [
            {
                "model_name": s.display_name,
                "litellm_params": {
                    "model": f"openai/{s.model_id}",
                    "api_base": "http://ocabra:8000/v1",
                    "api_key": "ocabra-internal",
                },
            }
            for s in loaded
        ]

        result = SyncResult()
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{settings.litellm_base_url}/config/update_config",
                    json={"model_list": model_list},
                    headers={
                        "Authorization": f"Bearer {settings.litellm_admin_key}",
                        "Content-Type": "application/json",
                    },
                )
                resp.raise_for_status()
                result.synced = len(model_list)
                logger.info("litellm_synced", models=result.synced)
        except Exception as e:
            result.errors.append(str(e))
            logger.error("litellm_sync_failed", error=str(e))

        return result

    async def on_model_event(self, event: str, model_id: str) -> None:
        """
        React to model state changes.

        If auto_sync is enabled, triggers a full re-sync whenever a model
        transitions to LOADED or UNLOADED.
        """
        if not settings.litellm_auto_sync:
            return
        if event in ("loaded", "unloaded"):
            await self.sync_all()
