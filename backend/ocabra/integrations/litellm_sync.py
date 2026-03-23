"""
LiteLLM Proxy auto-sync.

Keeps the LiteLLM proxy config in sync with oCabra's loaded models,
so that LiteLLM always knows which models are available to route to.
"""
from __future__ import annotations

import asyncio
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
        self._sync_task: asyncio.Task | None = None
        self._sync_pending = False

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

    async def handle_model_event(self, payload: dict) -> None:
        await self.on_model_event(
            str(payload.get("event", "")),
            str(payload.get("model_id", "")),
            new_status=str(payload.get("new_status", payload.get("status", ""))),
        )

    async def on_model_event(
        self,
        event: str,
        model_id: str,
        new_status: str | None = None,
    ) -> None:
        """
        React to model lifecycle changes.

        Auto-sync is scheduled in the background so model load/unload/register
        flows do not block on LiteLLM network calls.
        """
        if not settings.litellm_auto_sync:
            return
        if not self._should_sync(event, new_status):
            return

        logger.info(
            "litellm_sync_scheduled",
            lifecycle_event=event,
            model_id=model_id,
            new_status=new_status,
        )
        task = self._schedule_sync()
        if task is not None:
            await asyncio.sleep(0)

    def _should_sync(self, event: str, new_status: str | None) -> bool:
        event_name = str(event or "").lower()
        status_name = str(new_status or "").lower()
        return event_name in {"register", "load", "loaded", "unload", "unloaded"} or status_name in {
            "loaded",
            "unloaded",
        }

    def _schedule_sync(self) -> asyncio.Task | None:
        self._sync_pending = True
        if self._sync_task is not None and not self._sync_task.done():
            return self._sync_task
        self._sync_task = asyncio.create_task(self._drain_sync_queue())
        return self._sync_task

    async def _drain_sync_queue(self) -> None:
        while self._sync_pending:
            self._sync_pending = False
            try:
                await self.sync_all()
            except Exception as exc:
                logger.error("litellm_sync_background_failed", error=str(exc))
