from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import structlog

from ocabra.config import settings
from ocabra.redis_client import publish, set_key

logger = structlog.get_logger(__name__)


@dataclass
class ServiceState:
    service_id: str
    service_type: str
    display_name: str
    base_url: str
    ui_base_path: str
    health_path: str = "/health"
    unload_path: str | None = None
    unload_method: str = "POST"
    unload_payload: dict[str, Any] | None = None
    preferred_gpu: int | None = None
    idle_unload_after_seconds: int = 600
    service_alive: bool = False
    runtime_loaded: bool = False
    status: str = "unknown"
    active_model_ref: str | None = None
    last_activity_at: datetime | None = None
    last_health_check_at: datetime | None = None
    last_unload_at: datetime | None = None
    detail: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "service_id": self.service_id,
            "service_type": self.service_type,
            "display_name": self.display_name,
            "base_url": self.base_url,
            "ui_base_path": self.ui_base_path,
            "health_path": self.health_path,
            "unload_path": self.unload_path,
            "preferred_gpu": self.preferred_gpu,
            "idle_unload_after_seconds": self.idle_unload_after_seconds,
            "service_alive": self.service_alive,
            "runtime_loaded": self.runtime_loaded,
            "status": self.status,
            "active_model_ref": self.active_model_ref,
            "last_activity_at": (
                self.last_activity_at.isoformat() if self.last_activity_at else None
            ),
            "last_health_check_at": (
                self.last_health_check_at.isoformat() if self.last_health_check_at else None
            ),
            "last_unload_at": (
                self.last_unload_at.isoformat() if self.last_unload_at else None
            ),
            "detail": self.detail,
            "extra": self.extra,
        }


class ServiceManager:
    def __init__(self) -> None:
        self._states: dict[str, ServiceState] = {
            "hunyuan": ServiceState(
                service_id="hunyuan",
                service_type="hunyuan3d",
                display_name="Hunyuan3D",
                base_url=settings.hunyuan_base_url.rstrip("/"),
                ui_base_path=settings.hunyuan_ui_base_path,
                preferred_gpu=settings.hunyuan_preferred_gpu,
                idle_unload_after_seconds=settings.hunyuan_idle_unload_seconds,
                unload_path="/runtime/unload",
            ),
            "comfyui": ServiceState(
                service_id="comfyui",
                service_type="comfyui",
                display_name="ComfyUI",
                base_url=settings.comfyui_base_url.rstrip("/"),
                ui_base_path=settings.comfyui_ui_base_path,
                health_path="/",
                preferred_gpu=settings.comfyui_preferred_gpu,
                idle_unload_after_seconds=settings.comfyui_idle_unload_seconds,
                unload_path="/free",
                unload_payload={"unload_models": True, "free_memory": True},
            ),
            "a1111": ServiceState(
                service_id="a1111",
                service_type="automatic1111",
                display_name="Automatic1111",
                base_url=settings.a1111_base_url.rstrip("/"),
                ui_base_path=settings.a1111_ui_base_path,
                preferred_gpu=settings.a1111_preferred_gpu,
                idle_unload_after_seconds=settings.a1111_idle_unload_seconds,
                health_path="/sdapi/v1/memory",
                unload_path="/sdapi/v1/unload-checkpoint",
            ),
        }
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        for service_id in self._states:
            await self.refresh(service_id)

    async def list_states(self) -> list[ServiceState]:
        return list(self._states.values())

    async def get_state(self, service_id: str) -> ServiceState | None:
        return self._states.get(service_id)

    async def touch(
        self,
        service_id: str,
        *,
        runtime_loaded: bool | None = None,
        active_model_ref: str | None = None,
        detail: str | None = None,
    ) -> ServiceState:
        state = self._require(service_id)
        state.last_activity_at = datetime.now(timezone.utc)
        if runtime_loaded is not None:
            state.runtime_loaded = runtime_loaded
        if active_model_ref is not None:
            state.active_model_ref = active_model_ref
        if detail is not None:
            state.detail = detail
        state.status = "active" if state.service_alive else "unreachable"
        await self._publish_state(state, "touched")
        return state

    async def mark_runtime(
        self,
        service_id: str,
        *,
        runtime_loaded: bool,
        active_model_ref: str | None = None,
        detail: str | None = None,
    ) -> ServiceState:
        state = self._require(service_id)
        state.runtime_loaded = runtime_loaded
        state.active_model_ref = active_model_ref
        if detail is not None:
            state.detail = detail
        if runtime_loaded:
            state.last_activity_at = datetime.now(timezone.utc)
            state.status = "active" if state.service_alive else "unreachable"
        else:
            state.status = "idle" if state.service_alive else "unreachable"
        await self._publish_state(state, "runtime_changed")
        return state

    async def refresh(self, service_id: str) -> ServiceState:
        state = self._require(service_id)
        url = f"{state.base_url}{state.health_path}"
        now = datetime.now(timezone.utc)
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url)
                response.raise_for_status()
            state.service_alive = True
            state.last_health_check_at = now
            state.status = "active" if state.runtime_loaded else "idle"
            state.detail = None
        except Exception as exc:
            state.service_alive = False
            state.runtime_loaded = False
            state.active_model_ref = None
            state.last_health_check_at = now
            state.status = "unreachable"
            state.detail = str(exc)
        await self._publish_state(state, "health_checked")
        return state

    async def unload(self, service_id: str, reason: str = "manual") -> ServiceState:
        state = self._require(service_id)
        if not state.unload_path:
            raise RuntimeError(f"Service '{service_id}' does not expose an unload endpoint")

        url = f"{state.base_url}{state.unload_path}"
        method = state.unload_method.upper()
        request_kwargs: dict[str, Any] = {}
        if state.unload_payload is not None:
            request_kwargs["json"] = state.unload_payload

        async with self._lock:
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.request(method, url, **request_kwargs)
                    response.raise_for_status()
                state.runtime_loaded = False
                state.active_model_ref = None
                state.last_unload_at = datetime.now(timezone.utc)
                state.status = "idle" if state.service_alive else "unreachable"
                state.detail = f"unloaded:{reason}"
                logger.info("service_runtime_unloaded", service_id=service_id, reason=reason)
            except Exception as exc:
                state.detail = str(exc)
                logger.warning(
                    "service_runtime_unload_failed",
                    service_id=service_id,
                    reason=reason,
                    error=str(exc),
                )
                raise
            finally:
                await self._publish_state(state, "runtime_unloaded")
        return state

    async def check_idle_unloads(self) -> None:
        now = datetime.now(timezone.utc)
        for state in self._states.values():
            if not state.service_alive or not state.runtime_loaded:
                continue
            if state.last_activity_at is None:
                continue
            idle_for = now - state.last_activity_at
            if idle_for < timedelta(seconds=state.idle_unload_after_seconds):
                continue
            try:
                await self.unload(state.service_id, reason="idle")
            except Exception:
                continue

    def _require(self, service_id: str) -> ServiceState:
        state = self._states.get(service_id)
        if state is None:
            raise KeyError(f"Service '{service_id}' not found")
        return state

    async def _publish_state(self, state: ServiceState, event: str) -> None:
        payload = {
            "event": event,
            "service_id": state.service_id,
            "status": state.status,
            "service": state.to_dict(),
        }
        await publish("service:events", payload)
        await set_key(f"service:state:{state.service_id}", state.to_dict())
