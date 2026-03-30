from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import structlog

from ocabra.config import settings
from ocabra.redis_client import get_key, publish, set_key

logger = structlog.get_logger(__name__)
SERVICE_OVERRIDES_KEY = "service:overrides"


@dataclass
class ServiceState:
    service_id: str
    service_type: str
    display_name: str
    base_url: str
    ui_url: str = ""
    health_path: str = "/health"
    # Optional path to poll for runtime/model status after a successful health check.
    # Response must be JSON. For key-based detection: set runtime_check_key.
    # For A1111-style: set runtime_check_model_key to detect a loaded model name.
    runtime_check_path: str | None = None
    runtime_check_key: str = "runtime_loaded"
    runtime_check_model_key: str | None = None
    unload_path: str | None = None
    unload_method: str = "POST"
    unload_payload: dict[str, Any] | None = None
    # Optional extra POST called after unload to flush GPU memory (e.g. /free-memory on A1111).
    post_unload_flush_path: str | None = None
    # If set, stop this Docker container after unload to fully release VRAM.
    docker_container_name: str | None = None
    # When True, the service is assumed to have its model loaded whenever service_alive is True.
    # Use for services (like ACE-Step Gradio) that have no dedicated runtime-status endpoint.
    runtime_loaded_when_alive: bool = False
    preferred_gpu: int | None = None
    # Seconds of inactivity before the model is unloaded. 0 = disabled.
    idle_unload_after_seconds: int = 600
    # "stop"    — kill the container on idle (UI goes offline, VRAM freed immediately).
    # "restart" — restart the container on idle (UI comes back after startup, VRAM freed
    #             during the restart window). Use for Gradio services like ACE-Step where
    #             keeping the UI accessible matters more than instant VRAM release.
    idle_action: str = "stop"
    enabled: bool = True
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
            "ui_url": self.ui_url,
            "health_path": self.health_path,
            "unload_path": self.unload_path,
            "preferred_gpu": self.preferred_gpu,
            "idle_unload_after_seconds": self.idle_unload_after_seconds,
            "idle_action": self.idle_action,
            "enabled": self.enabled,
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
                ui_url=settings.hunyuan_ui_url,
                preferred_gpu=settings.hunyuan_preferred_gpu,
                idle_unload_after_seconds=settings.hunyuan_idle_unload_seconds,
                runtime_check_path="/runtime/status",
                runtime_check_key="runtime_loaded",
                unload_path="/runtime/unload",
            ),
            "comfyui": ServiceState(
                service_id="comfyui",
                service_type="comfyui",
                display_name="ComfyUI",
                base_url=settings.comfyui_base_url.rstrip("/"),
                ui_url=settings.comfyui_ui_url,
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
                ui_url=settings.a1111_ui_url,
                preferred_gpu=settings.a1111_preferred_gpu,
                idle_unload_after_seconds=settings.a1111_idle_unload_seconds,
                health_path="/sdapi/v1/memory",
                runtime_check_path="/sdapi/v1/options",
                runtime_check_model_key="sd_model_checkpoint",
                unload_path="/sdapi/v1/unload-checkpoint",
                post_unload_flush_path="/free-memory",
                docker_container_name=settings.a1111_docker_container,
            ),
            "acestep": ServiceState(
                service_id="acestep",
                service_type="acestep",
                display_name="ACE-Step",
                base_url=settings.acestep_base_url.rstrip("/"),
                ui_url=settings.acestep_ui_url,
                # Gradio mode: health_path="/" (HTML, port 7860 by default)
                # API mode:    health_path="/health" (JSON, port 8001)
                # Detected automatically from base_url port suffix.
                health_path="/" if settings.acestep_base_url.endswith(("7860", "7860/")) else "/health",
                preferred_gpu=settings.acestep_preferred_gpu,
                idle_unload_after_seconds=settings.acestep_idle_unload_seconds,
                # ACE-Step auto-loads models on startup; no dedicated unload endpoint.
                # On idle: restart (not stop) so the Gradio UI stays accessible while
                # VRAM is freed during the restart window.
                # On pressure eviction: pressure_evict() still does a full stop.
                docker_container_name="ocabra-acestep-1",
                runtime_loaded_when_alive=True,
                idle_action="restart",
            ),
        }
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        await self._load_persisted_overrides()
        for state in self._states.values():
            if not state.enabled and state.docker_container_name:
                await self._stop_container_if_running(state)
        for service_id in self._states:
            await self.refresh(service_id)

    async def refresh_all(self) -> None:
        for service_id in list(self._states):
            try:
                await self.refresh(service_id)
            except Exception:
                continue

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
        if not state.enabled:
            state.runtime_loaded = False
            state.active_model_ref = None
            state.status = "disabled"
            if detail is not None:
                state.detail = detail
            await self._publish_state(state, "touched")
            return state

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
        if not state.enabled:
            state.runtime_loaded = False
            state.active_model_ref = None
            state.status = "disabled"
            if detail is not None:
                state.detail = detail
            await self._publish_state(state, "runtime_changed")
            return state

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

    async def start_service(self, service_id: str) -> ServiceState:
        state = self._require(service_id)
        if not state.enabled:
            raise RuntimeError(f"Service '{service_id}' is disabled")
        if state.docker_container_name and not state.service_alive:
            await self._start_container(state)
        return state

    async def refresh(self, service_id: str) -> ServiceState:
        state = self._require(service_id)
        now = datetime.now(timezone.utc)
        if not state.enabled:
            container_running = await self._is_container_running(state)
            state.service_alive = bool(container_running)
            state.runtime_loaded = False
            state.active_model_ref = None
            state.last_health_check_at = now
            state.status = "disabled"
            state.detail = (
                f"disabled_but_container_running:{state.docker_container_name}"
                if container_running and state.docker_container_name
                else "disabled"
            )
            await self._publish_state(state, "health_checked")
            return state

        url = f"{state.base_url}{state.health_path}"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url)
                response.raise_for_status()
            state.service_alive = True
            state.last_health_check_at = now
            state.detail = None
            # Poll runtime/model status if configured
            await self._refresh_runtime_status(state, client=None)
            # For services with no runtime-check endpoint that always have a model loaded
            if state.runtime_loaded_when_alive and not state.runtime_check_path:
                state.runtime_loaded = True
                if state.last_activity_at is None:
                    state.last_activity_at = now
            state.status = "active" if state.runtime_loaded else "idle"
        except Exception as exc:
            state.service_alive = False
            state.runtime_loaded = False
            state.active_model_ref = None
            state.last_health_check_at = now
            state.status = "unreachable"
            state.detail = str(exc)
        await self._publish_state(state, "health_checked")
        return state

    async def _refresh_runtime_status(self, state: ServiceState, *, client: Any) -> None:
        """Call the runtime check endpoint and update runtime_loaded / active_model_ref."""
        if not state.runtime_check_path:
            return

        # Avoid re-detecting a model as loaded immediately after a manual unload.
        # A1111 keeps sd_model_checkpoint in /options even after unload-checkpoint,
        # so without this guard the health loop would undo the unload within 30s.
        if state.last_unload_at is not None:
            now = datetime.now(timezone.utc)
            if (now - state.last_unload_at).total_seconds() < 120:
                state.runtime_loaded = False
                state.active_model_ref = None
                return

        url = f"{state.base_url}{state.runtime_check_path}"
        try:
            async with httpx.AsyncClient(timeout=5.0) as c:
                resp = await c.get(url)
                if resp.is_error:
                    return
                payload = resp.json()
        except Exception:
            return

        if state.runtime_check_model_key:
            # e.g. A1111: sd_model_checkpoint is non-null when a model is loaded
            model_ref = payload.get(state.runtime_check_model_key)
            if model_ref:
                state.runtime_loaded = True
                state.active_model_ref = str(model_ref)
            else:
                state.runtime_loaded = False
                state.active_model_ref = None
        else:
            # e.g. Hunyuan: runtime_loaded bool
            val = payload.get(state.runtime_check_key)
            if isinstance(val, bool):
                state.runtime_loaded = val
                if not val:
                    state.active_model_ref = None

    async def unload(self, service_id: str, reason: str = "manual") -> ServiceState:
        state = self._require(service_id)
        if not state.enabled:
            raise RuntimeError(f"Service '{service_id}' is disabled")
        if not state.unload_path:
            # No HTTP unload endpoint — fall back to container restart/stop.
            if not state.docker_container_name:
                raise RuntimeError(f"Service '{service_id}' does not expose an unload endpoint")
            async with self._lock:
                now = datetime.now(timezone.utc)
                if state.idle_action == "restart":
                    await self._restart_container(state)
                else:
                    await self._stop_container(state)
                state.runtime_loaded = False
                state.active_model_ref = None
                state.last_unload_at = now
                state.detail = f"unloaded:{reason}"
                logger.info("service_runtime_unloaded", service_id=service_id, reason=reason)
                await self._publish_state(state, "runtime_unloaded")
            return state

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
                    if state.post_unload_flush_path:
                        flush_url = f"{state.base_url}{state.post_unload_flush_path}"
                        try:
                            await client.post(flush_url)
                        except Exception as flush_exc:
                            logger.warning(
                                "service_flush_failed",
                                service_id=service_id,
                                error=str(flush_exc),
                            )
                state.runtime_loaded = False
                state.active_model_ref = None
                state.last_unload_at = datetime.now(timezone.utc)
                state.status = "idle" if state.service_alive else "unreachable"
                state.detail = f"unloaded:{reason}"
                logger.info("service_runtime_unloaded", service_id=service_id, reason=reason)
                if state.docker_container_name:
                    await self._stop_container(state)
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

    async def _run_docker_command(self, *args: str) -> tuple[int, str, str]:
        process = await asyncio.create_subprocess_exec(
            "docker",
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        return (
            process.returncode or 0,
            stdout.decode("utf-8", errors="ignore").strip(),
            stderr.decode("utf-8", errors="ignore").strip(),
        )

    async def _stop_container(self, state: ServiceState) -> None:
        if not state.docker_container_name:
            return

        code, _out, err = await self._run_docker_command("stop", state.docker_container_name)
        if code == 0:
            state.service_alive = False
            state.status = "unreachable"
            logger.info("container_stopped", container=state.docker_container_name)
            return

        logger.warning(
            "container_stop_failed",
            container=state.docker_container_name,
            error=err or f"exit_code={code}",
        )

    async def _is_container_running(self, state: ServiceState) -> bool:
        if not state.docker_container_name:
            return False

        code, out, _err = await self._run_docker_command(
            "inspect",
            "-f",
            "{{.State.Running}}",
            state.docker_container_name,
        )
        if code != 0:
            return False

        return out.strip().lower() == "true"

    async def _stop_container_if_running(self, state: ServiceState) -> None:
        if not state.docker_container_name:
            return

        if not await self._is_container_running(state):
            return

        await self._stop_container(state)

    async def _restart_container(self, state: ServiceState) -> None:
        if not state.docker_container_name:
            return

        code, _out, err = await self._run_docker_command("restart", state.docker_container_name)
        if code == 0:
            state.service_alive = False
            state.status = "restarting"
            logger.info("container_restarted", container=state.docker_container_name)
            return

        logger.warning(
            "container_restart_failed",
            container=state.docker_container_name,
            error=err or f"exit_code={code}",
        )

    async def _start_container(self, state: ServiceState) -> None:
        if not state.docker_container_name:
            return

        code, _out, err = await self._run_docker_command("start", state.docker_container_name)
        if code == 0:
            logger.info("container_started", container=state.docker_container_name)
            return

        logger.warning(
            "container_start_failed",
            container=state.docker_container_name,
            error=err or f"exit_code={code}",
        )

    def get_pressure_eviction_candidates(self, preferred_gpu: int | None = None) -> list[str]:
        """Return service IDs that can be stopped to free VRAM under pressure.

        Only includes services that are alive, have a model loaded, and can
        actually be stopped (docker_container_name or unload_path).
        Filtered to `preferred_gpu` when provided.
        """
        return [
            state.service_id
            for state in self._states.values()
            if state.enabled
            and state.service_alive
            and state.runtime_loaded
            and (preferred_gpu is None or state.preferred_gpu == preferred_gpu)
            and (state.docker_container_name or state.unload_path)
        ]

    async def pressure_evict(self, service_id: str) -> bool:
        """Evict a service to free VRAM under model-load pressure.

        Respects idle_action: services with idle_action="restart" are restarted
        (UI stays accessible) rather than stopped. Returns True if evicted.
        """
        state = self._states.get(service_id)
        if not state or not state.enabled or not state.service_alive:
            return False

        now = datetime.now(timezone.utc)
        logger.info("service_pressure_evict", service_id=service_id, idle_action=state.idle_action)

        if state.docker_container_name:
            if state.idle_action == "restart":
                await self._restart_container(state)
                state.status = "restarting"
            else:
                await self._stop_container(state)
                state.status = "idle"
            state.runtime_loaded = False
            state.active_model_ref = None
            state.last_unload_at = now
            state.last_activity_at = None
            state.detail = "unloaded:pressure"
            await self._publish_state(state, "runtime_unloaded")
            return True

        if state.unload_path:
            try:
                await self.unload(service_id, reason="pressure")
                return True
            except Exception as exc:
                logger.warning("service_pressure_evict_failed", service_id=service_id, error=str(exc))
                return False

        return False

    async def check_idle_unloads(self) -> None:
        now = datetime.now(timezone.utc)
        for state in self._states.values():
            if not state.enabled:
                continue
            if state.idle_unload_after_seconds <= 0:
                continue
            if not state.service_alive or not state.runtime_loaded:
                continue
            if state.last_activity_at is None:
                continue
            idle_for = now - state.last_activity_at
            if idle_for < timedelta(seconds=state.idle_unload_after_seconds):
                continue
            try:
                if state.unload_path:
                    await self.unload(state.service_id, reason="idle")
                elif state.docker_container_name:
                    idle_seconds = int(idle_for.total_seconds())
                    if state.idle_action == "restart":
                        # Restart keeps the UI accessible; VRAM is freed during startup.
                        logger.info(
                            "service_idle_container_restart",
                            service_id=state.service_id,
                            idle_seconds=idle_seconds,
                        )
                        await self._restart_container(state)
                    else:
                        logger.info(
                            "service_idle_container_stop",
                            service_id=state.service_id,
                            idle_seconds=idle_seconds,
                        )
                        await self._stop_container(state)
                    state.runtime_loaded = False
                    state.active_model_ref = None
                    state.last_unload_at = now
                    state.last_activity_at = None  # reset so timer starts fresh after restart
                    state.status = "idle" if state.idle_action == "stop" else "restarting"
                    state.detail = f"unloaded:idle:{state.idle_action}"
                    await self._publish_state(state, "runtime_unloaded")
            except Exception:
                continue

    async def set_enabled(self, service_id: str, enabled: bool) -> ServiceState:
        state = self._require(service_id)
        if not enabled and state.enabled:
            if state.runtime_loaded and state.unload_path:
                try:
                    await self.unload(service_id, reason="disabled")
                except Exception as exc:
                    logger.warning(
                        "service_disable_unload_failed",
                        service_id=service_id,
                        error=str(exc),
                    )
            if state.docker_container_name:
                await self._stop_container_if_running(state)

        state.enabled = enabled
        if not enabled:
            state.service_alive = False
            state.runtime_loaded = False
            state.active_model_ref = None
            state.status = "disabled"
            state.detail = "disabled"
        else:
            state.status = "idle"
            state.detail = None
        await self._persist_overrides()
        await self._publish_state(state, "enabled_changed")
        return state

    async def _load_persisted_overrides(self) -> None:
        try:
            payload = await get_key(SERVICE_OVERRIDES_KEY)
        except Exception as exc:
            logger.warning("service_overrides_load_failed", error=str(exc))
            return

        if not isinstance(payload, dict):
            return

        for service_id, overrides in payload.items():
            state = self._states.get(str(service_id))
            if state is None or not isinstance(overrides, dict):
                continue
            if "enabled" in overrides:
                state.enabled = bool(overrides.get("enabled"))
            if not state.enabled:
                state.service_alive = False
                state.runtime_loaded = False
                state.active_model_ref = None
                state.status = "disabled"
                state.detail = "disabled"

    async def _persist_overrides(self) -> None:
        payload = {
            service_id: {"enabled": state.enabled}
            for service_id, state in self._states.items()
        }
        try:
            await set_key(SERVICE_OVERRIDES_KEY, payload)
        except Exception as exc:
            logger.warning("service_overrides_persist_failed", error=str(exc))

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
