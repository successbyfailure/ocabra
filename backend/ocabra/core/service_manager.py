from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

import httpx
import structlog

from ocabra.config import settings
from ocabra.redis_client import get_key, publish, set_key

if TYPE_CHECKING:
    from ocabra.core.gpu_manager import GPUManager

logger = structlog.get_logger(__name__)
SERVICE_OVERRIDES_KEY = "service:overrides"


@dataclass
class ServiceState:
    # ── Identity & config ─────────────────────────────────────────────────
    service_id: str
    service_type: str
    display_name: str
    base_url: str
    ui_url: str = ""
    health_path: str = "/health"
    # Optional path to poll for runtime/model status after a successful health check.
    runtime_check_path: str | None = None
    runtime_check_key: str = "runtime_loaded"
    runtime_check_model_key: str | None = None
    unload_path: str | None = None
    unload_method: str = "POST"
    unload_payload: dict[str, Any] | None = None
    post_unload_flush_path: str | None = None
    docker_container_name: str | None = None
    # When True, treat the service as model-loaded whenever it is alive.
    runtime_loaded_when_alive: bool = False
    preferred_gpu: int | None = None
    idle_unload_after_seconds: int = 600
    # "stop" — kill container; "restart" — restart container (UI stays accessible).
    idle_action: str = "stop"
    # Seconds to wait for an active generation before forcing eviction.
    # 0 = evict immediately; -1 = wait indefinitely (capped at 30 s for pressure eviction).
    generation_grace_period_s: int = 120

    # ── Live state ────────────────────────────────────────────────────────
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

    # ── Generation metrics (refreshed each health cycle) ──────────────────
    is_generating: bool = False
    queue_depth: int = 0
    vram_used_mb: int | None = None     # VRAM used by this service (MB)
    gpu_util_pct: float | None = None   # GPU utilisation % on preferred_gpu

    # ── Container resource metrics (refreshed each health cycle) ──────────
    cpu_pct: float | None = None        # Container CPU %
    mem_used_mb: int | None = None      # Container RSS memory (MB)
    mem_limit_mb: int | None = None     # Container memory limit (MB)

    # ── Internal generation tracking (not serialised) ─────────────────────
    _generation_started_at: datetime | None = field(default=None, repr=False)
    _generation_vram_peak_mb: int | None = field(default=None, repr=False)

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
            "generation_grace_period_s": self.generation_grace_period_s,
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
            # Generation metrics
            "is_generating": self.is_generating,
            "queue_depth": self.queue_depth,
            "vram_used_mb": self.vram_used_mb,
            "gpu_util_pct": self.gpu_util_pct,
            # Container resource metrics
            "cpu_pct": self.cpu_pct,
            "mem_used_mb": self.mem_used_mb,
            "mem_limit_mb": self.mem_limit_mb,
        }


class ServiceManager:
    def __init__(self) -> None:
        self._gpu_manager: GPUManager | None = None
        self._states: dict[str, ServiceState] = {
            "hunyuan": ServiceState(
                service_id="hunyuan",
                service_type="hunyuan3d",
                display_name="Hunyuan3D",
                base_url=settings.hunyuan_base_url.rstrip("/"),
                ui_url=settings.hunyuan_ui_url,
                preferred_gpu=settings.hunyuan_preferred_gpu,
                idle_unload_after_seconds=settings.hunyuan_idle_unload_seconds,
                generation_grace_period_s=settings.hunyuan_generation_grace_period_s,
                runtime_check_path="/runtime/status",
                runtime_check_key="runtime_loaded",
                unload_path="/runtime/unload",
                docker_container_name=settings.hunyuan_docker_container,
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
                generation_grace_period_s=settings.comfyui_generation_grace_period_s,
                unload_path="/free",
                unload_payload={"unload_models": True, "free_memory": True},
                docker_container_name=settings.comfyui_docker_container,
                runtime_loaded_when_alive=True,
            ),
            "a1111": ServiceState(
                service_id="a1111",
                service_type="automatic1111",
                display_name="Automatic1111",
                base_url=settings.a1111_base_url.rstrip("/"),
                ui_url=settings.a1111_ui_url,
                preferred_gpu=settings.a1111_preferred_gpu,
                idle_unload_after_seconds=settings.a1111_idle_unload_seconds,
                generation_grace_period_s=settings.a1111_generation_grace_period_s,
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
                health_path="/" if settings.acestep_base_url.endswith(("7860", "7860/")) else "/health",
                preferred_gpu=settings.acestep_preferred_gpu,
                idle_unload_after_seconds=settings.acestep_idle_unload_seconds,
                generation_grace_period_s=settings.acestep_generation_grace_period_s,
                docker_container_name=settings.acestep_docker_container,
                runtime_loaded_when_alive=True,
                idle_action="stop",
            ),
        }
        self._lock = asyncio.Lock()

    def set_gpu_manager(self, gpu_manager: GPUManager) -> None:
        self._gpu_manager = gpu_manager

    # ── Lifecycle ─────────────────────────────────────────────────────────

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

    # ── State accessors ───────────────────────────────────────────────────

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

    # ── Health & metrics refresh ──────────────────────────────────────────

    async def refresh(self, service_id: str) -> ServiceState:
        state = self._require(service_id)
        now = datetime.now(timezone.utc)
        if not state.enabled:
            container_running = await self._is_container_running(state)
            service_alive = bool(container_running)
            if not service_alive:
                service_alive = await self._check_service_alive(state)
            state.service_alive = service_alive
            state.runtime_loaded = False
            state.active_model_ref = None
            state.last_health_check_at = now
            state.status = "disabled"
            state.detail = (
                f"disabled_but_container_running:{state.docker_container_name}"
                if container_running and state.docker_container_name
                else "disabled_but_service_alive"
                if service_alive
                else "disabled"
            )
            await self._publish_state(state, "health_checked")
            return state

        try:
            service_alive = await self._check_service_alive(state)
            if not service_alive:
                raise RuntimeError("service health check failed")
            state.service_alive = True
            state.last_health_check_at = now
            state.detail = None

            await self._refresh_runtime_status(state, client=None)

            if state.runtime_loaded_when_alive and not state.runtime_check_path:
                state.runtime_loaded = True
                if state.last_activity_at is None:
                    state.last_activity_at = now

            state.status = "active" if state.runtime_loaded else "idle"

            # Refresh generation metrics (best-effort, never blocks health check)
            await self._refresh_generation_metrics(state)

            # Refresh container CPU/RAM stats (best-effort)
            await self._refresh_container_stats(state)

        except Exception as exc:
            state.service_alive = False
            state.runtime_loaded = False
            state.active_model_ref = None
            state.is_generating = False
            state.last_health_check_at = now
            state.status = "unreachable"
            state.detail = str(exc)
        await self._publish_state(state, "health_checked")
        return state

    async def _check_service_alive(self, state: ServiceState) -> bool:
        url = f"{state.base_url}{state.health_path}"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url)
                response.raise_for_status()
            return True
        except Exception:
            return False

    async def _refresh_runtime_status(self, state: ServiceState, *, client: Any) -> None:
        if not state.runtime_check_path:
            return

        if state.last_unload_at is not None:
            now = datetime.now(timezone.utc)
            if (now - state.last_unload_at).total_seconds() < 15:
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
            model_ref = payload.get(state.runtime_check_model_key)
            if model_ref:
                state.runtime_loaded = True
                state.active_model_ref = str(model_ref)
            else:
                state.runtime_loaded = False
                state.active_model_ref = None
        else:
            val = payload.get(state.runtime_check_key)
            if isinstance(val, bool):
                state.runtime_loaded = val
                if not val:
                    state.active_model_ref = None

    async def _refresh_generation_metrics(self, state: ServiceState) -> None:
        """Poll GPU and service-specific endpoints to update generation metrics.

        Detects start/end of generation and persists events to DB.
        Never raises — all failures are silently suppressed.
        """
        was_generating = state.is_generating

        # GPU utilisation from pynvml cache (sync, no IO)
        if self._gpu_manager is not None and state.preferred_gpu is not None:
            gpu_state = self._gpu_manager.get_state_nowait(state.preferred_gpu)
            if gpu_state is not None:
                state.gpu_util_pct = gpu_state.utilization_pct

        # Service-specific generation detection
        try:
            if state.service_type == "comfyui":
                await self._refresh_comfyui_metrics(state)
            elif state.service_type == "automatic1111":
                await self._refresh_a1111_metrics(state)
            else:
                # Hunyuan, ACE-Step: infer from GPU utilisation
                threshold = settings.generation_gpu_util_threshold_pct
                state.is_generating = bool(
                    state.gpu_util_pct is not None
                    and state.gpu_util_pct >= threshold
                )
                state.queue_depth = 0
        except Exception as exc:
            logger.debug(
                "generation_metrics_error",
                service_id=state.service_id,
                error=str(exc),
            )

        # Track peak VRAM during generation
        if state.is_generating and state.vram_used_mb is not None:
            peak = state._generation_vram_peak_mb or 0
            if state.vram_used_mb > peak:
                state._generation_vram_peak_mb = state.vram_used_mb

        # Detect generation start
        if state.is_generating and not was_generating:
            state._generation_started_at = datetime.now(timezone.utc)
            state._generation_vram_peak_mb = state.vram_used_mb
            logger.info("generation_started", service_id=state.service_id)

        # Detect generation end → persist event
        if not state.is_generating and was_generating and state._generation_started_at is not None:
            logger.info("generation_finished", service_id=state.service_id)
            asyncio.create_task(self._persist_generation_event(state, evicted=False))

    async def _refresh_container_stats(self, state: ServiceState) -> None:
        """Read container CPU % and memory from docker stats. Never raises."""
        if not state.docker_container_name:
            return
        try:
            code, out, _err = await self._run_docker_command(
                "stats", "--no-stream", "--format", "{{json .}}", state.docker_container_name
            )
            if code != 0 or not out:
                return
            import json as _json
            row = _json.loads(out)

            # CPU: "0.85%" → float
            cpu_str = str(row.get("CPUPerc", "")).replace("%", "").strip()
            if cpu_str:
                try:
                    state.cpu_pct = float(cpu_str)
                except ValueError:
                    pass

            # Memory: "117.2MiB / 94.17GiB" → (used_mb, limit_mb)
            mem_str = str(row.get("MemUsage", ""))
            parts = [p.strip() for p in mem_str.split("/")]
            if len(parts) == 2:
                state.mem_used_mb = self._parse_size_mb(parts[0])
                state.mem_limit_mb = self._parse_size_mb(parts[1])
        except Exception as exc:
            logger.debug("container_stats_error", service_id=state.service_id, error=str(exc))

    @staticmethod
    def _parse_size_mb(s: str) -> int | None:
        """Parse a Docker size string like '117.2MiB' or '94.17GiB' into MB."""
        s = s.strip()
        for suffix, factor in (
            ("GiB", 1024), ("GB", 1000), ("MiB", 1), ("MB", 1),
            ("KiB", 1 / 1024), ("KB", 1 / 1000),
            ("TiB", 1024 * 1024), ("TB", 1000 * 1000),
            ("B", 1 / (1024 * 1024)),
        ):
            if s.endswith(suffix):
                try:
                    return int(float(s[: -len(suffix)]) * factor)
                except ValueError:
                    return None
        return None

    async def _refresh_comfyui_metrics(self, state: ServiceState) -> None:
        async with httpx.AsyncClient(timeout=4.0) as client:
            # Queue status → is_generating + queue_depth
            try:
                r = await client.get(f"{state.base_url}/queue")
                if r.is_success:
                    data = r.json()
                    running = data.get("queue_running", [])
                    pending = data.get("queue_pending", [])
                    state.is_generating = len(running) > 0
                    state.queue_depth = len(pending)
            except Exception:
                pass

            # System stats → VRAM used
            try:
                r = await client.get(f"{state.base_url}/system_stats")
                if r.is_success:
                    devices = r.json().get("devices", [])
                    if devices:
                        d = devices[0]
                        total = d.get("vram_total", 0)
                        free = d.get("vram_free", 0)
                        state.vram_used_mb = (total - free) // (1024 * 1024)
            except Exception:
                pass

    async def _refresh_a1111_metrics(self, state: ServiceState) -> None:
        async with httpx.AsyncClient(timeout=4.0) as client:
            # Progress → is_generating
            try:
                r = await client.get(f"{state.base_url}/sdapi/v1/progress")
                if r.is_success:
                    data = r.json()
                    progress = data.get("progress", 0.0)
                    job = data.get("state", {}).get("job", "")
                    state.is_generating = bool(job) or (0 < progress < 1)
                    state.queue_depth = 0
            except Exception:
                pass

            # Memory → VRAM used
            try:
                r = await client.get(f"{state.base_url}/sdapi/v1/memory")
                if r.is_success:
                    cuda = r.json().get("cuda", {})
                    used_bytes = cuda.get("system", {}).get("used", 0)
                    if used_bytes:
                        state.vram_used_mb = used_bytes // (1024 * 1024)
            except Exception:
                pass

    # ── Service lifecycle (start / unload) ────────────────────────────────

    async def start_service(self, service_id: str) -> ServiceState:
        state = self._require(service_id)
        if not state.enabled:
            raise RuntimeError(f"Service '{service_id}' is disabled")
        if state.docker_container_name and not state.service_alive:
            await self._start_container(state)
        return state

    async def unload(self, service_id: str, reason: str = "manual") -> ServiceState:
        state = self._require(service_id)
        if not state.enabled:
            raise RuntimeError(f"Service '{service_id}' is disabled")

        # Grace period: wait for active generation to finish
        if state.is_generating:
            grace_finished = await self._wait_for_generation_grace(state, reason=reason)
            if not grace_finished:
                # Grace period expired — force evict, mark generation as interrupted
                asyncio.create_task(self._persist_generation_event(state, evicted=True))

        if not state.unload_path:
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
                state.is_generating = False
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
                state.is_generating = False
                state.last_unload_at = datetime.now(timezone.utc)
                state.status = "idle" if state.service_alive else "unreachable"
                state.detail = f"unloaded:{reason}"
                logger.info("service_runtime_unloaded", service_id=service_id, reason=reason)
                # Do NOT stop the container — the REST unload already freed GPU memory.
                # Stopping would cause Docker to restart it (restart: unless-stopped),
                # which looks like the service keeps reloading.
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

    # ── Idle eviction ─────────────────────────────────────────────────────

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
                idle_seconds = int(idle_for.total_seconds())
                logger.info(
                    "service_idle_eviction_triggered",
                    service_id=state.service_id,
                    idle_seconds=idle_seconds,
                    is_generating=state.is_generating,
                )

                # Refresh generation status before deciding
                try:
                    await self._refresh_generation_metrics(state)
                except Exception:
                    pass

                # Wait for active generation to finish (idle eviction: full grace period)
                if state.is_generating:
                    grace_finished = await self._wait_for_generation_grace(
                        state, reason="idle"
                    )
                    if not grace_finished:
                        asyncio.create_task(
                            self._persist_generation_event(state, evicted=True)
                        )

                if state.unload_path:
                    await self.unload(state.service_id, reason="idle")
                elif state.docker_container_name:
                    if state.idle_action == "restart":
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
                    state.is_generating = False
                    state.last_unload_at = now
                    state.last_activity_at = None
                    state.status = "idle" if state.idle_action == "stop" else "restarting"
                    state.detail = f"unloaded:idle:{state.idle_action}"
                    await self._publish_state(state, "runtime_unloaded")
            except Exception:
                continue

    # ── Pressure eviction ─────────────────────────────────────────────────

    def get_pressure_eviction_candidates(self, preferred_gpu: int | None = None) -> list[str]:
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

        For pressure eviction the grace period is capped at 30 s so the waiting
        model is not blocked indefinitely.
        """
        state = self._states.get(service_id)
        if not state or not state.enabled or not state.service_alive:
            return False

        now = datetime.now(timezone.utc)
        logger.info(
            "service_pressure_evict",
            service_id=service_id,
            is_generating=state.is_generating,
            idle_action=state.idle_action,
        )

        # Grace period — capped at 30 s for pressure eviction
        if state.is_generating and state.generation_grace_period_s != 0:
            max_wait = (
                min(state.generation_grace_period_s, 30)
                if state.generation_grace_period_s > 0
                else 30  # -1 = indefinite → cap at 30 s for pressure
            )
            grace_finished = await self._wait_for_generation_grace(
                state, reason="pressure", max_wait_s=max_wait
            )
            if not grace_finished:
                asyncio.create_task(self._persist_generation_event(state, evicted=True))

        if state.docker_container_name:
            if state.idle_action == "restart":
                await self._restart_container(state)
                state.status = "restarting"
            else:
                await self._stop_container(state)
                state.status = "idle"
            state.runtime_loaded = False
            state.active_model_ref = None
            state.is_generating = False
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
                logger.warning(
                    "service_pressure_evict_failed", service_id=service_id, error=str(exc)
                )
                return False

        return False

    # ── Enable / disable ──────────────────────────────────────────────────

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
            state.is_generating = False
            state.status = "disabled"
            state.detail = "disabled"
        else:
            state.status = "idle"
            state.detail = None
        await self._persist_overrides()
        await self._publish_state(state, "enabled_changed")
        return state

    # ── Generation event persistence ──────────────────────────────────────

    async def _persist_generation_event(
        self, state: ServiceState, *, evicted: bool = False
    ) -> None:
        started_at = state._generation_started_at
        if started_at is None:
            return
        now = datetime.now(timezone.utc)
        duration_ms = int((now - started_at).total_seconds() * 1000)

        # Reset tracking immediately so the next generation starts fresh
        state._generation_started_at = None
        vram_peak = state._generation_vram_peak_mb
        state._generation_vram_peak_mb = None

        try:
            from ocabra.database import AsyncSessionLocal
            from ocabra.db.stats import ServiceGenerationStat

            async with AsyncSessionLocal() as session:
                session.add(
                    ServiceGenerationStat(
                        service_id=state.service_id,
                        service_type=state.service_type,
                        started_at=started_at,
                        finished_at=now,
                        duration_ms=duration_ms,
                        gpu_index=state.preferred_gpu,
                        vram_peak_mb=vram_peak,
                        evicted=evicted,
                    )
                )
                await session.commit()
            logger.info(
                "generation_event_persisted",
                service_id=state.service_id,
                duration_ms=duration_ms,
                vram_peak_mb=vram_peak,
                evicted=evicted,
            )
        except Exception as exc:
            logger.warning(
                "generation_event_persist_failed",
                service_id=state.service_id,
                error=str(exc),
            )

    # ── Grace period wait ─────────────────────────────────────────────────

    async def _wait_for_generation_grace(
        self,
        state: ServiceState,
        reason: str,
        max_wait_s: int | None = None,
    ) -> bool:
        """Poll until the service stops generating or the grace period expires.

        Returns True if the generation finished within the grace period,
        False if it had to be forced.
        """
        grace = max_wait_s if max_wait_s is not None else state.generation_grace_period_s
        if grace == 0:
            return False
        if grace < 0:
            grace = 86400  # treat -1 as "very long" (24 h)

        state.status = "evicting_grace"
        state.detail = f"waiting_for_generation:{reason} (up to {grace}s)"
        await self._publish_state(state, "eviction_grace_started")
        logger.info(
            "eviction_grace_started",
            service_id=state.service_id,
            reason=reason,
            max_wait_s=grace,
        )

        deadline = asyncio.get_event_loop().time() + grace
        while asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(2)
            try:
                await self._refresh_generation_metrics(state)
            except Exception:
                pass
            if not state.is_generating:
                logger.info(
                    "eviction_grace_generation_finished",
                    service_id=state.service_id,
                    reason=reason,
                )
                return True

        logger.warning(
            "eviction_grace_timeout",
            service_id=state.service_id,
            reason=reason,
            max_wait_s=grace,
        )
        return False

    # ── Docker helpers ────────────────────────────────────────────────────

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
            "inspect", "-f", "{{.State.Running}}", state.docker_container_name
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

    # ── Redis persistence ─────────────────────────────────────────────────

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
