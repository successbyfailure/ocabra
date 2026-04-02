import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from threading import Lock
from urllib.parse import urlparse

import sqlalchemy as sa
import structlog

import ocabra.database as database
from ocabra.backends.base import BackendCapabilities, WorkerInfo
from ocabra.config import settings
from ocabra.core.model_manager_helpers import (
    build_diarized_extra_config,
    diarized_variant_model_id,
    estimate_bitnet_vram_from_config,
    is_diarized_model_id,
    resolve_bitnet_gpu_layers,
    resolve_bitnet_option,
    should_auto_create_diarized_variant,
)
from ocabra.core.model_ref import build_model_ref, normalize_model_ref, parse_model_ref
from ocabra.db.model_config import ModelConfig
from ocabra.db.stats import ModelLoadStat
from ocabra.redis_client import get_key, publish, set_key

logger = structlog.get_logger(__name__)


class ModelStatus(str, Enum):
    DISCOVERED = "discovered"
    CONFIGURED = "configured"
    LOADING = "loading"
    LOADED = "loaded"
    UNLOADING = "unloading"
    UNLOADED = "unloaded"
    ERROR = "error"


class LoadPolicy(str, Enum):
    PIN = "pin"
    WARM = "warm"
    ON_DEMAND = "on_demand"


@dataclass
class ModelState:
    # Canonical identifier: backend/model
    model_id: str
    display_name: str
    backend_type: str
    # Backend-native identifier (without backend prefix)
    backend_model_id: str = ""
    status: ModelStatus = ModelStatus.CONFIGURED
    load_policy: LoadPolicy = LoadPolicy.ON_DEMAND
    auto_reload: bool = False
    preferred_gpu: int | None = None
    current_gpu: list[int] = field(default_factory=list)
    vram_used_mb: int = 0
    capabilities: BackendCapabilities = field(default_factory=BackendCapabilities)
    last_request_at: datetime | None = None
    loaded_at: datetime | None = None
    worker_info: WorkerInfo | None = None
    error_message: str | None = None
    extra_config: dict = field(default_factory=dict)


    def __post_init__(self) -> None:
        if self.backend_model_id:
            return
        try:
            _, backend_model_id = parse_model_ref(self.model_id)
            self.backend_model_id = backend_model_id
        except ValueError:
            # Tests and in-memory fixtures may still build non-canonical ids.
            self.backend_model_id = self.model_id

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "backend_model_id": self.backend_model_id,
            "display_name": self.display_name,
            "backend_type": self.backend_type,
            "status": self.status.value,
            "load_policy": self.load_policy.value,
            "auto_reload": self.auto_reload,
            "preferred_gpu": self.preferred_gpu,
            "current_gpu": self.current_gpu,
            "vram_used_mb": self.vram_used_mb,
            "capabilities": self.capabilities.to_dict(),
            "last_request_at": (
                self.last_request_at.isoformat() if self.last_request_at else None
            ),
            "loaded_at": self.loaded_at.isoformat() if self.loaded_at else None,
            "error_message": self.error_message,
            "extra_config": self.extra_config,
        }

class ModelManager:
    def __init__(self, worker_pool, gpu_manager=None, gpu_scheduler=None) -> None:
        self._worker_pool = worker_pool
        self._gpu_manager = gpu_manager
        self._gpu_scheduler = gpu_scheduler
        self._service_manager = None
        self._states: dict[str, ModelState] = {}
        self._load_locks: dict[str, asyncio.Lock] = {}
        self._persisted_model_ids: set[str] = set()
        self._event_listeners: list[Callable[[dict], Awaitable[None]]] = []
        # In-flight request tracking: model_id → active request count
        self._in_flight: dict[str, int] = {}
        self._in_flight_lock = Lock()

    def begin_request(self, model_id: str) -> None:
        """Mark one request as in-flight for this model."""
        with self._in_flight_lock:
            self._in_flight[model_id] = self._in_flight.get(model_id, 0) + 1

    def end_request(self, model_id: str) -> None:
        """Mark one in-flight request for this model as complete."""
        with self._in_flight_lock:
            count = self._in_flight.get(model_id, 0)
            if count <= 1:
                self._in_flight.pop(model_id, None)
            else:
                self._in_flight[model_id] = count - 1

    def is_busy(self, model_id: str) -> bool:
        """Return True if there are in-flight requests for this model."""
        with self._in_flight_lock:
            return self._in_flight.get(model_id, 0) > 0

    async def _ensure_diarized_variants_for_whisper_models(self) -> None:
        snapshot = list(self._states.values())
        for state in snapshot:
            if not should_auto_create_diarized_variant(state):
                continue
            diarized_backend_model_id = diarized_variant_model_id(state.backend_model_id)
            diarized_model_id = build_model_ref(state.backend_type, diarized_backend_model_id)
            if diarized_model_id in self._states:
                continue
            try:
                await self.add_model(
                    model_id=diarized_model_id,
                    backend_type=state.backend_type,
                    display_name=f"{state.display_name} (diarized)",
                    load_policy=state.load_policy.value,
                    auto_reload=state.auto_reload,
                    preferred_gpu=state.preferred_gpu,
                    extra_config=build_diarized_extra_config(state.extra_config),
                    create_diarized_variant=False,
                )
            except Exception as exc:
                logger.warning(
                    "diarized_variant_auto_create_failed",
                    model_id=state.model_id,
                    diarized_model_id=diarized_model_id,
                    error=str(exc),
                )

    def set_gpu_manager(self, gpu_manager) -> None:
        self._gpu_manager = gpu_manager

    def set_gpu_scheduler(self, scheduler) -> None:
        self._gpu_scheduler = scheduler

    def set_service_manager(self, service_manager) -> None:
        self._service_manager = service_manager

    def register_event_listener(self, listener: Callable[[dict], Awaitable[None]]) -> None:
        self._event_listeners.append(listener)

    def _create_background_task(
        self,
        coro: Awaitable[object],
        *,
        task_name: str,
        **context: object,
    ) -> asyncio.Task:
        task = asyncio.create_task(coro, name=task_name)

        def _log_task_result(done: asyncio.Task) -> None:
            with suppress(asyncio.CancelledError):
                exc = done.exception()
                if exc is not None:
                    logger.warning(
                        "background_task_failed",
                        task=task_name,
                        error=str(exc),
                        **context,
                    )

        task.add_done_callback(_log_task_result)
        return task

    async def start(self) -> None:
        await self._load_configs_from_db()
        await self._ensure_diarized_variants_for_whisper_models()
        await self._hydrate_last_request_at_from_redis()
        for model_id, state in self._states.items():
            if state.load_policy == LoadPolicy.PIN:
                logger.info("auto_loading_pinned_model", model_id=model_id)
                self._create_background_task(
                    self._load_model(model_id),
                    task_name=f"load:{model_id}",
                    model_id=model_id,
                )

    async def _load_configs_from_db(self) -> None:
        async with database.AsyncSessionLocal() as session:
            result = await session.execute(sa.select(ModelConfig))
            configs = list(result.scalars().all())

            canonical_map: dict[str, tuple[ModelConfig, str]] = {}
            duplicates: list[ModelConfig] = []
            for cfg in configs:
                try:
                    canonical_model_id, backend_model_id = normalize_model_ref(
                        cfg.backend_type, cfg.model_id
                    )
                except ValueError as exc:
                    logger.warning(
                        "invalid_model_config_skipped",
                        model_id=cfg.model_id,
                        backend_type=cfg.backend_type,
                        error=str(exc),
                    )
                    continue

                normalized_backend = str(cfg.backend_type or "").strip().lower()
                if cfg.model_id != canonical_model_id:
                    cfg.model_id = canonical_model_id
                if cfg.backend_type != normalized_backend:
                    cfg.backend_type = normalized_backend

                if canonical_model_id in canonical_map:
                    duplicates.append(cfg)
                    continue
                canonical_map[canonical_model_id] = (cfg, backend_model_id)

            for duplicate in duplicates:
                await session.delete(duplicate)

            await session.commit()

        for canonical_model_id, (cfg, backend_model_id) in canonical_map.items():
            self._states[canonical_model_id] = ModelState(
                model_id=canonical_model_id,
                backend_model_id=backend_model_id,
                display_name=cfg.display_name or backend_model_id,
                backend_type=str(cfg.backend_type or "").strip().lower(),
                load_policy=LoadPolicy(cfg.load_policy),
                auto_reload=cfg.auto_reload,
                preferred_gpu=cfg.preferred_gpu,
                extra_config=cfg.extra_config or {},
            )
            self._load_locks[canonical_model_id] = asyncio.Lock()
            self._persisted_model_ids.add(canonical_model_id)

    async def _publish_event(self, model_id: str, event: str) -> None:
        state = self._states.get(model_id)
        if state:
            await publish(
                "model:events",
                {
                    "event": event,
                    "model_id": model_id,
                    "new_status": state.status.value,
                },
            )
            await self._persist_state(model_id)

    async def _persist_state(self, model_id: str) -> None:
        state = self._states.get(model_id)
        if not state:
            return
        try:
            await set_key(f"model:state:{model_id}", state.to_dict())
        except Exception as exc:
            logger.warning("model_state_persist_failed", model_id=model_id, error=str(exc))

    async def _hydrate_last_request_at_from_redis(self) -> None:
        for model_id, state in self._states.items():
            try:
                payload = await get_key(f"model:state:{model_id}")
            except Exception as exc:
                logger.warning(
                    "model_state_restore_failed",
                    model_id=model_id,
                    error=str(exc),
                )
                continue

            if not isinstance(payload, dict):
                continue

            parsed = self._parse_datetime(payload.get("last_request_at"))
            if parsed is not None:
                state.last_request_at = parsed

    @staticmethod
    def _parse_datetime(value: object) -> datetime | None:
        if not isinstance(value, str) or not value.strip():
            return None

        normalized = value.strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    async def touch_last_request_at(self, model_id: str, at: datetime | None = None) -> None:
        state = self._states.get(model_id)
        if not state:
            raise KeyError(f"Model '{model_id}' not configured")

        state.last_request_at = at or datetime.now(timezone.utc)
        await self._persist_state(model_id)

    def _notify_event_listeners(self, event: str, state: "ModelState") -> None:
        if not self._event_listeners:
            return

        payload = {
            "event": event,
            "model_id": state.model_id,
            "backend_type": state.backend_type,
            "display_name": state.display_name,
            "load_policy": state.load_policy.value,
            "new_status": state.status.value,
            "status": state.status.value,
        }
        for listener in list(self._event_listeners):
            self._create_background_task(
                self._run_event_listener(listener, payload),
                task_name=f"listener:{payload.get('event')}:{payload.get('model_id')}",
                model_id=payload.get("model_id"),
                event=payload.get("event"),
            )

    async def _run_event_listener(
        self,
        listener: Callable[[dict], Awaitable[None]],
        payload: dict,
    ) -> None:
        try:
            await listener(payload)
        except Exception as exc:
            logger.warning(
                "model_event_listener_failed",
                lifecycle_event=payload.get("event"),
                model_id=payload.get("model_id"),
                error=str(exc),
            )

    async def _load_model(
        self, model_id: str, force_gpu: int | None = None
    ) -> "ModelState":
        state = self._states[model_id]
        lock = self._load_locks.setdefault(model_id, asyncio.Lock())

        async with lock:
            if state.status == ModelStatus.LOADED:
                return state

            state.status = ModelStatus.LOADING
            await self._publish_event(model_id, "status_changed")
            load_started_at = datetime.now(timezone.utc)

            try:
                backend = await self._worker_pool.get_backend(state.backend_type)
                vram_needed = await backend.get_vram_estimate_mb(state.backend_model_id)
                if state.backend_type == "bitnet":
                    vram_needed = self._estimate_bitnet_vram_from_config(state)

                bitnet_cpu_only = (
                    state.backend_type == "bitnet" and self._resolve_bitnet_gpu_layers(state) <= 0
                )
                gpu_managed = state.backend_type != "ollama" and not bitnet_cpu_only
                needs_port = state.backend_type != "ollama"
                assigned_port = await self._worker_pool.assign_port() if needs_port else 0
                backend_loaded = False

                if not gpu_managed:
                    gpu_indices = []
                elif self._gpu_scheduler:
                    gpu_indices = await self._assign_gpus_for_load(
                        model_id=model_id,
                        vram_needed=vram_needed,
                        preferred_gpu=force_gpu or state.preferred_gpu,
                        enforce_vllm_headroom=(state.backend_type == "vllm"),
                    )
                else:
                    gpu_indices = [settings.default_gpu_index]

                # vLLM checks against a fraction of total VRAM, not only model weights.
                if state.backend_type == "vllm" and self._gpu_manager:
                    from ocabra.core.scheduler import InsufficientVRAMError

                    for gpu_idx in gpu_indices:
                        gpu_state = await self._gpu_manager.get_state(gpu_idx)
                        required_free_mb = int(
                            gpu_state.total_vram_mb * settings.vllm_gpu_memory_utilization
                        )
                        if gpu_state.free_vram_mb < required_free_mb:
                            raise InsufficientVRAMError(
                                "GPU "
                                f"{gpu_idx} has {gpu_state.free_vram_mb} MB free, "
                                f"vLLM requires at least {required_free_mb} MB free "
                                f"(vllm_gpu_memory_utilization={settings.vllm_gpu_memory_utilization})."
                            )

                worker_info = await backend.load(
                    state.backend_model_id,
                    gpu_indices,
                    port=assigned_port,
                    extra_config=state.extra_config,
                )
                backend_loaded = True
                capabilities = await backend.get_capabilities(state.backend_model_id)

                actual_gpu_indices = worker_info.gpu_indices or gpu_indices
                state.worker_info = worker_info
                state.current_gpu = actual_gpu_indices
                state.vram_used_mb = worker_info.vram_used_mb
                state.capabilities = capabilities
                state.status = ModelStatus.LOADED
                state.loaded_at = datetime.now(timezone.utc)
                state.error_message = None

                await self._record_model_load_stat(
                    model_id=model_id,
                    backend_type=state.backend_type,
                    started_at=load_started_at,
                    finished_at=state.loaded_at,
                    gpu_indices=actual_gpu_indices,
                )

                if gpu_managed and self._gpu_manager:
                    vram_per_gpu = worker_info.vram_used_mb // max(1, len(actual_gpu_indices))
                    for gpu_idx in actual_gpu_indices:
                        await self._gpu_manager.lock_vram(
                            gpu_idx, vram_per_gpu, model_id
                        )

                self._worker_pool.set_worker(model_id, worker_info)
                await self._publish_event(model_id, "status_changed")
                self._notify_event_listeners("load", state)
                logger.info(
                    "model_loaded",
                    model_id=model_id,
                    gpu=gpu_indices,
                    vram_mb=worker_info.vram_used_mb,
                )
            except Exception as e:
                state.status = ModelStatus.ERROR
                state.error_message = str(e)
                # Ensure no leaked subprocess/port after partial load failures.
                try:
                    if "backend_loaded" in locals() and backend_loaded:
                        await backend.unload(state.backend_model_id)
                except Exception:
                    logger.warning(
                        "model_load_cleanup_failed",
                        model_id=model_id,
                    )
                self._worker_pool.remove_worker(model_id)
                if "assigned_port" in locals() and assigned_port:
                    self._worker_pool.release_port(assigned_port)
                await self._publish_event(model_id, "load_failed")
                logger.error("model_load_failed", model_id=model_id, error=str(e))
                raise

        return state

    async def _record_model_load_stat(
        self,
        model_id: str,
        backend_type: str,
        started_at: datetime,
        finished_at: datetime | None,
        gpu_indices: list[int],
    ) -> None:
        duration_ms = None
        if finished_at is not None:
            duration_ms = max(0, int((finished_at - started_at).total_seconds() * 1000))

        try:
            async with database.AsyncSessionLocal() as session:
                session.add(
                    ModelLoadStat(
                        model_id=model_id,
                        backend_type=backend_type,
                        started_at=started_at,
                        finished_at=finished_at,
                        duration_ms=duration_ms,
                        gpu_count=len(gpu_indices),
                        gpu_indices=",".join(str(idx) for idx in gpu_indices) if gpu_indices else None,
                    )
                )
                await session.commit()
        except Exception as exc:
            logger.warning("model_load_stat_write_failed", model_id=model_id, error=str(exc))

    def _resolve_bitnet_option(self, state: "ModelState", key: str, default: int) -> int:
        return resolve_bitnet_option(state, key, default)

    def _resolve_bitnet_gpu_layers(self, state: "ModelState") -> int:
        return resolve_bitnet_gpu_layers(state, settings.bitnet_gpu_layers)

    def _estimate_bitnet_vram_from_config(self, state: "ModelState") -> int:
        return estimate_bitnet_vram_from_config(state, default_gpu_layers=settings.bitnet_gpu_layers)

    async def _assign_gpus_for_load(
        self,
        model_id: str,
        vram_needed: int,
        preferred_gpu: int | None,
        enforce_vllm_headroom: bool,
    ) -> list[int]:
        from ocabra.core.scheduler import InsufficientVRAMError

        if not self._gpu_scheduler:
            return [settings.default_gpu_index]

        try:
            return await self._gpu_scheduler.find_gpu_for_model(
                vram_needed,
                preferred_gpu,
                enforce_vllm_headroom=enforce_vllm_headroom,
            )
        except InsufficientVRAMError as initial_exc:
            last_exc = initial_exc

            # First: try stopping external services (ACE-Step, ComfyUI, etc.) on the
            # preferred GPU — they often hold large amounts of VRAM and can be restarted.
            if self._service_manager:
                service_candidates = self._service_manager.get_pressure_eviction_candidates(preferred_gpu)
                if service_candidates:
                    logger.info(
                        "pressure_eviction_service_candidates",
                        requested_model_id=model_id,
                        candidates=service_candidates,
                    )
                for service_id in service_candidates:
                    logger.info(
                        "pressure_eviction_service_attempt",
                        requested_model_id=model_id,
                        evicting_service_id=service_id,
                    )
                    evicted = await self._service_manager.pressure_evict(service_id)
                    if not evicted:
                        continue
                    await self._wait_for_vram_released(0)  # wait for VRAM to settle
                    try:
                        return await self._gpu_scheduler.find_gpu_for_model(
                            vram_needed,
                            preferred_gpu,
                            enforce_vllm_headroom=enforce_vllm_headroom,
                        )
                    except InsufficientVRAMError as exc:
                        last_exc = exc
                        continue

            # Then: evict model workers (on-demand first, then warm, then pinned)
            candidates = self._get_pressure_eviction_candidates(model_id)
            logger.info(
                "pressure_eviction_candidates",
                requested_model_id=model_id,
                candidates=candidates,
            )
            drain_timeout_s = max(1, int(settings.pressure_eviction_drain_timeout_s))
            for candidate_id in candidates:
                # Wait for any in-flight requests to drain before evicting
                if self.is_busy(candidate_id):
                    logger.info(
                        "pressure_eviction_drain_wait",
                        requested_model_id=model_id,
                        evicting_model_id=candidate_id,
                        drain_timeout_s=drain_timeout_s,
                    )
                    for _ in range(drain_timeout_s * 2):
                        if not self.is_busy(candidate_id):
                            break
                        await asyncio.sleep(0.5)
                    if self.is_busy(candidate_id):
                        logger.warning(
                            "pressure_eviction_drain_timeout_skip",
                            evicting_model_id=candidate_id,
                        )
                        continue  # skip — still busy, try next candidate

                logger.info(
                    "pressure_eviction_attempt",
                    requested_model_id=model_id,
                    evicting_model_id=candidate_id,
                )
                evicted_vram_mb = self._states.get(candidate_id).vram_used_mb if self._states.get(candidate_id) else 0
                await self.unload(candidate_id, reason="pressure")
                await self._wait_for_vram_released(evicted_vram_mb)
                try:
                    return await self._gpu_scheduler.find_gpu_for_model(
                        vram_needed,
                        preferred_gpu,
                        enforce_vllm_headroom=enforce_vllm_headroom,
                    )
                except InsufficientVRAMError as exc:
                    last_exc = exc
                    continue
            raise last_exc

    def _get_pressure_eviction_candidates(self, requested_model_id: str) -> list[str]:
        policy_order = {
            LoadPolicy.ON_DEMAND: 0,
            LoadPolicy.WARM: 1,
            LoadPolicy.PIN: 2,
        }
        fallback_time = datetime.min.replace(tzinfo=timezone.utc)

        candidates = [
            state
            for state in self._states.values()
            if state.model_id != requested_model_id
            and state.status == ModelStatus.LOADED
        ]
        candidates.sort(
            key=lambda state: (
                policy_order.get(state.load_policy, 99),
                state.last_request_at or state.loaded_at or fallback_time,
            )
        )
        if candidates:
            return [state.model_id for state in candidates]
        return [
            model_id
            for model_id, worker in self._worker_pool._workers.items()
            if model_id != requested_model_id and worker.backend_type != "ollama"
        ]

    async def _wait_for_vram_released(self, released_vram_mb: int) -> None:
        if not self._gpu_manager or released_vram_mb <= 0:
            return
        await asyncio.sleep(2.5)

    async def load(
        self, model_id: str, force_gpu: int | None = None
    ) -> "ModelState":
        if model_id not in self._states:
            raise KeyError(f"Model '{model_id}' not configured")
        return await self._load_model(model_id, force_gpu)

    async def unload(self, model_id: str, reason: str = "manual") -> None:
        state = self._states.get(model_id)
        if not state or state.status != ModelStatus.LOADED:
            return

        state.status = ModelStatus.UNLOADING
        await self._publish_event(model_id, "status_changed")

        try:
            backend = await self._worker_pool.get_backend(state.backend_type)
            await backend.unload(state.backend_model_id)

            if self._gpu_manager:
                for gpu_idx in state.current_gpu:
                    await self._gpu_manager.unlock_vram(gpu_idx, model_id)

            self._worker_pool.remove_worker(model_id)
            state.status = ModelStatus.UNLOADED
            state.current_gpu = []
            state.vram_used_mb = 0
            state.worker_info = None
            state.loaded_at = None
            await self._publish_event(model_id, "status_changed")
            self._notify_event_listeners("unload", state)
            logger.info("model_unloaded", model_id=model_id, reason=reason)

            if reason == "pressure" and state.auto_reload:
                self._create_background_task(
                    self._watch_and_reload(model_id),
                    task_name=f"watch-and-reload:{model_id}",
                    model_id=model_id,
                )
        except Exception as e:
            state.status = ModelStatus.ERROR
            state.error_message = str(e)
            await self._publish_event(model_id, "unload_failed")
            logger.error("model_unload_failed", model_id=model_id, error=str(e))
            raise

    async def _watch_and_reload(self, model_id: str) -> None:
        if not self._gpu_scheduler:
            return
        deadline = datetime.now(timezone.utc) + timedelta(
            seconds=max(30, int(settings.model_load_wait_timeout_s))
        )
        while model_id in self._states:
            state = self._states.get(model_id)
            if not state or not state.auto_reload:
                return
            if state.status != ModelStatus.UNLOADED:
                return
            if datetime.now(timezone.utc) >= deadline:
                logger.warning("watch_and_reload_timeout", model_id=model_id)
                return
            await asyncio.sleep(30)
            state = self._states.get(model_id)
            if not state or not state.auto_reload or state.status != ModelStatus.UNLOADED:
                return
            try:
                backend = await self._worker_pool.get_backend(state.backend_type)
                vram_needed = await backend.get_vram_estimate_mb(state.backend_model_id)
                gpu_indices = await self._gpu_scheduler.find_gpu_for_model(
                    vram_needed, state.preferred_gpu
                )
                if gpu_indices:
                    await self._load_model(model_id)
                    break
            except Exception:
                continue

    async def check_idle_evictions(self) -> None:
        if settings.idle_timeout_seconds <= 0:
            return
        now = datetime.now(timezone.utc)
        for model_id, state in list(self._states.items()):
            if state.status != ModelStatus.LOADED:
                continue
            if state.load_policy != LoadPolicy.ON_DEMAND:
                continue
            if state.last_request_at is None:
                continue
            idle_s = (now - state.last_request_at).total_seconds()
            if idle_s > settings.idle_timeout_seconds:
                logger.info("idle_eviction", model_id=model_id, idle_s=int(idle_s))
                self._create_background_task(
                    self.unload(model_id, reason="idle"),
                    task_name=f"idle-unload:{model_id}",
                    model_id=model_id,
                )

    async def get_state(self, model_id: str) -> "ModelState | None":
        return self._states.get(model_id)

    async def list_states(self) -> list["ModelState"]:
        return list(self._states.values())

    async def sync_ollama_models(self, model_ids: list[str]) -> int:
        """Ensure native Ollama models are present in the internal inventory."""
        added = 0
        seen: set[str] = set()
        for backend_model_id in model_ids:
            normalized = backend_model_id.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            model_id = build_model_ref("ollama", normalized)
            if model_id in self._states:
                continue

            self._states[model_id] = ModelState(
                model_id=model_id,
                backend_model_id=normalized,
                backend_type="ollama",
                display_name=normalized,
                load_policy=LoadPolicy.ON_DEMAND,
            )
            self._load_locks[model_id] = asyncio.Lock()
            added += 1
        return added

    async def sync_ollama_inventory(
        self,
        installed_model_ids: list[str],
        loaded_model_ids: list[str] | None = None,
        loaded_vram_mb: dict[str, int] | None = None,
    ) -> int:
        added = await self.sync_ollama_models(installed_model_ids)
        installed_set = {model_id.strip() for model_id in installed_model_ids if model_id.strip()}
        loaded_set = {model_id.strip() for model_id in (loaded_model_ids or []) if model_id.strip()}
        loaded_vram_map = {k.strip(): int(v) for k, v in (loaded_vram_mb or {}).items() if k and str(k).strip()}
        parsed = urlparse(settings.ollama_base_url)
        ollama_port = int(parsed.port or 11434)
        changed: list[str] = []
        try:
            ollama_backend = await self._worker_pool.get_backend("ollama")
        except Exception:
            ollama_backend = None

        for model_id, state in self._states.items():
            if state.backend_type != "ollama":
                continue
            backend_model_id = state.backend_model_id
            if backend_model_id not in installed_set:
                continue
            if ollama_backend is not None:
                state.capabilities = await ollama_backend.get_capabilities(backend_model_id)

            if backend_model_id in loaded_set:
                loaded_vram = max(0, int(loaded_vram_map.get(backend_model_id, 0)))
                if state.status != ModelStatus.LOADED:
                    state.status = ModelStatus.LOADED
                    state.loaded_at = state.loaded_at or datetime.now(timezone.utc)
                    state.error_message = None
                    state.current_gpu = []
                    state.vram_used_mb = loaded_vram
                    state.worker_info = WorkerInfo(
                        backend_type="ollama",
                        model_id=backend_model_id,
                        gpu_indices=[],
                        port=ollama_port,
                        pid=0,
                        vram_used_mb=loaded_vram,
                    )
                    self._worker_pool.set_worker(model_id, state.worker_info)
                    changed.append(model_id)
                else:
                    state.vram_used_mb = loaded_vram
                    if state.worker_info is not None:
                        state.worker_info.vram_used_mb = loaded_vram
                continue

            if state.status in {ModelStatus.LOADED, ModelStatus.LOADING, ModelStatus.UNLOADING}:
                state.status = ModelStatus.UNLOADED
                state.current_gpu = []
                state.vram_used_mb = 0
                state.worker_info = None
                state.loaded_at = None
                state.error_message = None
                self._worker_pool.remove_worker(model_id)
                changed.append(model_id)

        for model_id in changed:
            await self._publish_event(model_id, "status_changed")
            state = self._states.get(model_id)
            if not state:
                continue
            if state.status == ModelStatus.LOADED:
                self._notify_event_listeners("load", state)
            elif state.status == ModelStatus.UNLOADED:
                self._notify_event_listeners("unload", state)

        return added

    async def add_model(
        self,
        model_id: str,
        backend_type: str,
        display_name: str | None = None,
        load_policy: str = "on_demand",
        auto_reload: bool = False,
        preferred_gpu: int | None = None,
        extra_config: dict | None = None,
        create_diarized_variant: bool = True,
    ) -> "ModelState":
        normalized_model_id, backend_model_id = normalize_model_ref(backend_type, model_id)
        normalized_backend = str(backend_type or "").strip().lower()

        if normalized_model_id in self._states:
            return self._states[normalized_model_id]

        async with database.AsyncSessionLocal() as session:
            cfg = ModelConfig(
                model_id=normalized_model_id,
                display_name=display_name or backend_model_id,
                backend_type=normalized_backend,
                load_policy=load_policy,
                auto_reload=auto_reload,
                preferred_gpu=preferred_gpu,
                extra_config=extra_config,
            )
            session.add(cfg)
            await session.commit()

        state = ModelState(
            model_id=normalized_model_id,
            backend_model_id=backend_model_id,
            display_name=display_name or backend_model_id,
            backend_type=normalized_backend,
            load_policy=LoadPolicy(load_policy),
            auto_reload=auto_reload,
            preferred_gpu=preferred_gpu,
            extra_config=extra_config or {},
        )
        self._states[normalized_model_id] = state
        self._load_locks[normalized_model_id] = asyncio.Lock()
        self._persisted_model_ids.add(normalized_model_id)
        self._notify_event_listeners("register", state)

        if create_diarized_variant and should_auto_create_diarized_variant(state):
            diarized_backend_model_id = diarized_variant_model_id(backend_model_id)
            diarized_model_id = build_model_ref(normalized_backend, diarized_backend_model_id)
            if diarized_model_id not in self._states:
                try:
                    await self.add_model(
                        model_id=diarized_model_id,
                        backend_type=normalized_backend,
                        display_name=f"{display_name or backend_model_id} (diarized)",
                        load_policy=load_policy,
                        auto_reload=auto_reload,
                        preferred_gpu=preferred_gpu,
                        extra_config=build_diarized_extra_config(extra_config),
                        create_diarized_variant=False,
                    )
                except Exception as exc:
                    logger.warning(
                        "diarized_variant_auto_create_failed",
                        model_id=model_id,
                        diarized_model_id=diarized_model_id,
                        error=str(exc),
                    )
        return state

    async def update_config(self, model_id: str, patch: dict) -> "ModelState":
        state = self._states.get(model_id)
        if not state:
            raise KeyError(f"Model '{model_id}' not found")

        allowed_fields = {
            "display_name",
            "load_policy",
            "auto_reload",
            "preferred_gpu",
            "extra_config",
        }
        unsupported_fields = sorted(set(patch) - allowed_fields)
        if unsupported_fields:
            raise ValueError(
                "Unsupported model config fields: " + ", ".join(unsupported_fields)
            )

        if model_id in self._persisted_model_ids:
            async with database.AsyncSessionLocal() as session:
                result = await session.execute(
                    sa.select(ModelConfig).where(ModelConfig.model_id == model_id)
                )
                cfg = result.scalar_one_or_none()
                if cfg:
                    for key, val in patch.items():
                        if key == "load_policy":
                            setattr(cfg, key, LoadPolicy(val).value)
                        elif key == "preferred_gpu":
                            setattr(cfg, key, int(val) if val is not None else None)
                        elif key == "auto_reload":
                            setattr(cfg, key, bool(val))
                        elif key == "extra_config":
                            setattr(cfg, key, val or {})
                        else:
                            setattr(cfg, key, val)
                    await session.commit()

        for key, val in patch.items():
            if key == "load_policy":
                setattr(state, key, LoadPolicy(val))
            elif key == "preferred_gpu":
                setattr(state, key, int(val) if val is not None else None)
            elif key == "auto_reload":
                setattr(state, key, bool(val))
            elif key == "extra_config":
                setattr(state, key, val or {})
            else:
                setattr(state, key, val)

        return state

    async def delete_model(self, model_id: str) -> None:
        if (
            model_id in self._states
            and self._states[model_id].status == ModelStatus.LOADED
        ):
            await self.unload(model_id)

        async with database.AsyncSessionLocal() as session:
            await session.execute(
                sa.delete(ModelConfig).where(ModelConfig.model_id == model_id)
            )
            await session.commit()

        self._states.pop(model_id, None)
        self._load_locks.pop(model_id, None)
        self._persisted_model_ids.discard(model_id)
