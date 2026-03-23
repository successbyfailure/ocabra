import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from urllib.parse import urlparse

import structlog

from ocabra.backends.base import BackendCapabilities, WorkerInfo
from ocabra.config import settings
from ocabra.core.model_ref import build_model_ref, normalize_model_ref, parse_model_ref
from ocabra.redis_client import publish, set_key

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



def _is_diarized_model_id(model_id: str, extra_config: dict | None = None) -> bool:
    lowered = model_id.lower()
    if "diariz" in lowered:
        return True
    cfg = extra_config or {}
    return bool(cfg.get("diarization_enabled") is True)



def _diarized_variant_model_id(model_id: str) -> str:
    if model_id.endswith("::diarize"):
        return model_id
    return f"{model_id}::diarize"



def _should_auto_create_diarized_variant(state: "ModelState") -> bool:
    if state.backend_type != "whisper":
        return False
    if "::" in state.backend_model_id:
        return False
    if _is_diarized_model_id(state.backend_model_id, state.extra_config):
        return False
    return True



def _build_diarized_extra_config(base_extra_config: dict | None) -> dict:
    merged = dict(base_extra_config or {})
    merged["diarization_enabled"] = True
    return merged


class ModelManager:
    def __init__(self, worker_pool, gpu_manager=None, gpu_scheduler=None) -> None:
        self._worker_pool = worker_pool
        self._gpu_manager = gpu_manager
        self._gpu_scheduler = gpu_scheduler
        self._states: dict[str, ModelState] = {}
        self._load_locks: dict[str, asyncio.Lock] = {}

    async def _ensure_diarized_variants_for_whisper_models(self) -> None:
        snapshot = list(self._states.values())
        for state in snapshot:
            if not _should_auto_create_diarized_variant(state):
                continue
            diarized_backend_model_id = _diarized_variant_model_id(state.backend_model_id)
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
                    extra_config=_build_diarized_extra_config(state.extra_config),
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

    async def start(self) -> None:
        await self._load_configs_from_db()
        await self._ensure_diarized_variants_for_whisper_models()
        for model_id, state in self._states.items():
            if state.load_policy == LoadPolicy.PIN:
                logger.info("auto_loading_pinned_model", model_id=model_id)
                asyncio.create_task(self._load_model(model_id))

    async def _load_configs_from_db(self) -> None:
        import sqlalchemy as sa

        from ocabra.database import AsyncSessionLocal
        from ocabra.db.model_config import ModelConfig

        async with AsyncSessionLocal() as session:
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
            await set_key(f"model:state:{model_id}", state.to_dict())

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
                        model_id=normalized_model_id,
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

                state.worker_info = worker_info
                state.current_gpu = gpu_indices
                state.vram_used_mb = worker_info.vram_used_mb
                state.capabilities = capabilities
                state.status = ModelStatus.LOADED
                state.loaded_at = datetime.now(timezone.utc)
                state.error_message = None

                if gpu_managed and self._gpu_manager:
                    for gpu_idx in gpu_indices:
                        await self._gpu_manager.lock_vram(
                            gpu_idx, worker_info.vram_used_mb, model_id
                        )

                self._worker_pool.set_worker(model_id, worker_info)
                await self._publish_event(model_id, "status_changed")
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

    def _resolve_bitnet_option(self, state: "ModelState", key: str, default: int) -> int:
        extra = state.extra_config if isinstance(state.extra_config, dict) else {}
        nested = extra.get("bitnet") if isinstance(extra.get("bitnet"), dict) else None
        if nested and key in nested:
            return int(nested[key])
        if key in extra:
            return int(extra[key])
        return int(default)

    def _resolve_bitnet_gpu_layers(self, state: "ModelState") -> int:
        return self._resolve_bitnet_option(state, "gpu_layers", settings.bitnet_gpu_layers)

    def _estimate_bitnet_vram_from_config(self, state: "ModelState") -> int:
        gpu_layers = self._resolve_bitnet_gpu_layers(state)
        if gpu_layers <= 0:
            return 0
        total_layers = max(1, self._resolve_bitnet_option(state, "total_layers", 32))
        model_vram_mb = max(1, self._resolve_bitnet_option(state, "model_vram_mb", 400))
        return int(model_vram_mb * min(gpu_layers, total_layers) / total_layers)

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
            candidates = self._get_pressure_eviction_candidates(model_id)
            logger.info(
                "pressure_eviction_candidates",
                requested_model_id=model_id,
                candidates=candidates,
            )
            for candidate_id in candidates:
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
            logger.info("model_unloaded", model_id=model_id, reason=reason)

            if reason == "pressure" and state.auto_reload:
                asyncio.create_task(self._watch_and_reload(model_id))
        except Exception as e:
            state.status = ModelStatus.ERROR
            state.error_message = str(e)
            await self._publish_event(model_id, "unload_failed")
            logger.error("model_unload_failed", model_id=model_id, error=str(e))
            raise

    async def _watch_and_reload(self, model_id: str) -> None:
        state = self._states.get(model_id)
        if not state or not self._gpu_scheduler:
            return
        while True:
            await asyncio.sleep(30)
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

    async def on_request(self, model_id: str) -> None:
        state = self._states.get(model_id)
        if not state:
            raise KeyError(f"Model '{model_id}' not configured")

        state.last_request_at = datetime.now(timezone.utc)

        if state.status in (ModelStatus.UNLOADED, ModelStatus.CONFIGURED):
            await self._load_model(model_id)
        elif state.status == ModelStatus.LOADING:
            for _ in range(120):
                await asyncio.sleep(1)
                if state.status == ModelStatus.LOADED:
                    return
            raise TimeoutError(f"Model '{model_id}' did not load within 120s")
        elif state.status == ModelStatus.ERROR:
            raise RuntimeError(f"Model '{model_id}' is in error state: {state.error_message}")

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
                asyncio.create_task(self.unload(model_id, reason="idle"))

    async def get_state(self, model_id: str) -> "ModelState | None":
        return self._states.get(model_id)

    async def list_states(self) -> list["ModelState"]:
        return list(self._states.values())

    async def sync_ollama_models(self, model_ids: list[str]) -> int:
        """Ensure native Ollama models are present in the internal inventory."""
        added = 0
        for backend_model_id in model_ids:
            normalized = backend_model_id.strip()
            if not normalized:
                continue
            model_id = build_model_ref("ollama", normalized)
            if model_id in self._states:
                continue
            await self.add_model(
                model_id=model_id,
                backend_type="ollama",
                display_name=normalized,
                load_policy="on_demand",
            )
            added += 1
        return added

    async def sync_ollama_inventory(
        self,
        installed_model_ids: list[str],
        loaded_model_ids: list[str] | None = None,
    ) -> int:
        added = await self.sync_ollama_models(installed_model_ids)
        installed_set = {model_id.strip() for model_id in installed_model_ids if model_id.strip()}
        loaded_set = {model_id.strip() for model_id in (loaded_model_ids or []) if model_id.strip()}
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
                if state.status != ModelStatus.LOADED:
                    state.status = ModelStatus.LOADED
                    state.loaded_at = state.loaded_at or datetime.now(timezone.utc)
                    state.error_message = None
                    state.current_gpu = []
                    state.vram_used_mb = 0
                    state.worker_info = WorkerInfo(
                        backend_type="ollama",
                        model_id=backend_model_id,
                        gpu_indices=[],
                        port=ollama_port,
                        pid=0,
                        vram_used_mb=0,
                    )
                    self._worker_pool.set_worker(model_id, state.worker_info)
                    changed.append(model_id)
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
        from ocabra.database import AsyncSessionLocal
        from ocabra.db.model_config import ModelConfig

        normalized_model_id, backend_model_id = normalize_model_ref(backend_type, model_id)
        normalized_backend = str(backend_type or "").strip().lower()

        if normalized_model_id in self._states:
            return self._states[normalized_model_id]

        async with AsyncSessionLocal() as session:
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

        if create_diarized_variant and _should_auto_create_diarized_variant(state):
            diarized_backend_model_id = _diarized_variant_model_id(backend_model_id)
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
                        extra_config=_build_diarized_extra_config(extra_config),
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
        import sqlalchemy as sa

        from ocabra.database import AsyncSessionLocal
        from ocabra.db.model_config import ModelConfig

        state = self._states.get(model_id)
        if not state:
            raise KeyError(f"Model '{model_id}' not found")

        async with AsyncSessionLocal() as session:
            result = await session.execute(
                sa.select(ModelConfig).where(ModelConfig.model_id == model_id)
            )
            cfg = result.scalar_one_or_none()
            if cfg:
                for key, val in patch.items():
                    if hasattr(cfg, key):
                        setattr(cfg, key, val)
                await session.commit()

        for key, val in patch.items():
            if hasattr(state, key):
                if key == "load_policy":
                    val = LoadPolicy(val)
                setattr(state, key, val)

        return state

    async def delete_model(self, model_id: str) -> None:
        import sqlalchemy as sa

        if (
            model_id in self._states
            and self._states[model_id].status == ModelStatus.LOADED
        ):
            await self.unload(model_id)

        from ocabra.database import AsyncSessionLocal
        from ocabra.db.model_config import ModelConfig

        async with AsyncSessionLocal() as session:
            await session.execute(
                sa.delete(ModelConfig).where(ModelConfig.model_id == model_id)
            )
            await session.commit()

        self._states.pop(model_id, None)
        self._load_locks.pop(model_id, None)
