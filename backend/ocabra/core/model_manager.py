import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

import structlog

from ocabra.backends.base import BackendCapabilities, WorkerInfo
from ocabra.config import settings
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
    model_id: str
    display_name: str
    backend_type: str
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

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
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
        }


class ModelManager:
    def __init__(self, worker_pool, gpu_manager=None, gpu_scheduler=None) -> None:
        self._worker_pool = worker_pool
        self._gpu_manager = gpu_manager
        self._gpu_scheduler = gpu_scheduler
        self._states: dict[str, ModelState] = {}
        self._load_locks: dict[str, asyncio.Lock] = {}

    def set_gpu_manager(self, gpu_manager) -> None:
        self._gpu_manager = gpu_manager

    def set_gpu_scheduler(self, scheduler) -> None:
        self._gpu_scheduler = scheduler

    async def start(self) -> None:
        await self._load_configs_from_db()
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
            configs = result.scalars().all()

        for cfg in configs:
            self._states[cfg.model_id] = ModelState(
                model_id=cfg.model_id,
                display_name=cfg.display_name or cfg.model_id,
                backend_type=cfg.backend_type,
                load_policy=LoadPolicy(cfg.load_policy),
                auto_reload=cfg.auto_reload,
                preferred_gpu=cfg.preferred_gpu,
            )
            self._load_locks[cfg.model_id] = asyncio.Lock()

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
                vram_needed = await backend.get_vram_estimate_mb(model_id)
                gpu_managed = state.backend_type != "ollama"
                assigned_port = await self._worker_pool.assign_port() if gpu_managed else 0
                backend_loaded = False

                if not gpu_managed:
                    gpu_indices = []
                elif self._gpu_scheduler:
                    gpu_indices = await self._gpu_scheduler.find_gpu_for_model(
                        vram_needed,
                        force_gpu or state.preferred_gpu,
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
                    model_id,
                    gpu_indices,
                    port=assigned_port,
                )
                backend_loaded = True
                capabilities = await backend.get_capabilities(model_id)

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
                        await backend.unload(model_id)
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
            await backend.unload(model_id)

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
                vram_needed = await backend.get_vram_estimate_mb(model_id)
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

    async def add_model(
        self,
        model_id: str,
        backend_type: str,
        display_name: str | None = None,
        load_policy: str = "on_demand",
        auto_reload: bool = False,
        preferred_gpu: int | None = None,
        extra_config: dict | None = None,
    ) -> "ModelState":
        from ocabra.database import AsyncSessionLocal
        from ocabra.db.model_config import ModelConfig

        async with AsyncSessionLocal() as session:
            cfg = ModelConfig(
                model_id=model_id,
                display_name=display_name or model_id,
                backend_type=backend_type,
                load_policy=load_policy,
                auto_reload=auto_reload,
                preferred_gpu=preferred_gpu,
                extra_config=extra_config,
            )
            session.add(cfg)
            await session.commit()

        state = ModelState(
            model_id=model_id,
            display_name=display_name or model_id,
            backend_type=backend_type,
            load_policy=LoadPolicy(load_policy),
            auto_reload=auto_reload,
            preferred_gpu=preferred_gpu,
        )
        self._states[model_id] = state
        self._load_locks[model_id] = asyncio.Lock()
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
