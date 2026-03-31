import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from contextlib import suppress

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ocabra.config import settings

# ── Structlog configuration ───────────────────────────────────
_is_prod = os.getenv("OCABRA_ENV", "development").lower() == "production"

_processors = [
    structlog.contextvars.merge_contextvars,
    structlog.stdlib.add_log_level,
    structlog.processors.TimeStamper(fmt="iso"),
    structlog.processors.StackInfoRenderer(),
]

if _is_prod:
    _processors.append(structlog.processors.JSONRenderer())
else:
    _processors.append(structlog.dev.ConsoleRenderer(colors=True))

structlog.configure(
    processors=_processors,
    wrapper_class=structlog.make_filtering_bound_logger(
        logging.getLevelName(settings.log_level)
    ),
    logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


async def _idle_eviction_loop(model_manager, stop_event: asyncio.Event) -> None:
    interval_s = max(1, int(settings.idle_eviction_check_interval_seconds))
    logger.info(
        "idle_eviction_loop_started",
        interval_s=interval_s,
        idle_timeout_seconds=settings.idle_timeout_seconds,
    )
    while not stop_event.is_set():
        try:
            await model_manager.check_idle_evictions()
        except Exception as exc:
            logger.warning("idle_eviction_loop_error", error=str(exc))

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_s)
        except TimeoutError:
            continue
    logger.info("idle_eviction_loop_stopped")


async def _schedule_maintenance_loop(gpu_scheduler, stop_event: asyncio.Event) -> None:
    interval_s = 60
    logger.info("schedule_maintenance_loop_started", interval_s=interval_s)
    while not stop_event.is_set():
        try:
            await gpu_scheduler.check_schedule_evictions()
            await gpu_scheduler.check_schedule_reloads()
        except Exception as exc:
            logger.warning("schedule_maintenance_loop_error", error=str(exc))

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_s)
        except TimeoutError:
            continue
    logger.info("schedule_maintenance_loop_stopped")


async def _ollama_inventory_loop(model_manager, stop_event: asyncio.Event) -> None:
    from ocabra.registry.ollama_registry import OllamaRegistry

    interval_s = max(5, int(settings.ollama_inventory_sync_interval_seconds))
    registry = OllamaRegistry()
    logger.info("ollama_inventory_loop_started", interval_s=interval_s)
    while not stop_event.is_set():
        try:
            installed = await registry.list_installed()
            loaded = await registry.list_loaded()
            await model_manager.sync_ollama_inventory(installed, loaded)
        except Exception as exc:
            logger.warning("ollama_inventory_loop_error", error=str(exc))

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_s)
        except TimeoutError:
            continue
    logger.info("ollama_inventory_loop_stopped")


async def _service_idle_unload_loop(service_manager, stop_event: asyncio.Event) -> None:
    interval_s = 15
    logger.info("service_idle_unload_loop_started", interval_s=interval_s)
    while not stop_event.is_set():
        try:
            await service_manager.check_idle_unloads()
        except Exception as exc:
            logger.warning("service_idle_unload_loop_error", error=str(exc))

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_s)
        except TimeoutError:
            continue
    logger.info("service_idle_unload_loop_stopped")


async def _service_health_loop(service_manager, stop_event: asyncio.Event) -> None:
    interval_s = 30
    logger.info("service_health_loop_started", interval_s=interval_s)
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_s)
        except TimeoutError:
            pass
        if stop_event.is_set():
            break
        try:
            await service_manager.refresh_all()
        except Exception as exc:
            logger.warning("service_health_loop_error", error=str(exc))
    logger.info("service_health_loop_stopped")


def _find_tokenizer_for_engine(engine_name: str, models_root) -> str | None:
    """Try to find a HuggingFace model directory to use as tokenizer for the given engine.

    Strategy: strip known dtype/variant suffixes from engine_name and look for a
    matching HuggingFace directory that has tokenizer files.
    """
    import re
    from pathlib import Path

    hf_root = Path(models_root) / "huggingface"
    if not hf_root.exists():
        return None

    def has_tokenizer(p: Path) -> bool:
        return (p / "tokenizer.json").exists() or (p / "tokenizer_config.json").exists()

    # Exact match
    exact = hf_root / engine_name
    if exact.exists() and has_tokenizer(exact):
        return str(exact)

    # Strip common suffixes: -fp16, -bf16, -int8, -fp8, -test, -v2, -v3 etc.
    stripped = re.sub(r"[-_](fp16|bf16|int8|fp8|int4|test|v\d+)(\b.*)?$", "", engine_name, flags=re.IGNORECASE)
    if stripped != engine_name:
        candidate = hf_root / stripped
        if candidate.exists() and has_tokenizer(candidate):
            return str(candidate)

    # Find the longest-matching HF dir name that is a prefix of the engine name
    best: Path | None = None
    for hf_dir in hf_root.iterdir():
        if not hf_dir.is_dir():
            continue
        if engine_name.startswith(hf_dir.name) and has_tokenizer(hf_dir):
            if best is None or len(hf_dir.name) > len(best.name):
                best = hf_dir
    return str(best) if best else None


async def _scan_and_register_trtllm_engines(model_manager) -> None:
    """Discover compiled TRT-LLM engines on disk and register any not yet in the model manager."""
    from pathlib import Path
    from ocabra.config import settings

    # Use the container-side path (TENSORRT_LLM_ENGINES_DIR or models_dir/tensorrt_llm)
    if settings.tensorrt_llm_engines_dir:
        engines_root = Path(settings.tensorrt_llm_engines_dir)
    elif settings.models_dir:
        engines_root = Path(settings.models_dir) / "tensorrt_llm"
    else:
        return
    if not engines_root.exists():
        return

    registered = 0
    for engine_dir in sorted(engines_root.iterdir()):
        if not engine_dir.is_dir():
            continue
        engine_subdir = engine_dir / "engine"
        if not engine_subdir.exists():
            continue
        if not (engine_subdir / "config.json").exists():
            continue
        has_engine = any(engine_subdir.glob("*.engine"))
        if not has_engine:
            continue

        engine_name = engine_dir.name
        model_id = f"tensorrt_llm/{engine_name}"
        if model_id in model_manager._states:
            continue

        try:
            import json
            cfg = json.loads((engine_subdir / "config.json").read_text())
            build_cfg = cfg.get("build_config", {})
            tokenizer_path = _find_tokenizer_for_engine(engine_name, engines_root.parent)
            extra_config: dict = {
                "engine_dir": str(engine_subdir),
                "launch_mode": "docker",
                "max_batch_size": build_cfg.get("max_batch_size", 1),
                "context_length": build_cfg.get("max_seq_len", 4096),
            }
            if tokenizer_path:
                extra_config["tokenizer_path"] = tokenizer_path
            await model_manager.add_model(
                model_id=model_id,
                backend_type="tensorrt_llm",
                display_name=engine_name,
                extra_config=extra_config,
            )
            registered += 1
            logger.info("trtllm_engine_scanned_registered", model_id=model_id)
        except Exception as exc:
            logger.warning("trtllm_engine_scan_failed", engine_name=engine_name, error=str(exc))

    if registered:
        logger.info("trtllm_engines_registered_on_startup", count=registered)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────
    logger.info("starting_ocabra", version=settings.app_version)

    from ocabra.redis_client import init_redis
    await init_redis()
    logger.info("redis_connected")

    # Stream 1-A: GPU Manager + Scheduler
    from ocabra.core.gpu_manager import GPUManager
    from ocabra.core.scheduler import GPUScheduler
    gpu_manager = GPUManager()
    await gpu_manager.start()
    gpu_scheduler = GPUScheduler(gpu_manager)
    app.state.gpu_manager = gpu_manager
    app.state.gpu_scheduler = gpu_scheduler
    logger.info("gpu_manager_ready")

    # Stream 1-B: Worker Pool + Model Manager
    from ocabra.core.worker_pool import WorkerPool
    from ocabra.core.model_manager import ModelManager
    from ocabra.backends.acestep_backend import AceStepBackend
    from ocabra.backends.bitnet_backend import BitnetBackend
    from ocabra.backends.diffusers_backend import DiffusersBackend
    from ocabra.backends.llama_cpp_backend import LlamaCppBackend
    from ocabra.backends.ollama_backend import OllamaBackend
    from ocabra.backends.sglang_backend import SGLangBackend
    from ocabra.backends.tensorrt_llm_backend import TensorRTLLMBackend
    from ocabra.backends.tts_backend import TTSBackend
    from ocabra.backends.vllm_backend import VLLMBackend
    from ocabra.backends.whisper_backend import WhisperBackend
    worker_pool = WorkerPool()
    worker_pool.register_backend("acestep", AceStepBackend())
    worker_pool.register_backend("diffusers", DiffusersBackend())
    worker_pool.register_backend("bitnet", BitnetBackend())
    worker_pool.register_backend("llama_cpp", LlamaCppBackend())
    worker_pool.register_backend("ollama", OllamaBackend())
    worker_pool.register_backend("sglang", SGLangBackend())
    worker_pool.register_backend("whisper", WhisperBackend())
    worker_pool.register_backend("tts", TTSBackend())
    worker_pool.register_backend("vllm", VLLMBackend())
    tensorrt_llm_backend = TensorRTLLMBackend()
    if tensorrt_llm_backend.is_enabled():
        worker_pool.register_backend("tensorrt_llm", tensorrt_llm_backend)
    else:
        worker_pool.register_disabled_backend(
            "tensorrt_llm",
            tensorrt_llm_backend.disabled_reason or "feature flag disabled",
        )
    model_manager = ModelManager(worker_pool, gpu_manager, gpu_scheduler)
    gpu_scheduler.set_model_manager(model_manager)
    app.state.worker_pool = worker_pool
    app.state.model_manager = model_manager

    if settings.litellm_auto_sync:
        from ocabra.integrations.litellm_sync import LiteLLMSync

        litellm_syncer = LiteLLMSync(model_manager)
        model_manager.register_event_listener(litellm_syncer.handle_model_event)
        app.state.litellm_syncer = litellm_syncer
        logger.info("litellm_sync_enabled")

    await model_manager.start()
    from ocabra.registry.ollama_registry import OllamaRegistry
    try:
        installed_ollama = await OllamaRegistry().list_installed()
        loaded_ollama = await OllamaRegistry().list_loaded()
    except Exception:
        installed_ollama = []
        loaded_ollama = []
    if installed_ollama or loaded_ollama:
        await model_manager.sync_ollama_inventory(installed_ollama, loaded_ollama)
    logger.info("model_manager_ready")

    from ocabra.core.service_manager import ServiceManager
    service_manager = ServiceManager()
    await service_manager.start()
    app.state.service_manager = service_manager
    model_manager.set_service_manager(service_manager)
    logger.info("service_manager_ready")

    from ocabra.core.trtllm_compile_manager import TrtllmCompileManager
    trtllm_compile_manager = TrtllmCompileManager(model_manager=model_manager)
    await trtllm_compile_manager.start()
    app.state.trtllm_compile_manager = trtllm_compile_manager
    await _scan_and_register_trtllm_engines(model_manager)
    logger.info("trtllm_compile_manager_ready")

    idle_eviction_stop = asyncio.Event()
    idle_eviction_task = asyncio.create_task(
        _idle_eviction_loop(model_manager, idle_eviction_stop),
        name="idle-eviction-loop",
    )
    schedule_maintenance_stop = asyncio.Event()
    schedule_maintenance_task = asyncio.create_task(
        _schedule_maintenance_loop(gpu_scheduler, schedule_maintenance_stop),
        name="schedule-maintenance-loop",
    )
    ollama_inventory_stop = asyncio.Event()
    ollama_inventory_task = asyncio.create_task(
        _ollama_inventory_loop(model_manager, ollama_inventory_stop),
        name="ollama-inventory-loop",
    )
    service_idle_stop = asyncio.Event()
    service_idle_task = asyncio.create_task(
        _service_idle_unload_loop(service_manager, service_idle_stop),
        name="service-idle-unload-loop",
    )
    service_health_stop = asyncio.Event()
    service_health_task = asyncio.create_task(
        _service_health_loop(service_manager, service_health_stop),
        name="service-health-loop",
    )

    logger.info("vllm_backend_registered")
    logger.info("ocabra_ready")
    yield

    # ── Shutdown ─────────────────────────────────────────────
    logger.info("shutting_down_ocabra")

    idle_eviction_stop.set()
    idle_eviction_task.cancel()
    with suppress(asyncio.CancelledError):
        await idle_eviction_task

    schedule_maintenance_stop.set()
    schedule_maintenance_task.cancel()
    with suppress(asyncio.CancelledError):
        await schedule_maintenance_task

    ollama_inventory_stop.set()
    ollama_inventory_task.cancel()
    with suppress(asyncio.CancelledError):
        await ollama_inventory_task

    service_idle_stop.set()
    service_idle_task.cancel()
    with suppress(asyncio.CancelledError):
        await service_idle_task

    service_health_stop.set()
    service_health_task.cancel()
    with suppress(asyncio.CancelledError):
        await service_health_task

    await trtllm_compile_manager.stop()
    await gpu_manager.stop()

    from ocabra.redis_client import close_redis
    await close_redis()

    logger.info("ocabra_stopped")


app = FastAPI(
    title="oCabra",
    description="Multi-GPU AI model server — OpenAI & Ollama compatible",
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Stream 3-A: Stats middleware for /v1/* endpoints
from ocabra.stats.collector import StatsMiddleware  # noqa: E402
app.add_middleware(StatsMiddleware)

# ── ROUTERS ──────────────────────────────────────────────────
# Each stream adds its router here. Do not remove this comment.
from ocabra.api.health import router as health_router  # noqa: E402

app.include_router(health_router)

# Stream 1-A: GPU Manager
from ocabra.api.internal.gpus import router as gpus_router
app.include_router(gpus_router, prefix="/ocabra")

# Stream 1-B: Model Manager + WebSocket
from ocabra.api.internal.models import router as models_router
from ocabra.api.internal.ws import router as ws_router
app.include_router(models_router, prefix="/ocabra")
app.include_router(ws_router, prefix="/ocabra")

# Stream 1-C: Registry + Downloads
from ocabra.api.internal.downloads import router as downloads_router  # noqa: E402
from ocabra.api.internal.registry import router as registry_router  # noqa: E402
from ocabra.api.internal.services import router as services_router  # noqa: E402

app.include_router(registry_router, prefix="/ocabra")
app.include_router(downloads_router, prefix="/ocabra")
app.include_router(services_router, prefix="/ocabra")

# Stream 3-A: OpenAI API
from ocabra.api.openai import router as openai_router  # noqa: E402
app.include_router(openai_router, prefix="/v1")

# Stream 3-B: Ollama API
from ocabra.api.ollama import router as ollama_router  # noqa: E402
app.include_router(ollama_router)

# Stream 5: Metrics, Config, Stats
from ocabra.api.metrics import router as metrics_router  # noqa: E402
from ocabra.api.internal.config import router as config_router  # noqa: E402
from ocabra.api.internal.stats import router as stats_router  # noqa: E402
app.include_router(metrics_router)
app.include_router(config_router, prefix="/ocabra")
app.include_router(stats_router, prefix="/ocabra")

# C-7: TensorRT-LLM compile
from ocabra.api.internal.trtllm import router as trtllm_router  # noqa: E402
app.include_router(trtllm_router, prefix="/ocabra")
# ─────────────────────────────────────────────────────────────
