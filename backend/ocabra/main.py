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
    wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(settings.log_level)),
    logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


def _build_cors_config() -> dict[str, object]:
    """Restrict browser access to same-origin production and loopback dev origins."""
    return {
        "allow_origin_regex": r"^https?://(localhost|127\.0\.0\.1)(?::\d+)?$",
        "allow_methods": ["*"],
        "allow_headers": ["*"],
    }


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


async def _ollama_inventory_loop(
    model_manager, stop_event: asyncio.Event, profile_registry=None
) -> None:
    from ocabra.registry.ollama_registry import OllamaRegistry

    interval_s = max(5, int(settings.ollama_inventory_sync_interval_seconds))
    registry = OllamaRegistry()
    logger.info("ollama_inventory_loop_started", interval_s=interval_s)
    while not stop_event.is_set():
        try:
            installed = await registry.list_installed()
            loaded = await registry.list_loaded()
            added = await model_manager.sync_ollama_inventory(installed, loaded)
            # Auto-create default profiles for newly discovered models
            if added and profile_registry:
                from ocabra.database import AsyncSessionLocal as _ASL

                async with _ASL() as _session:
                    await profile_registry.ensure_default_profiles(_session)
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
    stripped = re.sub(
        r"[-_](fp16|bf16|int8|fp8|int4|test|v\d+)(\b.*)?$", "", engine_name, flags=re.IGNORECASE
    )
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
    from ocabra.backends.chatterbox_backend import ChatterboxBackend
    from ocabra.backends.diffusers_backend import DiffusersBackend
    from ocabra.backends.llama_cpp_backend import LlamaCppBackend
    from ocabra.backends.ollama_backend import OllamaBackend
    from ocabra.backends.sglang_backend import SGLangBackend
    from ocabra.backends.tensorrt_llm_backend import TensorRTLLMBackend
    from ocabra.backends.tts_backend import TTSBackend
    from ocabra.backends.vllm_backend import VLLMBackend
    from ocabra.backends.voxtral_backend import VoxtralBackend
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
    worker_pool.register_backend("chatterbox", ChatterboxBackend())
    worker_pool.register_backend("voxtral", VoxtralBackend())
    worker_pool.register_backend("vllm", VLLMBackend())
    tensorrt_llm_backend = TensorRTLLMBackend()
    if tensorrt_llm_backend.is_enabled():
        try:
            reaped = await tensorrt_llm_backend.reconcile_orphaned_processes()
            if reaped:
                logger.warning("tensorrt_llm_startup_reconciled_orphans", groups=reaped)
        except Exception as exc:
            logger.warning("tensorrt_llm_orphan_reconcile_failed", error=str(exc))
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

    # Bloque 15 — Modular backends installer.
    # Fase 1 runs in observational mode: every currently registered backend is
    # surfaced as "built-in / installed" so the UI can list them.  Installing
    # or uninstalling from source works end-to-end, but the fat image keeps
    # pre-installing every backend for now (migration happens in Fase 2).
    from pathlib import Path

    from ocabra.core.backend_installer import BackendInstaller

    backend_installer = BackendInstaller(
        backends_dir=Path(settings.backends_dir),
        worker_pool=worker_pool,
        backend_registry=worker_pool.registered_backends(),
    )
    await backend_installer.start()
    app.state.backend_installer = backend_installer
    logger.info("backend_installer_ready")

    if settings.litellm_auto_sync:
        from ocabra.integrations.litellm_sync import LiteLLMSync

        litellm_syncer = LiteLLMSync(model_manager)
        model_manager.register_event_listener(litellm_syncer.handle_model_event)
        app.state.litellm_syncer = litellm_syncer
        logger.info("litellm_sync_enabled")

    await model_manager.start()

    from ocabra.core.backend_process_manager import BackendProcessManager

    backend_process_manager = BackendProcessManager(model_manager, worker_pool, settings)
    await backend_process_manager.start()
    app.state.backend_process_manager = backend_process_manager
    logger.info("backend_process_manager_ready")

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
    service_manager.set_gpu_manager(gpu_manager)
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
    # ollama_inventory_task created after profile_registry (below)
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

    # Profile Registry
    from ocabra.core.profile_registry import ProfileRegistry

    profile_registry = ProfileRegistry()
    from ocabra.database import AsyncSessionLocal as _ASL

    async with _ASL() as _session:
        await profile_registry.load_all(_session)
    async with _ASL() as _session:
        default_count = await profile_registry.ensure_default_profiles(_session)
        if default_count:
            logger.info("default_profiles_seeded", count=default_count)
    async with _ASL() as _session:
        diarized_count = await profile_registry.ensure_diarized_profiles(_session)
        if diarized_count:
            logger.info("diarized_profiles_seeded", count=diarized_count)
    app.state.profile_registry = profile_registry
    logger.info("profile_registry_ready")

    # Start ollama inventory loop now that profile_registry is available
    ollama_inventory_task = asyncio.create_task(
        _ollama_inventory_loop(model_manager, ollama_inventory_stop, profile_registry),
        name="ollama-inventory-loop",
    )

    # Auth: seed first admin if the users table is empty
    from ocabra.core.auth_manager import seed_first_admin

    async with _ASL() as _session:
        await seed_first_admin(_session)
    logger.info("auth_seed_done")

    # Load persisted config overrides from DB (applied on top of .env values)
    from ocabra.db.server_config import (
        apply_overrides_to_settings,
        load_overrides,
        save_override,
    )

    async with _ASL() as _session:
        _overrides = await load_overrides(_session)
    if _overrides:
        apply_overrides_to_settings(settings, _overrides)
        logger.info("config_overrides_loaded", count=len(_overrides))
    else:
        logger.info("config_overrides_none")

    # Persist JWT secret in DB so sessions survive restarts
    if "jwt_secret" not in _overrides:
        async with _ASL() as _session:
            await save_override(_session, "jwt_secret", settings.jwt_secret)
            await _session.commit()
        logger.info("jwt_secret_persisted")

    # MCP registry (agents + MCP, Fase 1)
    from ocabra.agents.mcp_registry import MCPRegistry, set_registry

    mcp_registry = MCPRegistry(session_factory=_ASL)
    await mcp_registry.start()
    app.state.mcp_registry = mcp_registry
    set_registry(mcp_registry)
    logger.info("mcp_registry_ready")

    # Federation manager (optional, only if enabled)
    if settings.federation_enabled:
        from ocabra.core.federation import FederationManager

        federation_manager = FederationManager(settings, _ASL)
        await federation_manager.start()
        app.state.federation_manager = federation_manager
        logger.info("federation_manager_ready")
    else:
        app.state.federation_manager = None

    logger.info("vllm_backend_registered")

    # OpenAI Batches API processor
    from ocabra.core.batch_processor import BatchProcessor

    batch_processor = BatchProcessor(app)
    await batch_processor.start()
    app.state.batch_processor = batch_processor
    logger.info("batch_processor_ready")

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

    if getattr(app.state, "batch_processor", None) is not None:
        await app.state.batch_processor.stop()

    if getattr(app.state, "federation_manager", None) is not None:
        await app.state.federation_manager.stop()

    if getattr(app.state, "mcp_registry", None) is not None:
        await app.state.mcp_registry.stop()
        set_registry(None)

    await backend_process_manager.stop()
    await trtllm_compile_manager.stop()
    await gpu_manager.stop()

    if settings.langfuse_enabled:
        from ocabra.integrations.langfuse_tracer import shutdown as langfuse_shutdown

        await langfuse_shutdown()

    from ocabra.redis_client import close_redis

    await close_redis()

    logger.info("ocabra_stopped")


_API_DESCRIPTION = """
# oCabra — Multi-GPU AI Model Server

Compatible con las APIs de **OpenAI** (`/v1/`) y **Ollama** (`/api/`).

## Autenticacion

Todas las peticiones requieren API key en la cabecera:

```
Authorization: Bearer sk-...
```

Las API keys se crean desde el dashboard de oCabra o via la API interna.

## Modelos

El campo `model` acepta el **profile_id** del modelo (ej: `qwen3-8b`, `kokoro-82m`).
Consulta `GET /v1/models` para ver los modelos disponibles.

## Capacidades

| Capacidad | Endpoint | Backends |
|-----------|----------|----------|
| **Chat / Completions** | `/v1/chat/completions` | vllm, sglang, llama_cpp, ollama, bitnet, tensorrt_llm |
| **Embeddings** | `/v1/embeddings` | vllm, sglang, llama_cpp, ollama |
| **Text-to-Speech** | `/v1/audio/speech` | tts (Kokoro, Bark, Qwen3-TTS), chatterbox, voxtral |
| **Speech-to-Text** | `/v1/audio/transcriptions` | whisper (faster-whisper, Whisper) |
| **Generacion de imagenes** | `/v1/images/generations` | diffusers |
| **Generacion de musica** | `/v1/audio/generate` | acestep |
| **Reranking** | `/v1/rerank` | vllm |

## Formatos de audio (TTS)

`mp3` (default), `wav`, `opus`, `flac`, `pcm`, `aac`

## Voces TTS

Dependen del modelo. Usa `GET /v1/audio/voices?model=<profile_id>` para consultar las voces disponibles.
"""

app = FastAPI(
    title="oCabra",
    description=_API_DESCRIPTION,
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


def _custom_openapi():
    """Enrich the auto-generated OpenAPI spec with typed request schemas."""
    if app.openapi_schema:
        return app.openapi_schema

    from fastapi.openapi.utils import get_openapi

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    from ocabra.api.openai._schemas import (
        ChatCompletionRequest,
        CompletionRequest,
        EmbeddingRequest,
        ImageGenerationRequest,
        MusicGenerationRequest,
        RerankRequest,
        SpeechRequest,
    )

    # Inject Pydantic schemas into components
    schemas = schema.setdefault("components", {}).setdefault("schemas", {})
    for model_cls in (
        ChatCompletionRequest,
        CompletionRequest,
        EmbeddingRequest,
        SpeechRequest,
        ImageGenerationRequest,
        RerankRequest,
        MusicGenerationRequest,
    ):
        schemas[model_cls.__name__] = model_cls.model_json_schema(
            ref_template="#/components/schemas/{model}"
        )

    # Map endpoints → schemas
    endpoint_schemas = {
        "/v1/chat/completions": "ChatCompletionRequest",
        "/v1/completions": "CompletionRequest",
        "/v1/embeddings": "EmbeddingRequest",
        "/v1/audio/speech": "SpeechRequest",
        "/v1/images/generations": "ImageGenerationRequest",
        "/v1/rerank": "RerankRequest",
        "/v1/audio/generate": "MusicGenerationRequest",
    }

    for path, schema_name in endpoint_schemas.items():
        if path in schema.get("paths", {}):
            for method_info in schema["paths"][path].values():
                if isinstance(method_info, dict) and "summary" in method_info:
                    method_info["requestBody"] = {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": f"#/components/schemas/{schema_name}"}
                            }
                        },
                    }

    # Add WebSocket endpoints (Swagger can't auto-generate these)
    paths = schema.setdefault("paths", {})
    paths["/v1/realtime"] = {
        "get": {
            "tags": ["OpenAI Realtime"],
            "summary": "Realtime audio session (WebSocket)",
            "description": (
                "**WebSocket** endpoint for bidirectional audio streaming.\n\n"
                "Establishes a session that coordinates STT → LLM → TTS in real-time.\n\n"
                "### Connection\n"
                "```\n"
                "ws://<host>/v1/realtime?model=<profile_id>\n"
                "Authorization: Bearer <api_key>\n"
                "```\n\n"
                "### Client → Server events\n"
                "| Event | Descripcion |\n"
                "|-------|-------------|\n"
                "| `session.update` | Configurar sesion (modalities, voice, instructions, "
                "turn_detection, input_audio_format, output_audio_format) |\n"
                "| `input_audio_buffer.append` | Enviar chunk de audio PCM16 (base64) |\n"
                "| `input_audio_buffer.commit` | Forzar procesamiento del buffer |\n"
                "| `input_audio_buffer.clear` | Limpiar buffer de audio |\n"
                "| `response.create` | Solicitar respuesta del LLM |\n"
                "| `response.cancel` | Cancelar respuesta en curso |\n\n"
                "### Server → Client events\n"
                "| Event | Descripcion |\n"
                "|-------|-------------|\n"
                "| `session.created` | Sesion iniciada (devuelve session config) |\n"
                "| `session.updated` | Config de sesion actualizada |\n"
                "| `input_audio_buffer.speech_started` | VAD detecto inicio de habla |\n"
                "| `input_audio_buffer.speech_stopped` | VAD detecto fin de habla |\n"
                "| `input_audio_buffer.committed` | Buffer procesado |\n"
                "| `conversation.item.created` | Nuevo item en conversacion |\n"
                "| `response.created` | Respuesta iniciada |\n"
                "| `response.audio.delta` | Chunk de audio TTS (base64 PCM16) |\n"
                "| `response.audio.done` | Audio TTS completado |\n"
                "| `response.audio_transcript.delta` | Texto parcial de la respuesta |\n"
                "| `response.audio_transcript.done` | Texto completo de la respuesta |\n"
                "| `response.done` | Respuesta completada |\n"
                "| `error` | Error en la sesion |\n\n"
                "### Audio formats\n"
                "- Input: `pcm16` (16-bit PCM, 16kHz, mono, little-endian)\n"
                "- Output: `pcm16` (configurable via session.update)\n"
            ),
            "parameters": [
                {
                    "name": "model",
                    "in": "query",
                    "required": True,
                    "description": "Profile ID del modelo LLM para la sesion",
                    "schema": {"type": "string", "example": "qwen3-8b"},
                }
            ],
            "responses": {
                "101": {"description": "WebSocket upgrade exitoso"},
                "1008": {"description": "Autenticacion requerida (WebSocket close)"},
            },
        }
    }

    app.openapi_schema = schema
    return schema


app.openapi = _custom_openapi

app.add_middleware(
    CORSMiddleware,
    **_build_cors_config(),
)

# Stream 3-A: Stats middleware for /v1/* endpoints
from ocabra.stats.collector import StatsMiddleware  # noqa: E402

app.add_middleware(StatsMiddleware)

# ── ROUTERS ──────────────────────────────────────────────────
# Each stream adds its router here. Do not remove this comment.
from ocabra.api.health import router as health_router  # noqa: E402

app.include_router(health_router)

# ── Internal routers (hidden from /docs — admin dashboard only) ──
from ocabra.api.internal.gpus import router as gpus_router
from ocabra.api.internal.models import router as models_router
from ocabra.api.internal.ws import router as ws_router
from ocabra.api.internal.downloads import router as downloads_router  # noqa: E402
from ocabra.api.internal.registry import router as registry_router  # noqa: E402
from ocabra.api.internal.services import router as services_router  # noqa: E402
from ocabra.api.internal.host import router as host_router  # noqa: E402

app.include_router(gpus_router, prefix="/ocabra", include_in_schema=False)
app.include_router(models_router, prefix="/ocabra", include_in_schema=False)
app.include_router(ws_router, prefix="/ocabra", include_in_schema=False)
app.include_router(registry_router, prefix="/ocabra", include_in_schema=False)
app.include_router(downloads_router, prefix="/ocabra", include_in_schema=False)
app.include_router(services_router, prefix="/ocabra", include_in_schema=False)
app.include_router(host_router, prefix="/ocabra", include_in_schema=False)

# Bloque 15 — Modular backends router
from ocabra.api.internal.backends import router as backends_router  # noqa: E402

app.include_router(backends_router, prefix="/ocabra", include_in_schema=False)
from ocabra.api.internal.federation import router as federation_router  # noqa: E402

app.include_router(federation_router, prefix="/ocabra", include_in_schema=False)

# Stream 3-A: OpenAI API
from ocabra.api.openai import router as openai_router  # noqa: E402

app.include_router(openai_router, prefix="/v1")

# Stream 3-B: Ollama API
from ocabra.api.ollama import router as ollama_router  # noqa: E402

app.include_router(ollama_router)

# Stream 5: Metrics (public), Config/Stats/TRT/Auth/Users/Groups/Profiles (internal)
from ocabra.api.metrics import router as metrics_router  # noqa: E402
from ocabra.api.internal.config import router as config_router  # noqa: E402
from ocabra.api.internal.stats import router as stats_router  # noqa: E402
from ocabra.api.internal.trtllm import router as trtllm_router  # noqa: E402
from ocabra.api.internal.auth import router as auth_router  # noqa: E402
from ocabra.api.internal.users import router as users_router  # noqa: E402
from ocabra.api.internal.groups import router as groups_router  # noqa: E402
from ocabra.api.internal.profiles import router as profiles_router  # noqa: E402

app.include_router(metrics_router)
app.include_router(config_router, prefix="/ocabra", include_in_schema=False)
app.include_router(stats_router, prefix="/ocabra", include_in_schema=False)
app.include_router(trtllm_router, prefix="/ocabra", include_in_schema=False)
app.include_router(auth_router, prefix="/ocabra", include_in_schema=False)
app.include_router(users_router, prefix="/ocabra", include_in_schema=False)
app.include_router(groups_router, prefix="/ocabra", include_in_schema=False)

app.include_router(profiles_router, prefix="/ocabra", include_in_schema=False)

# Agents + MCP (plan: docs/tasks/agents-mcp-plan.md)
from ocabra.api.internal.agents import router as agents_router  # noqa: E402
from ocabra.api.internal.mcp_servers import router as mcp_servers_router  # noqa: E402

app.include_router(agents_router, prefix="/ocabra", include_in_schema=False)
app.include_router(mcp_servers_router, prefix="/ocabra", include_in_schema=False)
# ─────────────────────────────────────────────────────────────
