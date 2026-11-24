import asyncio
import logging
import os
import sys
import time
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack, asynccontextmanager, suppress

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

        # Same cadence: both are "the placement we chose earlier may no longer
        # be the right one" checks, and this one is a no-op unless a model is
        # sitting off its preferred GPU with that GPU now free.
        try:
            await model_manager.check_preferred_gpu_replacements()
        except Exception as exc:
            logger.warning("preferred_gpu_replacement_loop_error", error=str(exc))

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


_spill_reload_last: dict[str, float] = {}


async def _reload_spilled_ollama(model_manager, registry) -> None:
    """Reload WARM/PIN Ollama models that spilled to CPU once VRAM frees up.

    Ollama places a model at load time and never re-balances it: if a big model
    was resident when Ollama loaded it, part of the model stays on CPU forever
    (slow) even after the GPU pressure that caused the spill is gone. This
    monitor detects that (``size_vram < size`` on ``/api/ps``) and, when enough
    VRAM has since freed to hold the model fully, stops + reloads it so Ollama
    re-places it 100% on GPU. Scoped to resident-policy (warm/pinned) models
    that aren't serving a request, with a per-model cooldown.
    """
    if not settings.ollama_reload_spilled:
        return
    gm = getattr(model_manager, "_gpu_manager", None)
    if gm is None:
        return
    from ocabra.core.model_manager import LoadPolicy

    try:
        details = await registry.list_loaded_details()
        gpu_states = await gm.get_all_states()
    except Exception as exc:
        logger.warning("ollama_spill_reload_probe_error", error=str(exc))
        return

    free_total_mb = sum(max(0, int(getattr(s, "free_vram_mb", 0))) for s in gpu_states)
    margin = max(0, int(settings.ollama_reload_spilled_margin_mb))
    cooldown = max(30, int(settings.ollama_reload_spilled_cooldown_seconds))
    now = time.time()

    for d in details:
        size_total = int(d.get("size_total") or 0)
        size_vram = int(d.get("size_vram") or 0)
        if size_total <= 0:
            continue
        spilled_mb = max(0, size_total - size_vram) // (1024 * 1024)
        if spilled_mb < 256:  # effectively fully resident on GPU already
            continue
        name = d.get("name") or ""
        state = model_manager.get_ollama_state_by_name(name)
        if state is None or state.load_policy not in (LoadPolicy.WARM, LoadPolicy.PIN):
            continue
        if model_manager.is_busy(state.model_id):
            continue
        if now - _spill_reload_last.get(name, 0.0) < cooldown:
            continue
        # Unloading frees the model's current GPU slice too, so the reload has
        # ``free_total + size_vram`` to work with — require the spilled slice to
        # fit in the currently-free VRAM (plus a safety margin).
        if free_total_mb < spilled_mb + margin:
            continue
        logger.info(
            "ollama_spill_reload_start",
            model=name,
            spilled_mb=spilled_mb,
            free_total_mb=free_total_mb,
        )
        _spill_reload_last[name] = now
        try:
            await registry.unload(name)
            await asyncio.sleep(2)
            await registry.load(name, keep_alive=-1)
            logger.info("ollama_spill_reload_done", model=name)
        except Exception as exc:
            logger.warning("ollama_spill_reload_failed", model=name, error=str(exc))
        # One reload per pass: the VRAM figures above are now stale.
        return


async def _ollama_inventory_loop(
    model_manager, stop_event: asyncio.Event, profile_registry=None
) -> None:
    from ocabra.registry.ollama_registry import OllamaRegistry

    if not settings.ollama_enabled:
        logger.info("ollama_inventory_loop_disabled")
        return
    interval_s = max(5, int(settings.ollama_inventory_sync_interval_seconds))
    registry = OllamaRegistry()
    logger.info("ollama_inventory_loop_started", interval_s=interval_s)
    while not stop_event.is_set():
        try:
            # Installed catalog: all configured Ollama pools share the same
            # model storage on this host, so the default pool's view is
            # representative — only the LOADED/resident side genuinely
            # differs per pool.
            installed = await registry.list_installed()
            # 2026-10-30 — residency must be discovered across EVERY
            # configured pool (discover_ollama_resident_models), not just the
            # default OLLAMA_BASE_URL this loop used to poll alone. A model
            # resident on a non-default pool (e.g. ollama-3060) got adopted
            # correctly once at startup by _reconcile_ollama_state (which IS
            # pool-aware) — then this loop, seeing it absent from its own
            # single-pool /api/ps, demoted it back to UNLOADED ~60s later
            # (ollama_inventory_loaded_grace_s) every single cycle, forever.
            # Reproduced live: qwen3-embedding:8b sat warm and serving on
            # ollama-3060 the whole time while /ocabra/models reported it
            # unloaded with 0 VRAM.
            resident = await model_manager.discover_ollama_resident_models()
            loaded = [item["name"] for item in resident]
            loaded_vram_mb = {
                item["name"]: max(0, int(item.get("vram_mb", 0))) for item in resident
            }
            # 2026-10-30 — which POOL a resident model is actually on, not
            # just that it's loaded somewhere. Without this, ModelManager
            # .unload() falls back to the default pool's endpoint (it only
            # trusts state.worker_info.endpoint_url, which this loop used to
            # leave unset) — an unload request for a model resident on e.g.
            # ollama-3060 silently landed on the wrong daemon and did
            # nothing, while Ollama kept the real instance alive under its
            # own keep_alive. Reproduced live: unloading qwen3-embedding:8b
            # returned "unloaded" from oCabra but `ollama ps` on the 3060
            # pool still showed it resident with UNTIL=Forever.
            loaded_pool_url = {
                item["name"]: str(item.get("pool_url") or "") for item in resident
            }
            loaded_pool_gpus = {
                item["name"]: [int(index) for index in item.get("pool_gpus", [])]
                for item in resident
            }
            added = await model_manager.sync_ollama_inventory(
                installed,
                loaded,
                loaded_vram_mb,
                loaded_pool_url,
                loaded_pool_gpus,
            )
            # Auto-create default profiles for newly discovered models
            if added and profile_registry:
                from ocabra.database import AsyncSessionLocal as _ASL

                async with _ASL() as _session:
                    await profile_registry.ensure_default_profiles(_session)
        except Exception as exc:
            logger.warning("ollama_inventory_loop_error", error=str(exc))

        try:
            await _reload_spilled_ollama(model_manager, registry)
        except Exception as exc:
            logger.warning("ollama_spill_reload_loop_error", error=str(exc))

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


async def _image_outputs_cleanup_loop(stop_event: asyncio.Event) -> None:
    """Delete generated image files older than ``settings.image_url_ttl_seconds``.

    Implements the TTL promised by ``/v1/images/generations`` when
    ``response_format=url`` (URLs expire and disk doesn't grow unbounded).
    """
    from pathlib import Path

    from ocabra.config import settings as _settings

    interval_s = max(60, _settings.image_url_ttl_seconds // 6)
    logger.info("image_outputs_cleanup_loop_started",
                ttl_s=_settings.image_url_ttl_seconds, interval_s=interval_s)
    while not stop_event.is_set():
        try:
            out_dir = Path(_settings.image_outputs_dir)
            if out_dir.is_dir():  # noqa: ASYNC240 — solo metadatos (microsegundos); un to_thread costaria mas que la syscall
                cutoff = time.time() - _settings.image_url_ttl_seconds
                removed = 0
                for path in out_dir.iterdir():  # noqa: ASYNC240 — solo metadatos (microsegundos); un to_thread costaria mas que la syscall
                    if not path.is_file():
                        continue
                    try:
                        if path.stat().st_mtime < cutoff:
                            path.unlink(missing_ok=True)
                            removed += 1
                    except OSError as exc:
                        logger.warning("image_outputs_cleanup_unlink_failed",
                                       path=str(path), error=str(exc))
                if removed:
                    logger.info("image_outputs_cleanup_ran", removed=removed)
        except Exception as exc:
            logger.warning("image_outputs_cleanup_loop_error", error=str(exc))

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_s)
        except TimeoutError:
            continue
    logger.info("image_outputs_cleanup_loop_stopped")


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


# Teardown bounds. ``docker stop`` sends SIGKILL after 10 s by default, so a
# shutdown that hangs gains nothing: each step gets at most a few seconds and
# the whole teardown a global budget, after which the remaining steps still
# run but only get a token wait. Worker processes are children of the api and
# die with the container anyway; the teardown is about stopping our loops and
# closing connections cleanly, never about unloading models.
_SHUTDOWN_STEP_TIMEOUT_S = 5.0
_SHUTDOWN_BUDGET_S = 25.0


class _Teardown:
    """Each started component registers its own stop on an ``AsyncExitStack``.

    Teardown then runs in reverse start order, and it also runs when startup
    fails halfway (before 0.112 nothing was stopped in that case and the loops
    already started stayed alive). Every step is bounded and a failing one is
    logged, never raised, so one stuck component can't stop the rest from
    shutting down.
    """

    def __init__(self, stack: AsyncExitStack) -> None:
        self._stack = stack
        self._deadline: float | None = None

    def begin(self) -> None:
        """Start the global shutdown budget (on startup failure it starts at
        the first step that runs)."""
        self._deadline = asyncio.get_running_loop().time() + _SHUTDOWN_BUDGET_S

    def push(self, name: str, stop: Callable[[], Awaitable[object]]) -> None:
        async def _run() -> None:
            loop = asyncio.get_running_loop()
            if self._deadline is None:
                self._deadline = loop.time() + _SHUTDOWN_BUDGET_S
            timeout = max(0.1, min(_SHUTDOWN_STEP_TIMEOUT_S, self._deadline - loop.time()))
            try:
                await asyncio.wait_for(stop(), timeout)
            except TimeoutError:
                logger.warning("shutdown_step_timeout", step=name, timeout_s=round(timeout, 1))
            except Exception as exc:  # noqa: BLE001 - keep tearing down the rest
                logger.warning("shutdown_step_failed", step=name, error=str(exc))

        self._stack.push_async_callback(_run)

    def cancel_on_exit(self, task: asyncio.Task) -> None:
        async def _cancel() -> None:
            if not task.done():
                task.cancel()
            with suppress(asyncio.CancelledError):
                await task

        self.push(task.get_name(), _cancel)

    def loop(
        self, name: str, factory: Callable[[asyncio.Event], Awaitable[object]]
    ) -> asyncio.Task:
        """Start a ``while not stop.is_set()`` loop and register its stop."""
        stop_event = asyncio.Event()
        task = asyncio.create_task(factory(stop_event), name=name)

        async def _stop() -> None:
            stop_event.set()
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

        self.push(name, _stop)
        return task


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncExitStack() as stack:
        teardown = _Teardown(stack)
        try:
            await _startup(app, teardown)
        except BaseException as exc:
            logger.error("ocabra_startup_failed", error=str(exc) or type(exc).__name__)
            raise
        logger.info("ocabra_ready")
        try:
            yield
        finally:
            # ── Shutdown: the stack runs every registered stop, newest first.
            logger.info("shutting_down_ocabra")
            teardown.begin()
    logger.info("ocabra_stopped")


async def _startup(app: FastAPI, teardown: _Teardown) -> None:
    logger.info("starting_ocabra", version=settings.app_version)

    from ocabra.redis_client import close_redis, init_redis

    await init_redis()
    teardown.push("redis", close_redis)
    logger.info("redis_connected")

    # Apply persisted settings before starting any background loop or backend.
    # Otherwise startup logs and the first maintenance iteration can observe
    # stale .env values before DB overrides take effect.
    from ocabra import database
    from ocabra.database import AsyncSessionLocal as _ASL
    from ocabra.db.server_config import (
        apply_overrides_to_settings,
        load_overrides,
        save_override,
    )

    # Runs after everything else that may still use a DB session.
    teardown.push("db_engine", lambda: database.engine.dispose())

    async with _ASL() as _session:
        _overrides = await load_overrides(_session)
    if _overrides:
        apply_overrides_to_settings(settings, _overrides)
        logger.info("config_overrides_loaded", count=len(_overrides))
    else:
        logger.info("config_overrides_none")

    # Privacy: in zero data, temporary files (uploads, media the workers
    # process) go to RAM from the start — before any worker is spawned.
    from ocabra.core.privacy import apply_temp_dir

    apply_temp_dir()

    # Stream 1-A: GPU Manager + Scheduler
    from ocabra.core.gpu_manager import GPUManager
    from ocabra.core.scheduler import GPUScheduler

    gpu_manager = GPUManager()
    await gpu_manager.start()
    teardown.push("gpu_manager", gpu_manager.stop)
    gpu_scheduler = GPUScheduler(gpu_manager)
    app.state.gpu_manager = gpu_manager
    app.state.gpu_scheduler = gpu_scheduler
    logger.info("gpu_manager_ready")

    # Re-apply persisted GPU power caps + persistence-mode flags. Runs in a
    # background task so it can wait for hw-monitor's heartbeat (where the
    # privileged NVML calls actually happen) without blocking the rest of
    # startup. Best-effort: failures are logged, never raised.
    async def _reapply_gpu_power_caps() -> None:
        try:
            from ocabra.database import AsyncSessionLocal
            from ocabra.db.gpu_power_settings import reapply_persisted_settings

            summary = await reapply_persisted_settings(AsyncSessionLocal)
            logger.info("gpu_power_reapply_summary", **summary)
        except Exception as exc:  # noqa: BLE001
            logger.warning("gpu_power_reapply_unexpected_error", error=str(exc))

    teardown.cancel_on_exit(
        asyncio.create_task(_reapply_gpu_power_caps(), name="gpu-power-reapply")
    )

    # Stream 1-B: Worker Pool + Model Manager
    from ocabra.backends.acestep_backend import AceStepBackend
    from ocabra.backends.bitnet_backend import BitnetBackend
    from ocabra.backends.chatterbox_backend import ChatterboxBackend
    from ocabra.backends.comfyui_video_backend import ComfyUIVideoBackend
    from ocabra.backends.diffusers_backend import DiffusersBackend
    from ocabra.backends.flashvsr_backend import FlashVSRBackend
    from ocabra.backends.llama_cpp_backend import LlamaCppBackend
    from ocabra.backends.llamacpp_mks_backend import LlamaCppMksBackend
    from ocabra.backends.mageflow_backend import MageFlowBackend
    from ocabra.backends.ollama_backend import OllamaBackend
    from ocabra.backends.piper_backend import PiperBackend
    from ocabra.backends.sglang_backend import SGLangBackend
    from ocabra.backends.strata_backend import StrataBackend
    from ocabra.backends.tensorrt_llm_backend import TensorRTLLMBackend
    from ocabra.backends.tts_backend import TTSBackend
    from ocabra.backends.upscaler_backend import UpscalerBackend
    from ocabra.backends.vibeasr_backend import VibeAsrBackend
    from ocabra.backends.vllm_backend import VLLMBackend
    from ocabra.backends.voxtral_backend import VoxtralBackend
    from ocabra.backends.whisper_backend import WhisperBackend
    from ocabra.backends.whispercpp_backend import WhisperCppBackend
    from ocabra.core.model_manager import ModelManager
    from ocabra.core.worker_pool import WorkerPool

    worker_pool = WorkerPool()
    worker_pool.register_backend("acestep", AceStepBackend())
    worker_pool.register_backend("diffusers", DiffusersBackend())
    worker_pool.register_backend("mage", MageFlowBackend())
    worker_pool.register_backend("bitnet", BitnetBackend())
    # The backend reads GPU totals: tensor_split "evenly" and which card gets
    # the vision projector. Without it both silently did nothing (to 0.70.6).
    worker_pool.register_backend("llama_cpp", LlamaCppBackend(gpu_manager=gpu_manager))
    worker_pool.register_backend("llamacpp_mks", LlamaCppMksBackend(gpu_manager=gpu_manager))
    worker_pool.register_backend("ollama", OllamaBackend())
    worker_pool.register_backend("sglang", SGLangBackend())
    worker_pool.register_backend("strata", StrataBackend())
    worker_pool.register_backend("whisper", WhisperBackend())
    worker_pool.register_backend("whisper_cpp", WhisperCppBackend())
    worker_pool.register_backend("vibeasr", VibeAsrBackend())
    worker_pool.register_backend("tts", TTSBackend())
    worker_pool.register_backend("chatterbox", ChatterboxBackend())
    worker_pool.register_backend("piper", PiperBackend())
    worker_pool.register_backend("upscaler", UpscalerBackend())
    worker_pool.register_backend("flashvsr", FlashVSRBackend())
    worker_pool.register_backend("comfyui_video", ComfyUIVideoBackend())
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
    worker_pool.set_ollama_ctx_resolver(model_manager.resolve_ollama_num_ctx_cap)
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
        assume_fat_image=settings.backends_fat_image,
    )
    await backend_installer.start()
    app.state.backend_installer = backend_installer
    logger.info("backend_installer_ready")

    # ── Everything ModelManager's background loops use is wired BEFORE
    # model_manager.start() (BUG-26). They used to be injected afterwards,
    # up to the end of startup, and it only worked because the PIN autoload
    # waits ``pin_autoload_delay_s`` (60 s): with 0 the first PIN load ran
    # with no pin_worker_resolver (loaded the bare model instead of the
    # preset's worker), no service manager for pressure eviction, no
    # duration estimator / session registry / batch coordinator.

    # External services (ComfyUI, A1111, Hunyuan…) — pressure eviction can
    # unload them to make room.
    from ocabra.core.service_manager import ServiceManager

    service_manager = ServiceManager()
    await service_manager.start()
    app.state.service_manager = service_manager
    model_manager.set_service_manager(service_manager)
    service_manager.set_gpu_manager(gpu_manager)
    logger.info("service_manager_ready")

    # Profile Registry: only the DB. The default presets need the models
    # (model_manager.start loads them) and are seeded below.
    from ocabra.core.profile_registry import ProfileRegistry

    profile_registry = ProfileRegistry()

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

    # The PIN autoload (and its reload watcher) must load the worker requests
    # use: the default preset's ``model::hash``, like a manual "Load" does.
    async def _pin_worker_resolver(model_id: str) -> str:
        from ocabra.api.openai.realtime import resolve_realtime_llm_runtime_plan
        from ocabra.core.inference.loading import prepare_worker_state

        worker_key, overrides, preferred_gpu, load_policy = await resolve_realtime_llm_runtime_plan(
            model_id, profile_registry=profile_registry, model_manager=model_manager
        )
        if worker_key == model_id:
            return model_id
        await prepare_worker_state(
            model_manager,
            model_id,
            worker_key,
            overrides,
            preferred_gpu=preferred_gpu,
            load_policy=load_policy,
        )
        return worker_key

    model_manager.pin_worker_resolver = _pin_worker_resolver
    # A full card makes room for an on-demand vision projector before the first
    # image, instead of the worker dying with cudaMalloc (gemma4:26b, 2026-11-10).
    worker_pool.set_before_prepare(model_manager.make_room_for_vision)

    # Bloque 20 — DurationEstimator (Nivel 1). Runs a background refresh loop
    # that recomputes per-model calibrations from request_stats every
    # ``estimator_refresh_interval_s``. Kicked off eagerly here so /estimate
    # can respond from the first request rather than falling back to
    # FAMILY_DEFAULT for 30 minutes.
    from ocabra.core.duration_estimator import DurationEstimator

    duration_estimator = DurationEstimator()
    await duration_estimator.start()
    teardown.push("duration_estimator", duration_estimator.stop)
    app.state.duration_estimator = duration_estimator
    model_manager.set_duration_estimator(duration_estimator)
    logger.info("duration_estimator_ready")

    # Bloque 20 — SessionRegistry tracks live Realtime sessions holding
    # worker models. Consumers today: pressure_eviction (veto), router
    # (avoid stealing a reserved worker), Sessions admin UI.
    from ocabra.core.session_registry import SessionRegistry

    session_registry = SessionRegistry()
    await session_registry.start()
    teardown.push("session_registry", session_registry.stop)
    app.state.session_registry = session_registry
    model_manager.set_session_registry(session_registry)
    logger.info("session_registry_ready")

    # Batch-vs-interactive priority coordinator (2026-10-30 design) — tracks
    # in-flight batch-origin requests per worker so an arriving interactive
    # caller can preempt a saturated batch instead of queueing behind it.
    # See fair_slot_gate.py (admission priority) + batch_priority.py
    # (active preemption + cooldowns) for the two halves of the mechanism.
    from ocabra.core.batch_priority import BatchPriorityCoordinator

    batch_priority = BatchPriorityCoordinator()
    app.state.batch_priority = batch_priority
    worker_pool.set_batch_priority(batch_priority, duration_estimator)
    # Progreso de prefill de llama.cpp -> peticion en vuelo (dashboard).
    worker_pool.set_prompt_progress_sink(model_manager.update_prompt_progress)
    model_manager.set_batch_priority(batch_priority)
    logger.info("batch_priority_coordinator_ready")

    # Wired: now the loops (busy/VRAM watchdogs, PIN autoload) may start.
    await model_manager.start()
    teardown.push("model_manager", model_manager.stop)

    async def _warm_gguf_headers() -> None:
        """First GGUF header parse off the event loop (ASY-01)."""
        from ocabra.core import artifacts as model_artifacts
        from ocabra.core import vram_planner

        paths: list[str] = []
        for state in await model_manager.list_states():
            try:
                model = model_artifacts.resolve_paths(state.artifact).get("model")
            except Exception:  # noqa: BLE001 -- uninstalled tag, missing file...
                continue
            if model:
                paths.append(model)  # missing files: arch_from_gguf → None, in the thread
        read = await vram_planner.warm_arch_cache(paths)
        logger.info("gguf_header_cache_warmed", files=read)

    teardown.cancel_on_exit(asyncio.create_task(_warm_gguf_headers(), name="gguf-header-warmup"))

    from ocabra.core.backend_process_manager import BackendProcessManager

    backend_process_manager = BackendProcessManager(model_manager, worker_pool, settings)
    await backend_process_manager.start()
    teardown.push("backend_process_manager", backend_process_manager.stop)
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

    from ocabra.core.trtllm_compile_manager import TrtllmCompileManager

    trtllm_compile_manager = TrtllmCompileManager(model_manager=model_manager)
    await trtllm_compile_manager.start()
    teardown.push("trtllm_compile_manager", trtllm_compile_manager.stop)
    app.state.trtllm_compile_manager = trtllm_compile_manager
    # So completed engines get a default profile alongside their model_config
    # registration.
    trtllm_compile_manager.set_profile_registry(profile_registry)
    await _scan_and_register_trtllm_engines(model_manager)
    logger.info("trtllm_compile_manager_ready")

    # Every LLM gets a "normal" preset equal to what its base loads today; new
    # models get theirs when registered (on_model_added). Needs the models,
    # hence after model_manager.start().
    from ocabra.core.default_presets import ensure_llm_default_presets

    async def _seed_default_presets(model_id: str | None = None) -> None:
        seeded = await ensure_llm_default_presets(
            profile_registry, model_manager, only_model_id=model_id
        )
        if seeded["created"] or seeded["skipped"]:
            logger.info("default_presets", **seeded)

    profile_registry.default_preset_hook = _seed_default_presets
    try:
        await _seed_default_presets()
    except Exception as exc:  # noqa: BLE001 - never block startup
        logger.warning("default_presets_failed", error=str(exc))
    from ocabra.core.context_probe_service import ContextProbeService

    app.state.context_probe = ContextProbeService(model_manager, profile_registry, worker_pool)
    logger.info("profile_registry_ready")

    teardown.loop("idle-eviction-loop", lambda stop: _idle_eviction_loop(model_manager, stop))
    teardown.loop(
        "schedule-maintenance-loop", lambda stop: _schedule_maintenance_loop(gpu_scheduler, stop)
    )
    # oCabra updating itself (core/self_update.py): looks for releases and,
    # with policy "auto", applies one once the node is idle.
    from ocabra.core.self_update import SelfUpdater

    self_updater = SelfUpdater(app)
    app.state.self_updater = self_updater
    teardown.loop("self-update-loop", self_updater.run)
    teardown.loop(
        "service-idle-unload-loop", lambda stop: _service_idle_unload_loop(service_manager, stop)
    )
    teardown.loop("service-health-loop", lambda stop: _service_health_loop(service_manager, stop))
    teardown.loop("image-outputs-cleanup-loop", _image_outputs_cleanup_loop)
    # Batch/upload files: deleted once their retention passes (core/files_retention.py).
    from ocabra.core.files_retention import retention_loop as _files_retention_loop
    from ocabra.database import AsyncSessionLocal as _files_session_factory

    teardown.loop(
        "files-retention-loop", lambda stop: _files_retention_loop(stop, _files_session_factory)
    )
    # Stats tables: prune old rows once their retention passes (stats/retention.py).
    # Off by default (settings.stats_retention_enabled); see that module.
    from ocabra.stats.retention import retention_loop as _stats_retention_loop

    teardown.loop(
        "stats-retention-loop", lambda stop: _stats_retention_loop(stop, _files_session_factory)
    )
    # /v1/videos: cola de trabajos de vídeo, un trabajo a la vez (core/video_jobs.py).
    from ocabra.core.video_jobs import VideoJobManager

    video_job_manager = VideoJobManager(app, _files_session_factory)
    await video_job_manager.start()
    app.state.video_job_manager = video_job_manager
    teardown.push("video-job-manager", video_job_manager.stop)
    teardown.loop(
        "ollama-inventory-loop",
        lambda stop: _ollama_inventory_loop(model_manager, stop, profile_registry),
    )

    # OpenAI TTS voice-name compatibility shim — independent of routers on
    # purpose (see ocabra/core/openai_tts_compat.py).
    from ocabra.core.openai_tts_compat import OpenAITTSCompatRegistry

    openai_tts_compat = OpenAITTSCompatRegistry()
    async with _ASL() as _session:
        await openai_tts_compat.load_all(_session)
    app.state.openai_tts_compat = openai_tts_compat
    logger.info("openai_tts_compat_ready")

    # Same shim for OpenAI image model names (gpt-image-1, dall-e-3...).
    from ocabra.core.openai_image_compat import OpenAIImageCompatRegistry

    openai_image_compat = OpenAIImageCompatRegistry()
    async with _ASL() as _session:
        await openai_image_compat.load_all(_session)
    app.state.openai_image_compat = openai_image_compat

    # Bloque 20 — RouterResolver walks routing_targets to pick the best
    # candidate for router profiles.
    from ocabra.core.router_resolver import RouterResolver

    router_resolver = RouterResolver(
        profile_registry=profile_registry,
        model_manager=model_manager,
        duration_estimator=duration_estimator,
        session_registry=session_registry,
        gpu_manager=gpu_manager,
    )
    app.state.router_resolver = router_resolver
    logger.info("router_resolver_ready")

    # Auth: seed first admin if the users table is empty
    from ocabra.core.auth_manager import seed_first_admin

    async with _ASL() as _session:
        await seed_first_admin(_session)
    logger.info("auth_seed_done")

    # Persist JWT secret in DB so sessions survive restarts
    if "jwt_secret" not in _overrides:
        async with _ASL() as _session:
            await save_override(_session, "jwt_secret", settings.jwt_secret)
            await _session.commit()
        logger.info("jwt_secret_persisted")

    # Stored secrets onto OCABRA_SECRETS_KEY (SEG-17): after the overrides, so
    # the old session-secret key is the persisted one and still decrypts them.
    from ocabra.core.secrets_box import rotate_stored_secrets

    try:
        await rotate_stored_secrets(_ASL)
    except Exception as exc:  # noqa: BLE001 - the old key still reads them
        logger.error("stored_secrets_rotation_failed", error=str(exc))

    # MCP registry (agents + MCP, Fase 1)
    from ocabra.agents.mcp_registry import MCPRegistry, set_registry

    mcp_registry = MCPRegistry(session_factory=_ASL)
    await mcp_registry.start()

    async def _stop_mcp() -> None:
        try:
            await mcp_registry.stop()
        finally:
            set_registry(None)

    teardown.push("mcp_registry", _stop_mcp)
    app.state.mcp_registry = mcp_registry
    set_registry(mcp_registry)
    logger.info("mcp_registry_ready")

    # Federation manager (optional, only if enabled). An unattended node told to
    # join a hub on startup (cloud pod) needs it regardless of the toggle.
    if settings.federation_join_url and settings.federation_join_invite:
        settings.federation_enabled = True
    if settings.federation_enabled:
        from ocabra.core.federation import FederationManager

        federation_manager = FederationManager(settings, _ASL)
        federation_manager.gpu_manager = gpu_manager
        # «Mi servidor, mis reglas»: what is loaded here, and public names → ids.
        federation_manager.model_manager = model_manager
        federation_manager.profile_registry = profile_registry
        # Registered before start(): a start that fails halfway may already
        # have opened clients / started dialers, and stop() copes with that.
        teardown.push("federation_manager", federation_manager.stop)
        await federation_manager.start()
        app.state.federation_manager = federation_manager
        logger.info("federation_manager_ready")
    else:
        app.state.federation_manager = None

    logger.info("vllm_backend_registered")

    # OpenAI Batches API processor
    from ocabra.core.batch_processor import BatchProcessor

    batch_processor = BatchProcessor(app, coordinator=batch_priority)
    await batch_processor.start()
    teardown.push("batch_processor", batch_processor.stop)
    app.state.batch_processor = batch_processor
    logger.info("batch_processor_ready")

    # Application benchmarks: persistent queue with process-level pause/resume
    # and automatic preemption when non-benchmark inference traffic appears.
    from ocabra.benchmarks.manager import BenchmarkManager

    benchmark_manager = BenchmarkManager(app)
    await benchmark_manager.start()
    teardown.push("benchmark_manager", benchmark_manager.stop)
    app.state.benchmark_manager = benchmark_manager
    logger.info("benchmark_manager_ready")


# The /docs prose and OpenAPI customization live in openapi_docs.py. Re-exported
# here (``_API_DESCRIPTION``) so tests can keep importing it from ocabra.main.
from ocabra.openapi_docs import _API_DESCRIPTION, build_openapi_schema  # noqa: E402

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
    return build_openapi_schema(app)


app.openapi = _custom_openapi

app.add_middleware(
    CORSMiddleware,
    **_build_cors_config(),
)


# Global 500 logger: FastAPI's default exception handler returns a bare
# "Internal Server Error" with no traceback in the logs, which makes
# diagnosing federation/proxy issues painful. Log the full traceback for
# every uncaught exception so it's never silent.
from ocabra.api._errors import error_content, error_family, request_path  # noqa: E402
from ocabra.core.worker_pool import VisionNoRoomError  # noqa: E402


@app.exception_handler(VisionNoRoomError)
async def _vision_no_room_handler(request, exc):  # noqa: ANN001
    """An image a model cannot take right now (no room for its vision
    projector): 503 ``vision_no_room`` instead of a bare 500."""
    from fastapi.responses import JSONResponse

    logger.info("vision_no_room", path=request.url.path, error=str(exc))
    # Same envelope as every other oCabra error (``HTTPException`` detail),
    # plus the family's top-level ``error`` (REF-10).
    detail = {
        "error": {
            "message": str(exc),
            "type": "server_error",
            "code": "vision_no_room",
        }
    }
    return JSONResponse(status_code=503, content=error_content(request.url.path, 503, detail))


from starlette.exceptions import HTTPException as _StarletteHTTPException  # noqa: E402


@app.exception_handler(_StarletteHTTPException)
async def _peer_fallback_not_found_handler(request, exc):  # noqa: ANN001
    """A model only a peer has, whose peer failed: 503, not "does not exist".

    When the chosen peer refuses or is unreachable the request falls back to
    local, and a model that exists only on that peer (a pod's) then answered
    404 "model not found": wrong, and not retryable. The fallback is recorded
    in ``federation_info``; here a 404 after it becomes a retryable 503 that
    names the node.
    """
    from fastapi.exception_handlers import http_exception_handler
    from fastapi.responses import JSONResponse

    path = request_path(request)
    info = getattr(request.state, "federation_info", None) or {}
    if exc.status_code == 404 and info.get("fallback"):
        peer = info.get("fallback_peer") or "a federation peer"
        detail = info.get("fallback_detail")
        message = (
            f"The model is only available on node '{peer}', which did not serve the "
            f"request ({detail}). Retry shortly." if detail
            else f"The model is only available on node '{peer}', which did not serve "
            "the request. Retry shortly."
        )  # fmt: skip
        detail_body = {
            "error": {
                "message": message,
                "type": "server_error",
                "code": "peer_unavailable",
            }
        }
        return JSONResponse(
            status_code=503,
            content=error_content(path, 503, detail_body),
            headers={"Retry-After": "15"},
        )
    if error_family(path) is None:
        return await http_exception_handler(request, exc)
    # /v1 and /api (REF-10): FastAPI's handler plus the family's ``error``.
    from fastapi.utils import is_body_allowed_for_status_code
    from starlette.responses import Response

    headers = getattr(exc, "headers", None)
    if not is_body_allowed_for_status_code(exc.status_code):
        return Response(status_code=exc.status_code, headers=headers)
    return JSONResponse(
        status_code=exc.status_code,
        content=error_content(path, exc.status_code, exc.detail),
        headers=headers,
    )


from fastapi.exceptions import RequestValidationError as _RequestValidationError  # noqa: E402


@app.exception_handler(_RequestValidationError)
async def _request_validation_handler(request, exc):  # noqa: ANN001
    """FastAPI's 422, plus the family's top-level ``error`` on /v1 and /api."""
    from fastapi.encoders import jsonable_encoder
    from fastapi.exception_handlers import request_validation_exception_handler
    from fastapi.responses import JSONResponse

    if error_family(request.url.path) is None:
        return await request_validation_exception_handler(request, exc)
    detail = jsonable_encoder(exc.errors())
    return JSONResponse(status_code=422, content=error_content(request.url.path, 422, detail))


@app.exception_handler(Exception)
async def _log_unhandled_exception(request, exc):
    import traceback as _tb

    from fastapi.responses import JSONResponse

    logger.warning(
        "unhandled_exception",
        path=request.url.path,
        method=request.method,
        error=str(exc),
        error_type=type(exc).__name__,
        traceback=_tb.format_exc(),
    )
    detail = {
        "error": {
            "message": "Internal Server Error",
            "type": "server_error",
            "code": type(exc).__name__,
        }
    }
    return JSONResponse(status_code=500, content=error_content(request.url.path, 500, detail))

# Stream 3-A: Stats middleware for /v1/* endpoints. Records only.
from ocabra.core.admission import AdmissionMiddleware  # noqa: E402
from ocabra.stats.collector import StatsMiddleware, record_admission_rejection  # noqa: E402

app.add_middleware(StatsMiddleware)
# Admission (in-flight cap, maintenance, circuit breaker; REF-08): just outside
# the stats, so its slot spans the whole response, and inside the federation
# hop marker, whose ``federation_info`` it reads. Its rejections never reach
# the stats middleware: they are recorded through ``on_reject``.
app.add_middleware(AdmissionMiddleware, on_reject=record_admission_rejection)
# Outermost: marks requests that came from a federation peer before anything
# else runs, so they are never forwarded to yet another peer.
from ocabra.core.federation import FederationHopMiddleware  # noqa: E402

app.add_middleware(FederationHopMiddleware)

from ocabra.core.privacy import PrivacyHeaderMiddleware  # noqa: E402

# X-Ocabra-Privacy on every response: which privacy mode served it.
app.add_middleware(PrivacyHeaderMiddleware)

from ocabra.api._risky_config import RiskyConfigMiddleware  # noqa: E402

# Options that run arbitrary code (trust_remote_code, extra_args, …) are for
# system_admin only, however they get in (SEG-04).
app.add_middleware(RiskyConfigMiddleware)

# ── ROUTERS ──────────────────────────────────────────────────
# Each stream adds its router here. Do not remove this comment.
from ocabra.api.health import router as health_router  # noqa: E402

app.include_router(health_router)

# Paginas de documentacion en prosa (/docs/changelog, /docs/models). Se montan
# despues de las rutas de Swagger para que /docs y /docs/oauth2-redirect, que
# FastAPI registra al crear la app, sigan ganando.
from ocabra.api.docs_pages import router as docs_pages_router  # noqa: E402

app.include_router(docs_pages_router)

# ── Internal routers (hidden from /docs — admin dashboard only) ──
from ocabra.api.internal.downloads import router as downloads_router  # noqa: E402
from ocabra.api.internal.gpus import router as gpus_router  # noqa: E402
from ocabra.api.internal.host import router as host_router  # noqa: E402
from ocabra.api.internal.logical_models import router as logical_models_router  # noqa: E402
from ocabra.api.internal.models import router as models_router  # noqa: E402
from ocabra.api.internal.models_update import router as models_update_router  # noqa: E402
from ocabra.api.internal.ollama_admin import router as ollama_admin_router  # noqa: E402
from ocabra.api.internal.registry import router as registry_router  # noqa: E402
from ocabra.api.internal.services import router as services_router  # noqa: E402
from ocabra.api.internal.status import router as status_router  # noqa: E402
from ocabra.api.internal.system_update import router as system_update_router  # noqa: E402
from ocabra.api.internal.video import router as video_router  # noqa: E402
from ocabra.api.internal.ws import router as ws_router  # noqa: E402

app.include_router(gpus_router, prefix="/ocabra", include_in_schema=False)
app.include_router(status_router, prefix="/ocabra", include_in_schema=False)
app.include_router(video_router, prefix="/ocabra", include_in_schema=False)
# Profiles before models: models' catch-all GET /models/{model_id:path} would
# otherwise swallow GET /models/{id}/profiles (it answered 404 "model
# '<id>/profiles' not found").
from ocabra.api.internal.profiles import router as profiles_router  # noqa: E402

app.include_router(profiles_router, prefix="/ocabra", include_in_schema=False)
from ocabra.api.internal.openai_image_compat import (  # noqa: E402
    router as openai_image_compat_router,
)
from ocabra.api.internal.openai_tts_compat import router as openai_tts_compat_router  # noqa: E402

app.include_router(openai_tts_compat_router, prefix="/ocabra", include_in_schema=False)
app.include_router(openai_image_compat_router, prefix="/ocabra", include_in_schema=False)
app.include_router(models_router, prefix="/ocabra", include_in_schema=False)
app.include_router(system_update_router, prefix="/ocabra", include_in_schema=False)
app.include_router(logical_models_router, prefix="/ocabra", include_in_schema=False)
app.include_router(models_update_router, prefix="/ocabra", include_in_schema=False)
app.include_router(ws_router, prefix="/ocabra", include_in_schema=False)
app.include_router(registry_router, prefix="/ocabra", include_in_schema=False)
app.include_router(downloads_router, prefix="/ocabra", include_in_schema=False)
app.include_router(services_router, prefix="/ocabra", include_in_schema=False)
app.include_router(host_router, prefix="/ocabra", include_in_schema=False)
app.include_router(ollama_admin_router, prefix="/ocabra", include_in_schema=False)

# Bloque 15 — Modular backends router
from ocabra.api.internal.backends import router as backends_router  # noqa: E402

app.include_router(backends_router, prefix="/ocabra", include_in_schema=False)
from ocabra.api.internal.federation import router as federation_router  # noqa: E402

app.include_router(federation_router, prefix="/ocabra", include_in_schema=False)
from ocabra.api.internal.cloud import router as cloud_router  # noqa: E402

app.include_router(cloud_router, prefix="/ocabra", include_in_schema=False)

# Stream 3-A: OpenAI API
from ocabra.api.openai import router as openai_router  # noqa: E402

app.include_router(openai_router, prefix="/v1")

# Stream 3-B: Ollama API
from ocabra.api.ollama import router as ollama_router  # noqa: E402

app.include_router(ollama_router)

# Stream 5: Metrics (public), Config/Stats/TRT/Auth/Users/Groups/Profiles (internal)
from ocabra.api.internal.auth import router as auth_router  # noqa: E402
from ocabra.api.internal.benchmarks import router as benchmarks_router  # noqa: E402
from ocabra.api.internal.config import router as config_router  # noqa: E402
from ocabra.api.internal.estimate import router as estimate_router  # noqa: E402
from ocabra.api.internal.groups import router as groups_router  # noqa: E402
from ocabra.api.internal.sessions import router as sessions_router  # noqa: E402
from ocabra.api.internal.stats import router as stats_router  # noqa: E402
from ocabra.api.internal.trtllm import router as trtllm_router  # noqa: E402
from ocabra.api.internal.users import router as users_router  # noqa: E402
from ocabra.api.metrics import router as metrics_router  # noqa: E402

app.include_router(metrics_router)
app.include_router(config_router, prefix="/ocabra", include_in_schema=False)
app.include_router(stats_router, prefix="/ocabra", include_in_schema=False)
app.include_router(trtllm_router, prefix="/ocabra", include_in_schema=False)
app.include_router(auth_router, prefix="/ocabra", include_in_schema=False)
app.include_router(benchmarks_router, prefix="/ocabra", include_in_schema=False)
app.include_router(users_router, prefix="/ocabra", include_in_schema=False)
app.include_router(groups_router, prefix="/ocabra", include_in_schema=False)
app.include_router(estimate_router, prefix="/ocabra", include_in_schema=False)
app.include_router(sessions_router, prefix="/ocabra", include_in_schema=False)

# Agents + MCP (plan: docs/tasks/agents-mcp-plan.md)
from ocabra.api.internal.agents import router as agents_router  # noqa: E402
from ocabra.api.internal.mcp_servers import router as mcp_servers_router  # noqa: E402

app.include_router(agents_router, prefix="/ocabra", include_in_schema=False)
app.include_router(mcp_servers_router, prefix="/ocabra", include_in_schema=False)

from ocabra.api.internal.stats_agents import router as stats_agents_router  # noqa: E402

app.include_router(stats_agents_router, prefix="/ocabra", include_in_schema=False)
# ─────────────────────────────────────────────────────────────
