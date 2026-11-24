"""Periodic background loops started from the api's lifespan.

Each loop is a ``while not stop_event.is_set()`` coroutine that the lifespan
starts via ``_Teardown.loop`` (see ``ocabra/bootstrap.py``) and cancels on
shutdown. Intervals and behaviour live here; the wiring (which collaborators
each loop gets) lives in ``bootstrap.py``.
"""

import asyncio
import time

import structlog

from ocabra.config import settings

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
