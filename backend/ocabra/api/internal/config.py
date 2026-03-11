"""
GET/PATCH /ocabra/config — Server configuration API.
POST /ocabra/config/sync-litellm — Manual LiteLLM sync trigger.
"""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ocabra.config import settings

router = APIRouter(tags=["config"])


@router.get("/config", summary="Get server configuration")
async def get_config() -> dict:
    """Return current server configuration (non-secret values)."""
    return {
        "defaultGpuIndex": settings.default_gpu_index,
        "idleTimeoutSeconds": settings.idle_timeout_seconds,
        "idleEvictionCheckIntervalSeconds": settings.idle_eviction_check_interval_seconds,
        "vramBufferMb": settings.vram_buffer_mb,
        "vramPressureThresholdPct": settings.vram_pressure_threshold_pct,
        "logLevel": settings.log_level,
        "litellmBaseUrl": settings.litellm_base_url,
        "litellmAdminKey": "***" if settings.litellm_admin_key else "",
        "litellmAutoSync": settings.litellm_auto_sync,
        "energyCostEurKwh": settings.energy_cost_eur_kwh,
    }


@router.post("/config/sync-litellm", summary="Sync models to LiteLLM proxy")
async def sync_litellm(request: Request) -> JSONResponse:
    """
    Manually trigger synchronisation of all loaded models to LiteLLM proxy.

    Returns the number of models synced and any errors encountered.
    """
    from ocabra.integrations.litellm_sync import LiteLLMSync

    model_manager = request.app.state.model_manager
    syncer = LiteLLMSync(model_manager)
    result = await syncer.sync_all()

    return JSONResponse({
        "syncedModels": result.synced,
        "errors": result.errors,
    })
