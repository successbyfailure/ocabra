"""
GET/PATCH /ocabra/config — Server configuration API.
POST /ocabra/config/litellm/sync — Manual LiteLLM sync trigger.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from ocabra.config import settings
from ocabra.database import AsyncSessionLocal
from ocabra.db.model_config import global_schedule_rows_to_payload, get_global_schedule_rows, replace_global_schedules

router = APIRouter(tags=["config"])


class EvictionSchedulePayload(BaseModel):
    id: str
    days: list[int] = Field(default_factory=list)
    start: str = "00:00"
    end: str = "00:00"
    enabled: bool = True


class ServerConfigPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_gpu_index: int | None = Field(default=None, alias="defaultGpuIndex")
    idle_timeout_seconds: int | None = Field(default=None, alias="idleTimeoutSeconds")
    idle_eviction_check_interval_seconds: int | None = Field(
        default=None, alias="idleEvictionCheckIntervalSeconds"
    )
    vram_buffer_mb: int | None = Field(default=None, alias="vramBufferMb")
    vram_pressure_threshold_pct: float | None = Field(default=None, alias="vramPressureThresholdPct")
    log_level: str | None = Field(default=None, alias="logLevel")
    litellm_base_url: str | None = Field(default=None, alias="litellmBaseUrl")
    litellm_admin_key: str | None = Field(default=None, alias="litellmAdminKey")
    litellm_auto_sync: bool | None = Field(default=None, alias="litellmAutoSync")
    energy_cost_eur_kwh: float | None = Field(default=None, alias="energyCostEurKwh")
    models_dir: str | None = Field(default=None, alias="modelsDir")
    download_dir: str | None = Field(default=None, alias="downloadDir")
    max_temperature_c: int | None = Field(default=None, alias="maxTemperatureC")
    global_schedules: list[EvictionSchedulePayload] | None = Field(default=None, alias="globalSchedules")


def _config_overrides(request: Request) -> dict[str, Any]:
    if not hasattr(request.app.state, "config_overrides"):
        request.app.state.config_overrides = {}
    return request.app.state.config_overrides


def _masked_admin_key(value: str) -> str:
    return "***" if value else ""


def _build_config_response(request: Request) -> dict[str, Any]:
    overrides = _config_overrides(request)
    default_download_dir = f"{settings.models_dir.rstrip('/')}/downloads"
    return {
        "defaultGpuIndex": settings.default_gpu_index,
        "idleTimeoutSeconds": settings.idle_timeout_seconds,
        "idleEvictionCheckIntervalSeconds": settings.idle_eviction_check_interval_seconds,
        "vramBufferMb": settings.vram_buffer_mb,
        "vramPressureThresholdPct": settings.vram_pressure_threshold_pct,
        "logLevel": settings.log_level,
        "litellmBaseUrl": settings.litellm_base_url,
        "litellmAdminKey": _masked_admin_key(settings.litellm_admin_key),
        "litellmAutoSync": settings.litellm_auto_sync,
        "energyCostEurKwh": settings.energy_cost_eur_kwh,
        "modelsDir": settings.models_dir,
        "downloadDir": overrides.get("download_dir", default_download_dir),
        "maxTemperatureC": overrides.get("max_temperature_c", 88),
        "globalSchedules": [],
    }


async def _load_global_schedules() -> list[dict[str, Any]]:
    async with AsyncSessionLocal() as session:
        rows = await get_global_schedule_rows(session)
    return global_schedule_rows_to_payload(rows)


@router.get("/config", summary="Get server configuration")
async def get_config(request: Request) -> dict[str, Any]:
    """Return current server configuration (non-secret values)."""
    payload = _build_config_response(request)
    payload["globalSchedules"] = await _load_global_schedules()
    return payload


@router.patch("/config", summary="Patch server configuration")
async def patch_config(patch: ServerConfigPatch, request: Request) -> dict[str, Any]:
    """Patch mutable server configuration values for the running process."""
    payload = patch.model_dump(exclude_unset=True)
    if "models_dir" in payload:
        raise HTTPException(
            status_code=400,
            detail="modelsDir is managed by the container environment and cannot be changed at runtime",
        )

    if "global_schedules" in payload:
        try:
            async with AsyncSessionLocal() as session:
                await replace_global_schedules(
                    session,
                    [schedule.model_dump() for schedule in payload["global_schedules"]],
                )
                await session.commit()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    if "default_gpu_index" in payload:
        settings.default_gpu_index = int(payload["default_gpu_index"])
    if "idle_timeout_seconds" in payload:
        settings.idle_timeout_seconds = int(payload["idle_timeout_seconds"])
    if "idle_eviction_check_interval_seconds" in payload:
        settings.idle_eviction_check_interval_seconds = int(payload["idle_eviction_check_interval_seconds"])
    if "vram_buffer_mb" in payload:
        settings.vram_buffer_mb = int(payload["vram_buffer_mb"])
    if "vram_pressure_threshold_pct" in payload:
        settings.vram_pressure_threshold_pct = float(payload["vram_pressure_threshold_pct"])
    if "log_level" in payload:
        settings.log_level = str(payload["log_level"])
    if "litellm_base_url" in payload:
        settings.litellm_base_url = str(payload["litellm_base_url"])
    if "litellm_admin_key" in payload:
        new_key = str(payload["litellm_admin_key"])
        if new_key and new_key != "***":
            settings.litellm_admin_key = new_key
        elif new_key == "":
            settings.litellm_admin_key = ""
    if "litellm_auto_sync" in payload:
        settings.litellm_auto_sync = bool(payload["litellm_auto_sync"])
    if "energy_cost_eur_kwh" in payload:
        settings.energy_cost_eur_kwh = float(payload["energy_cost_eur_kwh"])

    overrides = _config_overrides(request)
    if "download_dir" in payload:
        overrides["download_dir"] = str(payload["download_dir"])
    if "max_temperature_c" in payload:
        overrides["max_temperature_c"] = int(payload["max_temperature_c"])

    response = _build_config_response(request)
    response["globalSchedules"] = await _load_global_schedules()
    return response


@router.post("/config/litellm/sync", summary="Sync models to LiteLLM proxy")
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
        "synced_models": result.synced,
        "errors": result.errors,
    })
