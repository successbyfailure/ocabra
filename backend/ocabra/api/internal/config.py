"""
GET/PATCH /ocabra/config — Server configuration API.
POST /ocabra/config/litellm/sync — Manual LiteLLM sync trigger.
"""
from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from ocabra.api._deps_auth import UserContext, require_role
from ocabra.config import settings
from ocabra.database import AsyncSessionLocal
from ocabra.db.model_config import global_schedule_rows_to_payload, get_global_schedule_rows, replace_global_schedules
from ocabra.db.server_config import save_override

logger = structlog.get_logger(__name__)

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
    model_load_wait_timeout_seconds: int | None = Field(default=None, alias="modelLoadWaitTimeoutSeconds")
    pressure_eviction_drain_timeout_seconds: int | None = Field(
        default=None, alias="pressureEvictionDrainTimeoutSeconds"
    )
    vram_buffer_mb: int | None = Field(default=None, alias="vramBufferMb")
    vram_pressure_threshold_pct: float | None = Field(default=None, alias="vramPressureThresholdPct")
    openai_audio_max_part_size_mb: int | None = Field(default=None, alias="openaiAudioMaxPartSizeMb")
    whisper_startup_timeout_seconds: int | None = Field(default=None, alias="whisperStartupTimeoutSeconds")
    log_level: str | None = Field(default=None, alias="logLevel")
    litellm_base_url: str | None = Field(default=None, alias="litellmBaseUrl")
    litellm_admin_key: str | None = Field(default=None, alias="litellmAdminKey")
    litellm_auto_sync: bool | None = Field(default=None, alias="litellmAutoSync")
    energy_cost_eur_kwh: float | None = Field(default=None, alias="energyCostEurKwh")
    models_dir: str | None = Field(default=None, alias="modelsDir")
    download_dir: str | None = Field(default=None, alias="downloadDir")
    max_temperature_c: int | None = Field(default=None, alias="maxTemperatureC")
    vllm_gpu_memory_utilization: float | None = Field(default=None, alias="vllmGpuMemoryUtilization")
    vllm_max_num_seqs: int | None = Field(default=None, alias="vllmMaxNumSeqs")
    vllm_max_num_batched_tokens: int | None = Field(default=None, alias="vllmMaxNumBatchedTokens")
    vllm_enable_prefix_caching: bool | None = Field(default=None, alias="vllmEnablePrefixCaching")
    vllm_enforce_eager: bool | None = Field(default=None, alias="vllmEnforceEager")
    sglang_mem_fraction_static: float | None = Field(default=None, alias="sglangMemFractionStatic")
    sglang_context_length: int | None = Field(default=None, alias="sglangContextLength")
    sglang_disable_radix_cache: bool | None = Field(default=None, alias="sglangDisableRadixCache")
    llama_cpp_gpu_layers: int | None = Field(default=None, alias="llamaCppGpuLayers")
    llama_cpp_ctx_size: int | None = Field(default=None, alias="llamaCppCtxSize")
    llama_cpp_flash_attn: bool | None = Field(default=None, alias="llamaCppFlashAttn")
    bitnet_gpu_layers: int | None = Field(default=None, alias="bitnetGpuLayers")
    bitnet_ctx_size: int | None = Field(default=None, alias="bitnetCtxSize")
    bitnet_flash_attn: bool | None = Field(default=None, alias="bitnetFlashAttn")
    diffusers_torch_dtype: str | None = Field(default=None, alias="diffusersTorchDtype")
    diffusers_offload_mode: str | None = Field(default=None, alias="diffusersOffloadMode")
    diffusers_enable_torch_compile: bool | None = Field(default=None, alias="diffusersEnableTorchCompile")
    diffusers_enable_xformers: bool | None = Field(default=None, alias="diffusersEnableXformers")
    diffusers_allow_tf32: bool | None = Field(default=None, alias="diffusersAllowTf32")
    tensorrt_llm_enabled: bool | None = Field(default=None, alias="tensorrtLlmEnabled")
    tensorrt_llm_max_batch_size: int | None = Field(default=None, alias="tensorrtLlmMaxBatchSize")
    tensorrt_llm_context_length: int | None = Field(default=None, alias="tensorrtLlmContextLength")
    global_schedules: list[EvictionSchedulePayload] | None = Field(default=None, alias="globalSchedules")
    require_api_key_openai: bool | None = Field(default=None, alias="requireApiKeyOpenai")
    require_api_key_ollama: bool | None = Field(default=None, alias="requireApiKeyOllama")
    realtime_default_stt_model: str | None = Field(
        default=None, alias="realtimeDefaultSttModel",
        description="Default STT model for Realtime API sessions (profile_id or model_id)",
    )
    realtime_default_tts_model: str | None = Field(
        default=None, alias="realtimeDefaultTtsModel",
        description="Default TTS model for Realtime API sessions (profile_id or model_id)",
    )
    federation_enabled: bool | None = Field(
        default=None, alias="federationEnabled",
        description="Enable/disable federation mode for multi-node inference (hot toggle)",
    )
    federation_node_name: str | None = Field(
        default=None, alias="federationNodeName",
        description="Human-readable name for this node in the federation",
    )

    # ── Generation services (Hunyuan, ComfyUI, A1111, ACE-Step, Unsloth) ───
    # Hot config: idle unload, generation grace, preferred GPU.
    hunyuan_idle_unload_seconds: int | None = Field(default=None, alias="hunyuanIdleUnloadSeconds")
    hunyuan_generation_grace_period_s: int | None = Field(default=None, alias="hunyuanGenerationGracePeriodS")
    hunyuan_preferred_gpu: int | None = Field(default=None, alias="hunyuanPreferredGpu")
    comfyui_idle_unload_seconds: int | None = Field(default=None, alias="comfyuiIdleUnloadSeconds")
    comfyui_generation_grace_period_s: int | None = Field(default=None, alias="comfyuiGenerationGracePeriodS")
    comfyui_preferred_gpu: int | None = Field(default=None, alias="comfyuiPreferredGpu")
    a1111_idle_unload_seconds: int | None = Field(default=None, alias="a1111IdleUnloadSeconds")
    a1111_generation_grace_period_s: int | None = Field(default=None, alias="a1111GenerationGracePeriodS")
    a1111_preferred_gpu: int | None = Field(default=None, alias="a1111PreferredGpu")
    acestep_idle_unload_seconds: int | None = Field(default=None, alias="acestepIdleUnloadSeconds")
    acestep_generation_grace_period_s: int | None = Field(default=None, alias="acestepGenerationGracePeriodS")
    acestep_preferred_gpu: int | None = Field(default=None, alias="acestepPreferredGpu")
    unsloth_idle_unload_seconds: int | None = Field(default=None, alias="unslothIdleUnloadSeconds")
    unsloth_generation_grace_period_s: int | None = Field(default=None, alias="unslothGenerationGracePeriodS")
    unsloth_preferred_gpu: int | None = Field(default=None, alias="unslothPreferredGpu")
    generation_gpu_util_threshold_pct: int | None = Field(
        default=None, alias="generationGpuUtilThresholdPct",
        description="GPU utilisation %% above which services without a dedicated generation-status endpoint are considered busy (Hunyuan, ACE-Step, Unsloth).",
    )


def _masked_admin_key(value: str) -> str:
    return "***" if value else ""


def _build_config_response(request: Request) -> dict[str, Any]:
    effective_download_dir = (
        settings.download_dir or f"{settings.models_dir.rstrip('/')}/downloads"
    )
    return {
        "defaultGpuIndex": settings.default_gpu_index,
        "idleTimeoutSeconds": settings.idle_timeout_seconds,
        "idleEvictionCheckIntervalSeconds": settings.idle_eviction_check_interval_seconds,
        "modelLoadWaitTimeoutSeconds": settings.model_load_wait_timeout_s,
        "pressureEvictionDrainTimeoutSeconds": settings.pressure_eviction_drain_timeout_s,
        "vramBufferMb": settings.vram_buffer_mb,
        "vramPressureThresholdPct": settings.vram_pressure_threshold_pct,
        "openaiAudioMaxPartSizeMb": settings.openai_audio_max_part_size_mb,
        "whisperStartupTimeoutSeconds": settings.whisper_startup_timeout_s,
        "logLevel": settings.log_level,
        "litellmBaseUrl": settings.litellm_base_url,
        "litellmAdminKey": _masked_admin_key(settings.litellm_admin_key),
        "litellmAutoSync": settings.litellm_auto_sync,
        "energyCostEurKwh": settings.energy_cost_eur_kwh,
        "modelsDir": settings.models_dir,
        "downloadDir": effective_download_dir,
        "maxTemperatureC": settings.max_temperature_c,
        "vllmGpuMemoryUtilization": settings.vllm_gpu_memory_utilization,
        "vllmMaxNumSeqs": settings.vllm_max_num_seqs,
        "vllmMaxNumBatchedTokens": settings.vllm_max_num_batched_tokens,
        "vllmEnablePrefixCaching": settings.vllm_enable_prefix_caching,
        "vllmEnforceEager": settings.vllm_enforce_eager,
        "sglangMemFractionStatic": settings.sglang_mem_fraction_static,
        "sglangContextLength": settings.sglang_context_length,
        "sglangDisableRadixCache": settings.sglang_disable_radix_cache,
        "llamaCppGpuLayers": settings.llama_cpp_gpu_layers,
        "llamaCppCtxSize": settings.llama_cpp_ctx_size,
        "llamaCppFlashAttn": settings.llama_cpp_flash_attn,
        "bitnetGpuLayers": settings.bitnet_gpu_layers,
        "bitnetCtxSize": settings.bitnet_ctx_size,
        "bitnetFlashAttn": settings.bitnet_flash_attn,
        "diffusersTorchDtype": settings.diffusers_torch_dtype,
        "diffusersOffloadMode": settings.diffusers_offload_mode,
        "diffusersEnableTorchCompile": settings.diffusers_enable_torch_compile,
        "diffusersEnableXformers": settings.diffusers_enable_xformers,
        "diffusersAllowTf32": settings.diffusers_allow_tf32,
        "tensorrtLlmEnabled": settings.tensorrt_llm_enabled,
        "tensorrtLlmMaxBatchSize": settings.tensorrt_llm_max_batch_size,
        "tensorrtLlmContextLength": settings.tensorrt_llm_context_length,
        "globalSchedules": [],
        "requireApiKeyOpenai": settings.require_api_key_openai,
        "requireApiKeyOllama": settings.require_api_key_ollama,
        "realtimeDefaultSttModel": settings.realtime_default_stt_model,
        "realtimeDefaultTtsModel": settings.realtime_default_tts_model,
        "federationEnabled": settings.federation_enabled,
        "federationNodeName": settings.federation_node_name,
        "hunyuanIdleUnloadSeconds": settings.hunyuan_idle_unload_seconds,
        "hunyuanGenerationGracePeriodS": settings.hunyuan_generation_grace_period_s,
        "hunyuanPreferredGpu": settings.hunyuan_preferred_gpu,
        "comfyuiIdleUnloadSeconds": settings.comfyui_idle_unload_seconds,
        "comfyuiGenerationGracePeriodS": settings.comfyui_generation_grace_period_s,
        "comfyuiPreferredGpu": settings.comfyui_preferred_gpu,
        "a1111IdleUnloadSeconds": settings.a1111_idle_unload_seconds,
        "a1111GenerationGracePeriodS": settings.a1111_generation_grace_period_s,
        "a1111PreferredGpu": settings.a1111_preferred_gpu,
        "acestepIdleUnloadSeconds": settings.acestep_idle_unload_seconds,
        "acestepGenerationGracePeriodS": settings.acestep_generation_grace_period_s,
        "acestepPreferredGpu": settings.acestep_preferred_gpu,
        "unslothIdleUnloadSeconds": settings.unsloth_idle_unload_seconds,
        "unslothGenerationGracePeriodS": settings.unsloth_generation_grace_period_s,
        "unslothPreferredGpu": settings.unsloth_preferred_gpu,
        "generationGpuUtilThresholdPct": settings.generation_gpu_util_threshold_pct,
    }


async def _load_global_schedules() -> list[dict[str, Any]]:
    async with AsyncSessionLocal() as session:
        rows = await get_global_schedule_rows(session)
    return global_schedule_rows_to_payload(rows)


@router.get(
    "/config",
    summary="Get server configuration",
    description=(
        "Return all server configuration values including GPU defaults, idle eviction, "
        "backend tuning, LiteLLM integration, energy pricing, and global schedules. "
        "Sensitive keys (e.g. litellmAdminKey) are masked."
    ),
)
async def get_config(
    request: Request,
    _user: UserContext = Depends(require_role("system_admin")),
) -> dict[str, Any]:
    """Return current server configuration (non-secret values)."""
    payload = _build_config_response(request)
    payload["globalSchedules"] = await _load_global_schedules()
    return payload


@router.patch(
    "/config",
    summary="Patch server configuration",
    description=(
        "Update one or more mutable server configuration values. Changes are applied "
        "to the running process and persisted to the database. "
        "modelsDir is read-only and cannot be changed at runtime."
    ),
    responses={400: {"description": "Invalid configuration value or read-only field"}},
)
async def patch_config(
    patch: ServerConfigPatch,
    request: Request,
    _user: UserContext = Depends(require_role("system_admin")),
) -> dict[str, Any]:
    """Patch mutable server configuration values for the running process."""
    payload = patch.model_dump(exclude_unset=True)

    async def _persist(key: str, value: object) -> None:
        async with AsyncSessionLocal() as _s:
            await save_override(_s, key, value)
            await _s.commit()

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
                    list(payload["global_schedules"]),
                )
                await session.commit()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    if "default_gpu_index" in payload:
        settings.default_gpu_index = int(payload["default_gpu_index"])
        await _persist("default_gpu_index", settings.default_gpu_index)
    if "idle_timeout_seconds" in payload:
        settings.idle_timeout_seconds = int(payload["idle_timeout_seconds"])
        await _persist("idle_timeout_seconds", settings.idle_timeout_seconds)
    if "idle_eviction_check_interval_seconds" in payload:
        settings.idle_eviction_check_interval_seconds = int(payload["idle_eviction_check_interval_seconds"])
        await _persist("idle_eviction_check_interval_seconds", settings.idle_eviction_check_interval_seconds)
    if "model_load_wait_timeout_seconds" in payload:
        settings.model_load_wait_timeout_s = int(payload["model_load_wait_timeout_seconds"])
        await _persist("model_load_wait_timeout_s", settings.model_load_wait_timeout_s)
    if "pressure_eviction_drain_timeout_seconds" in payload:
        settings.pressure_eviction_drain_timeout_s = int(payload["pressure_eviction_drain_timeout_seconds"])
        await _persist("pressure_eviction_drain_timeout_s", settings.pressure_eviction_drain_timeout_s)
    if "vram_buffer_mb" in payload:
        settings.vram_buffer_mb = int(payload["vram_buffer_mb"])
        await _persist("vram_buffer_mb", settings.vram_buffer_mb)
    if "vram_pressure_threshold_pct" in payload:
        settings.vram_pressure_threshold_pct = float(payload["vram_pressure_threshold_pct"])
        await _persist("vram_pressure_threshold_pct", settings.vram_pressure_threshold_pct)
    if "openai_audio_max_part_size_mb" in payload:
        settings.openai_audio_max_part_size_mb = int(payload["openai_audio_max_part_size_mb"])
        await _persist("openai_audio_max_part_size_mb", settings.openai_audio_max_part_size_mb)
    if "whisper_startup_timeout_seconds" in payload:
        settings.whisper_startup_timeout_s = int(payload["whisper_startup_timeout_seconds"])
        await _persist("whisper_startup_timeout_s", settings.whisper_startup_timeout_s)
    if "log_level" in payload:
        settings.log_level = str(payload["log_level"])
        await _persist("log_level", settings.log_level)
    if "litellm_base_url" in payload:
        settings.litellm_base_url = str(payload["litellm_base_url"])
        await _persist("litellm_base_url", settings.litellm_base_url)
    if "litellm_admin_key" in payload:
        new_key = str(payload["litellm_admin_key"])
        if new_key and new_key != "***":
            settings.litellm_admin_key = new_key
            await _persist("litellm_admin_key", settings.litellm_admin_key)
        elif new_key == "":
            settings.litellm_admin_key = ""
            await _persist("litellm_admin_key", "")
    if "litellm_auto_sync" in payload:
        settings.litellm_auto_sync = bool(payload["litellm_auto_sync"])
        await _persist("litellm_auto_sync", settings.litellm_auto_sync)
    if "energy_cost_eur_kwh" in payload:
        settings.energy_cost_eur_kwh = float(payload["energy_cost_eur_kwh"])
        await _persist("energy_cost_eur_kwh", settings.energy_cost_eur_kwh)
    if "vllm_gpu_memory_utilization" in payload:
        settings.vllm_gpu_memory_utilization = float(payload["vllm_gpu_memory_utilization"])
        await _persist("vllm_gpu_memory_utilization", settings.vllm_gpu_memory_utilization)
    if "vllm_max_num_seqs" in payload:
        settings.vllm_max_num_seqs = (
            int(payload["vllm_max_num_seqs"]) if payload["vllm_max_num_seqs"] is not None else None
        )
        await _persist("vllm_max_num_seqs", settings.vllm_max_num_seqs)
    if "vllm_max_num_batched_tokens" in payload:
        settings.vllm_max_num_batched_tokens = (
            int(payload["vllm_max_num_batched_tokens"])
            if payload["vllm_max_num_batched_tokens"] is not None
            else None
        )
        await _persist("vllm_max_num_batched_tokens", settings.vllm_max_num_batched_tokens)
    if "vllm_enable_prefix_caching" in payload:
        settings.vllm_enable_prefix_caching = bool(payload["vllm_enable_prefix_caching"])
        await _persist("vllm_enable_prefix_caching", settings.vllm_enable_prefix_caching)
    if "vllm_enforce_eager" in payload:
        settings.vllm_enforce_eager = bool(payload["vllm_enforce_eager"])
        await _persist("vllm_enforce_eager", settings.vllm_enforce_eager)
    if "sglang_mem_fraction_static" in payload:
        settings.sglang_mem_fraction_static = float(payload["sglang_mem_fraction_static"])
        await _persist("sglang_mem_fraction_static", settings.sglang_mem_fraction_static)
    if "sglang_context_length" in payload:
        settings.sglang_context_length = (
            int(payload["sglang_context_length"]) if payload["sglang_context_length"] is not None else None
        )
        await _persist("sglang_context_length", settings.sglang_context_length)
    if "sglang_disable_radix_cache" in payload:
        settings.sglang_disable_radix_cache = bool(payload["sglang_disable_radix_cache"])
        await _persist("sglang_disable_radix_cache", settings.sglang_disable_radix_cache)
    if "llama_cpp_gpu_layers" in payload:
        settings.llama_cpp_gpu_layers = int(payload["llama_cpp_gpu_layers"])
        await _persist("llama_cpp_gpu_layers", settings.llama_cpp_gpu_layers)
    if "llama_cpp_ctx_size" in payload:
        settings.llama_cpp_ctx_size = int(payload["llama_cpp_ctx_size"])
        await _persist("llama_cpp_ctx_size", settings.llama_cpp_ctx_size)
    if "llama_cpp_flash_attn" in payload:
        settings.llama_cpp_flash_attn = bool(payload["llama_cpp_flash_attn"])
        await _persist("llama_cpp_flash_attn", settings.llama_cpp_flash_attn)
    if "bitnet_gpu_layers" in payload:
        settings.bitnet_gpu_layers = int(payload["bitnet_gpu_layers"])
        await _persist("bitnet_gpu_layers", settings.bitnet_gpu_layers)
    if "bitnet_ctx_size" in payload:
        settings.bitnet_ctx_size = int(payload["bitnet_ctx_size"])
        await _persist("bitnet_ctx_size", settings.bitnet_ctx_size)
    if "bitnet_flash_attn" in payload:
        settings.bitnet_flash_attn = bool(payload["bitnet_flash_attn"])
        await _persist("bitnet_flash_attn", settings.bitnet_flash_attn)
    if "diffusers_torch_dtype" in payload:
        settings.diffusers_torch_dtype = str(payload["diffusers_torch_dtype"])
        await _persist("diffusers_torch_dtype", settings.diffusers_torch_dtype)
    if "diffusers_offload_mode" in payload:
        settings.diffusers_offload_mode = str(payload["diffusers_offload_mode"])
        await _persist("diffusers_offload_mode", settings.diffusers_offload_mode)
    if "diffusers_enable_torch_compile" in payload:
        settings.diffusers_enable_torch_compile = bool(payload["diffusers_enable_torch_compile"])
        await _persist("diffusers_enable_torch_compile", settings.diffusers_enable_torch_compile)
    if "diffusers_enable_xformers" in payload:
        settings.diffusers_enable_xformers = bool(payload["diffusers_enable_xformers"])
        await _persist("diffusers_enable_xformers", settings.diffusers_enable_xformers)
    if "diffusers_allow_tf32" in payload:
        settings.diffusers_allow_tf32 = bool(payload["diffusers_allow_tf32"])
        await _persist("diffusers_allow_tf32", settings.diffusers_allow_tf32)
    if "tensorrt_llm_enabled" in payload:
        settings.tensorrt_llm_enabled = bool(payload["tensorrt_llm_enabled"])
        await _persist("tensorrt_llm_enabled", settings.tensorrt_llm_enabled)
    if "tensorrt_llm_max_batch_size" in payload:
        settings.tensorrt_llm_max_batch_size = (
            int(payload["tensorrt_llm_max_batch_size"])
            if payload["tensorrt_llm_max_batch_size"] is not None
            else None
        )
        await _persist("tensorrt_llm_max_batch_size", settings.tensorrt_llm_max_batch_size)
    if "tensorrt_llm_context_length" in payload:
        settings.tensorrt_llm_context_length = (
            int(payload["tensorrt_llm_context_length"])
            if payload["tensorrt_llm_context_length"] is not None
            else None
        )
        await _persist("tensorrt_llm_context_length", settings.tensorrt_llm_context_length)

    if "require_api_key_openai" in payload:
        settings.require_api_key_openai = bool(payload["require_api_key_openai"])
        await _persist("require_api_key_openai", settings.require_api_key_openai)
    if "require_api_key_ollama" in payload:
        settings.require_api_key_ollama = bool(payload["require_api_key_ollama"])
        await _persist("require_api_key_ollama", settings.require_api_key_ollama)
    if "realtime_default_stt_model" in payload:
        settings.realtime_default_stt_model = str(payload["realtime_default_stt_model"] or "")
        await _persist("realtime_default_stt_model", settings.realtime_default_stt_model)
    if "realtime_default_tts_model" in payload:
        settings.realtime_default_tts_model = str(payload["realtime_default_tts_model"] or "")
        await _persist("realtime_default_tts_model", settings.realtime_default_tts_model)

    if "federation_node_name" in payload:
        settings.federation_node_name = str(payload["federation_node_name"] or "")
        await _persist("federation_node_name", settings.federation_node_name)

    if "federation_enabled" in payload:
        new_val = bool(payload["federation_enabled"])
        old_val = settings.federation_enabled
        settings.federation_enabled = new_val
        await _persist("federation_enabled", new_val)

        if new_val and not old_val:
            # Start federation manager on the fly
            fm = getattr(request.app.state, "federation_manager", None)
            if fm is None:
                from ocabra.core.federation import FederationManager
                from ocabra.database import AsyncSessionLocal as _ASL

                fm = FederationManager(settings, _ASL)
                await fm.start()
                request.app.state.federation_manager = fm
        elif not new_val and old_val:
            # Stop federation manager
            fm = getattr(request.app.state, "federation_manager", None)
            if fm is not None:
                await fm.stop()
                request.app.state.federation_manager = None

    if "download_dir" in payload:
        settings.download_dir = str(payload["download_dir"])
        await _persist("download_dir", settings.download_dir)
    if "max_temperature_c" in payload:
        settings.max_temperature_c = int(payload["max_temperature_c"])
        await _persist("max_temperature_c", settings.max_temperature_c)
        if hasattr(request.app.state, "gpu_manager"):
            request.app.state.gpu_manager.max_temperature_c = settings.max_temperature_c

    # ── Generation services: persist setting + propagate to live ServiceState ──
    # service_id (in ServiceManager) → settings prefix
    _SERVICE_PREFIXES = (
        ("hunyuan", "hunyuan"),
        ("comfyui", "comfyui"),
        ("a1111",   "a1111"),
        ("acestep", "acestep"),
        ("unsloth", "unsloth"),
    )
    sm = getattr(request.app.state, "service_manager", None)
    for service_id, prefix in _SERVICE_PREFIXES:
        live: dict[str, int] = {}
        for short, suffix, kw in (
            ("idle",  "idle_unload_seconds",        "idle_unload_after_seconds"),
            ("grace", "generation_grace_period_s",  "generation_grace_period_s"),
            ("gpu",   "preferred_gpu",              "preferred_gpu"),
        ):
            key = f"{prefix}_{suffix}"
            if key in payload:
                value = int(payload[key])
                setattr(settings, key, value)
                await _persist(key, value)
                live[kw] = value
        if live and sm is not None:
            try:
                await sm.update_runtime_config(service_id, **live)
            except KeyError:
                # Service not registered (shouldn't happen) — settings still persisted.
                logger.warning("service_runtime_config_missing", service_id=service_id)

    if "generation_gpu_util_threshold_pct" in payload:
        settings.generation_gpu_util_threshold_pct = int(payload["generation_gpu_util_threshold_pct"])
        await _persist(
            "generation_gpu_util_threshold_pct",
            settings.generation_gpu_util_threshold_pct,
        )

    response = _build_config_response(request)
    response["globalSchedules"] = await _load_global_schedules()
    return response


@router.post(
    "/config/litellm/sync",
    summary="Sync models to LiteLLM proxy",
    description="Manually trigger synchronisation of all loaded models to the configured LiteLLM proxy instance.",
)
async def sync_litellm(
    request: Request,
    _user: UserContext = Depends(require_role("system_admin")),
) -> JSONResponse:
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
