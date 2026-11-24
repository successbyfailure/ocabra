// Cliente tipado de /ocabra/* — dominio «system» (REF-15).
// Extraído de api/client.ts; usa la capa HTTP única de api/http.ts.

import { request } from "@/api/http"
import { isRecord } from "@/api/_helpers"
import type {
  ServerConfig,
} from "@/types"

export function toHostStats(raw: unknown): import("@/types").HostStats {
  const d = isRecord(raw) ? raw : {};
  const powerMax = d.cpu_power_max_w ?? d.cpuPowerMaxW
  return {
    cpuModel: (d.cpu_model ?? d.cpuModel ?? null) as string | null,
    cpuPowerMaxW: powerMax == null ? null : Number(powerMax),
    cpuPowerMaxSource: (d.cpu_power_max_source ?? d.cpuPowerMaxSource ?? null) as "config" | "rapl" | "tdp" | null,
    cpuPct: Number(d.cpu_pct ?? d.cpuPct ?? 0),
    cpuCount: Number(d.cpu_count ?? d.cpuCount ?? 0),
    cpuCountPhysical: Number(d.cpu_count_physical ?? d.cpuCountPhysical ?? 0),
    loadAvg1m: Number(d.load_avg_1m ?? d.loadAvg1m ?? 0),
    loadAvg5m: Number(d.load_avg_5m ?? d.loadAvg5m ?? 0),
    loadAvg15m: Number(d.load_avg_15m ?? d.loadAvg15m ?? 0),
    memTotalMb: Number(d.mem_total_mb ?? d.memTotalMb ?? 0),
    memUsedMb: Number(d.mem_used_mb ?? d.memUsedMb ?? 0),
    memFreeMb: Number(d.mem_free_mb ?? d.memFreeMb ?? 0),
    memPct: Number(d.mem_pct ?? d.memPct ?? 0),
    swapTotalMb: Number(d.swap_total_mb ?? d.swapTotalMb ?? 0),
    swapUsedMb: Number(d.swap_used_mb ?? d.swapUsedMb ?? 0),
    swapPct: Number(d.swap_pct ?? d.swapPct ?? 0),
  };
}


export function toServerConfig(raw: unknown): ServerConfig {
  const data = isRecord(raw) ? raw : {};
  const globalSchedulesRaw = data.globalSchedules;
  const globalSchedules = Array.isArray(globalSchedulesRaw)
    ? globalSchedulesRaw.map((schedule, idx) => {
        const s = isRecord(schedule) ? schedule : {};
        return {
          id: String(s.id ?? `global-schedule-${idx}`),
          days: Array.isArray(s.days)
            ? s.days.map((d) => Number(d)).filter((d) => Number.isFinite(d))
            : [],
          start: String(s.start ?? "00:00"),
          end: String(s.end ?? "00:00"),
          enabled: Boolean(s.enabled ?? true),
        };
      })
    : [];

  return {
    defaultGpuIndex: Number(data.defaultGpuIndex ?? 0),
    idleTimeoutSeconds: Number(data.idleTimeoutSeconds ?? 0),
    idleEvictionCheckIntervalSeconds: Number(
      data.idleEvictionCheckIntervalSeconds ??
        data.idle_eviction_check_interval_seconds ??
        15,
    ),
    modelLoadWaitTimeoutSeconds: Number(
      data.modelLoadWaitTimeoutSeconds ??
        data.model_load_wait_timeout_seconds ??
        720,
    ),
    pressureEvictionDrainTimeoutSeconds: Number(
      data.pressureEvictionDrainTimeoutSeconds ??
        data.pressure_eviction_drain_timeout_seconds ??
        60,
    ),
    evictionFollowupGraceSeconds: Number(data.evictionFollowupGraceSeconds ?? 4),
    evictionFollowupMaxChainSeconds: Number(data.evictionFollowupMaxChainSeconds ?? 60),
    pinReloadPollIntervalSeconds: Number(
      data.pinReloadPollIntervalSeconds ??
        data.pin_reload_poll_interval_seconds ??
        120,
    ),
    vramBufferMb: Number(data.vramBufferMb ?? 0),
    vramPressureThresholdPct: Number(data.vramPressureThresholdPct ?? 0),
    maxInflightPerModel: Number(data.maxInflightPerModel ?? 0),
    openaiAudioMaxPartSizeMb: Number(
      data.openaiAudioMaxPartSizeMb ??
        data.openai_audio_max_part_size_mb ??
        256,
    ),
    whisperStartupTimeoutSeconds: Number(
      data.whisperStartupTimeoutSeconds ??
        data.whisper_startup_timeout_seconds ??
        300,
    ),
    logLevel: String(data.logLevel ?? "info"),
    energyCostEurKwh: Number(data.energyCostEurKwh ?? 0),
    modelsDir: String(data.modelsDir ?? "/models"),
    downloadDir: String(data.downloadDir ?? "/models/downloads"),
    maxTemperatureC: Number(data.maxTemperatureC ?? 88),
    gpuThrottleEnabled: Boolean(data.gpuThrottleEnabled),
    gpuThrottlePowerReductionPct: Number(
      data.gpuThrottlePowerReductionPct ?? 20,
    ),
    gpuThrottleCooldownS: Number(data.gpuThrottleCooldownS ?? 300),
    gpuThrottleRecoveryMarginC: Number(data.gpuThrottleRecoveryMarginC ?? 5),
    vllmGpuMemoryUtilization: Number(
      data.vllmGpuMemoryUtilization ?? data.vllm_gpu_memory_utilization ?? 0.85,
    ),
    vllmMaxNumSeqs:
      data.vllmMaxNumSeqs == null && data.vllm_max_num_seqs == null
        ? null
        : Number(data.vllmMaxNumSeqs ?? data.vllm_max_num_seqs),
    vllmMaxNumBatchedTokens:
      data.vllmMaxNumBatchedTokens == null &&
      data.vllm_max_num_batched_tokens == null
        ? null
        : Number(
            data.vllmMaxNumBatchedTokens ?? data.vllm_max_num_batched_tokens,
          ),
    vllmEnablePrefixCaching: Boolean(
      data.vllmEnablePrefixCaching ?? data.vllm_enable_prefix_caching ?? true,
    ),
    vllmEnforceEager: Boolean(
      data.vllmEnforceEager ?? data.vllm_enforce_eager ?? true,
    ),
    sglangMemFractionStatic: Number(
      data.sglangMemFractionStatic ?? data.sglang_mem_fraction_static ?? 0.9,
    ),
    sglangContextLength:
      data.sglangContextLength == null && data.sglang_context_length == null
        ? null
        : Number(data.sglangContextLength ?? data.sglang_context_length),
    sglangDisableRadixCache: Boolean(
      data.sglangDisableRadixCache ?? data.sglang_disable_radix_cache ?? false,
    ),
    llamaCppGpuLayers: Number(
      data.llamaCppGpuLayers ?? data.llama_cpp_gpu_layers ?? 0,
    ),
    llamaCppCtxSize: Number(
      data.llamaCppCtxSize ?? data.llama_cpp_ctx_size ?? 4096,
    ),
    llamaCppFlashAttn: Boolean(
      data.llamaCppFlashAttn ?? data.llama_cpp_flash_attn ?? false,
    ),
    bitnetGpuLayers: Number(
      data.bitnetGpuLayers ?? data.bitnet_gpu_layers ?? 0,
    ),
    bitnetCtxSize: Number(data.bitnetCtxSize ?? data.bitnet_ctx_size ?? 4096),
    bitnetFlashAttn: Boolean(
      data.bitnetFlashAttn ?? data.bitnet_flash_attn ?? false,
    ),
    diffusersTorchDtype: String(
      data.diffusersTorchDtype ?? data.diffusers_torch_dtype ?? "auto",
    ),
    diffusersOffloadMode: String(
      data.diffusersOffloadMode ?? data.diffusers_offload_mode ?? "none",
    ),
    diffusersEnableTorchCompile: Boolean(
      data.diffusersEnableTorchCompile ??
      data.diffusers_enable_torch_compile ??
      false,
    ),
    diffusersEnableXformers: Boolean(
      data.diffusersEnableXformers ?? data.diffusers_enable_xformers ?? false,
    ),
    diffusersAllowTf32: Boolean(
      data.diffusersAllowTf32 ?? data.diffusers_allow_tf32 ?? true,
    ),
    tensorrtLlmEnabled: Boolean(
      data.tensorrtLlmEnabled ?? data.tensorrt_llm_enabled ?? false,
    ),
    tensorrtLlmMaxBatchSize:
      data.tensorrtLlmMaxBatchSize == null &&
      data.tensorrt_llm_max_batch_size == null
        ? null
        : Number(
            data.tensorrtLlmMaxBatchSize ?? data.tensorrt_llm_max_batch_size,
          ),
    tensorrtLlmContextLength:
      data.tensorrtLlmContextLength == null &&
      data.tensorrt_llm_context_length == null
        ? null
        : Number(
            data.tensorrtLlmContextLength ?? data.tensorrt_llm_context_length,
          ),
    globalSchedules,
    requireApiKeyOpenai: Boolean(
      data.requireApiKeyOpenai ?? data.require_api_key_openai ?? true,
    ),
    requireApiKeyOllama: Boolean(
      data.requireApiKeyOllama ?? data.require_api_key_ollama ?? true,
    ),
    ollamaNumParallel: Number(data.ollamaNumParallel ?? data.ollama_num_parallel ?? 2),
    ollamaMaxContext: Number(data.ollamaMaxContext ?? data.ollama_max_context ?? 65536),
    newUserGroupNames: Array.isArray(
      data.newUserGroupNames ?? data.new_user_group_names,
    )
      ? ((data.newUserGroupNames ?? data.new_user_group_names) as unknown[]).map(String)
      : ["default", "Makers"],
    defaultReasoningEffort: (String(
      data.defaultReasoningEffort ?? data.default_reasoning_effort ?? "medium",
    ) as ServerConfig["defaultReasoningEffort"]),
    realtimeDefaultSttModel: String(
      data.realtimeDefaultSttModel ?? data.realtime_default_stt_model ?? "",
    ),
    realtimeDefaultTtsModel: String(
      data.realtimeDefaultTtsModel ?? data.realtime_default_tts_model ?? "",
    ),
    federationEnabled: Boolean(
      data.federationEnabled ?? data.federation_enabled ?? false,
    ),
    federationAcceptJoinRequests: Boolean(
      data.federationAcceptJoinRequests ?? data.federation_accept_join_requests ?? false,
    ),
    privacyMode:
      (data.privacyMode ?? data.privacy_mode) === "zero_data" ? "zero_data" : "functional",
    privacyTempDir: (isRecord(data.privacyTempDir) ? data.privacyTempDir : null) as
      ServerConfig["privacyTempDir"],
    openaiFilesRetentionHours: Number(
      data.openaiFilesRetentionHours ?? data.openai_files_retention_hours ?? 168,
    ),
    federationNodeName: String(
      data.federationNodeName ?? data.federation_node_name ?? "",
    ),
    batchPriorityEnabled: Boolean(
      data.batchPriorityEnabled ?? data.batch_priority_enabled ?? true,
    ),
    batchPreemptMode: String(
      data.batchPreemptMode ?? data.batch_preempt_mode ?? "conditional",
    ) as ServerConfig["batchPreemptMode"],
    batchPreemptMinElapsedS: Number(
      data.batchPreemptMinElapsedS ?? data.batch_preempt_min_elapsed_s ?? 300,
    ),
    batchPreemptGraceS: Number(
      data.batchPreemptGraceS ?? data.batch_preempt_grace_s ?? 10,
    ),
    batchFreeSlotsPct: Number(
      data.batchFreeSlotsPct ?? data.batch_free_slots_pct ?? 1,
    ),
    batchResumeCooldownSlotsS: Number(
      data.batchResumeCooldownSlotsS ??
        data.batch_resume_cooldown_slots_s ??
        120,
    ),
    batchResumeCooldownEvictionS: Number(
      data.batchResumeCooldownEvictionS ??
        data.batch_resume_cooldown_eviction_s ??
        300,
    ),
    batchInterJobCooldownS: Number(
      data.batchInterJobCooldownS ?? data.batch_inter_job_cooldown_s ?? 0,
    ),
    trellis2IdleUnloadSeconds: Number(
      data.trellis2IdleUnloadSeconds ??
        data.trellis2_idle_unload_seconds ??
        300,
    ),
    trellis2GenerationGracePeriodS: Number(
      data.trellis2GenerationGracePeriodS ??
        data.trellis2_generation_grace_period_s ??
        120,
    ),
    trellis2PreferredGpu: Number(
      data.trellis2PreferredGpu ?? data.trellis2_preferred_gpu ?? 1,
    ),
    comfyuiIdleUnloadSeconds: Number(
      data.comfyuiIdleUnloadSeconds ?? data.comfyui_idle_unload_seconds ?? 600,
    ),
    comfyuiGenerationGracePeriodS: Number(
      data.comfyuiGenerationGracePeriodS ??
        data.comfyui_generation_grace_period_s ??
        120,
    ),
    comfyuiPreferredGpu: Number(
      data.comfyuiPreferredGpu ?? data.comfyui_preferred_gpu ?? 1,
    ),
    a1111IdleUnloadSeconds: Number(
      data.a1111IdleUnloadSeconds ?? data.a1111_idle_unload_seconds ?? 600,
    ),
    a1111GenerationGracePeriodS: Number(
      data.a1111GenerationGracePeriodS ??
        data.a1111_generation_grace_period_s ??
        120,
    ),
    a1111PreferredGpu: Number(
      data.a1111PreferredGpu ?? data.a1111_preferred_gpu ?? 1,
    ),
    acestepIdleUnloadSeconds: Number(
      data.acestepIdleUnloadSeconds ?? data.acestep_idle_unload_seconds ?? 1800,
    ),
    acestepGenerationGracePeriodS: Number(
      data.acestepGenerationGracePeriodS ??
        data.acestep_generation_grace_period_s ??
        180,
    ),
    acestepPreferredGpu: Number(
      data.acestepPreferredGpu ?? data.acestep_preferred_gpu ?? 1,
    ),
    unslothIdleUnloadSeconds: Number(
      data.unslothIdleUnloadSeconds ?? data.unsloth_idle_unload_seconds ?? 0,
    ),
    unslothGenerationGracePeriodS: Number(
      data.unslothGenerationGracePeriodS ??
        data.unsloth_generation_grace_period_s ??
        -1,
    ),
    unslothPreferredGpu: Number(
      data.unslothPreferredGpu ?? data.unsloth_preferred_gpu ?? 1,
    ),
    generationGpuUtilThresholdPct: Number(
      data.generationGpuUtilThresholdPct ??
        data.generation_gpu_util_threshold_pct ??
        50,
    ),
    routingEnabled:
      data.routingEnabled == null ? undefined : Boolean(data.routingEnabled),
    routerConfidenceFloor:
      data.routerConfidenceFloor == null
        ? undefined
        : Number(data.routerConfidenceFloor),
    maxDrainTimeoutS:
      data.maxDrainTimeoutS == null ? undefined : Number(data.maxDrainTimeoutS),
    sessionPauseThresholdS:
      data.sessionPauseThresholdS == null
        ? undefined
        : Number(data.sessionPauseThresholdS),
    sessionMaxIdleS:
      data.sessionMaxIdleS == null ? undefined : Number(data.sessionMaxIdleS),
  };
}


export function toDurationEstimate(raw: unknown): DurationEstimateResult {
  const d = (raw ?? {}) as Record<string, unknown>;
  return {
    expectedSeconds: Number(d.expected_seconds ?? 0),
    p50Seconds: Number(d.p50_seconds ?? 0),
    p95Seconds: Number(d.p95_seconds ?? 0),
    confidence: Number(d.confidence ?? 0),
    source: String(
      d.source ?? "family_default",
    ) as DurationEstimateResult["source"],
    sampleCount: Number(d.sample_count ?? 0),
    coldStartSeconds: Number(d.cold_start_seconds ?? 0),
  };
}


export function toRealtimeSession(raw: unknown): RealtimeSessionInfo {
  const d = (raw ?? {}) as Record<string, unknown>;
  return {
    sessionId: String(d.session_id ?? ""),
    userId: d.user_id == null ? null : String(d.user_id),
    apiKeyName: d.api_key_name == null ? null : String(d.api_key_name),
    workersHeld: Array.isArray(d.workers_held)
      ? d.workers_held.map(String)
      : [],
    startedAt: String(d.started_at ?? ""),
    lastActivityAt: String(d.last_activity_at ?? ""),
    isPaused: Boolean(d.is_paused),
    isZombie: Boolean(d.is_zombie),
  };
}


export type DurationEstimateResult = {
  expectedSeconds: number;
  p50Seconds: number;
  p95Seconds: number;
  confidence: number;
  source: "regression" | "percentile" | "family_default";
  sampleCount: number;
  coldStartSeconds: number;
};


export type RealtimeSessionInfo = {
  sessionId: string;
  userId: string | null;
  apiKeyName: string | null;
  workersHeld: string[];
  startedAt: string;
  lastActivityAt: string;
  isPaused: boolean;
  isZombie: boolean;
};


export const systemApiModule = {
  estimate: {
    // Bloque 20 — DurationEstimator entrypoint. Returns expected duration
    // plus confidence for the given model + request body. Callers that get
    // ``confidence < 0.3`` should treat the number as a rough hint and
    // prefer ``p95Seconds`` for any deadline-driven UI.
    predict: async (model: string, body: Record<string, unknown>) =>
      toDurationEstimate(
        await request<unknown>("POST", "/ocabra/estimate", { model, body }),
      ),
  },
  sessions: {
    // Live Realtime sessions holding worker models. Feeds the Sessions
    // admin page and any "who's using this model?" tooltips.
    list: async () =>
      (await request<unknown[]>("GET", "/ocabra/sessions")).map(
        toRealtimeSession,
      ),
    kill: (sessionId: string) =>
      request<{ session_id: string; killed: boolean }>(
        "POST",
        `/ocabra/sessions/${encodeURIComponent(sessionId)}/kill`,
      ),
  },
  host: {
    stats: async () =>
      toHostStats(await request<unknown>("GET", "/ocabra/host/stats")),
  },

  config: {
    get: async () =>
      toServerConfig(await request<unknown>("GET", "/ocabra/config")),
    patch: async (patch: Partial<ServerConfig>) =>
      toServerConfig(await request<unknown>("PATCH", "/ocabra/config", patch)),
  },

  ollama: {
    version: async (): Promise<{
      current: string | null;
      latest: string | null;
      updateAvailable: boolean;
    }> => {
      const raw = await request<{
        current: string | null;
        latest: string | null;
        update_available: boolean;
      }>("GET", "/ocabra/ollama/version");
      return {
        current: raw.current,
        latest: raw.latest,
        updateAvailable: Boolean(raw.update_available),
      };
    },
    serverUpdateStatus: async (): Promise<{
      status: "idle" | "pulling" | "restarting" | "done" | "error";
      detail: string | null;
      startedAt: string | null;
      finishedAt: string | null;
      fromVersion: string | null;
      toVersion: string | null;
    }> => {
      const raw = await request<{
        status: "idle" | "pulling" | "restarting" | "done" | "error";
        detail: string | null;
        started_at: string | null;
        finished_at: string | null;
        from_version: string | null;
        to_version: string | null;
      }>("GET", "/ocabra/ollama/server/update");
      return {
        status: raw.status,
        detail: raw.detail,
        startedAt: raw.started_at,
        finishedAt: raw.finished_at,
        fromVersion: raw.from_version,
        toVersion: raw.to_version,
      };
    },
    startServerUpdate: async () =>
      request<{ status: string; detail: string | null }>(
        "POST",
        "/ocabra/ollama/server/update",
      ),
  },

};
