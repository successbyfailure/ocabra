// Typed API client for oCabra internal API (/ocabra/*)
// Stream 1-D fills in the method implementations.
// Streams 4-A/B/C/D consume this client.

import type {
  GPUState,
  GPUStatHistory,
  ModelState,
  ModelPatchRequest,
  DownloadJob,
  DownloadSource,
  HFModelCard,
  HFModelDetail,
  HFModelVariant,
  OllamaModelCard,
  OllamaModelVariant,
  LocalModel,
  ServerConfig,
  RequestStats,
  EnergyStats,
  PerformanceStats,
  TokenStats,
  OverviewStats,
  StatsParams,
  ServiceState,
  CompileJob,
} from "@/types"

const BASE = ""

type AnyRecord = Record<string, unknown>

function isRecord(value: unknown): value is AnyRecord {
  return typeof value === "object" && value !== null
}

function toVLLMConfig(raw: unknown): ModelState["extraConfig"] {
  const data = isRecord(raw) ? raw : {}
  const vllmRaw = data.vllm
  const vllm = isRecord(vllmRaw) ? vllmRaw : {}

  return {
    ...data,
    vllm: {
      recipeId: (vllm.recipe_id ?? vllm.recipeId ?? null) as string | null,
      recipeNotes: Array.isArray(vllm.recipe_notes ?? vllm.recipeNotes)
        ? ((vllm.recipe_notes ?? vllm.recipeNotes) as unknown[]).map(String)
        : [],
      recipeModelImpl: (vllm.recipe_model_impl ?? vllm.recipeModelImpl ?? null) as
        | "auto"
        | "vllm"
        | "transformers"
        | null,
      recipeRunner: (vllm.recipe_runner ?? vllm.recipeRunner ?? null) as
        | "generate"
        | "pooling"
        | null,
      suggestedConfig: isRecord(vllm.suggested_config ?? vllm.suggestedConfig)
        ? ((vllm.suggested_config ?? vllm.suggestedConfig) as Record<string, unknown>)
        : {},
      suggestedTuning: isRecord(vllm.suggested_tuning ?? vllm.suggestedTuning)
        ? ((vllm.suggested_tuning ?? vllm.suggestedTuning) as Record<string, unknown>)
        : {},
      probeStatus: (vllm.probe_status ?? vllm.probeStatus ?? null) as import("@/types").HFVLLMRuntimeProbe["status"] | null,
      probeReason: (vllm.probe_reason ?? vllm.probeReason ?? null) as string | null,
      probeObservedAt: (vllm.probe_observed_at ?? vllm.probeObservedAt ?? null) as string | null,
      probeRecommendedModelImpl:
        (vllm.probe_recommended_model_impl ?? vllm.probeRecommendedModelImpl ?? null) as
          | "auto"
          | "vllm"
          | "transformers"
          | null,
      probeRecommendedRunner:
        (vllm.probe_recommended_runner ?? vllm.probeRecommendedRunner ?? null) as
          | "generate"
          | "pooling"
          | null,
      modelImpl: (vllm.model_impl ?? vllm.modelImpl ?? null) as string | null,
      runner: (vllm.runner ?? null) as string | null,
      hfOverrides: (vllm.hf_overrides ?? vllm.hfOverrides ?? null) as string | Record<string, unknown> | null,
      chatTemplate: (vllm.chat_template ?? vllm.chatTemplate ?? null) as string | null,
      chatTemplateContentFormat:
        (vllm.chat_template_content_format ?? vllm.chatTemplateContentFormat ?? null) as string | null,
      generationConfig: (vllm.generation_config ?? vllm.generationConfig ?? null) as string | null,
      overrideGenerationConfig:
        (vllm.override_generation_config ?? vllm.overrideGenerationConfig ?? null) as
          | string
          | Record<string, unknown>
          | null,
      toolCallParser: (vllm.tool_call_parser ?? vllm.toolCallParser ?? null) as string | null,
      toolParserPlugin: (vllm.tool_parser_plugin ?? vllm.toolParserPlugin ?? null) as string | null,
      reasoningParser: (vllm.reasoning_parser ?? vllm.reasoningParser ?? null) as string | null,
      languageModelOnly:
        vllm.language_model_only == null && vllm.languageModelOnly == null
          ? null
          : Boolean(vllm.language_model_only ?? vllm.languageModelOnly),
      maxNumSeqs:
        vllm.max_num_seqs == null && vllm.maxNumSeqs == null
          ? null
          : Number(vllm.max_num_seqs ?? vllm.maxNumSeqs ?? 0),
      maxNumBatchedTokens:
        vllm.max_num_batched_tokens == null && vllm.maxNumBatchedTokens == null
          ? null
          : Number(vllm.max_num_batched_tokens ?? vllm.maxNumBatchedTokens ?? 0),
      tensorParallelSize:
        vllm.tensor_parallel_size == null && vllm.tensorParallelSize == null
          ? null
          : Number(vllm.tensor_parallel_size ?? vllm.tensorParallelSize ?? 0),
      maxModelLen:
        vllm.max_model_len == null && vllm.maxModelLen == null
          ? null
          : Number(vllm.max_model_len ?? vllm.maxModelLen ?? 0),
      gpuMemoryUtilization:
        vllm.gpu_memory_utilization == null && vllm.gpuMemoryUtilization == null
          ? null
          : Number(vllm.gpu_memory_utilization ?? vllm.gpuMemoryUtilization ?? 0),
      enablePrefixCaching: Boolean(vllm.enable_prefix_caching ?? vllm.enablePrefixCaching),
      enableChunkedPrefill:
        vllm.enable_chunked_prefill == null && vllm.enableChunkedPrefill == null
          ? null
          : Boolean(vllm.enable_chunked_prefill ?? vllm.enableChunkedPrefill),
      enforceEager: Boolean(vllm.enforce_eager ?? vllm.enforceEager),
      trustRemoteCode: Boolean(vllm.trust_remote_code ?? vllm.trustRemoteCode),
      swapSpace:
        vllm.swap_space == null && vllm.swapSpace == null
          ? null
          : Number(vllm.swap_space ?? vllm.swapSpace ?? 0),
      kvCacheDtype: (vllm.kv_cache_dtype ?? vllm.kvCacheDtype ?? null) as string | null,
    },
  }
}

function toHFVLLMSupport(raw: unknown): HFModelCard["vllmSupport"] {
  const data = isRecord(raw) ? raw : null
  if (!data) return null
  const probeRaw = data.runtime_probe ?? data.runtimeProbe
  const probe = isRecord(probeRaw)
    ? {
        status: String(probeRaw.status ?? "unknown") as
          | "supported_native"
          | "supported_transformers_backend"
          | "supported_pooling"
          | "needs_remote_code"
          | "missing_chat_template"
          | "missing_tool_parser"
          | "missing_reasoning_parser"
          | "needs_hf_overrides"
          | "unsupported_tokenizer"
          | "unsupported_architecture"
          | "unavailable"
          | "unknown",
        reason: (probeRaw.reason ?? null) as string | null,
        recommendedModelImpl: (probeRaw.recommended_model_impl ?? probeRaw.recommendedModelImpl ?? null) as
          | "auto"
          | "vllm"
          | "transformers"
          | null,
        recommendedRunner: (probeRaw.recommended_runner ?? probeRaw.recommendedRunner ?? null) as
          | "generate"
          | "pooling"
          | null,
        tokenizerLoad:
          probeRaw.tokenizer_load == null && probeRaw.tokenizerLoad == null
            ? null
            : Boolean(probeRaw.tokenizer_load ?? probeRaw.tokenizerLoad),
        configLoad:
          probeRaw.config_load == null && probeRaw.configLoad == null
            ? null
            : Boolean(probeRaw.config_load ?? probeRaw.configLoad),
        observedAt: (probeRaw.observed_at ?? probeRaw.observedAt ?? null) as string | null,
      }
    : null

  return {
    classification: String(data.classification ?? "unknown") as
      | "native_vllm"
      | "transformers_backend"
      | "pooling"
      | "unsupported"
      | "unknown",
    label: String(data.label ?? "unknown"),
    modelImpl: (data.model_impl ?? data.modelImpl ?? null) as "auto" | "vllm" | "transformers" | null,
    runner: (data.runner ?? null) as "generate" | "pooling" | null,
    taskMode: (data.task_mode ?? data.taskMode ?? null) as
      | "generate"
      | "multimodal_generate"
      | "pooling"
      | "multimodal_pooling"
      | null,
    requiredOverrides: Array.isArray(data.required_overrides ?? data.requiredOverrides)
      ? ((data.required_overrides ?? data.requiredOverrides) as unknown[]).map(String)
      : [],
    recipeId: (data.recipe_id ?? data.recipeId ?? null) as string | null,
    recipeNotes: Array.isArray(data.recipe_notes ?? data.recipeNotes)
      ? ((data.recipe_notes ?? data.recipeNotes) as unknown[]).map(String)
      : [],
    recipeModelImpl: (data.recipe_model_impl ?? data.recipeModelImpl ?? null) as
      | "auto"
      | "vllm"
      | "transformers"
      | null,
    recipeRunner: (data.recipe_runner ?? data.recipeRunner ?? null) as
      | "generate"
      | "pooling"
      | null,
    suggestedConfig:
      isRecord(data.suggested_config ?? data.suggestedConfig)
        ? ((data.suggested_config ?? data.suggestedConfig) as Record<string, unknown>)
        : {},
    suggestedTuning:
      isRecord(data.suggested_tuning ?? data.suggestedTuning)
        ? ((data.suggested_tuning ?? data.suggestedTuning) as Record<string, unknown>)
        : {},
    runtimeProbe: probe,
  }
}

function toModelCapabilities(raw: unknown): ModelState["capabilities"] {
  const data = isRecord(raw) ? raw : {}
  return {
    chat: Boolean(data.chat),
    completion: Boolean(data.completion),
    tools: Boolean(data.tools),
    vision: Boolean(data.vision),
    embeddings: Boolean(data.embeddings),
    pooling: Boolean(data.pooling),
    rerank: Boolean(data.rerank),
    classification: Boolean(data.classification),
    score: Boolean(data.score),
    reasoning: Boolean(data.reasoning),
    imageGeneration: Boolean(data.image_generation ?? data.imageGeneration),
    audioTranscription: Boolean(data.audio_transcription ?? data.audioTranscription),
    tts: Boolean(data.tts),
    streaming: Boolean(data.streaming),
    contextLength: Number(data.context_length ?? data.contextLength ?? 0),
  }
}

function toGpuState(raw: unknown): GPUState {
  const data = isRecord(raw) ? raw : {}
  return {
    index: Number(data.index ?? 0),
    name: String(data.name ?? "GPU"),
    totalVramMb: Number(data.total_vram_mb ?? data.totalVramMb ?? 0),
    freeVramMb: Number(data.free_vram_mb ?? data.freeVramMb ?? 0),
    usedVramMb: Number(data.used_vram_mb ?? data.usedVramMb ?? 0),
    utilizationPct: Number(data.utilization_pct ?? data.utilizationPct ?? 0),
    temperatureC: Number(data.temperature_c ?? data.temperatureC ?? 0),
    powerDrawW: Number(data.power_draw_w ?? data.powerDrawW ?? 0),
    powerLimitW: Number(data.power_limit_w ?? data.powerLimitW ?? 0),
    lockedVramMb: Number(data.locked_vram_mb ?? data.lockedVramMb ?? 0),
    processes: Array.isArray(data.processes)
      ? data.processes.map((processRaw) => {
          const process = isRecord(processRaw) ? processRaw : {}
          return {
            pid: Number(process.pid ?? 0),
            processName: (process.process_name ?? process.processName ?? null) as string | null,
            processType: String(process.process_type ?? process.processType ?? "compute") as
              | "compute"
              | "graphics",
            usedVramMb: Number(process.used_vram_mb ?? process.usedVramMb ?? 0),
          }
        })
      : [],
  }
}

function toModelState(raw: unknown): ModelState {
  const data = isRecord(raw) ? raw : {}
  const schedulesRaw = data.schedules
  const schedules = Array.isArray(schedulesRaw)
    ? schedulesRaw.map((schedule, idx) => {
        const s = isRecord(schedule) ? schedule : {}
        return {
          id: String(s.id ?? `schedule-${idx}`),
          days: Array.isArray(s.days) ? s.days.map((d) => Number(d)).filter((d) => Number.isFinite(d)) : [],
          start: String(s.start ?? "00:00"),
          end: String(s.end ?? "00:00"),
          enabled: Boolean(s.enabled ?? true),
        }
      })
    : undefined
  return {
    modelId: String(data.model_id ?? data.modelId ?? ""),
    displayName: String(data.display_name ?? data.displayName ?? data.model_id ?? data.modelId ?? "Unknown model"),
    backendType: String(data.backend_type ?? data.backendType ?? "vllm") as ModelState["backendType"],
    status: String(data.status ?? "configured") as ModelState["status"],
    loadPolicy: String(data.load_policy ?? data.loadPolicy ?? "on_demand") as ModelState["loadPolicy"],
    autoReload: Boolean(data.auto_reload ?? data.autoReload),
    preferredGpu: (data.preferred_gpu ?? data.preferredGpu ?? null) as number | null,
    currentGpu: Array.isArray(data.current_gpu ?? data.currentGpu)
      ? ((data.current_gpu ?? data.currentGpu) as unknown[]).map((gpu) => Number(gpu))
      : [],
    vramUsedMb: Number(data.vram_used_mb ?? data.vramUsedMb ?? 0),
    diskSizeBytes:
      data.disk_size_bytes == null && data.diskSizeBytes == null
        ? null
        : Number(data.disk_size_bytes ?? data.diskSizeBytes ?? 0),
    capabilities: toModelCapabilities(data.capabilities),
    lastRequestAt: (data.last_request_at ?? data.lastRequestAt ?? null) as string | null,
    loadedAt: (data.loaded_at ?? data.loadedAt ?? null) as string | null,
    errorMessage: (data.error_message ?? data.errorMessage ?? null) as string | null,
    schedules,
    extraConfig: toVLLMConfig(data.extra_config ?? data.extraConfig),
  }
}

function toDownloadJob(raw: unknown): DownloadJob {
  const data = isRecord(raw) ? raw : {}
  const registerConfigRaw = isRecord(data.register_config ?? data.registerConfig)
    ? ((data.register_config ?? data.registerConfig) as Record<string, unknown>)
    : null
  return {
    jobId: String(data.job_id ?? data.jobId ?? ""),
    source: String(data.source ?? "huggingface") as DownloadSource,
    modelRef: String(data.model_ref ?? data.modelRef ?? ""),
    artifact: (data.artifact ?? null) as string | null,
    registerConfig: registerConfigRaw
      ? {
          displayName: registerConfigRaw.display_name as string | undefined,
          loadPolicy: (registerConfigRaw.load_policy ?? "on_demand") as "pin" | "warm" | "on_demand",
          autoReload: Boolean(registerConfigRaw.auto_reload),
          preferredGpu: (registerConfigRaw.preferred_gpu ?? null) as number | null,
          extraConfig: isRecord(registerConfigRaw.extra_config)
            ? (registerConfigRaw.extra_config as { vllm?: import("@/types").VLLMConfig })
            : undefined,
        }
      : null,
    status: String(data.status ?? "queued") as DownloadJob["status"],
    progressPct: Number(data.progress_pct ?? data.progressPct ?? 0),
    speedMbS: data.speed_mb_s === null || data.speedMbS === null ? null : Number(data.speed_mb_s ?? data.speedMbS ?? 0),
    etaSeconds: data.eta_seconds === null || data.etaSeconds === null ? null : Number(data.eta_seconds ?? data.etaSeconds ?? 0),
    error: (data.error ?? null) as string | null,
    startedAt: String(data.started_at ?? data.startedAt ?? new Date(0).toISOString()),
    completedAt: (data.completed_at ?? data.completedAt ?? null) as string | null,
  }
}

function toHFModelCard(raw: unknown): HFModelCard {
  const data = isRecord(raw) ? raw : {}
  return {
    repoId: String(data.repo_id ?? data.repoId ?? ""),
    modelName: String(data.model_name ?? data.modelName ?? ""),
    task: (data.task ?? null) as string | null,
    downloads: Number(data.downloads ?? 0),
    likes: Number(data.likes ?? 0),
    sizeGb: data.size_gb != null ? Number(data.size_gb) : (data.sizeGb != null ? Number(data.sizeGb) : null),
    tags: Array.isArray(data.tags) ? data.tags.map(String) : [],
    gated: Boolean(data.gated),
    suggestedBackend: String(data.suggested_backend ?? data.suggestedBackend ?? "vllm") as HFModelCard["suggestedBackend"],
    compatibility: String(data.compatibility ?? "unknown"),
    compatibilityReason: (data.compatibility_reason ?? data.compatibilityReason ?? null) as string | null,
    vllmSupport: toHFVLLMSupport(data.vllm_support ?? data.vllmSupport),
  }
}

function toHFModelDetail(raw: unknown): HFModelDetail {
  const data = isRecord(raw) ? raw : {}
  return {
    ...toHFModelCard(raw),
    siblings: Array.isArray(data.siblings) ? (data.siblings as Record<string, unknown>[]) : [],
    readmeExcerpt: (data.readme_excerpt ?? data.readmeExcerpt ?? null) as string | null,
    suggestedBackend: String(data.suggested_backend ?? data.suggestedBackend ?? "vllm") as HFModelDetail["suggestedBackend"],
    estimatedVramGb: data.estimated_vram_gb != null ? Number(data.estimated_vram_gb) : (data.estimatedVramGb != null ? Number(data.estimatedVramGb) : null),
    vllmSupport: toHFVLLMSupport(data.vllm_support ?? data.vllmSupport),
  }
}

function toHFModelVariant(raw: unknown): HFModelVariant {
  const data = isRecord(raw) ? raw : {}
  return {
    variantId: String(data.variant_id ?? data.variantId ?? ""),
    label: String(data.label ?? ""),
    artifact: (data.artifact ?? null) as string | null,
    sizeGb: data.size_gb != null ? Number(data.size_gb) : (data.sizeGb != null ? Number(data.sizeGb) : null),
    format: String(data.format ?? "unknown"),
    quantization: (data.quantization ?? null) as string | null,
    backendType: String(data.backend_type ?? data.backendType ?? "vllm") as HFModelVariant["backendType"],
    isDefault: Boolean(data.is_default ?? data.isDefault),
    installable: data.installable == null ? true : Boolean(data.installable),
    compatibility: String(data.compatibility ?? "unknown"),
    compatibilityReason: (data.compatibility_reason ?? data.compatibilityReason ?? null) as string | null,
    vllmSupport: toHFVLLMSupport(data.vllm_support ?? data.vllmSupport),
  }
}

function toLocalModel(raw: unknown): LocalModel {
  const data = isRecord(raw) ? raw : {}
  const modelRef = String(data.model_ref ?? data.modelRef ?? data.model_id ?? data.modelId ?? "")
  return {
    modelId: modelRef,
    path: String(data.path ?? ""),
    sizeGb: Number(data.size_gb ?? data.sizeGb ?? 0),
    backendType: String(data.backend_type ?? data.backendType ?? "vllm") as LocalModel["backendType"],
    configured: Boolean(data.configured ?? false),
  }
}

function toOllamaVariant(raw: unknown): OllamaModelVariant {
  const data = isRecord(raw) ? raw : {}
  return {
    name: String(data.name ?? ""),
    tag: String(data.tag ?? ""),
    sizeGb: data.size_gb != null ? Number(data.size_gb) : (data.sizeGb != null ? Number(data.sizeGb) : null),
    parameterSize: (data.parameter_size ?? data.parameterSize ?? null) as string | null,
    quantization: (data.quantization ?? null) as string | null,
    contextWindow: (data.context_window ?? data.contextWindow ?? null) as string | null,
    modality: (data.modality ?? null) as string | null,
    updatedHint: (data.updated_hint ?? data.updatedHint ?? null) as string | null,
  }
}

function toServerConfig(raw: unknown): ServerConfig {
  const data = isRecord(raw) ? raw : {}
  const globalSchedulesRaw = data.globalSchedules
  const globalSchedules = Array.isArray(globalSchedulesRaw)
    ? globalSchedulesRaw.map((schedule, idx) => {
        const s = isRecord(schedule) ? schedule : {}
        return {
          id: String(s.id ?? `global-schedule-${idx}`),
          days: Array.isArray(s.days) ? s.days.map((d) => Number(d)).filter((d) => Number.isFinite(d)) : [],
          start: String(s.start ?? "00:00"),
          end: String(s.end ?? "00:00"),
          enabled: Boolean(s.enabled ?? true),
        }
      })
    : []

  return {
    defaultGpuIndex: Number(data.defaultGpuIndex ?? 0),
    idleTimeoutSeconds: Number(data.idleTimeoutSeconds ?? 0),
    vramBufferMb: Number(data.vramBufferMb ?? 0),
    vramPressureThresholdPct: Number(data.vramPressureThresholdPct ?? 0),
    logLevel: String(data.logLevel ?? "info"),
    litellmBaseUrl: String(data.litellmBaseUrl ?? ""),
    litellmAdminKey: String(data.litellmAdminKey ?? ""),
    litellmAutoSync: Boolean(data.litellmAutoSync),
    energyCostEurKwh: Number(data.energyCostEurKwh ?? 0),
    modelsDir: String(data.modelsDir ?? "/models"),
    downloadDir: String(data.downloadDir ?? "/models/downloads"),
    maxTemperatureC: Number(data.maxTemperatureC ?? 88),
    globalSchedules,
  }
}

async function request<T>(
  method: string,
  path: string,
  body?: unknown,
): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method,
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail ?? res.statusText)
  }
  if (res.status === 204) {
    return undefined as T
  }
  return res.json() as Promise<T>
}

function buildQuery(params: Record<string, string | number | undefined>): string {
  const search = new URLSearchParams()
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== "") {
      search.set(key, String(value))
    }
  })
  const query = search.toString()
  return query ? `?${query}` : ""
}

function toServiceState(raw: unknown): ServiceState {
  const data = isRecord(raw) ? raw : {}
  return {
    serviceId: String(data.service_id ?? data.serviceId ?? ""),
    serviceType: String(data.service_type ?? data.serviceType ?? ""),
    displayName: String(data.display_name ?? data.displayName ?? ""),
    uiUrl: String(data.ui_url ?? data.uiUrl ?? ""),
    preferredGpu: data.preferred_gpu == null && data.preferredGpu == null ? null : Number(data.preferred_gpu ?? data.preferredGpu),
    idleUnloadAfterSeconds: Number(data.idle_unload_after_seconds ?? data.idleUnloadAfterSeconds ?? 600),
    enabled: Boolean(data.enabled ?? true),
    serviceAlive: Boolean(data.service_alive ?? data.serviceAlive),
    runtimeLoaded: Boolean(data.runtime_loaded ?? data.runtimeLoaded),
    status: String(data.status ?? "unknown") as ServiceState["status"],
    activeModelRef: (data.active_model_ref ?? data.activeModelRef ?? null) as string | null,
    lastActivityAt: (data.last_activity_at ?? data.lastActivityAt ?? null) as string | null,
    lastHealthCheckAt: (data.last_health_check_at ?? data.lastHealthCheckAt ?? null) as string | null,
    detail: (data.detail ?? null) as string | null,
  }
}

function toCompileJob(raw: unknown): CompileJob {
  const data = isRecord(raw) ? raw : {}
  const cfg = isRecord(data.config) ? data.config : {}
  return {
    jobId: String(data.job_id ?? data.jobId ?? ""),
    sourceModel: String(data.source_model ?? data.sourceModel ?? ""),
    engineName: String(data.engine_name ?? data.engineName ?? ""),
    gpuIndices: Array.isArray(data.gpu_indices ?? data.gpuIndices)
      ? ((data.gpu_indices ?? data.gpuIndices) as unknown[]).map(Number)
      : [],
    dtype: String(data.dtype ?? "fp16"),
    config: {
      maxBatchSize: Number(cfg.max_batch_size ?? cfg.maxBatchSize ?? 1),
      maxInputLen: Number(cfg.max_input_len ?? cfg.maxInputLen ?? 2048),
      maxSeqLen: Number(cfg.max_seq_len ?? cfg.maxSeqLen ?? 4096),
    },
    status: String(data.status ?? "pending") as CompileJob["status"],
    phase: (data.phase ?? null) as CompileJob["phase"],
    progressPct: Number(data.progress_pct ?? data.progressPct ?? 0),
    errorDetail: (data.error_detail ?? data.errorDetail ?? null) as string | null,
    engineDir: (data.engine_dir ?? data.engineDir ?? null) as string | null,
    startedAt: (data.started_at ?? data.startedAt ?? null) as string | null,
    finishedAt: (data.finished_at ?? data.finishedAt ?? null) as string | null,
  }
}

export const api = {
  gpus: {
    list: async () => (await request<unknown[]>("GET", "/ocabra/gpus")).map(toGpuState),
    get: async (index: number) => toGpuState(await request<unknown>("GET", `/ocabra/gpus/${index}`)),
    stats: (index: number, window = "5m") =>
      request<GPUStatHistory>("GET", `/ocabra/gpus/${index}/stats${buildQuery({ window })}`),
  },

  models: {
    list: async () => (await request<unknown[]>("GET", "/ocabra/models")).map(toModelState),
    get: async (modelId: string) =>
      toModelState(await request<unknown>("GET", `/ocabra/models/${encodeURIComponent(modelId)}`)),
    load: async (modelId: string) =>
      toModelState(await request<unknown>("POST", `/ocabra/models/${encodeURIComponent(modelId)}/load`)),
    unload: (modelId: string) => request<void>("POST", `/ocabra/models/${encodeURIComponent(modelId)}/unload`),
    patch: (modelId: string, patch: ModelPatchRequest) =>
      request<unknown>("PATCH", `/ocabra/models/${encodeURIComponent(modelId)}`, patch).then(toModelState),
    delete: (modelId: string) => request<void>("DELETE", `/ocabra/models/${encodeURIComponent(modelId)}`),
  },

  downloads: {
    list: async () => (await request<unknown[]>("GET", "/ocabra/downloads")).map(toDownloadJob),
    enqueue: (
      source: DownloadSource,
      modelRef: string,
      artifact?: string | null,
      registerConfig?: DownloadJob["registerConfig"],
    ) =>
      request<unknown>("POST", "/ocabra/downloads", {
        source,
        model_ref: modelRef,
        artifact: artifact ?? null,
        register_config: registerConfig ?? null,
      }).then(toDownloadJob),
    cancel: (jobId: string) => request<void>("DELETE", `/ocabra/downloads/${encodeURIComponent(jobId)}`),
    clearHistory: (status = "failed,cancelled,completed") =>
      request<{ deleted: number }>("DELETE", `/ocabra/downloads?status=${encodeURIComponent(status)}`),
    streamProgress: (jobId: string): EventSource =>
      new EventSource(`/ocabra/downloads/${jobId}/stream`),
  },

  registry: {
    searchHF: async (q: string, task?: string, limit = 20): Promise<HFModelCard[]> => {
      const params = new URLSearchParams({ q, limit: String(limit) })
      if (task) params.set("task", task)
      const raw = await request<unknown[]>("GET", `/ocabra/registry/hf/search?${params}`)
      return raw.map(toHFModelCard)
    },
    getHFDetail: async (repoId: string): Promise<HFModelDetail> =>
      toHFModelDetail(await request<unknown>("GET", `/ocabra/registry/hf/${encodeURIComponent(repoId)}`)),
    getHFVariants: async (repoId: string): Promise<HFModelVariant[]> =>
      (await request<unknown[]>("GET", `/ocabra/registry/hf/${encodeURIComponent(repoId)}/variants`)).map(
        toHFModelVariant,
      ),
    searchBitnet: async (q: string, limit = 20): Promise<HFModelCard[]> => {
      const params = new URLSearchParams({ q, limit: String(limit) })
      const raw = await request<unknown[]>("GET", `/ocabra/registry/bitnet/search?${params}`)
      return raw.map(toHFModelCard)
    },
    getBitnetVariants: async (repoId: string): Promise<HFModelVariant[]> =>
      (await request<unknown[]>("GET", `/ocabra/registry/bitnet/${encodeURIComponent(repoId)}/variants`)).map(
        toHFModelVariant,
      ),
    searchOllama: (q: string) =>
      request<OllamaModelCard[]>("GET", `/ocabra/registry/ollama/search?q=${encodeURIComponent(q)}`),
    getOllamaVariants: async (modelName: string): Promise<OllamaModelVariant[]> =>
      (await request<unknown[]>("GET", `/ocabra/registry/ollama/${encodeURIComponent(modelName)}/variants`)).map(
        toOllamaVariant,
      ),
    listLocal: async () => (await request<unknown[]>("GET", "/ocabra/registry/local")).map(toLocalModel),
  },

  stats: {
    requests: (params: StatsParams) => {
      const query = buildQuery({ from: params.from, to: params.to, model_id: params.modelId })
      return request<RequestStats>("GET", `/ocabra/stats/requests${query}`)
    },
    energy: (params: StatsParams) => {
      const query = buildQuery({ from: params.from, to: params.to })
      return request<EnergyStats>("GET", `/ocabra/stats/energy${query}`)
    },
    tokens: (params: StatsParams) => {
      const query = buildQuery({ from: params.from, to: params.to, model_id: params.modelId })
      return request<TokenStats>("GET", `/ocabra/stats/tokens${query}`)
    },
    performance: (params: StatsParams) => {
      const query = buildQuery({ from: params.from, to: params.to, model_id: params.modelId })
      return request<PerformanceStats>("GET", `/ocabra/stats/performance${query}`)
    },
    overview: (params: StatsParams) => {
      const query = buildQuery({ from: params.from, to: params.to, model_id: params.modelId })
      return request<OverviewStats>("GET", `/ocabra/stats/overview${query}`)
    },
  },

  services: {
    list: async () => (await request<unknown[]>("GET", "/ocabra/services")).map(toServiceState),
    get: async (serviceId: string) =>
      toServiceState(await request<unknown>("GET", `/ocabra/services/${encodeURIComponent(serviceId)}`)),
    refresh: async (serviceId: string) =>
      toServiceState(await request<unknown>("POST", `/ocabra/services/${encodeURIComponent(serviceId)}/refresh`)),
    unload: async (serviceId: string) =>
      toServiceState(await request<unknown>("POST", `/ocabra/services/${encodeURIComponent(serviceId)}/unload`)),
    start: async (serviceId: string) =>
      toServiceState(await request<unknown>("POST", `/ocabra/services/${encodeURIComponent(serviceId)}/start`)),
    patch: async (serviceId: string, patch: { enabled: boolean }) =>
      toServiceState(await request<unknown>("PATCH", `/ocabra/services/${encodeURIComponent(serviceId)}`, patch)),
  },

  config: {
    get: async () => toServerConfig(await request<unknown>("GET", "/ocabra/config")),
    patch: async (patch: Partial<ServerConfig>) =>
      toServerConfig(await request<unknown>("PATCH", "/ocabra/config", patch)),
    syncLiteLLM: async () => {
      const result = await request<{ synced_models: number }>("POST", "/ocabra/config/litellm/sync")
      return { syncedModels: result.synced_models }
    },
  },

  trtllm: {
    compile: async (body: {
      modelId: string
      gpuIndices: number[]
      dtype: string
      maxBatchSize: number
      maxInputLen: number
      maxSeqLen: number
      engineName: string
    }): Promise<CompileJob> =>
      toCompileJob(
        await request<unknown>("POST", "/ocabra/trtllm/compile", {
          model_id: body.modelId,
          gpu_indices: body.gpuIndices,
          dtype: body.dtype,
          max_batch_size: body.maxBatchSize,
          max_input_len: body.maxInputLen,
          max_seq_len: body.maxSeqLen,
          engine_name: body.engineName,
        }),
      ),
    list: async (): Promise<CompileJob[]> =>
      (await request<unknown[]>("GET", "/ocabra/trtllm/compile")).map(toCompileJob),
    cancel: async (jobId: string): Promise<CompileJob> =>
      toCompileJob(await request<unknown>("DELETE", `/ocabra/trtllm/compile/${encodeURIComponent(jobId)}`)),
    streamUrl: (jobId: string): string => `/ocabra/trtllm/compile/${encodeURIComponent(jobId)}/stream`,
    deleteEngine: async (engineName: string): Promise<void> => {
      await request<unknown>("DELETE", `/ocabra/trtllm/engines/${encodeURIComponent(engineName)}`)
    },
    estimate: async (params: {
      modelId: string
      tpSize: number
      dtype: string
      maxBatchSize: number
      maxSeqLen: number
    }): Promise<VramEstimate> => {
      const qs = new URLSearchParams({
        model_id: params.modelId,
        tp_size: String(params.tpSize),
        dtype: params.dtype,
        max_batch_size: String(params.maxBatchSize),
        max_seq_len: String(params.maxSeqLen),
      })
      const raw = await request<Record<string, unknown>>("GET", `/ocabra/trtllm/estimate?${qs}`)
      return {
        estimatedParamsB: raw.estimated_params_b as number | null,
        quant: raw.quant as string,
        tpSize: raw.tp_size as number,
        configFound: raw.config_found as boolean,
        serve: raw.serve as VramEstimate["serve"],
        build: raw.build as VramEstimate["build"],
        disk: raw.disk as VramEstimate["disk"],
        warnings: raw.warnings as string[],
      }
    },
  },
}

export interface VramEstimate {
  estimatedParamsB: number | null
  quant: string
  tpSize: number
  configFound: boolean
  serve: { vramPerGpuMb: number; vramTotalMb: number; breakdown: { weightsMb: number; kvCacheMb: number; overheadMb: number } }
  build: { vramPerGpuMb: number; vramTotalMb: number }
  disk: { engineMb: number; checkpointMbTemp: number; totalPeakMb: number }
  warnings: string[]
}
