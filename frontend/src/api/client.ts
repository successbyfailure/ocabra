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
  ModelsStorageStats,
  ModelMemoryEstimate,
} from "@/types"

const BASE = ""

type AnyRecord = Record<string, unknown>

function isRecord(value: unknown): value is AnyRecord {
  return typeof value === "object" && value !== null
}

function hasAnyKey(record: AnyRecord, keys: string[]): boolean {
  return keys.some((key) => key in record)
}

function toNumberOrNull(record: AnyRecord, snakeKey: string, camelKey = snakeKey): number | null {
  const value = record[snakeKey] ?? record[camelKey]
  return value == null ? null : Number(value)
}

function toBooleanOrNull(record: AnyRecord, snakeKey: string, camelKey = snakeKey): boolean | null {
  const value = record[snakeKey] ?? record[camelKey]
  return value == null ? null : Boolean(value)
}

function toBackendExtraConfig(raw: unknown): ModelState["extraConfig"] {
  const data = isRecord(raw) ? raw : {}
  const next: ModelState["extraConfig"] = { ...data }

  const vllmRaw = isRecord(data.vllm) ? data.vllm : {}
  const vllmSource = isRecord(data.vllm) ? vllmRaw : data
  if (
    isRecord(data.vllm) ||
    hasAnyKey(vllmSource, [
      "recipe_id",
      "recipeId",
      "probe_status",
      "probeStatus",
      "model_impl",
      "modelImpl",
      "runner",
      "tensor_parallel_size",
      "tensorParallelSize",
      "max_model_len",
      "maxModelLen",
      "max_num_seqs",
      "maxNumSeqs",
      "gpu_memory_utilization",
      "gpuMemoryUtilization",
      "enable_prefix_caching",
      "enablePrefixCaching",
      "trust_remote_code",
      "trustRemoteCode",
      "hf_overrides",
      "hfOverrides",
      "chat_template",
      "chatTemplate",
      "tool_call_parser",
      "toolCallParser",
      "reasoning_parser",
      "reasoningParser",
    ])
  ) {
    next.vllm = {
      recipeId: (vllmSource.recipe_id ?? vllmSource.recipeId ?? null) as string | null,
      recipeNotes: Array.isArray(vllmSource.recipe_notes ?? vllmSource.recipeNotes)
        ? ((vllmSource.recipe_notes ?? vllmSource.recipeNotes) as unknown[]).map(String)
        : [],
      recipeModelImpl: (vllmSource.recipe_model_impl ?? vllmSource.recipeModelImpl ?? null) as
        | "auto"
        | "vllm"
        | "transformers"
        | null,
      recipeRunner: (vllmSource.recipe_runner ?? vllmSource.recipeRunner ?? null) as
        | "generate"
        | "pooling"
        | null,
      suggestedConfig: isRecord(vllmSource.suggested_config ?? vllmSource.suggestedConfig)
        ? ((vllmSource.suggested_config ?? vllmSource.suggestedConfig) as Record<string, unknown>)
        : {},
      suggestedTuning: isRecord(vllmSource.suggested_tuning ?? vllmSource.suggestedTuning)
        ? ((vllmSource.suggested_tuning ?? vllmSource.suggestedTuning) as Record<string, unknown>)
        : {},
      probeStatus: (vllmSource.probe_status ?? vllmSource.probeStatus ?? null) as import("@/types").HFVLLMRuntimeProbe["status"] | null,
      probeReason: (vllmSource.probe_reason ?? vllmSource.probeReason ?? null) as string | null,
      probeObservedAt: (vllmSource.probe_observed_at ?? vllmSource.probeObservedAt ?? null) as string | null,
      probeRecommendedModelImpl:
        (vllmSource.probe_recommended_model_impl ?? vllmSource.probeRecommendedModelImpl ?? null) as
          | "auto"
          | "vllm"
          | "transformers"
          | null,
      probeRecommendedRunner:
        (vllmSource.probe_recommended_runner ?? vllmSource.probeRecommendedRunner ?? null) as
          | "generate"
          | "pooling"
          | null,
      modelImpl: (vllmSource.model_impl ?? vllmSource.modelImpl ?? null) as
        | "auto"
        | "vllm"
        | "transformers"
        | null,
      runner: (vllmSource.runner ?? null) as "generate" | "pooling" | null,
      hfOverrides: (vllmSource.hf_overrides ?? vllmSource.hfOverrides ?? null) as string | Record<string, unknown> | null,
      chatTemplate: (vllmSource.chat_template ?? vllmSource.chatTemplate ?? null) as string | null,
      chatTemplateContentFormat:
        (vllmSource.chat_template_content_format ?? vllmSource.chatTemplateContentFormat ?? null) as string | null,
      generationConfig: (vllmSource.generation_config ?? vllmSource.generationConfig ?? null) as string | null,
      overrideGenerationConfig:
        (vllmSource.override_generation_config ?? vllmSource.overrideGenerationConfig ?? null) as
          | string
          | Record<string, unknown>
          | null,
      toolCallParser: (vllmSource.tool_call_parser ?? vllmSource.toolCallParser ?? null) as string | null,
      toolParserPlugin: (vllmSource.tool_parser_plugin ?? vllmSource.toolParserPlugin ?? null) as string | null,
      reasoningParser: (vllmSource.reasoning_parser ?? vllmSource.reasoningParser ?? null) as string | null,
      languageModelOnly: toBooleanOrNull(vllmSource, "language_model_only", "languageModelOnly"),
      maxNumSeqs: toNumberOrNull(vllmSource, "max_num_seqs", "maxNumSeqs"),
      maxNumBatchedTokens: toNumberOrNull(vllmSource, "max_num_batched_tokens", "maxNumBatchedTokens"),
      tensorParallelSize: toNumberOrNull(vllmSource, "tensor_parallel_size", "tensorParallelSize"),
      maxModelLen: toNumberOrNull(vllmSource, "max_model_len", "maxModelLen"),
      gpuMemoryUtilization: toNumberOrNull(vllmSource, "gpu_memory_utilization", "gpuMemoryUtilization"),
      enablePrefixCaching: Boolean(vllmSource.enable_prefix_caching ?? vllmSource.enablePrefixCaching),
      enableChunkedPrefill: toBooleanOrNull(vllmSource, "enable_chunked_prefill", "enableChunkedPrefill"),
      enforceEager: Boolean(vllmSource.enforce_eager ?? vllmSource.enforceEager),
      trustRemoteCode: Boolean(vllmSource.trust_remote_code ?? vllmSource.trustRemoteCode),
      swapSpace: toNumberOrNull(vllmSource, "swap_space", "swapSpace"),
      kvCacheDtype: (vllmSource.kv_cache_dtype ?? vllmSource.kvCacheDtype ?? null) as string | null,
    }
  }

  const sglangRaw = isRecord(data.sglang) ? data.sglang : {}
  const sglangSource = isRecord(data.sglang) ? sglangRaw : data
  if (isRecord(data.sglang) || hasAnyKey(sglangSource, ["tensor_parallel_size", "context_length", "mem_fraction_static", "trust_remote_code", "disable_radix_cache"])) {
    next.sglang = {
      tensorParallelSize: toNumberOrNull(sglangSource, "tensor_parallel_size", "tensorParallelSize"),
      contextLength: toNumberOrNull(sglangSource, "context_length", "contextLength"),
      memFractionStatic: toNumberOrNull(sglangSource, "mem_fraction_static", "memFractionStatic"),
      trustRemoteCode: Boolean(sglangSource.trust_remote_code ?? sglangSource.trustRemoteCode),
      disableRadixCache: Boolean(sglangSource.disable_radix_cache ?? sglangSource.disableRadixCache),
    }
  }

  const llamaCppRaw = isRecord(data.llama_cpp) ? data.llama_cpp : {}
  const llamaCppSource = isRecord(data.llama_cpp) ? llamaCppRaw : data
  if (isRecord(data.llama_cpp) || hasAnyKey(llamaCppSource, ["gpu_layers", "ctx_size", "flash_attn", "embedding"])) {
    next.llama_cpp = {
      gpuLayers: toNumberOrNull(llamaCppSource, "gpu_layers", "gpuLayers"),
      ctxSize: toNumberOrNull(llamaCppSource, "ctx_size", "ctxSize"),
      flashAttn: Boolean(llamaCppSource.flash_attn ?? llamaCppSource.flashAttn),
      embedding: Boolean(llamaCppSource.embedding),
    }
  }

  const bitnetRaw = isRecord(data.bitnet) ? data.bitnet : {}
  const bitnetSource = isRecord(data.bitnet) ? bitnetRaw : data
  if (isRecord(data.bitnet) || hasAnyKey(bitnetSource, ["gpu_layers", "ctx_size", "flash_attn"])) {
    next.bitnet = {
      gpuLayers: toNumberOrNull(bitnetSource, "gpu_layers", "gpuLayers"),
      ctxSize: toNumberOrNull(bitnetSource, "ctx_size", "ctxSize"),
      flashAttn: Boolean(bitnetSource.flash_attn ?? bitnetSource.flashAttn),
    }
  }

  const tensorrtRaw = isRecord(data.tensorrt_llm) ? data.tensorrt_llm : {}
  const tensorrtSource = isRecord(data.tensorrt_llm) ? tensorrtRaw : data
  if (isRecord(data.tensorrt_llm) || hasAnyKey(tensorrtSource, ["max_batch_size", "context_length", "trust_remote_code"])) {
    next.tensorrt_llm = {
      maxBatchSize: toNumberOrNull(tensorrtSource, "max_batch_size", "maxBatchSize"),
      contextLength: toNumberOrNull(tensorrtSource, "context_length", "contextLength"),
      trustRemoteCode: Boolean(tensorrtSource.trust_remote_code ?? tensorrtSource.trustRemoteCode),
    }
  }

  const whisperRaw = isRecord(data.whisper) ? data.whisper : {}
  const whisperSource = isRecord(data.whisper) ? whisperRaw : data
  if (isRecord(data.whisper) || hasAnyKey(whisperSource, ["diarization_enabled", "diarization_model_id"])) {
    next.whisper = {
      diarizationEnabled: Boolean(whisperSource.diarization_enabled ?? whisperSource.diarizationEnabled),
      diarizationModelId: (whisperSource.diarization_model_id ?? whisperSource.diarizationModelId ?? null) as string | null,
    }
  }

  return next
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
    musicGeneration: Boolean(data.music_generation ?? data.musicGeneration),
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

function toModelsStorageStats(raw: unknown): ModelsStorageStats {
  const data = isRecord(raw) ? raw : {}
  return {
    path: String(data.path ?? ""),
    totalBytes: Number(data.total_bytes ?? data.totalBytes ?? 0),
    usedBytes: Number(data.used_bytes ?? data.usedBytes ?? 0),
    freeBytes: Number(data.free_bytes ?? data.freeBytes ?? 0),
  }
}

function toModelMemoryEstimate(raw: unknown): ModelMemoryEstimate {
  const data = isRecord(raw) ? raw : {}
  return {
    backendType: String(data.backend_type ?? data.backendType ?? "vllm") as ModelMemoryEstimate["backendType"],
    gpuIndex:
      data.gpu_index == null && data.gpuIndex == null ? null : Number(data.gpu_index ?? data.gpuIndex),
    totalVramMb:
      data.total_vram_mb == null && data.totalVramMb == null ? null : Number(data.total_vram_mb ?? data.totalVramMb),
    freeVramMb:
      data.free_vram_mb == null && data.freeVramMb == null ? null : Number(data.free_vram_mb ?? data.freeVramMb),
    budgetVramMb:
      data.budget_vram_mb == null && data.budgetVramMb == null ? null : Number(data.budget_vram_mb ?? data.budgetVramMb),
    requestedContextLength:
      data.requested_context_length == null && data.requestedContextLength == null
        ? null
        : Number(data.requested_context_length ?? data.requestedContextLength),
    estimatedWeightsMb:
      data.estimated_weights_mb == null && data.estimatedWeightsMb == null
        ? null
        : Number(data.estimated_weights_mb ?? data.estimatedWeightsMb),
    estimatedEngineMbPerGpu:
      data.estimated_engine_mb_per_gpu == null && data.estimatedEngineMbPerGpu == null
        ? null
        : Number(data.estimated_engine_mb_per_gpu ?? data.estimatedEngineMbPerGpu),
    estimatedKvCacheMb:
      data.estimated_kv_cache_mb == null && data.estimatedKvCacheMb == null
        ? null
        : Number(data.estimated_kv_cache_mb ?? data.estimatedKvCacheMb),
    estimatedMaxContextLength:
      data.estimated_max_context_length == null && data.estimatedMaxContextLength == null
        ? null
        : Number(data.estimated_max_context_length ?? data.estimatedMaxContextLength),
    modelLoadingMemoryMb:
      data.model_loading_memory_mb == null && data.modelLoadingMemoryMb == null
        ? null
        : Number(data.model_loading_memory_mb ?? data.modelLoadingMemoryMb),
    maximumConcurrency:
      data.maximum_concurrency == null && data.maximumConcurrency == null
        ? null
        : Number(data.maximum_concurrency ?? data.maximumConcurrency),
    tensorParallelSize:
      data.tensor_parallel_size == null && data.tensorParallelSize == null
        ? null
        : Number(data.tensor_parallel_size ?? data.tensorParallelSize),
    fitsCurrentGpu:
      data.fits_current_gpu == null && data.fitsCurrentGpu == null ? null : Boolean(data.fits_current_gpu ?? data.fitsCurrentGpu),
    enginePresent:
      data.engine_present == null && data.enginePresent == null ? null : Boolean(data.engine_present ?? data.enginePresent),
    source: String(data.source ?? "heuristic") as ModelMemoryEstimate["source"],
    status: String(data.status ?? "ok") as ModelMemoryEstimate["status"],
    warning: (data.warning ?? null) as string | null,
    notes: Array.isArray(data.notes) ? data.notes.map(String) : [],
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
    extraConfig: toBackendExtraConfig(data.extra_config ?? data.extraConfig),
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
    idleEvictionCheckIntervalSeconds: Number(
      data.idleEvictionCheckIntervalSeconds ?? data.idle_eviction_check_interval_seconds ?? 15,
    ),
    modelLoadWaitTimeoutSeconds: Number(
      data.modelLoadWaitTimeoutSeconds ?? data.model_load_wait_timeout_seconds ?? 720,
    ),
    pressureEvictionDrainTimeoutSeconds: Number(
      data.pressureEvictionDrainTimeoutSeconds ?? data.pressure_eviction_drain_timeout_seconds ?? 60,
    ),
    vramBufferMb: Number(data.vramBufferMb ?? 0),
    vramPressureThresholdPct: Number(data.vramPressureThresholdPct ?? 0),
    openaiAudioMaxPartSizeMb: Number(
      data.openaiAudioMaxPartSizeMb ?? data.openai_audio_max_part_size_mb ?? 256,
    ),
    whisperStartupTimeoutSeconds: Number(
      data.whisperStartupTimeoutSeconds ?? data.whisper_startup_timeout_seconds ?? 300,
    ),
    logLevel: String(data.logLevel ?? "info"),
    litellmBaseUrl: String(data.litellmBaseUrl ?? ""),
    litellmAdminKey: String(data.litellmAdminKey ?? ""),
    litellmAutoSync: Boolean(data.litellmAutoSync),
    energyCostEurKwh: Number(data.energyCostEurKwh ?? 0),
    modelsDir: String(data.modelsDir ?? "/models"),
    downloadDir: String(data.downloadDir ?? "/models/downloads"),
    maxTemperatureC: Number(data.maxTemperatureC ?? 88),
    vllmGpuMemoryUtilization: Number(data.vllmGpuMemoryUtilization ?? data.vllm_gpu_memory_utilization ?? 0.85),
    vllmMaxNumSeqs:
      data.vllmMaxNumSeqs == null && data.vllm_max_num_seqs == null
        ? null
        : Number(data.vllmMaxNumSeqs ?? data.vllm_max_num_seqs),
    vllmMaxNumBatchedTokens:
      data.vllmMaxNumBatchedTokens == null && data.vllm_max_num_batched_tokens == null
        ? null
        : Number(data.vllmMaxNumBatchedTokens ?? data.vllm_max_num_batched_tokens),
    vllmEnablePrefixCaching: Boolean(
      data.vllmEnablePrefixCaching ?? data.vllm_enable_prefix_caching ?? true,
    ),
    vllmEnforceEager: Boolean(data.vllmEnforceEager ?? data.vllm_enforce_eager ?? true),
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
    llamaCppGpuLayers: Number(data.llamaCppGpuLayers ?? data.llama_cpp_gpu_layers ?? 0),
    llamaCppCtxSize: Number(data.llamaCppCtxSize ?? data.llama_cpp_ctx_size ?? 4096),
    llamaCppFlashAttn: Boolean(data.llamaCppFlashAttn ?? data.llama_cpp_flash_attn ?? false),
    bitnetGpuLayers: Number(data.bitnetGpuLayers ?? data.bitnet_gpu_layers ?? 0),
    bitnetCtxSize: Number(data.bitnetCtxSize ?? data.bitnet_ctx_size ?? 4096),
    bitnetFlashAttn: Boolean(data.bitnetFlashAttn ?? data.bitnet_flash_attn ?? false),
    diffusersTorchDtype: String(data.diffusersTorchDtype ?? data.diffusers_torch_dtype ?? "auto"),
    diffusersOffloadMode: String(data.diffusersOffloadMode ?? data.diffusers_offload_mode ?? "none"),
    diffusersEnableTorchCompile: Boolean(
      data.diffusersEnableTorchCompile ?? data.diffusers_enable_torch_compile ?? false,
    ),
    diffusersEnableXformers: Boolean(
      data.diffusersEnableXformers ?? data.diffusers_enable_xformers ?? false,
    ),
    diffusersAllowTf32: Boolean(data.diffusersAllowTf32 ?? data.diffusers_allow_tf32 ?? true),
    tensorrtLlmEnabled: Boolean(data.tensorrtLlmEnabled ?? data.tensorrt_llm_enabled ?? false),
    tensorrtLlmMaxBatchSize:
      data.tensorrtLlmMaxBatchSize == null && data.tensorrt_llm_max_batch_size == null
        ? null
        : Number(data.tensorrtLlmMaxBatchSize ?? data.tensorrt_llm_max_batch_size),
    tensorrtLlmContextLength:
      data.tensorrtLlmContextLength == null && data.tensorrt_llm_context_length == null
        ? null
        : Number(data.tensorrtLlmContextLength ?? data.tensorrt_llm_context_length),
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
    storage: async () => toModelsStorageStats(await request<unknown>("GET", "/ocabra/models/storage")),
    get: async (modelId: string) =>
      toModelState(await request<unknown>("GET", `/ocabra/models/${encodeURIComponent(modelId)}`)),
    load: async (modelId: string) =>
      toModelState(await request<unknown>("POST", `/ocabra/models/${encodeURIComponent(modelId)}/load`)),
    unload: (modelId: string) => request<void>("POST", `/ocabra/models/${encodeURIComponent(modelId)}/unload`),
    patch: (modelId: string, patch: ModelPatchRequest) =>
      request<unknown>("PATCH", `/ocabra/models/${encodeURIComponent(modelId)}`, patch).then(toModelState),
    estimateMemory: (
      modelId: string,
      body: {
        preferredGpu?: number | null
        extraConfig?: ModelPatchRequest["extraConfig"]
        runProbe?: boolean
      },
    ) =>
      request<unknown>("POST", `/ocabra/models/${encodeURIComponent(modelId)}/memory-estimate`, {
        preferred_gpu: body.preferredGpu ?? null,
        extra_config: body.extraConfig ?? null,
        run_probe: Boolean(body.runProbe),
      }).then(toModelMemoryEstimate),
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
      new EventSource(`/ocabra/downloads/${encodeURIComponent(jobId)}/stream`),
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
      gpuIndices: number[]
      tpSize: number
      dtype: string
      maxBatchSize: number
      maxSeqLen: number
    }): Promise<VramEstimate> => {
      const qs = new URLSearchParams()
      qs.set("model_id", params.modelId)
      qs.set("tp_size", String(params.tpSize))
      qs.set("dtype", params.dtype)
      qs.set("max_batch_size", String(params.maxBatchSize))
      qs.set("max_seq_len", String(params.maxSeqLen))
      params.gpuIndices.forEach((gpuIndex) => qs.append("gpu_indices", String(gpuIndex)))
      const raw = await request<Record<string, unknown>>("GET", `/ocabra/trtllm/estimate?${qs}`)
      return toVramEstimate(raw)
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

function toVramEstimate(raw: Record<string, unknown>): VramEstimate {
  const serveRaw = (raw.serve ?? {}) as Record<string, unknown>
  const buildRaw = (raw.build ?? {}) as Record<string, unknown>
  const diskRaw = (raw.disk ?? {}) as Record<string, unknown>
  const breakdownRaw = (serveRaw.breakdown ?? {}) as Record<string, unknown>

  return {
    estimatedParamsB: raw.estimated_params_b == null ? null : Number(raw.estimated_params_b),
    quant: String(raw.quant ?? "unknown"),
    tpSize: Number(raw.tp_size ?? 1),
    configFound: Boolean(raw.config_found),
    serve: {
      vramPerGpuMb: Number(serveRaw.vram_per_gpu_mb ?? 0),
      vramTotalMb: Number(serveRaw.vram_total_mb ?? 0),
      breakdown: {
        weightsMb: Number(breakdownRaw.weights_mb ?? 0),
        kvCacheMb: Number(breakdownRaw.kv_cache_mb ?? 0),
        overheadMb: Number(breakdownRaw.overhead_mb ?? 0),
      },
    },
    build: {
      vramPerGpuMb: Number(buildRaw.vram_per_gpu_mb ?? 0),
      vramTotalMb: Number(buildRaw.vram_total_mb ?? 0),
    },
    disk: {
      engineMb: Number(diskRaw.engine_mb ?? 0),
      checkpointMbTemp: Number(diskRaw.checkpoint_mb_temp ?? 0),
      totalPeakMb: Number(diskRaw.total_peak_mb ?? 0),
    },
    warnings: Array.isArray(raw.warnings) ? raw.warnings.map(String) : [],
  }
}
