// Cliente tipado de /ocabra/* — dominio «models» (REF-15).
// Extraído de api/client.ts; usa la capa HTTP única de api/http.ts.

import { extractErrorMessage, httpFetch, request } from "@/api/http"
import { buildQuery, hasAnyKey, isRecord, numberMap, toBooleanOrNull, toNumberOrNull } from "@/api/_helpers"
import { toDownloadJob } from "@/api/registry"
import type {
  BackendType,
  CompatibleBackend,
  ContextProbeJob,
  DownloadJob,
  LlamaCppConfig,
  LoadPolicy,
  LogicalModel,
  ModelDeployment,
  ModelProfile,
  ModelState,
  ModelsStorageStats,
  OpenAIImageModelMapEntry,
  OpenAITTSVoiceMapEntry,
  ProfileCreate,
  ProfileUpdate,
  RemoveDeploymentResult,
  RouterContextPolicy,
  RouterCreate,
  RuntimePolicy,
  RuntimePreset,
  RuntimePresetBenchmark,
  RuntimePresetCreate,
  RuntimePresetResolution,
  RuntimePresetUpdate,
  WhisperConfig,
} from "@/types"

export function toBackendExtraConfig(raw: unknown): ModelState["extraConfig"] {
  const data = isRecord(raw) ? raw : {};
  const next: ModelState["extraConfig"] = { ...data };

  const vllmRaw = isRecord(data.vllm) ? data.vllm : {};
  const vllmSource = isRecord(data.vllm) ? vllmRaw : data;
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
      recipeId: (vllmSource.recipe_id ?? vllmSource.recipeId ?? null) as
        string | null,
      recipeNotes: Array.isArray(
        vllmSource.recipe_notes ?? vllmSource.recipeNotes,
      )
        ? (
            (vllmSource.recipe_notes ?? vllmSource.recipeNotes) as unknown[]
          ).map(String)
        : [],
      recipeModelImpl: (vllmSource.recipe_model_impl ??
        vllmSource.recipeModelImpl ??
        null) as "auto" | "vllm" | "transformers" | null,
      recipeRunner: (vllmSource.recipe_runner ??
        vllmSource.recipeRunner ??
        null) as "generate" | "pooling" | null,
      suggestedConfig: isRecord(
        vllmSource.suggested_config ?? vllmSource.suggestedConfig,
      )
        ? ((vllmSource.suggested_config ??
            vllmSource.suggestedConfig) as Record<string, unknown>)
        : {},
      suggestedTuning: isRecord(
        vllmSource.suggested_tuning ?? vllmSource.suggestedTuning,
      )
        ? ((vllmSource.suggested_tuning ??
            vllmSource.suggestedTuning) as Record<string, unknown>)
        : {},
      probeStatus: (vllmSource.probe_status ??
        vllmSource.probeStatus ??
        null) as import("@/types").HFVLLMRuntimeProbe["status"] | null,
      probeReason: (vllmSource.probe_reason ??
        vllmSource.probeReason ??
        null) as string | null,
      probeObservedAt: (vllmSource.probe_observed_at ??
        vllmSource.probeObservedAt ??
        null) as string | null,
      probeRecommendedModelImpl: (vllmSource.probe_recommended_model_impl ??
        vllmSource.probeRecommendedModelImpl ??
        null) as "auto" | "vllm" | "transformers" | null,
      probeRecommendedRunner: (vllmSource.probe_recommended_runner ??
        vllmSource.probeRecommendedRunner ??
        null) as "generate" | "pooling" | null,
      modelImpl: (vllmSource.model_impl ?? vllmSource.modelImpl ?? null) as
        "auto" | "vllm" | "transformers" | null,
      runner: (vllmSource.runner ?? null) as "generate" | "pooling" | null,
      hfOverrides: (vllmSource.hf_overrides ??
        vllmSource.hfOverrides ??
        null) as string | Record<string, unknown> | null,
      chatTemplate: (vllmSource.chat_template ??
        vllmSource.chatTemplate ??
        null) as string | null,
      chatTemplateContentFormat: (vllmSource.chat_template_content_format ??
        vllmSource.chatTemplateContentFormat ??
        null) as string | null,
      generationConfig: (vllmSource.generation_config ??
        vllmSource.generationConfig ??
        null) as string | null,
      overrideGenerationConfig: (vllmSource.override_generation_config ??
        vllmSource.overrideGenerationConfig ??
        null) as string | Record<string, unknown> | null,
      toolCallParser: (vllmSource.tool_call_parser ??
        vllmSource.toolCallParser ??
        null) as string | null,
      toolParserPlugin: (vllmSource.tool_parser_plugin ??
        vllmSource.toolParserPlugin ??
        null) as string | null,
      reasoningParser: (vllmSource.reasoning_parser ??
        vllmSource.reasoningParser ??
        null) as string | null,
      languageModelOnly: toBooleanOrNull(
        vllmSource,
        "language_model_only",
        "languageModelOnly",
      ),
      maxNumSeqs: toNumberOrNull(vllmSource, "max_num_seqs", "maxNumSeqs"),
      maxNumBatchedTokens: toNumberOrNull(
        vllmSource,
        "max_num_batched_tokens",
        "maxNumBatchedTokens",
      ),
      tensorParallelSize: toNumberOrNull(
        vllmSource,
        "tensor_parallel_size",
        "tensorParallelSize",
      ),
      maxModelLen: toNumberOrNull(vllmSource, "max_model_len", "maxModelLen"),
      gpuMemoryUtilization: toNumberOrNull(
        vllmSource,
        "gpu_memory_utilization",
        "gpuMemoryUtilization",
      ),
      enablePrefixCaching: Boolean(
        vllmSource.enable_prefix_caching ?? vllmSource.enablePrefixCaching,
      ),
      enableChunkedPrefill: toBooleanOrNull(
        vllmSource,
        "enable_chunked_prefill",
        "enableChunkedPrefill",
      ),
      enforceEager: Boolean(
        vllmSource.enforce_eager ?? vllmSource.enforceEager,
      ),
      trustRemoteCode: Boolean(
        vllmSource.trust_remote_code ?? vllmSource.trustRemoteCode,
      ),
      swapSpace: toNumberOrNull(vllmSource, "swap_space", "swapSpace"),
      kvCacheDtype: (vllmSource.kv_cache_dtype ??
        vllmSource.kvCacheDtype ??
        null) as string | null,
    };
  }

  const sglangRaw = isRecord(data.sglang) ? data.sglang : {};
  const sglangSource = isRecord(data.sglang) ? sglangRaw : data;
  if (
    isRecord(data.sglang) ||
    hasAnyKey(sglangSource, [
      "tensor_parallel_size",
      "context_length",
      "mem_fraction_static",
      "trust_remote_code",
      "disable_radix_cache",
    ])
  ) {
    next.sglang = {
      tensorParallelSize: toNumberOrNull(
        sglangSource,
        "tensor_parallel_size",
        "tensorParallelSize",
      ),
      contextLength: toNumberOrNull(
        sglangSource,
        "context_length",
        "contextLength",
      ),
      memFractionStatic: toNumberOrNull(
        sglangSource,
        "mem_fraction_static",
        "memFractionStatic",
      ),
      trustRemoteCode: Boolean(
        sglangSource.trust_remote_code ?? sglangSource.trustRemoteCode,
      ),
      disableRadixCache: Boolean(
        sglangSource.disable_radix_cache ?? sglangSource.disableRadixCache,
      ),
    };
  }

  const llamaCppRaw = isRecord(data.llama_cpp) ? data.llama_cpp : {};
  const llamaCppSource = isRecord(data.llama_cpp) ? llamaCppRaw : data;
  if (
    isRecord(data.llama_cpp) ||
    hasAnyKey(llamaCppSource, [
      "gpu_layers",
      "ctx_size",
      "flash_attn",
      "embedding",
      "cache_type_k",
      "cache_type_v",
      "mmproj_model_id",
      "vision_mode",
      "jinja",
      "reasoning",
      "reasoning_format",
      "reasoning_budget",
    ])
  ) {
    const cacheK = (llamaCppSource.cache_type_k ??
      llamaCppSource.cacheTypeK) as string | null | undefined;
    const cacheV = (llamaCppSource.cache_type_v ??
      llamaCppSource.cacheTypeV) as string | null | undefined;
    next.llama_cpp = {
      // Spread the raw source first so fields this parser doesn't know
      // about (model_path, speculative, mmproj_model_id, disabled_gpus,
      // ...) survive the round-trip instead of being silently dropped —
      // reproduced live 2026-10-29: this rebuild used to replace the whole
      // object with only the 7 fields below, so any save from the Models
      // UI wiped llama_cpp.model_path and broke context-capacity planning
      // for every subsequent request. The explicit fields below still
      // normalize/override with typed values.
      ...(llamaCppSource as unknown as LlamaCppConfig),
      gpuLayers: toNumberOrNull(llamaCppSource, "gpu_layers", "gpuLayers"),
      ctxSize: toNumberOrNull(llamaCppSource, "ctx_size", "ctxSize"),
      parallelSlots: toNumberOrNull(
        llamaCppSource,
        "parallel_slots",
        "parallelSlots",
      ),
      kvUnified: (llamaCppSource.kv_unified ??
        llamaCppSource.kvUnified ??
        null) as boolean | null,
      kvUnifiedPerSlot: toNumberOrNull(
        llamaCppSource,
        "kv_unified_per_slot",
        "kvUnifiedPerSlot",
      ),
      flashAttn: Boolean(llamaCppSource.flash_attn ?? llamaCppSource.flashAttn),
      embedding: Boolean(llamaCppSource.embedding),
      cacheTypeK: (cacheK ?? null) as LlamaCppConfig["cacheTypeK"],
      cacheTypeV: (cacheV ?? null) as LlamaCppConfig["cacheTypeV"],
      mmprojModelId: (llamaCppSource.mmproj_model_id ??
        llamaCppSource.mmprojModelId ??
        null) as string | null,
      visionMode: (llamaCppSource.vision_mode ??
        llamaCppSource.visionMode ??
        null) as LlamaCppConfig["visionMode"],
      jinja: (llamaCppSource.jinja ?? null) as boolean | null,
      reasoning: (llamaCppSource.reasoning ??
        null) as LlamaCppConfig["reasoning"],
      reasoningFormat: (llamaCppSource.reasoning_format ??
        llamaCppSource.reasoningFormat ??
        null) as LlamaCppConfig["reasoningFormat"],
      reasoningBudget: toNumberOrNull(
        llamaCppSource,
        "reasoning_budget",
        "reasoningBudget",
      ),
    };
  }

  const bitnetRaw = isRecord(data.bitnet) ? data.bitnet : {};
  const bitnetSource = isRecord(data.bitnet) ? bitnetRaw : data;
  if (
    isRecord(data.bitnet) ||
    hasAnyKey(bitnetSource, ["gpu_layers", "ctx_size", "flash_attn"])
  ) {
    next.bitnet = {
      gpuLayers: toNumberOrNull(bitnetSource, "gpu_layers", "gpuLayers"),
      ctxSize: toNumberOrNull(bitnetSource, "ctx_size", "ctxSize"),
      flashAttn: Boolean(bitnetSource.flash_attn ?? bitnetSource.flashAttn),
    };
  }

  const tensorrtRaw = isRecord(data.tensorrt_llm) ? data.tensorrt_llm : {};
  const tensorrtSource = isRecord(data.tensorrt_llm) ? tensorrtRaw : data;
  if (
    isRecord(data.tensorrt_llm) ||
    hasAnyKey(tensorrtSource, [
      "max_batch_size",
      "context_length",
      "trust_remote_code",
    ])
  ) {
    next.tensorrt_llm = {
      maxBatchSize: toNumberOrNull(
        tensorrtSource,
        "max_batch_size",
        "maxBatchSize",
      ),
      contextLength: toNumberOrNull(
        tensorrtSource,
        "context_length",
        "contextLength",
      ),
      trustRemoteCode: Boolean(
        tensorrtSource.trust_remote_code ?? tensorrtSource.trustRemoteCode,
      ),
    };
  }

  const whisperRaw = isRecord(data.whisper) ? data.whisper : {};
  const whisperSource = isRecord(data.whisper) ? whisperRaw : data;
  if (
    isRecord(data.whisper) ||
    hasAnyKey(whisperSource, [
      "diarization_enabled",
      "diarization_model_id",
      "num_workers",
    ])
  ) {
    next.whisper = {
      // Keep the keys this parser does not know about (same reason as the
      // llama_cpp section above). Only for a real nested section: in the
      // legacy flat shape the source is the whole extra_config.
      ...(isRecord(data.whisper) ? (whisperRaw as WhisperConfig) : {}),
      diarizationEnabled: Boolean(
        whisperSource.diarization_enabled ?? whisperSource.diarizationEnabled,
      ),
      diarizationModelId: (whisperSource.diarization_model_id ??
        whisperSource.diarizationModelId ??
        null) as string | null,
      numWorkers: toNumberOrNull(whisperSource, "num_workers", "numWorkers"),
    };
  }

  return next;
}


export function toModelCapabilities(raw: unknown): ModelState["capabilities"] {
  const data = isRecord(raw) ? raw : {};
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
    // Honour what the backend reports (default false). It used to default to
    // *true* when absent, which made the category inference tag any model with
    // an unpopulated flag as "image" (BUG-51); the edit endpoint still returns
    // ``edit_unsupported`` for a pipeline that truly can't edit.
    imageEditing: Boolean(data.image_editing ?? data.imageEditing),
    audioTranscription: Boolean(
      data.audio_transcription ?? data.audioTranscription,
    ),
    musicGeneration: Boolean(data.music_generation ?? data.musicGeneration),
    tts: Boolean(data.tts),
    audioInput: Boolean(data.audio_input ?? data.audioInput),
    audioOutput: Boolean(data.audio_output ?? data.audioOutput),
    videoInput: Boolean(data.video_input ?? data.videoInput),
    videoGeneration: Boolean(data.video_generation ?? data.videoGeneration),
    videoUpscaling: Boolean(data.video_upscaling ?? data.videoUpscaling),
    imageUpscaling: Boolean(data.image_upscaling ?? data.imageUpscaling),
    streaming: Boolean(data.streaming),
    contextLength: Number(data.context_length ?? data.contextLength ?? 0),
    nativeContextLength: Number(
      data.native_context_length ?? data.nativeContextLength ?? 0,
    ),
    maxOutputTokens: Number(data.max_output_tokens ?? data.maxOutputTokens ?? 0),
    parallelSlots: Number(data.parallel_slots ?? data.parallelSlots ?? 0),
    kvCacheShared: Boolean(data.kv_cache_shared ?? data.kvCacheShared),
    contextPerRequest: Number(data.context_per_request ?? data.contextPerRequest ?? 0),
  };
}


export function toModelsStorageStats(raw: unknown): ModelsStorageStats {
  const data = isRecord(raw) ? raw : {};
  return {
    path: String(data.path ?? ""),
    totalBytes: Number(data.total_bytes ?? data.totalBytes ?? 0),
    usedBytes: Number(data.used_bytes ?? data.usedBytes ?? 0),
    freeBytes: Number(data.free_bytes ?? data.freeBytes ?? 0),
  };
}


// Claves de runtime_policy que este cliente mapea; el resto viaja tal cual en
// `passthrough` para que guardar la política no borre lo que la UI no edita
// (reserve_mb_by_preset, campos futuros).
export const RUNTIME_POLICY_KEYS = new Set([
  "selection",
  "restore_default_when_idle",
  "downgrade_grace_seconds",
  "minimum_residency_seconds",
  "small_requests_extend_large_ttl",
  "remote_warm_threshold_seconds",
  "max_context_tokens",
  "max_output_tokens",
  "validated_context_tokens",
  "validated_vram_mb",
  "context_probe",
  "default_output_reservation_tokens",
]);


export function toRouterContext(raw: unknown): RouterContextPolicy | null {
  if (!isRecord(raw)) return null;
  const mode = raw.mode === "common" || raw.mode === "fixed" ? raw.mode : "max";
  const tokens = Number(raw.tokens);
  return { mode, tokens: Number.isFinite(tokens) && tokens > 0 ? tokens : null };
}


export function toRuntimePolicy(raw: unknown): RuntimePolicy | null {
  if (!isRecord(raw)) return null;
  return {
    passthrough: Object.fromEntries(
      Object.entries(raw).filter(([key]) => !RUNTIME_POLICY_KEYS.has(key) && !/[A-Z]/.test(key)),
    ),
    selection: String(
      raw.selection ?? "prefer_loaded",
    ) as RuntimePolicy["selection"],
    restoreDefaultWhenIdle: Boolean(
      raw.restore_default_when_idle ?? raw.restoreDefaultWhenIdle ?? true,
    ),
    downgradeGraceSeconds: Number(
      raw.downgrade_grace_seconds ?? raw.downgradeGraceSeconds ?? 600,
    ),
    minimumResidencySeconds: Number(
      raw.minimum_residency_seconds ?? raw.minimumResidencySeconds ?? 120,
    ),
    smallRequestsExtendLargeTtl: Boolean(
      raw.small_requests_extend_large_ttl ??
      raw.smallRequestsExtendLargeTtl ??
      false,
    ),
    remoteWarmThresholdSeconds: Number(
      raw.remote_warm_threshold_seconds ?? raw.remoteWarmThresholdSeconds ?? 10,
    ),
    maxContextTokens:
      raw.max_context_tokens == null && raw.maxContextTokens == null
        ? null
        : Number(raw.max_context_tokens ?? raw.maxContextTokens),
    maxOutputTokens:
      raw.max_output_tokens == null && raw.maxOutputTokens == null
        ? null
        : Number(raw.max_output_tokens ?? raw.maxOutputTokens),
    validatedContextTokens: Object.fromEntries(
      Object.entries(
        (raw.validated_context_tokens ??
          raw.validatedContextTokens ??
          {}) as Record<string, unknown>,
      ).map(([key, value]) => [key, Number(value)]),
    ),
    validatedVramMb: Object.fromEntries(
      Object.entries(
        (raw.validated_vram_mb ?? raw.validatedVramMb ?? {}) as Record<
          string,
          unknown
        >,
      ).map(([key, value]) => [key, Number(value)]),
    ),
    contextProbe: (isRecord(raw.context_probe ?? raw.contextProbe)
      ? (raw.context_probe ?? raw.contextProbe)
      : {}) as Record<string, Record<string, unknown>>,
    defaultOutputReservationTokens:
      raw.default_output_reservation_tokens == null &&
      raw.defaultOutputReservationTokens == null
        ? null
        : Number(
            raw.default_output_reservation_tokens ??
              raw.defaultOutputReservationTokens,
          ),
  };
}


export function fromRuntimePolicy(
  policy: RuntimePolicy | null,
): Record<string, unknown> | null {
  if (policy === null) return null;
  return {
    ...(policy.passthrough ?? {}),
    selection: policy.selection,
    restore_default_when_idle: policy.restoreDefaultWhenIdle,
    downgrade_grace_seconds: policy.downgradeGraceSeconds,
    minimum_residency_seconds: policy.minimumResidencySeconds,
    small_requests_extend_large_ttl: policy.smallRequestsExtendLargeTtl,
    remote_warm_threshold_seconds: policy.remoteWarmThresholdSeconds,
    max_context_tokens: policy.maxContextTokens,
    max_output_tokens: policy.maxOutputTokens,
    validated_context_tokens: policy.validatedContextTokens,
    validated_vram_mb: policy.validatedVramMb,
    // Pass-through: saving the selection policy must not wipe the probes.
    context_probe: policy.contextProbe ?? {},
    default_output_reservation_tokens: policy.defaultOutputReservationTokens,
  };
}


export function toContextProbeJob(raw: unknown): ContextProbeJob {
  const data = isRecord(raw) ? raw : {};
  const result = isRecord(data.result) ? data.result : null;
  const vision = result && isRecord(result.vision) ? result.vision : null;
  return {
    jobId: String(data.job_id ?? ""),
    profileId: String(data.profile_id ?? ""),
    presetKey: String(data.preset_key ?? ""),
    status: String(data.status ?? "pending") as ContextProbeJob["status"],
    fullValidation: Boolean(data.full_validation),
    measureVision: Boolean(data.measure_vision),
    targetContext: data.target_context == null ? null : Number(data.target_context),
    gpuIndices: Array.isArray(data.gpu_indices) ? data.gpu_indices.map(Number) : [],
    contextIsPool: data.context_is_pool !== false,
    parallelism: Number(data.parallelism ?? 1),
    steps: (Array.isArray(data.steps) ? data.steps : []).map((step) => {
      const row = isRecord(step) ? step : {};
      return {
        context: Number(row.context ?? 0),
        usedMb: numberMap(row.used_mb),
        fits: Boolean(row.fits),
        note: String(row.note ?? ""),
        ...(row.in_ram == null ? {} : { inRam: Number(row.in_ram) }),
        ...(row.projected_mb ? { projectedMb: numberMap(row.projected_mb) } : {}),
        ...(row.ram_need_mb == null ? {} : { ramNeedMb: Number(row.ram_need_mb) }),
      };
    }),
    evicted: Array.isArray(data.evicted) ? data.evicted.map(String) : [],
    error: data.error == null ? null : String(data.error),
    startedAt: Number(data.started_at ?? 0),
    finishedAt: data.finished_at == null ? null : Number(data.finished_at),
    result: result
      ? {
          maxContext: Number(result.max_context ?? result.pool_tokens ?? 0),
          offload:
            result.mode === "offload"
              ? {
                  targetContext: Number(result.target_context ?? 0),
                  poolTokens: Number(result.pool_tokens ?? 0),
                  inRam: Number(result.in_ram ?? 0),
                  layers: Number(result.layers ?? 0),
                  moe: Boolean(result.moe),
                  placement: numberMap(result.placement),
                  ramNeedMb: result.ram_need_mb == null ? null : Number(result.ram_need_mb),
                  ramAvailableMb:
                    result.ram_available_mb == null ? null : Number(result.ram_available_mb),
                  mlock: Boolean(result.mlock),
                  firstGuess: Number(result.first_guess ?? 0),
                }
              : null,
          usedMb: numberMap(result.used_mb),
          limitMb: numberMap(result.limit_mb),
          kvMbPer1k: numberMap(result.kv_mb_per_1k),
          cappedBy: String(result.capped_by ?? "memory"),
          trainedContext: Number(result.trained_context ?? 0),
          marginMb: Number(result.margin_mb ?? 0),
          fullValidated: Boolean(result.full_validated),
          peakMb: result.peak_mb ? numberMap(result.peak_mb) : undefined,
          validationPromptTokens:
            result.validation_prompt_tokens == null
              ? null
              : Number(result.validation_prompt_tokens),
          reducedByValidation:
            result.reduced_by_validation == null
              ? undefined
              : Number(result.reduced_by_validation),
          vision: vision
            ? {
                projector: String(vision.projector ?? ""),
                costMb: numberMap(vision.cost_mb),
                maxContextWithVision: Number(vision.max_context_with_vision ?? 0),
              }
            : null,
        }
      : null,
  };
}


export function toRuntimePreset(raw: unknown): RuntimePreset {
  const data = isRecord(raw) ? raw : {};
  return {
    id: String(data.id ?? ""),
    profileId: String(data.profile_id ?? data.profileId ?? ""),
    presetKey: String(data.preset_key ?? data.presetKey ?? ""),
    label: String(data.label ?? ""),
    contextMode: String(
      data.context_mode ?? data.contextMode ?? "fixed",
    ) as RuntimePreset["contextMode"],
    contextTokens:
      data.context_tokens == null && data.contextTokens == null
        ? null
        : Number(data.context_tokens ?? data.contextTokens),
    parallelism: Number(data.parallelism ?? 2),
    kvCacheDtype: String(
      data.kv_cache_dtype ?? data.kvCacheDtype ?? "q8",
    ) as RuntimePreset["kvCacheDtype"],
    placementPolicy: String(
      data.placement_policy ?? data.placementPolicy ?? "auto",
    ) as RuntimePreset["placementPolicy"],
    preferredGpus: Array.isArray(data.preferred_gpus ?? data.preferredGpus)
      ? ((data.preferred_gpus ?? data.preferredGpus) as unknown[]).map(Number)
      : null,
    enabled: Boolean(data.enabled ?? true),
    isDefault: Boolean(data.is_default ?? data.isDefault ?? false),
    priority: Number(data.priority ?? 100),
    placement: toPresetPlacement(data.placement),
    createdAt: (data.created_at ?? data.createdAt ?? null) as string | null,
    updatedAt: (data.updated_at ?? data.updatedAt ?? null) as string | null,
  };
}


export function toPresetPlacement(raw: unknown): import("@/types").PresetPlacement | null {
  if (!isRecord(raw)) return null;
  const num = (v: unknown) => (v == null ? null : Number(v));
  return {
    splitPolicy: (raw.split_policy ?? raw.splitPolicy ?? null) as import("@/types").PresetPlacement["splitPolicy"],
    splitMode: (raw.split_mode ?? raw.splitMode ?? null) as import("@/types").PresetPlacement["splitMode"],
    tensorSplit: Array.isArray(raw.tensor_split ?? raw.tensorSplit)
      ? ((raw.tensor_split ?? raw.tensorSplit) as unknown[]).map(Number)
      : null,
    gpuLayers: num(raw.gpu_layers ?? raw.gpuLayers),
    nCpuMoe: num(raw.n_cpu_moe ?? raw.nCpuMoe),
  };
}


export function fromPresetPlacement(p: import("@/types").PresetPlacement | null): Record<string, unknown> | null {
  if (!p) return null;
  return {
    split_policy: p.splitPolicy ?? null,
    split_mode: p.splitMode ?? null,
    tensor_split: p.tensorSplit ?? null,
    gpu_layers: p.gpuLayers ?? null,
    n_cpu_moe: p.nCpuMoe ?? null,
  };
}


export function fromRuntimePreset(
  data: RuntimePresetCreate | RuntimePresetUpdate,
): Record<string, unknown> {
  return {
    ...("presetKey" in data && { preset_key: data.presetKey }),
    ...(data.label !== undefined && { label: data.label }),
    ...(data.contextMode !== undefined && { context_mode: data.contextMode }),
    ...(data.contextTokens !== undefined && {
      context_tokens: data.contextTokens,
    }),
    ...(data.parallelism !== undefined && { parallelism: data.parallelism }),
    ...(data.kvCacheDtype !== undefined && {
      kv_cache_dtype: data.kvCacheDtype,
    }),
    ...(data.placementPolicy !== undefined && {
      placement_policy: data.placementPolicy,
    }),
    ...(data.preferredGpus !== undefined && {
      preferred_gpus: data.preferredGpus,
    }),
    ...(data.enabled !== undefined && { enabled: data.enabled }),
    ...(data.isDefault !== undefined && { is_default: data.isDefault }),
    ...(data.priority !== undefined && { priority: data.priority }),
    ...(data.placement !== undefined && { placement: fromPresetPlacement(data.placement) }),
  };
}


export function toRuntimePresetResolution(raw: unknown): RuntimePresetResolution {
  const data = isRecord(raw) ? raw : {};
  return {
    presetKey: String(data.preset_key ?? data.presetKey ?? ""),
    resolved: Boolean(data.resolved),
    requestedContext: toNumberOrNull(
      data,
      "requested_context",
      "requestedContext",
    ),
    effectiveContext: toNumberOrNull(
      data,
      "effective_context",
      "effectiveContext",
    ),
    modelMax: toNumberOrNull(data, "model_max", "modelMax"),
    hardwareMax: toNumberOrNull(data, "hardware_max", "hardwareMax"),
    policyMax: toNumberOrNull(data, "policy_max", "policyMax"),
    requestedParallelism: Number(
      data.requested_parallelism ?? data.requestedParallelism ?? 1,
    ),
    effectiveParallelism: Number(
      data.effective_parallelism ?? data.effectiveParallelism ?? 1,
    ),
    targetGpus: Array.isArray(data.target_gpus ?? data.targetGpus)
      ? ((data.target_gpus ?? data.targetGpus) as unknown[]).map(Number)
      : [],
    fits: Boolean(data.fits),
    runtimeKey: (data.runtime_key ?? data.runtimeKey ?? null) as string | null,
    warnings: Array.isArray(data.warnings) ? data.warnings.map(String) : [],
    estimatedVramMb: toNumberOrNull(data, "estimated_vram_mb", "estimatedVramMb"),
    cpuMb: toNumberOrNull(data, "cpu_mb", "cpuMb"),
    vramByGpu: isRecord(data.vram_by_gpu ?? data.vramByGpu)
      ? Object.fromEntries(
          Object.entries((data.vram_by_gpu ?? data.vramByGpu) as Record<string, unknown>).map(([k, v]) => [k, Number(v)]),
        )
      : {},
    vramBreakdownByGpu: isRecord(data.vram_breakdown_by_gpu ?? data.vramBreakdownByGpu)
      ? Object.fromEntries(
          Object.entries((data.vram_breakdown_by_gpu ?? data.vramBreakdownByGpu) as Record<string, unknown>).map(([gpu, parts]) => [
            gpu,
            Object.fromEntries(Object.entries(isRecord(parts) ? parts : {}).map(([k, v]) => [k, Number(v)])),
          ]),
        )
      : {},
  };
}


export function toRuntimePresetBenchmark(raw: unknown): RuntimePresetBenchmark {
  const data = isRecord(raw) ? raw : {};
  return {
    presetKey: String(data.preset_key ?? data.presetKey ?? ""),
    effectiveContext: Number(
      data.effective_context ?? data.effectiveContext ?? 0,
    ),
    workerKey: String(data.worker_key ?? data.workerKey ?? ""),
    wasCold: Boolean(data.was_cold ?? data.wasCold),
    loadMs: Number(data.load_ms ?? data.loadMs ?? 0),
    generationMs: Number(data.generation_ms ?? data.generationMs ?? 0),
    outputTokens: toNumberOrNull(data, "output_tokens", "outputTokens"),
    tokensPerSecond: toNumberOrNull(
      data,
      "tokens_per_second",
      "tokensPerSecond",
    ),
  };
}


export function toModelProfile(raw: unknown): ModelProfile {
  const data = isRecord(raw) ? raw : {};
  return {
    profileId: String(data.profile_id ?? data.profileId ?? ""),
    profileKind: String(
      data.profile_kind ?? data.profileKind ?? "execution",
    ) as ModelProfile["profileKind"],
    baseModelId:
      data.base_model_id == null && data.baseModelId == null
        ? null
        : String(data.base_model_id ?? data.baseModelId),
    displayName: (data.display_name ?? data.displayName ?? null) as
      string | null,
    description: (data.description ?? null) as string | null,
    category: String(data.category ?? "llm") as ModelProfile["category"],
    loadOverrides: isRecord(data.load_overrides ?? data.loadOverrides)
      ? ((data.load_overrides ?? data.loadOverrides) as Record<string, unknown>)
      : null,
    requestDefaults: isRecord(data.request_defaults ?? data.requestDefaults)
      ? ((data.request_defaults ?? data.requestDefaults) as Record<
          string,
          unknown
        >)
      : null,
    assets: isRecord(data.assets)
      ? (data.assets as Record<string, unknown>)
      : null,
    enabled: Boolean(data.enabled ?? true),
    isDefault: Boolean(data.is_default ?? data.isDefault ?? false),
    routingTargets: Array.isArray(data.routing_targets ?? data.routingTargets)
      ? ((data.routing_targets ?? data.routingTargets) as unknown[]).map(String)
      : null,
    routerContext: toRouterContext(data.router_context ?? data.routerContext),
    runtimePolicy: toRuntimePolicy(data.runtime_policy ?? data.runtimePolicy),
    runtimePresets: Array.isArray(data.runtime_presets ?? data.runtimePresets)
      ? ((data.runtime_presets ?? data.runtimePresets) as unknown[]).map(
          toRuntimePreset,
        )
      : [],
    aliases: Array.isArray(data.aliases)
      ? data.aliases.map((rawAlias) => {
          const alias = isRecord(rawAlias) ? rawAlias : {};
          return {
            aliasId: String(alias.alias_id ?? alias.aliasId ?? ""),
            profileId: String(alias.profile_id ?? alias.profileId ?? ""),
            presetKey: (alias.preset_key ?? alias.presetKey ?? null) as
              string | null,
            hidden: Boolean(alias.hidden ?? true),
          };
        })
      : [],
    createdAt: (data.created_at ?? data.createdAt ?? null) as string | null,
    updatedAt: (data.updated_at ?? data.updatedAt ?? null) as string | null,
  };
}


export function toOpenAIImageModelMapEntry(raw: unknown): OpenAIImageModelMapEntry {
  const data = isRecord(raw) ? raw : {};
  return {
    openaiModel: String(data.openai_model ?? data.openaiModel ?? ""),
    targetProfileId: String(data.target_profile_id ?? data.targetProfileId ?? ""),
    createdAt: (data.created_at ?? data.createdAt ?? null) as string | null,
    updatedAt: (data.updated_at ?? data.updatedAt ?? null) as string | null,
  };
}


export function toOpenAITTSVoiceMapEntry(raw: unknown): OpenAITTSVoiceMapEntry {
  const data = isRecord(raw) ? raw : {};
  return {
    openaiModel: String(data.openai_model ?? data.openaiModel ?? ""),
    voiceName: String(data.voice_name ?? data.voiceName ?? ""),
    targetProfileId: String(data.target_profile_id ?? data.targetProfileId ?? ""),
    createdAt: (data.created_at ?? data.createdAt ?? null) as string | null,
    updatedAt: (data.updated_at ?? data.updatedAt ?? null) as string | null,
  };
}


export function toModelState(raw: unknown): ModelState {
  const data = isRecord(raw) ? raw : {};
  const schedulesRaw = data.schedules;
  const schedules = Array.isArray(schedulesRaw)
    ? schedulesRaw.map((schedule, idx) => {
        const s = isRecord(schedule) ? schedule : {};
        return {
          id: String(s.id ?? `schedule-${idx}`),
          days: Array.isArray(s.days)
            ? s.days.map((d) => Number(d)).filter((d) => Number.isFinite(d))
            : [],
          start: String(s.start ?? "00:00"),
          end: String(s.end ?? "00:00"),
          enabled: Boolean(s.enabled ?? true),
        };
      })
    : undefined;
  return {
    modelId: String(data.model_id ?? data.modelId ?? ""),
    backendModelId: (data.backend_model_id ?? data.backendModelId ?? null) as
      string | null,
    maintenance: Boolean(data.maintenance),
    displayName: String(
      data.display_name ??
        data.displayName ??
        data.model_id ??
        data.modelId ??
        "Unknown model",
    ),
    backendType: String(
      data.backend_type ?? data.backendType ?? "vllm",
    ) as ModelState["backendType"],
    status: String(data.status ?? "configured") as ModelState["status"],
    loadPolicy: String(
      data.load_policy ?? data.loadPolicy ?? "on_demand",
    ) as ModelState["loadPolicy"],
    pinPermanent: Boolean(data.pin_permanent ?? data.pinPermanent),
    preferredGpu: (data.preferred_gpu ?? data.preferredGpu ?? null) as
      number | null,
    currentGpu: Array.isArray(data.current_gpu ?? data.currentGpu)
      ? ((data.current_gpu ?? data.currentGpu) as unknown[]).map((gpu) =>
          Number(gpu),
        )
      : [],
    vramUsedMb: Number(data.vram_used_mb ?? data.vramUsedMb ?? 0),
    vramEstimateMb: Number(data.vram_estimate_mb ?? data.vramEstimateMb ?? 0),
    visionState: (data.vision_state ?? data.visionState ?? null) as
      | "active"
      | "inactive"
      | null,
    ramUsedMb: Number(data.ram_used_mb ?? data.ramUsedMb ?? 0),
    diskSizeBytes:
      data.disk_size_bytes == null && data.diskSizeBytes == null
        ? null
        : Number(data.disk_size_bytes ?? data.diskSizeBytes ?? 0),
    capabilities: toModelCapabilities(data.capabilities),
    lastRequestAt: (data.last_request_at ?? data.lastRequestAt ?? null) as
      string | null,
    loadedAt: (data.loaded_at ?? data.loadedAt ?? null) as string | null,
    errorMessage: (data.error_message ?? data.errorMessage ?? null) as
      string | null,
    runtimePhase: String(
      data.runtime_phase ?? data.runtimePhase ?? data.status ?? "configured",
    ) as ModelState["runtimePhase"],
    runtimeWorkerId: (data.runtime_worker_id ?? data.runtimeWorkerId ?? null) as
      string | null,
    runtimeSlots: Math.max(
      1,
      Number(data.runtime_slots ?? data.runtimeSlots ?? 1),
    ),
    canCancelLoad: Boolean(
      data.can_cancel_load ?? data.canCancelLoad ?? false,
    ),
    canDownsize: Boolean(data.can_downsize ?? data.canDownsize ?? false),
    liveContext: isRecord(data.live_context ?? data.liveContext)
      ? ((data.live_context ?? data.liveContext) as Record<string, unknown>)
      : null,
    schedules,
    extraConfig: toBackendExtraConfig(data.extra_config ?? data.extraConfig),
    profiles: Array.isArray(data.profiles)
      ? data.profiles.map(toModelProfile)
      : undefined,
    federation: (() => {
      // The backend sends {local, remote, nodes: [{node_name, node_id, loaded}]};
      // this used to require a ``remote`` key it never sent, so the
      // "federated model" badge never showed.
      const fed = isRecord(data.federation) ? data.federation : null;
      if (!fed) return null;
      const nodes = Array.isArray(fed.nodes) ? fed.nodes.filter(isRecord) : [];
      const loadedOn = nodes
        .filter((n) => n.loaded !== false)
        .map((n) => (n.node_name === "local" ? "aquí" : String(n.node_name ?? n.node_id ?? "")))
        .filter(Boolean);
      const remote = fed.remote === true || fed.local === false;
      const first = nodes.find((n) => n.node_name !== "local");
      if (!remote && !loadedOn.some((n) => n !== "aquí")) return null;
      return {
        remote,
        nodeName: String(first?.node_name ?? ""),
        nodeId: String(first?.node_id ?? ""),
        loadedOn,
      };
    })(),
    releaseDate: (data.release_date ?? data.releaseDate ?? null) as
      string | null,
    lastUpdated: (data.last_updated ?? data.lastUpdated ?? null) as
      string | null,
    metadataFetchedAt: (data.metadata_fetched_at ??
      data.metadataFetchedAt ??
      null) as string | null,
  };
}


export function toCompatibleBackends(raw: unknown): CompatibleBackend[] {
  if (!Array.isArray(raw)) return [];
  return raw.filter(isRecord).map((item) => ({
    backend: String(item.backend ?? "") as BackendType,
    status: (item.status === "probable" ? "probable" : "compatible") as
      CompatibleBackend["status"],
    installed: item.installed == null ? true : Boolean(item.installed),
    reason: String(item.reason ?? ""),
    ...(item.deployed == null ? {} : { deployed: Boolean(item.deployed) }),
  }));
}


export function toModelDeployment(raw: unknown): ModelDeployment {
  const data = isRecord(raw) ? raw : {};
  const artifact = isRecord(data.artifact) ? data.artifact : null;
  return {
    modelId: String(data.modelId ?? data.model_id ?? ""),
    backend: String(data.backend ?? "") as BackendType,
    isDefault: Boolean(data.isDefault ?? data.is_default),
    installed: (data.installed ?? true) !== false,
    status: (data.status ?? null) as ModelDeployment["status"],
    loadPolicy: String(
      data.loadPolicy ?? data.load_policy ?? "on_demand",
    ) as LoadPolicy,
    pinPermanent: Boolean(data.pinPermanent ?? data.pin_permanent),
    preferredGpu: (() => {
      const gpu = data.preferredGpu ?? data.preferred_gpu;
      return gpu == null || gpu === "" ? null : Number(gpu);
    })(),
    artifact: artifact as ModelDeployment["artifact"],
    profiles: Array.isArray(data.profiles) ? data.profiles.map(String) : [],
  };
}


export function toLogicalModel(raw: unknown): LogicalModel {
  const data = isRecord(raw) ? raw : {};
  const modelKey = String(data.modelKey ?? data.model_key ?? "");
  const registryMetadata = data.registryMetadata ?? data.registry_metadata;
  return {
    modelKey,
    displayName: String(data.displayName ?? data.display_name ?? modelKey),
    category: (data.category ?? null) as string | null,
    defaultBackend: String(
      data.defaultBackend ?? data.default_backend ?? "",
    ) as BackendType,
    deployments: Array.isArray(data.deployments)
      ? data.deployments.map(toModelDeployment)
      : [],
    compatibleBackends: toCompatibleBackends(
      data.compatibleBackends ?? data.compatible_backends,
    ),
    publicProfiles: Array.isArray(data.publicProfiles ?? data.public_profiles)
      ? ((data.publicProfiles ?? data.public_profiles) as unknown[]).map(String)
      : [],
    hiddenProfiles: Array.isArray(data.hiddenProfiles ?? data.hidden_profiles)
      ? ((data.hiddenProfiles ?? data.hidden_profiles) as unknown[]).map(String)
      : [],
    registryMetadata: isRecord(registryMetadata) ? registryMetadata : null,
  };
}


export const modelsApiModule = {
  models: {
    list: async () =>
      (await request<unknown[]>("GET", "/ocabra/models")).map(toModelState),
    storage: async () =>
      toModelsStorageStats(
        await request<unknown>("GET", "/ocabra/models/storage"),
      ),
    get: async (modelId: string) =>
      toModelState(
        await request<unknown>(
          "GET",
          `/ocabra/models/${encodeURIComponent(modelId)}`,
        ),
      ),
    activity: () =>
      request<import("@/types").ModelActivityMap>(
        "GET",
        "/ocabra/models/activity",
      ),
    ollamaRuntime: async () => {
      const data = await request<{
        models?: Record<string, import("@/types").OllamaRuntimeInfo>;
      }>("GET", "/ocabra/models/ollama-runtime");
      return data.models ?? {};
    },
    capacity: (
      modelId: string,
      params?: {
        gpu?: number;
        slots?: string;
        kvDtype?: string;
        context?: string;
      },
    ) =>
      request<import("@/types").ModelCapacity>(
        "GET",
        `/ocabra/models/${encodeURIComponent(modelId)}/capacity${buildQuery({
          gpu: params?.gpu,
          slots: params?.slots,
          kv_dtype: params?.kvDtype,
          context: params?.context,
        })}`,
      ),
    load: async (modelId: string) =>
      toModelState(
        await request<unknown>(
          "POST",
          `/ocabra/models/${encodeURIComponent(modelId)}/load`,
        ),
      ),
    unload: (modelId: string) =>
      request<void>(
        "POST",
        `/ocabra/models/${encodeURIComponent(modelId)}/unload`,
      ),
    cancelLoad: (modelId: string) =>
      request<{ model_id: string; cancelled_workers: string[] }>(
        "POST",
        `/ocabra/models/${encodeURIComponent(modelId)}/cancel-load`,
      ),
    downsize: (modelId: string) =>
      request<{ model_id: string; downsized_worker: string }>(
        "POST",
        `/ocabra/models/${encodeURIComponent(modelId)}/downsize`,
      ),
    delete: (modelId: string) =>
      request<void>("DELETE", `/ocabra/models/${encodeURIComponent(modelId)}`),
    update: async (modelId: string) =>
      toDownloadJob(
        await request<unknown>(
          "POST",
          `/ocabra/models/${encodeURIComponent(modelId)}/update`,
        ),
      ),
    refreshMetadata: async (): Promise<{
      refreshed: string[];
      skipped: { modelId: string; reason: string }[];
    }> => {
      const raw = await request<{
        refreshed: string[];
        skipped: { model_id: string; reason: string }[];
      }>("POST", "/ocabra/models/refresh-metadata");
      return {
        refreshed: raw.refreshed ?? [],
        skipped: (raw.skipped ?? []).map((s) => ({
          modelId: s.model_id,
          reason: s.reason,
        })),
      };
    },
    updateAll: async (): Promise<{
      enqueued: DownloadJob[];
      skipped: { modelId: string; reason: string }[];
    }> => {
      const raw = await request<{
        enqueued: unknown[];
        skipped: { model_id: string; reason: string }[];
      }>("POST", "/ocabra/models/update-all");
      return {
        enqueued: (raw.enqueued ?? []).map(toDownloadJob),
        skipped: (raw.skipped ?? []).map((s) => ({
          modelId: s.model_id,
          reason: s.reason,
        })),
      };
    },
    // Sprint 17.4 — list llama.cpp models compatible as speculative-decoding drafts.
    speculativeCandidates: async (
      modelId: string,
    ): Promise<
      {
        modelId: string;
        displayName: string;
        vocabSize: number | null;
        bosId: number | null;
        eosId: number | null;
      }[]
    > => {
      const rows = await request<
        {
          model_id: string;
          display_name: string;
          vocab_size: number | null;
          bos_id: number | null;
          eos_id: number | null;
        }[]
      >(
        "GET",
        `/ocabra/models/${encodeURIComponent(modelId)}/speculative-candidates`,
      );
      return rows.map((row) => ({
        modelId: row.model_id,
        displayName: row.display_name,
        vocabSize: row.vocab_size,
        bosId: row.bos_id,
        eosId: row.eos_id,
      }));
    },
  },

  /**
   * Model-centric view (model/backends unification): one entry per public
   * model name with its per-backend deployments. `/ocabra/models` keeps
   * listing deployments. Keys may contain ':' and '/'; the server routes take
   * `{model_key:path}`, so percent-encoding the slash is fine (same as the
   * `/ocabra/models/{id}` calls above).
   */
  logicalModels: {
    list: async (): Promise<LogicalModel[]> =>
      (await request<unknown[]>("GET", "/ocabra/logical-models")).map(
        toLogicalModel,
      ),
    get: async (modelKey: string): Promise<LogicalModel> =>
      toLogicalModel(
        await request<unknown>(
          "GET",
          `/ocabra/logical-models/${encodeURIComponent(modelKey)}`,
        ),
      ),
    patch: async (
      modelKey: string,
      body: { defaultBackend?: BackendType; displayName?: string },
    ): Promise<LogicalModel> =>
      toLogicalModel(
        await request<unknown>(
          "PATCH",
          `/ocabra/logical-models/${encodeURIComponent(modelKey)}`,
          body,
        ),
      ),
    addDeployment: async (
      modelKey: string,
      body: {
        backend: BackendType;
        sourceBackend?: BackendType;
        extraConfig?: Record<string, unknown>;
      },
    ): Promise<LogicalModel> =>
      toLogicalModel(
        await request<unknown>(
          "POST",
          `/ocabra/logical-models/${encodeURIComponent(modelKey)}/deployments`,
          body,
        ),
      ),
    removeDeployment: async (
      modelKey: string,
      backend: BackendType,
    ): Promise<RemoveDeploymentResult> => {
      const data = await request<Record<string, unknown>>(
        "DELETE",
        `/ocabra/logical-models/${encodeURIComponent(modelKey)}/deployments/${encodeURIComponent(backend)}`,
      );
      return {
        modelKey: String(data.modelKey ?? data.model_key ?? modelKey),
        removed: String(data.removed ?? ""),
        newDefaultBackend: (data.newDefaultBackend ??
          data.new_default_backend ??
          null) as BackendType | null,
      };
    },
  },

  profiles: {
    // Bloque 20 — every profile in the registry. Used by the Routers admin
    // page to filter for profiles whose ``routing_targets`` is non-empty.
    listAll: async (): Promise<ModelProfile[]> =>
      (await request<unknown[]>("GET", "/ocabra/profiles")).map(toModelProfile),

    listByModel: async (modelId: string): Promise<ModelProfile[]> =>
      (
        await request<unknown[]>(
          "GET",
          `/ocabra/models/${encodeURIComponent(modelId)}/profiles`,
        )
      ).map(toModelProfile),

    create: async (
      modelId: string,
      data: ProfileCreate,
    ): Promise<ModelProfile> =>
      toModelProfile(
        await request<unknown>(
          "POST",
          `/ocabra/models/${encodeURIComponent(modelId)}/profiles`,
          {
            profile_id: data.profileId,
            display_name: data.displayName,
            description: data.description,
            category: data.category,
            load_overrides: data.loadOverrides,
            request_defaults: data.requestDefaults,
            enabled: data.enabled,
            is_default: data.isDefault,
            ...(data.routingTargets !== undefined && {
              routing_targets: data.routingTargets,
            }),
            ...(data.runtimePolicy !== undefined && {
              runtime_policy: fromRuntimePolicy(data.runtimePolicy),
            }),
          },
        ),
      ),

    createRouter: async (data: RouterCreate): Promise<ModelProfile> =>
      toModelProfile(
        await request<unknown>("POST", "/ocabra/profiles", {
          profile_id: data.profileId,
          display_name: data.displayName,
          description: data.description,
          category: data.category,
          request_defaults: data.requestDefaults,
          enabled: data.enabled,
          routing_targets: data.routingTargets,
          ...(data.routerContext !== undefined && {
            router_context: data.routerContext,
          }),
        }),
      ),

    update: async (
      profileId: string,
      data: ProfileUpdate,
    ): Promise<ModelProfile> =>
      toModelProfile(
        await request<unknown>(
          "PATCH",
          `/ocabra/profiles/${encodeURIComponent(profileId)}`,
          {
            ...(data.displayName !== undefined && {
              display_name: data.displayName,
            }),
            ...(data.description !== undefined && {
              description: data.description,
            }),
            ...(data.category !== undefined && { category: data.category }),
            ...(data.loadOverrides !== undefined && {
              load_overrides: data.loadOverrides,
            }),
            ...(data.requestDefaults !== undefined && {
              request_defaults: data.requestDefaults,
            }),
            ...(data.enabled !== undefined && { enabled: data.enabled }),
            ...(data.isDefault !== undefined && { is_default: data.isDefault }),
            ...(data.routingTargets !== undefined && {
              routing_targets: data.routingTargets,
            }),
            ...(data.runtimePolicy !== undefined && {
              runtime_policy: fromRuntimePolicy(data.runtimePolicy),
            }),
            ...(data.routerContext !== undefined && {
              router_context: data.routerContext,
            }),
          },
        ),
      ),

    resolveRuntimePresets: async (
      profileId: string,
    ): Promise<RuntimePresetResolution[]> =>
      (
        await request<unknown[]>(
          "GET",
          `/ocabra/profiles/${encodeURIComponent(profileId)}/runtime-presets/resolved`,
        )
      ).map(toRuntimePresetResolution),

    startContextProbe: async (
      profileId: string,
      presetKey: string,
      options: { fullValidation: boolean; measureVision: boolean; targetContext?: number | null },
    ): Promise<ContextProbeJob> =>
      toContextProbeJob(
        await request<unknown>(
          "POST",
          `/ocabra/profiles/${encodeURIComponent(profileId)}/runtime-presets/${encodeURIComponent(presetKey)}/context-probe`,
          {
            full_validation: options.fullValidation,
            measure_vision: options.measureVision,
            ...(options.targetContext ? { target_context: options.targetContext } : {}),
          },
        ),
      ),

    getContextProbe: async (
      profileId: string,
      presetKey: string,
    ): Promise<ContextProbeJob> =>
      toContextProbeJob(
        await request<unknown>(
          "GET",
          `/ocabra/profiles/${encodeURIComponent(profileId)}/runtime-presets/${encodeURIComponent(presetKey)}/context-probe`,
        ),
      ),

    applyContextProbe: async (profileId: string, presetKey: string): Promise<void> => {
      await request<unknown>(
        "POST",
        `/ocabra/profiles/${encodeURIComponent(profileId)}/runtime-presets/${encodeURIComponent(presetKey)}/context-probe/apply`,
      );
    },

    benchmarkRuntimePreset: async (
      profileId: string,
      presetKey: string,
      coldStart = true,
    ): Promise<RuntimePresetBenchmark> =>
      toRuntimePresetBenchmark(
        await request<unknown>(
          "POST",
          `/ocabra/profiles/${encodeURIComponent(profileId)}/runtime-presets/${encodeURIComponent(presetKey)}/benchmark`,
          { cold_start: coldStart },
        ),
      ),

    previewRuntimePreset: async (
      profileId: string,
      data: RuntimePresetCreate,
    ): Promise<RuntimePresetResolution> =>
      toRuntimePresetResolution(
        await request<unknown>(
          "POST",
          `/ocabra/profiles/${encodeURIComponent(profileId)}/runtime-presets/preview`,
          fromRuntimePreset(data),
        ),
      ),

    createRuntimePreset: async (
      profileId: string,
      data: RuntimePresetCreate,
    ): Promise<RuntimePreset> =>
      toRuntimePreset(
        await request<unknown>(
          "POST",
          `/ocabra/profiles/${encodeURIComponent(profileId)}/runtime-presets`,
          fromRuntimePreset(data),
        ),
      ),

    updateRuntimePreset: async (
      profileId: string,
      presetKey: string,
      data: RuntimePresetUpdate,
    ): Promise<RuntimePreset> =>
      toRuntimePreset(
        await request<unknown>(
          "PATCH",
          `/ocabra/profiles/${encodeURIComponent(profileId)}/runtime-presets/${encodeURIComponent(presetKey)}`,
          fromRuntimePreset(data),
        ),
      ),

    deleteRuntimePreset: async (
      profileId: string,
      presetKey: string,
    ): Promise<void> => {
      await request<unknown>(
        "DELETE",
        `/ocabra/profiles/${encodeURIComponent(profileId)}/runtime-presets/${encodeURIComponent(presetKey)}`,
      );
    },

    delete: async (profileId: string): Promise<void> => {
      await request<unknown>(
        "DELETE",
        `/ocabra/profiles/${encodeURIComponent(profileId)}`,
      );
    },

    rename: async (
      profileId: string,
      newProfileId: string,
    ): Promise<ModelProfile> =>
      toModelProfile(
        await request<unknown>(
          "POST",
          `/ocabra/profiles/${encodeURIComponent(profileId)}/rename`,
          { new_profile_id: newProfileId },
        ),
      ),

    clone: async (
      profileId: string,
      newProfileId: string,
      displayName?: string,
    ): Promise<ModelProfile> =>
      toModelProfile(
        await request<unknown>(
          "POST",
          `/ocabra/profiles/${encodeURIComponent(profileId)}/clone`,
          {
            new_profile_id: newProfileId,
            ...(displayName !== undefined && { display_name: displayName }),
          },
        ),
      ),

    uploadAsset: async (
      profileId: string,
      file: File,
    ): Promise<ModelProfile> => {
      const form = new FormData();
      form.append("file", file);
      const res = await httpFetch(
        `/ocabra/profiles/${encodeURIComponent(profileId)}/assets`,
        { method: "POST", body: form },
      );
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(extractErrorMessage(err) ?? res.statusText);
      }
      const json = (await res.json()) as Record<string, unknown>;
      return toModelProfile(json.profile);
    },

    deleteAsset: async (
      profileId: string,
      assetKey: string,
    ): Promise<ModelProfile> => {
      const raw = await request<Record<string, unknown>>(
        "DELETE",
        `/ocabra/profiles/${encodeURIComponent(profileId)}/assets/${encodeURIComponent(assetKey)}`,
      );
      return toModelProfile(raw.profile);
    },
  },

  // Compatibilidad de voces OpenAI (independiente de los routers): fija a
  // qué perfil propio responde cada (modelo oficial de OpenAI, voz) para
  // un cliente que solo sabe mandar esos nombres.
  openaiTtsCompat: {
    list: async (): Promise<OpenAITTSVoiceMapEntry[]> =>
      (await request<unknown[]>("GET", "/ocabra/openai-tts-voice-map")).map(
        toOpenAITTSVoiceMapEntry,
      ),

    upsert: async (
      openaiModel: string,
      voiceName: string,
      targetProfileId: string,
    ): Promise<OpenAITTSVoiceMapEntry> =>
      toOpenAITTSVoiceMapEntry(
        await request<unknown>(
          "PUT",
          `/ocabra/openai-tts-voice-map/${encodeURIComponent(openaiModel)}/${encodeURIComponent(voiceName)}`,
          { target_profile_id: targetProfileId },
        ),
      ),

    delete: async (openaiModel: string, voiceName: string): Promise<void> => {
      await request<unknown>(
        "DELETE",
        `/ocabra/openai-tts-voice-map/${encodeURIComponent(openaiModel)}/${encodeURIComponent(voiceName)}`,
      );
    },
  },

  openaiImageCompat: {
    list: async (): Promise<OpenAIImageModelMapEntry[]> =>
      (await request<unknown[]>("GET", "/ocabra/openai-image-model-map")).map(
        toOpenAIImageModelMapEntry,
      ),

    upsert: async (
      openaiModel: string,
      targetProfileId: string,
    ): Promise<OpenAIImageModelMapEntry> =>
      toOpenAIImageModelMapEntry(
        await request<unknown>(
          "PUT",
          `/ocabra/openai-image-model-map/${encodeURIComponent(openaiModel)}`,
          { target_profile_id: targetProfileId },
        ),
      ),

    delete: async (openaiModel: string): Promise<void> => {
      await request<unknown>(
        "DELETE",
        `/ocabra/openai-image-model-map/${encodeURIComponent(openaiModel)}`,
      );
    },
  },

};
