import type {
  BackendExtraConfig,
  BitNetConfig,
  LlamaCppConfig,
  ModelState,
  SGLangConfig,
  TensorRTLLMConfig,
  VLLMConfig,
  WhisperConfig,
} from "@/types"

export interface ModelContextSummary {
  nativeContext: number | null
  configuredContext: number | null
  maxInputTokens: number | null
  maxOutputTokens: number | null
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null
}

function getConfigSection<T>(model: ModelState, key: keyof BackendExtraConfig): T | null {
  const extraConfig = model.extraConfig
  if (!isRecord(extraConfig)) return null
  const nested = extraConfig[key]
  if (isRecord(nested)) return nested as T
  return extraConfig as T
}

function toPositiveNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? value : null
}

export function getWritableExtraConfig(model: ModelState): BackendExtraConfig {
  return isRecord(model.extraConfig) ? ({ ...model.extraConfig } as BackendExtraConfig) : {}
}

export function getVllmConfig(model: ModelState): VLLMConfig | null {
  if (model.backendType !== "vllm") return null
  return getConfigSection<VLLMConfig>(model, "vllm")
}

export function getSGLangConfig(model: ModelState): SGLangConfig | null {
  if (model.backendType !== "sglang") return null
  return getConfigSection<SGLangConfig>(model, "sglang")
}

export function getLlamaCppConfig(model: ModelState): LlamaCppConfig | null {
  if (model.backendType !== "llama_cpp") return null
  const raw = getConfigSection<Record<string, unknown>>(model, "llama_cpp")
  if (raw == null) return null
  // Accept both camelCase (frontend write path) and snake_case (backend
  // schemas / persisted state) for the Sprint 17.3 multi-GPU + MoE knobs.
  const config: LlamaCppConfig = { ...(raw as LlamaCppConfig) }
  if (config.mainGpu == null && typeof raw.main_gpu === "number") config.mainGpu = raw.main_gpu
  if (config.tensorSplit == null && Array.isArray(raw.tensor_split)) {
    config.tensorSplit = (raw.tensor_split as unknown[]).filter(
      (v): v is number => typeof v === "number" && Number.isFinite(v),
    )
  }
  if (config.splitMode == null && typeof raw.split_mode === "string") {
    config.splitMode = raw.split_mode as LlamaCppConfig["splitMode"]
  }
  if (config.disabledGpus == null && Array.isArray(raw.disabled_gpus)) {
    config.disabledGpus = (raw.disabled_gpus as unknown[]).filter(
      (v): v is number => typeof v === "number" && Number.isInteger(v),
    )
  }
  if (config.splitStrategy == null && typeof raw.split_strategy === "string") {
    config.splitStrategy = raw.split_strategy as LlamaCppConfig["splitStrategy"]
  }
  if (config.nCpuMoe == null && typeof raw.n_cpu_moe === "number") config.nCpuMoe = raw.n_cpu_moe
  if (config.overrideTensor == null && typeof raw.override_tensor === "string") {
    config.overrideTensor = raw.override_tensor
  }
  return config
}

export function getBitNetConfig(model: ModelState): BitNetConfig | null {
  if (model.backendType !== "bitnet") return null
  return getConfigSection<BitNetConfig>(model, "bitnet")
}

export function getTensorRTLLMConfig(model: ModelState): TensorRTLLMConfig | null {
  if (model.backendType !== "tensorrt_llm") return null
  return getConfigSection<TensorRTLLMConfig>(model, "tensorrt_llm")
}

export function getWhisperConfig(model: ModelState): WhisperConfig | null {
  if (model.backendType !== "whisper") return null
  return getConfigSection<WhisperConfig>(model, "whisper")
}

export function getModelContextSummary(model: ModelState): ModelContextSummary {
  // Prefer the actual native ceiling (e.g. 262144 for gemma4:26b) when the
  // backend reports it; fall back to the served context otherwise. Before,
  // "nativo" showed OLLAMA_CONTEXT_LENGTH (65536) which is the effective cap,
  // not what the model was trained for.
  const nativeFromBackend =
    typeof model.capabilities.nativeContextLength === "number" && model.capabilities.nativeContextLength > 0
      ? model.capabilities.nativeContextLength
      : null
  const servedContext =
    typeof model.capabilities.contextLength === "number" && model.capabilities.contextLength > 0
      ? model.capabilities.contextLength
      : null
  const nativeContext = nativeFromBackend ?? servedContext

  let configuredContext: number | null = null
  switch (model.backendType) {
    case "vllm": {
      const vllm = getVllmConfig(model)
      configuredContext = toPositiveNumber(vllm?.maxModelLen)
      break
    }
    case "sglang": {
      const sglang = getSGLangConfig(model)
      configuredContext = toPositiveNumber(sglang?.contextLength ?? (sglang as Record<string, unknown> | null)?.context_length)
      break
    }
    case "llama_cpp": {
      const llamaCpp = getLlamaCppConfig(model)
      configuredContext = toPositiveNumber(llamaCpp?.ctxSize ?? (llamaCpp as Record<string, unknown> | null)?.ctx_size)
      break
    }
    case "bitnet": {
      const bitnet = getBitNetConfig(model)
      configuredContext = toPositiveNumber(bitnet?.ctxSize ?? (bitnet as Record<string, unknown> | null)?.ctx_size)
      break
    }
    case "tensorrt_llm": {
      const tensorrt = getTensorRTLLMConfig(model)
      configuredContext = toPositiveNumber(
        tensorrt?.contextLength ?? (tensorrt as Record<string, unknown> | null)?.context_length,
      )
      break
    }
  }

  configuredContext ??= servedContext ?? nativeContext

  return {
    nativeContext,
    configuredContext,
    maxInputTokens: configuredContext,
    maxOutputTokens: configuredContext,
  }
}

export function formatTokenCount(value: number | null): string {
  if (!value || value <= 0) return "-"
  if (value >= 1000) {
    const compact = value / 1000
    return Number.isInteger(compact) ? `${compact}k` : `${compact.toFixed(1)}k`
  }
  return String(value)
}

export type ModelCategory = "chat" | "embeddings" | "image" | "tts" | "stt" | "music" | "video" | "other"

export const MODEL_CATEGORY_ORDER: ModelCategory[] = [
  "chat",
  "embeddings",
  "image",
  "tts",
  "stt",
  "music",
  "video",
  "other",
]

export const MODEL_CATEGORY_LABELS: Record<ModelCategory, string> = {
  chat: "Chat",
  embeddings: "Embeddings",
  image: "Imagen",
  tts: "TTS",
  stt: "STT",
  music: "Música",
  video: "Video",
  other: "Otros",
}

// Capability-first, backendType as a fallback for backends the capability
// flags don't fully cover yet (e.g. video upscalers — BackendCapabilities
// has video_upscaling server-side but it isn't wired through to the
// frontend ModelCapabilities type).
export function categorizeModel(model: ModelState): ModelCategory {
  const c = model.capabilities
  if (c.audioTranscription) return "stt"
  if (c.tts) return "tts"
  if (c.musicGeneration) return "music"
  if (c.imageGeneration || c.imageEditing) return "image"
  if (model.backendType === "upscaler" || model.backendType === "flashvsr") return "video"
  if (c.embeddings || c.pooling || c.rerank || c.classification || c.score) return "embeddings"
  if (c.chat || c.completion) return "chat"
  return "other"
}
