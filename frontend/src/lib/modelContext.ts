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
  return getConfigSection<LlamaCppConfig>(model, "llama_cpp")
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
  const nativeContext =
    typeof model.capabilities.contextLength === "number" && model.capabilities.contextLength > 0
      ? model.capabilities.contextLength
      : null

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

  configuredContext ??= nativeContext

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
