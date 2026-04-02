import type { ModelState, VLLMConfig } from "@/types"

export interface ModelContextSummary {
  nativeContext: number | null
  configuredContext: number | null
  maxInputTokens: number | null
  maxOutputTokens: number | null
}

function getVllmConfig(model: ModelState): VLLMConfig | null {
  const vllm = model.extraConfig?.vllm
  if (vllm && typeof vllm === "object") return vllm as VLLMConfig
  if (model.extraConfig && typeof model.extraConfig === "object") return model.extraConfig as VLLMConfig
  return null
}

export function getModelContextSummary(model: ModelState): ModelContextSummary {
  const vllm = getVllmConfig(model)
  const nativeContext =
    typeof model.capabilities.contextLength === "number" && model.capabilities.contextLength > 0
      ? model.capabilities.contextLength
      : null
  const configuredContext =
    typeof vllm?.maxModelLen === "number" && vllm.maxModelLen > 0
      ? vllm.maxModelLen
      : nativeContext

  return {
    nativeContext,
    configuredContext,
    // In vLLM the request must fit into a shared context window.
    // Showing both bounds as the configured window is clearer than inventing
    // a fixed split that the runtime does not enforce.
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
