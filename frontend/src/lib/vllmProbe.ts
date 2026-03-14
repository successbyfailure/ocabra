import type { HFVLLMRuntimeProbe } from "@/types"

export function getProbeStatusLabel(
  status: HFVLLMRuntimeProbe["status"] | null | undefined,
): string | null {
  switch (status) {
    case "supported_native":
      return "soporte nativo verificado"
    case "supported_transformers_backend":
      return "compatibilidad verificada via Transformers backend"
    case "supported_pooling":
      return "modo pooling verificado"
    case "needs_remote_code":
      return "requiere trust_remote_code"
    case "missing_chat_template":
      return "falta chat_template"
    case "missing_tool_parser":
      return "falta tool_call_parser"
    case "missing_reasoning_parser":
      return "falta reasoning_parser"
    case "needs_hf_overrides":
      return "faltan hf_overrides"
    case "unsupported_tokenizer":
      return "tokenizer no soportado"
    case "unsupported_architecture":
      return "arquitectura no soportada"
    case "unavailable":
      return "probe no disponible"
    case "unknown":
    default:
      return null
  }
}

export function getProbeOverrideHint(
  status: HFVLLMRuntimeProbe["status"] | null | undefined,
): string | null {
  switch (status) {
    case "needs_remote_code":
      return "override sugerido: trust_remote_code"
    case "missing_chat_template":
      return "override sugerido: chat_template"
    case "missing_tool_parser":
      return "override sugerido: tool_call_parser"
    case "missing_reasoning_parser":
      return "override sugerido: reasoning_parser"
    case "needs_hf_overrides":
      return "override sugerido: hf_overrides"
    default:
      return null
  }
}
