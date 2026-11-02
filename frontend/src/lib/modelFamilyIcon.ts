import {
  siDeepmind,
  siDeepseek,
  siMeta,
  siMistralai,
  siNvidia,
  siOllama,
  siQwen,
  siVllm,
  type SimpleIcon,
} from "simple-icons"

// Order matters: more specific patterns first. "nemotron-3-nano" must match
// nvidia (nemotron) before any llama-family pattern would otherwise apply.
const FAMILY_PATTERNS: [RegExp, SimpleIcon][] = [
  [/qwen/i, siQwen],
  [/gemma/i, siDeepmind],
  [/nemotron/i, siNvidia],
  [/deepseek/i, siDeepseek],
  [/mistral/i, siMistralai],
  [/\bllama\b/i, siMeta],
]

const BACKEND_FALLBACK: Record<string, SimpleIcon> = {
  ollama: siOllama,
  vllm: siVllm,
}

export interface ResolvedModelIcon {
  icon: SimpleIcon
  hex: string
}

/** Best-effort brand icon for a model, by family name then by backend. Returns
 * null when nothing matches — callers fall back to an initials avatar. */
export function resolveModelIcon(
  modelId: string,
  displayName: string,
  backendType: string,
): ResolvedModelIcon | null {
  const haystack = `${modelId} ${displayName}`.toLowerCase()
  for (const [pattern, icon] of FAMILY_PATTERNS) {
    if (pattern.test(haystack)) return { icon, hex: icon.hex }
  }
  const fallback = BACKEND_FALLBACK[backendType]
  return fallback ? { icon: fallback, hex: fallback.hex } : null
}
