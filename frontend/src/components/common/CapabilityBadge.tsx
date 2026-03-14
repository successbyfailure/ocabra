import type { ModelCapabilities } from "@/types"

interface CapabilityBadgeProps {
  capability: keyof ModelCapabilities
}

const LABELS: Partial<Record<keyof ModelCapabilities, string>> = {
  chat: "Chat",
  tools: "Tools",
  vision: "Vision",
  reasoning: "Reasoning",
  embeddings: "Embeddings",
  pooling: "Pooling",
  rerank: "Rerank",
  classification: "Classify",
  score: "Score",
  imageGeneration: "Image",
  audioTranscription: "Audio",
  tts: "TTS",
}

export function CapabilityBadge({ capability }: CapabilityBadgeProps) {
  const label = LABELS[capability]
  if (!label) return null

  return (
    <span className="inline-flex items-center rounded border border-cyan-500/40 bg-cyan-500/10 px-2 py-0.5 text-xs text-cyan-100">
      {label}
    </span>
  )
}
