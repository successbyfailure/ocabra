import { AudioLines, Boxes, Cpu, Image as ImageIcon, Layers, Music, type LucideIcon } from "lucide-react"
import { siHuggingface, siNvidia, siOllama, siVllm, type SimpleIcon } from "simple-icons"
import { backendLabel } from "@/components/models/BackendIcons"

// Logo de marca cuando existe en simple-icons. La familia ggml (llama.cpp, el
// fork mks, whisper.cpp, bitnet) no tiene logo publicado, así que usa un icono
// de familia y se distingue por la etiqueta.
const BRAND: Record<string, SimpleIcon> = {
  ollama: siOllama,
  vllm: siVllm,
  tensorrt_llm: siNvidia,
  diffusers: siHuggingface,
  tts: siHuggingface,
}

const FAMILY: Record<string, LucideIcon> = {
  llama_cpp: Layers,
  llamacpp_mks: Layers,
  bitnet: Layers,
  sglang: Cpu,
  whisper: AudioLines,
  whisper_cpp: AudioLines,
  whispercpp: AudioLines,
  vibeasr: AudioLines,
  voxtral: AudioLines,
  piper: AudioLines,
  chatterbox: AudioLines,
  mageflow: ImageIcon,
  upscaler: ImageIcon,
  flashvsr: ImageIcon,
  acestep: Music,
}

/** Icono + nombre corto del backend de un modelo. */
export function BackendIcon({
  backendType,
  showLabel = true,
  className = "",
}: {
  backendType: string
  showLabel?: boolean
  className?: string
}) {
  const brand = BRAND[backendType]
  const Family = FAMILY[backendType] ?? Boxes
  const label = backendLabel(backendType)
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-md border border-border bg-muted/60 px-1.5 py-0.5 text-[11px] font-medium text-foreground/85 ${className}`}
      title={`Backend: ${label}`}
      data-backend={backendType}
    >
      {brand ? (
        <svg role="img" aria-hidden viewBox="0 0 24 24" width={12} height={12} fill="currentColor">
          <path d={brand.path} />
        </svg>
      ) : (
        <Family size={12} aria-hidden />
      )}
      {showLabel && <span>{label}</span>}
    </span>
  )
}
