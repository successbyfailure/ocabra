import {
  Binary,
  Brain,
  Captions,
  Eye,
  EyeOff,
  Image as ImageIcon,
  ListOrdered,
  Mic,
  Music,
  Video,
  Volume2,
  Wrench,
} from "lucide-react"
import type { ModelState } from "@/types"

/**
 * Capability chips for a loaded model.
 *
 * Two distinct things are shown, and conflating them would be misleading:
 *  - *supported*: the model can do this (from `capabilities`), drawn muted;
 *  - *active now*: it is actually holding the resources for it, drawn
 *    highlighted. Only multimodal vision on the Makespace llama.cpp fork has
 *    a live state today — it loads and drops the projector at runtime, so
 *    "supports vision" and "has vision loaded" genuinely differ there.
 *    Everything else is static, so supported == active and it renders muted.
 */
export function CapabilityIcons({ model }: { model: ModelState }) {
  const caps = model.capabilities
  if (!caps) return null

  const visionLive = model.visionState // null when not applicable
  const items: Array<{ key: string; icon: typeof Eye; label: string; active?: boolean }> = []

  if (caps.vision) {
    const on = visionLive !== "inactive"
    items.push({
      key: "vision",
      icon: on ? Eye : EyeOff,
      active: visionLive === "active",
      label:
        visionLive === "active"
          ? "Visión activa — el proyector multimodal está cargado"
          : visionLive === "inactive"
            ? "Visión disponible pero no cargada: se activará bajo demanda (ocupará VRAM al hacerlo)"
            : "Acepta imágenes",
    })
  }
  if (caps.audioInput) items.push({ key: "audioIn", icon: Mic, label: "Acepta audio de entrada" })
  if (caps.audioTranscription)
    items.push({ key: "stt", icon: Captions, label: "Transcripción de audio" })
  if (caps.tts || caps.audioOutput) items.push({ key: "tts", icon: Volume2, label: "Genera voz" })
  if (caps.videoInput) items.push({ key: "video", icon: Video, label: "Acepta vídeo" })
  if (caps.imageGeneration || caps.imageEditing)
    items.push({ key: "image", icon: ImageIcon, label: "Genera o edita imágenes" })
  if (caps.musicGeneration) items.push({ key: "music", icon: Music, label: "Genera música" })
  if (caps.tools) items.push({ key: "tools", icon: Wrench, label: "Tool calling" })
  if (caps.reasoning) items.push({ key: "reasoning", icon: Brain, label: "Razonamiento" })
  if (caps.embeddings || caps.pooling) items.push({ key: "emb", icon: Binary, label: "Embeddings" })
  if (caps.rerank) items.push({ key: "rerank", icon: ListOrdered, label: "Reranking" })

  if (items.length === 0) return null

  return (
    <span className="inline-flex items-center gap-1" aria-label="Capacidades">
      {items.map(({ key, icon: Icon, label, active }) => (
        <span
          key={key}
          title={label}
          className={
            active
              ? "inline-flex items-center rounded bg-emerald-500/20 p-0.5 text-emerald-300"
              : "inline-flex items-center rounded p-0.5 text-muted-foreground/70"
          }
        >
          <Icon size={12} aria-hidden />
        </span>
      ))}
    </span>
  )
}
