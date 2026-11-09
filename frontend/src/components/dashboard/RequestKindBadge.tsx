import { AudioLines, Binary, Bot, Image as ImageIcon, MessageSquare, Radio, Volume2 } from "lucide-react"

const META: Record<string, { icon: React.ElementType; label: string; className: string }> = {
  chat: { icon: MessageSquare, label: "chat", className: "border-sky-500/30 bg-sky-500/10 text-sky-300" },
  ollama_chat: { icon: MessageSquare, label: "ollama chat", className: "border-sky-500/30 bg-sky-500/10 text-sky-300" },
  completion: { icon: MessageSquare, label: "completion", className: "border-sky-500/30 bg-sky-500/10 text-sky-300" },
  ollama_generate: { icon: MessageSquare, label: "generate", className: "border-sky-500/30 bg-sky-500/10 text-sky-300" },
  embedding: { icon: Binary, label: "embedding", className: "border-violet-500/30 bg-violet-500/10 text-violet-300" },
  ollama_embedding: { icon: Binary, label: "embedding", className: "border-violet-500/30 bg-violet-500/10 text-violet-300" },
  rerank: { icon: Binary, label: "rerank", className: "border-violet-500/30 bg-violet-500/10 text-violet-300" },
  image_generation: { icon: ImageIcon, label: "imagen", className: "border-fuchsia-500/30 bg-fuchsia-500/10 text-fuchsia-300" },
  audio_transcription: { icon: AudioLines, label: "transcripción", className: "border-amber-500/30 bg-amber-500/10 text-amber-300" },
  tts: { icon: Volume2, label: "tts", className: "border-amber-500/30 bg-amber-500/10 text-amber-300" },
  realtime_session: { icon: Radio, label: "realtime", className: "border-emerald-500/30 bg-emerald-500/10 text-emerald-300" },
  realtime_chat: { icon: Radio, label: "realtime chat", className: "border-emerald-500/30 bg-emerald-500/10 text-emerald-300" },
  realtime_transcription: { icon: Radio, label: "realtime stt", className: "border-emerald-500/30 bg-emerald-500/10 text-emerald-300" },
  realtime_tts: { icon: Radio, label: "realtime tts", className: "border-emerald-500/30 bg-emerald-500/10 text-emerald-300" },
  agent_hop: { icon: Bot, label: "agent", className: "border-orange-500/30 bg-orange-500/10 text-orange-300" },
}
const DEFAULT_META = { icon: MessageSquare, label: "otro", className: "border-border bg-muted/50 text-muted-foreground" }

export function RequestKindBadge({ kind }: { kind: string | null | undefined }) {
  if (!kind) return null
  const meta = META[kind] ?? { ...DEFAULT_META, label: kind }
  const Icon = meta.icon
  return (
    <span className={`inline-flex shrink-0 items-center gap-1 rounded-md border px-1.5 py-0.5 text-[10.5px] font-medium ${meta.className}`}>
      <Icon size={10} />
      {meta.label}
    </span>
  )
}
