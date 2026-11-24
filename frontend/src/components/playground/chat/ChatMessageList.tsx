import { Loader2 } from "lucide-react"
import { MessageBubble, type ChatMessage } from "@/components/playground/MessageBubble"
import { detectAudioFormat } from "@/lib/chatAudio"

interface ChatMessageListProps {
  messages: ChatMessage[]
  streamingMessageId: string | null
  slowResponse: boolean
  lastRouterInfo: { router: string; target: string } | null
  onImageDrop: (file: File) => void
  onAudioDrop: (file: File) => void
}

export function ChatMessageList({
  messages,
  streamingMessageId,
  slowResponse,
  lastRouterInfo,
  onImageDrop,
  onAudioDrop,
}: ChatMessageListProps) {
  return (
    <div
      className="flex-1 space-y-4 overflow-y-auto p-3"
      onDragOver={(event) => event.preventDefault()}
      onDrop={(event) => {
        event.preventDefault()
        const file = event.dataTransfer.files[0]
        if (!file) return
        if (file.type.startsWith("image/")) {
          onImageDrop(file)
          return
        }
        if (file.type.startsWith("audio/") || detectAudioFormat(file)) {
          onAudioDrop(file)
        }
      }}
    >
      {messages.length === 0 && (
        <p className="text-sm text-muted-foreground">
          Escribe un prompt o arrastra una imagen / audio (.wav, .mp3, .m4a, .webm, .ogg, .flac).
        </p>
      )}
      {messages.map((message) => (
        <MessageBubble
          key={message.id}
          message={message}
          streaming={message.id === streamingMessageId}
        />
      ))}
      {slowResponse && (
        <div
          role="status"
          className="flex items-center gap-2 rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-100"
        >
          <Loader2 size={14} className="shrink-0 animate-spin text-amber-400" aria-hidden="true" />
          <span>
            Cargando modelo o compilando kernels (primera vez puede tardar 60-120 s)...
          </span>
        </div>
      )}
      {lastRouterInfo && (
        <div
          role="status"
          className="flex items-center gap-2 rounded-md border border-primary/30 bg-primary/5 px-3 py-2 text-xs text-primary"
        >
          <span className="rounded border border-primary/40 bg-primary/10 px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide">
            router
          </span>
          <span>
            <code className="font-mono">{lastRouterInfo.router}</code> reenvió a{" "}
            <code className="font-mono">{lastRouterInfo.target}</code>
          </span>
        </div>
      )}
    </div>
  )
}
