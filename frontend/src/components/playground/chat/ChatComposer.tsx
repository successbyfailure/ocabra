import { useRef } from "react"
import { AudioLines, Copy, ImagePlus, Mic, Send, X } from "lucide-react"
import { toast } from "sonner"
import type { DurationEstimateResult } from "@/api/client"
import { EtaBadge } from "@/components/playground/chat/EtaBadge"
import { AUDIO_ACCEPT, formatDurationLabel, type AudioAttachment } from "@/lib/chatAudio"

interface ChatComposerProps {
  input: string
  onInputChange: (value: string) => void
  onSend: () => void
  onStop: () => void
  sending: boolean
  modelId: string
  audioInputCapable: boolean
  recording: boolean
  dragImage: string | null
  audioAttachment: AudioAttachment | null
  onRemoveImage: () => void
  onRemoveAudio: () => void
  onImageSelected: (file: File) => void
  onAudioSelected: (file: File) => void
  onToggleRecord: () => void
  curlPreview: string
  estimate: DurationEstimateResult | null
  estimateLoading: boolean
}

export function ChatComposer({
  input,
  onInputChange,
  onSend,
  onStop,
  sending,
  modelId,
  audioInputCapable,
  recording,
  dragImage,
  audioAttachment,
  onRemoveImage,
  onRemoveAudio,
  onImageSelected,
  onAudioSelected,
  onToggleRecord,
  curlPreview,
  estimate,
  estimateLoading,
}: ChatComposerProps) {
  const fileRef = useRef<HTMLInputElement>(null)
  const audioFileRef = useRef<HTMLInputElement>(null)

  return (
    <>
      {(dragImage || audioAttachment) && (
        <div className="flex flex-wrap items-center gap-2 border-t border-border px-3 py-2">
          {dragImage && (
            <span className="inline-flex items-center gap-2 rounded-md border border-primary/40 bg-primary/10 px-2 py-1 text-xs">
              <ImagePlus size={14} />
              Imagen adjuntada
              <button
                type="button"
                onClick={onRemoveImage}
                className="ml-1 rounded-sm p-0.5 text-primary/70 hover:bg-primary/20 hover:text-primary"
                aria-label="Quitar imagen"
              >
                <X size={12} />
              </button>
            </span>
          )}
          {audioAttachment && (
            <span className="inline-flex items-center gap-2 rounded-md border border-violet-500/40 bg-violet-500/10 px-2 py-1 text-xs text-violet-100">
              <AudioLines size={14} />
              <span className="max-w-[16rem] truncate font-medium">{audioAttachment.name}</span>
              <span className="text-violet-100/70">{audioAttachment.format.toUpperCase()}</span>
              {typeof audioAttachment.durationSec === "number" && audioAttachment.durationSec > 0 && (
                <span className="text-violet-100/70">{formatDurationLabel(audioAttachment.durationSec)}</span>
              )}
              <button
                type="button"
                onClick={onRemoveAudio}
                className="ml-1 rounded-sm p-0.5 text-violet-100/70 hover:bg-violet-500/20 hover:text-violet-50"
                aria-label="Quitar audio"
              >
                <X size={12} />
              </button>
            </span>
          )}
        </div>
      )}

      <div className="border-t border-border p-3">
        <textarea
          value={input}
          onChange={(event) => onInputChange(event.target.value)}
          onKeyDown={(event) => {
            if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
              event.preventDefault()
              onSend()
            }
          }}
          placeholder="Escribe tu mensaje..."
          className="mb-2 min-h-20 w-full rounded-md border border-border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
        />
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => fileRef.current?.click()}
              className="rounded-md border border-border px-3 py-2 text-xs hover:bg-muted"
            >
              Vision input
            </button>
            <button
              type="button"
              onClick={() => audioFileRef.current?.click()}
              disabled={!audioInputCapable}
              title={
                audioInputCapable
                  ? "Adjuntar audio (.wav, .mp3, .m4a, .webm, .ogg, .flac)"
                  : "Selected model does not support audio input"
              }
              className="inline-flex items-center gap-1 rounded-md border border-border px-3 py-2 text-xs hover:bg-muted disabled:cursor-not-allowed disabled:opacity-50"
            >
              <AudioLines size={14} />
              Audio input
            </button>
            <button
              type="button"
              onClick={onToggleRecord}
              disabled={!audioInputCapable}
              title={
                audioInputCapable
                  ? recording
                    ? "Detener grabación"
                    : "Grabar audio del micrófono"
                  : "Selected model does not support audio input"
              }
              className={`inline-flex items-center gap-1 rounded-md border px-3 py-2 text-xs disabled:cursor-not-allowed disabled:opacity-50 ${
                recording
                  ? "border-rose-500/50 bg-rose-500/15 text-rose-100 hover:bg-rose-500/25"
                  : "border-border hover:bg-muted"
              }`}
            >
              <Mic size={14} className={recording ? "animate-pulse" : ""} />
              {recording ? "Detener" : "Grabar"}
            </button>
            <button
              type="button"
              onClick={async () => {
                await navigator.clipboard.writeText(curlPreview)
                toast.success("curl copiado")
              }}
              className="inline-flex items-center gap-1 rounded-md border border-border px-3 py-2 text-xs hover:bg-muted"
            >
              <Copy size={14} />
              Copy as OpenAI API call
            </button>
          </div>
          <div className="flex items-center gap-2">
            {modelId && !sending && (estimate || estimateLoading) && (
              <EtaBadge estimate={estimate} loading={estimateLoading} />
            )}
            {sending && (
              <button
                type="button"
                onClick={onStop}
                className="inline-flex items-center gap-2 rounded-md border border-rose-500/50 bg-rose-500/15 px-4 py-2.5 text-sm font-medium text-rose-100 hover:bg-rose-500/25"
              >
                <X size={14} />
                Detener
              </button>
            )}
            <button
              type="button"
              onClick={onSend}
              disabled={sending || !modelId}
              className="inline-flex items-center gap-2 rounded-md bg-primary px-5 py-2.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
            >
              <Send size={14} />
              {sending ? "Enviando..." : "Enviar"}
              <span className="ml-0.5 text-xs text-primary-foreground/50">⌘↵</span>
            </button>
          </div>
        </div>
      </div>

      <input
        ref={fileRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(event) => {
          const file = event.target.files?.[0]
          if (file) onImageSelected(file)
        }}
      />

      <input
        ref={audioFileRef}
        type="file"
        accept={AUDIO_ACCEPT}
        className="hidden"
        onChange={(event) => {
          const file = event.target.files?.[0]
          if (file) onAudioSelected(file)
          event.target.value = ""
        }}
      />
    </>
  )
}
