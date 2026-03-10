import { useMemo, useRef, useState } from "react"
import { Mic, Square } from "lucide-react"
import { toast } from "sonner"
import type { PlaygroundParams } from "@/components/playground/ParamsPanel"

interface AudioInterfaceProps {
  modelId: string
  params: PlaygroundParams
}

interface TranscriptionSegment {
  start: string
  end: string
  text: string
}

const FALLBACK_TTS_AUDIO =
  "data:audio/wav;base64,UklGRlQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YTAAAAAA"

export function AudioInterface({ modelId, params }: AudioInterfaceProps) {
  const [recording, setRecording] = useState(false)
  const [audioUrl, setAudioUrl] = useState<string | null>(null)
  const [transcript, setTranscript] = useState("")
  const [segments, setSegments] = useState<TranscriptionSegment[]>([])
  const [ttsText, setTtsText] = useState("Hola desde oCabra")
  const [voice, setVoice] = useState("alloy")
  const [speed, setSpeed] = useState(1)
  const [ttsAudioUrl, setTtsAudioUrl] = useState<string | null>(null)

  const recorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])

  const availableVoices = useMemo(() => ["alloy", "nova", "echo", "onyx"], [])

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const recorder = new MediaRecorder(stream)
      recorderRef.current = recorder
      chunksRef.current = []

      recorder.ondataavailable = (event) => {
        chunksRef.current.push(event.data)
      }

      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: "audio/webm" })
        setAudioUrl(URL.createObjectURL(blob))
        setRecording(false)
      }

      recorder.start()
      setRecording(true)
    } catch {
      toast.error("No se pudo iniciar grabacion")
    }
  }

  const stopRecording = () => {
    recorderRef.current?.stop()
    recorderRef.current?.stream.getTracks().forEach((track) => track.stop())
  }

  const transcribe = () => {
    if (!audioUrl) {
      toast.error("Sube o graba un audio primero")
      return
    }
    const text = `Transcripcion simulada por ${modelId} en formato ${params.responseFormat}.`
    setTranscript(text)
    if (params.responseFormat === "verbose_json") {
      setSegments([
        { start: "00:00", end: "00:03", text: "Hola, este es un ejemplo." },
        { start: "00:03", end: "00:07", text: "Incluye timestamps en verbose_json." },
      ])
    } else {
      setSegments([])
    }
  }

  const generateTTS = () => {
    if (!ttsText.trim()) return
    setTtsAudioUrl(FALLBACK_TTS_AUDIO)
    toast.success(`TTS generado con voz ${voice} a velocidad ${speed.toFixed(2)}`)
  }

  return (
    <div className="space-y-4 rounded-lg border border-border bg-card p-4">
      <section className="space-y-3">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">Transcripcion</h2>
        <div className="flex flex-wrap gap-2">
          {!recording ? (
            <button
              type="button"
              onClick={() => void startRecording()}
              className="inline-flex items-center gap-1 rounded-md border border-border px-3 py-2 text-sm hover:bg-muted"
            >
              <Mic size={14} />
              Grabar
            </button>
          ) : (
            <button
              type="button"
              onClick={stopRecording}
              className="inline-flex items-center gap-1 rounded-md border border-red-500/40 px-3 py-2 text-sm text-red-200 hover:bg-red-500/20"
            >
              <Square size={14} />
              Detener
            </button>
          )}

          <label className="rounded-md border border-border px-3 py-2 text-sm hover:bg-muted">
            Upload
            <input
              type="file"
              accept="audio/*"
              className="hidden"
              onChange={(event) => {
                const file = event.target.files?.[0]
                if (file) setAudioUrl(URL.createObjectURL(file))
              }}
            />
          </label>

          <button
            type="button"
            onClick={transcribe}
            className="rounded-md bg-primary px-3 py-2 text-sm font-medium text-primary-foreground"
          >
            Transcribir
          </button>
        </div>

        {audioUrl && <audio controls src={audioUrl} className="w-full" />}

        {transcript && (
          <div className="rounded-md border border-border bg-background/60 p-3">
            <p className="text-sm">{transcript}</p>
            {segments.length > 0 && (
              <div className="mt-2 space-y-1 text-xs text-muted-foreground">
                {segments.map((segment) => (
                  <p key={`${segment.start}-${segment.end}`}>
                    [{segment.start} - {segment.end}] {segment.text}
                  </p>
                ))}
              </div>
            )}
          </div>
        )}
      </section>

      <section className="space-y-3 border-t border-border pt-3">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">TTS</h2>

        <textarea
          value={ttsText}
          onChange={(event) => setTtsText(event.target.value)}
          className="min-h-20 w-full rounded-md border border-border bg-background px-3 py-2"
        />

        <div className="grid gap-2 md:grid-cols-2">
          <label className="text-sm text-muted-foreground">
            Voz
            <select
              value={voice}
              onChange={(event) => setVoice(event.target.value)}
              className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2"
            >
              {availableVoices.map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </label>

          <label className="text-sm text-muted-foreground">
            Velocidad {speed.toFixed(2)}
            <input
              type="range"
              min={0.5}
              max={1.8}
              step={0.05}
              value={speed}
              onChange={(event) => setSpeed(Number(event.target.value))}
              className="mt-2 w-full"
            />
          </label>
        </div>

        <button
          type="button"
          onClick={generateTTS}
          className="rounded-md bg-primary px-3 py-2 text-sm font-medium text-primary-foreground"
        >
          Generar TTS
        </button>

        {ttsAudioUrl && <audio controls src={ttsAudioUrl} className="w-full" />}
      </section>
    </div>
  )
}
