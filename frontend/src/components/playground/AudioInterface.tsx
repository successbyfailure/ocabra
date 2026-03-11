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
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null)
  const [transcript, setTranscript] = useState("")
  const [segments, setSegments] = useState<TranscriptionSegment[]>([])
  const [ttsText, setTtsText] = useState("Hola desde oCabra")
  const [voice, setVoice] = useState("alloy")
  const [speed, setSpeed] = useState(1)
  const [ttsAudioUrl, setTtsAudioUrl] = useState<string | null>(null)
  const [runningTranscription, setRunningTranscription] = useState(false)
  const [runningTTS, setRunningTTS] = useState(false)

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
        setAudioBlob(blob)
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

  const transcribe = async () => {
    if (!modelId) {
      toast.error("Selecciona un modelo")
      return
    }
    if (!audioBlob) {
      toast.error("Sube o graba un audio primero")
      return
    }
    setRunningTranscription(true)
    try {
      const form = new FormData()
      form.append("file", audioBlob, "audio.webm")
      form.append("model", modelId)
      form.append("response_format", params.responseFormat)
      form.append("temperature", String(params.temperature))

      const response = await fetch("/v1/audio/transcriptions", {
        method: "POST",
        body: form,
      })
      if (!response.ok) {
        const err = await response.json().catch(() => ({}))
        throw new Error(String(err?.error?.message ?? err?.detail ?? `HTTP ${response.status}`))
      }

      if (params.responseFormat === "text") {
        setTranscript(await response.text())
        setSegments([])
      } else {
        const payload = await response.json()
        setTranscript(String(payload?.text ?? ""))
        if (params.responseFormat === "verbose_json" && Array.isArray(payload?.segments)) {
          setSegments(
            payload.segments.map((segment: { start?: number; end?: number; text?: string }) => ({
              start: String(segment?.start ?? ""),
              end: String(segment?.end ?? ""),
              text: String(segment?.text ?? ""),
            })),
          )
        } else {
          setSegments([])
        }
      }
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Error en transcripcion")
    } finally {
      setRunningTranscription(false)
    }
  }

  const generateTTS = async () => {
    if (!modelId) {
      toast.error("Selecciona un modelo")
      return
    }
    if (!ttsText.trim()) return
    setRunningTTS(true)
    try {
      const response = await fetch("/v1/audio/speech", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: modelId,
          input: ttsText.trim(),
          voice,
          speed,
          response_format: "mp3",
        }),
      })
      if (!response.ok) {
        const err = await response.json().catch(() => ({}))
        throw new Error(String(err?.error?.message ?? err?.detail ?? `HTTP ${response.status}`))
      }
      const blob = await response.blob()
      setTtsAudioUrl(URL.createObjectURL(blob))
      toast.success(`TTS generado con voz ${voice} a velocidad ${speed.toFixed(2)}`)
    } catch {
      setTtsAudioUrl(FALLBACK_TTS_AUDIO)
      toast.error("Fallo TTS real, mostrando audio fallback")
    } finally {
      setRunningTTS(false)
    }
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
                if (!file) return
                setAudioUrl(URL.createObjectURL(file))
                setAudioBlob(file)
              }}
            />
          </label>

          <button
            type="button"
            onClick={() => void transcribe()}
            disabled={runningTranscription || !modelId}
            className="rounded-md bg-primary px-3 py-2 text-sm font-medium text-primary-foreground"
          >
            {runningTranscription ? "Transcribiendo..." : "Transcribir"}
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
          onClick={() => void generateTTS()}
          disabled={runningTTS || !modelId}
          className="rounded-md bg-primary px-3 py-2 text-sm font-medium text-primary-foreground"
        >
          {runningTTS ? "Generando..." : "Generar TTS"}
        </button>

        {ttsAudioUrl && <audio controls src={ttsAudioUrl} className="w-full" />}
      </section>
    </div>
  )
}
