import { useEffect, useMemo, useRef, useState } from "react"
import { Mic, Square, Upload } from "lucide-react"
import { toast } from "sonner"
import type { PlaygroundParams } from "@/components/playground/ParamsPanel"

interface AudioInterfaceProps {
  modelId: string
  params: PlaygroundParams
  canTranscribe?: boolean
  canTTS?: boolean
}

interface TranscriptionSegment {
  start: string
  end: string
  text: string
}

interface VoiceInfo {
  voices: string[]
  model_type: "base" | "custom_voice" | "placeholder"
  languages: string[]
  supports_voice_clone: boolean
}

const OPENAI_VOICES = ["alloy", "echo", "fable", "nova", "onyx", "shimmer"]

/**
 * Convert any browser-recorded audio Blob (WebM, OGG, etc.) to PCM16 WAV
 * using the Web Audio API so the backend receives a valid WAV file.
 */
async function blobToWav(blob: Blob, targetSr = 16000): Promise<Blob> {
  const ctx = new OfflineAudioContext(1, 1, targetSr)
  const arrayBuf = await blob.arrayBuffer()
  const decoded = await ctx.decodeAudioData(arrayBuf)

  // Resample to targetSr mono
  const offline = new OfflineAudioContext(1, Math.ceil(decoded.duration * targetSr), targetSr)
  const src = offline.createBufferSource()
  src.buffer = decoded
  src.connect(offline.destination)
  src.start()
  const rendered = await offline.startRendering()

  const pcm = rendered.getChannelData(0)
  const pcm16 = new Int16Array(pcm.length)
  for (let i = 0; i < pcm.length; i++) {
    const s = Math.max(-1, Math.min(1, pcm[i]))
    pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff
  }

  // Build WAV header + PCM data
  const wavBuf = new ArrayBuffer(44 + pcm16.byteLength)
  const view = new DataView(wavBuf)
  const writeStr = (off: number, s: string) => { for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i)) }
  writeStr(0, "RIFF")
  view.setUint32(4, 36 + pcm16.byteLength, true)
  writeStr(8, "WAVE")
  writeStr(12, "fmt ")
  view.setUint32(16, 16, true)          // PCM subchunk size
  view.setUint16(20, 1, true)           // PCM format
  view.setUint16(22, 1, true)           // mono
  view.setUint32(24, targetSr, true)    // sample rate
  view.setUint32(28, targetSr * 2, true) // byte rate
  view.setUint16(32, 2, true)           // block align
  view.setUint16(34, 16, true)          // bits per sample
  writeStr(36, "data")
  view.setUint32(40, pcm16.byteLength, true)
  new Int16Array(wavBuf, 44).set(pcm16)

  return new Blob([wavBuf], { type: "audio/wav" })
}

export function AudioInterface({ modelId, params, canTranscribe = true, canTTS = true }: AudioInterfaceProps) {
  // Transcription state
  const [recording, setRecording] = useState(false)
  const [audioUrl, setAudioUrl] = useState<string | null>(null)
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null)
  const [transcript, setTranscript] = useState("")
  const [segments, setSegments] = useState<TranscriptionSegment[]>([])
  const [runningTranscription, setRunningTranscription] = useState(false)

  // TTS state
  const [ttsText, setTtsText] = useState("Hola desde oCabra")
  const [voice, setVoice] = useState("alloy")
  const [speed, setSpeed] = useState(1)
  const [language, setLanguage] = useState("Auto")
  const [ttsAudioUrl, setTtsAudioUrl] = useState<string | null>(null)
  const [runningTTS, setRunningTTS] = useState(false)
  const [instruct, setInstruct] = useState("")

  // Voice cloning state (Base model)
  const [refAudioUrl, setRefAudioUrl] = useState<string | null>(null)
  const [refAudioBlob, setRefAudioBlob] = useState<Blob | null>(null)
  const [refText, setRefText] = useState("")
  const [refRecording, setRefRecording] = useState(false)

  // Voice metadata from worker
  const [voiceInfo, setVoiceInfo] = useState<VoiceInfo | null>(null)

  const recorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const refRecorderRef = useRef<MediaRecorder | null>(null)
  const refChunksRef = useRef<Blob[]>([])

  // Fetch voice info when model changes
  useEffect(() => {
    if (!modelId || !canTTS) return
    setVoiceInfo(null)
    fetch(`/v1/audio/voices?model=${encodeURIComponent(modelId)}`)
      .then((r) => r.json())
      .then((data: VoiceInfo) => {
        setVoiceInfo(data)
        // Reset voice to first available
        if (data.voices?.length) setVoice(data.voices[0])
      })
      .catch(() => setVoiceInfo(null))
  }, [modelId, canTTS])

  const availableVoices = useMemo(() => voiceInfo?.voices ?? OPENAI_VOICES, [voiceInfo])
  const modelType = voiceInfo?.model_type ?? "placeholder"
  const supportsVoiceClone = voiceInfo?.supports_voice_clone ?? false
  const availableLanguages = voiceInfo?.languages ?? ["Auto"]

  // ─── Transcription helpers ─────────────────────────────────────────────────

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const recorder = new MediaRecorder(stream)
      recorderRef.current = recorder
      chunksRef.current = []
      recorder.ondataavailable = (e) => chunksRef.current.push(e.data)
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
    recorderRef.current?.stream.getTracks().forEach((t) => t.stop())
  }

  const transcribe = async () => {
    if (!modelId) return toast.error("Selecciona un modelo")
    if (!audioBlob) return toast.error("Sube o graba un audio primero")
    setRunningTranscription(true)
    try {
      const form = new FormData()
      form.append("file", audioBlob, "audio.webm")
      form.append("model", modelId)
      form.append("response_format", params.responseFormat)
      form.append("temperature", String(params.temperature))

      const response = await fetch("/v1/audio/transcriptions", { method: "POST", body: form })
      if (!response.ok) {
        const err = await response.json().catch(() => ({}))
        const msg = err?.error?.message ?? err?.detail?.error?.message ?? (typeof err?.detail === "string" ? err.detail : null) ?? `HTTP ${response.status}`
        throw new Error(String(msg))
      }
      if (params.responseFormat === "text") {
        setTranscript(await response.text())
        setSegments([])
      } else {
        const payload = await response.json()
        setTranscript(String(payload?.text ?? ""))
        setSegments(
          params.responseFormat === "verbose_json" && Array.isArray(payload?.segments)
            ? payload.segments.map((s: { start?: number; end?: number; text?: string }) => ({
                start: String(s?.start ?? ""),
                end: String(s?.end ?? ""),
                text: String(s?.text ?? ""),
              }))
            : [],
        )
      }
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Error en transcripcion")
    } finally {
      setRunningTranscription(false)
    }
  }

  // ─── Reference audio recording (voice clone) ───────────────────────────────

  const startRefRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const recorder = new MediaRecorder(stream)
      refRecorderRef.current = recorder
      refChunksRef.current = []
      recorder.ondataavailable = (e) => refChunksRef.current.push(e.data)
      recorder.onstop = () => {
        const blob = new Blob(refChunksRef.current, { type: "audio/webm" })
        setRefAudioUrl(URL.createObjectURL(blob))
        setRefAudioBlob(blob)
        setRefRecording(false)
      }
      recorder.start()
      setRefRecording(true)
    } catch {
      toast.error("No se pudo iniciar grabacion de referencia")
    }
  }

  const stopRefRecording = () => {
    refRecorderRef.current?.stop()
    refRecorderRef.current?.stream.getTracks().forEach((t) => t.stop())
  }

  // ─── TTS generation ────────────────────────────────────────────────────────

  const generateTTS = async () => {
    if (!modelId) return toast.error("Selecciona un modelo")
    if (!ttsText.trim()) return
    setRunningTTS(true)
    try {
      const body: Record<string, unknown> = {
        model: modelId,
        input: ttsText.trim(),
        voice,
        speed,
        response_format: "wav",
        language,
      }

      // Voice cloning: convert reference audio to WAV and attach as base64
      if (supportsVoiceClone && refAudioBlob) {
        const wavBlob = await blobToWav(refAudioBlob)
        const arrayBuf = await wavBlob.arrayBuffer()
        const bytes = new Uint8Array(arrayBuf)
        let binary = ""
        for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i])
        body.reference_audio = btoa(binary)
        if (refText.trim()) body.reference_text = refText.trim()
      }

      // CustomVoice: speaker is in voice dropdown already (speaker name)
      if (modelType === "custom_voice") {
        body.speaker = voice
        if (instruct.trim()) body.instruct = instruct.trim()
      }

      const response = await fetch("/v1/audio/speech", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      })
      if (!response.ok) {
        const err = await response.json().catch(() => ({}))
        const msg = err?.error?.message ?? err?.detail?.error?.message ?? (typeof err?.detail === "string" ? err.detail : null) ?? `HTTP ${response.status}`
        throw new Error(String(msg))
      }
      const blob = await response.blob()
      if (ttsAudioUrl) URL.revokeObjectURL(ttsAudioUrl)
      setTtsAudioUrl(URL.createObjectURL(blob))
      toast.success("TTS generado")
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Error generando TTS")
    } finally {
      setRunningTTS(false)
    }
  }

  // ─── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="space-y-4 rounded-lg border border-border bg-card p-4">

      {/* ── Transcription section ──────────────────────────────── */}
      {canTranscribe && (
        <section className="space-y-3">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">Transcripcion</h2>
          <div className="flex flex-wrap gap-2">
            {!recording ? (
              <button type="button" onClick={() => void startRecording()}
                className="inline-flex items-center gap-1 rounded-md border border-border px-3 py-2 text-sm hover:bg-muted">
                <Mic size={14} /> Grabar
              </button>
            ) : (
              <button type="button" onClick={stopRecording}
                className="inline-flex items-center gap-1 rounded-md border border-red-500/40 px-3 py-2 text-sm text-red-200 hover:bg-red-500/20">
                <Square size={14} /> Detener
              </button>
            )}
            <label className="rounded-md border border-border px-3 py-2 text-sm hover:bg-muted">
              Upload
              <input type="file" accept="audio/*" className="hidden"
                onChange={(e) => {
                  const file = e.target.files?.[0]
                  if (!file) return
                  setAudioUrl(URL.createObjectURL(file))
                  setAudioBlob(file)
                }} />
            </label>
            <button type="button" onClick={() => void transcribe()}
              disabled={runningTranscription || !modelId}
              className="rounded-md bg-primary px-3 py-2 text-sm font-medium text-primary-foreground disabled:opacity-50">
              {runningTranscription ? "Transcribiendo..." : "Transcribir"}
            </button>
          </div>
          {audioUrl && <audio controls src={audioUrl} className="w-full" />}
          {transcript && (
            <div className="rounded-md border border-border bg-background/60 p-3">
              <p className="text-sm">{transcript}</p>
              {segments.length > 0 && (
                <div className="mt-2 space-y-1 text-xs text-muted-foreground">
                  {segments.map((s) => (
                    <p key={`${s.start}-${s.end}`}>[{s.start} - {s.end}] {s.text}</p>
                  ))}
                </div>
              )}
            </div>
          )}
        </section>
      )}

      {/* ── TTS section ───────────────────────────────────────── */}
      {canTTS && (
        <section className={`space-y-3 ${canTranscribe ? "border-t border-border pt-3" : ""}`}>
          <h2 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">
            TTS
            {modelType !== "placeholder" && (
              <span className="ml-2 rounded px-1.5 py-0.5 text-[10px] font-normal bg-muted text-muted-foreground">
                {modelType === "custom_voice" ? "Custom Voice" : "Voice Clone"}
              </span>
            )}
          </h2>

          {/* Text to synthesize */}
          <textarea
            value={ttsText}
            onChange={(e) => setTtsText(e.target.value)}
            placeholder="Texto a sintetizar..."
            className="min-h-20 w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
          />

          <div className="grid gap-2 md:grid-cols-2">
            {/* Voice / Speaker selector */}
            <label className="text-sm text-muted-foreground">
              {modelType === "custom_voice" ? "Speaker" : "Voz"}
              <select value={voice} onChange={(e) => setVoice(e.target.value)}
                className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2 text-sm">
                {availableVoices.map((v) => (
                  <option key={v} value={v}>{v}</option>
                ))}
              </select>
            </label>

            {/* Language selector */}
            <label className="text-sm text-muted-foreground">
              Idioma
              <select value={language} onChange={(e) => setLanguage(e.target.value)}
                className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2 text-sm">
                {availableLanguages.map((l) => (
                  <option key={l} value={l}>{l}</option>
                ))}
              </select>
            </label>

            {/* Speed */}
            <label className="text-sm text-muted-foreground">
              Velocidad {speed.toFixed(2)}
              <input type="range" min={0.5} max={1.8} step={0.05} value={speed}
                onChange={(e) => setSpeed(Number(e.target.value))}
                className="mt-2 w-full" />
            </label>
          </div>

          {/* Style instruction (CustomVoice only) */}
          {modelType === "custom_voice" && (
            <label className="text-sm text-muted-foreground">
              Instrucción de estilo (opcional)
              <input type="text" value={instruct}
                onChange={(e) => setInstruct(e.target.value)}
                placeholder='Ej: "con tono muy alegre", "Very angry"'
                className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2 text-sm" />
            </label>
          )}

          {/* Voice cloning (Base model only) */}
          {supportsVoiceClone && (
            <div className="rounded-md border border-border/60 bg-muted/20 p-3 space-y-2">
              <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                Audio de referencia para clonación (recomendado +5s)
              </p>
              <div className="flex flex-wrap gap-2">
                {!refRecording ? (
                  <button type="button" onClick={() => void startRefRecording()}
                    className="inline-flex items-center gap-1 rounded-md border border-border px-3 py-1.5 text-xs hover:bg-muted">
                    <Mic size={12} /> Grabar
                  </button>
                ) : (
                  <button type="button" onClick={stopRefRecording}
                    className="inline-flex items-center gap-1 rounded-md border border-red-500/40 px-3 py-1.5 text-xs text-red-200 hover:bg-red-500/20">
                    <Square size={12} /> Detener
                  </button>
                )}
                <label className="inline-flex items-center gap-1 rounded-md border border-border px-3 py-1.5 text-xs hover:bg-muted cursor-pointer">
                  <Upload size={12} /> Subir audio
                  <input type="file" accept="audio/*" className="hidden"
                    onChange={(e) => {
                      const file = e.target.files?.[0]
                      if (!file) return
                      setRefAudioUrl(URL.createObjectURL(file))
                      setRefAudioBlob(file)
                    }} />
                </label>
                {refAudioBlob && (
                  <button type="button" onClick={() => { setRefAudioBlob(null); setRefAudioUrl(null) }}
                    className="rounded-md border border-border px-3 py-1.5 text-xs text-muted-foreground hover:bg-muted">
                    Quitar
                  </button>
                )}
              </div>
              {refAudioUrl && <audio controls src={refAudioUrl} className="w-full h-8" />}
              <input type="text" value={refText} onChange={(e) => setRefText(e.target.value)}
                placeholder="Transcripción del audio de referencia (mejora la calidad)"
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-xs" />
              {!refAudioBlob && (
                <p className="text-xs text-muted-foreground">
                  Sin audio de referencia se usará una voz por defecto.
                </p>
              )}
            </div>
          )}

          <button type="button" onClick={() => void generateTTS()}
            disabled={runningTTS || !modelId}
            className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground disabled:opacity-50">
            {runningTTS ? "Generando..." : "Generar TTS"}
          </button>

          {ttsAudioUrl && <audio controls src={ttsAudioUrl} className="w-full" />}
        </section>
      )}
    </div>
  )
}
