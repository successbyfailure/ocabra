import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { AlertCircle, Mic, Square, Wifi, WifiOff } from "lucide-react"
import { toast } from "sonner"
import { api } from "@/api/client"
import type { ModelState, ServerConfig } from "@/types"

interface RealtimeInterfaceProps {
  models: ModelState[]
}

type ConnState = "idle" | "connecting" | "connected" | "error"
type ActivityState = "idle" | "recording" | "thinking" | "speaking"
type InputMode = "ptt" | "vad"

interface TurnEntry {
  id: string
  role: "user" | "assistant"
  text: string
  native?: boolean
}

const INPUT_SAMPLE_RATE = 16000
const OUTPUT_SAMPLE_RATE = 16000
const PTT_MAX_MS = 30_000
const VAD_RMS_THRESHOLD = 0.012
const VAD_SILENCE_MS = 800
const VAD_MIN_SPEECH_MS = 300

function floatTo16BitPCM(input: Float32Array): Int16Array {
  const out = new Int16Array(input.length)
  for (let i = 0; i < input.length; i++) {
    const s = Math.max(-1, Math.min(1, input[i]))
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff
  }
  return out
}

function downsampleBuffer(buffer: Float32Array, inSampleRate: number, outSampleRate: number): Float32Array {
  if (outSampleRate >= inSampleRate) return buffer
  const ratio = inSampleRate / outSampleRate
  const newLen = Math.floor(buffer.length / ratio)
  const result = new Float32Array(newLen)
  let offsetResult = 0
  let offsetBuffer = 0
  while (offsetResult < newLen) {
    const nextOffsetBuffer = Math.floor((offsetResult + 1) * ratio)
    let accum = 0
    let count = 0
    for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
      accum += buffer[i]
      count++
    }
    result[offsetResult] = count > 0 ? accum / count : 0
    offsetResult++
    offsetBuffer = nextOffsetBuffer
  }
  return result
}

function int16ToBase64(pcm: Int16Array): string {
  const bytes = new Uint8Array(pcm.buffer, pcm.byteOffset, pcm.byteLength)
  let binary = ""
  const CHUNK = 0x8000
  for (let i = 0; i < bytes.length; i += CHUNK) {
    binary += String.fromCharCode.apply(null, Array.from(bytes.subarray(i, i + CHUNK)) as number[])
  }
  return btoa(binary)
}

function base64ToInt16(b64: string): Int16Array {
  const binary = atob(b64)
  const len = binary.length
  const bytes = new Uint8Array(len)
  for (let i = 0; i < len; i++) bytes[i] = binary.charCodeAt(i)
  return new Int16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2)
}

function getRealtimeUrl(modelId: string): string {
  const proto = window.location.protocol === "https:" ? "wss" : "ws"
  const params = new URLSearchParams({ model: modelId })
  return `${proto}://${window.location.host}/v1/realtime?${params.toString()}`
}

export function RealtimeInterface({ models }: RealtimeInterfaceProps) {
  const chatModels = useMemo(
    () =>
      models
        .filter((m) => m.capabilities.chat)
        .sort((a, b) => {
          const al = a.status === "loaded" ? 0 : 1
          const bl = b.status === "loaded" ? 0 : 1
          if (al !== bl) return al - bl
          return a.displayName.localeCompare(b.displayName)
        }),
    [models],
  )

  const [selectedModelId, setSelectedModelId] = useState<string>("")
  const [conn, setConn] = useState<ConnState>("idle")
  const [activity, setActivity] = useState<ActivityState>("idle")
  const [inputMode, setInputMode] = useState<InputMode>("ptt")
  const [turns, setTurns] = useState<TurnEntry[]>([])
  const [error, setError] = useState<string | null>(null)
  const [serverConfig, setServerConfig] = useState<ServerConfig | null>(null)
  const [transcribeUserAudio, setTranscribeUserAudio] = useState<boolean>(true)

  const wsRef = useRef<WebSocket | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const mediaStreamRef = useRef<MediaStream | null>(null)
  const processorRef = useRef<ScriptProcessorNode | null>(null)
  const sourceNodeRef = useRef<MediaStreamAudioSourceNode | null>(null)
  const recordingRef = useRef(false)
  const recStartRef = useRef(0)
  const vadActiveRef = useRef(false)
  const vadSilenceStartRef = useRef<number | null>(null)
  const vadSpeechStartRef = useRef<number | null>(null)
  const playbackQueueRef = useRef<AudioBuffer[]>([])
  const playbackTimeRef = useRef(0)
  const playbackCtxRef = useRef<AudioContext | null>(null)
  const inputAudioRoutingRef = useRef<"auto" | "native" | "stt">("auto")
  const currentRespIdRef = useRef<string | null>(null)
  const pendingAssistantRef = useRef<{ text: string; id: string } | null>(null)
  const pttTimerRef = useRef<number | null>(null)

  const selectedModel = useMemo(
    () => chatModels.find((m) => m.modelId === selectedModelId) ?? null,
    [chatModels, selectedModelId],
  )
  const audioNative = Boolean(selectedModel?.capabilities.audioInput)

  useEffect(() => {
    if (!selectedModelId && chatModels.length > 0) {
      setSelectedModelId(chatModels[0].modelId)
    }
  }, [chatModels, selectedModelId])

  useEffect(() => {
    api.config
      .get()
      .then((c) => setServerConfig(c))
      .catch(() => setServerConfig(null))
  }, [])

  const appendTurn = useCallback((entry: TurnEntry) => {
    setTurns((prev) => [...prev, entry])
  }, [])

  const updateLastAssistant = useCallback((deltaText: string, native: boolean) => {
    setTurns((prev) => {
      const pending = pendingAssistantRef.current
      if (!pending) {
        const id = `a_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`
        pendingAssistantRef.current = { id, text: deltaText }
        return [...prev, { id, role: "assistant", text: deltaText, native }]
      }
      pending.text += deltaText
      return prev.map((t) => (t.id === pending.id ? { ...t, text: pending.text } : t))
    })
  }, [])

  const stopRecordingInternal = useCallback(() => {
    if (pttTimerRef.current !== null) {
      window.clearTimeout(pttTimerRef.current)
      pttTimerRef.current = null
    }
    if (processorRef.current) {
      try {
        processorRef.current.disconnect()
      } catch {
        // ignore
      }
      processorRef.current.onaudioprocess = null
      processorRef.current = null
    }
    if (sourceNodeRef.current) {
      try {
        sourceNodeRef.current.disconnect()
      } catch {
        // ignore
      }
      sourceNodeRef.current = null
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((t) => t.stop())
      mediaStreamRef.current = null
    }
    recordingRef.current = false
  }, [])

  const teardown = useCallback(() => {
    stopRecordingInternal()
    if (audioContextRef.current && audioContextRef.current.state !== "closed") {
      void audioContextRef.current.close()
    }
    audioContextRef.current = null
    if (playbackCtxRef.current && playbackCtxRef.current.state !== "closed") {
      void playbackCtxRef.current.close()
    }
    playbackCtxRef.current = null
    playbackQueueRef.current = []
    playbackTimeRef.current = 0
    if (wsRef.current) {
      try {
        wsRef.current.close()
      } catch {
        // ignore
      }
      wsRef.current = null
    }
    pendingAssistantRef.current = null
    currentRespIdRef.current = null
    setActivity("idle")
    setConn("idle")
  }, [stopRecordingInternal])

  useEffect(() => {
    return () => {
      teardown()
    }
  }, [teardown])

  const playPcmChunk = useCallback((pcm: Int16Array) => {
    if (!playbackCtxRef.current) {
      try {
        playbackCtxRef.current = new AudioContext({ sampleRate: OUTPUT_SAMPLE_RATE })
      } catch {
        playbackCtxRef.current = new AudioContext()
      }
      playbackTimeRef.current = playbackCtxRef.current.currentTime
    }
    const ctx = playbackCtxRef.current
    if (!ctx || pcm.length === 0) return
    const buffer = ctx.createBuffer(1, pcm.length, OUTPUT_SAMPLE_RATE)
    const channel = buffer.getChannelData(0)
    for (let i = 0; i < pcm.length; i++) channel[i] = pcm[i] / 0x8000
    const src = ctx.createBufferSource()
    src.buffer = buffer
    src.connect(ctx.destination)
    const now = ctx.currentTime
    const startAt = Math.max(now, playbackTimeRef.current)
    src.start(startAt)
    playbackTimeRef.current = startAt + buffer.duration
    setActivity("speaking")
  }, [])

  const handleServerEvent = useCallback(
    (data: unknown) => {
      if (typeof data !== "object" || data === null) return
      const ev = data as { type?: string; [k: string]: unknown }
      switch (ev.type) {
        case "session.created":
        case "session.updated": {
          const session = ev.session as { input_audio_routing?: string } | undefined
          if (session?.input_audio_routing === "native" || session?.input_audio_routing === "stt" || session?.input_audio_routing === "auto") {
            inputAudioRoutingRef.current = session.input_audio_routing
          }
          break
        }
        case "conversation.item.created": {
          const item = ev.item as { id?: string; role?: string; content?: Array<{ transcript?: string; text?: string; type?: string }> } | undefined
          if (item?.role === "user") {
            const transcript = item.content?.find((c) => typeof c?.transcript === "string")?.transcript ?? ""
            const native = !transcript
            const itemId = item.id ?? `u_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`
            appendTurn({
              id: itemId,
              role: "user",
              text: native ? (transcribeUserAudio ? "<transcribiendo…>" : "<audio enviado>") : transcript,
              native,
            })
          }
          break
        }
        case "conversation.item.input_audio_transcription.completed": {
          const itemId = typeof ev.item_id === "string" ? ev.item_id : null
          const transcript = typeof ev.transcript === "string" ? ev.transcript : ""
          if (!itemId || !transcript) break
          setTurns((prev) =>
            prev.map((t) => (t.id === itemId ? { ...t, text: transcript } : t)),
          )
          break
        }
        case "response.created": {
          const resp = ev.response as { id?: string } | undefined
          currentRespIdRef.current = resp?.id ?? null
          pendingAssistantRef.current = null
          setActivity("thinking")
          break
        }
        case "response.audio_transcript.delta":
        case "response.text.delta": {
          const delta = typeof ev.delta === "string" ? ev.delta : ""
          if (delta) updateLastAssistant(delta, audioNative)
          break
        }
        case "response.audio.delta": {
          const delta = typeof ev.delta === "string" ? ev.delta : ""
          if (!delta) break
          try {
            const pcm = base64ToInt16(delta)
            playPcmChunk(pcm)
          } catch {
            // ignore decode errors
          }
          break
        }
        case "response.done": {
          currentRespIdRef.current = null
          pendingAssistantRef.current = null
          window.setTimeout(() => {
            setActivity((a) => (a === "speaking" || a === "thinking" ? "idle" : a))
          }, 200)
          break
        }
        case "error": {
          const err = ev.error as { message?: string; code?: string } | undefined
          const msg = err?.message ?? "Error del servidor Realtime"
          setError(msg)
          toast.error(msg)
          break
        }
        default:
          break
      }
    },
    [appendTurn, audioNative, playPcmChunk, updateLastAssistant],
  )

  const sendSessionUpdate = useCallback(() => {
    const ws = wsRef.current
    if (!ws || ws.readyState !== WebSocket.OPEN) return
    const session: Record<string, unknown> = {
      modalities: ["text", "audio"],
      input_audio_format: "pcm16",
      output_audio_format: "pcm16",
      voice: "alloy",
      input_audio_routing: audioNative ? "auto" : "stt",
      transcribe_user_audio: transcribeUserAudio,
      turn_detection:
        inputMode === "vad"
          ? { type: "server_vad", threshold: 0.5, silence_duration_ms: 600, prefix_padding_ms: 300 }
          : { type: "none" },
    }
    // Skip declaring the STT model entirely in native mode + transcript off:
    // the backend won't need Whisper and avoids unnecessary cold loads.
    if (
      serverConfig?.realtimeDefaultSttModel &&
      (!audioNative || transcribeUserAudio)
    ) {
      session.input_audio_transcription = { model: serverConfig.realtimeDefaultSttModel }
    }
    if (serverConfig?.realtimeDefaultTtsModel) {
      session.tts_model = serverConfig.realtimeDefaultTtsModel
    }
    ws.send(JSON.stringify({ type: "session.update", session }))
  }, [audioNative, inputMode, serverConfig, transcribeUserAudio])

  const connect = useCallback(async () => {
    if (!selectedModelId) {
      toast.error("Selecciona un modelo")
      return
    }
    if (wsRef.current) return
    setError(null)
    setConn("connecting")
    try {
      const ws = new WebSocket(getRealtimeUrl(selectedModelId))
      wsRef.current = ws
      ws.onopen = () => {
        setConn("connected")
        sendSessionUpdate()
      }
      ws.onmessage = (evt) => {
        try {
          const data = JSON.parse(typeof evt.data === "string" ? evt.data : "")
          handleServerEvent(data)
        } catch {
          // ignore non-JSON messages
        }
      }
      ws.onerror = () => {
        setError("Error de conexión WebSocket")
        setConn("error")
      }
      ws.onclose = (evt) => {
        wsRef.current = null
        if (evt.code !== 1000 && evt.code !== 1005) {
          const reason = evt.reason || `code ${evt.code}`
          setError(`Conexión cerrada: ${reason}`)
          setConn("error")
        } else {
          setConn("idle")
        }
        setActivity("idle")
        stopRecordingInternal()
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "No se pudo conectar")
      setConn("error")
    }
  }, [handleServerEvent, selectedModelId, sendSessionUpdate, stopRecordingInternal])

  const disconnect = useCallback(() => {
    teardown()
  }, [teardown])

  const sendCommit = useCallback(() => {
    const ws = wsRef.current
    if (!ws || ws.readyState !== WebSocket.OPEN) return
    ws.send(JSON.stringify({ type: "input_audio_buffer.commit" }))
    ws.send(JSON.stringify({ type: "response.create" }))
    setActivity("thinking")
  }, [])

  const startCapture = useCallback(async () => {
    const ws = wsRef.current
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      toast.error("Conecta primero")
      return
    }
    if (recordingRef.current) return
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
      })
      mediaStreamRef.current = stream
      const ctx = new AudioContext()
      audioContextRef.current = ctx
      const source = ctx.createMediaStreamSource(stream)
      sourceNodeRef.current = source
      const bufferSize = 4096
      const processor = ctx.createScriptProcessor(bufferSize, 1, 1)
      processorRef.current = processor
      const inSampleRate = ctx.sampleRate

      vadSilenceStartRef.current = null
      vadSpeechStartRef.current = null
      vadActiveRef.current = false

      processor.onaudioprocess = (event) => {
        if (!recordingRef.current) return
        const wsActive = wsRef.current
        if (!wsActive || wsActive.readyState !== WebSocket.OPEN) return
        const input = event.inputBuffer.getChannelData(0)
        let rms = 0
        for (let i = 0; i < input.length; i++) rms += input[i] * input[i]
        rms = Math.sqrt(rms / input.length)

        const down = downsampleBuffer(input, inSampleRate, INPUT_SAMPLE_RATE)
        const pcm16 = floatTo16BitPCM(down)
        const b64 = int16ToBase64(pcm16)
        wsActive.send(JSON.stringify({ type: "input_audio_buffer.append", audio: b64 }))

        if (inputMode === "vad") {
          const now = performance.now()
          if (rms > VAD_RMS_THRESHOLD) {
            if (!vadActiveRef.current) {
              vadActiveRef.current = true
              vadSpeechStartRef.current = now
            }
            vadSilenceStartRef.current = null
          } else if (vadActiveRef.current) {
            if (vadSilenceStartRef.current === null) {
              vadSilenceStartRef.current = now
            } else if (
              now - vadSilenceStartRef.current > VAD_SILENCE_MS &&
              vadSpeechStartRef.current !== null &&
              now - vadSpeechStartRef.current > VAD_MIN_SPEECH_MS
            ) {
              vadActiveRef.current = false
              vadSpeechStartRef.current = null
              vadSilenceStartRef.current = null
              sendCommit()
            }
          }
        }

        if (inputMode === "ptt" && performance.now() - recStartRef.current > PTT_MAX_MS) {
          stopRecordingInternal()
          sendCommit()
        }
      }

      source.connect(processor)
      processor.connect(ctx.destination)
      recordingRef.current = true
      recStartRef.current = performance.now()
      setActivity("recording")

      if (inputMode === "ptt") {
        pttTimerRef.current = window.setTimeout(() => {
          if (recordingRef.current) {
            stopRecordingInternal()
            sendCommit()
          }
        }, PTT_MAX_MS)
      }
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "No se pudo acceder al micrófono")
      stopRecordingInternal()
    }
  }, [inputMode, sendCommit, stopRecordingInternal])

  const stopCapture = useCallback(() => {
    if (!recordingRef.current) return
    stopRecordingInternal()
    if (inputMode === "ptt") {
      sendCommit()
    }
  }, [inputMode, sendCommit, stopRecordingInternal])

  const onModelChange = (id: string) => {
    setSelectedModelId(id)
    if (wsRef.current) {
      // Reset connection: model is bound to ?model= query param
      teardown()
    }
  }

  const onInputModeChange = (mode: InputMode) => {
    setInputMode(mode)
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      const session = {
        turn_detection:
          mode === "vad"
            ? { type: "server_vad", threshold: 0.5, silence_duration_ms: 600, prefix_padding_ms: 300 }
            : { type: "none" },
      }
      wsRef.current.send(JSON.stringify({ type: "session.update", session }))
    }
  }

  const connBadgeClass =
    conn === "connected"
      ? "bg-emerald-500/20 text-emerald-200 border-emerald-500/40"
      : conn === "connecting"
        ? "bg-amber-500/20 text-amber-200 border-amber-500/40"
        : conn === "error"
          ? "bg-red-500/20 text-red-200 border-red-500/40"
          : "bg-muted text-muted-foreground border-border"

  const activityLabel: Record<ActivityState, string> = {
    idle: "En espera",
    recording: "Grabando",
    thinking: "Pensando",
    speaking: "Hablando",
  }

  return (
    <div className="space-y-4 rounded-lg border border-border bg-card p-4">
      <section className="space-y-3">
        <div className="flex flex-wrap items-center gap-2">
          <label className="flex-1 min-w-[200px] text-sm text-muted-foreground">
            Modelo LLM
            <select
              value={selectedModelId}
              onChange={(e) => onModelChange(e.target.value)}
              className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
              disabled={conn === "connected" || conn === "connecting"}
            >
              <option value="">— Selecciona un modelo —</option>
              {chatModels.map((m) => (
                <option key={m.modelId} value={m.modelId}>
                  {m.displayName} {m.capabilities.audioInput ? "• audio nativo" : ""}{" "}
                  {m.status === "loaded" ? "• loaded" : ""}
                </option>
              ))}
            </select>
          </label>

          {selectedModel && (
            <div className="flex flex-col gap-1 pt-5">
              {selectedModel.capabilities.audioInput && (
                <span className="inline-flex items-center rounded border border-purple-500/40 bg-purple-500/10 px-2 py-0.5 text-xs text-purple-100">
                  Audio nativo
                </span>
              )}
              {!selectedModel.capabilities.audioInput && (
                <span className="inline-flex items-center rounded border border-border bg-muted px-2 py-0.5 text-xs text-muted-foreground">
                  Pipeline STT
                </span>
              )}
            </div>
          )}
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <span
            className={`inline-flex items-center gap-1 rounded-md border px-2 py-1 text-xs ${connBadgeClass}`}
          >
            {conn === "connected" ? <Wifi size={12} /> : <WifiOff size={12} />}
            {conn === "connected"
              ? "Conectado"
              : conn === "connecting"
                ? "Conectando..."
                : conn === "error"
                  ? "Error"
                  : "Desconectado"}
          </span>
          <span className="inline-flex items-center rounded-md border border-border bg-muted px-2 py-1 text-xs text-muted-foreground">
            {activityLabel[activity]}
          </span>
          <div className="ml-auto flex gap-2">
            {conn !== "connected" && conn !== "connecting" ? (
              <button
                type="button"
                onClick={() => void connect()}
                disabled={!selectedModelId}
                className="rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground disabled:opacity-50"
              >
                Conectar
              </button>
            ) : (
              <button
                type="button"
                onClick={disconnect}
                className="rounded-md border border-border px-3 py-1.5 text-sm hover:bg-muted"
              >
                Desconectar
              </button>
            )}
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs text-muted-foreground">Modo entrada:</span>
          <div className="inline-flex rounded-md border border-border bg-background p-0.5">
            <button
              type="button"
              onClick={() => onInputModeChange("ptt")}
              className={`px-3 py-1 text-xs ${
                inputMode === "ptt" ? "bg-primary text-primary-foreground rounded-sm" : "text-muted-foreground"
              }`}
            >
              Push-to-Talk
            </button>
            <button
              type="button"
              onClick={() => onInputModeChange("vad")}
              className={`px-3 py-1 text-xs ${
                inputMode === "vad" ? "bg-primary text-primary-foreground rounded-sm" : "text-muted-foreground"
              }`}
            >
              VAD continuo
            </button>
          </div>
        </div>

        <label
          className="flex items-center gap-2 text-xs text-muted-foreground"
          title={
            audioNative
              ? "En modo audio nativo el LLM escucha el audio directamente. Whisper solo se usa para mostrar tu transcripción en la UI; puedes desactivarlo para evitar cargarlo."
              : "Whisper transcribe tu voz antes de pasarla al LLM. No se puede desactivar en este modo."
          }
        >
          <input
            type="checkbox"
            className="h-3.5 w-3.5 cursor-pointer accent-primary"
            checked={transcribeUserAudio || !audioNative}
            disabled={!audioNative}
            onChange={(e) => setTranscribeUserAudio(e.target.checked)}
          />
          Transcribir audio del usuario para mostrar en UI
          {!audioNative && <span className="text-muted-foreground/70">(siempre activo en modo STT)</span>}
        </label>

        {inputMode === "ptt" ? (
          <button
            type="button"
            onMouseDown={() => void startCapture()}
            onMouseUp={stopCapture}
            onMouseLeave={() => recordingRef.current && stopCapture()}
            onTouchStart={(e) => {
              e.preventDefault()
              void startCapture()
            }}
            onTouchEnd={(e) => {
              e.preventDefault()
              stopCapture()
            }}
            disabled={conn !== "connected"}
            className={`inline-flex w-full items-center justify-center gap-2 rounded-md px-4 py-3 text-sm font-medium transition-colors ${
              activity === "recording"
                ? "bg-red-500/30 text-red-100 border border-red-500/60"
                : "bg-primary text-primary-foreground"
            } disabled:opacity-50`}
          >
            <Mic size={16} />
            {activity === "recording" ? "Soltar para enviar" : "Mantener para hablar"}
          </button>
        ) : (
          <div className="flex gap-2">
            {!recordingRef.current ? (
              <button
                type="button"
                onClick={() => void startCapture()}
                disabled={conn !== "connected"}
                className="inline-flex flex-1 items-center justify-center gap-2 rounded-md bg-primary px-4 py-3 text-sm font-medium text-primary-foreground disabled:opacity-50"
              >
                <Mic size={16} /> Iniciar escucha VAD
              </button>
            ) : (
              <button
                type="button"
                onClick={() => {
                  stopRecordingInternal()
                }}
                className="inline-flex flex-1 items-center justify-center gap-2 rounded-md border border-red-500/40 bg-red-500/10 px-4 py-3 text-sm font-medium text-red-100"
              >
                <Square size={16} /> Detener escucha
              </button>
            )}
          </div>
        )}

        {error && (
          <div className="flex items-start gap-2 rounded-md border border-red-500/40 bg-red-500/10 px-3 py-2 text-sm text-red-100">
            <AlertCircle size={16} className="mt-0.5 shrink-0" />
            <span>{error}</span>
          </div>
        )}
      </section>

      <section className="space-y-2 border-t border-border pt-3">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">Conversación</h2>
        <div className="max-h-[50vh] space-y-2 overflow-y-auto">
          {turns.length === 0 ? (
            <p className="text-sm text-muted-foreground">
              {conn === "connected"
                ? "Habla para iniciar la conversación."
                : "Selecciona un modelo y conecta para empezar."}
            </p>
          ) : (
            turns.map((t) => (
              <div
                key={t.id}
                className={`rounded-md border px-3 py-2 text-sm ${
                  t.role === "user"
                    ? "border-border bg-background/60"
                    : "border-primary/40 bg-primary/10"
                }`}
              >
                <div className="mb-1 flex items-center gap-2 text-xs text-muted-foreground">
                  <span className="font-medium">{t.role === "user" ? "Tú" : "Asistente"}</span>
                  {t.native && t.role === "user" && (
                    <span className="rounded border border-purple-500/40 bg-purple-500/10 px-1.5 py-0.5 text-[10px] text-purple-100">
                      audio nativo
                    </span>
                  )}
                </div>
                <p className="whitespace-pre-wrap">{t.text || <span className="text-muted-foreground">…</span>}</p>
              </div>
            ))
          )}
        </div>
      </section>
    </div>
  )
}
