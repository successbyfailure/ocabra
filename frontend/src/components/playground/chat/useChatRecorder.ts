import { useEffect, useRef, useState } from "react"
import { toast } from "sonner"
import {
  blobToBase64,
  mimeToFormat,
  pickRecorderMimeType,
  stopRecorderStream,
  type AudioAttachment,
} from "@/lib/chatAudio"

interface UseChatRecorderOptions {
  audioInputCapable: boolean
  onRecorded: (attachment: AudioAttachment) => void
}

/**
 * Microphone capture for the Playground composer (REF-18). Owns the
 * ``MediaRecorder`` lifecycle and the stream cleanup, and hands a ready-made
 * ``AudioAttachment`` back when a recording stops. Extracted from
 * ChatInterface so the component stays focused on the chat flow.
 */
export function useChatRecorder({ audioInputCapable, onRecorded }: UseChatRecorderOptions) {
  const [recording, setRecording] = useState(false)
  const recorderRef = useRef<MediaRecorder | null>(null)
  const recorderChunksRef = useRef<Blob[]>([])
  const recorderStartRef = useRef<number>(0)
  const recorderStreamRef = useRef<MediaStream | null>(null)

  useEffect(() => {
    return () => {
      stopRecorderStream(recorderStreamRef.current)
    }
  }, [])

  const startRecording = async () => {
    if (!audioInputCapable) {
      toast.error("El modelo seleccionado no acepta audio como entrada")
      return
    }
    if (recording) return
    if (typeof navigator === "undefined" || !navigator.mediaDevices?.getUserMedia) {
      toast.error("Tu navegador no soporta grabación de audio")
      return
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mimeType = pickRecorderMimeType()
      const recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream)
      recorderChunksRef.current = []
      recorderStartRef.current = performance.now()
      recorderStreamRef.current = stream
      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          recorderChunksRef.current.push(event.data)
        }
      }
      recorder.onstop = async () => {
        const elapsedSec = (performance.now() - recorderStartRef.current) / 1000
        stopRecorderStream(recorderStreamRef.current)
        recorderStreamRef.current = null
        const chunks = recorderChunksRef.current
        recorderChunksRef.current = []
        if (chunks.length === 0) {
          setRecording(false)
          return
        }
        const blob = new Blob(chunks, { type: recorder.mimeType || "audio/webm" })
        const format = mimeToFormat(blob.type) ?? "webm"
        const ext = format === "m4a" ? "m4a" : format
        const name = `recording-${Date.now()}.${ext}`
        try {
          const dataBase64 = await blobToBase64(blob)
          onRecorded({
            name,
            format,
            dataBase64,
            durationSec: Number.isFinite(elapsedSec) ? Math.round(elapsedSec) : null,
          })
          toast.success("Grabación lista")
        } catch (err) {
          toast.error(err instanceof Error ? err.message : "Error al codificar audio")
        } finally {
          setRecording(false)
        }
      }
      recorderRef.current = recorder
      recorder.start()
      setRecording(true)
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "No se pudo acceder al micrófono")
    }
  }

  const stopRecording = () => {
    const recorder = recorderRef.current
    if (recorder && recorder.state !== "inactive") {
      recorder.stop()
    } else {
      setRecording(false)
    }
  }

  const toggle = () => (recording ? stopRecording() : void startRecording())

  return { recording, toggle }
}
