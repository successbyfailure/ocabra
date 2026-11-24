// Ayudantes de audio del Playground (REF-18).
//
// Extraído de ChatInterface.tsx: detección de formato, codificación a base64,
// sondeo de duración y el MIME de grabación del micrófono. Sin React.

import type { ChatAudioFormat } from "@/types"

export interface AudioAttachment {
  name: string
  format: ChatAudioFormat
  /** base64 of raw bytes (no data: prefix). */
  dataBase64: string
  durationSec: number | null
}

export const AUDIO_ACCEPT = ".wav,.mp3,.m4a,.webm,.ogg,.flac,audio/*"

const AUDIO_FORMAT_BY_MIME: Record<string, ChatAudioFormat> = {
  "audio/wav": "wav",
  "audio/wave": "wav",
  "audio/x-wav": "wav",
  "audio/mpeg": "mp3",
  "audio/mp3": "mp3",
  "audio/mp4": "m4a",
  "audio/x-m4a": "m4a",
  "audio/aac": "m4a",
  "audio/webm": "webm",
  "audio/ogg": "ogg",
  "audio/oga": "ogg",
  "audio/flac": "flac",
  "audio/x-flac": "flac",
}
const AUDIO_FORMAT_BY_EXT: Record<string, ChatAudioFormat> = {
  wav: "wav",
  mp3: "mp3",
  m4a: "m4a",
  webm: "webm",
  ogg: "ogg",
  oga: "ogg",
  flac: "flac",
}

export function detectAudioFormat(file: File): ChatAudioFormat | null {
  const mime = file.type.toLowerCase()
  const fromMime = AUDIO_FORMAT_BY_MIME[mime] ?? mimeToFormat(mime)
  if (fromMime) return fromMime
  const ext = file.name.split(".").pop()?.toLowerCase() ?? ""
  return AUDIO_FORMAT_BY_EXT[ext] ?? null
}

export function mimeToFormat(mime: string): ChatAudioFormat | null {
  if (!mime) return null
  if (mime.includes("webm")) return "webm"
  if (mime.includes("ogg")) return "ogg"
  if (mime.includes("wav")) return "wav"
  if (mime.includes("mpeg") || mime.includes("mp3")) return "mp3"
  if (mime.includes("mp4") || mime.includes("m4a") || mime.includes("aac")) return "m4a"
  if (mime.includes("flac")) return "flac"
  return null
}

export function pickRecorderMimeType(): string | null {
  if (typeof MediaRecorder === "undefined") return null
  const candidates = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/ogg;codecs=opus",
    "audio/ogg",
    "audio/mp4",
  ]
  for (const candidate of candidates) {
    if (MediaRecorder.isTypeSupported?.(candidate)) return candidate
  }
  return null
}

export function stopRecorderStream(stream: MediaStream | null): void {
  if (!stream) return
  for (const track of stream.getTracks()) {
    try {
      track.stop()
    } catch {
      // ignore
    }
  }
}

export function fileToBase64(file: File): Promise<string> {
  return blobToBase64(file)
}

export function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onerror = () => reject(reader.error ?? new Error("FileReader error"))
    reader.onload = () => {
      const result = String(reader.result ?? "")
      const comma = result.indexOf(",")
      resolve(comma === -1 ? result : result.slice(comma + 1))
    }
    reader.readAsDataURL(blob)
  })
}

export function probeAudioDurationSec(file: File): Promise<number | null> {
  return new Promise((resolve) => {
    if (typeof window === "undefined" || typeof URL === "undefined") {
      resolve(null)
      return
    }
    const url = URL.createObjectURL(file)
    const audio = new Audio()
    audio.preload = "metadata"
    const cleanup = () => URL.revokeObjectURL(url)
    audio.onloadedmetadata = () => {
      const value = Number.isFinite(audio.duration) ? Math.round(audio.duration) : null
      cleanup()
      resolve(value)
    }
    audio.onerror = () => {
      cleanup()
      resolve(null)
    }
    audio.src = url
  })
}

export function formatDurationLabel(seconds: number): string {
  const total = Math.max(0, Math.round(seconds))
  const m = Math.floor(total / 60)
  const s = total % 60
  return `${m}:${s.toString().padStart(2, "0")}`
}
