import type { ProfileCategory } from "@/types"

export type Defaults = Record<string, unknown>

const DIARIZATION_SWITCHES = ["diarization_enabled", "diarizationEnabled"]

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
}

/** ``load_overrides`` sin el interruptor de diarización, igual que
 *  ``strip_diarization_overrides`` del backend: diarizar se decide por
 *  petición y no forma parte de la identidad del worker. */
export function stripDiarizationOverrides(overrides: Record<string, unknown> | null | undefined): Record<string, unknown> {
  const out: Record<string, unknown> = {}
  for (const [k, v] of Object.entries(overrides ?? {})) {
    if (!DIARIZATION_SWITCHES.includes(k)) out[k] = v
  }
  if (isRecord(out.whisper)) {
    const section = Object.fromEntries(
      Object.entries(out.whisper).filter(([k]) => !DIARIZATION_SWITCHES.includes(k)),
    )
    if (Object.keys(section).length) out.whisper = section
    else delete out.whisper
  }
  return out
}

/** Un perfil con sobrescrituras de carga arranca su propio proceso. */
export function isDedicatedWorker(overrides: Record<string, unknown> | null | undefined): boolean {
  return Object.keys(stripDiarizationOverrides(overrides)).length > 0
}

/** Los ids de perfil son el nombre que los clientes ponen en ``model``. */
export function isValidSlug(value: string): boolean {
  return /^[a-z0-9][a-z0-9._:-]*$/.test(value) && !value.includes("/")
}

export interface OverrideDiffRow {
  key: string
  base: unknown
  override: unknown
}

function getPath(obj: unknown, path: string[]): unknown {
  let node = obj
  for (const p of path) {
    if (!isRecord(node) || !(p in node)) return undefined
    node = node[p]
  }
  return node
}

/** Cada hoja que el perfil sobrescribe, con el valor del despliegue base. */
export function overridesDiff(
  base: Record<string, unknown> | null | undefined,
  overrides: Record<string, unknown> | null | undefined,
): OverrideDiffRow[] {
  const rows: OverrideDiffRow[] = []
  const walk = (node: Record<string, unknown>, prefix: string[]) => {
    for (const [k, v] of Object.entries(node)) {
      const path = [...prefix, k]
      if (isRecord(v) && Object.keys(v).length) walk(v, path)
      else rows.push({ key: path.join("."), base: getPath(base ?? {}, path), override: v })
    }
  }
  walk(overrides ?? {}, [])
  return rows
}

export function fmt(value: unknown): string {
  if (value === undefined) return "—"
  if (value === null) return "null (se quita)"
  if (typeof value === "boolean") return value ? "sí" : "no"
  if (typeof value === "object") return JSON.stringify(value)
  return String(value)
}

export function extractField(obj: Defaults, key: string): string {
  const v = obj[key]
  return v === undefined || v === null ? "" : String(v)
}

export function extractBool(obj: Defaults, key: string): boolean | null {
  const v = obj[key]
  return v === true ? true : v === false ? false : null
}

/** Texto vacío quita la clave; lo numérico se guarda como número. */
export function mergeField(obj: Defaults, key: string, value: string, asText = false): Defaults {
  const next = { ...obj }
  if (value === "") delete next[key]
  else {
    const n = Number(value)
    next[key] = asText || Number.isNaN(n) ? value : n
  }
  return next
}

/** ``null`` (Automático) quita la clave: manda el cliente o el servidor. */
export function mergeBool(obj: Defaults, key: string, value: boolean | null): Defaults {
  const next = { ...obj }
  if (value === null) delete next[key]
  else next[key] = value
  return next
}

export const CATEGORIES: { value: ProfileCategory; label: string }[] = [
  { value: "llm", label: "LLM" },
  { value: "embedding", label: "Embeddings" },
  { value: "rerank", label: "Reranker" },
  { value: "tts", label: "Voz (TTS)" },
  { value: "stt", label: "Transcripción (STT)" },
  { value: "image", label: "Imagen" },
  { value: "music", label: "Música" },
  { value: "video_upscaling", label: "Escalado de vídeo" },
]

export const CATEGORY_TONE: Record<ProfileCategory, string> = {
  llm: "border-blue-500/40 bg-blue-500/15 text-blue-200",
  embedding: "border-cyan-500/40 bg-cyan-500/15 text-cyan-200",
  rerank: "border-indigo-500/40 bg-indigo-500/15 text-indigo-200",
  tts: "border-emerald-500/40 bg-emerald-500/15 text-emerald-200",
  stt: "border-amber-500/40 bg-amber-500/15 text-amber-200",
  image: "border-purple-500/40 bg-purple-500/15 text-purple-200",
  music: "border-pink-500/40 bg-pink-500/15 text-pink-200",
  video_upscaling: "border-orange-500/40 bg-orange-500/15 text-orange-200",
}

export function inferCategory(caps: Record<string, unknown> | undefined): ProfileCategory {
  const c = caps ?? {}
  if (c.rerank) return "rerank"
  if (c.embeddings) return "embedding"
  if (c.tts) return "tts"
  if (c.audio_transcription || c.audioTranscription) return "stt"
  if (c.image_generation || c.imageGeneration || c.image_editing) return "image"
  if (c.music_generation || c.musicGeneration) return "music"
  if (c.video_upscaling) return "video_upscaling"
  return "llm"
}

export const AUDIO_EXTENSIONS = [".wav", ".mp3", ".flac", ".ogg"]
export const ASSET_ACCEPT = ".wav,.mp3,.flac,.ogg,.safetensors,.bin,.pt,.gguf,.json"

export function isAudioFile(name: string): boolean {
  const lower = name.toLowerCase()
  return AUDIO_EXTENSIONS.some((ext) => lower.endsWith(ext))
}
