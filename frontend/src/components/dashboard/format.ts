import type { RequestStage } from "@/types"

export function fmtTokens(n: number | null | undefined): string {
  if (n == null) return "—"
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(n >= 100_000 ? 0 : 1)}k`
  return String(Math.round(n))
}

/** Duración compacta: 850 ms · 12 s · 3m05s · 1h12m. */
export function fmtDuration(ms: number | null | undefined): string {
  if (ms == null) return "—"
  if (ms < 1000) return `${Math.round(ms)} ms`
  const s = Math.round(ms / 1000)
  if (s < 60) return `${s} s`
  const m = Math.floor(s / 60)
  if (m < 60) return `${m}m${String(s % 60).padStart(2, "0")}s`
  const h = Math.floor(m / 60)
  return `${h}h${String(m % 60).padStart(2, "0")}m`
}

export function fmtTps(tps: number | null | undefined): string {
  if (tps == null || tps <= 0) return "—"
  if (tps >= 1000) return `${(tps / 1000).toFixed(1)}k t/s`
  return `${tps >= 100 ? Math.round(tps) : tps.toFixed(1)} t/s`
}

export function fmtAgo(totalSeconds: number): string {
  if (totalSeconds < 60) return `${totalSeconds}s`
  const minutes = Math.floor(totalSeconds / 60)
  if (minutes < 60) return `${minutes}m`
  const hours = Math.floor(minutes / 60)
  const remMinutes = minutes % 60
  if (hours < 24) return remMinutes === 0 ? `${hours}h` : `${hours}h ${remMinutes}m`
  const days = Math.floor(hours / 24)
  const remHours = hours % 24
  return remHours === 0 ? `${days}d` : `${days}d ${remHours}h`
}

export function loadedMeta(loadedAt: string | null | undefined, nowMs: number): { at: string; ago: string } | null {
  if (!loadedAt) return null
  const loadedTs = Date.parse(loadedAt)
  if (Number.isNaN(loadedTs)) return null
  const seconds = Math.max(0, Math.floor((nowMs - loadedTs) / 1000))
  const at = new Date(loadedTs).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })
  return { at, ago: fmtAgo(seconds) }
}

/**
 * Lenguaje visual de estados, compartido por la píldora del modelo, el borde
 * de la card, el punto de cada petición y sus barras. Cada estado tiene un
 * color propio y una animación que dice si algo avanza (pulso rápido), espera
 * (parpadeo lento) o está parado (sin animación):
 *
 * - verde, pulso: generando · azul cielo, pulso: prefill
 * - violeta, pulso: cargando el modelo
 * - naranja, parpadeo: esperando espacio en GPU (desalojo o servicio)
 * - ámbar, parpadeo: en cola o esperando a que el backend empiece
 * - pizarra, pulso lento: procesando sin señal (sin streaming)
 * - rojo, fijo: atascada o sin actividad · gris, fijo: inactivo
 */
export type VisualState =
  | "generating"
  | "prefill"
  | "loading"
  | "waiting_space"
  | "waiting"
  | "processing"
  | "stuck"
  | "idle"

export const VISUAL: Record<
  VisualState,
  { color: string; text: string; bg: string; border: string; anim: string }
> = {
  generating: { color: "#22c55e", text: "text-emerald-300", bg: "bg-emerald-500/15", border: "border-emerald-500/50", anim: "animate-pulse" },
  prefill: { color: "#0ea5e9", text: "text-sky-300", bg: "bg-sky-500/15", border: "border-sky-500/50", anim: "animate-pulse" },
  loading: { color: "#a78bfa", text: "text-violet-300", bg: "bg-violet-500/15", border: "border-violet-500/50", anim: "animate-pulse" },
  waiting_space: { color: "#f97316", text: "text-orange-300", bg: "bg-orange-500/15", border: "border-orange-500/60", anim: "animate-blink-slow" },
  waiting: { color: "#f59e0b", text: "text-amber-300", bg: "bg-amber-500/15", border: "border-amber-500/50", anim: "animate-blink-slow" },
  processing: { color: "#94a3b8", text: "text-slate-300", bg: "bg-slate-500/15", border: "border-slate-500/40", anim: "animate-pulse-slow" },
  stuck: { color: "#ef4444", text: "text-red-300", bg: "bg-red-500/15", border: "border-red-500/60", anim: "" },
  idle: { color: "#64748b", text: "text-slate-400", bg: "bg-muted", border: "border-border", anim: "" },
}

export const STAGE_META: Record<RequestStage, { label: string; color: string; text: string; visual: VisualState }> = {
  queued: { label: "en cola", color: VISUAL.waiting.color, text: VISUAL.waiting.text, visual: "waiting" },
  backend_wait: { label: "esperando al backend", color: VISUAL.waiting.color, text: VISUAL.waiting.text, visual: "waiting" },
  loading: { label: "cargando modelo", color: VISUAL.loading.color, text: VISUAL.loading.text, visual: "loading" },
  waiting_for_gpu: { label: "esperando espacio en GPU", color: VISUAL.waiting_space.color, text: VISUAL.waiting_space.text, visual: "waiting_space" },
  prefill: { label: "prefill", color: VISUAL.prefill.color, text: VISUAL.prefill.text, visual: "prefill" },
  generating: { label: "generando", color: VISUAL.generating.color, text: VISUAL.generating.text, visual: "generating" },
  processing: { label: "procesando", color: VISUAL.processing.color, text: VISUAL.processing.text, visual: "processing" },
}

/** Sin actividad del worker durante este tiempo = probablemente colgada,
 *  mucho antes de que salte el umbral global de "atascada". */
export const STALL_SECONDS = 15

/** Tamaños de contexto: se configuran en potencias de dos (98304 = "96k"),
 *  así que se muestran en k de 1024 cuando el valor es múltiplo exacto. */
export function fmtContext(n: number | null | undefined): string {
  if (n == null || n <= 0) return "—"
  if (n >= 1024 && n % 1024 === 0) return `${n / 1024}k`
  return fmtTokens(n)
}


/** Reloj de audio: 75 → "1:15", 3725 → "1:02:05". */
export function fmtClock(seconds: number | null | undefined): string {
  if (seconds == null) return "—"
  const s = Math.max(0, Math.round(seconds))
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const ss = String(s % 60).padStart(2, "0")
  return h > 0 ? `${h}:${String(m).padStart(2, "0")}:${ss}` : `${m}:${ss}`
}

/** Velocidad respecto al tiempo real: 14.2 → "×14", 0.8 → "×0.8". */
export function fmtRealtime(factor: number | null | undefined): string {
  if (factor == null || factor <= 0) return "—"
  return `×${factor >= 10 ? Math.round(factor) : factor.toFixed(1)}`
}

/** Texto del avance de una transcripción en curso. */
export function audioProgressText(audio: {
  phase: string | null
  totalSeconds: number | null
  processedSeconds: number
  pct: number | null
  realtimeFactor: number | null
}): string {
  if (audio.phase === "diarizing") return "diarizando (quién habla)"
  const pct = audio.pct != null ? ` (${audio.pct.toFixed(0)}%)` : ""
  const speed = audio.realtimeFactor ? ` · ${fmtRealtime(audio.realtimeFactor)} tiempo real` : ""
  return `transcribiendo ${fmtClock(audio.processedSeconds)} / ${fmtClock(audio.totalSeconds)}${pct}${speed}`
}
