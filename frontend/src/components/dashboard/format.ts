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

export const STAGE_META: Record<RequestStage, { label: string; color: string; text: string }> = {
  queued: { label: "en cola", color: "#d97706", text: "text-amber-300" },
  backend_wait: { label: "esperando al backend", color: "#d97706", text: "text-amber-200" },
  loading: { label: "cargando modelo", color: "#8b5cf6", text: "text-violet-300" },
  waiting_for_gpu: { label: "esperando GPU", color: "#8b5cf6", text: "text-violet-300" },
  prefill: { label: "prefill", color: "#0ea5e9", text: "text-sky-300" },
  generating: { label: "generando", color: "#16a34a", text: "text-emerald-300" },
  processing: { label: "procesando", color: "#64748b", text: "text-slate-300" },
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
