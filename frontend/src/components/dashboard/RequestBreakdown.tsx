import type { RecentRequest } from "@/types"
import { fmtTokens } from "./format"

/**
 * Desglose de una petición terminada: en qué se fue el tiempo (cola, carga del
 * modelo, prefill, generación) y cuánto de la entrada salió de la cache de
 * prompt. Lo comparten la tabla del dashboard, el log de Stats, la página de
 * Logs y el resumen del editor de modelos.
 *
 * Fuentes: ``generationMs`` lo informa el motor (llama.cpp ``predicted_ms``,
 * Ollama ``eval_duration``); prefill, cola y carga vienen de la telemetría de
 * la petición. Lo que no cuadra con la duración total (red, streaming al
 * cliente, espera de slot) sale como «resto».
 */

export interface RequestPhase {
  key: "queue" | "load" | "prefill" | "generation" | "other"
  label: string
  ms: number
  color: string
}

const PHASE_META: Record<RequestPhase["key"], { label: string; color: string }> = {
  queue: { label: "cola", color: "#64748b" },
  load: { label: "carga", color: "#f59e0b" },
  prefill: { label: "prefill", color: "#38bdf8" },
  generation: { label: "generación", color: "#a78bfa" },
  other: { label: "resto", color: "#334155" },
}

const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) && v >= 0 ? v : null)

/** Duración con un decimal por debajo de 10 s: 850 ms · 1.2 s · 36 s · 2m05s. */
export function fmtSpan(ms: number | null | undefined): string {
  if (ms == null) return "—"
  if (ms < 1000) return `${Math.round(ms)} ms`
  if (ms < 10_000) return `${(ms / 1000).toFixed(1)} s`
  const s = Math.round(ms / 1000)
  if (s < 60) return `${s} s`
  return `${Math.floor(s / 60)}m${String(s % 60).padStart(2, "0")}s`
}

/** Fases con tiempo de una petición, o null si el backend no informa de
 *  prefill ni de generación (no hay nada que desglosar). */
export function requestPhases(r: RecentRequest): { phases: RequestPhase[]; total: number } | null {
  const t = r.runtimeTelemetry ?? {}
  const prefill = num(t.prefill_ms)
  const generation = num(r.generationMs)
  if (prefill == null && generation == null) return null
  const parts: [RequestPhase["key"], number | null][] = [
    ["queue", num(r.queueMs ?? t.queue_ms)],
    ["load", num(t.load_ms)],
    ["prefill", prefill],
    ["generation", generation],
  ]
  const phases: RequestPhase[] = parts
    .filter(([, ms]) => ms != null && ms > 0)
    .map(([key, ms]) => ({ key, ms: ms!, ...PHASE_META[key] }))
  const known = phases.reduce((a, p) => a + p.ms, 0)
  const total = Math.max(num(r.durationMs) ?? known, known)
  const rest = total - known
  // Lo que queda por debajo del 3 % es ruido de medida: no merece segmento.
  if (rest > Math.max(50, total * 0.03)) phases.push({ key: "other", ms: rest, ...PHASE_META.other })
  return { phases, total }
}

/** Duración total con su barra por fases y el texto «prefill X · gen Y». */
export function TimingBreakdown({ r, compact = false }: { r: RecentRequest; compact?: boolean }) {
  const split = requestPhases(r)
  const ttft = num(r.runtimeTelemetry?.ttft_ms)
  if (!split) return <span className="tabular-nums">{fmtSpan(r.durationMs)}</span>
  const { phases, total } = split
  const title = [
    ...phases.map((p) => `${p.label}: ${fmtSpan(p.ms)}`),
    ttft != null ? `primer token a los ${fmtSpan(ttft)}` : null,
    `total: ${fmtSpan(total)}`,
  ]
    .filter(Boolean)
    .join("\n")
  const main = phases.filter((p) => p.key === "prefill" || p.key === "generation")
  // Cola y carga solo se nombran cuando pesan (la barra ya las muestra).
  const extra = phases.filter((p) => (p.key === "queue" || p.key === "load") && p.ms >= 100)
  return (
    <div className={compact ? "min-w-[110px]" : "min-w-[150px]"} title={title} data-testid="timing-breakdown">
      <div className="flex items-baseline justify-end gap-1.5 tabular-nums">
        <span className="font-medium text-foreground/90">{fmtSpan(total)}</span>
      </div>
      <div className="mt-0.5 flex h-1.5 overflow-hidden rounded-full bg-muted/40">
        {phases.map((p) => (
          <div key={p.key} style={{ width: `${(p.ms / total) * 100}%`, background: p.color }} />
        ))}
      </div>
      <div className="mt-0.5 flex flex-wrap justify-end gap-x-1.5 text-[10.5px] tabular-nums text-muted-foreground">
        {main.map((p) => (
          <span key={p.key}>
            <span style={{ color: p.color }}>{p.key === "generation" ? "gen" : p.label}</span> {fmtSpan(p.ms)}
          </span>
        ))}
        {!compact &&
          extra.map((p) => (
            <span key={p.key}>
              <span style={{ color: p.color }}>{p.label}</span> {fmtSpan(p.ms)}
            </span>
          ))}
      </div>
    </div>
  )
}

/** Tokens de entrada servidos desde la cache de prompt. */
export function cacheInfo(r: RecentRequest): { cached: number; input: number; pct: number } | null {
  const cached = num(r.cachedTokens)
  const input = num(r.inputTokens)
  if (cached == null || input == null || input <= 0) return null
  return { cached, input, pct: Math.min(100, Math.round((cached / input) * 100)) }
}

export function CacheCell({ r }: { r: RecentRequest }) {
  const info = cacheInfo(r)
  if (!info) return <span className="text-muted-foreground/50" title="El backend no informa de la cache">—</span>
  if (info.cached === 0) return <span className="text-muted-foreground/70">sin cache</span>
  const tone = info.pct >= 80 ? "text-emerald-300" : info.pct >= 30 ? "text-sky-300" : "text-muted-foreground"
  return (
    <div className="min-w-[80px]" title={`${info.cached.toLocaleString()} de ${info.input.toLocaleString()} tokens de entrada salieron de la cache`} data-testid="cache-cell">
      <div className={`tabular-nums ${tone}`}>
        {fmtTokens(info.cached)} <span className="text-[10.5px]">· {info.pct}%</span>
      </div>
      <div className="mt-0.5 h-1 overflow-hidden rounded-full bg-muted/40">
        <div className="h-full bg-emerald-400/70" style={{ width: `${info.pct}%` }} />
      </div>
    </div>
  )
}

/** Leyenda de colores de las fases, para la cabecera de una tabla. */
export function PhaseLegend() {
  return (
    <span className="inline-flex flex-wrap items-center gap-x-2 text-[10.5px] text-muted-foreground">
      {(["queue", "load", "prefill", "generation", "other"] as const).map((k) => (
        <span key={k} className="inline-flex items-center gap-1">
          <span className="inline-block h-1.5 w-2.5 rounded-sm" style={{ background: PHASE_META[k].color }} />
          {PHASE_META[k].label}
        </span>
      ))}
    </span>
  )
}
