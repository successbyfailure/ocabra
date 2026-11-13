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
 * la petición.
 *
 * «preparación» es lo que pasa entre la admisión y el prefill: el motor
 * tokeniza el prompt ENTERO y aplica la plantilla de chat en cada petición,
 * aunque casi todo salga de la cache. Medido el 2026-11-13: 236 ms para 70k
 * tokens en gemma4:26b, ~1,1 s con 141k en ornith. Con mucha cache el prefill
 * se queda en nada y este tramo es lo que se ve. Sale de ``ttft_ms``. Lo que
 * queda tras la generación (red, streaming al cliente) es «resto».
 */

export interface RequestPhase {
  key: "queue" | "load" | "prep" | "prefill" | "generation" | "other"
  label: string
  ms: number
  color: string
}

const PHASE_META: Record<RequestPhase["key"], { label: string; color: string }> = {
  queue: { label: "cola", color: "#64748b" },
  load: { label: "carga", color: "#f59e0b" },
  prep: { label: "preparación", color: "#0e7490" },
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
  const queue = num(r.queueMs ?? t.queue_ms)
  const load = num(t.load_ms)
  const ttft = num(t.ttft_ms)
  // Del inicio al primer token: cola + carga + preparación + prefill.
  const prep = ttft != null && prefill != null ? ttft - (queue ?? 0) - (load ?? 0) - prefill : null
  const total0 = num(r.durationMs)
  const noise = Math.max(50, (total0 ?? 0) * 0.03)
  const parts: [RequestPhase["key"], number | null][] = [
    ["queue", queue],
    ["load", load],
    ["prep", prep != null && prep > noise ? prep : null],
    ["prefill", prefill],
    ["generation", generation],
  ]
  const phases: RequestPhase[] = parts
    .filter(([, ms]) => ms != null && ms > 0)
    .map(([key, ms]) => ({ key, ms: ms!, ...PHASE_META[key] }))
  const known = phases.reduce((a, p) => a + p.ms, 0)
  const total = Math.max(total0 ?? known, known)
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
  // Cola, carga y preparación solo se nombran cuando pesan (la barra ya las muestra).
  const extra = phases.filter((p) => (p.key === "queue" || p.key === "load" || p.key === "prep") && p.ms >= 100)
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
      {(["queue", "load", "prep", "prefill", "generation", "other"] as const).map((k) => (
        <span key={k} className="inline-flex items-center gap-1">
          <span className="inline-block h-1.5 w-2.5 rounded-sm" style={{ background: PHASE_META[k].color }} />
          {PHASE_META[k].label}
        </span>
      ))}
    </span>
  )
}

/** Velocidad de prefill con sentido: el motor la mide solo sobre los tokens
 *  que no estaban en cache, así que con casi todo en cache (p. ej. 1 token
 *  nuevo) da cifras absurdas como 2 t/s. Entonces se dice cuántos eran. */
export function prefillSpeedText(r: RecentRequest, fmt: (tps: number | null | undefined) => string): string | null {
  const fresh = r.inputTokens != null && r.cachedTokens != null ? Math.max(0, r.inputTokens - r.cachedTokens) : null
  if (fresh != null && r.cachedTokens && fresh < 256) return fresh === 0 ? "todo en cache" : `${fresh} ${fresh === 1 ? "token nuevo" : "tokens nuevos"}`
  if (!r.promptTokensPerSecond) return null
  // Con parte en cache la velocidad es la de los tokens nuevos, y unos pocos
  // cientos no llenan la GPU: 332 nuevos sobre 141k salen a ~800 t/s en un
  // modelo que hace miles con un prompt entero. Se dice sobre cuántos es.
  if (fresh != null && r.cachedTokens) return `${fmt(r.promptTokensPerSecond)} · ${fmtTokens(fresh)} nuevos`
  return fmt(r.promptTokensPerSecond)
}
