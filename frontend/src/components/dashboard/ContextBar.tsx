import type { ModelContextInfo } from "@/types"
import { fmtContext, fmtTokens } from "./format"
import { gpuColor } from "./GpuBadge"
import { useGpuLabels } from "./gpuNames"

/** Tono estable por petición, compartido con la lista de peticiones. */
export const REQUEST_HUES = ["#2a78d6", "#1baf7a", "#b07cf0", "#e0a93b", "#4ec9d6", "#d4679a"] as const

export function requestHue(index: number): string {
  return REQUEST_HUES[index % REQUEST_HUES.length]
}

const HATCH =
  "repeating-linear-gradient(135deg, rgba(148,163,184,0.18) 0 4px, transparent 4px 8px)"

/**
 * Barra de contexto de un modelo.
 *
 * La escala es el mayor contexto que admite el modelo en cualquier
 * colocación (p. ej. 256k con 2 GPU). Una marca señala el tope de una sola
 * GPU (p. ej. 96k). Lo que queda por encima del contexto cargado ahora se
 * raya: solo está disponible escalando, con una recarga drenada.
 *
 * - `pool` (llama.cpp con KV unificada): una barra apilada; cada petición
 *   ocupa su reserva (claro) y lo que ya ha usado (sólido).
 * - `slots` (Ollama, llama.cpp sin pool): cada petición tiene su parcela
 *   fija; una línea por petición sobre la misma escala.
 * - `paged` (vLLM): solo un porcentaje de ocupación del pool.
 */
export function ContextBar({
  ctx,
  requestOrder,
}: {
  ctx: ModelContextInfo
  /** requestId → índice de color, el mismo que usa la lista de peticiones. */
  requestOrder: Map<string, number>
}) {
  const { list: gpuLabel } = useGpuLabels()
  const scale = ctx.scaleMaxTokens ?? ctx.activeMaxTokens ?? 0
  if (scale <= 0) return null
  const pct = (tokens: number) => Math.max(0, Math.min(100, (tokens / scale) * 100))
  const active = ctx.activeMaxTokens ?? scale
  const activePct = pct(active)
  const mark = ctx.singleGpuMaxTokens
  const showMark = mark != null && mark < scale
  const biggest = [...ctx.tiers].sort((a, b) => b.maxTokens - a.maxTokens)[0]
  const singleTier = ctx.tiers.find((t) => t.gpus.length === 1 && t.maxTokens === mark)

  const reserved = ctx.segments.reduce((acc, s) => acc + (s.reservedTokens ?? 0), 0)
  const used = ctx.segments.reduce((acc, s) => acc + (s.usedTokens ?? 0), 0)
  const anyUsed = ctx.segments.some((s) => s.usedTokens != null)

  const title =
    `Contexto activo ${fmtContext(active)}${ctx.activePresetKey ? ` (preset ${ctx.activePresetKey})` : ""}` +
    ` en ${gpuLabel(ctx.activeGpus)}. ` +
    (showMark ? `Hasta ${fmtContext(mark)} cabe en una GPU; hasta ${fmtContext(scale)} con ${gpuLabel(biggest?.gpus ?? [])}. ` : "") +
    (ctx.mode === "pool"
      ? "Pool KV compartido: cada petición reserva su presupuesto al entrar (claro) y va ocupando (sólido)."
      : ctx.mode === "slots"
        ? "Cada petición tiene una parcela fija de contexto."
        : "")

  return (
    <div className="space-y-1" title={title}>
      <div className="relative h-3 overflow-hidden rounded-full bg-muted">
        {/* Zona fuera del contexto cargado: solo escalando de colocación. */}
        {activePct < 100 && (
          <span
            className="absolute inset-y-0 right-0"
            style={{ left: `${activePct}%`, background: HATCH }}
            aria-hidden
          />
        )}
        {ctx.mode === "pool" && <PoolSegments ctx={ctx} pct={pct} requestOrder={requestOrder} />}
        {ctx.mode === "paged" && ctx.usagePct != null && (
          <span
            className="absolute inset-y-0 left-0 bg-sky-500/80"
            style={{ width: `${(activePct * ctx.usagePct) / 100}%` }}
          />
        )}
        {showMark && (
          <span
            className="absolute inset-y-0 w-0.5 bg-foreground/70"
            style={{ left: `calc(${pct(mark)}% - 1px)` }}
            aria-hidden
          />
        )}
        {/* Fin del contexto cargado, si no coincide con la marca. */}
        {activePct < 100 && active !== mark && (
          <span
            className="absolute inset-y-0 w-px bg-foreground/40"
            style={{ left: `${activePct}%` }}
            aria-hidden
          />
        )}
      </div>

      {ctx.mode === "slots" && ctx.segments.length > 0 && (
        <div className="space-y-0.5">
          {ctx.segments.map((s) => {
            const hue = requestHue(requestOrder.get(s.requestId) ?? 0)
            return (
              <div key={s.requestId} className="relative h-1 overflow-hidden rounded-full bg-muted">
                <span
                  className="absolute inset-y-0 left-0 opacity-30"
                  style={{ width: `${activePct}%`, background: hue }}
                />
                {s.usedTokens != null && (
                  <span
                    className="absolute inset-y-0 left-0"
                    style={{ width: `${pct(s.usedTokens)}%`, background: hue }}
                  />
                )}
              </div>
            )
          })}
        </div>
      )}

      <div className="relative h-4 text-[10px] tabular-nums text-muted-foreground">
        <span className="absolute left-0">0</span>
        {showMark && (
          <span
            className="absolute -translate-x-1/2 whitespace-nowrap font-medium"
            style={{
              left: `${pct(mark)}%`,
              color: singleTier?.gpus.length === 1 ? gpuColor(singleTier.gpus[0]) : undefined,
            }}
          >
            {fmtContext(mark)} · {gpuLabel(singleTier?.gpus ?? [])}
          </span>
        )}
        <span className="absolute right-0 whitespace-nowrap">
          {fmtContext(scale)}
          {biggest && biggest.gpus.length > 1 ? ` · ${gpuLabel(biggest.gpus)}` : ""}
        </span>
      </div>

      <p className="text-[10.5px] text-muted-foreground">
        {ctx.mode === "pool" && (
          <>
            <span title="Cada tramo es lo que reserva una petición (su prompt + su max_tokens), no un slot fijo: el pool es compartido">
              Pool compartido ·{" "}
            </span>
            Reservado {fmtTokens(ctx.reservedTokens ?? reserved)}
            {anyUsed ? ` · usado ${fmtTokens(used)}` : ""} de {fmtContext(active)} cargados
          </>
        )}
        {ctx.mode === "slots" && (
          <>
            {ctx.segments.length} de {ctx.tiers.find((t) => t.maxTokens === active)?.slots ?? "?"} parcelas de{" "}
            {fmtContext(active)} en uso
          </>
        )}
        {ctx.mode === "paged" &&
          (ctx.usagePct != null ? `Pool KV al ${ctx.usagePct.toFixed(0)}%` : "vLLM no publica aún la ocupación del pool")}
        {activePct < 100 && <span className="text-muted-foreground/70"> · rayado: requiere escalar</span>}
        {ctx.liveResized && ctx.nominalMaxTokens != null && (
          <span className="text-sky-300" title="El contexto se cambió en vivo, sin recargar el modelo">
            {" "}· cambiado en vivo (cargado con {fmtContext(ctx.nominalMaxTokens)})
          </span>
        )}
      </p>
    </div>
  )
}

function PoolSegments({
  ctx,
  pct,
  requestOrder,
}: {
  ctx: ModelContextInfo
  pct: (tokens: number) => number
  requestOrder: Map<string, number>
}) {
  let offset = 0
  return (
    <>
      {ctx.segments.map((s) => {
        const reserved = s.reservedTokens ?? s.usedTokens ?? 0
        if (reserved <= 0) return null
        const left = offset
        offset += reserved
        const hue = requestHue(requestOrder.get(s.requestId) ?? 0)
        const usedTokens = s.usedTokens != null ? Math.min(s.usedTokens, reserved) : null
        return (
          <span key={s.requestId} className="absolute inset-y-0" style={{ left: `${pct(left)}%`, width: `${pct(reserved)}%` }}>
            <span className="absolute inset-0 border-r border-background/60 opacity-35" style={{ background: hue }} />
            {usedTokens != null && (
              <span
                className="absolute inset-y-0 left-0"
                style={{ width: `${(usedTokens / reserved) * 100}%`, background: hue }}
              />
            )}
          </span>
        )
      })}
    </>
  )
}
