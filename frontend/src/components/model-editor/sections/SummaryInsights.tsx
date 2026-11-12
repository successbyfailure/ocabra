import { useEffect, useState } from "react"
import { useNavigate } from "react-router-dom"
import { FlaskConical, Gauge, MessageSquareText } from "lucide-react"
import { api } from "@/api/client"
import { fmtDuration, fmtTokens, fmtTps } from "@/components/dashboard/format"
import type { RecentRequest } from "@/types"
import { cn } from "@/lib/cn"
import { Chip, Hint, Panel } from "../ui"
import type { SectionProps } from "./types"

/** Tiempo esperado de una petición típica con los ritmos medidos. */
export function typicalRequestMs(
  rates: { prefillTps: number | null; generationTps: number | null; typicalOutputTokens: number | null },
  promptTokens = 4000,
): number | null {
  if (!rates.prefillTps || !rates.generationTps) return null
  const output = rates.typicalOutputTokens ?? 500
  return Math.round((promptTokens / rates.prefillTps + output / rates.generationTps) * 1000)
}

function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="rounded-xl border border-border bg-muted/20 px-3 py-2" title={hint}>
      <p className="text-[11px] text-muted-foreground">{label}</p>
      <p className="text-lg font-semibold tabular-nums">{value}</p>
    </div>
  )
}

export function PerformancePanel({ editor, insights, onGoTo }: SectionProps & { onGoTo: (s: string) => void }) {
  const navigate = useNavigate()
  const m = editor.model
  if (!m) return null
  const r = insights?.rates
  const typical = r ? typicalRequestMs(r) : null
  const hasRates = Boolean(r && (r.prefillTps || r.generationTps))
  return (
    <Panel
      title="Rendimiento medido"
      subtitle={hasRates ? "Con el tráfico real de las últimas horas." : "Todavía no hay medidas de este modelo."}
      actions={
        <>
          <button
            type="button"
            className="inline-flex items-center gap-1 rounded-md border border-border px-2 py-1 text-xs hover:bg-muted"
            onClick={() => navigate(`/playground?model=${encodeURIComponent(m.model_id)}`)}
          >
            <MessageSquareText size={12} /> Probar en Playground
          </button>
          {editor.schema?.isLlm && (
            <button
              type="button"
              className="inline-flex items-center gap-1 rounded-md border border-border px-2 py-1 text-xs hover:bg-muted"
              onClick={() => onGoTo("presets")}
              title="Medir carga en frío y velocidad por preset"
            >
              <FlaskConical size={12} /> Medir
            </button>
          )}
        </>
      }
    >
      <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-5" data-testid="performance">
        <Stat
          label="Prefill"
          value={fmtTps(r?.prefillTps)}
          hint={r?.prefillSource === "regression" ? "Estimado con la regresión del estimador (pocas medidas)" : "Mediana medida"}
        />
        <Stat label="Generación" value={fmtTps(r?.generationTps)} hint="Mediana medida" />
        <Stat label="Salida típica" value={r?.typicalOutputTokens ? `${fmtTokens(r.typicalOutputTokens)} tok` : "—"} hint="p75 de los tokens de salida" />
        <Stat label="Carga en frío" value={fmtDuration(insights?.typicalLoadMs)} hint="Mediana de los últimos 30 días" />
        <Stat label="Petición típica" value={fmtDuration(typical)} hint="4k tokens de entrada + la salida típica, con los ritmos medidos" />
      </div>
      {r?.prefillSource === "regression" && <Hint>El prefill sale de la regresión del estimador: hay pocas medidas reales.</Hint>}
    </Panel>
  )
}

/** Últimas peticiones terminadas de este modelo (por cualquiera de sus perfiles). */
export function RecentForModel({ editor, insights, logical }: SectionProps) {
  const [rows, setRows] = useState<RecentRequest[] | null>(null)
  const key = logical?.modelKey
  const names = new Set([...(insights?.profileIds ?? []), key, editor.model?.model_id].filter(Boolean) as string[])
  useEffect(() => {
    let cancelled = false
    api.stats
      .recent(200)
      .then((res) => {
        if (!cancelled) setRows(res.requests.filter((r) => !r.inFlight))
      })
      .catch(() => !cancelled && setRows([]))
    return () => {
      cancelled = true
    }
  }, [])
  const mine = (rows ?? []).filter((r) => names.has(r.modelId) || (r.requestedModel != null && names.has(r.requestedModel))).slice(0, 8)
  return (
    <Panel title="Últimas peticiones" subtitle="De cualquiera de los perfiles de este modelo.">
      {rows === null ? (
        <Hint>Cargando…</Hint>
      ) : mine.length === 0 ? (
        <Hint>No hay peticiones recientes de este modelo.</Hint>
      ) : (
        <table className="w-full text-xs" data-testid="model-recent">
          <tbody className="divide-y divide-border/60">
            {mine.map((r) => {
              const error = Boolean(r.error) || (r.statusCode != null && r.statusCode >= 400)
              return (
                <tr key={r.id}>
                  <td className="py-1 pr-2 font-mono text-muted-foreground">
                    {new Date(r.startedAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                  </td>
                  <td className="py-1 pr-2">{r.requestedModel && r.requestedModel !== r.modelId ? r.requestedModel : r.modelId}</td>
                  <td className="py-1 pr-2 text-muted-foreground">{r.username ?? r.apiKeyName ?? "anónimo"}</td>
                  <td className="py-1 pr-2 text-right tabular-nums">{fmtDuration(r.durationMs)}</td>
                  <td className="py-1 pr-2 text-right tabular-nums text-muted-foreground">
                    {r.inputTokens != null ? `${fmtTokens(r.inputTokens)} → ${fmtTokens(r.outputTokens ?? 0)}` : "—"}
                  </td>
                  <td className="py-1 pr-2 text-right tabular-nums text-muted-foreground">{fmtTps(r.predictedTokensPerSecond)}</td>
                  <td className="py-1 text-right">
                    <Chip tone={error ? "error" : "ok"}>{r.statusCode ?? (r.error ? "error" : "—")}</Chip>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      )}
    </Panel>
  )
}

/** Recomendación de política a partir del coste de carga en frío. */
export function LoadCostHint({ insights, policy }: { insights?: SectionProps["insights"]; policy: string }) {
  const ms = insights?.typicalLoadMs
  if (!ms) return null
  const slow = ms > 30_000
  return (
    <div className={cn("mt-3 flex items-center gap-2 rounded-lg border px-3 py-2 text-xs", slow ? "border-amber-500/40 bg-amber-500/10" : "border-border bg-muted/20")}>
      <Gauge size={14} className={slow ? "text-amber-300" : "text-muted-foreground"} />
      <span>
        Cargarlo en frío tarda <span className="font-semibold tabular-nums">{fmtDuration(ms)}</span> (mediana).
        {slow && policy === "on_demand" && " Si recibe tráfico a menudo, «Caliente» evita esa espera en la primera petición tras descargarse."}
        {!slow && policy !== "on_demand" && " Carga rápido: «Bajo demanda» liberaría la GPU sin apenas coste."}
      </span>
    </div>
  )
}
