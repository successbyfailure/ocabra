import { useEffect, useMemo, useState } from "react"
import { Activity, AlertTriangle, Gauge, Hourglass, PenLine, Plug, Zap } from "lucide-react"
import { api } from "@/api/client"
import type { EnergyStats, ModelActivity, RequestStats, ServerPower, TokenStats } from "@/types"
import { fmtTokens, fmtTps } from "./format"

function Kpi({
  icon: Icon,
  label,
  value,
  sub,
  color,
  title,
}: {
  icon: React.ElementType
  label: string
  value: string | number
  sub?: string
  color: string
  title?: string
}) {
  return (
    <div className="rounded-lg border border-border bg-card px-3.5 py-3" title={title}>
      <div className="flex items-center gap-3">
        <div className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-lg ${color}`}>
          <Icon size={17} />
        </div>
        <div className="min-w-0">
          <p className="truncate text-[11px] text-muted-foreground">{label}</p>
          <p className="truncate text-lg font-semibold tabular-nums">{value}</p>
          {sub && <p className="truncate text-[11px] text-muted-foreground">{sub}</p>}
        </div>
      </div>
    </div>
  )
}

interface WindowStats {
  tokens: TokenStats | null
  requests: RequestStats | null
}

async function loadWindow(fromMs: number): Promise<WindowStats> {
  const to = new Date().toISOString()
  const from = new Date(Date.now() - fromMs).toISOString()
  const [tokens, requests] = await Promise.all([
    api.stats.tokens({ from, to, includeSeries: false }).catch(() => null),
    api.stats.requests({ from, to }).catch(() => null),
  ])
  return { tokens, requests }
}

const pctLabel = (rate: number | undefined) => (rate == null ? "—" : `${(rate * 100).toFixed(1)}%`)

/**
 * KPIs de cabecera: primero lo que pasa ahora (de la actividad en vivo) y
 * después la última hora frente a las últimas 24 h.
 */
export function KpiStrip({
  activity,
  serverPower,
}: {
  activity: Record<string, ModelActivity>
  serverPower: ServerPower | null
}) {
  const [h1, setH1] = useState<WindowStats>({ tokens: null, requests: null })
  const [h24, setH24] = useState<WindowStats>({ tokens: null, requests: null })
  const [energy, setEnergy] = useState<EnergyStats | null>(null)

  useEffect(() => {
    let cancelled = false
    const load = async () => {
      const [a, b, e] = await Promise.all([
        loadWindow(3_600_000),
        loadWindow(86_400_000),
        api.stats
          .energy({ from: new Date(Date.now() - 86_400_000).toISOString(), to: new Date().toISOString() })
          .catch(() => null),
      ])
      if (cancelled) return
      setH1(a)
      setH24(b)
      setEnergy(e)
    }
    void load()
    const id = window.setInterval(() => void load(), 60_000)
    return () => {
      cancelled = true
      window.clearInterval(id)
    }
  }, [])

  const live = useMemo(() => {
    let running = 0
    let queued = 0
    let prefill = 0
    let generation = 0
    for (const act of Object.values(activity)) {
      const q = act.queuedCount ?? act.admission?.queuedRequests ?? 0
      queued += q
      running += Math.max(0, act.inFlight - q)
      prefill += act.prefillTokensPerSecondTotal ?? 0
      generation += act.generationTokensPerSecondTotal ?? 0
    }
    return { running, queued, prefill, generation }
  }, [activity])

  const tokens = (w: WindowStats) => (w.tokens ? w.tokens.totalInputTokens + w.tokens.totalOutputTokens : null)

  return (
    <div className="space-y-2.5">
      <div className="grid gap-2.5 sm:grid-cols-3 xl:grid-cols-5">
        <Kpi icon={Activity} label="En curso ahora" value={live.running} color="bg-sky-500/15 text-sky-300" />
        <Kpi icon={Hourglass} label="En cola ahora" value={live.queued} color={live.queued > 0 ? "bg-amber-500/15 text-amber-300" : "bg-muted text-muted-foreground"} />
        <Kpi icon={Gauge} label="Prefill ahora" value={fmtTps(live.prefill)} color="bg-sky-500/15 text-sky-300" title="Suma de las peticiones en prefill" />
        <Kpi icon={PenLine} label="Generación ahora" value={fmtTps(live.generation)} color="bg-emerald-500/15 text-emerald-300" title="Suma de las peticiones generando" />
        <Kpi
          icon={Plug}
          label="Potencia del servidor"
          value={serverPower?.totalPowerW != null ? `${Math.round(serverPower.totalPowerW)} W` : "—"}
          sub={
            serverPower?.totalGpuPowerW != null
              ? `GPU ${Math.round(serverPower.totalGpuPowerW)} W · CPU ${Math.round(serverPower.cpuPowerW ?? 0)} W`
              : undefined
          }
          color="bg-orange-500/15 text-orange-300"
        />
      </div>
      <div className="grid gap-2.5 sm:grid-cols-2 xl:grid-cols-4">
        <Kpi
          icon={Activity}
          label="Peticiones · 1 h / 24 h"
          value={`${h1.requests?.totalRequests.toLocaleString() ?? "—"} / ${h24.requests?.totalRequests.toLocaleString() ?? "—"}`}
          sub={h24.requests ? `p50 ${Math.round(h24.requests.p50DurationMs)} ms · p95 ${Math.round(h24.requests.p95DurationMs)} ms (24 h)` : undefined}
          color="bg-sky-500/15 text-sky-300"
        />
        <Kpi
          icon={Zap}
          label="Tokens · 1 h / 24 h"
          value={`${fmtTokens(tokens(h1))} / ${fmtTokens(tokens(h24))}`}
          sub={h24.tokens ? `${fmtTokens(h24.tokens.totalInputTokens)} entrada · ${fmtTokens(h24.tokens.totalOutputTokens)} salida (24 h)` : undefined}
          color="bg-fuchsia-500/15 text-fuchsia-300"
        />
        <Kpi
          icon={AlertTriangle}
          label="Errores · 1 h / 24 h"
          value={`${pctLabel(h1.requests?.errorRate)} / ${pctLabel(h24.requests?.errorRate)}`}
          sub={h24.requests ? `${h1.requests?.rejections ?? 0} / ${h24.requests.rejections} rechazos por capacidad (429)` : undefined}
          color={(h1.requests?.errorRate ?? 0) > 0.05 ? "bg-red-500/15 text-red-300" : "bg-muted text-muted-foreground"}
          title="Porcentaje de peticiones con error (sin contar los 429, que van aparte)"
        />
        <Kpi
          icon={Plug}
          label="Energía · 24 h"
          value={energy ? `${(energy.totalServerKwh ?? energy.totalKwh).toFixed(1)} kWh` : "—"}
          sub={energy ? `${energy.estimatedCostEur.toFixed(2)} €` : undefined}
          color="bg-orange-500/15 text-orange-300"
        />
      </div>
    </div>
  )
}
