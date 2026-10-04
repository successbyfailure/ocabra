import { useMemo, useState } from "react"
import {
  Area,
  AreaChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts"
import type { GpuUsageStats } from "@/types"

type Metric = "utilizationPct" | "vramMb" | "powerW" | "tempC"

const METRIC_LABEL: Record<Metric, string> = {
  utilizationPct: "Uso (%)",
  vramMb: "VRAM (MB)",
  powerW: "Potencia (W)",
  tempC: "Temp (°C)",
}

const METRIC_UNIT: Record<Metric, string> = {
  utilizationPct: "%",
  vramMb: " MB",
  powerW: " W",
  tempC: " °C",
}

const GPU_COLORS = ["#3b82f6", "#f97316", "#10b981", "#a855f7"]

function formatHours(h: number): string {
  if (h >= 24) {
    const days = Math.floor(h / 24)
    const rem = h - days * 24
    return `${days}d ${rem.toFixed(1)}h`
  }
  return `${h.toFixed(2)} h`
}

function formatVram(mb: number): string {
  if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`
  return `${mb} MB`
}

function groupSeriesByTimestamp(
  data: GpuUsageStats | null,
  metric: Metric,
): Array<Record<string, number | string>> {
  if (!data) return []
  const byT = new Map<string, Record<string, number | string>>()
  for (const point of data.series) {
    const existing = byT.get(point.t) ?? { t: point.t }
    existing[`gpu${point.gpuIndex}`] = point[metric]
    byT.set(point.t, existing)
  }
  return Array.from(byT.values()).sort((a, b) =>
    String(a.t).localeCompare(String(b.t)),
  )
}

interface GpuUsagePanelProps {
  data: GpuUsageStats | null
  loading?: boolean
}

export function GpuUsagePanel({ data, loading }: GpuUsagePanelProps) {
  const [metric, setMetric] = useState<Metric>("utilizationPct")

  const chartRows = useMemo(() => groupSeriesByTimestamp(data, metric), [data, metric])
  const gpuIndexes = useMemo(() => {
    if (!data) return [] as number[]
    const set = new Set<number>()
    data.series.forEach((p) => set.add(p.gpuIndex))
    return Array.from(set).sort((a, b) => a - b)
  }, [data])

  if (loading && !data) {
    return (
      <div className="rounded-lg border border-border bg-card p-4 text-sm text-muted-foreground">
        Cargando estadísticas de GPU…
      </div>
    )
  }

  if (!data || data.totals.length === 0) {
    return (
      <div className="rounded-lg border border-border bg-card p-4 text-sm text-muted-foreground">
        Sin muestras de GPU en el rango seleccionado.
      </div>
    )
  }

  const tickFormatter = (value: string) => {
    const d = new Date(value)
    if (Number.isNaN(d.getTime())) return value
    if (data.bucket === "day") return d.toLocaleDateString()
    return d.toLocaleString([], { month: "short", day: "numeric", hour: "numeric" })
  }

  return (
    <section className="space-y-3 rounded-lg border border-border bg-card p-4">
      <header className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 className="text-sm font-semibold text-foreground">Uso de GPU</h3>
          <p className="text-xs text-muted-foreground">
            Rango agregado · granularidad {data.bucket === "day" ? "diaria" : "horaria"} · umbral activo ≥ {data.activeThresholdPct}% uso
          </p>
        </div>
        <div className="inline-flex rounded-md border border-border bg-background p-0.5 text-xs">
          {(Object.keys(METRIC_LABEL) as Metric[]).map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => setMetric(m)}
              className={`rounded px-2 py-1 transition ${
                metric === m
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:bg-muted"
              }`}
            >
              {METRIC_LABEL[m]}
            </button>
          ))}
        </div>
      </header>

      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead className="text-left text-muted-foreground">
            <tr>
              <th className="py-1 pr-3">GPU</th>
              <th className="py-1 pr-3">Horas cubiertas</th>
              <th className="py-1 pr-3">Horas activas</th>
              <th className="py-1 pr-3">% activo</th>
              <th className="py-1 pr-3">Util media</th>
              <th className="py-1 pr-3">VRAM media / pico</th>
              <th className="py-1 pr-3">Temp media</th>
              <th className="py-1 pr-3">Energía</th>
            </tr>
          </thead>
          <tbody>
            {data.totals.map((t) => (
              <tr key={t.gpuIndex} className="border-t border-border/60">
                <td className="py-1 pr-3 font-mono font-semibold">
                  <span
                    className="mr-1 inline-block h-2 w-2 rounded-full align-middle"
                    style={{ backgroundColor: GPU_COLORS[t.gpuIndex % GPU_COLORS.length] }}
                  />
                  GPU {t.gpuIndex}
                </td>
                <td className="py-1 pr-3">{formatHours(t.hoursTotal)}</td>
                <td className="py-1 pr-3">{formatHours(t.hoursActive)}</td>
                <td className="py-1 pr-3">{(t.activeRatio * 100).toFixed(1)}%</td>
                <td className="py-1 pr-3">{t.avgUtilizationPct.toFixed(1)}%</td>
                <td className="py-1 pr-3">
                  {formatVram(t.avgVramMb)} · <span className="text-muted-foreground">pico {formatVram(t.peakVramMb)}</span>
                </td>
                <td className="py-1 pr-3">{t.avgTempC.toFixed(1)} °C</td>
                <td className="py-1 pr-3">{t.kwh.toFixed(2)} kWh</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          {metric === "utilizationPct" ? (
            <AreaChart data={chartRows}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="t" tickFormatter={tickFormatter} stroke="#94a3b8" minTickGap={40} />
              <YAxis
                stroke="#94a3b8"
                domain={[0, 100]}
                unit={METRIC_UNIT[metric]}
              />
              <Tooltip
                labelFormatter={(v) => tickFormatter(String(v))}
                formatter={(value: number, name: string) => [
                  `${Number(value).toFixed(1)}${METRIC_UNIT[metric]}`,
                  name,
                ]}
                contentStyle={{ background: "#0f172a", border: "1px solid #334155", borderRadius: 6 }}
              />
              <Legend />
              {gpuIndexes.map((idx) => (
                <Area
                  key={idx}
                  type="monotone"
                  dataKey={`gpu${idx}`}
                  name={`GPU ${idx}`}
                  stroke={GPU_COLORS[idx % GPU_COLORS.length]}
                  fill={GPU_COLORS[idx % GPU_COLORS.length]}
                  fillOpacity={0.15}
                  strokeWidth={2}
                  isAnimationActive={false}
                />
              ))}
            </AreaChart>
          ) : (
            <LineChart data={chartRows}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="t" tickFormatter={tickFormatter} stroke="#94a3b8" minTickGap={40} />
              <YAxis stroke="#94a3b8" unit={METRIC_UNIT[metric]} />
              <Tooltip
                labelFormatter={(v) => tickFormatter(String(v))}
                formatter={(value: number, name: string) => [
                  `${Number(value).toFixed(metric === "vramMb" ? 0 : 1)}${METRIC_UNIT[metric]}`,
                  name,
                ]}
                contentStyle={{ background: "#0f172a", border: "1px solid #334155", borderRadius: 6 }}
              />
              <Legend />
              {gpuIndexes.map((idx) => (
                <Line
                  key={idx}
                  type="monotone"
                  dataKey={`gpu${idx}`}
                  name={`GPU ${idx}`}
                  stroke={GPU_COLORS[idx % GPU_COLORS.length]}
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={false}
                />
              ))}
            </LineChart>
          )}
        </ResponsiveContainer>
      </div>
    </section>
  )
}
