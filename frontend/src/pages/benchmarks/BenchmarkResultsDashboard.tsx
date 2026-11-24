import { useEffect, useMemo, useState } from "react"
import { Award, Clock3, Cpu, Gauge, Layers3, Trophy } from "lucide-react"
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts"
import type { BenchmarkRun } from "@/types"
import { BenchmarkHistory } from "./BenchmarkHistory"
import { BenchmarkModelRanking } from "./BenchmarkModelRanking"
import { BenchmarkResults } from "./BenchmarkResults"
import {
  BENCHMARK_TOOLTIP_ITEM_STYLE,
  BENCHMARK_TOOLTIP_LABEL_STYLE,
  BENCHMARK_TOOLTIP_STYLE,
  BENCHMARK_TOOLTIP_WRAPPER_STYLE,
} from "./chartStyles"
import {
  formatMetric,
  parseBenchmarkResults,
  type BenchmarkMetric,
  type BenchmarkResultRow,
} from "./resultAdapters"

const PRIMARY_METRICS = [
  "non_live_accuracy",
  "strict_prompt_accuracy",
  "pass_rate",
  "coverage",
  "precision",
  "simple_accuracy",
  "speed_tok_s",
]

function durationLabel(run: BenchmarkRun): string {
  if (!run.startedAt || !run.finishedAt) return "—"
  const minutes = Math.max(0, Math.round(
    (new Date(run.finishedAt).getTime() - new Date(run.startedAt).getTime()) / 60_000,
  ))
  if (minutes < 60) return `${minutes} min`
  return `${Math.floor(minutes / 60)} h ${minutes % 60} min`
}

export function rankedMetricRows(rows: BenchmarkResultRow[], metricKey: string) {
  const values = rows.flatMap((row) => {
    const metric = row.metrics.find((item) => item.key === metricKey)
    return metric && typeof metric.value === "number" ? [{ row, metric }] : []
  })
  const direction = values[0]?.metric.better ?? "higher"
  return values.sort((a, b) => direction === "lower"
    ? a.metric.value as number - (b.metric.value as number)
    : (b.metric.value as number) - (a.metric.value as number))
}

export function preferredMetric(metrics: BenchmarkMetric[]): string {
  return PRIMARY_METRICS.find((key) => metrics.some((metric) => metric.key === key))
    ?? metrics.find((metric) => typeof metric.value === "number" && metric.better)?.key
    ?? metrics.find((metric) => typeof metric.value === "number")?.key
    ?? ""
}

function chartTick(value: number, metric: BenchmarkMetric | undefined): string {
  if (metric?.format === "percent") return `${Math.round(value * 100)}%`
  if (metric?.format === "seconds") return `${value.toFixed(0)}s`
  return value >= 1_000 ? `${(value / 1_000).toFixed(1)}k` : value.toFixed(value < 10 ? 1 : 0)
}

function WinnerCard({ metric, rows }: { metric: BenchmarkMetric; rows: BenchmarkResultRow[] }) {
  const winner = rankedMetricRows(rows, metric.key)[0]
  if (!winner) return null
  return (
    <div className="rounded-lg border border-border bg-background p-3">
      <p className="flex items-center gap-1.5 text-xs text-muted-foreground">
        <Award size={13} className="text-amber-400" /> {metric.label}
      </p>
      <p className="mt-2 truncate font-mono text-sm font-semibold" title={winner.row.model}>
        {winner.row.model}
      </p>
      <p className="mt-0.5 text-lg font-semibold text-emerald-400">
        {formatMetric(winner.metric)}
      </p>
    </div>
  )
}

export function BenchmarkResultsDashboard({ runs }: { runs: BenchmarkRun[] }) {
  const completed = useMemo(() => runs.filter((run) => (
    run.status === "completed" && Object.keys(run.result).length > 0
  )), [runs])
  const suiteKeys = useMemo(() => Array.from(new Set(
    completed.map((run) => `${run.app}/${run.suite}`),
  )), [completed])
  const [suiteKey, setSuiteKey] = useState("")
  const [runId, setRunId] = useState("")
  const [scope, setScope] = useState("")
  const [metricKey, setMetricKey] = useState("")
  const effectiveSuite = suiteKeys.includes(suiteKey) ? suiteKey : suiteKeys[0] ?? ""
  const eligible = completed.filter((run) => `${run.app}/${run.suite}` === effectiveSuite)
  const run = eligible.find((candidate) => candidate.id === runId) ?? eligible[0]
  const rows = useMemo(() => run ? parseBenchmarkResults(run.result) : [], [run])
  const scopes = useMemo(() => Array.from(new Set(rows.map((row) => row.scope))), [rows])
  const effectiveScope = scopes.includes(scope) ? scope : scopes[0] ?? ""
  const scopedRows = rows.filter((row) => row.scope === effectiveScope)
  const metrics = Array.from(new Map(
    scopedRows.flatMap((row) => row.metrics.map((metric) => [metric.key, metric])),
  ).values()).filter((metric) => typeof metric.value === "number")
  const effectiveMetric = metrics.some((metric) => metric.key === metricKey)
    ? metricKey
    : preferredMetric(metrics)
  const metricDefinition = metrics.find((metric) => metric.key === effectiveMetric)
  const ranking = rankedMetricRows(scopedRows, effectiveMetric)
  const chartData = ranking.map(({ row, metric }) => ({
    model: row.model,
    value: metric.value as number,
    formatted: formatMetric(metric),
  }))

  useEffect(() => {
    setRunId("")
    setScope("")
    setMetricKey("")
  }, [effectiveSuite])

  useEffect(() => {
    setScope("")
    setMetricKey("")
  }, [run?.id])

  if (completed.length === 0) {
    return (
      <div className="rounded-xl border border-dashed border-border p-12 text-center">
        <Trophy size={28} className="mx-auto text-muted-foreground" />
        <h2 className="mt-3 font-medium">Todavía no hay resultados completos</h2>
        <p className="mt-1 text-sm text-muted-foreground">Lanza una batería para empezar a comparar modelos.</p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <BenchmarkModelRanking runs={completed} />
      <section className="overflow-hidden rounded-xl border border-border bg-card">
        <div className="flex flex-wrap items-end justify-between gap-4 border-b border-border px-4 py-4">
          <div>
            <p className="text-xs font-medium uppercase tracking-wider text-primary">Comparativa</p>
            <h2 className="mt-1 text-lg font-semibold">Resultados por modelo</h2>
            <p className="text-xs text-muted-foreground">Elige una batería, ejecución y prueba para comparar sus métricas.</p>
          </div>
          <div className="flex flex-wrap gap-2">
            <label className="space-y-1 text-xs text-muted-foreground">
              <span className="block">Batería</span>
              <select value={effectiveSuite} onChange={(event) => setSuiteKey(event.target.value)} className="min-w-44 rounded-md border border-border bg-background px-2.5 py-2 text-sm text-foreground">
                {suiteKeys.map((key) => <option key={key} value={key}>{key}</option>)}
              </select>
            </label>
            <label className="space-y-1 text-xs text-muted-foreground">
              <span className="block">Ejecución</span>
              <select value={run?.id ?? ""} onChange={(event) => setRunId(event.target.value)} className="min-w-60 rounded-md border border-border bg-background px-2.5 py-2 text-sm text-foreground">
                {eligible.map((candidate) => <option key={candidate.id} value={candidate.id}>{new Date(candidate.createdAt).toLocaleString()} · {candidate.models.length} modelos</option>)}
              </select>
            </label>
          </div>
        </div>

        {run && (
          <>
            <div className="grid gap-px border-b border-border bg-border sm:grid-cols-2 xl:grid-cols-4">
              {[
                [<Layers3 size={15} />, "Modelos", String(run.models.length)],
                [<Clock3 size={15} />, "Duración", durationLabel(run)],
                [<Gauge size={15} />, "Intentos descartados", String(run.contaminatedAttempts)],
                [<Cpu size={15} />, "Revisión", run.revision.slice(0, 12)],
                ...(run.node ? [[<Cpu size={15} />, "Nodo", run.node]] : []),
              ].map(([icon, label, value]) => <div key={String(label)} className="bg-card px-4 py-3"><p className="flex items-center gap-1.5 text-xs text-muted-foreground">{icon}{label}</p><p className="mt-1 font-mono text-sm font-semibold">{value}</p></div>)}
            </div>

            <div className="space-y-5 p-4">
              {scopes.length > 1 && <div><p className="mb-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">Prueba</p><div className="flex flex-wrap gap-2">{scopes.map((item) => <button key={item} type="button" onClick={() => { setScope(item); setMetricKey("") }} className={`rounded-full border px-3 py-1.5 text-xs ${effectiveScope === item ? "border-primary bg-primary/10 text-primary" : "border-border text-muted-foreground hover:text-foreground"}`}>{item}</button>)}</div></div>}

              <div className="grid gap-4 xl:grid-cols-[minmax(0,1.8fr)_minmax(260px,0.7fr)]">
                <div className="rounded-lg border border-border bg-background p-3">
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div><h3 className="text-sm font-medium">Clasificación</h3><p className="text-xs text-muted-foreground">{metricDefinition?.better === "lower" ? "Menos es mejor" : "Más es mejor"}</p></div>
                    <div className="flex flex-wrap justify-end gap-1.5">{metrics.map((metric) => <button key={metric.key} type="button" onClick={() => setMetricKey(metric.key)} className={`rounded-md px-2 py-1 text-xs ${effectiveMetric === metric.key ? "bg-primary text-primary-foreground" : "bg-muted text-muted-foreground hover:text-foreground"}`}>{metric.label}</button>)}</div>
                  </div>
                  {chartData.length > 0 ? <div className="mt-3" style={{ height: Math.max(260, chartData.length * 38) }}><ResponsiveContainer width="100%" height="100%"><BarChart data={chartData} layout="vertical" margin={{ left: 8, right: 28 }}><CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="hsl(var(--border))" /><XAxis type="number" tickFormatter={(value) => chartTick(Number(value), metricDefinition)} tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }} axisLine={false} tickLine={false} /><YAxis type="category" dataKey="model" width={180} tick={{ fontSize: 11, fill: "hsl(var(--foreground))" }} axisLine={false} tickLine={false} /><Tooltip cursor={{ fill: "hsl(var(--muted) / 0.35)" }} contentStyle={BENCHMARK_TOOLTIP_STYLE} wrapperStyle={BENCHMARK_TOOLTIP_WRAPPER_STYLE} labelStyle={BENCHMARK_TOOLTIP_LABEL_STYLE} itemStyle={BENCHMARK_TOOLTIP_ITEM_STYLE} formatter={(value) => [metricDefinition ? formatMetric({ ...metricDefinition, value: Number(value) }) : String(value), metricDefinition?.label ?? "Valor"]} /><Bar dataKey="value" radius={[0, 5, 5, 0]}>{chartData.map((item, index) => <Cell key={item.model} fill={index === 0 ? "hsl(var(--primary))" : "hsl(var(--primary) / 0.42)"} />)}</Bar></BarChart></ResponsiveContainer></div> : <p className="py-12 text-center text-sm text-muted-foreground">No hay valores numéricos comparables.</p>}
                </div>

                <div className="space-y-3">
                  <div><h3 className="text-sm font-medium">Líderes por métrica</h3><p className="text-xs text-muted-foreground">Mejor valor dentro de esta prueba.</p></div>
                  <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-1">{metrics.filter((metric) => metric.better).slice(0, 5).map((metric) => <WinnerCard key={metric.key} metric={metric} rows={scopedRows} />)}</div>
                </div>
              </div>
            </div>
          </>
        )}
      </section>

      {run && <BenchmarkResults run={run} />}
      <BenchmarkHistory runs={completed} />
    </div>
  )
}
