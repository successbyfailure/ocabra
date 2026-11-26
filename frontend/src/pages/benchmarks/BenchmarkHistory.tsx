import { useEffect, useMemo, useState } from "react"
import { GitCompareArrows, Star } from "lucide-react"
import type { BenchmarkRun } from "@/types"
import { formatMetric, parseBenchmarkResults, type BenchmarkMetric } from "./resultAdapters"

export interface HistoricalRow {
  run: BenchmarkRun
  model: string
  scope: string
  metrics: BenchmarkMetric[]
}

export function historicalRows(runs: BenchmarkRun[]): HistoricalRow[] {
  return runs.flatMap((run) => parseBenchmarkResults(run.result).map((row) => ({
    run, model: row.model, scope: row.scope, metrics: row.metrics,
  })))
}

function deltaLabel(metric: BenchmarkMetric, baseline: BenchmarkMetric | undefined): string | null {
  if (typeof metric.value !== "number" || typeof baseline?.value !== "number") return null
  const delta = metric.value - baseline.value
  if (Math.abs(delta) < 1e-9) return "="
  const positive = delta > 0
  const good = metric.better === "lower" ? !positive : metric.better === "higher" ? positive : null
  const amount = metric.format === "percent"
    ? `${positive ? "+" : ""}${(delta * 100).toFixed(1)} pp`
    : `${positive ? "+" : ""}${delta.toFixed(metric.format === "integer" ? 0 : 2)}`
  return `${good === true ? "▲" : good === false ? "▼" : "Δ"} ${amount}`
}

export function runFingerprint(run: BenchmarkRun, model: string): string {
  const attempt = [...run.attempts].reverse().find(
    (item) => item.status === "completed" && item.model === model,
  )
  const software = attempt?.environment.software
  const backend = attempt?.environment.model?.backends?.[0]
  const gpu = attempt?.environment.hardware?.gpus?.map((item) => `GPU${item.index}`).join("+")
  return [
    software?.ocabra ? `v${software.ocabra}` : null,
    backend,
    gpu,
    run.node ? `en ${run.node}` : null,
  ].filter(Boolean).join(" · ")
}

export function BenchmarkHistory({ runs }: { runs: BenchmarkRun[] }) {
  const completed = useMemo(() => runs.filter((run) => run.status === "completed"), [runs])
  const suites = useMemo(() => Array.from(new Set(completed.map((run) => `${run.app}/${run.suite}`))), [completed])
  const [suiteKey, setSuiteKey] = useState("")
  const [selected, setSelected] = useState<string[]>([])
  const effectiveSuite = suites.includes(suiteKey) ? suiteKey : suites[0] ?? ""
  const eligible = completed.filter((run) => `${run.app}/${run.suite}` === effectiveSuite)

  useEffect(() => {
    const baseline = eligible.find((run) => run.isBaseline)
    const defaults = [baseline, ...eligible].filter((run): run is BenchmarkRun => Boolean(run))
    setSelected(Array.from(new Set(defaults.map((run) => run.id))).slice(0, 4))
  // The suite key intentionally resets the selection; run refreshes preserve it.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [effectiveSuite])

  const chosen = eligible.filter((run) => selected.includes(run.id))
  const rows = historicalRows(chosen)
  const definitions = Array.from(new Map(
    rows.flatMap((row) => row.metrics.map((metric) => [metric.key, metric])),
  ).values())
  const baselineRun = eligible.find((run) => run.isBaseline)
  const baselineRows = baselineRun ? historicalRows([baselineRun]) : []

  if (completed.length === 0) return null
  return (
    <section className="rounded-xl border border-border bg-card">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border px-4 py-3">
        <div><h2 className="inline-flex items-center gap-2 font-medium"><GitCompareArrows size={16} /> Comparación histórica</h2><p className="text-xs text-muted-foreground">Selecciona hasta cuatro ejecuciones de la misma suite.</p></div>
        <select value={effectiveSuite} onChange={(event) => setSuiteKey(event.target.value)} className="rounded-md border border-border bg-background px-2 py-1.5 text-xs">
          {suites.map((suite) => <option key={suite} value={suite}>{suite}</option>)}
        </select>
      </div>
      <div className="space-y-3 p-4">
        <div className="flex flex-wrap gap-2">
          {eligible.map((run) => {
            const active = selected.includes(run.id)
            return <button key={run.id} type="button" onClick={() => setSelected((current) => active ? current.filter((id) => id !== run.id) : current.length < 4 ? [...current, run.id] : current)} className={`rounded-full border px-2.5 py-1 text-xs ${active ? "border-primary bg-primary/10 text-primary" : "border-border text-muted-foreground"}`}>
              {run.isBaseline && <Star size={11} className="mr-1 inline fill-current" />}{new Date(run.createdAt).toLocaleDateString()} · {run.id.slice(0, 6)}{run.scoringIssue && <span className="ml-1 text-destructive" title={run.scoringIssue.reason}>· nota no válida</span>}
            </button>
          })}
        </div>
        {rows.length > 0 ? <div className="overflow-x-auto rounded-lg border border-border"><table className="w-full min-w-max text-left text-xs">
          <thead className="bg-muted/30 text-muted-foreground"><tr><th className="px-3 py-2">Ejecución</th><th className="px-3 py-2">Modelo</th><th className="px-3 py-2">Prueba</th>{definitions.map((metric) => <th key={metric.key} className="px-3 py-2 text-right">{metric.label}</th>)}</tr></thead>
          <tbody>{rows.map((row, index) => {
            const values = new Map(row.metrics.map((metric) => [metric.key, metric]))
            const baseline = baselineRows.find((candidate) => candidate.model === row.model && candidate.scope === row.scope)
            const baselineValues = new Map((baseline?.metrics ?? []).map((metric) => [metric.key, metric]))
            return <tr key={`${row.run.id}-${row.model}-${row.scope}-${index}`} className="border-t border-border/60">
              <td className="px-3 py-2">{row.run.isBaseline && <Star size={11} className="mr-1 inline fill-amber-400 text-amber-400" />}{row.run.id.slice(0, 8)}<span className="block text-[10px] text-muted-foreground">{runFingerprint(row.run, row.model)}</span></td><td className="px-3 py-2 font-mono">{row.model}</td><td className="px-3 py-2 text-muted-foreground">{row.scope}</td>
              {definitions.map((definition) => { const metric = values.get(definition.key); const delta = metric && !row.run.isBaseline ? deltaLabel(metric, baselineValues.get(definition.key)) : null; return <td key={definition.key} className="px-3 py-2 text-right"><span>{metric ? formatMetric(metric) : "—"}</span>{delta && <span className="ml-1 text-[10px] text-muted-foreground">{delta}</span>}</td> })}
            </tr>
          })}</tbody>
        </table></div> : <p className="py-5 text-center text-sm text-muted-foreground">Selecciona al menos una ejecución.</p>}
      </div>
    </section>
  )
}
