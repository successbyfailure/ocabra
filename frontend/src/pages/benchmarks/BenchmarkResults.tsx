import { Check, Download, Trophy } from "lucide-react"
import type { BenchmarkRun } from "@/types"
import {
  bestMetricValues,
  formatMetric,
  parseBenchmarkResults,
} from "./resultAdapters"

function downloadJson(run: BenchmarkRun) {
  const blob = new Blob([JSON.stringify(run.result, null, 2)], { type: "application/json" })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement("a")
  anchor.href = url
  anchor.download = `benchmark-${run.app}-${run.suite}-${run.id}.json`
  anchor.click()
  URL.revokeObjectURL(url)
}

export function BenchmarkResults({ run }: { run: BenchmarkRun }) {
  const rows = parseBenchmarkResults(run.result)
  const metricDefinitions = Array.from(
    new Map(rows.flatMap((row) => row.metrics.map((item) => [item.key, item]))).values(),
  )
  const best = bestMetricValues(rows)

  return (
    <section className="mt-4 overflow-hidden rounded-lg border border-border">
      <div className="flex items-center justify-between gap-3 border-b border-border bg-muted/30 px-3 py-2">
        <div>
          <h4 className="text-sm font-medium">Resultados</h4>
          <p className="text-xs text-muted-foreground">
            {rows.length} resultado{rows.length === 1 ? "" : "s"} aceptado{rows.length === 1 ? "" : "s"}
          </p>
        </div>
        <button
          type="button"
          onClick={() => downloadJson(run)}
          className="inline-flex items-center gap-1 rounded-md border border-border px-2.5 py-1.5 text-xs hover:bg-muted"
        >
          <Download size={13} /> JSON
        </button>
      </div>

      {metricDefinitions.length > 0 ? (
        <div className="overflow-x-auto">
          <table className="w-full min-w-max text-left text-xs">
            <thead className="bg-muted/20 text-muted-foreground">
              <tr>
                <th className="px-3 py-2">Modelo</th>
                <th className="px-3 py-2">Prueba</th>
                <th className="px-3 py-2">Estado</th>
                {metricDefinitions.map((item) => (
                  <th key={item.key} className="px-3 py-2 text-right font-medium">{item.label}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, rowIndex) => {
                const byKey = new Map(row.metrics.map((item) => [item.key, item]))
                return (
                  <tr key={`${row.model}-${row.scope}-${rowIndex}`} className="border-t border-border/70">
                    <td className="px-3 py-2 font-mono font-medium">{row.model}</td>
                    <td className="px-3 py-2 text-muted-foreground">{row.scope}</td>
                    <td className="px-3 py-2">
                      {row.status ? (
                        <span className="inline-flex items-center gap-1 text-emerald-400">
                          <Check size={12} /> {row.status}
                        </span>
                      ) : "—"}
                    </td>
                    {metricDefinitions.map((definition) => {
                      const item = byKey.get(definition.key)
                      const isBest = item
                        && typeof item.value === "number"
                        && best.get(item.key) === item.value
                      return (
                        <td
                          key={definition.key}
                          className={`px-3 py-2 text-right tabular-nums ${isBest ? "font-semibold text-emerald-400" : ""}`}
                        >
                          <span className="inline-flex items-center gap-1">
                            {isBest && <Trophy size={11} />}
                            {item ? formatMetric(item) : "—"}
                          </span>
                        </td>
                      )
                    })}
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="px-3 py-5 text-sm text-muted-foreground">
          El runner devolvió datos, pero no hay métricas escalares para tabular.
        </div>
      )}

      <details className="border-t border-border px-3 py-2">
        <summary className="cursor-pointer text-xs text-muted-foreground">Resultado bruto</summary>
        <pre className="mt-2 max-h-96 overflow-auto rounded-md bg-black/60 p-3 text-[11px] text-zinc-300">
          {JSON.stringify(run.result, null, 2)}
        </pre>
      </details>
    </section>
  )
}
