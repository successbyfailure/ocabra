import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"
import type { BatchVsInteractiveStats } from "@/types"

interface BatchVsInteractiveChartProps {
  data: BatchVsInteractiveStats
}

// Dark-mode-validated slots from the project's categorical palette
// (blue=slot1, orange=slot2) — see dataviz skill's palette.md. Passes CVD
// separation (ΔE 26.8 protan / 32.4 tritan), normal-vision floor (ΔE 31.8)
// and ≥3:1 contrast against this app's dark card surface (#050b1e).
const COLOR_INTERACTIVE = "#3987e5"
const COLOR_BATCH = "#d95926"

export function BatchVsInteractiveChart({ data }: BatchVsInteractiveChartProps) {
  const rows = data.dailySeries.map((point) => ({
    date: new Date(point.date).toLocaleDateString(),
    interactive: point.interactive,
    batch: point.batch,
  }))

  return (
    <div className="h-72 rounded-lg border border-border bg-card p-3">
      <h3 className="mb-2 text-sm font-semibold text-muted-foreground">Peticiones: lotes vs. interactivas</h3>
      <ResponsiveContainer width="100%" height="75%">
        <BarChart data={rows}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis dataKey="date" stroke="#94a3b8" tick={{ fontSize: 11 }} />
          <YAxis stroke="#94a3b8" />
          <Tooltip />
          <Legend />
          <Bar dataKey="interactive" name="Interactivas" fill={COLOR_INTERACTIVE} radius={[2, 2, 0, 0]} />
          <Bar dataKey="batch" name="Lotes" fill={COLOR_BATCH} radius={[2, 2, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
      <div className="mt-1 grid grid-cols-2 gap-3 text-xs text-muted-foreground">
        <div>
          <span className="font-medium" style={{ color: COLOR_INTERACTIVE }}>Interactivas</span>:{" "}
          {data.interactive.totalRequests.toLocaleString()} peticiones, {data.interactive.totalErrors} errores,{" "}
          {data.interactive.avgDurationMs}ms avg, ${data.interactive.estimatedCostUsd}
        </div>
        <div>
          <span className="font-medium" style={{ color: COLOR_BATCH }}>Lotes</span>:{" "}
          {data.batch.totalRequests.toLocaleString()} peticiones, {data.batch.totalErrors} errores,{" "}
          {data.batch.avgDurationMs}ms avg, ${data.batch.estimatedCostUsd}
        </div>
      </div>
    </div>
  )
}
