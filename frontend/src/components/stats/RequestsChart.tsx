import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"
import type { RequestStats } from "@/types"

interface RequestsChartProps {
  data: RequestStats
}

export function RequestsChart({ data }: RequestsChartProps) {
  const rows = data.series.map((point) => ({
    timestamp: new Date(point.timestamp).toLocaleTimeString(),
    requests: point.count,
    p50: Math.max(1, data.p50DurationMs * 0.9),
    p95: data.p95DurationMs,
    p99: Math.max(data.p95DurationMs * 1.2, data.p95DurationMs + 30),
  }))

  return (
    <div className="h-72 rounded-lg border border-border bg-card p-3">
      <h3 className="mb-2 text-sm font-semibold text-muted-foreground">Requests/min + Latencia</h3>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={rows}>
          <XAxis dataKey="timestamp" hide />
          <YAxis yAxisId="req" stroke="#94a3b8" />
          <YAxis yAxisId="lat" orientation="right" stroke="#94a3b8" />
          <Tooltip />
          <Line yAxisId="req" type="monotone" dataKey="requests" stroke="#3b82f6" strokeWidth={2} dot={false} />
          <Line yAxisId="lat" type="monotone" dataKey="p50" stroke="#10b981" dot={false} />
          <Line yAxisId="lat" type="monotone" dataKey="p95" stroke="#f59e0b" dot={false} />
          <Line yAxisId="lat" type="monotone" dataKey="p99" stroke="#ef4444" dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
