import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"
import type { TokenStats } from "@/types"

interface TokensChartProps {
  data: TokenStats
}

export function TokensChart({ data }: TokensChartProps) {
  const rows = data.series.map((point) => ({
    timestamp: new Date(point.timestamp).toLocaleTimeString(),
    input: point.inputTokens,
    output: point.outputTokens,
  }))

  return (
    <div className="h-72 rounded-lg border border-border bg-card p-3">
      <h3 className="mb-2 text-sm font-semibold text-muted-foreground">Tokens entrada/salida</h3>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={rows}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis dataKey="timestamp" hide />
          <YAxis stroke="#94a3b8" />
          <Tooltip />
          <Legend />
          <Bar dataKey="input" stackId="tokens" fill="#0ea5e9" />
          <Bar dataKey="output" stackId="tokens" fill="#22c55e" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
