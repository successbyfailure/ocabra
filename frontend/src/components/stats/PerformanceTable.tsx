import { useMemo, useState } from "react"
import type { PerformanceStats } from "@/types"

type SortKey = "modelId" | "totalRequests" | "avgLatencyMs" | "tokensPerSecond" | "errorCount" | "uptimePct"

interface PerformanceTableProps {
  data: PerformanceStats
}

function toCsv(data: PerformanceStats): string {
  const header = "model_id,total_requests,avg_latency_ms,tokens_per_second,error_count,uptime_pct"
  const rows = data.byModel.map((row) =>
    [row.modelId, row.totalRequests, row.avgLatencyMs, row.tokensPerSecond, row.errorCount, row.uptimePct].join(","),
  )
  return [header, ...rows].join("\n")
}

export function PerformanceTable({ data }: PerformanceTableProps) {
  const [sortKey, setSortKey] = useState<SortKey>("totalRequests")
  const [descending, setDescending] = useState(true)

  const rows = useMemo(() => {
    const copy = [...data.byModel]
    copy.sort((a, b) => {
      const left = a[sortKey]
      const right = b[sortKey]
      if (typeof left === "string" && typeof right === "string") {
        return descending ? right.localeCompare(left) : left.localeCompare(right)
      }
      return descending ? Number(right) - Number(left) : Number(left) - Number(right)
    })
    return copy
  }, [data.byModel, descending, sortKey])

  const setSort = (next: SortKey) => {
    if (sortKey === next) {
      setDescending((prev) => !prev)
      return
    }
    setSortKey(next)
    setDescending(true)
  }

  const downloadCsv = () => {
    const blob = new Blob([toCsv(data)], { type: "text/csv;charset=utf-8" })
    const url = URL.createObjectURL(blob)
    const link = document.createElement("a")
    link.href = url
    link.download = "performance.csv"
    link.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="rounded-lg border border-border bg-card p-3">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-muted-foreground">Performance por modelo</h3>
        <button
          type="button"
          onClick={downloadCsv}
          className="rounded-md border border-border px-3 py-1 text-xs hover:bg-muted"
        >
          Export CSV
        </button>
      </div>

      <div className="overflow-x-auto">
        <table className="min-w-full text-left text-sm">
          <thead className="text-xs uppercase text-muted-foreground">
            <tr>
              <th className="px-2 py-2"><button type="button" onClick={() => setSort("modelId")}>Modelo</button></th>
              <th className="px-2 py-2"><button type="button" onClick={() => setSort("totalRequests")}>Requests</button></th>
              <th className="px-2 py-2"><button type="button" onClick={() => setSort("avgLatencyMs")}>Avg latency</button></th>
              <th className="px-2 py-2"><button type="button" onClick={() => setSort("tokensPerSecond")}>Tokens/s</button></th>
              <th className="px-2 py-2"><button type="button" onClick={() => setSort("errorCount")}>Errores</button></th>
              <th className="px-2 py-2"><button type="button" onClick={() => setSort("uptimePct")}>Uptime %</button></th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.modelId} className="border-t border-border/60">
                <td className="px-2 py-2">{row.modelId}</td>
                <td className="px-2 py-2">{row.totalRequests}</td>
                <td className="px-2 py-2">{row.avgLatencyMs.toFixed(1)} ms</td>
                <td className="px-2 py-2">{row.tokensPerSecond.toFixed(1)}</td>
                <td className="px-2 py-2">{row.errorCount}</td>
                <td className="px-2 py-2">{row.uptimePct.toFixed(1)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
