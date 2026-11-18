import { useMemo, useState } from "react"
import type { PerformanceStats, WorkSpeed } from "@/types"

type SortKey =
  | "modelId"
  | "backendType"
  | "totalRequests"
  | "avgLatencyMs"
  | "p95LatencyMs"
  | "avgLoadMs"
  | "lastLoadMs"
  | "requestsPerMinute"
  | "tokensPerSecond"
  | "errorCount"
  | "uptimePct"

interface PerformanceTableProps {
  data: PerformanceStats
}

function toCsv(data: PerformanceStats): string {
  const header =
    "model_id,backend_type,request_kinds,total_requests,avg_latency_ms,p95_latency_ms,avg_load_ms,p95_load_ms,last_load_ms,load_count,requests_per_minute,tokens_per_second,input_tokens,output_tokens,tokenized_requests,error_count,uptime_pct"
  const rows = data.byModel.map((row) =>
    [
      row.modelId,
      row.backendType,
      row.requestKinds.join("|"),
      row.totalRequests,
      row.avgLatencyMs,
      row.p95LatencyMs,
      row.avgLoadMs,
      row.p95LoadMs,
      row.lastLoadMs,
      row.loadCount,
      row.requestsPerMinute,
      row.tokensPerSecond,
      row.totalInputTokens,
      row.totalOutputTokens,
      row.tokenizedRequests,
      row.errorCount,
      row.uptimePct,
    ].join(","),
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
              <th className="px-2 py-2"><button type="button" onClick={() => setSort("backendType")}>Backend</button></th>
              <th className="px-2 py-2">Tipos</th>
              <th className="px-2 py-2"><button type="button" onClick={() => setSort("totalRequests")}>Requests</button></th>
              <th className="px-2 py-2"><button type="button" onClick={() => setSort("avgLatencyMs")}>Avg ms</button></th>
              <th className="px-2 py-2"><button type="button" onClick={() => setSort("p95LatencyMs")}>P95 ms</button></th>
              <th className="px-2 py-2"><button type="button" onClick={() => setSort("avgLoadMs")}>Load avg</button></th>
              <th className="px-2 py-2"><button type="button" onClick={() => setSort("lastLoadMs")}>Load last</button></th>
              <th className="px-2 py-2"><button type="button" onClick={() => setSort("requestsPerMinute")}>Req/min</button></th>
              <th className="px-2 py-2"><button type="button" onClick={() => setSort("tokensPerSecond")}>Tokens/s</button></th>
              <th className="px-2 py-2" title="Para lo que no genera texto: audio transcrito o hablado por segundo, caracteres, textos, documentos o imágenes">Velocidad</th>
              <th className="px-2 py-2">Tok in/out</th>
              <th className="px-2 py-2"><button type="button" onClick={() => setSort("errorCount")}>Err</button></th>
              <th className="px-2 py-2"><button type="button" onClick={() => setSort("uptimePct")}>Uptime</button></th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.modelId} className="border-t border-border/60">
                <td className="px-2 py-2">{row.modelId}</td>
                <td className="px-2 py-2">{row.backendType}</td>
                <td className="px-2 py-2">{row.requestKinds.join(", ") || "-"}</td>
                <td className="px-2 py-2">{row.totalRequests}</td>
                <td className="px-2 py-2">{row.avgLatencyMs.toFixed(1)}</td>
                <td className="px-2 py-2">{row.p95LatencyMs.toFixed(1)}</td>
                <td className="px-2 py-2">{row.loadCount > 0 ? row.avgLoadMs.toFixed(0) : "-"}</td>
                <td className="px-2 py-2">{row.loadCount > 0 ? row.lastLoadMs.toFixed(0) : "-"}</td>
                <td className="px-2 py-2">{row.requestsPerMinute.toFixed(2)}</td>
                <td className="px-2 py-2">{row.tokensPerSecond.toFixed(2)}</td>
                <td className="px-2 py-2" data-testid="work-speed">
                  {row.workSpeeds?.length
                    ? row.workSpeeds.map((w) => (
                        <span key={w.unit + w.kinds.join()} className="mr-2 whitespace-nowrap" title={`${w.kinds.join(", ")} · ${w.samples} peticiones`}>
                          {fmtWorkSpeed(w)}
                        </span>
                      ))
                    : "-"}
                </td>
                <td className="px-2 py-2">{row.totalInputTokens}/{row.totalOutputTokens}</td>
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

const WORK_UNIT: Record<string, (v: number) => string> = {
  x_realtime: (v) => `×${v.toFixed(1)} tiempo real`,
  chars_per_s: (v) => `${v.toFixed(0)} car/s`,
  inputs_per_s: (v) => `${v.toFixed(1)} textos/s`,
  tokens_per_s: (v) => `${v.toFixed(0)} tok/s`,
  documents_per_s: (v) => `${v.toFixed(1)} docs/s`,
  s_per_image: (v) => `${v.toFixed(1)} s/imagen`,
  frames_per_s: (v) => `${v.toFixed(1)} fps`,
}

export function fmtWorkSpeed(w: WorkSpeed): string {
  return (WORK_UNIT[w.unit] ?? ((v: number) => `${v} ${w.unit}`))(w.value)
}
