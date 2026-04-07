import { useEffect, useState } from "react"
import { api } from "@/api/client"
import { useIsModelManager } from "@/hooks/useAuth"
import type { RecentRequest } from "@/types"

function formatDate(iso: string): string {
  try {
    const d = new Date(iso)
    return `${d.toLocaleDateString("es-ES", { day: "2-digit", month: "2-digit" })} ${d.toLocaleTimeString("es-ES", { hour12: false })}`
  } catch {
    return "—"
  }
}

function StatusBadge({ code, error }: { code: number | null; error: string | null }) {
  if (error) {
    return (
      <span className="rounded border border-red-500/40 bg-red-500/10 px-1.5 py-0.5 text-[10px] text-red-300" title={error}>
        {code ?? "ERR"}
      </span>
    )
  }
  if (code && code >= 200 && code < 300) {
    return (
      <span className="rounded border border-emerald-500/40 bg-emerald-500/10 px-1.5 py-0.5 text-[10px] text-emerald-300">
        {code}
      </span>
    )
  }
  return (
    <span className="rounded border border-border bg-muted/30 px-1.5 py-0.5 text-[10px] text-muted-foreground">
      {code ?? "—"}
    </span>
  )
}

export function Logs() {
  const isManager = useIsModelManager()
  const [requests, setRequests] = useState<RecentRequest[]>([])
  const [loading, setLoading] = useState(true)
  const [limit, setLimit] = useState(100)
  const [autoRefresh, setAutoRefresh] = useState(true)

  const fetchLogs = async () => {
    try {
      const data = isManager
        ? await api.stats.recent(limit)
        : await api.stats.myRecent(limit)
      setRequests(data.requests ?? [])
    } catch {
      // silently fail
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchLogs()
  }, [limit])

  useEffect(() => {
    if (!autoRefresh) return
    const id = setInterval(fetchLogs, 10_000)
    return () => clearInterval(id)
  }, [autoRefresh, limit])

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold">Logs</h1>
        <div className="flex items-center gap-3">
          <label className="flex items-center gap-2 text-xs text-muted-foreground">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
              className="h-3.5 w-3.5 rounded border-border"
            />
            Auto-refresh
          </label>
          <select
            value={limit}
            onChange={(e) => setLimit(Number(e.target.value))}
            className="rounded-md border border-border bg-background px-2 py-1 text-xs"
          >
            <option value={50}>50</option>
            <option value={100}>100</option>
            <option value={200}>200</option>
          </select>
          <button
            onClick={fetchLogs}
            className="rounded-md border border-border px-2.5 py-1 text-xs hover:bg-muted"
          >
            Refrescar
          </button>
        </div>
      </div>

      <div className="overflow-x-auto rounded-lg border border-border">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border bg-muted/30 text-xs text-muted-foreground">
              <th className="px-3 py-2 text-left font-medium">Hora</th>
              <th className="px-3 py-2 text-left font-medium">Modelo</th>
              <th className="px-3 py-2 text-left font-medium">Endpoint</th>
              <th className="px-3 py-2 text-left font-medium">Tipo</th>
              <th className="px-3 py-2 text-right font-medium">Duracion</th>
              <th className="px-3 py-2 text-right font-medium">Tokens</th>
              {isManager && <th className="px-3 py-2 text-left font-medium">Usuario</th>}
              <th className="px-3 py-2 text-left font-medium">API Key</th>
              <th className="px-3 py-2 text-center font-medium">Estado</th>
            </tr>
          </thead>
          <tbody>
            {loading && (
              <tr>
                <td colSpan={isManager ? 9 : 8} className="px-3 py-8 text-center text-muted-foreground">
                  Cargando...
                </td>
              </tr>
            )}
            {!loading && requests.length === 0 && (
              <tr>
                <td colSpan={isManager ? 9 : 8} className="px-3 py-8 text-center text-muted-foreground">
                  Sin peticiones recientes
                </td>
              </tr>
            )}
            {requests.map((r) => (
              <tr key={r.id} className="border-b border-border/30 hover:bg-muted/10">
                <td className="whitespace-nowrap px-3 py-1.5 text-xs text-muted-foreground" title={r.startedAt}>
                  {formatDate(r.startedAt)}
                </td>
                <td className="max-w-[180px] truncate px-3 py-1.5 font-mono text-xs" title={r.modelId}>
                  {r.modelId}
                </td>
                <td className="whitespace-nowrap px-3 py-1.5 text-xs text-muted-foreground">
                  {r.endpointPath ?? "—"}
                </td>
                <td className="whitespace-nowrap px-3 py-1.5 text-xs text-muted-foreground">
                  {r.requestKind ?? r.backendType ?? "—"}
                </td>
                <td className="whitespace-nowrap px-3 py-1.5 text-right text-xs">
                  {r.durationMs != null ? `${r.durationMs} ms` : "—"}
                </td>
                <td className="whitespace-nowrap px-3 py-1.5 text-right text-xs text-muted-foreground">
                  {r.inputTokens != null || r.outputTokens != null
                    ? `${r.inputTokens ?? 0} / ${r.outputTokens ?? 0}`
                    : "—"}
                </td>
                {isManager && (
                  <td className="whitespace-nowrap px-3 py-1.5 text-xs">
                    {r.username ?? "—"}
                  </td>
                )}
                <td className="whitespace-nowrap px-3 py-1.5 text-xs text-muted-foreground">
                  {r.apiKeyName ?? "—"}
                </td>
                <td className="px-3 py-1.5 text-center">
                  <StatusBadge code={r.statusCode} error={r.error} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
