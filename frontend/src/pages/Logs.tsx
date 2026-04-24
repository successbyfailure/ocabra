import { Fragment, useEffect, useMemo, useState } from "react"
import { ChevronDown, ChevronRight, Search, X } from "lucide-react"
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

function ExpandedRow({ request, isManager }: { request: RecentRequest; isManager: boolean }) {
  return (
    <tr className="bg-muted/10">
      <td colSpan={isManager ? 9 : 8} className="px-5 py-3">
        <div className="space-y-2 text-xs">
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
            <div>
              <span className="text-muted-foreground">Modelo: </span>
              <span className="font-mono">{request.modelId}</span>
            </div>
            <div>
              <span className="text-muted-foreground">Endpoint: </span>
              <span className="font-mono">{request.endpointPath ?? "—"}</span>
            </div>
            <div>
              <span className="text-muted-foreground">Tipo: </span>
              <span>{request.requestKind ?? request.backendType ?? "—"}</span>
            </div>
            <div>
              <span className="text-muted-foreground">Duracion: </span>
              <span>{request.durationMs != null ? `${request.durationMs} ms` : "—"}</span>
            </div>
            <div>
              <span className="text-muted-foreground">Tokens: </span>
              <span>
                {request.inputTokens != null || request.outputTokens != null
                  ? `${request.inputTokens ?? 0} in / ${request.outputTokens ?? 0} out`
                  : "—"}
              </span>
            </div>
            {isManager && (
              <div>
                <span className="text-muted-foreground">Usuario: </span>
                <span>{request.username ?? "—"}</span>
              </div>
            )}
            <div>
              <span className="text-muted-foreground">API Key: </span>
              <span>{request.apiKeyName ?? "—"}</span>
            </div>
            <div>
              <span className="text-muted-foreground">Status: </span>
              <span>{request.statusCode ?? "—"}</span>
            </div>
            <div>
              <span className="text-muted-foreground">Inicio: </span>
              <span className="font-mono">{request.startedAt}</span>
            </div>
          </div>
          {request.error && (
            <div className="rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 text-red-200">
              <span className="font-medium">Error: </span>
              {request.error}
            </div>
          )}
        </div>
      </td>
    </tr>
  )
}

export function Logs() {
  const isManager = useIsModelManager()
  const [requests, setRequests] = useState<RecentRequest[]>([])
  const [loading, setLoading] = useState(true)
  const [limit, setLimit] = useState(100)
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [searchQuery, setSearchQuery] = useState("")
  const [expandedId, setExpandedId] = useState<string | null>(null)

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

  const filtered = useMemo(() => {
    if (!searchQuery.trim()) return requests
    const q = searchQuery.toLowerCase()
    return requests.filter(
      (r) =>
        r.modelId?.toLowerCase().includes(q) ||
        r.endpointPath?.toLowerCase().includes(q) ||
        r.error?.toLowerCase().includes(q) ||
        r.username?.toLowerCase().includes(q) ||
        r.requestKind?.toLowerCase().includes(q) ||
        r.apiKeyName?.toLowerCase().includes(q),
    )
  }, [requests, searchQuery])

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-semibold">Logs</h1>
        <p className="text-sm text-muted-foreground">Historial de peticiones al servidor.</p>
      </div>

      {/* Filter bar */}
      <div className="flex flex-wrap items-center gap-3 rounded-lg border border-border bg-card p-3">
        <div className="relative flex-1 min-w-[200px]">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Buscar por modelo, endpoint, error, usuario..."
            className="w-full rounded-md border border-border bg-background pl-9 pr-8 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
          />
          {searchQuery && (
            <button type="button" onClick={() => setSearchQuery("")}
              className="absolute right-2.5 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground">
              <X size={12} />
            </button>
          )}
        </div>

        <label className="flex items-center gap-2 text-xs text-muted-foreground">
          <input type="checkbox" checked={autoRefresh} onChange={(e) => setAutoRefresh(e.target.checked)}
            className="h-3.5 w-3.5 rounded border-border accent-primary" />
          Auto-refresh
        </label>

        <select value={limit} onChange={(e) => setLimit(Number(e.target.value))}
          className="rounded-md border border-border bg-background px-2 py-1.5 text-xs focus:outline-none focus:ring-2 focus:ring-primary/50">
          <option value={50}>50</option>
          <option value={100}>100</option>
          <option value={200}>200</option>
        </select>

        <button onClick={fetchLogs}
          className="rounded-md border border-border px-2.5 py-1.5 text-xs hover:bg-muted">
          Refrescar
        </button>

        {searchQuery && (
          <span className="text-xs text-muted-foreground">
            {filtered.length} de {requests.length} resultados
          </span>
        )}
      </div>

      <div className="overflow-x-auto rounded-lg border border-border">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border bg-muted/30 text-xs text-muted-foreground">
              <th className="w-6 px-1 py-2" />
              <th className="px-3 py-2 text-left font-medium">Hora</th>
              <th className="px-3 py-2 text-left font-medium">Modelo</th>
              <th className="px-3 py-2 text-left font-medium hidden sm:table-cell">Endpoint</th>
              <th className="px-3 py-2 text-left font-medium hidden md:table-cell">Tipo</th>
              <th className="px-3 py-2 text-right font-medium">Duracion</th>
              <th className="px-3 py-2 text-right font-medium hidden lg:table-cell">Tokens</th>
              {isManager && <th className="px-3 py-2 text-left font-medium hidden lg:table-cell">Usuario</th>}
              <th className="px-3 py-2 text-center font-medium">Estado</th>
            </tr>
          </thead>
          <tbody>
            {loading && (
              <tr>
                <td colSpan={isManager ? 9 : 8} className="px-3 py-8 text-center text-muted-foreground">
                  <div className="inline-flex items-center gap-2">
                    <div className="h-4 w-4 animate-spin rounded-full border-2 border-primary border-t-transparent" />
                    Cargando...
                  </div>
                </td>
              </tr>
            )}
            {!loading && filtered.length === 0 && (
              <tr>
                <td colSpan={isManager ? 9 : 8} className="px-3 py-8 text-center text-muted-foreground">
                  {searchQuery ? "Sin resultados para la busqueda" : "Sin peticiones recientes"}
                </td>
              </tr>
            )}
            {filtered.map((r) => {
              const isExpanded = expandedId === r.id
              const isError = Boolean(r.error) || (r.statusCode != null && r.statusCode >= 500)
              return (
                <Fragment key={r.id}>
                  <tr
                    className={`border-b border-border/30 cursor-pointer transition-colors hover:bg-muted/10${isError ? " bg-red-500/5" : ""}${isExpanded ? " bg-muted/20" : ""}`}
                    onClick={() => setExpandedId(isExpanded ? null : r.id)}
                  >
                    <td className="px-1 py-1.5 text-center text-muted-foreground">
                      {isExpanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
                    </td>
                    <td className="whitespace-nowrap px-3 py-1.5 text-xs text-muted-foreground" title={r.startedAt}>
                      {formatDate(r.startedAt)}
                    </td>
                    <td className="max-w-[180px] truncate px-3 py-1.5 font-mono text-xs" title={r.modelId}>
                      {r.modelId}
                    </td>
                    <td className="whitespace-nowrap px-3 py-1.5 text-xs text-muted-foreground hidden sm:table-cell">
                      {r.endpointPath ?? "—"}
                    </td>
                    <td className="whitespace-nowrap px-3 py-1.5 text-xs text-muted-foreground hidden md:table-cell">
                      {r.requestKind ?? r.backendType ?? "—"}
                    </td>
                    <td className="whitespace-nowrap px-3 py-1.5 text-right text-xs">
                      {r.durationMs != null ? `${r.durationMs} ms` : "—"}
                    </td>
                    <td className="whitespace-nowrap px-3 py-1.5 text-right text-xs text-muted-foreground hidden lg:table-cell">
                      {r.inputTokens != null || r.outputTokens != null
                        ? `${r.inputTokens ?? 0} / ${r.outputTokens ?? 0}` : "—"}
                    </td>
                    {isManager && (
                      <td className="whitespace-nowrap px-3 py-1.5 text-xs hidden lg:table-cell">
                        {r.username ?? "—"}
                      </td>
                    )}
                    <td className="px-3 py-1.5 text-center">
                      <StatusBadge code={r.statusCode} error={r.error} />
                    </td>
                  </tr>
                  {isExpanded && <ExpandedRow request={r} isManager={isManager} />}
                </Fragment>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
