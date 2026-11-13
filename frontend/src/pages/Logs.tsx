import { Fragment, useEffect, useMemo, useState } from "react"
import { ChevronDown, ChevronRight, Loader2, Search, X } from "lucide-react"
import { api } from "@/api/client"
import { useIsModelManager } from "@/hooks/useAuth"
import type { RecentRequest } from "@/types"
import { CacheCell, cacheInfo, fmtSpan, prefillSpeedText, requestPhases, TimingBreakdown } from "@/components/dashboard/RequestBreakdown"
import { fmtClock, fmtContext, fmtRealtime, fmtTps } from "@/components/dashboard/format"

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

function Field({ label, children, mono = false, wide = false }: { label: string; children: React.ReactNode; mono?: boolean; wide?: boolean }) {
  return (
    <div className={wide ? "sm:col-span-2 lg:col-span-3" : undefined}>
      <span className="text-muted-foreground">{label}: </span>
      <span className={mono ? "break-all font-mono" : undefined}>{children}</span>
    </div>
  )
}

function Group({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="space-y-1.5">
      <p className="text-[10.5px] font-semibold uppercase tracking-wide text-muted-foreground/80">{title}</p>
      <div className="grid gap-x-4 gap-y-1 sm:grid-cols-2 lg:grid-cols-3">{children}</div>
    </div>
  )
}

/** Detalle de una petición: qué se pidió y dónde corrió, quién la hizo (con IP
 *  y user agent, que el backend solo rellena para admins), cómo rindió y cómo
 *  acabó. Es el sitio para lo que no cabe en la tabla del dashboard. */
function ExpandedRow({ request, isManager }: { request: RecentRequest; isManager: boolean }) {
  const r = request
  const split = requestPhases(r)
  const cache = cacheInfo(r)
  const asked = r.requestedModel && r.requestedModel !== r.modelId ? r.requestedModel : null
  return (
    <tr className="bg-muted/10">
      <td colSpan={isManager ? 10 : 9} className="px-5 py-3">
        <div className="space-y-3 text-xs" data-testid="log-detail">
          <Group title="Petición">
            <Field label="Modelo" mono>{r.modelId}</Field>
            {asked && <Field label="Pedido como" mono>{asked}</Field>}
            {r.backendType && <Field label="Backend">{r.backendType}</Field>}
            {r.requestKind && <Field label="Tipo">{r.requestKind}</Field>}
            {r.endpointPath && <Field label="Endpoint" mono>{r.endpointPath}</Field>}
            {r.runtimePreset && <Field label="Peldaño">{r.runtimePreset}</Field>}
            {r.effectiveContext ? <Field label="Contexto cargado">{fmtContext(r.effectiveContext)}</Field> : null}
            {r.gpuIndex != null && <Field label="GPU">#{r.gpuIndex}</Field>}
          </Group>
          <Group title="Origen">
            {isManager && <Field label="Usuario">{r.username ?? "anónimo"}</Field>}
            {r.groupName && <Field label="Grupo">{r.groupName}</Field>}
            {r.apiKeyName && <Field label="API key" mono>{r.apiKeyName}</Field>}
            {r.clientAddr && <Field label="IP" mono>{r.clientAddr}</Field>}
            {r.userAgent && <Field label="User agent" mono wide>{r.userAgent}</Field>}
          </Group>
          {r.inFlight ? (
            <Group title="En curso">
              <Field label="Lleva">{fmtSpan(r.elapsedMs ?? null)}</Field>
              {r.queued ? <Field label="En cola">posición {r.queuePosition ?? "?"}{r.queueWaitEstimateMs ? ` · ~${fmtSpan(r.queueWaitEstimateMs)} de espera` : ""}</Field> : null}
              {r.estimatedRemainingMs != null && <Field label="Le queda">~{fmtSpan(r.estimatedRemainingMs)}</Field>}
            </Group>
          ) : (
          <Group title="Rendimiento">
            <Field label="Duración">{fmtSpan(r.durationMs)}</Field>
            <Field label="Tokens">
              {r.inputTokens != null || r.outputTokens != null ? `${r.inputTokens ?? 0} entrada → ${r.outputTokens ?? 0} salida` : "—"}
            </Field>
            <Field label="Cache de prompt">
              {!cache ? "el backend no lo informa" : cache.cached ? `${cache.cached} de ${cache.input} tokens (${cache.pct} %)` : "ningún token de la entrada"}
            </Field>
            {(r.promptTokensPerSecond || r.predictedTokensPerSecond) && (
              <Field label="Velocidad">
                prefill {prefillSpeedText(r, fmtTps) ?? "—"} · generación {fmtTps(r.predictedTokensPerSecond)}
              </Field>
            )}
            {r.audioSeconds && r.durationMs ? (
              <Field label="Audio">
                {fmtClock(r.audioSeconds)} transcritos a {fmtRealtime(r.audioSeconds / (r.durationMs / 1000))} tiempo real
              </Field>
            ) : null}
          </Group>
          )}
          {split && (
            <div className="space-y-1" data-testid="timing-detail">
              <div className="max-w-md">
                <TimingBreakdown r={r} />
              </div>
              <p className="text-muted-foreground">
                {split.phases.map((p) => `${p.label} ${fmtSpan(p.ms)}`).join(" · ")}
                {r.runtimeTelemetry?.ttft_ms != null && ` · primer token a los ${fmtSpan(r.runtimeTelemetry.ttft_ms)}`}
              </p>
            </div>
          )}
          <Group title="Resultado">
            <Field label="Estado">{r.inFlight ? "en curso" : (r.statusCode ?? "—")}</Field>
            <Field label="Inicio">
              <span title={r.startedAt}>{formatDate(r.startedAt)}</span>
            </Field>
            <Field label="ID" mono>{r.id}</Field>
          </Group>
          {r.error && (
            <div className="rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 text-red-200">
              <span className="font-medium">Error: </span>
              <span className="break-words">{r.error}</span>
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
        r.apiKeyName?.toLowerCase().includes(q) ||
        r.requestedModel?.toLowerCase().includes(q) ||
        r.groupName?.toLowerCase().includes(q) ||
        r.runtimePreset?.toLowerCase().includes(q) ||
        r.clientAddr?.toLowerCase().includes(q) ||
        r.userAgent?.toLowerCase().includes(q),
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
            placeholder="Buscar por modelo, endpoint, error, usuario, API key, IP…"
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
              <th className="px-3 py-2 text-right font-medium" title="Duración total con su desglose en prefill y generación">Tiempo</th>
              <th className="px-3 py-2 text-right font-medium hidden lg:table-cell">Tokens</th>
              <th className="px-3 py-2 text-right font-medium hidden md:table-cell" title="Tokens de entrada servidos desde la cache de prompt">Cache</th>
              {isManager && <th className="px-3 py-2 text-left font-medium hidden lg:table-cell">Usuario</th>}
              <th className="px-3 py-2 text-center font-medium">Estado</th>
            </tr>
          </thead>
          <tbody>
            {loading && (
              <tr>
                <td colSpan={isManager ? 10 : 9} className="px-3 py-8 text-center text-muted-foreground">
                  <div className="inline-flex items-center gap-2">
                    <div className="h-4 w-4 animate-spin rounded-full border-2 border-primary border-t-transparent" />
                    Cargando...
                  </div>
                </td>
              </tr>
            )}
            {!loading && filtered.length === 0 && (
              <tr>
                <td colSpan={isManager ? 10 : 9} className="px-3 py-8 text-center text-muted-foreground">
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
                    <td className="max-w-[200px] px-3 py-1.5 font-mono text-xs" title={r.modelId}>
                      <div className="truncate">{r.modelId}</div>
                      {r.requestedModel && r.requestedModel !== r.modelId && (
                        <div className="truncate text-[10.5px] text-sky-300/80" title="Nombre con el que lo pidió el cliente">
                          como {r.requestedModel}
                        </div>
                      )}
                    </td>
                    <td className="whitespace-nowrap px-3 py-1.5 text-xs text-muted-foreground hidden sm:table-cell">
                      {r.endpointPath ?? "—"}
                    </td>
                    <td className="whitespace-nowrap px-3 py-1.5 text-xs text-muted-foreground hidden md:table-cell">
                      {r.requestKind ?? r.backendType ?? "—"}
                    </td>
                    <td className="px-3 py-1.5 text-right text-xs text-muted-foreground">
                      {r.inFlight ? (
                        <span className="inline-flex items-center gap-1 text-sky-300" data-testid="log-inflight">
                          <Loader2 size={11} className="animate-spin" />
                          {r.queued ? "en cola" : "en curso"} · {fmtSpan(r.elapsedMs ?? null)}
                        </span>
                      ) : (
                        <TimingBreakdown r={r} compact />
                      )}
                    </td>
                    <td className="whitespace-nowrap px-3 py-1.5 text-right text-xs text-muted-foreground hidden lg:table-cell">
                      {r.inputTokens != null || r.outputTokens != null
                        ? `${r.inputTokens ?? 0} → ${r.outputTokens ?? 0}` : "—"}
                    </td>
                    <td className="whitespace-nowrap px-3 py-1.5 text-right text-xs hidden md:table-cell">
                      <CacheCell r={r} />
                    </td>
                    {isManager && (
                      <td className="whitespace-nowrap px-3 py-1.5 text-xs hidden lg:table-cell">
                        {r.username ?? <span className="italic text-muted-foreground/60">anónimo</span>}
                      </td>
                    )}
                    <td className="px-3 py-1.5 text-center">
                      {r.inFlight ? (
                        <span className="rounded border border-sky-500/40 bg-sky-500/10 px-1.5 py-0.5 text-[10px] text-sky-300">en curso</span>
                      ) : (
                        <StatusBadge code={r.statusCode} error={r.error} />
                      )}
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
