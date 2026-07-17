import { useEffect, useMemo, useState } from "react"
import { MetricGauge } from "@/components/gpu/MetricGauge"
import { MemoryBars } from "@/components/gpu/MemoryBars"
import { PowerBlock } from "@/components/gpu/PowerBlock"
import { METRIC, tempColor } from "@/components/gpu/metrics"
import {
  Activity,
  Boxes,
  ChevronDown,
  ChevronRight,
  Cpu,
  Database,
  Download,
  Globe,
  Layers,
  MemoryStick,
  Network,
  Sparkles,
  Zap,
} from "lucide-react"
import { agentStatsApi } from "@/api/agents"
import type { AgentStatRow } from "@/types/agents"
import { api } from "@/api/client"
import { EmptyState } from "@/components/common/EmptyState"
import { GpuCard } from "@/components/gpu/GpuCard"
import { LoadPolicyBadge } from "@/components/models/LoadPolicyBadge"
import { ModelStatusBadge } from "@/components/models/ModelStatusBadge"
import { useIsModelManager } from "@/hooks/useAuth"
import { useWebSocket } from "@/hooks/useWebSocket"
import { useDownloadStore } from "@/stores/downloadStore"
import { useGpuStore } from "@/stores/gpuStore"
import { useModelStore } from "@/stores/modelStore"
import { useServiceStore } from "@/stores/serviceStore"
import type { FederationPeer, HostStats, RecentRequestsData, ServerPower, ServiceState } from "@/types"

function formatLoadedAgo(totalSeconds: number): string {
  if (totalSeconds < 60) return `${totalSeconds}s`
  const minutes = Math.floor(totalSeconds / 60)
  if (minutes < 60) return `${minutes}m`
  const hours = Math.floor(minutes / 60)
  const remMinutes = minutes % 60
  if (hours < 24) return remMinutes === 0 ? `${hours}h` : `${hours}h ${remMinutes}m`
  const days = Math.floor(hours / 24)
  const remHours = hours % 24
  return remHours === 0 ? `${days}d` : `${days}d ${remHours}h`
}

function loadedMeta(loadedAt: string | null, nowMs: number): { at: string; ago: string } | null {
  if (!loadedAt) return null
  const loadedTs = Date.parse(loadedAt)
  if (Number.isNaN(loadedTs)) return null
  const seconds = Math.max(0, Math.floor((nowMs - loadedTs) / 1000))
  const at = new Date(loadedTs).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  })
  return { at, ago: formatLoadedAgo(seconds) }
}

/* ─── KPI Card ─── */
function KpiCard({ icon: Icon, label, value, sub, color }: {
  icon: React.ElementType
  label: string
  value: string | number
  sub?: string
  color: string
}) {
  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="flex items-center gap-3">
        <div className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-lg ${color}`}>
          <Icon size={18} />
        </div>
        <div className="min-w-0">
          <p className="text-xs text-muted-foreground">{label}</p>
          <p className="text-xl font-semibold">{value}</p>
          {sub && <p className="text-xs text-muted-foreground">{sub}</p>}
        </div>
      </div>
    </div>
  )
}

/* ─── Collapsible Section ─── */
function Section({ title, defaultOpen = true, children, badge }: {
  title: string
  defaultOpen?: boolean
  children: React.ReactNode
  badge?: React.ReactNode
}) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <section className="space-y-3">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="flex w-full items-center gap-2 text-left"
      >
        {open ? <ChevronDown size={16} className="text-muted-foreground" /> : <ChevronRight size={16} className="text-muted-foreground" />}
        <h2 className="text-lg font-semibold">{title}</h2>
        {badge}
      </button>
      {open && children}
    </section>
  )
}

/* ─── Host Stats ─── */
interface HostHistoryPoint {
  t: number
  cpuPct: number
  memPct: number
  tempC: number | null
  powerW: number | null
}

function HostStatsCard({
  stats,
  serverPower,
  history = [],
}: {
  stats: HostStats
  serverPower?: ServerPower | null
  history?: HostHistoryPoint[]
}) {
  const gb = (mb: number) => (mb / 1024).toFixed(1)
  const cpuTemp = serverPower?.cpuTempC
  const cpuPower = serverPower?.cpuPowerW
  const powerHistory = history.map((p) => p.powerW ?? 0)

  return (
    <article className="rounded-2xl border border-border bg-card p-[18px] shadow-sm">
      <div className="mb-4 flex items-start justify-between gap-2.5">
        <div className="min-w-0">
          <p className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">Host</p>
          <h3 className="truncate text-[15.5px] font-semibold tracking-tight text-foreground">
            Threadripper · {stats.cpuCountPhysical}C/{stats.cpuCount}T
          </h3>
        </div>
        {cpuTemp != null && (
          <span className="inline-flex shrink-0 items-center gap-1.5 rounded-full border border-border bg-muted/50 px-2.5 py-1 text-[12.5px] font-semibold tabular-nums">
            <span className="h-[7px] w-[7px] rounded-full" style={{ background: tempColor(cpuTemp) }} />
            {cpuTemp.toFixed(0)}°C
          </span>
        )}
      </div>

      <div className="grid grid-cols-[auto_1fr] items-center gap-[18px]">
        <MetricGauge
          pct={stats.cpuPct}
          color={METRIC.util}
          label="Carga CPU"
          value={stats.cpuPct.toFixed(0)}
          unit="%"
          caption="Carga CPU"
        />
        <MemoryBars
          name="RAM"
          usedLabel={gb(stats.memUsedMb)}
          totalLabel={gb(stats.memTotalMb)}
          usedPct={stats.memPct}
          secondaryName="swap"
          secondaryPct={stats.swapTotalMb > 0 ? stats.swapPct : 0}
          secondaryMode="aside"
        />
      </div>

      {cpuPower != null ? (
        <PowerBlock
          label="Consumo CPU"
          powerW={cpuPower}
          history={powerHistory}
          subtitle={`load ${stats.loadAvg1m} / ${stats.loadAvg5m} / ${stats.loadAvg15m} · últimos 10 min`}
        />
      ) : (
        <div className="mt-[18px] border-t border-border pt-3 text-[11px] text-muted-foreground">
          load {stats.loadAvg1m} / {stats.loadAvg5m} / {stats.loadAvg15m}
        </div>
      )}

      <details className="group mt-[15px] border-t border-border">
        <summary className="flex cursor-pointer list-none items-center justify-between pt-3 text-[11.5px] font-medium text-muted-foreground [&::-webkit-details-marker]:hidden">
          <span>Detalles</span>
          <span className="text-muted-foreground/70 transition-transform group-open:rotate-180">▾</span>
        </summary>
        <div className="flex flex-col gap-1.5 pb-0.5 pt-2.5 text-[11.5px]">
          <div className="flex items-center justify-between rounded-lg bg-muted/50 px-2.5 py-1.5">
            <span className="text-muted-foreground">Swap</span>
            <span className="tabular-nums">{gb(stats.swapUsedMb)} / {gb(stats.swapTotalMb)} GB</span>
          </div>
          <div className="flex items-center justify-between rounded-lg bg-muted/50 px-2.5 py-1.5">
            <span className="text-muted-foreground">Load (1/5/15m)</span>
            <span className="tabular-nums">{stats.loadAvg1m} / {stats.loadAvg5m} / {stats.loadAvg15m}</span>
          </div>
        </div>
      </details>
    </article>
  )
}

/* ─── Service Card ─── */
function ServiceCard({ service }: { service: ServiceState }) {
  const unloadService = useServiceStore((s) => s.unloadService)
  const startService = useServiceStore((s) => s.startService)
  const refreshService = useServiceStore((s) => s.refreshService)
  const setServiceEnabled = useServiceStore((s) => s.setServiceEnabled)
  const [busy, setBusy] = useState(false)
  const [unloadError, setUnloadError] = useState<string | null>(null)

  const statusMap: Record<string, { color: string; label: string }> = {
    active: { color: "bg-emerald-500/20 text-emerald-200 border-emerald-500/30", label: "Activo" },
    idle: { color: "bg-blue-500/20 text-blue-200 border-blue-500/30", label: "Inactivo" },
    unreachable: { color: "bg-red-500/20 text-red-200 border-red-500/30", label: "No disponible" },
    disabled: { color: "bg-amber-500/20 text-amber-200 border-amber-500/30", label: "Desactivado" },
    building: { color: "bg-violet-500/20 text-violet-200 border-violet-500/30 animate-pulse", label: "Construyendo" },
    starting: { color: "bg-sky-500/20 text-sky-200 border-sky-500/30 animate-pulse", label: "Arrancando" },
  }
  const { color: statusColor, label: statusLabel } = statusMap[service.status] ?? {
    color: "bg-muted text-muted-foreground border-border",
    label: "Desconocido",
  }

  async function handleUnload() {
    setBusy(true)
    setUnloadError(null)
    try { await unloadService(service.serviceId) } catch (err) {
      setUnloadError(err instanceof Error ? err.message : "Error al descargar el modelo")
    } finally { setBusy(false) }
  }

  async function handleStart() {
    setBusy(true)
    setUnloadError(null)
    try { await startService(service.serviceId) } catch (err) {
      setUnloadError(err instanceof Error ? err.message : "Error al iniciar el servicio")
    } finally { setBusy(false) }
  }

  async function handleToggleEnabled() {
    setBusy(true)
    setUnloadError(null)
    try { await setServiceEnabled(service.serviceId, !service.enabled) } catch (err) {
      setUnloadError(err instanceof Error ? err.message : "Error al cambiar el estado del servicio")
    } finally { setBusy(false) }
  }

  async function handleRefresh() {
    setBusy(true)
    try { await refreshService(service.serviceId) } finally { setBusy(false) }
  }

  return (
    <div className="rounded-lg border border-border bg-card px-4 py-3 space-y-2">
      {/* Header row */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <p className="font-medium">{service.displayName}</p>
          <span className={`rounded-full border px-2 py-0.5 text-xs font-medium ${statusColor}`}>
            {statusLabel}
          </span>
        </div>
        <div className="flex items-center gap-2">
          {service.uiUrl && service.serviceAlive && (
            <a href={service.uiUrl} target="_blank" rel="noopener noreferrer"
              className="rounded-md border border-border px-3 py-1 text-xs hover:bg-muted">
              Abrir UI
            </a>
          )}
          <button type="button" onClick={() => void handleToggleEnabled()} disabled={busy}
            className="rounded-md border border-amber-500/40 px-3 py-1 text-xs text-amber-200 hover:bg-amber-500/20 disabled:opacity-50">
            {service.enabled ? "Desactivar" : "Activar"}
          </button>
          <button type="button" onClick={() => void handleRefresh()} disabled={busy || !service.enabled}
            className="rounded-md border border-border px-3 py-1 text-xs hover:bg-muted disabled:opacity-50">
            Actualizar
          </button>
          {service.enabled && !service.serviceAlive && !["building", "starting"].includes(service.status) && (
            <button type="button" onClick={() => void handleStart()} disabled={busy}
              className="rounded-md border border-emerald-500/40 px-3 py-1 text-xs text-emerald-200 hover:bg-emerald-500/20 disabled:opacity-50">
              Iniciar
            </button>
          )}
          {service.enabled && service.runtimeLoaded && (
            <button type="button" onClick={() => void handleUnload()} disabled={busy}
              className="rounded-md border border-red-500/40 px-3 py-1 text-xs text-red-200 hover:bg-red-500/20 disabled:opacity-50">
              Descargar
            </button>
          )}
        </div>
      </div>

      {/* Metrics grid */}
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-5 text-xs">
        {service.serviceAlive ? (
          <span className="rounded-md bg-emerald-500/10 px-2 py-1 text-emerald-300 text-center">UI online</span>
        ) : (
          <span className="rounded-md bg-red-500/10 px-2 py-1 text-red-300 text-center">UI offline</span>
        )}
        {service.enabled && service.runtimeLoaded && (
          <span className="rounded-md bg-emerald-500/10 px-2 py-1 text-emerald-300 text-center truncate" title={service.activeModelRef ?? undefined}>
            {service.activeModelRef ? `${service.activeModelRef}` : "Runtime OK"}
          </span>
        )}
        {service.isGenerating && (
          <span className="inline-flex items-center justify-center gap-1 rounded-md bg-emerald-900/40 px-2 py-1 text-emerald-300 border border-emerald-500/30">
            <Activity size={10} />
            Generando{service.queueDepth > 0 ? ` +${service.queueDepth}` : ""}
          </span>
        )}
        {service.vramUsedMb != null && (
          <span className="inline-flex items-center justify-center gap-1 rounded-md bg-muted px-2 py-1">
            <Database size={10} /> {(service.vramUsedMb / 1024).toFixed(1)} GB
          </span>
        )}
        {service.gpuUtilPct != null && (
          <span className="inline-flex items-center justify-center gap-1 rounded-md bg-muted px-2 py-1">
            <Zap size={10} /> GPU {Math.round(service.gpuUtilPct)}%
          </span>
        )}
        {service.cpuPct != null && (
          <span className="inline-flex items-center justify-center gap-1 rounded-md bg-muted px-2 py-1">
            <Cpu size={10} /> CPU {service.cpuPct.toFixed(1)}%
          </span>
        )}
        {service.memUsedMb != null && (
          <span className="inline-flex items-center justify-center gap-1 rounded-md bg-muted px-2 py-1">
            <Layers size={10} /> RAM {(service.memUsedMb / 1024).toFixed(1)}
            {service.memLimitMb != null ? `/${(service.memLimitMb / 1024).toFixed(0)}` : ""} GB
          </span>
        )}
      </div>

      {/* Errors */}
      {unloadError && (
        <p className="text-xs text-red-400 truncate" title={unloadError}>{unloadError}</p>
      )}
      {!service.enabled && service.serviceAlive && (
        <p className="text-xs text-red-300">Runtime activo fuera de oCabra</p>
      )}
      {service.detail && (
        <p className="text-xs text-muted-foreground/70 truncate" title={service.detail}>{service.detail}</p>
      )}
    </div>
  )
}

/* ─── Federation Summary ─── */
function FederationSummary() {
  const [peers, setPeers] = useState<FederationPeer[]>([])

  useEffect(() => {
    let active = true
    async function load() {
      try {
        const data = await api.federation.getPeers()
        if (active) setPeers(data)
      } catch { /* Federation may not be enabled */ }
    }
    void load()
    const timer = window.setInterval(() => { void load() }, 30_000)
    return () => { active = false; window.clearInterval(timer) }
  }, [])

  const onlinePeers = peers.filter((p) => p.enabled && p.online)
  if (onlinePeers.length === 0) return null

  const totalModels = onlinePeers.reduce((acc, p) => acc + p.models.length, 0)
  const totalGpus = onlinePeers.reduce((acc, p) => acc + p.gpus.length, 0)

  return (
    <Section title="Federacion">
      <div className="rounded-lg border border-border bg-card px-4 py-3">
        <div className="flex flex-wrap items-center gap-5 text-sm">
          <div className="flex items-center gap-2">
            <Network size={16} className="text-muted-foreground" />
            <span className="font-medium">{onlinePeers.length}</span>
            <span className="text-muted-foreground">nodo{onlinePeers.length !== 1 ? "s" : ""} online</span>
          </div>
          <div className="flex items-center gap-2">
            <Globe size={16} className="text-muted-foreground" />
            <span className="font-medium">{totalModels}</span>
            <span className="text-muted-foreground">modelo{totalModels !== 1 ? "s" : ""} remotos</span>
          </div>
          <div className="flex items-center gap-2">
            <Cpu size={16} className="text-muted-foreground" />
            <span className="font-medium">{totalGpus}</span>
            <span className="text-muted-foreground">GPU{totalGpus !== 1 ? "s" : ""} remota{totalGpus !== 1 ? "s" : ""}</span>
          </div>
        </div>
      </div>
    </Section>
  )
}

/* ─── Active Agents (last hour) ─── */
// TODO: remove mock once backend exposes /ocabra/stats/by-agent.
const MOCK_ACTIVE_AGENTS: AgentStatRow[] = [
  {
    agentId: "mock-agent-1",
    slug: "research-bot",
    displayName: "Research bot",
    requestCount: 12,
    toolCallCount: 27,
    errorCount: 0,
    p50DurationMs: 1800,
    p95DurationMs: 5200,
    totalTokens: 24100,
  },
]

function ActiveAgentsSection() {
  const [loading, setLoading] = useState(true)
  const [rows, setRows] = useState<AgentStatRow[]>([])
  const [usingMock, setUsingMock] = useState(false)

  useEffect(() => {
    let active = true
    const load = async () => {
      const now = new Date()
      const from = new Date(now.getTime() - 60 * 60 * 1000)
      try {
        const data = await agentStatsApi.byAgent({ from: from.toISOString(), to: now.toISOString() })
        if (!active) return
        setRows(data.byAgent)
        setUsingMock(false)
      } catch {
        if (!active) return
        setRows(MOCK_ACTIVE_AGENTS)
        setUsingMock(true)
      } finally {
        if (active) setLoading(false)
      }
    }
    void load()
    const timer = window.setInterval(() => void load(), 60_000)
    return () => {
      active = false
      window.clearInterval(timer)
    }
  }, [])

  return (
    <Section title="Active agents (last hour)" defaultOpen={rows.length > 0}>
      {loading ? (
        <div className="h-20 animate-pulse rounded-lg bg-muted" />
      ) : rows.length === 0 ? (
        <EmptyState
          icon={<Sparkles size={16} />}
          title="Sin agents activos"
          description="Los agents aparecerán aquí cuando reciban requests."
        />
      ) : (
        <div className="space-y-2">
          {usingMock && (
            <p className="text-xs text-muted-foreground">
              (mock) Endpoint <code>/ocabra/stats/by-agent</code> todavía no expuesto por el backend.
            </p>
          )}
          {rows.map((row) => (
            <div
              key={row.agentId}
              className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-border bg-card px-4 py-3 text-sm"
            >
              <div className="flex items-center gap-2">
                <Sparkles size={14} className="text-primary" />
                <span className="font-medium">{row.displayName}</span>
                <code className="font-mono text-xs text-muted-foreground">agent/{row.slug}</code>
              </div>
              <div className="flex flex-wrap gap-3 text-xs text-muted-foreground">
                <span>{row.requestCount} requests</span>
                <span>{row.toolCallCount} tool calls</span>
                <span>{row.totalTokens.toLocaleString()} tokens</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </Section>
  )
}

/* ─── Recent Requests ─── */
function RecentRequestsSection() {
  const [data, setData] = useState<RecentRequestsData>({ requests: [] })
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let active = true
    async function load() {
      try {
        const result = await api.stats.recent(20)
        if (active) { setData(result); setLoading(false) }
      } catch { if (active) setLoading(false) }
    }
    void load()
    const timer = window.setInterval(() => { void load() }, 30_000)
    return () => { active = false; window.clearInterval(timer) }
  }, [])

  return (
    <Section title="Ultimas peticiones" defaultOpen={false}>
      {loading ? (
        <div className="flex items-center gap-3 rounded-lg border border-border bg-card px-4 py-4 text-sm text-muted-foreground">
          <div className="h-4 w-4 animate-spin rounded-full border-2 border-primary border-t-transparent" role="status" aria-label="Cargando peticiones recientes" />
          Cargando...
        </div>
      ) : data.requests.length === 0 ? (
        <EmptyState title="Sin peticiones recientes" description="Las peticiones apareceran aqui cuando se procesen." />
      ) : (
        <div className="overflow-auto rounded-lg border border-border">
          <table className="min-w-full divide-y divide-border text-sm">
            <thead className="bg-muted/50">
              <tr>
                <th className="px-3 py-2 text-left font-medium text-muted-foreground">Hora</th>
                <th className="px-3 py-2 text-left font-medium text-muted-foreground">Modelo</th>
                <th className="px-3 py-2 text-left font-medium text-muted-foreground">Tipo</th>
                <th className="px-3 py-2 text-right font-medium text-muted-foreground">Duracion</th>
                <th className="px-3 py-2 text-right font-medium text-muted-foreground">Tokens</th>
                <th className="px-3 py-2 text-left font-medium text-muted-foreground">Usuario / API key</th>
                <th className="px-3 py-2 text-left font-medium text-muted-foreground">Estado</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border bg-card">
              {data.requests.map((req) => {
                const statusColor =
                  req.error ? "text-red-400"
                  : req.statusCode != null && req.statusCode < 400 ? "text-emerald-400"
                  : "text-muted-foreground"
                const statusLabel = req.error ? `Error: ${req.error.substring(0, 40)}` : req.statusCode != null ? String(req.statusCode) : "—"
                const hora = new Date(req.startedAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })
                return (
                  <tr key={req.id} className={`hover:bg-muted/30 transition-colors${req.error ? " bg-red-500/5" : ""}`}>
                    <td className="px-3 py-2 font-mono text-xs">{hora}</td>
                    <td className="px-3 py-2 max-w-[10rem] truncate" title={req.modelId}>{req.modelId}</td>
                    <td className="px-3 py-2 text-muted-foreground">{req.requestKind ?? req.backendType ?? "—"}</td>
                    <td className="px-3 py-2 text-right">{req.durationMs != null ? `${req.durationMs} ms` : "—"}</td>
                    <td className="px-3 py-2 text-right">
                      {req.inputTokens != null || req.outputTokens != null
                        ? `${req.inputTokens ?? 0} / ${req.outputTokens ?? 0}` : "—"}
                    </td>
                    <td className="px-3 py-2">
                      <div className={req.username ? "text-muted-foreground" : "italic text-muted-foreground/60"}>
                        {req.username ?? "anónimo"}
                      </div>
                      {req.apiKeyName && (
                        <div
                          className="max-w-[12rem] truncate font-mono text-[11px] text-muted-foreground/70"
                          title={`API key: ${req.apiKeyName}`}
                        >
                          🔑 {req.apiKeyName}
                        </div>
                      )}
                      {!req.username && (req.clientAddr || req.userAgent) && (
                        <div
                          className="max-w-[12rem] truncate font-mono text-[11px] text-muted-foreground/50"
                          title={req.userAgent ?? undefined}
                        >
                          {req.clientAddr ?? "?"}{req.userAgent ? ` · ${req.userAgent}` : ""}
                        </div>
                      )}
                    </td>
                    <td className={`px-3 py-2 text-xs font-medium ${statusColor}`} title={req.error ?? undefined}>{statusLabel}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </Section>
  )
}

/* ─── Dashboard ─── */
export function Dashboard() {
  const isModelManager = useIsModelManager()
  const [error, setError] = useState<string | null>(null)
  const [nowMs, setNowMs] = useState<number>(() => Date.now())
  const [hostStats, setHostStats] = useState<HostStats | null>(null)
  const [serverPower, setServerPower] = useState<ServerPower | null>(null)
  const [hostHistory, setHostHistory] = useState<HostHistoryPoint[]>([])

  useWebSocket()

  const gpus = useGpuStore((state) => state.gpus)
  const setGpus = useGpuStore((state) => state.setGpus)

  const models = useModelStore((state) => state.models)
  const setModels = useModelStore((state) => state.setModels)
  const unloadModel = useModelStore((state) => state.unloadModel)

  const jobs = useDownloadStore((state) => state.jobs)
  const setJobs = useDownloadStore((state) => state.setJobs)

  const services = useServiceStore((state) => state.services)
  const setServices = useServiceStore((state) => state.setServices)

  const activeModels = useMemo(
    () => Object.values(models).filter((model) => model.status === "loaded" || model.status === "loading"),
    [models],
  )
  const activeDownloads = useMemo(
    () => jobs.filter((job) => job.status === "queued" || job.status === "downloading"),
    [jobs],
  )
  const serviceList = useMemo(() => Object.values(services), [services])
  const externalRuntimeServices = useMemo(
    () => serviceList.filter((service) => !service.enabled && service.serviceAlive),
    [serviceList],
  )

  // Total VRAM
  const totalVramMb = useMemo(() => gpus.reduce((acc, g) => acc + g.totalVramMb, 0), [gpus])
  const usedVramMb = useMemo(() => gpus.reduce((acc, g) => acc + g.usedVramMb, 0), [gpus])

  useEffect(() => {
    const timer = window.setInterval(() => setNowMs(Date.now()), 1000)
    return () => window.clearInterval(timer)
  }, [])

  useEffect(() => {
    async function pollHostStats() {
      let stats: HostStats | null = null
      let power: ServerPower | null = null
      try { stats = await api.host.stats(); setHostStats(stats) } catch { /* keep stale */ }
      // CPU temperature/power comes from hw-monitor via server:power (k10temp),
      // not from the host psutil stats — fetch it alongside.
      try { power = await api.stats.serverPower(); setServerPower(power) } catch { /* keep stale */ }
      // Accumulate a ring buffer (~10 min at 5s) to feed the host sparklines.
      if (stats) {
        const point: HostHistoryPoint = {
          t: Date.now(),
          cpuPct: stats.cpuPct,
          memPct: stats.memPct,
          tempC: power?.cpuTempC ?? null,
          powerW: power?.cpuPowerW ?? null,
        }
        setHostHistory((prev) => [...prev.slice(-119), point])
      }
    }
    void pollHostStats()
    const timer = window.setInterval(() => void pollHostStats(), 5000)
    return () => window.clearInterval(timer)
  }, [])

  useEffect(() => {
    async function bootstrap() {
      try {
        const [gpuList, modelList, downloadList, servicesList] = await Promise.all([
          api.gpus.list(), api.models.list(), api.downloads.list(), api.services.list(),
        ])
        setGpus(gpuList); setModels(modelList); setJobs(downloadList); setServices(servicesList)
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load dashboard data")
      }
    }
    void bootstrap()
  }, [setGpus, setJobs, setModels, setServices])

  return (
    <div className="space-y-6">
      {/* KPI Cards */}
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <KpiCard
          icon={Cpu}
          label="GPUs"
          value={gpus.length}
          sub={gpus.length > 0 ? `${gpus.filter(g => g.utilizationPct > 10).length} activa${gpus.filter(g => g.utilizationPct > 10).length !== 1 ? "s" : ""}` : undefined}
          color="bg-indigo-500/15 text-indigo-300"
        />
        <KpiCard
          icon={Boxes}
          label="Modelos cargados"
          value={activeModels.length}
          sub={`${Object.keys(models).length} registrados`}
          color="bg-emerald-500/15 text-emerald-300"
        />
        <KpiCard
          icon={MemoryStick}
          label="VRAM usada"
          value={totalVramMb > 0 ? `${(usedVramMb / 1024).toFixed(1)} GB` : "—"}
          sub={totalVramMb > 0 ? `de ${(totalVramMb / 1024).toFixed(0)} GB (${Math.round(usedVramMb / totalVramMb * 100)}%)` : undefined}
          color="bg-cyan-500/15 text-cyan-300"
        />
        <KpiCard
          icon={Activity}
          label="Servicios"
          value={serviceList.filter(s => s.status === "active").length}
          sub={`${serviceList.length} configurados`}
          color="bg-amber-500/15 text-amber-300"
        />
      </div>

      {/* GPUs + Host — 2 column grid */}
      <Section title="GPUs y host">
        <div className="grid gap-4 xl:grid-cols-2">
          {gpus.map((gpu) => (
            <GpuCard key={gpu.index} gpu={gpu} />
          ))}
          {hostStats && <HostStatsCard stats={hostStats} serverPower={serverPower} history={hostHistory} />}
          {gpus.length === 0 && !hostStats && (
            <EmptyState title="Sin datos de GPU" description="No se detectaron GPUs o el servicio de monitoreo no esta disponible." />
          )}
        </div>
      </Section>

      {/* Services */}
      <Section title="Servicios de generacion" badge={
        <span className="ml-2 rounded-full bg-muted px-2 py-0.5 text-xs text-muted-foreground">{serviceList.length}</span>
      }>
        <div className="space-y-3">
          {serviceList.map((service) => (
            <ServiceCard key={service.serviceId} service={service} />
          ))}
          {serviceList.length === 0 && (
            <EmptyState title="Sin servicios configurados" description="Anade servicios (ComfyUI, A1111...) en la seccion de configuracion." />
          )}
        </div>
      </Section>

      <FederationSummary />

      {/* Active Models */}
      <Section title="Modelos activos" badge={
        activeModels.length > 0 ? <span className="ml-2 rounded-full bg-emerald-500/15 px-2 py-0.5 text-xs text-emerald-300">{activeModels.length}</span> : undefined
      }>
        <div className="space-y-3">
          {activeModels.length === 0 && externalRuntimeServices.length > 0 && (
            <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-sm text-amber-100">
              Hay runtimes externos ocupando GPU en servicios de generacion.
            </div>
          )}
          {activeModels.map((model) => (
            <div
              key={model.modelId}
              className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-border bg-card px-4 py-3"
            >
              <div className="space-y-1">
                <p className="font-medium">{model.displayName}</p>
                <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                  <ModelStatusBadge status={model.status} />
                  <LoadPolicyBadge policy={model.loadPolicy} />
                  <span className="rounded-md bg-muted px-2 py-0.5">GPU {model.currentGpu.join(", ") || "-"}</span>
                  <span>{model.vramUsedMb.toLocaleString()} MB</span>
                  {loadedMeta(model.loadedAt, nowMs) && (
                    <span className="rounded-md bg-muted px-2 py-0.5">
                      Cargado {loadedMeta(model.loadedAt, nowMs)!.at} · {loadedMeta(model.loadedAt, nowMs)!.ago}
                    </span>
                  )}
                </div>
              </div>
              {model.status === "loaded" && (
                <button type="button" onClick={() => void unloadModel(model.modelId)}
                  className="rounded-md border border-red-500/40 px-3 py-1 text-sm text-red-200 hover:bg-red-500/20">
                  Descargar
                </button>
              )}
            </div>
          ))}
          {activeModels.length === 0 && (
            <EmptyState title="Sin modelos cargados" description="Carga un modelo desde la pagina de Models." />
          )}
        </div>
      </Section>

      {/* Downloads */}
      <Section title="Descargas activas" defaultOpen={activeDownloads.length > 0}>
        <div className="space-y-3">
          {activeDownloads.map((job) => (
            <div key={job.jobId} className="rounded-lg border border-border bg-card px-4 py-3">
              <div className="mb-2 flex items-center justify-between gap-2 text-sm">
                <p className="font-medium">{job.modelRef || job.jobId}</p>
                <p className="text-muted-foreground">
                  {job.speedMbS ? `${job.speedMbS.toFixed(1)} MB/s` : "--"}{" · "}
                  ETA {job.etaSeconds ? `${Math.ceil(job.etaSeconds)}s` : "--"}
                </p>
              </div>
              <div className="h-2 overflow-hidden rounded-full bg-muted">
                <div className="h-full animate-pulse bg-blue-500 transition-all"
                  style={{ width: `${Math.min(100, Math.max(0, job.progressPct))}%` }} />
              </div>
            </div>
          ))}
          {activeDownloads.length === 0 && (
            <EmptyState icon={<Download size={16} />} title="Sin descargas activas" />
          )}
        </div>
      </Section>

      {isModelManager && <ActiveAgentsSection />}

      {isModelManager && <RecentRequestsSection />}

      {error && (
        <div className="rounded-md border border-red-500/40 bg-red-500/10 px-4 py-3 text-sm text-red-200">{error}</div>
      )}
    </div>
  )
}
