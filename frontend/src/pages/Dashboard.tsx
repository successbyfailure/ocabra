import { useEffect, useMemo, useState } from "react"
import { Activity, Cpu, Database, Download, Globe, Layers, Network, Zap } from "lucide-react"
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
import type { FederationPeer, HostStats, RecentRequestsData, ServiceState } from "@/types"

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
function MiniBar({ pct, color }: { pct: number; color: string }) {
  return (
    <div className="h-1.5 w-full rounded-full bg-muted overflow-hidden">
      <div
        className={`h-full rounded-full transition-all ${color}`}
        style={{ width: `${Math.min(100, Math.max(0, pct))}%` }}
      />
    </div>
  )
}

function HostStatsCard({ stats }: { stats: HostStats }) {
  const cpuColor =
    stats.cpuPct > 90 ? "bg-red-500" : stats.cpuPct > 70 ? "bg-amber-500" : "bg-emerald-500"
  const memColor =
    stats.memPct > 90 ? "bg-red-500" : stats.memPct > 70 ? "bg-amber-500" : "bg-blue-500"

  const memUsedGb = (stats.memUsedMb / 1024).toFixed(1)
  const memTotalGb = (stats.memTotalMb / 1024).toFixed(1)

  return (
    <div className="rounded-lg border border-border bg-card px-4 py-4 space-y-3">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium">Host · {stats.cpuCountPhysical}C/{stats.cpuCount}T</span>
        <span className="text-xs text-muted-foreground">
          load {stats.loadAvg1m} / {stats.loadAvg5m} / {stats.loadAvg15m}
        </span>
      </div>

      <div className="space-y-1.5">
        <div className="flex justify-between text-xs text-muted-foreground">
          <span>CPU</span>
          <span className="font-mono">{stats.cpuPct.toFixed(1)}%</span>
        </div>
        <MiniBar pct={stats.cpuPct} color={cpuColor} />
      </div>

      <div className="space-y-1.5">
        <div className="flex justify-between text-xs text-muted-foreground">
          <span>RAM</span>
          <span className="font-mono">{memUsedGb} / {memTotalGb} GB</span>
        </div>
        <MiniBar pct={stats.memPct} color={memColor} />
      </div>

      {stats.swapTotalMb > 0 && (
        <div className="space-y-1.5">
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>Swap</span>
            <span className="font-mono">{(stats.swapUsedMb / 1024).toFixed(1)} / {(stats.swapTotalMb / 1024).toFixed(1)} GB</span>
          </div>
          <MiniBar pct={stats.swapPct} color="bg-purple-500" />
        </div>
      )}
    </div>
  )
}

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
    try {
      await unloadService(service.serviceId)
    } catch (err) {
      setUnloadError(err instanceof Error ? err.message : "Error al descargar el modelo")
    } finally {
      setBusy(false)
    }
  }

  async function handleStart() {
    setBusy(true)
    setUnloadError(null)
    try {
      await startService(service.serviceId)
    } catch (err) {
      setUnloadError(err instanceof Error ? err.message : "Error al iniciar el servicio")
    } finally {
      setBusy(false)
    }
  }


  async function handleToggleEnabled() {
    setBusy(true)
    setUnloadError(null)
    try {
      await setServiceEnabled(service.serviceId, !service.enabled)
    } catch (err) {
      setUnloadError(err instanceof Error ? err.message : "Error al cambiar el estado del servicio")
    } finally {
      setBusy(false)
    }
  }

  async function handleRefresh() {
    setBusy(true)
    try {
      await refreshService(service.serviceId)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-border bg-card px-4 py-3">
      <div className="space-y-1">
        <div className="flex items-center gap-2">
          <p className="font-medium">{service.displayName}</p>
          <span className={`rounded-full border px-2 py-0.5 text-xs font-medium ${statusColor}`}>
            {statusLabel}
          </span>
        </div>
        <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
          {service.serviceAlive ? (
            <span className="text-emerald-400">UI online</span>
          ) : (
            <span className="text-red-400">UI offline</span>
          )}
          {!service.enabled && (
            <span className="rounded-md bg-amber-500/10 px-2 py-0.5 text-amber-300">Desactivado</span>
          )}
          {service.enabled && service.runtimeLoaded && (
            <span className="rounded-md bg-emerald-500/10 px-2 py-0.5 text-emerald-300">
              {service.activeModelRef ? `Modelo: ${service.activeModelRef}` : "Runtime cargado"}
            </span>
          )}
          {service.isGenerating && (
            <span className="inline-flex items-center gap-1 rounded-md bg-emerald-900/40 px-2 py-0.5 text-emerald-300 border border-emerald-500/30">
              <Activity size={11} aria-hidden="true" />
              Generando{service.queueDepth > 0 ? ` (+${service.queueDepth} en cola)` : ""}
            </span>
          )}
          {service.vramUsedMb != null && (
            <span className="inline-flex items-center gap-1 rounded-md bg-muted px-2 py-0.5">
              <Database size={11} aria-hidden="true" />
              {(service.vramUsedMb / 1024).toFixed(1)} GB VRAM
            </span>
          )}
          {service.gpuUtilPct != null && (
            <span className="inline-flex items-center gap-1 rounded-md bg-muted px-2 py-0.5">
              <Zap size={11} aria-hidden="true" />
              GPU {Math.round(service.gpuUtilPct)}%
            </span>
          )}
          {service.cpuPct != null && (
            <span className="inline-flex items-center gap-1 rounded-md bg-muted px-2 py-0.5">
              <Cpu size={11} aria-hidden="true" />
              CPU {service.cpuPct.toFixed(1)}%
            </span>
          )}
          {service.memUsedMb != null && (
            <span className="inline-flex items-center gap-1 rounded-md bg-muted px-2 py-0.5">
              <Layers size={11} aria-hidden="true" />
              RAM {(service.memUsedMb / 1024).toFixed(1)}
              {service.memLimitMb != null ? `/${(service.memLimitMb / 1024).toFixed(0)}` : ""} GB
            </span>
          )}
          {!service.enabled && service.serviceAlive && (
            <span className="rounded-md bg-red-500/10 px-2 py-0.5 text-red-300">
              Runtime activo fuera de oCabra
            </span>
          )}
          {service.preferredGpu != null && (
            <span className="rounded-md bg-muted px-2 py-0.5">GPU {service.preferredGpu}</span>
          )}
          {service.detail && (
            <span className="text-muted-foreground/70 max-w-xs truncate" title={service.detail}>
              {service.detail}
            </span>
          )}
          {unloadError && (
            <span className="text-red-400 max-w-xs truncate" title={unloadError}>
              {unloadError}
            </span>
          )}
        </div>
      </div>

      <div className="flex items-center gap-2">
        {service.uiUrl && service.serviceAlive && (
          <a
            href={service.uiUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="rounded-md border border-border px-3 py-1 text-sm hover:bg-muted"
          >
            Abrir UI ↗
          </a>
        )}
        <button
          type="button"
          onClick={() => void handleToggleEnabled()}
          disabled={busy}
          className="rounded-md border border-amber-500/40 px-3 py-1 text-sm text-amber-200 hover:bg-amber-500/20 disabled:opacity-50"
        >
          {service.enabled ? "Desactivar" : "Activar"}
        </button>
        <button
          type="button"
          onClick={() => void handleRefresh()}
          disabled={busy || !service.enabled}
          className="rounded-md border border-border px-3 py-1 text-sm hover:bg-muted disabled:opacity-50"
        >
          Actualizar
        </button>
        {service.enabled && !service.serviceAlive && !["building", "starting"].includes(service.status) && (
          <button
            type="button"
            onClick={() => void handleStart()}
            disabled={busy}
            className="rounded-md border border-emerald-500/40 px-3 py-1 text-sm text-emerald-200 hover:bg-emerald-500/20 disabled:opacity-50"
          >
            Iniciar
          </button>
        )}
        {service.enabled && service.runtimeLoaded && (
          <button
            type="button"
            onClick={() => void handleUnload()}
            disabled={busy}
            className="rounded-md border border-red-500/40 px-3 py-1 text-sm text-red-200 hover:bg-red-500/20 disabled:opacity-50"
          >
            Descargar
          </button>
        )}
      </div>
    </div>
  )
}

function FederationSummary() {
  const [peers, setPeers] = useState<FederationPeer[]>([])

  useEffect(() => {
    let active = true

    async function load() {
      try {
        const data = await api.federation.getPeers()
        if (active) setPeers(data)
      } catch {
        // Federation may not be enabled
      }
    }

    void load()
    const timer = window.setInterval(() => { void load() }, 30_000)
    return () => {
      active = false
      window.clearInterval(timer)
    }
  }, [])

  const onlinePeers = peers.filter((p) => p.enabled && p.online)
  if (onlinePeers.length === 0) return null

  const totalModels = onlinePeers.reduce((acc, p) => acc + p.models.length, 0)
  const totalGpus = onlinePeers.reduce((acc, p) => acc + p.gpus.length, 0)

  return (
    <section className="space-y-2">
      <h2 className="text-xl font-semibold">Federacion</h2>
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
    </section>
  )
}

function RecentRequestsSection() {
  const [data, setData] = useState<RecentRequestsData>({ requests: [] })
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let active = true

    async function load() {
      try {
        const result = await api.stats.recent(20)
        if (active) {
          setData(result)
          setLoading(false)
        }
      } catch {
        if (active) setLoading(false)
      }
    }

    void load()
    const timer = window.setInterval(() => { void load() }, 30_000)
    return () => {
      active = false
      window.clearInterval(timer)
    }
  }, [])

  return (
    <section className="space-y-4">
      <h2 className="text-xl font-semibold">Últimas peticiones</h2>
      {loading ? (
        <div className="flex items-center gap-3 rounded-lg border border-border bg-card px-4 py-4 text-sm text-muted-foreground">
          <div className="h-4 w-4 animate-spin rounded-full border-2 border-primary border-t-transparent" role="status" aria-label="Cargando peticiones recientes" />
          Cargando…
        </div>
      ) : data.requests.length === 0 ? (
        <EmptyState title="Sin peticiones recientes" description="Las peticiones aparecerán aquí cuando se procesen." />
      ) : (
        <div className="overflow-auto rounded-lg border border-border">
          <table className="min-w-full divide-y divide-border text-sm">
            <thead className="bg-muted/50">
              <tr>
                <th className="px-3 py-2 text-left font-medium text-muted-foreground">Hora</th>
                <th className="px-3 py-2 text-left font-medium text-muted-foreground">Modelo</th>
                <th className="px-3 py-2 text-left font-medium text-muted-foreground">Tipo</th>
                <th className="px-3 py-2 text-right font-medium text-muted-foreground">Duración</th>
                <th className="px-3 py-2 text-right font-medium text-muted-foreground">Tokens</th>
                <th className="px-3 py-2 text-left font-medium text-muted-foreground">Usuario</th>
                <th className="px-3 py-2 text-left font-medium text-muted-foreground">Estado</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border bg-card">
              {data.requests.map((req) => {
                const statusColor =
                  req.error
                    ? "text-red-400"
                    : req.statusCode != null && req.statusCode < 400
                      ? "text-emerald-400"
                      : "text-muted-foreground"
                const statusLabel =
                  req.error
                    ? `Error: ${req.error.substring(0, 40)}`
                    : req.statusCode != null
                      ? String(req.statusCode)
                      : "—"
                const hora = new Date(req.startedAt).toLocaleTimeString([], {
                  hour: "2-digit",
                  minute: "2-digit",
                  second: "2-digit",
                })
                return (
                  <tr key={req.id} className="hover:bg-muted/30 transition-colors">
                    <td className="px-3 py-2 font-mono text-xs">{hora}</td>
                    <td className="px-3 py-2 max-w-[10rem] truncate" title={req.modelId}>{req.modelId}</td>
                    <td className="px-3 py-2 text-muted-foreground">{req.requestKind ?? req.backendType ?? "—"}</td>
                    <td className="px-3 py-2 text-right">{req.durationMs != null ? `${req.durationMs} ms` : "—"}</td>
                    <td className="px-3 py-2 text-right">
                      {req.inputTokens != null || req.outputTokens != null
                        ? `${req.inputTokens ?? 0} / ${req.outputTokens ?? 0}`
                        : "—"}
                    </td>
                    <td className="px-3 py-2 text-muted-foreground">{req.username ?? "—"}</td>
                    <td className={`px-3 py-2 text-xs font-medium ${statusColor}`} title={req.error ?? undefined}>
                      {statusLabel}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}

export function Dashboard() {
  const isModelManager = useIsModelManager()
  const [error, setError] = useState<string | null>(null)
  const [nowMs, setNowMs] = useState<number>(() => Date.now())
  const [hostStats, setHostStats] = useState<HostStats | null>(null)

  const { connected } = useWebSocket()

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

  useEffect(() => {
    const timer = window.setInterval(() => setNowMs(Date.now()), 1000)
    return () => window.clearInterval(timer)
  }, [])

  useEffect(() => {
    async function pollHostStats() {
      try {
        const stats = await api.host.stats()
        setHostStats(stats)
      } catch {
        // non-critical, keep stale data
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
          api.gpus.list(),
          api.models.list(),
          api.downloads.list(),
          api.services.list(),
        ])
        setGpus(gpuList)
        setModels(modelList)
        setJobs(downloadList)
        setServices(servicesList)
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load dashboard data")
      }
    }

    void bootstrap()
  }, [setGpus, setJobs, setModels, setServices])

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-semibold">Dashboard</h1>
        <p className="text-muted-foreground">Estado en tiempo real de GPUs, modelos cargados y servicios de generación.</p>
      </div>

      <section className="space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-semibold">GPUs y host</h2>
          <span
            role="status"
            aria-live="polite"
            className={`rounded-full px-3 py-1 text-xs font-medium ${
              connected ? "bg-emerald-500/20 text-emerald-200" : "bg-amber-500/20 text-amber-200"
            }`}
          >
            {connected ? "● Live" : "⟳ Reconectando..."}
          </span>
        </div>

        <div className="grid gap-4 xl:grid-cols-2">
          {gpus.map((gpu) => (
            <GpuCard key={gpu.index} gpu={gpu} />
          ))}
          {hostStats && <HostStatsCard stats={hostStats} />}
          {gpus.length === 0 && !hostStats && (
            <EmptyState title="Sin datos de GPU" description="No se detectaron GPUs o el servicio de monitoreo no está disponible." />
          )}
        </div>
      </section>

      <section className="space-y-4">
        <h2 className="text-xl font-semibold">Servicios de generación</h2>
        <div className="space-y-3">
          {serviceList.map((service) => (
            <ServiceCard key={service.serviceId} service={service} />
          ))}
          {serviceList.length === 0 && (
            <EmptyState title="Sin servicios configurados" description="Añade servicios (ComfyUI, A1111…) en la sección de configuración." />
          )}
        </div>
      </section>

      <FederationSummary />

      <section className="space-y-4">
        <h2 className="text-xl font-semibold">Modelos activos</h2>
        <div className="space-y-3">
          {activeModels.length === 0 && externalRuntimeServices.length > 0 && (
            <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-sm text-amber-100">
              Hay runtimes externos ocupando GPU en servicios de generación. No se cuentan como modelos activos
              porque no los gestiona `model_manager`.
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
                  <span className="rounded-md bg-muted px-2 py-0.5">
                    GPU {model.currentGpu.join(", ") || "-"}
                  </span>
                  <span>{model.vramUsedMb.toLocaleString()} MB</span>
                  {loadedMeta(model.loadedAt, nowMs) && (
                    <span className="rounded-md bg-muted px-2 py-0.5">
                      Cargado {loadedMeta(model.loadedAt, nowMs)!.at} · {loadedMeta(model.loadedAt, nowMs)!.ago}
                    </span>
                  )}
                </div>
              </div>

              {model.status === "loaded" && (
                <button
                  type="button"
                  onClick={() => void unloadModel(model.modelId)}
                  className="rounded-md border border-red-500/40 px-3 py-1 text-sm text-red-200 hover:bg-red-500/20"
                >
                  Unload
                </button>
              )}
            </div>
          ))}
          {activeModels.length === 0 && (
            <EmptyState title="Sin modelos cargados" description="Carga un modelo desde la página de Models." />
          )}
        </div>
      </section>

      <section className="space-y-4" aria-label="Descargas activas">
        <h2 className="text-xl font-semibold">Descargas activas</h2>
        <div className="space-y-3">
          {activeDownloads.map((job) => (
            <div key={job.jobId} className="rounded-lg border border-border bg-card px-4 py-3">
              <div className="mb-2 flex items-center justify-between gap-2 text-sm">
                <p className="font-medium">{job.modelRef || job.jobId}</p>
                <p className="text-muted-foreground">
                  {job.speedMbS ? `${job.speedMbS.toFixed(1)} MB/s` : "--"}
                  {" · "}
                  ETA {job.etaSeconds ? `${Math.ceil(job.etaSeconds)}s` : "--"}
                </p>
              </div>

              <div className="h-2 overflow-hidden rounded-full bg-muted">
                <div
                  className="h-full animate-pulse bg-blue-500 transition-all"
                  style={{ width: `${Math.min(100, Math.max(0, job.progressPct))}%` }}
                />
              </div>
            </div>
          ))}
          {activeDownloads.length === 0 && (
            <EmptyState icon={<Download size={16} />} title="Sin descargas activas" />
          )}
        </div>
      </section>

      {isModelManager && <RecentRequestsSection />}

      {error && (
        <div className="rounded-md border border-red-500/40 bg-red-500/10 px-4 py-3 text-sm text-red-200">
          {error}
        </div>
      )}
    </div>
  )
}
