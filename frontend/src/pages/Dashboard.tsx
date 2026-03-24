import { useEffect, useMemo, useState } from "react"
import { api } from "@/api/client"
import { GpuCard } from "@/components/gpu/GpuCard"
import { LoadPolicyBadge } from "@/components/models/LoadPolicyBadge"
import { ModelStatusBadge } from "@/components/models/ModelStatusBadge"
import { useWebSocket } from "@/hooks/useWebSocket"
import { useDownloadStore } from "@/stores/downloadStore"
import { useGpuStore } from "@/stores/gpuStore"
import { useModelStore } from "@/stores/modelStore"
import { useServiceStore } from "@/stores/serviceStore"
import type { ServiceState } from "@/types"

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
function ServiceCard({ service }: { service: ServiceState }) {
  const unloadService = useServiceStore((s) => s.unloadService)
  const startService = useServiceStore((s) => s.startService)
  const refreshService = useServiceStore((s) => s.refreshService)
  const setServiceEnabled = useServiceStore((s) => s.setServiceEnabled)
  const [busy, setBusy] = useState(false)
  const [unloadError, setUnloadError] = useState<string | null>(null)

  const statusColor =
    service.status === "active"
      ? "bg-emerald-500/20 text-emerald-200 border-emerald-500/30"
      : service.status === "idle"
        ? "bg-blue-500/20 text-blue-200 border-blue-500/30"
        : service.status === "unreachable"
          ? "bg-red-500/20 text-red-200 border-red-500/30"
          : service.status === "disabled"
            ? "bg-amber-500/20 text-amber-200 border-amber-500/30"
            : "bg-muted text-muted-foreground border-border"

  const statusLabel =
    service.status === "active"
      ? "Activo"
      : service.status === "idle"
        ? "Inactivo"
        : service.status === "unreachable"
          ? "No disponible"
          : service.status === "disabled"
            ? "Desactivado"
            : "Desconocido"

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
        {service.enabled && !service.serviceAlive && (
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

export function Dashboard() {
  const [error, setError] = useState<string | null>(null)
  const [nowMs, setNowMs] = useState<number>(() => Date.now())

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

  useEffect(() => {
    const timer = window.setInterval(() => setNowMs(Date.now()), 1000)
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
      <section className="space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-semibold">GPU Cards</h2>
          <span
            className={`rounded-full px-3 py-1 text-xs font-medium ${
              connected ? "bg-emerald-500/20 text-emerald-200" : "bg-amber-500/20 text-amber-200"
            }`}
          >
            {connected ? "Live updates connected" : "Reconnecting WebSocket..."}
          </span>
        </div>

        <div className="grid gap-4 xl:grid-cols-2">
          {gpus.map((gpu) => (
            <GpuCard key={gpu.index} gpu={gpu} />
          ))}
          {gpus.length === 0 && (
            <div className="rounded-lg border border-dashed border-border p-8 text-center text-muted-foreground">
              No GPU stats available.
            </div>
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
            <div className="rounded-lg border border-dashed border-border p-6 text-muted-foreground">
              No hay servicios configurados.
            </div>
          )}
        </div>
      </section>

      <section className="space-y-4">
        <h2 className="text-xl font-semibold">Modelos activos</h2>
        <div className="space-y-3">
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
            <div className="rounded-lg border border-dashed border-border p-6 text-muted-foreground">
              No hay modelos cargados en este momento.
            </div>
          )}
        </div>
      </section>

      <section className="space-y-4">
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
            <div className="rounded-lg border border-dashed border-border p-6 text-muted-foreground">
              No hay descargas en progreso.
            </div>
          )}
        </div>
      </section>

      {error && (
        <div className="rounded-md border border-red-500/40 bg-red-500/10 px-4 py-3 text-sm text-red-200">
          {error}
        </div>
      )}
    </div>
  )
}
