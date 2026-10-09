import { useState } from "react"
import { Activity, Cpu, Database, Layers, Zap } from "lucide-react"
import { useServiceStore } from "@/stores/serviceStore"
import type { ServiceState } from "@/types"

/* Service card (ComfyUI, A1111, ACE-Step, TRELLIS.2, …).
 * Antes vivía inline en Dashboard; extraída para reutilizarla en la nueva
 * página /services, donde se gestionan desde su propia entrada del menú. */
export function ServiceCard({ service }: { service: ServiceState }) {
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
    <div className="rounded-lg border border-border bg-card px-4 py-3 space-y-2">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <p className="font-medium">{service.displayName}</p>
          <span className={`rounded-full border px-2 py-0.5 text-xs font-medium ${statusColor}`}>
            {statusLabel}
          </span>
        </div>
        <div className="flex items-center gap-2">
          {service.uiUrl && service.serviceAlive && (
            <a
              href={service.uiUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="rounded-md border border-border px-3 py-1 text-xs hover:bg-muted"
            >
              Abrir UI
            </a>
          )}
          <button
            type="button"
            onClick={() => void handleToggleEnabled()}
            disabled={busy}
            className="rounded-md border border-amber-500/40 px-3 py-1 text-xs text-amber-200 hover:bg-amber-500/20 disabled:opacity-50"
          >
            {service.enabled ? "Desactivar" : "Activar"}
          </button>
          <button
            type="button"
            onClick={() => void handleRefresh()}
            disabled={busy || !service.enabled}
            className="rounded-md border border-border px-3 py-1 text-xs hover:bg-muted disabled:opacity-50"
          >
            Actualizar
          </button>
          {service.enabled && !service.serviceAlive && !["building", "starting"].includes(service.status) && (
            <button
              type="button"
              onClick={() => void handleStart()}
              disabled={busy}
              className="rounded-md border border-emerald-500/40 px-3 py-1 text-xs text-emerald-200 hover:bg-emerald-500/20 disabled:opacity-50"
            >
              Iniciar
            </button>
          )}
          {service.enabled && service.runtimeLoaded && (
            <button
              type="button"
              onClick={() => void handleUnload()}
              disabled={busy}
              className="rounded-md border border-red-500/40 px-3 py-1 text-xs text-red-200 hover:bg-red-500/20 disabled:opacity-50"
            >
              Descargar
            </button>
          )}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-5 text-xs">
        {service.serviceAlive ? (
          <span className="rounded-md bg-emerald-500/10 px-2 py-1 text-emerald-300 text-center">UI online</span>
        ) : (
          <span className="rounded-md bg-red-500/10 px-2 py-1 text-red-300 text-center">UI offline</span>
        )}
        {service.enabled && service.runtimeLoaded && (
          <span
            className="rounded-md bg-emerald-500/10 px-2 py-1 text-emerald-300 text-center truncate"
            title={service.activeModelRef ?? undefined}
          >
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

      {unloadError && (
        <p className="text-xs text-red-400 truncate" title={unloadError}>
          {unloadError}
        </p>
      )}
      {!service.enabled && service.serviceAlive && (
        <p className="text-xs text-red-300">Runtime activo fuera de oCabra</p>
      )}
      {service.detail && (
        <p className="text-xs text-muted-foreground/70 truncate" title={service.detail}>
          {service.detail}
        </p>
      )}
    </div>
  )
}
