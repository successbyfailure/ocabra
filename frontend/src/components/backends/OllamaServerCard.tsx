import { useEffect, useRef, useState } from "react"
import { ArrowUpCircle, RefreshCw, Loader2, AlertCircle, CheckCircle2 } from "lucide-react"
import { toast } from "sonner"
import { api } from "@/api/client"

type VersionInfo = {
  current: string | null
  latest: string | null
  updateAvailable: boolean
}

type UpdateStatus = {
  status: "idle" | "pulling" | "restarting" | "done" | "error"
  detail: string | null
  startedAt: string | null
  finishedAt: string | null
  fromVersion: string | null
  toVersion: string | null
}

const POLL_MS = 3000

export function OllamaServerCard() {
  const [version, setVersion] = useState<VersionInfo | null>(null)
  const [updateStatus, setUpdateStatus] = useState<UpdateStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [bulkBusy, setBulkBusy] = useState(false)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const fetchAll = async () => {
    try {
      const [v, s] = await Promise.all([api.ollama.version(), api.ollama.serverUpdateStatus()])
      setVersion(v)
      setUpdateStatus(s)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : "No se pudo consultar Ollama")
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void fetchAll()
  }, [])

  useEffect(() => {
    const running =
      updateStatus?.status === "pulling" || updateStatus?.status === "restarting"
    if (running && pollRef.current === null) {
      pollRef.current = setInterval(() => {
        void fetchAll()
      }, POLL_MS)
    } else if (!running && pollRef.current !== null) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
    return () => {
      if (pollRef.current !== null) {
        clearInterval(pollRef.current)
        pollRef.current = null
      }
    }
  }, [updateStatus?.status])

  const handleUpdateServer = async () => {
    if (busy) return
    if (!confirm(
      "Esto pulará la última imagen de ollama/ollama y recreará el contenedor. " +
      "Cualquier modelo cargado actualmente se descargará temporalmente. ¿Continuar?",
    )) return
    setBusy(true)
    try {
      await api.ollama.startServerUpdate()
      toast.success("Actualización de Ollama iniciada")
      await fetchAll()
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Error al iniciar la actualización")
    } finally {
      setBusy(false)
    }
  }

  const handleUpdateAllModels = async () => {
    if (bulkBusy) return
    if (!confirm("Se encolará un re-pull de todos los modelos actualizables. ¿Continuar?")) return
    setBulkBusy(true)
    try {
      const result = await api.models.updateAll()
      const enq = result.enqueued.length
      const skipped = result.skipped.length
      toast.success(
        `Re-pull encolado: ${enq} modelo${enq === 1 ? "" : "s"}` +
          (skipped > 0 ? ` · ${skipped} omitido${skipped === 1 ? "" : "s"}` : ""),
      )
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Error al actualizar modelos")
    } finally {
      setBulkBusy(false)
    }
  }

  if (loading) {
    return <div className="h-32 animate-pulse rounded-lg border border-border bg-muted/30" />
  }

  if (error) {
    return (
      <div className="flex items-start gap-2 rounded-lg border border-red-500/40 bg-red-500/10 p-4 text-sm text-red-100">
        <AlertCircle size={16} className="mt-0.5 shrink-0" />
        <div>
          <span className="font-semibold">Ollama:</span> {error}
        </div>
      </div>
    )
  }

  const current = version?.current ?? "—"
  const latest = version?.latest ?? "—"
  const updateAvailable = Boolean(version?.updateAvailable)
  const status = updateStatus?.status ?? "idle"
  const inProgress = status === "pulling" || status === "restarting"

  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold">Servidor Ollama</h2>
          <p className="text-sm text-muted-foreground">
            Daemon que gestiona los modelos GGUF de Ollama y los pulls compartidos con llama.cpp.
          </p>
        </div>
        <button
          type="button"
          onClick={() => void fetchAll()}
          className="inline-flex items-center gap-1.5 rounded-md border border-border px-2.5 py-1 text-xs text-muted-foreground hover:bg-muted hover:text-foreground"
          title="Recargar"
        >
          <RefreshCw size={12} />
          Recargar
        </button>
      </div>

      <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-3">
        <div>
          <div className="text-xs uppercase text-muted-foreground">Instalada</div>
          <div className="mt-1 font-mono text-sm">{current}</div>
        </div>
        <div>
          <div className="text-xs uppercase text-muted-foreground">Última disponible</div>
          <div className="mt-1 font-mono text-sm">{latest}</div>
        </div>
        <div>
          <div className="text-xs uppercase text-muted-foreground">Estado</div>
          <div className="mt-1 text-sm">
            {updateAvailable && status === "idle" && (
              <span className="inline-flex items-center gap-1 rounded-md bg-amber-500/15 px-2 py-0.5 text-xs text-amber-300">
                <ArrowUpCircle size={12} /> actualización disponible
              </span>
            )}
            {!updateAvailable && status === "idle" && (
              <span className="inline-flex items-center gap-1 rounded-md bg-emerald-500/15 px-2 py-0.5 text-xs text-emerald-300">
                <CheckCircle2 size={12} /> al día
              </span>
            )}
            {inProgress && (
              <span className="inline-flex items-center gap-1 rounded-md bg-blue-500/15 px-2 py-0.5 text-xs text-blue-300">
                <Loader2 size={12} className="animate-spin" />
                {status === "pulling" ? "descargando imagen…" : "reiniciando…"}
              </span>
            )}
            {status === "done" && (
              <span className="inline-flex items-center gap-1 rounded-md bg-emerald-500/15 px-2 py-0.5 text-xs text-emerald-300">
                <CheckCircle2 size={12} /> última actualización completada
              </span>
            )}
            {status === "error" && (
              <span className="inline-flex items-center gap-1 rounded-md bg-red-500/15 px-2 py-0.5 text-xs text-red-300">
                <AlertCircle size={12} /> error: {updateStatus?.detail ?? "desconocido"}
              </span>
            )}
          </div>
        </div>
      </div>

      <div className="mt-4 flex flex-wrap gap-2">
        <button
          type="button"
          onClick={() => void handleUpdateServer()}
          disabled={busy || inProgress}
          className="inline-flex items-center gap-1.5 rounded-md border border-border bg-primary px-3 py-1.5 text-sm text-primary-foreground hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {busy || inProgress ? <Loader2 size={14} className="animate-spin" /> : <ArrowUpCircle size={14} />}
          Actualizar daemon
        </button>
        <button
          type="button"
          onClick={() => void handleUpdateAllModels()}
          disabled={bulkBusy}
          className="inline-flex items-center gap-1.5 rounded-md border border-border px-3 py-1.5 text-sm hover:bg-muted disabled:cursor-not-allowed disabled:opacity-50"
        >
          {bulkBusy ? <Loader2 size={14} className="animate-spin" /> : <RefreshCw size={14} />}
          Actualizar todos los modelos
        </button>
      </div>
    </div>
  )
}
