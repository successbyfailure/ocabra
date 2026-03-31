import { useEffect, useState } from "react"
import { toast } from "sonner"
import { Cpu, Play, Square, Trash2, RefreshCw } from "lucide-react"
import { api } from "@/api/client"
import { useModelStore } from "@/stores/modelStore"
import type { CompileJob } from "@/types"

function formatDuration(startedAt: string | null, finishedAt: string | null): string {
  if (!startedAt || !finishedAt) return "-"
  const ms = new Date(finishedAt).getTime() - new Date(startedAt).getTime()
  const s = Math.round(ms / 1000)
  if (s < 60) return `${s}s`
  const m = Math.floor(s / 60)
  const rem = s % 60
  return `${m}m ${rem}s`
}

function formatDate(iso: string | null): string {
  if (!iso) return "-"
  return new Date(iso).toLocaleString()
}

function dtypeLabel(dtype: string): string {
  const map: Record<string, string> = { fp16: "FP16", bf16: "BF16", int8: "INT8", fp8: "FP8" }
  return map[dtype] ?? dtype.toUpperCase()
}

export function TrtllmEngines() {
  const [jobs, setJobs] = useState<CompileJob[]>([])
  const [loading, setLoading] = useState(true)
  const [busyEngine, setBusyEngine] = useState<string | null>(null)

  const models = useModelStore((s) => s.models)
  const loadModel = useModelStore((s) => s.loadModel)
  const unloadModel = useModelStore((s) => s.unloadModel)

  const refresh = async () => {
    const all = await api.trtllm.list()
    setJobs(all.filter((j) => j.status === "done"))
  }

  useEffect(() => {
    void refresh()
      .catch(() => toast.error("Error al cargar engines"))
      .finally(() => setLoading(false))
  }, [])

  const handleLoad = async (engineName: string) => {
    const modelId = `tensorrt_llm/${engineName}`
    setBusyEngine(engineName)
    try {
      await loadModel(modelId)
      toast.success("Engine cargado")
      await refresh()
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Error al cargar")
    } finally {
      setBusyEngine(null)
    }
  }

  const handleUnload = async (engineName: string) => {
    const modelId = `tensorrt_llm/${engineName}`
    setBusyEngine(engineName)
    try {
      await unloadModel(modelId)
      toast.success("Engine descargado")
      await refresh()
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Error al descargar")
    } finally {
      setBusyEngine(null)
    }
  }

  const handleDelete = async (engineName: string) => {
    setBusyEngine(engineName)
    try {
      await api.trtllm.deleteEngine(engineName)
      toast.success("Engine eliminado")
      setJobs((prev) => prev.filter((j) => j.engineName !== engineName))
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Error al eliminar")
    } finally {
      setBusyEngine(null)
    }
  }

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold flex items-center gap-2">
            <Cpu size={22} className="text-purple-400" />
            TensorRT-LLM Engines
          </h1>
          <p className="text-muted-foreground">Engines compilados y disponibles para inferencia.</p>
        </div>
        <button
          type="button"
          onClick={() => void refresh()}
          className="rounded-md border border-border p-2 text-muted-foreground hover:bg-muted"
          title="Refrescar"
        >
          <RefreshCw size={16} />
        </button>
      </div>

      {loading ? (
        <div className="space-y-2">
          {Array.from({ length: 3 }).map((_, i) => (
            <div key={`sk-${i}`} className="h-14 animate-pulse rounded-md bg-muted" />
          ))}
        </div>
      ) : jobs.length === 0 ? (
        <div className="rounded-lg border border-border bg-card px-6 py-12 text-center text-sm text-muted-foreground">
          No hay engines compilados aún. Usa el botón{" "}
          <Cpu size={13} className="inline text-purple-400" /> en un modelo vLLM para compilar uno.
        </div>
      ) : (
        <div className="overflow-x-auto rounded-lg border border-border bg-card">
          <table className="min-w-full text-left text-sm">
            <thead className="bg-muted/40 text-xs uppercase text-muted-foreground">
              <tr>
                <th className="px-3 py-2">Engine</th>
                <th className="px-3 py-2">Origen</th>
                <th className="px-3 py-2">GPU(s)</th>
                <th className="px-3 py-2">Dtype</th>
                <th className="px-3 py-2">Compilado</th>
                <th className="px-3 py-2">Tiempo</th>
                <th className="px-3 py-2">Estado</th>
                <th className="px-3 py-2">Acciones</th>
              </tr>
            </thead>
            <tbody>
              {jobs.map((job) => {
                const modelId = `tensorrt_llm/${job.engineName}`
                const modelState = models[modelId]
                const status: string = modelState?.status ?? "unregistered"
                const isLoaded = status === "loaded" || status === "loading"
                const isUnloaded = !modelState || status === "unloaded" || status === "unloading" || status === "configured"
                const busy = busyEngine === job.engineName

                return (
                  <tr key={job.jobId} className="border-b border-border/60">
                    <td className="px-3 py-3">
                      <div className="font-medium text-purple-300">{job.engineName}</div>
                      <div className="text-xs text-muted-foreground">{job.config.maxSeqLen} tok max</div>
                    </td>
                    <td className="px-3 py-3 text-muted-foreground text-xs">
                      {job.sourceModel.replace(/^vllm\//, "")}
                    </td>
                    <td className="px-3 py-3 text-muted-foreground">
                      GPU {job.gpuIndices.join("+")}
                    </td>
                    <td className="px-3 py-3">
                      <span className="rounded bg-purple-500/10 px-1.5 py-0.5 text-xs text-purple-300">
                        {dtypeLabel(job.dtype)}
                      </span>
                    </td>
                    <td className="px-3 py-3 text-xs text-muted-foreground">
                      {formatDate(job.finishedAt)}
                    </td>
                    <td className="px-3 py-3 text-xs text-muted-foreground">
                      {formatDuration(job.startedAt, job.finishedAt)}
                    </td>
                    <td className="px-3 py-3">
                      <StatusBadge status={status} />
                    </td>
                    <td className="px-3 py-3">
                      <div className="flex items-center gap-2">
                        <button
                          type="button"
                          onClick={() => void handleLoad(job.engineName)}
                          disabled={busy || isLoaded}
                          className="rounded-md border border-emerald-500/40 p-1.5 text-emerald-200 disabled:opacity-40"
                          title="Cargar"
                        >
                          <Play size={14} />
                        </button>
                        <button
                          type="button"
                          onClick={() => void handleUnload(job.engineName)}
                          disabled={busy || isUnloaded}
                          className="rounded-md border border-red-500/40 p-1.5 text-red-200 disabled:opacity-40"
                          title="Descargar"
                        >
                          <Square size={14} />
                        </button>
                        <button
                          type="button"
                          onClick={() => void handleDelete(job.engineName)}
                          disabled={busy}
                          className="rounded-md border border-red-500/40 p-1.5 text-red-200 disabled:opacity-40"
                          title="Eliminar engine"
                        >
                          <Trash2 size={14} />
                        </button>
                      </div>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

function StatusBadge({ status }: { status: string }) {
  const styles: Record<string, string> = {
    loaded: "bg-emerald-500/20 text-emerald-300",
    loading: "bg-yellow-500/20 text-yellow-300",
    unloading: "bg-yellow-500/20 text-yellow-300",
    unloaded: "bg-muted text-muted-foreground",
    configured: "bg-muted text-muted-foreground",
    error: "bg-red-500/20 text-red-300",
    unregistered: "bg-zinc-500/20 text-zinc-400",
  }
  return (
    <span className={`rounded px-1.5 py-0.5 text-xs ${styles[status] ?? "bg-muted text-muted-foreground"}`}>
      {status}
    </span>
  )
}
