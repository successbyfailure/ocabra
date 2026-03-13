import { useEffect, useMemo, useState } from "react"
import { api } from "@/api/client"
import { GpuCard } from "@/components/gpu/GpuCard"
import { LoadPolicyBadge } from "@/components/models/LoadPolicyBadge"
import { ModelStatusBadge } from "@/components/models/ModelStatusBadge"
import { useWebSocket } from "@/hooks/useWebSocket"
import { useDownloadStore } from "@/stores/downloadStore"
import { useGpuStore } from "@/stores/gpuStore"
import { useModelStore } from "@/stores/modelStore"

export function Dashboard() {
  const [error, setError] = useState<string | null>(null)

  const { connected } = useWebSocket()

  const gpus = useGpuStore((state) => state.gpus)
  const setGpus = useGpuStore((state) => state.setGpus)

  const models = useModelStore((state) => state.models)
  const setModels = useModelStore((state) => state.setModels)
  const unloadModel = useModelStore((state) => state.unloadModel)

  const jobs = useDownloadStore((state) => state.jobs)
  const setJobs = useDownloadStore((state) => state.setJobs)

  const activeModels = useMemo(
    () => Object.values(models).filter((model) => model.status === "loaded" || model.status === "loading"),
    [models],
  )
  const activeDownloads = useMemo(
    () => jobs.filter((job) => job.status === "queued" || job.status === "downloading"),
    [jobs],
  )

  useEffect(() => {
    async function bootstrap() {
      try {
        const [gpuList, modelList, downloadList] = await Promise.all([
          api.gpus.list(),
          api.models.list(),
          api.downloads.list(),
        ])
        setGpus(gpuList)
        setModels(modelList)
        setJobs(downloadList)
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load dashboard data")
      }
    }

    void bootstrap()
  }, [setGpus, setJobs, setModels])

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
