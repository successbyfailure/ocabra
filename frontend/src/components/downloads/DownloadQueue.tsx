import { useEffect, useState } from "react"
import * as Dialog from "@radix-ui/react-dialog"
import { Download, X } from "lucide-react"
import { toast } from "sonner"
import { api } from "@/api/client"
import { useDownloadStore } from "@/stores/downloadStore"
import type { DownloadJob } from "@/types"

interface DownloadQueueProps {
  jobs?: DownloadJob[]
  onCancel?: (job: DownloadJob) => Promise<void>
}

export function DownloadQueue({ jobs: jobsProp, onCancel: onCancelProp }: DownloadQueueProps = {}) {
  const storeJobs = useDownloadStore((state) => state.jobs)
  const setJobs = useDownloadStore((state) => state.setJobs)
  const updateJob = useDownloadStore((state) => state.updateJob)
  const [open, setOpen] = useState(false)
  const [targetJob, setTargetJob] = useState<DownloadJob | null>(null)

  const jobs = jobsProp ?? storeJobs

  // Self-managed polling when no parent supplies jobs (i.e. global mount).
  useEffect(() => {
    if (jobsProp !== undefined) return
    const refresh = async () => {
      try {
        const list = await api.downloads.list()
        setJobs(list)
      } catch {
        // silent — sidebar component
      }
    }
    void refresh()
    const timer = window.setInterval(() => {
      void refresh()
    }, 5_000)
    return () => window.clearInterval(timer)
  }, [jobsProp, setJobs])

  useEffect(() => {
    const activeJobs = jobs.filter((job) => job.status === "queued" || job.status === "downloading")
    if (activeJobs.length === 0) return

    const streams = activeJobs.map((job) => {
      const source = api.downloads.streamProgress(job.jobId)
      source.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data) as {
            progress_pct?: number
            progressPct?: number
            speed_mb_s?: number
            speedMbS?: number
            eta_seconds?: number
            etaSeconds?: number
            status?: DownloadJob["status"]
            error?: string
          }
          const pct = Number(payload.progress_pct ?? payload.progressPct ?? 0)
          updateJob(job.jobId, {
            progressPct: pct,
            speedMbS: payload.speed_mb_s ?? payload.speedMbS ?? null,
            etaSeconds: payload.eta_seconds ?? payload.etaSeconds ?? null,
            status: payload.status ?? (pct >= 100 ? "completed" : "downloading"),
            error: payload.error ?? null,
            completedAt: pct >= 100 ? new Date().toISOString() : null,
          })
        } catch {
          // ignore malformed SSE payloads
        }
      }
      return source
    })

    return () => {
      streams.forEach((stream) => stream.close())
    }
  }, [jobs, updateJob])

  const handleCancel = async (job: DownloadJob) => {
    if (onCancelProp) {
      await onCancelProp(job)
      return
    }
    try {
      await api.downloads.cancel(job.jobId)
      toast.success("Descarga cancelada")
      const list = await api.downloads.list()
      setJobs(list)
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "No se pudo cancelar")
    }
  }

  // Hide cancelled and stale finished jobs (older than 1h) so the widget
  // doesn't accumulate forever when mounted globally.
  const cutoff = Date.now() - 60 * 60 * 1000
  const visibleJobs = jobs.filter((job) => {
    if (job.status === "cancelled") return false
    if (job.status === "completed" || job.status === "failed") {
      const finished = job.completedAt ? Date.parse(job.completedAt) : NaN
      if (Number.isFinite(finished) && finished < cutoff) return false
    }
    return true
  })
  const activeCount = visibleJobs.filter(
    (job) => job.status === "queued" || job.status === "downloading",
  ).length
  const updatesActive = visibleJobs.some(
    (job) => job.kind === "update" && (job.status === "queued" || job.status === "downloading"),
  )
  if (visibleJobs.length === 0) return null

  return (
    <div className="fixed bottom-4 right-4 z-30 w-[90vw] max-w-sm rounded-lg border border-border bg-card/95 p-3 shadow-xl backdrop-blur">
      <button
        type="button"
        onClick={() => setOpen((curr) => !curr)}
        className="mb-3 flex w-full items-center justify-between rounded-md border border-border px-3 py-2 text-sm"
      >
        <span className="inline-flex items-center gap-2 font-medium">
          <Download size={15} />
          {updatesActive ? "Descargas y actualizaciones" : "Cola de descargas"} ({activeCount}
          {activeCount !== visibleJobs.length ? `/${visibleJobs.length}` : ""})
        </span>
        <span className="text-xs text-muted-foreground">{open ? "Ocultar" : "Mostrar"}</span>
      </button>

      {open && (
        <div className="max-h-72 space-y-2 overflow-y-auto pr-1">
          {visibleJobs.map((job) => {
            const isUpdate = job.kind === "update"
            const kindLabel = isUpdate ? "Actualizando" : "Instalando"
            const kindBadgeClass = isUpdate
              ? "bg-amber-500/15 text-amber-300 border-amber-500/30"
              : "bg-blue-500/15 text-blue-300 border-blue-500/30"
            const barColor = isUpdate ? "bg-amber-500" : "bg-blue-500"
            return (
              <div key={job.jobId} className="rounded-md border border-border bg-background/60 p-2">
                <div className="mb-1 flex items-start justify-between gap-2">
                  <div className="min-w-0">
                    <p className="truncate text-sm font-medium">{job.modelRef}</p>
                    <div className="mt-0.5 flex items-center gap-2 text-xs text-muted-foreground">
                      <span
                        className={`inline-block rounded border px-1.5 py-0.5 text-[10px] font-medium uppercase ${kindBadgeClass}`}
                      >
                        {kindLabel}
                      </span>
                      <span>{job.source}</span>
                    </div>
                  </div>
                  {(job.status === "queued" || job.status === "downloading") && (
                    <button
                      type="button"
                      onClick={() => setTargetJob(job)}
                      className="rounded p-1 text-red-300 hover:bg-red-500/10"
                      aria-label="Cancel download"
                    >
                      <X size={14} />
                    </button>
                  )}
                </div>

                <div className="mb-1 h-1.5 overflow-hidden rounded-full bg-muted">
                  <div
                    className={`h-full transition-all ${barColor}`}
                    style={{ width: `${Math.min(100, Math.max(0, job.progressPct))}%` }}
                  />
                </div>

                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>{job.status}</span>
                  <span>
                    {Math.round(job.progressPct)}% · {job.speedMbS ? `${job.speedMbS.toFixed(1)} MB/s` : "--"}
                  </span>
                </div>
                {job.error && <p className="mt-1 text-xs text-red-300">{job.error}</p>}
              </div>
            )
          })}
        </div>
      )}

      <Dialog.Root open={Boolean(targetJob)} onOpenChange={(next) => !next && setTargetJob(null)}>
        <Dialog.Portal>
          <Dialog.Overlay className="fixed inset-0 z-40 bg-black/60" />
          <Dialog.Content className="fixed left-1/2 top-1/2 z-50 w-[95vw] max-w-md -translate-x-1/2 -translate-y-1/2 rounded-lg border border-border bg-card p-5">
            <Dialog.Title className="text-lg font-semibold">Cancelar descarga</Dialog.Title>
            <Dialog.Description className="mt-1 text-sm text-muted-foreground">
              Esta accion detendra la descarga en curso.
            </Dialog.Description>
            <div className="mt-4 flex justify-end gap-2">
              <Dialog.Close className="rounded-md border border-border px-3 py-2 text-sm hover:bg-muted">
                Volver
              </Dialog.Close>
              <button
                type="button"
                onClick={async () => {
                  if (!targetJob) return
                  await handleCancel(targetJob)
                  setTargetJob(null)
                }}
                className="rounded-md border border-red-500/40 px-3 py-2 text-sm text-red-200 hover:bg-red-500/20"
              >
                Cancelar descarga
              </button>
            </div>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
    </div>
  )
}
