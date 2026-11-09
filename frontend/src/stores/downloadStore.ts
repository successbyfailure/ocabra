import { create } from "zustand"
import type { DownloadJob } from "@/types"

/** Trabajos que se enseñan: fuera los cancelados y los terminados hace más
 *  de una hora, para que la lista no crezca sin fin. */
export function visibleDownloadJobs(jobs: DownloadJob[], nowMs = Date.now()): DownloadJob[] {
  const cutoff = nowMs - 60 * 60 * 1000
  return jobs.filter((job) => {
    if (job.status === "cancelled") return false
    if (job.status === "completed" || job.status === "failed") {
      const finished = job.completedAt ? Date.parse(job.completedAt) : NaN
      if (Number.isFinite(finished) && finished < cutoff) return false
    }
    return true
  })
}

interface DownloadStore {
  jobs: DownloadJob[]
  /** Panel de descargas abierto (lo abre/cierra la píldora de la cabecera). */
  queueOpen: boolean
  setQueueOpen: (open: boolean) => void
  setJobs: (jobs: DownloadJob[]) => void
  addJob: (job: DownloadJob) => void
  updateJob: (jobId: string, patch: Partial<DownloadJob>) => void
}

export const useDownloadStore = create<DownloadStore>((set) => ({
  jobs: [],
  queueOpen: false,
  setQueueOpen: (queueOpen) => set({ queueOpen }),
  setJobs: (jobs) => set({ jobs }),
  addJob: (job) => set((state) => ({ jobs: [...state.jobs, job] })),
  updateJob: (jobId, patch) =>
    set((state) => ({
      jobs: state.jobs.some((j) => j.jobId === jobId)
        ? state.jobs.map((j) => (j.jobId === jobId ? { ...j, ...patch } : j))
        : [
            ...state.jobs,
            {
              jobId,
              source: "huggingface",
              modelRef: "",
              status: "downloading",
              progressPct: 0,
              speedMbS: null,
              etaSeconds: null,
              error: null,
              startedAt: new Date().toISOString(),
              completedAt: null,
              kind: "install",
              ...patch,
            },
          ],
    })),
}))
