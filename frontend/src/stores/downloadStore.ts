import { create } from "zustand"
import type { DownloadJob } from "@/types"

interface DownloadStore {
  jobs: DownloadJob[]
  setJobs: (jobs: DownloadJob[]) => void
  addJob: (job: DownloadJob) => void
  updateJob: (jobId: string, patch: Partial<DownloadJob>) => void
}

export const useDownloadStore = create<DownloadStore>((set) => ({
  jobs: [],
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
              ...patch,
            },
          ],
    })),
}))
