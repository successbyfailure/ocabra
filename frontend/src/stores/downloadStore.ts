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
      jobs: state.jobs.map((j) => (j.jobId === jobId ? { ...j, ...patch } : j)),
    })),
}))
