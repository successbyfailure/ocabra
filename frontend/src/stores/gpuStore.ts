import { create } from "zustand"
import type { GPUState } from "@/types"

interface GPUStore {
  gpus: GPUState[]
  lastUpdated: Date | null
  setGpus: (gpus: GPUState[]) => void
}

export const useGpuStore = create<GPUStore>((set) => ({
  gpus: [],
  lastUpdated: null,
  setGpus: (gpus) => set({ gpus, lastUpdated: new Date() }),
}))
