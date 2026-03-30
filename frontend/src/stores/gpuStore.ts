import { create } from "zustand"
import type { GPUState } from "@/types"

export interface GpuHistoryPoint {
  t: number          // timestamp ms
  util: number       // utilizationPct
  vramPct: number    // usedVramMb / totalVramMb * 100
  powerPct: number   // powerDrawW / powerLimitW * 100
}

const HISTORY_LEN = 1800 // 60 min at 2s poll interval

interface GPUStore {
  gpus: GPUState[]
  lastUpdated: Date | null
  history: Record<number, GpuHistoryPoint[]>  // gpuIndex → ring buffer
  setGpus: (gpus: GPUState[]) => void
}

export const useGpuStore = create<GPUStore>((set) => ({
  gpus: [],
  lastUpdated: null,
  history: {},
  setGpus: (gpus) =>
    set((state) => {
      const now = Date.now()
      const history = { ...state.history }
      for (const gpu of gpus) {
        const prev = history[gpu.index] ?? []
        const point: GpuHistoryPoint = {
          t: now,
          util: gpu.utilizationPct,
          vramPct: gpu.totalVramMb > 0 ? (gpu.usedVramMb / gpu.totalVramMb) * 100 : 0,
          powerPct: gpu.powerLimitW > 0 ? (gpu.powerDrawW / gpu.powerLimitW) * 100 : 0,
        }
        history[gpu.index] = [...prev.slice(-(HISTORY_LEN - 1)), point]
      }
      return { gpus, lastUpdated: new Date(), history }
    }),
}))
