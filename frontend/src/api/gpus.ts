// Cliente tipado de /ocabra/* — dominio «gpus» (REF-15).
// Extraído de api/client.ts; usa la capa HTTP única de api/http.ts.

import { request } from "@/api/http"
import { isRecord } from "@/api/_helpers"
import type {
  GPUState,
} from "@/types"

export function toGpuState(raw: unknown): GPUState {
  const data = isRecord(raw) ? raw : {};
  return {
    index: Number(data.index ?? 0),
    name: String(data.name ?? "GPU"),
    totalVramMb: Number(data.total_vram_mb ?? data.totalVramMb ?? 0),
    freeVramMb: Number(data.free_vram_mb ?? data.freeVramMb ?? 0),
    usedVramMb: Number(data.used_vram_mb ?? data.usedVramMb ?? 0),
    utilizationPct: Number(data.utilization_pct ?? data.utilizationPct ?? 0),
    temperatureC: Number(data.temperature_c ?? data.temperatureC ?? 0),
    powerDrawW: Number(data.power_draw_w ?? data.powerDrawW ?? 0),
    powerLimitW: Number(data.power_limit_w ?? data.powerLimitW ?? 0),
    lockedVramMb: Number(data.locked_vram_mb ?? data.lockedVramMb ?? 0),
    processes: Array.isArray(data.processes)
      ? data.processes.map((processRaw) => {
          const process = isRecord(processRaw) ? processRaw : {};
          return {
            pid: Number(process.pid ?? 0),
            processName: (process.process_name ??
              process.processName ??
              null) as string | null,
            processType: String(
              process.process_type ?? process.processType ?? "compute",
            ) as "compute" | "graphics",
            usedVramMb: Number(process.used_vram_mb ?? process.usedVramMb ?? 0),
          };
        })
      : [],
  };
}


export const gpusApiModule = {
  gpus: {
    list: async () =>
      (await request<unknown[]>("GET", "/ocabra/gpus")).map(toGpuState),
  },
};
