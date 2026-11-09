import { useCallback } from "react"
import { useGpuStore } from "@/stores/gpuStore"
import type { GPUState } from "@/types"

/** Nombre corto de una GPU: "NVIDIA GeForce RTX 3060" -> "RTX 3060". */
export function shortGpuName(name: string | null | undefined): string {
  return (name ?? "")
    .replace(/^NVIDIA\s+/i, "")
    .replace(/^GeForce\s+/i, "")
    .replace(/^AMD\s+Radeon\s+/i, "Radeon ")
    .replace(/\s+/g, " ")
    .trim()
}

/** Etiqueta de una GPU por índice: "#0 RTX 3060" ("#0" si no se conoce). */
export function gpuLabel(index: number, gpus: GPUState[]): string {
  const name = shortGpuName(gpus.find((g) => g.index === index)?.name)
  return name ? `#${index} ${name}` : `#${index}`
}

/** Varias GPUs: "#0 RTX 3060 + #1 RTX 3090"; ninguna: "CPU". */
export function gpuListLabel(indices: number[], gpus: GPUState[]): string {
  if (indices.length === 0) return "CPU"
  return indices.map((i) => gpuLabel(i, gpus)).join(" + ")
}

/** Igual que `gpuLabel`/`gpuListLabel`, leyendo las GPUs del store. */
export function useGpuLabels() {
  const gpus = useGpuStore((s) => s.gpus)
  const one = useCallback((index: number) => gpuLabel(index, gpus), [gpus])
  const list = useCallback((indices: number[]) => gpuListLabel(indices, gpus), [gpus])
  return { one, list }
}
