import type { CSSProperties } from "react"
import type { GPUState } from "@/types"

/** Rayado diagonal del banco de trabajo. Un solo dibujo con dos colores:
 *  gris para la memoria que no es de ningún modelo (sistema) y rojo para el
 *  tramo imposible (lo que no cabe con esa configuración). */
const hatch = (rgb: string, strong: number, weak: number) =>
  `repeating-linear-gradient(135deg, rgba(${rgb},${strong}) 0 3px, rgba(${rgb},${weak}) 3px 6px)`

export const HATCH_SYSTEM: CSSProperties = { backgroundImage: hatch("148,163,184", 0.6, 0.15) }
export const HATCH_IMPOSSIBLE: CSSProperties = { backgroundImage: hatch("239,68,68", 0.8, 0.08) }

// Sin datos del servidor (un nodo más viejo), una aproximación: lo no
// atribuido a procesos, si es verosímil, más los procesos gráficos (el
// escritorio: ningún modelo lo puede desalojar) y el margen que guarda el
// planificador. Antes los procesos gráficos contaban como «otros modelos» y
// el editor decía «cabe desalojando» para una carga que luego fallaba.
const DEFAULT_SYSTEM_MB = 512
const MAX_PLAUSIBLE_SYSTEM_MB = 1024
const SCHEDULER_BUFFER_MB = 512

export interface GpuOccupancy {
  /** Lo que ningún modelo puede usar: reserva del driver, contextos CUDA,
   *  escritorio y el margen del planificador. */
  system: number
  /** Otros modelos cargados (se pueden desalojar). */
  others: number
  /** Lo que ocupa ahora mismo este modelo. */
  own: number
}

/** Reparte la memoria de una GPU en sistema / otros / este modelo. Con
 *  `editorCapacity` (del servidor) cuadra con lo que concede el planificador:
 *  total − sistema − otros = lo que tendría este modelo ahora. */
export function gpuOccupancy(g: GPUState, ownMb: number): GpuOccupancy {
  const own = Math.max(0, ownMb)
  const cap = g.editorCapacity
  if (cap) {
    const system = Math.max(0, Math.min(g.totalVramMb, cap.systemMb))
    return { system, others: Math.max(0, cap.budgetMb - cap.availableMb), own }
  }
  const used = Math.max(0, g.usedVramMb)
  const procs = g.processes ?? []
  const computePids = new Set(procs.filter((p) => p.processType === "compute").map((p) => p.pid))
  const desktop = procs
    .filter((p) => p.processType === "graphics" && !computePids.has(p.pid))
    .reduce((a, p) => a + Math.max(0, p.usedVramMb || 0), 0)
  // Un mismo pid puede salir como compute y graphics: se cuenta una vez.
  const seen = new Map<number, number>()
  for (const p of procs) seen.set(p.pid, Math.max(seen.get(p.pid) ?? 0, Math.max(0, p.usedVramMb || 0)))
  const attributed = Math.max(own, [...seen.values()].reduce((a, b) => a + b, 0))
  const unattributed = used - attributed
  const reserve =
    unattributed >= 0 && unattributed <= MAX_PLAUSIBLE_SYSTEM_MB ? unattributed : Math.min(DEFAULT_SYSTEM_MB, Math.max(0, used - own - desktop))
  const system = Math.min(g.totalVramMb, reserve + desktop + SCHEDULER_BUFFER_MB)
  return { system, others: Math.max(0, used - own - reserve - desktop), own }
}

/** Las GPUs con el hueco que da el servidor en la vista previa, si lo da. */
export function withCapacity(gpus: GPUState[], capacity: { index: number; availableMb: number; budgetMb: number; systemMb: number }[] | undefined | null): GPUState[] {
  if (!capacity?.length) return gpus
  const by = new Map(capacity.map((c) => [c.index, c]))
  return gpus.map((g) => {
    const c = by.get(g.index)
    return c ? { ...g, editorCapacity: { availableMb: c.availableMb, budgetMb: c.budgetMb, systemMb: c.systemMb } } : g
  })
}
