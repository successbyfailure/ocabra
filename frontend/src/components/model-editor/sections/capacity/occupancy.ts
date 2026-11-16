import type { CSSProperties } from "react"
import type { GPUState } from "@/types"

/** Rayado diagonal del banco de trabajo. Un solo dibujo con dos colores:
 *  gris para la memoria que no es de ningún modelo (sistema) y rojo para el
 *  tramo imposible (lo que no cabe con esa configuración). */
const hatch = (rgb: string, strong: number, weak: number) =>
  `repeating-linear-gradient(135deg, rgba(${rgb},${strong}) 0 3px, rgba(${rgb},${weak}) 3px 6px)`

export const HATCH_SYSTEM: CSSProperties = { backgroundImage: hatch("148,163,184", 0.6, 0.15) }
export const HATCH_IMPOSSIBLE: CSSProperties = { backgroundImage: hatch("239,68,68", 0.8, 0.08) }

// Igual que el backend (`_unattributed_vram_mb` en runtime_preset_capacity.py):
// si lo no atribuido a procesos es inverosímil, se cuenta un fijo y el resto
// se trata como ocupado por otros (procesos que oCabra no ve, p. ej. otros
// contenedores).
const DEFAULT_SYSTEM_MB = 512
const MAX_PLAUSIBLE_SYSTEM_MB = 1024

export interface GpuOccupancy {
  /** Lo que no es de ningún proceso de oCabra y ningún modelo puede usar:
   *  reserva del driver, contextos CUDA, escritorio. */
  system: number
  /** Otros modelos cargados (y procesos fuera de oCabra que sí ocupan). */
  others: number
  /** Lo que ocupa ahora mismo este modelo. */
  own: number
}

/** Reparte la memoria usada de una GPU en sistema / otros / este modelo. */
export function gpuOccupancy(g: GPUState, ownMb: number): GpuOccupancy {
  const own = Math.max(0, ownMb)
  const used = Math.max(0, g.usedVramMb)
  const attributed = Math.max(
    own,
    (g.processes ?? []).reduce((a, p) => a + Math.max(0, p.usedVramMb || 0), 0),
  )
  const unattributed = used - attributed
  const system = unattributed >= 0 && unattributed <= MAX_PLAUSIBLE_SYSTEM_MB ? unattributed : Math.min(DEFAULT_SYSTEM_MB, Math.max(0, used - own))
  return { system, others: Math.max(0, used - own - system), own }
}
