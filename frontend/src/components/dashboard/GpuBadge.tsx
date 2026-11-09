import { gpuLabel } from "./gpuNames"
import type { GPUState } from "@/types"

/**
 * Color propio de cada GPU, por índice, igual en todo el dashboard. Elegidos
 * para no confundirse con los colores de estado (verde, azul cielo, violeta,
 * naranja, ámbar, pizarra y rojo de `VISUAL`).
 */
const GPU_COLORS = ["#e879f9", "#2dd4bf", "#a3e635", "#818cf8", "#fb7185", "#facc15"] as const

export function gpuColor(index: number): string {
  return GPU_COLORS[((index % GPU_COLORS.length) + GPU_COLORS.length) % GPU_COLORS.length]
}

/** Burbuja con el nombre de la GPU ("#1 RTX 3090") en su color. */
export function GpuBadge({
  index,
  gpus,
  className = "",
}: {
  index: number
  gpus: GPUState[]
  className?: string
}) {
  const color = gpuColor(index)
  const name = gpus.find((g) => g.index === index)?.name
  return (
    <span
      className={`inline-flex shrink-0 items-center gap-1 rounded-md border px-1.5 py-0 text-[11px] font-medium ${className}`}
      style={{ color, borderColor: `${color}66`, background: `${color}1a` }}
      title={name ?? `GPU ${index}`}
      data-gpu-badge={index}
    >
      <span className="h-1.5 w-1.5 rounded-full" style={{ background: color }} />
      {gpuLabel(index, gpus)}
    </span>
  )
}
