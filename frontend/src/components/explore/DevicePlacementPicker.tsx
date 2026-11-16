import type { GPUState } from "@/types"

/** What the install dialog can say before the file exists: a rough check from
 * the download size. The real plan (context per rung, MoE experts in RAM) is
 * computed by the server after the download, from llama.cpp's projection. */
export type PlacementVerdict = "unknown" | "fits_one" | "split" | "needs_cpu" | "cpu_only" | "nothing"

export interface PlacementAssessment {
  verdict: PlacementVerdict
  message: string
  /** False when the download should not start (it can't run where chosen). */
  canInstall: boolean
}

// Weights plus the smallest useful context and CUDA buffers.
const OVERHEAD_FACTOR = 1.08
const MIN_CONTEXT_MB = 1024

export function assessPlacement(
  sizeGb: number | null | undefined,
  gpus: Pick<GPUState, "index" | "name" | "totalVramMb">[],
  selected: number[],
  cpu: boolean,
): PlacementAssessment {
  const chosen = gpus.filter((g) => selected.includes(g.index))
  if (chosen.length === 0 && !cpu) {
    return { verdict: "nothing", message: "Elige al menos una GPU o la CPU.", canInstall: false }
  }
  if (chosen.length === 0) {
    return {
      verdict: "cpu_only",
      message: "Solo CPU: funcionará, pero mucho más lento que en GPU.",
      canInstall: true,
    }
  }
  if (!sizeGb) {
    return {
      verdict: "unknown",
      message: "Tras la descarga se calculan los peldaños con lo que de verdad ocupa.",
      canInstall: true,
    }
  }
  const needMb = sizeGb * 1024 * OVERHEAD_FACTOR + MIN_CONTEXT_MB
  const biggest = chosen.reduce((a, b) => (b.totalVramMb > a.totalVramMb ? b : a))
  const sum = chosen.reduce((acc, g) => acc + g.totalVramMb, 0)
  if (needMb <= biggest.totalVramMb) {
    return {
      verdict: "fits_one",
      message:
        chosen.length > 1
          ? `Cabe en la ${biggest.name}: un peldaño ahí con todo el contexto que quepa y otro en las ${chosen.length} GPU para más contexto.`
          : `Cabe en la ${biggest.name}: un peldaño con todo el contexto que quepa.`,
      canInstall: true,
    }
  }
  if (needMb <= sum) {
    return {
      verdict: "split",
      message: `No cabe en una sola GPU: se repartirá entre las ${chosen.length} elegidas.`,
      canInstall: true,
    }
  }
  if (!cpu) {
    return {
      verdict: "needs_cpu",
      message: "No cabe en las GPU elegidas: marca también la CPU para dejar parte en RAM.",
      canInstall: false,
    }
  }
  return {
    verdict: "needs_cpu",
    message:
      "Parte irá a RAM: en los MoE solo los expertos que haga falta, en los densos las capas que no quepan.",
    canInstall: true,
  }
}

interface Props {
  gpus: GPUState[]
  selected: number[]
  cpu: boolean
  onChange: (next: { selected: number[]; cpu: boolean }) => void
  assessment: PlacementAssessment
}

export function DevicePlacementPicker({ gpus, selected, cpu, onChange, assessment }: Props) {
  const toggle = (index: number, on: boolean) =>
    onChange({
      selected: on ? [...selected, index].sort((a, b) => a - b) : selected.filter((g) => g !== index),
      cpu,
    })
  const tone =
    assessment.canInstall
      ? assessment.verdict === "needs_cpu" || assessment.verdict === "cpu_only"
        ? "text-amber-500"
        : "text-muted-foreground"
      : "text-destructive"
  return (
    <fieldset className="block text-sm text-muted-foreground" data-testid="device-placement">
      <legend className="mb-1">Dónde correrá</legend>
      <div className="flex flex-wrap gap-2">
        {gpus.map((g) => (
          <label
            key={g.index}
            className="flex items-center gap-1.5 rounded-md border border-border px-2 py-1 text-xs text-foreground"
          >
            <input
              type="checkbox"
              checked={selected.includes(g.index)}
              onChange={(e) => toggle(g.index, e.target.checked)}
            />
            #{g.index} {g.name.replace(/^NVIDIA (GeForce )?/, "")} · {(g.totalVramMb / 1024).toFixed(0)} GB
          </label>
        ))}
        <label className="flex items-center gap-1.5 rounded-md border border-border px-2 py-1 text-xs text-foreground">
          <input
            type="checkbox"
            checked={cpu}
            onChange={(e) => onChange({ selected, cpu: e.target.checked })}
          />
          CPU (RAM)
        </label>
      </div>
      <span className={`mt-1 block text-xs ${tone}`} data-testid="device-placement-verdict">
        {assessment.message}
      </span>
    </fieldset>
  )
}
