import type { PresetPlacement, RuntimePresetCreate } from "@/types"
import type { Json } from "../../types"
import type { Placement } from "./DevicePlacement"

const SPLIT_MODES = new Set(["layer", "row", "tensor"])
const GPU_KEYS: (keyof Placement)[] = ["marked", "star", "together", "splitWhenNeeded", "ratios", "splitMode"]
const CPU_KEYS: (keyof Placement)[] = ["cpu", "cpuMode", "gpuLayers", "nCpuMoe"]
const touches = (patch: Partial<Placement>, keys: (keyof Placement)[]) => keys.some((k) => k in patch)

/** GPUs que usa a la vez (techo de la gráfica, GPUs resaltadas). */
export function activeGpus(p: Placement): number[] | null {
  if (!p.marked.length) return null
  if (p.marked.length === 1) return p.marked
  if (p.together) return p.marked
  return p.star != null ? [p.star] : null
}

function cpuFrom(gpuLayers: number | null, nCpuMoe: number | null, layers: number | null) {
  const moe = nCpuMoe != null && nCpuMoe > 0
  const partial = gpuLayers != null && gpuLayers > 0 && (layers == null ? gpuLayers < 999 : gpuLayers < layers)
  return {
    cpu: moe || partial || gpuLayers === 0,
    cpuMode: moe ? ("moe" as const) : ("layers" as const),
    gpuLayers: partial || gpuLayers === 0 ? gpuLayers : null,
    nCpuMoe: moe ? nCpuMoe : null,
  }
}

// ── Base de llama.cpp ───────────────────────────────────────────────────────

/** Lee el reparto de la base (`llama_cpp.*` + GPU preferida del modelo).
 *  Sin `split_policy` explícita vale lo de siempre: `split_mode: none` es una
 *  sola GPU; cualquier otra cosa reparte solo si no cabe en ninguna. */
export function readBasePlacement(
  sec: Record<string, Json>,
  preferredGpu: number | null,
  gpuIndices: number[],
  layers: number | null,
  gpuLayersDefault: number | null,
): Placement {
  const disabled = new Set((Array.isArray(sec.disabled_gpus) ? sec.disabled_gpus : []) as number[])
  const gpuLayers = typeof sec.gpu_layers === "number" ? sec.gpu_layers : gpuLayersDefault
  const nCpuMoe = typeof sec.n_cpu_moe === "number" ? sec.n_cpu_moe : null
  const marked = gpuLayers === 0 ? [] : gpuIndices.filter((g) => !disabled.has(g))
  const policy = typeof sec.split_policy === "string" ? sec.split_policy : sec.split_mode === "none" ? "single" : "when_needed"
  return {
    marked,
    star: preferredGpu != null && marked.includes(preferredGpu) ? preferredGpu : null,
    together: policy === "always" && marked.length > 1,
    splitWhenNeeded: policy === "when_needed",
    ratios: Array.isArray(sec.tensor_split) ? (sec.tensor_split as number[]) : null,
    splitMode: typeof sec.split_mode === "string" && SPLIT_MODES.has(sec.split_mode) ? (sec.split_mode as Placement["splitMode"]) : null,
    ...cpuFrom(gpuLayers, nCpuMoe, layers),
  }
}

/** Escribe solo las claves que toca el cambio: mover la ★ no reescribe las
 *  capas, y un modelo sin tocar no gana claves nuevas. */
export function writeBasePlacement(
  setKey: (k: string, v: Json | undefined) => void,
  setPreferred: (g: number | null) => void,
  gpuIndices: number[],
  next: Placement,
  patch: Partial<Placement>,
) {
  if (touches(patch, GPU_KEYS) && next.marked.length) {
    const others = gpuIndices.filter((g) => !next.marked.includes(g))
    setKey("disabled_gpus", others.length ? others : undefined)
    setPreferred(next.marked.length === 1 ? next.marked[0] : next.star)
    const mainPos = next.star != null ? next.marked.indexOf(next.star) : -1
    if (next.marked.length === 1 || (!next.together && !next.splitWhenNeeded)) {
      setKey("split_mode", "none")
      setKey("split_policy", undefined)
      setKey("tensor_split", undefined)
      setKey("main_gpu", undefined)
    } else {
      setKey("split_policy", next.together ? "always" : undefined)
      setKey("split_mode", next.splitMode ?? undefined)
      setKey("tensor_split", next.ratios && next.ratios.length === next.marked.length ? next.ratios : undefined)
      setKey("main_gpu", mainPos > 0 ? mainPos : undefined)
    }
  }
  if (touches(patch, ["marked"]) && !next.marked.length) {
    setKey("gpu_layers", 0)
    setKey("n_cpu_moe", undefined)
    return
  }
  if (touches(patch, CPU_KEYS) || (touches(patch, ["marked"]) && next.gpuLayers === 0)) {
    if (!next.cpu || (next.cpuMode === "layers" && next.gpuLayers === 0 && next.marked.length)) {
      setKey("gpu_layers", undefined)
      setKey("n_cpu_moe", undefined)
    } else if (next.cpuMode === "moe") {
      setKey("gpu_layers", undefined)
      setKey("n_cpu_moe", next.nCpuMoe ?? undefined)
    } else {
      setKey("gpu_layers", next.gpuLayers ?? undefined)
      setKey("n_cpu_moe", undefined)
    }
  }
}

// ── Peldaño ─────────────────────────────────────────────────────────────────

/** Un peldaño guarda sus GPUs en placement_policy/preferred_gpus y el resto
 *  (reparto, CPU) en `placement`. */
export function readPresetPlacement(form: RuntimePresetCreate, gpuIndices: number[], layers: number | null): Placement {
  const pl = form.placement ?? null
  const listed = (form.preferredGpus ?? []).filter((g) => gpuIndices.includes(g))
  const cpu = cpuFrom(pl?.gpuLayers ?? null, pl?.nCpuMoe ?? null, layers)
  let marked = gpuIndices
  let star: number | null = null
  let together = pl?.splitPolicy === "always" && gpuIndices.length > 1
  if (form.placementPolicy === "preferred") star = listed[0] ?? null
  if (form.placementPolicy === "forced" && listed.length) {
    marked = [...listed].sort((a, b) => a - b)
    together = marked.length > 1
    star = together ? null : marked[0]
  }
  if (pl?.gpuLayers === 0) marked = []
  return {
    marked,
    star,
    together,
    splitWhenNeeded: pl?.splitPolicy === "when_needed",
    ratios: pl?.tensorSplit ?? null,
    splitMode: pl?.splitMode ?? null,
    ...cpu,
  }
}

export function writePresetPlacement(form: RuntimePresetCreate, next: Placement, gpuIndices: number[]): RuntimePresetCreate {
  let placementPolicy: RuntimePresetCreate["placementPolicy"] = "auto"
  let preferredGpus: number[] | null = null
  let splitPolicy: PresetPlacement["splitPolicy"] = null
  let tensorSplit: number[] | null = null
  let splitMode: PresetPlacement["splitMode"] = null
  if (next.marked.length === 1) {
    placementPolicy = "forced"
    preferredGpus = next.marked
  } else if (next.marked.length > 1 && next.together) {
    placementPolicy = "forced"
    preferredGpus = next.marked
    splitPolicy = "always"
    tensorSplit = next.ratios && next.ratios.length === next.marked.length ? next.ratios : null
    splitMode = next.splitMode
  } else if (next.marked.length > 1) {
    const all = next.marked.length === gpuIndices.length
    if (next.star != null || !all) {
      placementPolicy = "preferred"
      const first = next.star ?? next.marked[0]
      preferredGpus = [first, ...next.marked.filter((g) => g !== first)]
    }
    if (next.splitWhenNeeded) {
      splitPolicy = "when_needed"
      splitMode = next.splitMode
    }
  }
  const cpuOnly = next.marked.length === 0
  const gpuLayers = cpuOnly ? 0 : next.cpu && next.cpuMode === "layers" && next.gpuLayers ? next.gpuLayers : null
  const nCpuMoe = !cpuOnly && next.cpu && next.cpuMode === "moe" ? next.nCpuMoe : null
  const placement: PresetPlacement = { splitPolicy, splitMode, tensorSplit, gpuLayers, nCpuMoe }
  const empty = Object.values(placement).every((v) => v == null)
  return { ...form, placementPolicy, preferredGpus, placement: empty ? null : placement }
}

// ── bitnet ──────────────────────────────────────────────────────────────────

/** bitnet no excluye GPUs ni reparte: GPU preferida del modelo + capas en GPU. */
export function readBitnetPlacement(sec: Record<string, Json>, preferredGpu: number | null, gpuIndices: number[], layers: number | null, gpuLayersDefault: number | null): Placement {
  const gpuLayers = typeof sec.gpu_layers === "number" ? sec.gpu_layers : gpuLayersDefault
  return {
    marked: gpuLayers === 0 ? [] : gpuIndices,
    star: preferredGpu != null && gpuIndices.includes(preferredGpu) ? preferredGpu : null,
    together: false,
    splitWhenNeeded: false,
    ratios: null,
    splitMode: null,
    ...cpuFrom(gpuLayers, null, layers),
  }
}

export function writeBitnetPlacement(setKey: (k: string, v: Json | undefined) => void, setPreferred: (g: number | null) => void, next: Placement, patch: Partial<Placement>) {
  if ("star" in patch) setPreferred(next.star)
  if (touches(patch, CPU_KEYS)) setKey("gpu_layers", next.cpu && next.gpuLayers != null ? next.gpuLayers : undefined)
}
