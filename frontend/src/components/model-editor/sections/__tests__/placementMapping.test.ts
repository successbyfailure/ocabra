import { describe, expect, it } from "vitest"
import type { RuntimePresetCreate } from "@/types"
import type { Json } from "../../types"
import { EMPTY_PRESET } from "../presets/presetLogic"
import { readBasePlacement, readPresetPlacement, writeBasePlacement, writePresetPlacement } from "../capacity/placementMapping"
import type { Placement } from "../capacity/DevicePlacement"

const GPUS = [0, 1]

function applyBase(sec: Record<string, Json>, pref: number | null, next: Placement, patch: Partial<Placement>) {
  const out: Record<string, Json> = { ...sec }
  let preferred = pref
  writeBasePlacement(
    (k, v) => {
      if (v === undefined) delete out[k]
      else out[k] = v
    },
    (g) => (preferred = g),
    GPUS,
    next,
    patch,
  )
  return { sec: out, preferred }
}

describe("base placement", () => {
  it("legacy split_mode none + disabled [0] is 'only GPU 1' and writes back unchanged", () => {
    const sec = { split_mode: "none", disabled_gpus: [0], gpu_layers: 999 }
    const p = readBasePlacement(sec, 1, GPUS, 48, 999)
    expect(p.marked).toEqual([1])
    expect(p.cpu).toBe(false)
    const { sec: out, preferred } = applyBase(sec, 1, { ...p, star: 1 }, { star: 1 })
    expect(out).toEqual(sec)
    expect(preferred).toBe(1)
  })

  it("no split_mode = one GPU, split only when needed", () => {
    const p = readBasePlacement({ tensor_split: [3, 1] }, 1, GPUS, 48, 999)
    expect(p).toMatchObject({ marked: [0, 1], star: 1, together: false, splitWhenNeeded: true, ratios: [3, 1] })
  })

  it("'todas a la vez' writes split_policy always and keeps the ratios", () => {
    const p = readBasePlacement({}, null, GPUS, 48, 999)
    const { sec } = applyBase({}, null, { ...p, together: true, ratios: [60, 40] }, { together: true, ratios: [60, 40] })
    expect(sec).toEqual({ split_policy: "always", tensor_split: [60, 40] })
    expect(readBasePlacement(sec, null, GPUS, 48, 999).together).toBe(true)
  })

  it("moving the star does not touch the CPU keys", () => {
    const sec = { gpu_layers: 999 }
    const p = readBasePlacement(sec, null, GPUS, 48, 999)
    const { sec: out } = applyBase(sec, null, { ...p, star: 0 }, { star: 0 })
    expect(out.gpu_layers).toBe(999)
  })

  it("MoE experts on CPU", () => {
    const p = readBasePlacement({}, null, GPUS, 48, 999)
    const next = { ...p, cpu: true, cpuMode: "moe" as const, nCpuMoe: 12 }
    const { sec } = applyBase({ gpu_layers: 40 }, null, next, { cpu: true, cpuMode: "moe", nCpuMoe: 12 })
    expect(sec).toEqual({ n_cpu_moe: 12 })
    expect(readBasePlacement(sec, null, GPUS, 48, 999)).toMatchObject({ cpu: true, cpuMode: "moe", nCpuMoe: 12 })
  })
})

describe("preset placement", () => {
  const form = (patch: Partial<RuntimePresetCreate>): RuntimePresetCreate => ({ ...EMPTY_PRESET, ...patch })

  it("auto with a star becomes preferred; with when_needed keeps the policy", () => {
    const p = readPresetPlacement(form({}), GPUS, 48)
    const out = writePresetPlacement(form({}), { ...p, star: 1, splitWhenNeeded: true }, GPUS)
    expect(out.placementPolicy).toBe("preferred")
    expect(out.preferredGpus).toEqual([1, 0])
    expect(out.placement).toMatchObject({ splitPolicy: "when_needed" })
    expect(readPresetPlacement(out, GPUS, 48)).toMatchObject({ star: 1, splitWhenNeeded: true, together: false })
  })

  it("together is forced over the marked GPUs with split_policy always", () => {
    const p = readPresetPlacement(form({}), GPUS, 48)
    const out = writePresetPlacement(form({}), { ...p, together: true, ratios: [70, 30] }, GPUS)
    expect(out).toMatchObject({ placementPolicy: "forced", preferredGpus: [0, 1], placement: { splitPolicy: "always", tensorSplit: [70, 30] } })
    expect(readPresetPlacement(out, GPUS, 48)).toMatchObject({ together: true, ratios: [70, 30] })
  })

  it("an untouched auto preset keeps placement null", () => {
    const f = form({ placement: null })
    expect(writePresetPlacement(f, readPresetPlacement(f, GPUS, 48), GPUS)).toEqual(f)
  })

  it("forced single GPU with layers on CPU round-trips", () => {
    const f = form({ placementPolicy: "forced", preferredGpus: [1], placement: { splitPolicy: null, splitMode: null, tensorSplit: null, gpuLayers: 40, nCpuMoe: null } })
    const p = readPresetPlacement(f, GPUS, 48)
    expect(p).toMatchObject({ marked: [1], cpu: true, cpuMode: "layers", gpuLayers: 40 })
    expect(writePresetPlacement(f, p, GPUS)).toEqual(f)
  })

  it("CPU only and back", () => {
    const p = readPresetPlacement(form({}), GPUS, 48)
    const cpuOnly = writePresetPlacement(form({}), { ...p, marked: [], cpu: true }, GPUS)
    expect(cpuOnly.placement?.gpuLayers).toBe(0)
    const back = readPresetPlacement(cpuOnly, GPUS, 48)
    expect(back.marked).toEqual([])
    const again = writePresetPlacement(cpuOnly, { ...back, marked: [1] }, GPUS)
    expect(again).toMatchObject({ placementPolicy: "forced", preferredGpus: [1], placement: null })
  })
})
