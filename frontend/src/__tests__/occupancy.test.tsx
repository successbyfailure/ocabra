import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { gpuOccupancy, withCapacity } from "@/components/model-editor/sections/capacity/occupancy"
import { MemoryBars } from "@/components/model-editor/sections/capacity/Workbench"
import { SegmentBar } from "@/components/model-editor/sections/capacity/DevicePlacement"
import type { GPUState } from "@/types"

// La 1660 de Mapache: 1.268 MB usados, casi todo el escritorio (procesos
// gráficos); un pid sale dos veces (compute y graphics).
const gtx1660: GPUState = {
  index: 0, name: "NVIDIA GeForce GTX 1660", totalVramMb: 6144, freeVramMb: 4875, usedVramMb: 1268,
  utilizationPct: 0, temperatureC: 50, powerDrawW: 9, powerLimitW: 130, lockedVramMb: 0,
  processes: [
    { pid: 1865, processName: null, processType: "graphics", usedVramMb: 428 },
    { pid: 632300, processName: null, processType: "graphics", usedVramMb: 124 },
    { pid: 3603882, processName: null, processType: "compute", usedVramMb: 65 },
    { pid: 3603882, processName: null, processType: "graphics", usedVramMb: 65 },
    { pid: 343072, processName: null, processType: "graphics", usedVramMb: 263 },
  ],
}

describe("gpuOccupancy", () => {
  it("counts a desktop as system, not as evictable models", () => {
    const occ = gpuOccupancy(gtx1660, 0)
    // 815 de escritorio + 388 sin atribuir + 512 de margen del planificador.
    expect(occ.system).toBe(815 + 388 + 512)
    expect(occ.others).toBe(65)
  })

  it("uses the server's room when the preview gives it", () => {
    const [g] = withCapacity([gtx1660], [{ index: 0, totalMb: 6144, availableMb: 4363, budgetMb: 4364, systemMb: 1780 }])
    const occ = gpuOccupancy(g, 0)
    expect(occ.system).toBe(1780)
    expect(g.totalVramMb - occ.system - occ.others).toBe(4363)
  })
})

describe("memory verdict against the scheduler", () => {
  it("gemma4 on Mapache (4.486 MB) does not fit, even evicting", () => {
    const [g] = withCapacity([gtx1660], [{ index: 0, totalMb: 6144, availableMb: 4363, budgetMb: 4364, systemMb: 1780 }])
    render(<MemoryBars gpus={[g]} memory={[{ gpu: 0, parts: { weights: 2474, kv: 346, fixed: 1266, overhead: 400 } }]} liveVramByGpu={{}} />)
    expect(screen.getByTestId("memory-bars").textContent).toContain("No cabe en estas GPUs.")
  })
})

describe("SegmentBar", () => {
  it("hatches the part of a GPU segment that cannot fit", () => {
    render(
      <SegmentBar
        editable={false}
        onChange={() => {}}
        segments={[
          { key: "gpu0", label: "#0", color: "#0f0", share: 0.3, impossible: 0.25, impossibleMb: 122 },
          { key: "cpu", label: "CPU", color: "#999", share: 0.7 },
        ]}
      />,
    )
    const hatch = screen.getByTestId("segment-bar").querySelector('[data-part="impossible"]') as HTMLElement
    expect(hatch.style.width).toBe("25%")
    expect(hatch.style.backgroundImage).toContain("repeating-linear-gradient")
    expect(hatch.title).toContain("122 MB")
  })
})
