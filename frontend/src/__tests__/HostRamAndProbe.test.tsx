import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { MemoryBars } from "@/components/model-editor/sections/capacity/Workbench"
import type { GPUState } from "@/types"

const gtx1660: GPUState = {
  index: 0, name: "NVIDIA GeForce GTX 1660", totalVramMb: 6144, freeVramMb: 4844, usedVramMb: 1300,
  utilizationPct: 0, temperatureC: 50, powerDrawW: 9, powerLimitW: 130, lockedVramMb: 0, processes: [],
  editorCapacity: { availableMb: 4600, budgetMb: 4600, systemMb: 1544 },
}
// gemma4:26b en Mapache: 3,7 GB en la GPU y los expertos de 30 capas en RAM.
const memory = [{ gpu: 0, parts: { weights: 2459, kv: 776, overhead: 495 } }]

describe("RAM del nodo en el configurador", () => {
  it("muestra lo que deja residente frente a lo libre", () => {
    render(
      <MemoryBars gpus={[gtx1660]} memory={memory} liveVramByGpu={{}}
        host={{ totalMb: 32000, availableMb: 21000, needMb: 14564, projectedMb: 14564, modelFileMb: 17900, mlock: false, marginMb: 1024, fits: true }} />,
    )
    const bar = screen.getByTestId("host-ram-bar")
    expect(bar.textContent).toContain("14 GB de 21 GB libres")
    expect(bar.querySelector('[data-part="impossible"]')).toBeNull()
    expect(screen.getByTestId("memory-bars").textContent).toContain("Cabe con lo que hay cargado ahora.")
  })

  it("con mlock bloquea el fichero entero y avisa cuando no cabe en la RAM", () => {
    render(
      <MemoryBars gpus={[gtx1660]} memory={memory} liveVramByGpu={{}}
        host={{ totalMb: 32000, availableMb: 16000, needMb: 17900, projectedMb: 14564, modelFileMb: 17900, mlock: true, marginMb: 1024, fits: false }} />,
    )
    const bar = screen.getByTestId("host-ram-bar")
    expect(bar.querySelector('[data-part="host-mlock"]')).toBeTruthy()
    expect(bar.querySelector('[data-part="impossible"]')).toBeTruthy()
    expect(bar.textContent).toContain("mlock")
    expect(screen.getByTestId("memory-bars").textContent).toContain("no en la RAM libre del nodo")
  })
})

const startContextProbe = vi.fn()
vi.mock("@/api/client", () => ({
  api: {
    profiles: {
      getContextProbe: vi.fn().mockRejectedValue(new Error("none")),
      startContextProbe: (...a: unknown[]) => startContextProbe(...a),
      applyContextProbe: vi.fn(),
    },
  },
  request: vi.fn(),
}))

import { ContextProbe } from "@/components/models/ContextProbe"

describe("Sonda en modo contexto objetivo", () => {
  it("un peldaño con expertos en RAM arranca en ese modo y manda el contexto", async () => {
    startContextProbe.mockResolvedValue({
      jobId: "j", profileId: "p", presetKey: "normal", status: "pending", fullValidation: false, measureVision: false,
      targetContext: 32768, gpuIndices: [0], contextIsPool: true, parallelism: 4, steps: [], evicted: [], error: null,
      startedAt: 0, finishedAt: null, result: null,
    })
    const preset = { presetKey: "normal", label: "normal", contextTokens: 22528, placement: { nCpuMoe: 30 }, preferredGpus: [0] }
    render(<ContextProbe profileId="p" preset={preset as never} onApplied={() => {}} />)
    fireEvent.click(screen.getByText("Medir contexto máximo"))
    fireEvent.change(await screen.findByTestId("probe-target"), { target: { value: "32" } })
    fireEvent.click(screen.getByText("Empezar"))
    await waitFor(() =>
      expect(startContextProbe).toHaveBeenCalledWith("p", "normal", { fullValidation: false, measureVision: false, targetContext: 32768 }),
    )
  })
})
