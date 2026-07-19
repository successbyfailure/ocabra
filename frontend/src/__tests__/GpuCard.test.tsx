// @vitest-environment jsdom
import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { GpuCard } from "@/components/gpu/GpuCard"
import type { GPUState } from "@/types"

const gpu: GPUState = {
  index: 0,
  name: "RTX 3090",
  totalVramMb: 24576,
  freeVramMb: 2000,
  usedVramMb: 22576,
  utilizationPct: 92,
  temperatureC: 84,
  powerDrawW: 320,
  powerLimitW: 350,
  lockedVramMb: 4096,
  processes: [
    {
      pid: 1234,
      processName: "python",
      processType: "compute",
      usedVramMb: 1024,
    },
  ],
}

describe("GpuCard", () => {
  it("renders high temperature and token stats", () => {
    render(
      <GpuCard
        gpu={gpu}
        tokenStats={{ gpuIndex: 0, inputTokens: 1_000_000, outputTokens: 500_000 }}
      />,
    )

    expect(screen.getByText("RTX 3090")).toBeTruthy()
    expect(screen.getByText("84°C")).toBeTruthy()
    expect(screen.getByText("1.5M")).toBeTruthy()
    expect(screen.getByText("1.0M")).toBeTruthy()
    expect(screen.getByText("500k")).toBeTruthy()
    expect(screen.getByText("Procesos (1)")).toBeTruthy()
    expect(screen.getByText("1234 · python")).toBeTruthy()
    expect(screen.getByText("compute · 1,024 MB")).toBeTruthy()
  })
})
