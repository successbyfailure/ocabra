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
  it("renders high temperature/utilization alert state", () => {
    const { container } = render(<GpuCard gpu={gpu} />)

    expect(screen.getByText("RTX 3090")).toBeTruthy()
    expect(screen.getByText("92.0%").className).toContain("animate-pulse")
    expect(screen.getByText("84.0°C").className).toContain("text-orange-400")
    expect(screen.getByText("Processes (nvidia-smi)")).toBeTruthy()
    expect(screen.getByText("1234 · python")).toBeTruthy()
    expect(screen.getByText("compute · 1024 MB")).toBeTruthy()
    expect(container.firstChild).toMatchSnapshot()
  })
})
