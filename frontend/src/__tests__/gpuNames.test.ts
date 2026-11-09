import { describe, expect, it } from "vitest"
import { gpuLabel, gpuListLabel, shortGpuName } from "@/components/dashboard/gpuNames"
import type { GPUState } from "@/types"

const gpu = (index: number, name: string) => ({ index, name }) as GPUState

describe("nombres de GPU", () => {
  it("quita el fabricante y la marca comercial", () => {
    expect(shortGpuName("NVIDIA GeForce RTX 3060")).toBe("RTX 3060")
    expect(shortGpuName("NVIDIA GeForce GTX 1660 SUPER")).toBe("GTX 1660 SUPER")
    expect(shortGpuName("NVIDIA RTX A6000")).toBe("RTX A6000")
    expect(shortGpuName(undefined)).toBe("")
  })

  it("antepone el índice y une varias con +", () => {
    const gpus = [gpu(0, "NVIDIA GeForce RTX 3060"), gpu(1, "NVIDIA GeForce RTX 3090")]
    expect(gpuLabel(0, gpus)).toBe("#0 RTX 3060")
    expect(gpuLabel(2, gpus)).toBe("#2")
    expect(gpuListLabel([0, 1], gpus)).toBe("#0 RTX 3060 + #1 RTX 3090")
    expect(gpuListLabel([], gpus)).toBe("CPU")
  })
})
