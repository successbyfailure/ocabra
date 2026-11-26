// @vitest-environment jsdom
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { useState } from "react"
import { describe, expect, it, vi } from "vitest"

vi.mock("@/api/client", () => ({
  api: {
    federation: {
      cloud: {
        gpuTypes: vi.fn().mockResolvedValue([
          { id: "NVIDIA RTX 4090", name: "RTX 4090", memory_gb: 24, availability: "NONE" },
          { id: "NVIDIA RTX 3090", name: "RTX 3090", memory_gb: 24, availability: "HIGH" },
        ]),
      },
    },
    backends: { list: vi.fn().mockResolvedValue([{ backendType: "vllm" }, { backendType: "llamacpp_mks" }]) },
  },
}))

import { BenchmarkCloudFields, DEFAULT_CLOUD_SPEC } from "@/pages/benchmarks/BenchmarkCloudFields"
import type { BenchmarkCloudSpec } from "@/types"

let latest: BenchmarkCloudSpec = DEFAULT_CLOUD_SPEC

function Harness() {
  const [value, setValue] = useState(DEFAULT_CLOUD_SPEC)
  latest = value
  return <BenchmarkCloudFields value={value} onChange={setValue} />
}

describe("BenchmarkCloudFields", () => {
  it("guarda las GPUs por orden de preferencia, los backends y el apagado", async () => {
    render(<Harness />)
    await waitFor(() => expect(screen.getByText(/RTX 3090/)).toBeTruthy())
    fireEvent.click(screen.getByText(/RTX 3090/))
    fireEvent.click(screen.getByText(/RTX 4090/))
    expect(latest.gpu_type_ids).toEqual(["NVIDIA RTX 3090", "NVIDIA RTX 4090"])
    expect(screen.getByText(/^1\. RTX 3090/)).toBeTruthy()
    fireEvent.click(await screen.findByText("vllm"))
    expect(latest.backends).toEqual(["vllm"])
    expect(latest.terminate_after).toBe(true)
    fireEvent.click(screen.getByRole("checkbox"))
    expect(latest.terminate_after).toBe(false)
  })
})
