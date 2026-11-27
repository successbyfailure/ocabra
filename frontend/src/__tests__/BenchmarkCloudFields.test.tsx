// @vitest-environment jsdom
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { useState } from "react"
import { describe, expect, it, vi } from "vitest"

vi.mock("@/api/client", () => ({
  api: {
    federation: {
      cloud: {
        gpuTypes: vi.fn().mockResolvedValue([
          { id: "4090", name: "RTX 4090", memory_gb: 24, availability: "HIGH", community: true, secure: true, price_community: 0.34, price_secure: 0.69 },
          { id: "3090", name: "RTX 3090", memory_gb: 24, availability: "LOW", community: true, secure: false, price_community: 0.22, price_secure: null },
          { id: "A100", name: "A100", memory_gb: 80, availability: "NONE", community: true, secure: true, price_community: 1.2, price_secure: 1.6 },
          { id: "L40", name: "L40", memory_gb: 48, availability: "HIGH", community: false, secure: true, price_community: null, price_secure: 0.75 },
        ]),
        getRunpod: vi.fn().mockResolvedValue({ settings: { cloud_type: "COMMUNITY", container_disk_gb: 80 } }),
        replicableModels: vi.fn().mockResolvedValue([
          { model_id: "llamacpp_mks/gemma4:12b", display_name: "g", size_bytes: 8 * 1024 ** 3, replicable: true, loaded: false, reason: null },
        ]),
      },
    },
    backends: { list: vi.fn().mockResolvedValue([{ backendType: "vllm" }]) },
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
  it("combina ambas nubes y conserva las GPU que igualan el mínimo de VRAM", async () => {
    render(<Harness />)
    fireEvent.click(await screen.findByText(/RTX 3090/))
    fireEvent.change(screen.getByLabelText("Nube"), { target: { value: "BOTH" } })
    expect(screen.getByText(/RTX 4090.*Community 0.34.*Secure 0.69/)).toBeTruthy()
    fireEvent.click(screen.getByText(/L40.*Secure/))
    expect(latest.cloud_type).toBe("BOTH")
    expect(latest.gpu_type_ids).toEqual(["3090", "L40"])
    const minimum = screen.getByLabelText(/VRAM mínima por GPU/)
    fireEvent.change(minimum, { target: { value: "24" } })
    expect(screen.getByText(/RTX 3090/)).toBeTruthy()
    expect(latest.gpu_type_ids).toEqual(["3090", "L40"])
    fireEvent.change(minimum, { target: { value: "25" } })
    expect(screen.queryByText(/RTX 3090/)).toBeNull()
    expect(screen.queryByText(/RTX 4090/)).toBeNull()
    expect(latest.gpu_type_ids).toEqual(["L40"])
  })

  it("solo ofrece lo que RunPod tiene ahora y respeta el orden de preferencia", async () => {
    render(<Harness />)
    await waitFor(() => expect(screen.getByText(/RTX 3090/)).toBeTruthy())
    expect(screen.queryByText(/A100/)).toBeNull() // agotada
    fireEvent.click(screen.getByText(/RTX 3090/))
    fireEvent.click(screen.getByText(/RTX 4090/))
    expect(latest.gpu_type_ids).toEqual(["3090", "4090"])
    expect(screen.getByText(/^1\. RTX 3090/)).toBeTruthy()
    fireEvent.click(await screen.findByText("vllm"))
    expect(latest.backends).toEqual(["vllm"])
    fireEvent.click(screen.getByRole("checkbox"))
    expect(latest.terminate_after).toBe(false)
  })

  it("al pasar a Secure se caen las GPUs que esa nube no ofrece", async () => {
    render(<Harness />)
    await waitFor(() => expect(screen.getByText(/RTX 3090/)).toBeTruthy())
    fireEvent.click(screen.getByText(/RTX 3090/))
    fireEvent.click(screen.getByText(/RTX 4090/))
    fireEvent.change(screen.getByDisplayValue(/La de los ajustes/), { target: { value: "SECURE" } })
    await waitFor(() => expect(latest.gpu_type_ids).toEqual(["4090"]))
    expect(screen.queryByText(/RTX 3090/)).toBeNull()
    expect(screen.getByText(/0\.69 \$\/h/)).toBeTruthy()
  })

  it("sugiere el disco por los modelos y lo manda al pedir el pod", async () => {
    function WithModels() {
      const [value, setValue] = useState(DEFAULT_CLOUD_SPEC)
      latest = value
      return <BenchmarkCloudFields value={value} onChange={setValue} modelIds={["llamacpp_mks/gemma4:12b"]} />
    }
    render(<WithModels />)
    fireEvent.click(await screen.findByText("Usar 40 GB"))
    expect(latest.container_disk_gb).toBe(40)
    fireEvent.change(screen.getByLabelText("Disco del pod en GB"), { target: { value: "120" } })
    expect(latest.container_disk_gb).toBe(120)
    fireEvent.click(screen.getByText("Volver a los ajustes"))
    expect(latest.container_disk_gb).toBeNull()
  })
})
