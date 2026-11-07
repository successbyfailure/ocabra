// @vitest-environment jsdom
import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { BackendIcons, mergeBackendItems } from "@/components/models/BackendIcons"
import type { CompatibleBackend, ModelDeployment } from "@/types"

const deployment = (overrides: Partial<ModelDeployment>): ModelDeployment => ({
  modelId: "llamacpp_mks/gemma4:26b",
  backend: "llamacpp_mks",
  isDefault: false,
  installed: true,
  status: "configured",
  loadPolicy: "on_demand",
  pinPermanent: false,
  artifact: null,
  profiles: [],
  ...overrides,
})

describe("BackendIcons", () => {
  const deployments = [
    deployment({ modelId: "ollama/gemma4:26b", backend: "ollama" }),
    deployment({ isDefault: true }),
  ]
  const compatible: CompatibleBackend[] = [
    { backend: "llamacpp_mks", status: "compatible", installed: true, reason: "GGUF", deployed: true },
    { backend: "llama_cpp", status: "compatible", installed: true, reason: "GGUF del tag", deployed: false },
    { backend: "bitnet", status: "probable", installed: true, reason: "GGUF ternario/1-bit", deployed: false },
    { backend: "vllm", status: "compatible", installed: false, reason: "", deployed: false },
    { backend: "ollama", status: "compatible", installed: true, reason: "tag", deployed: true },
  ]

  it("merges deployments (default first) with compatible backends, each once", () => {
    const items = mergeBackendItems(deployments, compatible)
    expect(items.map((item) => item.backend)).toEqual([
      "llamacpp_mks",
      "ollama",
      "llama_cpp",
      "bitnet",
      "vllm",
    ])
    expect(items[0]).toMatchObject({ isDefault: true, deployed: true, modelId: "llamacpp_mks/gemma4:26b" })
  })

  it("distinguishes default, deployed, compatible, probable and not installed", () => {
    const { container } = render(<BackendIcons items={mergeBackendItems(deployments, compatible)} />)
    const kind = (backend: string) =>
      container.querySelector(`[data-backend="${backend}"]`)?.getAttribute("data-state-kind")

    expect(kind("llamacpp_mks")).toBe("default")
    expect(kind("ollama")).toBe("deployed")
    expect(kind("llama_cpp")).toBe("compatible")
    expect(kind("bitnet")).toBe("probable")
    expect(kind("vllm")).toBe("not-installed")

    // ★ only on the default.
    expect(container.querySelector('[data-backend="llamacpp_mks"]')?.textContent).toContain("★")
    expect(container.querySelector('[data-backend="ollama"]')?.textContent).not.toContain("★")

    // Tooltip text is mirrored in the accessible name.
    expect(screen.getByLabelText(/bitnet · probablemente compatible · GGUF ternario\/1-bit/)).toBeTruthy()
    expect(screen.getByLabelText(/vllm · compatible · no instalado/)).toBeTruthy()
    expect(screen.getByLabelText(/llama.cpp mks · por defecto · desplegado/)).toBeTruthy()
  })

  it("only deployed backends are clickable", () => {
    const onSelect = vi.fn()
    render(<BackendIcons items={mergeBackendItems(deployments, compatible)} onSelect={onSelect} />)

    const buttons = screen.getAllByRole("listitem").filter((el) => el.tagName === "BUTTON")
    expect(buttons.map((el) => el.getAttribute("data-backend"))).toEqual(["llamacpp_mks", "ollama"])

    fireEvent.click(buttons[1])
    expect(onSelect).toHaveBeenCalledWith(expect.objectContaining({ modelId: "ollama/gemma4:26b" }))
  })

  it("renders the empty label when there is nothing to show", () => {
    render(<BackendIcons items={[]} emptyLabel="sin backends" />)
    expect(screen.getByText("sin backends")).toBeTruthy()
  })
})
