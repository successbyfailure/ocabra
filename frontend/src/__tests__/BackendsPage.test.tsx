import { fireEvent, render, screen, within } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"
import { describe, expect, it, vi } from "vitest"
import type { BackendModuleState } from "@/types"

const b = (backendType: string, installStatus: BackendModuleState["installStatus"], displayName = backendType): BackendModuleState => ({
  backendType, displayName, description: `desc ${backendType}`, tags: [], installStatus, installedVersion: null, installedAt: null,
  installSource: null, estimatedSizeMb: 1200, actualSizeMb: null, modelsLoaded: 0, hasUpdate: false, installProgress: null,
  installDetail: null, error: null, alwaysAvailable: false, ociRef: null, ociDigest: null, packages: {}, declaredPackages: [], gitRef: null,
})

const state = {
  backends: [b("ollama", "installed", "Ollama"), b("tensorrt_llm", "installed", "TensorRT-LLM"), b("vllm", "not_installed", "vLLM"), b("piper", "not_installed", "Piper")],
  loading: false, usingMock: false, error: null,
  fetchAll: vi.fn(), install: vi.fn(), cancelInstall: vi.fn(), uninstall: vi.fn(),
}
vi.mock("@/stores/backendsStore", () => ({ useBackendsStore: (sel: (s: typeof state) => unknown) => sel(state) }))
vi.mock("@/api/client", () => ({
  api: {
    backends: { published: vi.fn().mockResolvedValue({ published: null, updateAvailable: false }), logs: vi.fn() },
    ollama: {
      version: vi.fn().mockResolvedValue({ current: "0.40.13", latest: "0.40.13", updateAvailable: false }),
      serverUpdateStatus: vi.fn().mockResolvedValue({ status: "idle", detail: null, startedAt: null, finishedAt: null, fromVersion: null, toVersion: null }),
      startServerUpdate: vi.fn(),
    },
    models: {
      updateAll: vi.fn(),
      list: vi.fn().mockResolvedValue([
        { modelId: "ollama/a", backendType: "ollama", status: "loaded" },
        { modelId: "ollama/b", backendType: "ollama", status: "loaded" },
        { modelId: "ollama/c", backendType: "ollama", status: "configured" },
      ]),
    },
    config: { get: vi.fn().mockResolvedValue({ ollamaNumParallel: 2, ollamaMaxContext: 65536 }), patch: vi.fn() },
  },
}))
vi.mock("@/api/modelEditor", () => ({
  modelEditorApi: { ollamaSettings: vi.fn().mockResolvedValue({ pending: false, running: { numParallel: 2, maxContext: 65536 } }), applyOllamaSettings: vi.fn() },
}))

async function renderPage() {
  const { Backends } = await import("@/pages/Backends")
  return render(
    <MemoryRouter>
      <Backends />
    </MemoryRouter>,
  )
}

describe("Backends", () => {
  it("solo muestra los instalados, con icono, y el resto en «Añadir backend»", async () => {
    const { container } = await renderPage()
    expect(container.querySelector('[data-backend-icon="ollama"]')).toBeTruthy()
    expect(container.querySelector('[data-backend-icon="tensorrt_llm"]')).toBeTruthy()
    expect(screen.queryByText("vLLM")).toBeNull()
    fireEvent.click(screen.getByRole("button", { name: /Añadir backend \(2\)/ }))
    const dialog = await screen.findByTestId("add-backend-dialog")
    expect(dialog.querySelector('[data-catalog="vllm"]')).toBeTruthy()
    expect(dialog.querySelector('[data-catalog="piper"]')).toBeTruthy()
    expect(dialog.querySelector('[data-catalog="ollama"]')).toBeNull()
  })

  it("TensorRT-LLM enlaza a sus engines y Ollama lleva sus controles dentro", async () => {
    await renderPage()
    expect(screen.getByTestId("trt-engines-link").getAttribute("href")).toBe("/engines")
    // Los modelos cargados se cuentan de la lista de modelos (la API de backends no lo da).
    const ollamaCard = (await screen.findByTestId("ollama-controls")).closest("[data-backend-card]") as HTMLElement
    await within(ollamaCard).findByText("2")
    const controls = await screen.findByTestId("ollama-controls")
    expect(within(controls).getByText("Actualizar daemon")).toBeTruthy()
    expect(await within(controls).findByTestId("ollama-server-settings")).toBeTruthy()
  })
})
