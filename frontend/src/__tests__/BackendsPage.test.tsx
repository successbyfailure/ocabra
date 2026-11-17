import { fireEvent, render, screen, within } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"
import { describe, expect, it, vi } from "vitest"
import type { BackendModuleState } from "@/types"

const b = (backendType: string, installStatus: BackendModuleState["installStatus"], displayName = backendType, tags: string[] = [], modelsLoaded = 0): BackendModuleState => ({
  backendType, displayName, description: `desc ${backendType}`, tags, installStatus, installedVersion: null, installedAt: null,
  installSource: null, estimatedSizeMb: 1200, actualSizeMb: 2048, modelsLoaded, hasUpdate: false, installProgress: null,
  installDetail: null, error: null, alwaysAvailable: false, ociRef: null, ociDigest: null, packages: {}, declaredPackages: [], gitRef: null,
})

const state = {
  backends: [
    b("ollama", "installed", "Ollama", ["LLM"], 2),
    b("tensorrt_llm", "installed", "TensorRT-LLM", ["LLM"]),
    b("whisper", "installed", "Whisper", ["STT"], 1),
    b("vllm", "not_installed", "vLLM", ["LLM"]),
    b("piper", "not_installed", "Piper", ["TTS"]),
  ],
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
    models: { updateAll: vi.fn() },
    config: {
      get: vi.fn().mockResolvedValue({ ollamaNumParallel: 2, ollamaMaxContext: 65536, whisperStartupTimeoutSeconds: 120, openaiAudioMaxPartSizeMb: 24 }),
      patch: vi.fn(async (p: object) => ({ ollamaNumParallel: 2, ollamaMaxContext: 65536, whisperStartupTimeoutSeconds: 120, openaiAudioMaxPartSizeMb: 24, ...p })),
    },
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
    // «Modelos cargados» viene de la API (models_loaded).
    const ollamaCard = (await screen.findByTestId("ollama-controls")).closest("[data-backend-card]") as HTMLElement
    await within(ollamaCard).findByText("2")
    const controls = await screen.findByTestId("ollama-controls")
    expect(within(controls).getByText("Actualizar daemon")).toBeTruthy()
    expect(await within(controls).findByTestId("ollama-server-settings")).toBeTruthy()
  })
  it("filtra por uso y resume disco, modelos cargados y nada que actualizar", async () => {
    const { container } = await renderPage()
    const summary = screen.getByTestId("backends-summary")
    expect(summary.textContent).toContain("6.0 GB en disco · 3 modelos cargados")
    fireEvent.click(within(summary).getByText("Audio (1)"))
    expect(container.querySelector('[data-backend-card="whisper"]')).toBeTruthy()
    expect(container.querySelector('[data-backend-card="ollama"]')).toBeNull()
    expect(screen.queryByTestId("backends-updates")).toBeNull()
  })

  it("cada backend lleva sus valores por defecto en su tarjeta y se guardan solo los suyos", async () => {
    const { api } = await import("@/api/client")
    await renderPage()
    const defaults = await screen.findByTestId("backend-defaults-whisper")
    fireEvent.click(within(defaults).getByText(/Valores por defecto \(2\)/))
    const timeout = within(defaults).getByLabelText(/Tiempo máximo de arranque/) as HTMLInputElement
    fireEvent.change(timeout, { target: { value: "300" } })
    fireEvent.click(within(defaults).getByRole("button", { name: "Guardar" }))
    await vi.waitFor(() => expect(api.config.patch).toHaveBeenCalledWith({ whisperStartupTimeoutSeconds: 300, openaiAudioMaxPartSizeMb: 24 }))
  })
})
