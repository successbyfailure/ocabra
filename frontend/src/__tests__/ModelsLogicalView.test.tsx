// @vitest-environment jsdom
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { Models } from "@/pages/Models"
import { useModelStore } from "@/stores/modelStore"
import type { LogicalModel, ModelState } from "@/types"

const api = vi.hoisted(() => ({
  gpus: { list: vi.fn() },
  models: {
    list: vi.fn(),
    storage: vi.fn(),
    get: vi.fn(),
    patch: vi.fn(),
  },
  logicalModels: {
    list: vi.fn(),
    patch: vi.fn(),
    addDeployment: vi.fn(),
    removeDeployment: vi.fn(),
  },
  backends: { list: vi.fn() },
}))

vi.mock("@/api/client", () => ({ api }))
vi.mock("@/hooks/useWebSocket", () => ({ useWebSocket: vi.fn() }))
vi.mock("sonner", () => ({ toast: { success: vi.fn(), error: vi.fn() } }))
// The real modal pulls in a lot (groups, estimates…); only which deployment
// it was opened for matters here.
vi.mock("@/components/models/ModelConfigModal", () => ({
  ModelConfigModal: ({ model, open }: { model: ModelState | null; open: boolean }) =>
    open && model ? <div data-testid="config-modal">{model.modelId}</div> : null,
  ModelConfigForm: ({
    model,
    active,
    footerStart,
  }: {
    model: ModelState | null
    active?: boolean
    footerStart?: import("react").ReactNode
  }) => (
    <div data-testid="config-form" data-active={String(active)}>
      {model?.modelId}
      {footerStart}
    </div>
  ),
}))

function deploymentState(modelId: string, backendType: ModelState["backendType"]): ModelState {
  return {
    modelId,
    displayName: modelId,
    backendType,
    status: "configured",
    loadPolicy: "on_demand",
    pinPermanent: false,
    preferredGpu: null,
    currentGpu: [],
    vramUsedMb: 0,
    ramUsedMb: 0,
    diskSizeBytes: null,
    capabilities: {
      chat: true,
      completion: true,
      tools: false,
      vision: false,
      embeddings: false,
      pooling: false,
      rerank: false,
      classification: false,
      score: false,
      reasoning: false,
      imageGeneration: false,
      imageEditing: false,
      audioTranscription: false,
      musicGeneration: false,
      tts: false,
      audioInput: false,
      audioOutput: false,
      videoInput: false,
      streaming: true,
      contextLength: 8192,
      nativeContextLength: 8192,
    },
    lastRequestAt: null,
    loadedAt: null,
  }
}

const gemma: LogicalModel = {
  modelKey: "gemma4:26b",
  displayName: "gemma4:26b",
  category: "llm",
  defaultBackend: "llamacpp_mks",
  deployments: [
    {
      modelId: "llamacpp_mks/gemma4:26b",
      backend: "llamacpp_mks",
      isDefault: true,
      installed: true,
      status: "configured",
      loadPolicy: "pin",
      pinPermanent: false,
      preferredGpu: 1,
      artifact: { kind: "gguf", source: "ollama", ref: "gemma4:26b" },
      profiles: ["gemma4:26b"],
    },
    {
      modelId: "ollama/gemma4:26b",
      backend: "ollama",
      isDefault: false,
      installed: true,
      status: "configured",
      loadPolicy: "on_demand",
      pinPermanent: false,
      preferredGpu: null,
      artifact: { kind: "ollama_tag", source: "ollama", ref: "gemma4:26b" },
      profiles: ["gemma4:26b-ollama"],
    },
  ],
  compatibleBackends: [
    { backend: "llamacpp_mks", status: "compatible", installed: true, reason: "GGUF", deployed: true },
    { backend: "llama_cpp", status: "compatible", installed: true, reason: "GGUF", deployed: false },
    { backend: "vllm", status: "probable", installed: false, reason: "", deployed: false },
    { backend: "ollama", status: "compatible", installed: true, reason: "tag", deployed: true },
  ],
  publicProfiles: ["gemma4:26b"],
  hiddenProfiles: ["gemma4:26b-ollama"],
  registryMetadata: null,
}

async function renderModelView() {
  render(<Models />)
  return await screen.findByTestId("logical-model-gemma4:26b")
}

describe("Models · vista por modelo", () => {
  beforeEach(() => {
    useModelStore.setState({ models: {} })
    api.gpus.list.mockResolvedValue([])
    api.backends.list.mockResolvedValue([])
    api.models.list.mockResolvedValue([
      deploymentState("llamacpp_mks/gemma4:26b", "llamacpp_mks"),
      deploymentState("ollama/gemma4:26b", "ollama"),
    ])
    api.models.storage.mockResolvedValue({ path: "/models", totalBytes: 0, usedBytes: 0, freeBytes: 0 })
    api.logicalModels.list.mockResolvedValue([gemma])
    api.logicalModels.patch.mockResolvedValue(gemma)
    api.logicalModels.addDeployment.mockResolvedValue(gemma)
    api.logicalModels.removeDeployment.mockResolvedValue({
      modelKey: "gemma4:26b",
      removed: "llamacpp_mks/gemma4:26b",
      newDefaultBackend: "ollama",
    })
  })

  afterEach(() => {
    vi.clearAllMocks()
  })

  it("is the default view and lists one card per model", async () => {
    const card = await renderModelView()
    expect(screen.getByRole("button", { name: "Por modelo" }).getAttribute("aria-pressed")).toBe("true")
    expect(within(card).getByLabelText(/llama.cpp mks · por defecto · desplegado/)).toBeTruthy()
    expect(screen.getByText(/1 modelo en chat/)).toBeTruthy()
  })

  it("changes the default backend with PATCH", async () => {
    const card = await renderModelView()
    fireEvent.change(within(card).getByLabelText("Backend por defecto"), { target: { value: "ollama" } })
    await waitFor(() =>
      expect(api.logicalModels.patch).toHaveBeenCalledWith("gemma4:26b", { defaultBackend: "ollama" }),
    )
  })

  it("only offers installed, not-deployed compatible backends to add, and POSTs the choice", async () => {
    const card = await renderModelView()
    const select = within(card).getByLabelText("Añadir backend") as HTMLSelectElement
    const values = Array.from(select.options).map((option) => option.value)
    expect(values).toEqual(["", "llama_cpp"])

    fireEvent.change(select, { target: { value: "llama_cpp" } })
    fireEvent.click(within(card).getByRole("button", { name: /Añadir$/ }))
    await waitFor(() =>
      expect(api.logicalModels.addDeployment).toHaveBeenCalledWith("gemma4:26b", { backend: "llama_cpp" }),
    )
  })

  it("asks for confirmation before removing a deployment and explains what happens", async () => {
    const card = await renderModelView()
    fireEvent.click(within(card).getByLabelText("Quitar despliegue llamacpp_mks"))

    const dialog = await screen.findByRole("dialog")
    expect(dialog.textContent).toContain("los ficheros del modelo se conservan")
    expect(dialog.textContent).toContain("el nombre público gemma4:26b pasará antes a otro despliegue")
    expect(api.logicalModels.removeDeployment).not.toHaveBeenCalled()

    fireEvent.click(within(dialog).getByRole("button", { name: "Quitar" }))
    await waitFor(() =>
      expect(api.logicalModels.removeDeployment).toHaveBeenCalledWith("gemma4:26b", "llamacpp_mks"),
    )
  })

  it("opens the deployment config modal when a deployment is clicked", async () => {
    const card = await renderModelView()
    fireEvent.click(within(card).getByLabelText(/^ollama · desplegado/))
    expect((await screen.findByTestId("config-modal")).textContent).toBe("ollama/gemma4:26b")
  })

  it("keeps the per-deployment view reachable", async () => {
    await renderModelView()
    fireEvent.click(screen.getByRole("button", { name: "Por despliegue" }))
    await waitFor(() => expect(screen.getByText(/2 despliegues en chat/)).toBeTruthy())
    expect(screen.queryByTestId("logical-model-gemma4:26b")).toBeNull()
  })
  it("shows each deployment's preferred GPU (auto when unset)", async () => {
    const card = await renderModelView()
    expect(within(card).getByTestId("preferred-gpu-llamacpp_mks").textContent).toBe("GPU 1")
    expect(within(card).getByTestId("preferred-gpu-ollama").textContent).toBe("GPU auto")
  })

  describe("editor", () => {
    async function openEditor() {
      const card = await renderModelView()
      fireEvent.click(within(card).getByRole("button", { name: "Editar gemma4:26b" }))
      return await screen.findByRole("dialog")
    }

    it("General tab renames, switches the default and adds a backend", async () => {
      const dialog = await openEditor()
      expect(within(dialog).getByText("Editar modelo")).toBeTruthy()

      fireEvent.change(within(dialog).getByLabelText("Nombre visible"), { target: { value: "Gemma 4 26B" } })
      fireEvent.click(within(dialog).getByRole("button", { name: "Guardar nombre" }))
      await waitFor(() =>
        expect(api.logicalModels.patch).toHaveBeenCalledWith("gemma4:26b", { displayName: "Gemma 4 26B" }),
      )

      fireEvent.change(within(dialog).getByLabelText("Backend por defecto"), { target: { value: "ollama" } })
      await waitFor(() =>
        expect(api.logicalModels.patch).toHaveBeenCalledWith("gemma4:26b", { defaultBackend: "ollama" }),
      )

      fireEvent.change(within(dialog).getByLabelText("Añadir backend"), { target: { value: "llama_cpp" } })
      fireEvent.click(within(dialog).getByRole("button", { name: /Añadir$/ }))
      await waitFor(() =>
        expect(api.logicalModels.addDeployment).toHaveBeenCalledWith("gemma4:26b", { backend: "llama_cpp" }),
      )
    })

    it("has one tab per deployment (★ on the default) embedding that deployment's form", async () => {
      const dialog = await openEditor()
      const tabs = within(dialog).getAllByRole("tab").map((tab) => tab.textContent)
      expect(tabs).toEqual(["General", "★ llama.cpp mks", "ollama"])

      fireEvent.mouseDown(within(dialog).getByRole("tab", { name: "ollama" }))
      const form = await within(dialog).findByTestId("config-form")
      expect(form.textContent).toContain("ollama/gemma4:26b")
      expect(form.getAttribute("data-active")).toBe("true")
    })

    it("removes a deployment from its tab through the same confirmation", async () => {
      const dialog = await openEditor()
      fireEvent.mouseDown(within(dialog).getByRole("tab", { name: "★ llama.cpp mks" }))
      const form = await within(dialog).findByTestId("config-form")
      fireEvent.click(within(form).getByRole("button", { name: /Quitar despliegue/ }))

      const confirm = await screen.findByRole("dialog", { name: "Quitar despliegue" })
      expect(confirm.textContent).toContain("los ficheros del modelo se conservan")
      fireEvent.click(within(confirm).getByRole("button", { name: "Quitar" }))
      await waitFor(() =>
        expect(api.logicalModels.removeDeployment).toHaveBeenCalledWith("gemma4:26b", "llamacpp_mks"),
      )
    })
  })
})
