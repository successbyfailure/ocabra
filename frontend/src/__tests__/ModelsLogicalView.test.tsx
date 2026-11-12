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
// Sin socket ni sondeo en los tests: la actividad en vivo llega vacía.
vi.mock("@/components/dashboard/useLiveActivity", () => ({
  useLiveActivity: () => ({ data: { activity: {}, contexts: {}, stuckThresholdSeconds: 300 }, live: false, receivedAt: 0 }),
}))
// El editor es una página: solo importa a qué despliegue navega.
const navigate = vi.fn()
vi.mock("react-router-dom", async (importOriginal) => ({
  ...(await importOriginal<typeof import("react-router-dom")>()),
  useNavigate: () => navigate,
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

async function openMenu(card: HTMLElement) {
  fireEvent.keyDown(within(card).getByRole("button", { name: /Más acciones/ }), { key: "Enter" })
  await screen.findByRole("menu")
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
    expect(within(card).getByText("Apagado")).toBeTruthy()
  })

  it("changes the default backend from the menu with PATCH", async () => {
    const card = await renderModelView()
    await openMenu(card)
    fireEvent.click(await screen.findByRole("menuitem", { name: /Servir gemma4:26b con ollama/ }))
    await waitFor(() =>
      expect(api.logicalModels.patch).toHaveBeenCalledWith("gemma4:26b", { defaultBackend: "ollama" }),
    )
  })

  it("only offers installed, not-deployed compatible backends to add, and POSTs the choice", async () => {
    const card = await renderModelView()
    await openMenu(card)
    const adds = (await screen.findAllByRole("menuitem")).filter((item) => /Añadir backend/.test(item.textContent ?? ""))
    expect(adds.map((item) => item.textContent)).toEqual(["Añadir backend llama.cpp"])

    fireEvent.click(adds[0])
    await waitFor(() =>
      expect(api.logicalModels.addDeployment).toHaveBeenCalledWith("gemma4:26b", { backend: "llama_cpp" }),
    )
  })

  it("asks for confirmation before removing a deployment and explains what happens", async () => {
    const card = await renderModelView()
    await openMenu(card)
    fireEvent.click(await screen.findByRole("menuitem", { name: "Quitar despliegue llamacpp_mks" }))

    const dialog = await screen.findByRole("dialog")
    expect(dialog.textContent).toContain("los ficheros del modelo se conservan")
    expect(dialog.textContent).toContain("el nombre público gemma4:26b pasará antes a otro despliegue")
    expect(api.logicalModels.removeDeployment).not.toHaveBeenCalled()

    fireEvent.click(within(dialog).getByRole("button", { name: "Quitar" }))
    await waitFor(() =>
      expect(api.logicalModels.removeDeployment).toHaveBeenCalledWith("gemma4:26b", "llamacpp_mks"),
    )
  })

  it("opens the editor page of the clicked deployment", async () => {
    const card = await renderModelView()
    fireEvent.click(within(card).getByLabelText(/^ollama · desplegado/))
    expect(navigate).toHaveBeenCalledWith(`/models/edit/${encodeURIComponent("ollama/gemma4:26b")}`)
  })

  it("keeps the per-deployment view reachable", async () => {
    await renderModelView()
    fireEvent.click(screen.getByRole("button", { name: "Por despliegue" }))
    await waitFor(() => expect(screen.getByText(/2 despliegues en chat/)).toBeTruthy())
    expect(screen.queryByTestId("logical-model-gemma4:26b")).toBeNull()
  })
  it("shows the preferred GPU of the shown deployment, and of each one in the menu", async () => {
    const card = await renderModelView()
    expect(within(card).getByTestId("preferred-gpu").textContent).toBe("GPU 1")
    await openMenu(card)
    expect((await screen.findByTestId("preferred-gpu-llamacpp_mks")).textContent).toBe("GPU 1")
    expect(screen.getByTestId("preferred-gpu-ollama").textContent).toBe("GPU auto")
  })

  it("opens a deployment's editor from the menu", async () => {
    const card = await renderModelView()
    await openMenu(card)
    fireEvent.click(await screen.findByRole("menuitem", { name: /Editar ollama/ }))
    expect(navigate).toHaveBeenCalledWith(`/models/edit/${encodeURIComponent("ollama/gemma4:26b")}`)
  })

  it("Probar opens the Playground on the shown deployment", async () => {
    const card = await renderModelView()
    fireEvent.click(within(card).getByRole("button", { name: /Probar/ }))
    expect(navigate).toHaveBeenCalledWith(`/playground?model=${encodeURIComponent("llamacpp_mks/gemma4:26b")}`)
  })

  it("Editar opens the editor on the default deployment", async () => {
    const card = await renderModelView()
    fireEvent.click(within(card).getByRole("button", { name: "Editar gemma4:26b" }))
    expect(navigate).toHaveBeenCalledWith(`/models/edit/${encodeURIComponent("llamacpp_mks/gemma4:26b")}`)
  })
})
