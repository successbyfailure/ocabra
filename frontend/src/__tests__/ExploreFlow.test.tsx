// @vitest-environment jsdom
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { Explore } from "@/pages/Explore"
import { useDownloadStore } from "@/stores/downloadStore"

const {
  listDownloads,
  searchHF,
  searchBitnet,
  getHFVariants,
  getBitnetVariants,
  enqueue,
  streamProgress,
  searchOllama,
  getOllamaVariants,
} = vi.hoisted(() => ({
  searchOllama: vi.fn(),
  getOllamaVariants: vi.fn(),
  listDownloads: vi.fn(),
  searchHF: vi.fn(),
  searchBitnet: vi.fn(),
  getHFVariants: vi.fn(),
  getBitnetVariants: vi.fn(),
  enqueue: vi.fn(),
  streamProgress: vi.fn(),
}))

vi.mock("sonner", () => ({
  toast: {
    success: vi.fn(),
    error: vi.fn(),
  },
}))

vi.mock("@/api/client", () => ({
  api: {
    downloads: {
      list: listDownloads,
      enqueue,
      cancel: vi.fn(),
      clearHistory: vi.fn(),
      streamProgress,
    },
    registry: {
      searchHF,
      searchBitnet,
      getHFVariants,
      getBitnetVariants,
      searchOllama,
      getOllamaVariants,
    },
  },
}))

describe("Explore flow", () => {
  beforeEach(() => {
    useDownloadStore.setState({ jobs: [] })
    listDownloads.mockResolvedValue([])
    searchHF.mockResolvedValue([
      {
        repoId: "Qwen/Qwen3-8B-Instruct",
        modelName: "Qwen3-8B-Instruct",
        task: "text-generation",
        downloads: 123,
        likes: 45,
        sizeGb: 15.2,
        tags: ["chat"],
        gated: false,
        suggestedBackend: "vllm",
        compatibility: "warning",
        compatibilityReason: "Compatibilidad prevista via Transformers backend.",
        vllmSupport: {
          classification: "transformers_backend",
          label: "transformers backend",
          modelImpl: "transformers",
          runner: "generate",
          taskMode: "generate",
          requiredOverrides: ["chat_template", "tool_call_parser"],
          recipeId: "qwen3",
          recipeNotes: ["Qwen3 necesita parser de tools y reasoning."],
          recipeModelImpl: "vllm",
          recipeRunner: "generate",
          suggestedConfig: { tool_call_parser: "qwen3_json", reasoning_parser: "qwen3" },
          suggestedTuning: { gpu_memory_utilization: 0.9 },
          runtimeProbe: {
            status: "supported_transformers_backend",
            reason: "Compatibilidad verificada via Transformers backend.",
            recommendedModelImpl: "transformers",
            recommendedRunner: "generate",
            tokenizerLoad: true,
            configLoad: true,
            observedAt: "2026-03-14T12:00:00Z",
          },
        },
      },
    ])
    searchBitnet.mockResolvedValue([])
    searchOllama.mockResolvedValue([])
    getOllamaVariants.mockResolvedValue([])
    getBitnetVariants.mockResolvedValue([])
    getHFVariants.mockResolvedValue([
      {
        variantId: "standard",
        label: "standard",
        artifact: null,
        sizeGb: 15.2,
        format: "safetensors",
        quantization: null,
        backendType: "vllm",
        isDefault: true,
        installable: true,
        compatibility: "warning",
        compatibilityReason: "Compatibilidad prevista via Transformers backend.",
        vllmSupport: {
          classification: "transformers_backend",
          label: "transformers backend",
          modelImpl: "transformers",
          runner: "generate",
          taskMode: "generate",
          requiredOverrides: ["chat_template", "tool_call_parser"],
          recipeId: "qwen3",
          recipeNotes: ["Qwen3 necesita parser de tools y reasoning."],
          recipeModelImpl: "vllm",
          recipeRunner: "generate",
          suggestedConfig: { tool_call_parser: "qwen3_json", reasoning_parser: "qwen3" },
          suggestedTuning: { gpu_memory_utilization: 0.9 },
          runtimeProbe: {
            status: "supported_transformers_backend",
            reason: "Compatibilidad verificada via Transformers backend.",
            recommendedModelImpl: "transformers",
            recommendedRunner: "generate",
            tokenizerLoad: true,
            configLoad: true,
            observedAt: "2026-03-14T12:00:00Z",
          },
        },
      },
    ])
    enqueue.mockResolvedValue({
      jobId: "job-1",
      source: "huggingface",
      modelRef: "Qwen/Qwen3-8B-Instruct",
      status: "queued",
      progressPct: 0,
      speedMbS: null,
      etaSeconds: null,
      error: null,
      startedAt: "2026-03-14T00:00:00Z",
      completedAt: null,
    })
    streamProgress.mockReturnValue({
      close: vi.fn(),
      onmessage: null,
    })
  })

  afterEach(() => {
    vi.clearAllMocks()
  })

  it("shows recipe vs probe guidance and enqueues install with persisted probe metadata", async () => {
    render(<Explore />)

    await waitFor(() => {
      expect(screen.getByText("Qwen3-8B-Instruct")).toBeTruthy()
    }, { timeout: 3000 })

    fireEvent.click(screen.getAllByText("Instalar")[0])

    // El detalle del probe vive dentro del colapsable "Detalles avanzados",
    // que arranca cerrado. El test se escribio cuando todo iba plano en el
    // modal, asi que esperaba ese texto sin desplegar nada.
    await waitFor(() => {
      expect(screen.getByText("Detalles avanzados")).toBeTruthy()
    }, { timeout: 3000 })
    fireEvent.click(screen.getByText("Detalles avanzados"))

    await waitFor(() => {
      const bodyText = document.body.textContent ?? ""
      expect(bodyText).toContain("Instalar modelo")
      expect(bodyText).toContain("Recomendacion verificada por probe")
      expect(bodyText).toContain("Compatibilidad verificada via Transformers backend")
      expect(bodyText).toContain("La preconfiguracion automatica usara la recomendacion verificada por probe")
      expect(bodyText).toContain("La recomendacion final del probe difiere de la recipe base")
      expect(bodyText).toContain("recipe_model_impl=vllm")
    }, { timeout: 3000 })

    fireEvent.click(screen.getByRole("button", { name: "Iniciar descarga" }))

    await waitFor(() => {
      expect(enqueue).toHaveBeenCalledTimes(1)
    }, { timeout: 3000 })

    expect(enqueue).toHaveBeenCalledWith(
      "huggingface",
      "Qwen/Qwen3-8B-Instruct",
      null,
      expect.objectContaining({
        displayName: "Qwen3-8B-Instruct",
        loadPolicy: "on_demand",
        extraConfig: {
          vllm: expect.objectContaining({
            recipeId: "qwen3",
            recipeModelImpl: "vllm",
            recipeRunner: "generate",
            probeStatus: "supported_transformers_backend",
            probeRecommendedModelImpl: "transformers",
            probeRecommendedRunner: "generate",
            modelImpl: "transformers",
            runner: "generate",
          }),
        },
      }),
    )
  }, 10000)
  it("Ollama: proposes the tag as name, llamacpp_mks as default plus ollama, and sends them", async () => {
    const compat = [
      { backend: "llamacpp_mks", status: "compatible", installed: true, reason: "GGUF del tag" },
      { backend: "llama_cpp", status: "compatible", installed: true, reason: "GGUF del tag" },
      { backend: "ollama", status: "compatible", installed: true, reason: "tag de Ollama" },
    ]
    searchOllama.mockResolvedValue([
      { name: "gemma4", description: "Gemma 4", tags: [], sizeGb: 9.6, pulls: 10, compatibleBackends: compat },
    ])
    getOllamaVariants.mockResolvedValue([
      {
        name: "gemma4:e4b",
        tag: "e4b",
        sizeGb: 9.6,
        parameterSize: "8B",
        quantization: "Q4_K_M",
        contextWindow: null,
        modality: null,
        updatedHint: null,
        compatibleBackends: compat,
      },
    ])

    render(<Explore />)
    fireEvent.mouseDown(screen.getByRole("tab", { name: "Ollama" }))
    await waitFor(() => expect(screen.getByText("gemma4")).toBeTruthy(), { timeout: 3000 })
    fireEvent.click(screen.getAllByText("Instalar")[0])

    const nameInput = (await screen.findByLabelText("Nombre del modelo")) as HTMLInputElement
    await waitFor(() => expect(nameInput.value).toBe("gemma4:e4b"), { timeout: 3000 })
    expect((screen.getByLabelText("Desplegar con llamacpp_mks") as HTMLInputElement).checked).toBe(true)
    expect((screen.getByLabelText("Desplegar con ollama") as HTMLInputElement).checked).toBe(true)
    expect((screen.getByLabelText("Desplegar con llama_cpp") as HTMLInputElement).checked).toBe(false)
    expect((screen.getByLabelText("Por defecto: llamacpp_mks") as HTMLInputElement).checked).toBe(true)

    fireEvent.click(screen.getByRole("button", { name: "Iniciar descarga" }))
    await waitFor(() => expect(enqueue).toHaveBeenCalledTimes(1), { timeout: 3000 })
    expect(enqueue).toHaveBeenCalledWith(
      "ollama",
      "gemma4:e4b",
      null,
      expect.objectContaining({
        modelKey: "gemma4:e4b",
        defaultBackend: "llamacpp_mks",
        backends: ["llamacpp_mks", "ollama"],
        backendType: "llamacpp_mks",
      }),
    )
  }, 10000)

  it("HF GGUF: names it <repo>-<quant>, lets the user pick deployments and the default", async () => {
    const compat = [
      { backend: "llamacpp_mks", status: "compatible", installed: true, reason: "GGUF" },
      { backend: "llama_cpp", status: "compatible", installed: true, reason: "GGUF" },
      { backend: "ollama", status: "probable", installed: true, reason: "importable" },
      { backend: "bitnet", status: "probable", installed: false, reason: "ternario" },
    ]
    searchHF.mockResolvedValue([
      {
        repoId: "Qwen/Qwen3-8B-GGUF",
        modelName: "Qwen3-8B-GGUF",
        task: "text-generation",
        downloads: 1,
        likes: 1,
        sizeGb: 5,
        tags: [],
        gated: false,
        suggestedBackend: "llamacpp_mks",
        compatibleBackends: compat,
      },
    ])
    getHFVariants.mockResolvedValue([
      {
        variantId: "q4",
        label: "Q4_K_M",
        artifact: "Qwen3-8B-Q4_K_M.gguf",
        sizeGb: 5,
        format: "gguf",
        quantization: "Q4_K_M",
        backendType: "llamacpp_mks",
        isDefault: true,
        installable: true,
        compatibleBackends: compat,
      },
    ])

    render(<Explore />)
    await waitFor(() => expect(screen.getByText("Qwen3-8B-GGUF")).toBeTruthy(), { timeout: 3000 })
    fireEvent.click(screen.getAllByText("Instalar")[0])

    const nameInput = (await screen.findByLabelText("Nombre del modelo")) as HTMLInputElement
    await waitFor(() => expect(nameInput.value).toBe("qwen3-8b-q4_k_m"), { timeout: 3000 })
    // Not-installed backends cannot be selected.
    expect((screen.getByLabelText("Desplegar con bitnet") as HTMLInputElement).disabled).toBe(true)

    fireEvent.click(screen.getByLabelText("Desplegar con llama_cpp"))
    fireEvent.click(screen.getByLabelText("Por defecto: llama_cpp"))
    fireEvent.change(nameInput, { target: { value: "qwen3-8b" } })
    fireEvent.click(screen.getByRole("button", { name: "Iniciar descarga" }))

    await waitFor(() => expect(enqueue).toHaveBeenCalledTimes(1), { timeout: 3000 })
    expect(enqueue).toHaveBeenCalledWith(
      "huggingface",
      "Qwen/Qwen3-8B-GGUF",
      "Qwen3-8B-Q4_K_M.gguf",
      expect.objectContaining({
        modelKey: "qwen3-8b",
        defaultBackend: "llama_cpp",
        backends: ["llama_cpp", "llamacpp_mks"],
      }),
    )
  }, 10000)
})
