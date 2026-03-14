// @vitest-environment jsdom
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { Explore } from "@/pages/Explore"
import { useDownloadStore } from "@/stores/downloadStore"

const {
  listDownloads,
  searchHF,
  getHFVariants,
  enqueue,
  streamProgress,
} = vi.hoisted(() => ({
  listDownloads: vi.fn(),
  searchHF: vi.fn(),
  getHFVariants: vi.fn(),
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
      getHFVariants,
      searchOllama: vi.fn(async () => []),
      getOllamaVariants: vi.fn(async () => []),
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
})
