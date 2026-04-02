// @vitest-environment jsdom
import { act, fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { api } from "@/api/client"
import { ModelConfigModal } from "@/components/models/ModelConfigModal"

vi.mock("@/api/client", () => ({
  api: {
    models: {
      estimateMemory: vi.fn(async () => ({
        backendType: "vllm",
        gpuIndex: 1,
        totalVramMb: 24576,
        freeVramMb: 24115,
        budgetVramMb: 22118,
        requestedContextLength: 7800,
        estimatedWeightsMb: 18000,
        estimatedEngineMbPerGpu: null,
        estimatedKvCacheMb: 1956,
        estimatedMaxContextLength: 7824,
        modelLoadingMemoryMb: 18575,
        maximumConcurrency: 1,
        tensorParallelSize: 1,
        fitsCurrentGpu: true,
        enginePresent: null,
        source: "heuristic",
        status: "ok",
        warning: null,
        notes: ["estimacion de prueba"],
      })),
    },
  },
}))

describe("ModelConfigModal", () => {
  it("renders persisted recipe metadata and reapplies suggested config", () => {
    render(
      <ModelConfigModal
        open
        onOpenChange={vi.fn()}
        onSave={vi.fn(async () => undefined)}
        gpus={[]}
        model={{
          modelId: "Qwen/Qwen3-8B-Instruct",
          displayName: "Qwen3 8B",
          backendType: "vllm",
          status: "configured",
          loadPolicy: "warm",
          autoReload: false,
          preferredGpu: null,
          currentGpu: [],
          vramUsedMb: 0,
          diskSizeBytes: null,
          capabilities: {
            chat: true,
            completion: true,
            tools: true,
            vision: false,
            embeddings: false,
            pooling: false,
            rerank: false,
            classification: false,
            score: false,
            reasoning: true,
            imageGeneration: false,
            audioTranscription: false,
            musicGeneration: false,
            tts: false,
            streaming: true,
            contextLength: 32768,
          },
          lastRequestAt: null,
          loadedAt: null,
          extraConfig: {
            vllm: {
              recipeId: "qwen3",
              recipeNotes: ["Qwen3 necesita parser de tools y reasoning."],
              recipeModelImpl: "vllm",
              recipeRunner: "generate",
              suggestedConfig: {
                tool_call_parser: "qwen3_json",
                reasoning_parser: "qwen3",
              },
              suggestedTuning: {
                gpu_memory_utilization: 0.9,
                enable_prefix_caching: true,
              },
              probeStatus: "supported_transformers_backend",
              probeRecommendedModelImpl: "transformers",
              probeRecommendedRunner: "generate",
              probeReason: "Compatibilidad verificada via Transformers backend.",
              probeObservedAt: "2026-03-14T12:00:00Z",
              modelImpl: "vllm",
              runner: "generate",
            },
          },
        }}
      />,
    )

    expect(screen.getByText(/recipe=qwen3/i)).toBeTruthy()
    expect(screen.getByText(/recipe=qwen3 · model_impl=vllm · runner=generate/i)).toBeTruthy()
    expect(screen.getByText(/Qwen3 necesita parser/i)).toBeTruthy()
    expect(screen.getByText(/probe=compatibilidad verificada via Transformers backend/i)).toBeTruthy()
    expect(screen.getByText(/recomendacion verificada · model_impl=transformers · runner=generate/i)).toBeTruthy()
    expect(screen.getByText(/difiere de la recomendacion verificada por probe/i)).toBeTruthy()
    expect(screen.getByText(/La recipe base y la recomendacion final del probe no coinciden/i)).toBeTruthy()
    expect(screen.getByText(/observado=2026-03-14T12:00:00Z/i)).toBeTruthy()

    fireEvent.click(screen.getByText(/Reaplicar sugerencias de recipe/i))

    expect(screen.getByDisplayValue("qwen3_json")).toBeTruthy()
    expect(screen.getByDisplayValue("qwen3")).toBeTruthy()

    fireEvent.click(screen.getByText(/Aplicar tuning recomendado/i))

    expect((screen.getAllByLabelText(/GPU memory utilization/i)[1] as HTMLInputElement).value).toBe("0.9")

    fireEvent.click(screen.getByText(/Aplicar recomendacion del probe/i))

    expect(screen.getByDisplayValue("transformers")).toBeTruthy()
  })

  it("shows override hint when probe flags a missing parser", () => {
    render(
      <ModelConfigModal
        open
        onOpenChange={vi.fn()}
        onSave={vi.fn(async () => undefined)}
        gpus={[]}
        model={{
          modelId: "DeepSeek/DeepSeek-R1",
          displayName: "DeepSeek R1",
          backendType: "vllm",
          status: "configured",
          loadPolicy: "warm",
          autoReload: false,
          preferredGpu: null,
          currentGpu: [],
          vramUsedMb: 0,
          diskSizeBytes: null,
          capabilities: {
            chat: true,
            completion: true,
            tools: true,
            vision: false,
            embeddings: false,
            pooling: false,
            rerank: false,
            classification: false,
            score: false,
            reasoning: true,
            imageGeneration: false,
            audioTranscription: false,
            musicGeneration: false,
            tts: false,
            streaming: true,
            contextLength: 32768,
          },
          lastRequestAt: null,
          loadedAt: null,
          extraConfig: {
            vllm: {
              probeStatus: "missing_reasoning_parser",
              probeRecommendedModelImpl: "vllm",
              probeRecommendedRunner: "generate",
              probeReason: "Reasoning needs parser.",
              modelImpl: "vllm",
              runner: "generate",
            },
          },
        }}
      />,
    )

    expect(screen.getByText(/probe=falta reasoning_parser/i)).toBeTruthy()
    expect(screen.getByText(/override sugerido: reasoning_parser/i)).toBeTruthy()
  })

  it("saves llama.cpp options in nested backend config", async () => {
    const onSave = vi.fn(async () => undefined)

    render(
      <ModelConfigModal
        open
        onOpenChange={vi.fn()}
        onSave={onSave}
        gpus={[]}
        model={{
          modelId: "llama_cpp/Qwen/Qwen2.5-0.5B-Instruct-GGUF",
          displayName: "Qwen2.5 GGUF",
          backendType: "llama_cpp",
          status: "configured",
          loadPolicy: "on_demand",
          autoReload: false,
          preferredGpu: null,
          currentGpu: [],
          vramUsedMb: 0,
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
            audioTranscription: false,
            musicGeneration: false,
            tts: false,
            streaming: true,
            contextLength: 8192,
          },
          lastRequestAt: null,
          loadedAt: null,
          extraConfig: {
            model_path: "/models/qwen.gguf",
          },
        }}
      />,
    )

    fireEvent.change(screen.getByLabelText(/GPU layers/i), { target: { value: "24" } })
    fireEvent.change(screen.getByLabelText(/Context size/i), { target: { value: "16384" } })
    fireEvent.click(screen.getByLabelText(/flash attention/i))
    fireEvent.click(screen.getByLabelText(/embedding mode/i))
    await act(async () => {
      fireEvent.click(screen.getByText(/^Guardar$/i))
    })

    expect(onSave).toHaveBeenCalledWith(
      "llama_cpp/Qwen/Qwen2.5-0.5B-Instruct-GGUF",
      expect.objectContaining({
        extraConfig: expect.objectContaining({
          model_path: "/models/qwen.gguf",
          llama_cpp: expect.objectContaining({
            gpuLayers: 24,
            ctxSize: 16384,
            flashAttn: true,
            embedding: true,
          }),
        }),
      }),
    )
  })

  it("renders memory estimate and lets the user trigger runtime validation", async () => {
    render(
      <ModelConfigModal
        open
        onOpenChange={vi.fn()}
        onSave={vi.fn(async () => undefined)}
        gpus={[]}
        model={{
          modelId: "vllm/Qwen/Qwen3-32B-AWQ",
          displayName: "Qwen3 32B AWQ",
          backendType: "vllm",
          status: "configured",
          loadPolicy: "on_demand",
          autoReload: false,
          preferredGpu: null,
          currentGpu: [],
          vramUsedMb: 0,
          diskSizeBytes: null,
          capabilities: {
            chat: true,
            completion: true,
            tools: true,
            vision: false,
            embeddings: false,
            pooling: false,
            rerank: false,
            classification: false,
            score: false,
            reasoning: true,
            imageGeneration: false,
            audioTranscription: false,
            musicGeneration: false,
            tts: false,
            streaming: true,
            contextLength: 40960,
          },
          lastRequestAt: null,
          loadedAt: null,
          extraConfig: {
            vllm: {
              maxModelLen: 7800,
              gpuMemoryUtilization: 0.9,
            },
          },
        }}
      />,
    )

    await screen.findByText(/Prevision de memoria/i)
    await screen.findByText(/pesos estimados: 18000 MB/i)
    await screen.findByText(/contexto maximo estimado por engine: 7.8k/i)

    await act(async () => {
      fireEvent.click(screen.getByText(/Validar con engine/i))
    })

    expect(api.models.estimateMemory).toHaveBeenCalled()
  })
})
