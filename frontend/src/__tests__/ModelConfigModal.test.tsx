// @vitest-environment jsdom
import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { ModelConfigModal } from "@/components/models/ModelConfigModal"

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

    expect((screen.getByLabelText(/GPU memory utilization/i) as HTMLInputElement).value).toBe("0.9")

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
})
