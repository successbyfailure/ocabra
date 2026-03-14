// @vitest-environment jsdom
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { HFModelCard } from "@/components/explore/HFModelCard"
import type { HFModelCard as HFModelCardType } from "@/types"

const model: HFModelCardType = {
  repoId: "Qwen/Qwen3-8B",
  modelName: "Qwen3-8B",
  task: "text-generation",
  downloads: 123,
  likes: 45,
  sizeGb: 15.2,
  tags: ["chat"],
  gated: false,
  suggestedBackend: "vllm",
  compatibility: "warning",
  compatibilityReason: "vLLM probablemente requiere Transformers backend para este repo.",
  vllmSupport: {
    classification: "transformers_backend",
    label: "transformers backend",
    modelImpl: "transformers",
    runner: "generate",
    taskMode: "generate",
    requiredOverrides: ["chat_template", "tool_call_parser"],
    recipeId: "qwen3",
    recipeNotes: ["Qwen3 suele necesitar parser específico para tools y reasoning."],
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
    runtimeProbe: {
      status: "supported_transformers_backend",
      reason: null,
      recommendedModelImpl: "transformers",
      recommendedRunner: "generate",
      tokenizerLoad: true,
      configLoad: true,
      observedAt: "2026-03-14T12:00:00Z",
    },
  },
}

describe("HFModelCard", () => {
  it("renders vLLM compatibility guidance", () => {
    render(<HFModelCard model={model} onInstall={vi.fn()} />)

    expect(screen.getByText("vllm transformers backend / generate")).toBeTruthy()
    expect(screen.getByText(/requiere: chat_template, tool_call_parser/i)).toBeTruthy()
    expect(screen.getByText(/recipe: qwen3/i)).toBeTruthy()
    expect(screen.getByText(/config sugerida:/i)).toBeTruthy()
    expect(screen.getByText(/tuning recomendado:/i)).toBeTruthy()
    expect(screen.getByText("probe: compatibilidad verificada via Transformers backend")).toBeTruthy()
    expect(screen.getByText(/probe verificado: model_impl=transformers, runner=generate/i)).toBeTruthy()
  })

  it("renders actionable probe override hints", () => {
    render(
      <HFModelCard
        model={{
          ...model,
          vllmSupport: {
            ...model.vllmSupport!,
            runtimeProbe: {
              status: "missing_tool_parser",
              reason: "Automatic tool choice requires parser.",
              recommendedModelImpl: "vllm",
              recommendedRunner: "generate",
              tokenizerLoad: true,
              configLoad: true,
              observedAt: "2026-03-14T12:00:00Z",
            },
          },
        }}
        onInstall={vi.fn()}
      />,
    )

    expect(screen.getAllByText(/override sugerido: tool_call_parser/i).length).toBeGreaterThan(0)
    expect(screen.getByText("probe: falta tool_call_parser")).toBeTruthy()
  })
})
