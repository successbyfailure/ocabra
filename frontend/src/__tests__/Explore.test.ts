import { describe, expect, it } from "vitest"
import { getProbePreconfigurationNotice, getRecipeProbeDifferenceNotice } from "@/pages/Explore"
import type { HFModelVariant } from "@/types"

describe("getProbePreconfigurationNotice", () => {
  it("returns a notice when probe recommendation exists", () => {
    const variant: HFModelVariant = {
      variantId: "standard",
      label: "standard",
      artifact: null,
      sizeGb: 4,
      format: "safetensors",
      quantization: null,
      backendType: "vllm",
      isDefault: true,
      installable: true,
      compatibility: "warning",
      compatibilityReason: null,
      vllmSupport: {
        classification: "transformers_backend",
        label: "transformers backend",
        modelImpl: "transformers",
        runner: "generate",
        taskMode: "generate",
        requiredOverrides: [],
        recipeId: "qwen3",
        recipeNotes: [],
        recipeModelImpl: "vllm",
        recipeRunner: "generate",
        suggestedConfig: {},
        suggestedTuning: {},
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

    expect(getProbePreconfigurationNotice(variant)).toContain("preconfiguracion automatica")
    expect(getProbePreconfigurationNotice(variant)).toContain("model_impl=transformers")
    expect(getProbePreconfigurationNotice(variant)).toContain("runner=generate")
  })

  it("returns null when probe has no recommendation", () => {
    const variant: HFModelVariant = {
      variantId: "standard",
      label: "standard",
      artifact: null,
      sizeGb: 4,
      format: "safetensors",
      quantization: null,
      backendType: "vllm",
      isDefault: true,
      installable: true,
      compatibility: "compatible",
      compatibilityReason: null,
      vllmSupport: {
        classification: "native_vllm",
        label: "native",
        modelImpl: "vllm",
        runner: "generate",
        taskMode: "generate",
        requiredOverrides: [],
        recipeId: null,
        recipeNotes: [],
        recipeModelImpl: null,
        recipeRunner: null,
        suggestedConfig: {},
        suggestedTuning: {},
        runtimeProbe: {
          status: "supported_native",
          reason: null,
          recommendedModelImpl: null,
          recommendedRunner: null,
          tokenizerLoad: true,
          configLoad: true,
          observedAt: null,
        },
      },
    }

    expect(getProbePreconfigurationNotice(variant)).toBeNull()
  })

  it("returns a difference notice when recipe and probe disagree", () => {
    const variant: HFModelVariant = {
      variantId: "standard",
      label: "standard",
      artifact: null,
      sizeGb: 4,
      format: "safetensors",
      quantization: null,
      backendType: "vllm",
      isDefault: true,
      installable: true,
      compatibility: "warning",
      compatibilityReason: null,
      vllmSupport: {
        classification: "transformers_backend",
        label: "transformers backend",
        modelImpl: "transformers",
        runner: "generate",
        taskMode: "generate",
        requiredOverrides: [],
        recipeId: "qwen3",
        recipeNotes: [],
        recipeModelImpl: "vllm",
        recipeRunner: "generate",
        suggestedConfig: {},
        suggestedTuning: {},
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

    expect(getRecipeProbeDifferenceNotice(variant)).toContain("recipe base")
    expect(getRecipeProbeDifferenceNotice(variant)).toContain("recipe model_impl=vllm")
    expect(getRecipeProbeDifferenceNotice(variant)).toContain("final model_impl=transformers")
  })
})
