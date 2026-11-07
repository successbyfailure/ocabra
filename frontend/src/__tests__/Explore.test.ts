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

describe("download naming and default backend", () => {
  it("suggests a public name per source", async () => {
    const { suggestModelKey } = await import("@/pages/Explore")
    expect(suggestModelKey("ollama", "gemma4:e4b")).toBe("gemma4:e4b")
    expect(suggestModelKey("huggingface", "Qwen/Qwen3-8B")).toBe("qwen3-8b")
    const gguf = {
      variantId: "q4",
      label: "Q4_K_M",
      artifact: "Qwen3-8B-Q4_K_M.gguf",
      sizeGb: 5,
      format: "gguf",
      quantization: "Q4_K_M",
      backendType: "llamacpp_mks" as const,
      isDefault: true,
    }
    expect(suggestModelKey("huggingface", "Qwen/Qwen3-8B", gguf)).toBe("qwen3-8b-q4_k_m")
    expect(suggestModelKey("huggingface", "Qwen/Qwen3-8B-GGUF", gguf)).toBe("qwen3-8b-q4_k_m")
  })

  it("prefers llamacpp_mks for GGUF only when installed", async () => {
    const { pickDefaultBackend } = await import("@/pages/Explore")
    const entry = (backend: "llamacpp_mks" | "llama_cpp" | "vllm", installed: boolean) => ({
      backend,
      status: "compatible" as const,
      installed,
      reason: "",
    })
    expect(pickDefaultBackend([entry("llama_cpp", true), entry("llamacpp_mks", true)], { gguf: true })).toBe(
      "llamacpp_mks",
    )
    expect(
      pickDefaultBackend([entry("llamacpp_mks", false), entry("llama_cpp", true)], {
        gguf: true,
        variantBackend: "llamacpp_mks",
      }),
    ).toBe("llama_cpp")
    expect(pickDefaultBackend([entry("llamacpp_mks", true), entry("vllm", true)], { gguf: false, variantBackend: "vllm" })).toBe(
      "vllm",
    )
  })
})
