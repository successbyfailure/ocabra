import { describe, expect, it } from "vitest"
import {
  categorizeModel,
  inferProfileCategory,
  profileToModelCategory,
} from "@/lib/modelContext"
import type { ModelCapabilities, ModelProfile, ModelState } from "@/types"

const CAPS: ModelCapabilities = {
  chat: false,
  completion: false,
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
  streaming: false,
  contextLength: 0,
  nativeContextLength: 0,
}

function model(partial: Partial<ModelState>): ModelState {
  return {
    modelId: "m",
    displayName: "m",
    backendType: "llama_cpp",
    status: "configured",
    loadPolicy: "on_demand",
    pinPermanent: false,
    preferredGpu: null,
    currentGpu: [],
    vramUsedMb: 0,
    ramUsedMb: 0,
    diskSizeBytes: null,
    capabilities: { ...CAPS },
    lastRequestAt: null,
    loadedAt: null,
    ...partial,
  }
}

function profile(category: ModelProfile["category"], isDefault = true): ModelProfile {
  return {
    profileId: "p",
    profileKind: "execution",
    baseModelId: "m",
    displayName: null,
    description: null,
    category,
    loadOverrides: null,
    requestDefaults: null,
    assets: null,
    enabled: true,
    isDefault,
    routingTargets: null,
    runtimePolicy: null,
    runtimePresets: [],
    aliases: [],
    createdAt: null,
    updatedAt: null,
  }
}

describe("inferProfileCategory (single capability fallback)", () => {
  it("maps capabilities to a profile category", () => {
    expect(inferProfileCategory({ tts: true })).toBe("tts")
    expect(inferProfileCategory({ audioTranscription: true })).toBe("stt")
    expect(inferProfileCategory({ rerank: true })).toBe("rerank")
    expect(inferProfileCategory({ embeddings: true })).toBe("embedding")
    expect(inferProfileCategory({ chat: true })).toBe("llm")
    expect(inferProfileCategory({})).toBeNull()
  })
})

describe("categorizeModel", () => {
  it("prefers the backend profile category as the single source", () => {
    // Capabilities say "chat" but the backend profile says "embedding".
    const m = model({
      capabilities: { ...CAPS, chat: true },
      profiles: [profile("embedding")],
    })
    expect(categorizeModel(m)).toBe("embeddings")
  })

  it("falls back to capabilities when there is no profile", () => {
    expect(categorizeModel(model({ capabilities: { ...CAPS, tts: true } }))).toBe("tts")
    expect(categorizeModel(model({}))).toBe("other")
  })

  it("does not treat a model as image just because imageEditing is set", () => {
    // imageGeneration false + imageEditing false → not image.
    expect(categorizeModel(model({ capabilities: { ...CAPS, chat: true } }))).toBe("chat")
  })

  it("recognises video backends by type", () => {
    expect(categorizeModel(model({ backendType: "comfyui_video" }))).toBe("video")
  })
})

describe("profileToModelCategory", () => {
  it("maps rerank into embeddings and null for unknowns", () => {
    expect(profileToModelCategory("rerank")).toBe("embeddings")
    expect(profileToModelCategory(null)).toBeNull()
    expect(profileToModelCategory("nonsense")).toBeNull()
  })
})
