import { create } from "zustand"
import type { ModelState } from "@/types"
import { api } from "@/api/client"

interface ModelStore {
  models: Record<string, ModelState>
  setModels: (models: ModelState[]) => void
  updateModel: (modelId: string, patch: Partial<ModelState>) => void
  loadModel: (modelId: string) => Promise<void>
  unloadModel: (modelId: string) => Promise<void>
}

export const useModelStore = create<ModelStore>((set) => ({
  models: {},
  setModels: (models) =>
    set({
      models: Object.fromEntries(models.map((m) => [m.modelId, m])),
    }),
  updateModel: (modelId, patch) =>
    set((state) => ({
      models: {
        ...state.models,
        [modelId]: state.models[modelId]
          ? { ...state.models[modelId], ...patch }
          : {
              modelId,
              displayName: modelId,
              backendType: "vllm",
              status: "configured",
              loadPolicy: "on_demand",
              autoReload: false,
              preferredGpu: null,
              currentGpu: [],
              vramUsedMb: 0,
              diskSizeBytes: null,
              capabilities: {
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
                audioTranscription: false,
                tts: false,
                streaming: false,
                contextLength: 0,
              },
              lastRequestAt: null,
              loadedAt: null,
              ...patch,
            },
      },
    })),
  loadModel: async (modelId) => {
    const updated = await api.models.load(modelId)
    set((state) => ({
      models: { ...state.models, [modelId]: updated },
    }))
  },
  unloadModel: async (modelId) => {
    await api.models.unload(modelId)
    set((state) => ({
      models: {
        ...state.models,
        [modelId]: {
          ...state.models[modelId],
          status: "unloaded",
        },
      },
    }))
  },
}))
