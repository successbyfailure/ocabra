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
        [modelId]: { ...state.models[modelId], ...patch },
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
