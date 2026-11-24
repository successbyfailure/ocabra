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
    set((state) => {
      const existing = state.models[modelId]
      if (existing) {
        return { models: { ...state.models, [modelId]: { ...existing, ...patch } } }
      }
      // Unknown model. Only insert when the patch is a *complete* ModelState
      // (the WebSocket resync hands over the full object from GET/list). We no
      // longer fabricate a phantom with backendType:"vllm" and all-false
      // capabilities from a partial status event — that invented a card that
      // never matched the real model and drifted from the backend. A partial
      // update for an unknown id is dropped; the next list resync fills it in.
      if (typeof (patch as Partial<ModelState>).backendType === "string") {
        return { models: { ...state.models, [modelId]: patch as ModelState } }
      }
      return state
    }),
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
