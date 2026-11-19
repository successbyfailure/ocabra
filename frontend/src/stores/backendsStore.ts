import { create } from "zustand"
import { api, toBackendModuleState } from "@/api/client"
import type {
  BackendInstallMethod,
  BackendModuleState,
} from "@/types"

function sortBackends(list: BackendModuleState[]): BackendModuleState[] {
  return [...list].sort((a, b) => {
    // built-in first, then alphabetical by display name
    if (a.installStatus === "built-in" && b.installStatus !== "built-in") return -1
    if (b.installStatus === "built-in" && a.installStatus !== "built-in") return 1
    return a.displayName.localeCompare(b.displayName)
  })
}

interface BackendsStore {
  backends: BackendModuleState[]
  loading: boolean
  error: string | null
  // Active install streams keyed by backendType
  streams: Record<string, EventSource>
  fetchAll: () => Promise<void>
  install: (backendType: string, method: BackendInstallMethod) => void
  cancelInstall: (backendType: string) => void
  uninstall: (backendType: string) => Promise<void>
  upsert: (state: BackendModuleState) => void
  remove: (backendType: string) => void
  handleWSEvent: (
    type: "backend_installed" | "backend_uninstalled" | "backend_progress",
    data: unknown,
  ) => void
}

export const useBackendsStore = create<BackendsStore>((set, get) => ({
  backends: [],
  loading: false,
  error: null,
  streams: {},

  fetchAll: async () => {
    set({ loading: true, error: null })
    try {
      const data = await api.backends.list()
      set({ backends: sortBackends(data), loading: false })
    } catch (err) {
      const message = err instanceof Error ? err.message : "unknown error"
      set({ loading: false, error: message })
    }
  },

  upsert: (state) =>
    set((prev) => {
      const without = prev.backends.filter((b) => b.backendType !== state.backendType)
      return { backends: sortBackends([...without, state]) }
    }),

  remove: (backendType) =>
    set((prev) => ({
      backends: prev.backends.filter((b) => b.backendType !== backendType),
    })),

  install: (backendType, method) => {
    const store = get()

    // Optimistically mark as "installing" so the UI reacts immediately.
    const current = store.backends.find((b) => b.backendType === backendType)
    if (current) {
      store.upsert({
        ...current,
        installStatus: "installing",
        installProgress: 0,
        installDetail: "Queued...",
        error: null,
      })
    }

    const existing = store.streams[backendType]
    if (existing) existing.close()

    const source = api.backends.install(backendType, method)
    source.onmessage = (ev) => {
      try {
        const payload = JSON.parse(ev.data)
        const normalized = toBackendModuleState(payload)
        get().upsert(normalized)
        if (
          normalized.installStatus === "installed" ||
          normalized.installStatus === "error" ||
          normalized.installStatus === "not_installed"
        ) {
          source.close()
          set((prev) => {
            const next = { ...prev.streams }
            delete next[backendType]
            return { streams: next }
          })
        }
      } catch {
        // ignore malformed events
      }
    }
    source.onerror = () => {
      source.close()
      set((prev) => {
        const next = { ...prev.streams }
        delete next[backendType]
        return { streams: next }
      })
    }

    set((prev) => ({ streams: { ...prev.streams, [backendType]: source } }))
  },

  cancelInstall: (backendType) => {
    const stream = get().streams[backendType]
    if (stream) {
      stream.close()
      set((prev) => {
        const next = { ...prev.streams }
        delete next[backendType]
        return { streams: next }
      })
    }
    // Re-fetch the real state from the server.
    void get().fetchAll()
  },

  uninstall: async (backendType) => {
    const current = get().backends.find((b) => b.backendType === backendType)
    if (current) {
      get().upsert({
        ...current,
        installStatus: "uninstalling",
        installProgress: null,
        installDetail: null,
      })
    }
    try {
      const updated = await api.backends.uninstall(backendType)
      if (updated) {
        get().upsert(updated)
      } else {
        await get().fetchAll()
      }
    } catch (err) {
      if (current) {
        get().upsert({
          ...current,
          installStatus: "error",
          error: err instanceof Error ? err.message : "Error al desinstalar",
        })
      }
      throw err
    }
  },

  handleWSEvent: (type, data) => {
    if (type === "backend_installed" || type === "backend_progress") {
      try {
        const normalized = toBackendModuleState(data)
        get().upsert(normalized)
      } catch {
        // ignore
      }
    } else if (type === "backend_uninstalled") {
      const payload = (data ?? {}) as Record<string, unknown>
      const backendType = String(payload.backend_type ?? payload.backendType ?? "")
      if (backendType) {
        // Refresh the entry so the card reflects "not_installed" state.
        void get().fetchAll()
      }
    }
  },
}))
