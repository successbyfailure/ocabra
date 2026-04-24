import { create } from "zustand"
import { api, toBackendModuleState } from "@/api/client"
import type {
  BackendInstallMethod,
  BackendModuleState,
} from "@/types"

// Mock fixture used only when VITE_MOCK_BACKENDS=1. Kept for offline/dev work —
// the real /ocabra/backends API is the default source of truth.
const MOCK_BACKENDS: BackendModuleState[] = [
  {
    backendType: "vllm",
    displayName: "vLLM",
    description: "High-throughput LLM inference with PagedAttention.",
    tags: ["LLM", "GPU", "CUDA"],
    installStatus: "installed",
    installedVersion: "0.17.1",
    installedAt: "2026-04-10T10:30:00Z",
    installSource: "oci",
    estimatedSizeMb: 8000,
    actualSizeMb: 7850,
    modelsLoaded: 1,
    hasUpdate: false,
    installProgress: null,
    installDetail: null,
    error: null,
    alwaysAvailable: false,
  },
  {
    backendType: "diffusers",
    displayName: "Diffusers",
    description: "Image generation with Stable Diffusion, SDXL and Flux.",
    tags: ["Image", "GPU", "CUDA"],
    installStatus: "not_installed",
    installedVersion: null,
    installedAt: null,
    installSource: null,
    estimatedSizeMb: 3000,
    actualSizeMb: null,
    modelsLoaded: 0,
    hasUpdate: false,
    installProgress: null,
    installDetail: null,
    error: null,
    alwaysAvailable: false,
  },
  {
    backendType: "whisper",
    displayName: "faster-whisper",
    description: "Fast audio transcription with CTranslate2.",
    tags: ["Audio", "STT", "GPU"],
    installStatus: "installed",
    installedVersion: "1.0.3",
    installedAt: "2026-04-08T12:00:00Z",
    installSource: "source",
    estimatedSizeMb: 1200,
    actualSizeMb: 1180,
    modelsLoaded: 0,
    hasUpdate: true,
    installProgress: null,
    installDetail: null,
    error: null,
    alwaysAvailable: false,
  },
  {
    backendType: "llama_cpp",
    displayName: "llama.cpp",
    description: "CPU/GPU inference for GGUF models.",
    tags: ["LLM", "CPU", "GPU", "GGUF"],
    installStatus: "not_installed",
    installedVersion: null,
    installedAt: null,
    installSource: null,
    estimatedSizeMb: 500,
    actualSizeMb: null,
    modelsLoaded: 0,
    hasUpdate: false,
    installProgress: null,
    installDetail: null,
    error: null,
    alwaysAvailable: false,
  },
  {
    backendType: "ollama",
    displayName: "Ollama",
    description: "External Ollama server integration.",
    tags: ["LLM", "External"],
    installStatus: "built-in",
    installedVersion: null,
    installedAt: null,
    installSource: "built-in",
    estimatedSizeMb: 0,
    actualSizeMb: null,
    modelsLoaded: 0,
    hasUpdate: false,
    installProgress: null,
    installDetail: null,
    error: null,
    alwaysAvailable: true,
  },
]

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
  usingMock: boolean
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
  usingMock: false,
  error: null,
  streams: {},

  fetchAll: async () => {
    set({ loading: true, error: null })
    // Flag to force mock fallback during development.
    const meta = import.meta as unknown as { env?: Record<string, string | undefined> }
    if (meta.env?.VITE_MOCK_BACKENDS === "1") {
      set({
        backends: sortBackends(MOCK_BACKENDS),
        loading: false,
        usingMock: true,
        error: null,
      })
      return
    }
    try {
      const data = await api.backends.list()
      set({ backends: sortBackends(data), loading: false, usingMock: false })
    } catch (err) {
      const message = err instanceof Error ? err.message : "unknown error"
      set({ loading: false, error: message, usingMock: false })
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

    if (store.usingMock) {
      // Simulate progress in mock mode.
      let pct = 0
      const interval = window.setInterval(() => {
        pct = Math.min(1, pct + 0.1)
        const latest = get().backends.find((b) => b.backendType === backendType)
        if (!latest) {
          window.clearInterval(interval)
          return
        }
        if (pct >= 1) {
          window.clearInterval(interval)
          get().upsert({
            ...latest,
            installStatus: "installed",
            installProgress: null,
            installDetail: null,
            installedAt: new Date().toISOString(),
            installedVersion: latest.installedVersion ?? "mock-1.0.0",
            installSource: method,
            actualSizeMb: latest.estimatedSizeMb,
          })
        } else {
          get().upsert({
            ...latest,
            installStatus: "installing",
            installProgress: pct,
            installDetail:
              pct < 0.3
                ? "Pulling image..."
                : pct < 0.7
                ? "Extracting..."
                : "Verifying...",
          })
        }
      }, 600)
      return
    }

    // Real SSE flow.
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
    // Re-fetch the real state from the server; in mock mode revert to not_installed.
    if (get().usingMock) {
      const current = get().backends.find((b) => b.backendType === backendType)
      if (current) {
        get().upsert({
          ...current,
          installStatus: "not_installed",
          installProgress: null,
          installDetail: null,
        })
      }
    } else {
      void get().fetchAll()
    }
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
      if (get().usingMock) {
        await new Promise((r) => window.setTimeout(r, 500))
        if (current) {
          get().upsert({
            ...current,
            installStatus: "not_installed",
            installedVersion: null,
            installedAt: null,
            installSource: null,
            actualSizeMb: null,
            modelsLoaded: 0,
            installProgress: null,
            installDetail: null,
            error: null,
          })
        }
        return
      }
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
