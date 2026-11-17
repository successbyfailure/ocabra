import { create } from "zustand"
import { mcpApi } from "@/api/mcp"
import type { MCPServer, MCPServerCreate, MCPServerUpdate } from "@/types/agents"

// TODO: remove mock once Stream A merges /ocabra/mcp-servers.
function sortServers(list: MCPServer[]): MCPServer[] {
  return [...list].sort((a, b) => a.displayName.localeCompare(b.displayName))
}

interface MCPStore {
  servers: MCPServer[]
  loading: boolean
  error: string | null
  fetchAll: () => Promise<void>
  create: (data: MCPServerCreate) => Promise<MCPServer>
  update: (id: string, data: MCPServerUpdate) => Promise<MCPServer>
  remove: (id: string) => Promise<void>
  refresh: (id: string) => Promise<MCPServer>
  upsert: (server: MCPServer) => void
  handleWSEvent: (type: string, data: unknown) => void
}

export const useMCPStore = create<MCPStore>((set, get) => ({
  servers: [],
  loading: false,
  error: null,

  fetchAll: async () => {
    set({ loading: true, error: null })
    try {
      const data = await mcpApi.list()
      set({ servers: sortServers(data), loading: false })
    } catch (err) {
      // Shown as an error; it used to swap in made-up example servers.
      const message = err instanceof Error ? err.message : "unknown error"
      set({ loading: false, error: message })
    }
  },

  upsert: (server) =>
    set((prev) => {
      const without = prev.servers.filter((s) => s.id !== server.id)
      return { servers: sortServers([...without, server]) }
    }),

  create: async (data) => {
    const created = await mcpApi.create(data)
    get().upsert(created)
    return created
  },

  update: async (id, data) => {
    const updated = await mcpApi.update(id, data)
    get().upsert(updated)
    // The backend invalidates tools_cache on PATCH. Repopulate it eagerly so
    // the next edit dialog opens on the checklist, not the comma-separated
    // textarea fallback.
    try {
      const refreshed = await mcpApi.refresh(id)
      get().upsert(refreshed)
      return refreshed
    } catch {
      return updated
    }
  },

  remove: async (id) => {
    await mcpApi.delete(id)
    set((prev) => ({ servers: prev.servers.filter((s) => s.id !== id) }))
  },

  refresh: async (id) => {
    const refreshed = await mcpApi.refresh(id)
    get().upsert(refreshed)
    return refreshed
  },

  handleWSEvent: (type, data) => {
    if (type !== "mcp_server_health_changed") return
    const payload = (data ?? {}) as Record<string, unknown>
    const id = String(payload.id ?? payload.mcp_server_id ?? payload.mcpServerId ?? "")
    if (!id) return
    const existing = get().servers.find((s) => s.id === id)
    if (!existing) {
      void get().fetchAll()
      return
    }
    get().upsert({
      ...existing,
      healthStatus: (payload.health_status ??
        payload.healthStatus ??
        existing.healthStatus) as MCPServer["healthStatus"],
      lastError: (payload.last_error ?? payload.lastError ?? existing.lastError) as string | null,
    })
  },
}))
