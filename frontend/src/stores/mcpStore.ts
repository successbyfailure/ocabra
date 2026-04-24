import { create } from "zustand"
import { mcpApi } from "@/api/mcp"
import type { MCPServer, MCPServerCreate, MCPServerUpdate } from "@/types/agents"

// TODO: remove mock once Stream A merges /ocabra/mcp-servers.
const MOCK_SERVERS: MCPServer[] = [
  {
    id: "mock-mcp-1",
    alias: "filesystem",
    displayName: "Local Filesystem",
    description: "Local filesystem MCP server (stdio).",
    transport: "stdio",
    url: null,
    command: "uvx",
    args: ["mcp-server-filesystem", "/data"],
    env: {},
    authType: "none",
    allowedTools: null,
    groupId: null,
    groupName: null,
    toolsCache: [
      {
        name: "read_file",
        description: "Read a file from the allowed root.",
        inputSchema: {
          type: "object",
          properties: { path: { type: "string" } },
          required: ["path"],
        },
      },
      {
        name: "list_dir",
        description: "List directory contents.",
        inputSchema: {
          type: "object",
          properties: { path: { type: "string" } },
          required: ["path"],
        },
      },
    ],
    toolsCacheUpdatedAt: "2026-04-24T09:00:00Z",
    healthStatus: "healthy",
    lastError: null,
    createdAt: "2026-04-20T08:00:00Z",
    updatedAt: "2026-04-24T09:00:00Z",
  },
  {
    id: "mock-mcp-2",
    alias: "github",
    displayName: "GitHub API",
    description: "GitHub MCP server (HTTP).",
    transport: "http",
    url: "https://mcp.example.com/github",
    command: null,
    args: null,
    env: null,
    authType: "bearer",
    allowedTools: ["create_issue", "list_repos"],
    groupId: null,
    groupName: null,
    toolsCache: [
      {
        name: "create_issue",
        description: "Create an issue on a repo.",
        inputSchema: {
          type: "object",
          properties: { repo: { type: "string" }, title: { type: "string" } },
          required: ["repo", "title"],
        },
      },
      {
        name: "list_repos",
        description: "List repos owned by user.",
        inputSchema: { type: "object", properties: {} },
      },
    ],
    toolsCacheUpdatedAt: "2026-04-24T08:30:00Z",
    healthStatus: "unhealthy",
    lastError: "401 Unauthorized (mock)",
    createdAt: "2026-04-22T10:00:00Z",
    updatedAt: "2026-04-24T08:30:00Z",
  },
]

function sortServers(list: MCPServer[]): MCPServer[] {
  return [...list].sort((a, b) => a.displayName.localeCompare(b.displayName))
}

function isNotFoundError(err: unknown): boolean {
  const status = (err as { status?: number } | null)?.status
  return status === 404 || status === 501
}

interface MCPStore {
  servers: MCPServer[]
  loading: boolean
  usingMock: boolean
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
  usingMock: false,
  error: null,

  fetchAll: async () => {
    set({ loading: true, error: null })
    const meta = import.meta as unknown as { env?: Record<string, string | undefined> }
    if (meta.env?.VITE_MOCK_AGENTS === "1") {
      set({ servers: sortServers(MOCK_SERVERS), loading: false, usingMock: true, error: null })
      return
    }
    try {
      const data = await mcpApi.list()
      set({ servers: sortServers(data), loading: false, usingMock: false })
    } catch (err) {
      if (isNotFoundError(err)) {
        // TODO: remove mock once Stream A merges /ocabra/mcp-servers.
        set({
          servers: sortServers(MOCK_SERVERS),
          loading: false,
          usingMock: true,
          error: null,
        })
        return
      }
      const message = err instanceof Error ? err.message : "unknown error"
      set({ loading: false, error: message, usingMock: false })
    }
  },

  upsert: (server) =>
    set((prev) => {
      const without = prev.servers.filter((s) => s.id !== server.id)
      return { servers: sortServers([...without, server]) }
    }),

  create: async (data) => {
    if (get().usingMock) {
      const created: MCPServer = {
        id: `mock-${Date.now()}`,
        alias: data.alias,
        displayName: data.displayName,
        description: data.description ?? null,
        transport: data.transport,
        url: data.url ?? null,
        command: data.command ?? null,
        args: data.args ?? null,
        env: data.env ?? null,
        authType: data.authType,
        allowedTools: data.allowedTools ?? null,
        groupId: data.groupId ?? null,
        groupName: null,
        toolsCache: null,
        toolsCacheUpdatedAt: null,
        healthStatus: "unknown",
        lastError: null,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      }
      get().upsert(created)
      return created
    }
    const created = await mcpApi.create(data)
    get().upsert(created)
    return created
  },

  update: async (id, data) => {
    if (get().usingMock) {
      const existing = get().servers.find((s) => s.id === id)
      if (!existing) throw new Error("not found")
      const merged: MCPServer = {
        ...existing,
        ...(data.alias !== undefined && { alias: data.alias }),
        ...(data.displayName !== undefined && { displayName: data.displayName }),
        ...(data.description !== undefined && { description: data.description ?? null }),
        ...(data.transport !== undefined && { transport: data.transport }),
        ...(data.url !== undefined && { url: data.url ?? null }),
        ...(data.command !== undefined && { command: data.command ?? null }),
        ...(data.args !== undefined && { args: data.args ?? null }),
        ...(data.env !== undefined && { env: data.env ?? null }),
        ...(data.authType !== undefined && { authType: data.authType }),
        ...(data.allowedTools !== undefined && { allowedTools: data.allowedTools ?? null }),
        ...(data.groupId !== undefined && { groupId: data.groupId ?? null }),
        updatedAt: new Date().toISOString(),
      }
      get().upsert(merged)
      return merged
    }
    const updated = await mcpApi.update(id, data)
    get().upsert(updated)
    return updated
  },

  remove: async (id) => {
    if (get().usingMock) {
      set((prev) => ({ servers: prev.servers.filter((s) => s.id !== id) }))
      return
    }
    await mcpApi.delete(id)
    set((prev) => ({ servers: prev.servers.filter((s) => s.id !== id) }))
  },

  refresh: async (id) => {
    if (get().usingMock) {
      const existing = get().servers.find((s) => s.id === id)
      if (!existing) throw new Error("not found")
      const refreshed: MCPServer = {
        ...existing,
        toolsCacheUpdatedAt: new Date().toISOString(),
        healthStatus: "healthy",
        lastError: null,
      }
      get().upsert(refreshed)
      return refreshed
    }
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
