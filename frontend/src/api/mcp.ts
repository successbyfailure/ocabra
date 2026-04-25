// Typed client for /ocabra/mcp-servers.
// Stream C (frontend). Backend (Stream A) not merged yet — UI falls back to
// mocks in the store when the API responds with 404.

import type {
  MCPServer,
  MCPServerCreate,
  MCPServerTestResult,
  MCPServerUpdate,
  MCPToolSpec,
} from "@/types/agents"

const BASE = ""

type AnyRecord = Record<string, unknown>

function isRecord(value: unknown): value is AnyRecord {
  return typeof value === "object" && value !== null
}

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method,
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    const error = new Error(err.detail ?? res.statusText) as Error & { status?: number }
    error.status = res.status
    throw error
  }
  if (res.status === 204) return undefined as T
  return res.json() as Promise<T>
}

function toToolSpec(raw: unknown): MCPToolSpec {
  const d = isRecord(raw) ? raw : {}
  return {
    name: String(d.name ?? ""),
    description: String(d.description ?? ""),
    inputSchema: isRecord(d.input_schema ?? d.inputSchema)
      ? ((d.input_schema ?? d.inputSchema) as Record<string, unknown>)
      : {},
  }
}

function toMCPServer(raw: unknown): MCPServer {
  const d = isRecord(raw) ? raw : {}
  const tools = Array.isArray(d.tools_cache ?? d.toolsCache)
    ? ((d.tools_cache ?? d.toolsCache) as unknown[]).map(toToolSpec)
    : null
  return {
    id: String(d.id ?? ""),
    alias: String(d.alias ?? ""),
    displayName: String(d.display_name ?? d.displayName ?? d.alias ?? ""),
    description: (d.description ?? null) as string | null,
    transport: (d.transport ?? "http") as MCPServer["transport"],
    url: (d.url ?? null) as string | null,
    command: (d.command ?? null) as string | null,
    args: Array.isArray(d.args) ? (d.args as unknown[]).map(String) : null,
    env: isRecord(d.env) ? (d.env as Record<string, string>) : null,
    authType: (d.auth_type ?? d.authType ?? "none") as MCPServer["authType"],
    allowedTools: Array.isArray(d.allowed_tools ?? d.allowedTools)
      ? ((d.allowed_tools ?? d.allowedTools) as unknown[]).map(String)
      : null,
    groupId: (d.group_id ?? d.groupId ?? null) as string | null,
    groupName: (d.group_name ?? d.groupName ?? null) as string | null,
    toolsCache: tools,
    toolsCacheUpdatedAt: (d.tools_cache_updated_at ?? d.toolsCacheUpdatedAt ?? null) as string | null,
    healthStatus: (d.health_status ?? d.healthStatus ?? "unknown") as MCPServer["healthStatus"],
    lastError: (d.last_error ?? d.lastError ?? null) as string | null,
    createdAt: String(d.created_at ?? d.createdAt ?? ""),
    updatedAt: String(d.updated_at ?? d.updatedAt ?? ""),
  }
}

function serializeCreate(data: MCPServerCreate | MCPServerUpdate): Record<string, unknown> {
  const out: Record<string, unknown> = {}
  if (data.alias !== undefined) out.alias = data.alias
  if (data.displayName !== undefined) out.display_name = data.displayName
  if (data.description !== undefined) out.description = data.description
  if (data.transport !== undefined) out.transport = data.transport
  if (data.url !== undefined) out.url = data.url
  if (data.command !== undefined) out.command = data.command
  if (data.args !== undefined) out.args = data.args
  if (data.env !== undefined) out.env = data.env
  if (data.authType !== undefined) out.auth_type = data.authType
  if (data.authValue !== undefined) out.auth_value = data.authValue
  if (data.allowedTools !== undefined) out.allowed_tools = data.allowedTools
  if (data.groupId !== undefined) out.group_id = data.groupId
  return out
}

function toTestResult(raw: unknown): MCPServerTestResult {
  const d = isRecord(raw) ? raw : {}
  return {
    healthy: Boolean(d.healthy ?? false),
    toolsCount: Number(d.tools_count ?? d.toolsCount ?? 0),
    error: (d.error ?? null) as string | null,
  }
}

export const mcpApi = {
  list: async (): Promise<MCPServer[]> =>
    (await request<unknown[]>("GET", "/ocabra/mcp-servers")).map(toMCPServer),

  get: async (id: string): Promise<MCPServer> =>
    toMCPServer(await request<unknown>("GET", `/ocabra/mcp-servers/${encodeURIComponent(id)}`)),

  create: async (data: MCPServerCreate): Promise<MCPServer> =>
    toMCPServer(await request<unknown>("POST", "/ocabra/mcp-servers", serializeCreate(data))),

  update: async (id: string, data: MCPServerUpdate): Promise<MCPServer> =>
    toMCPServer(
      await request<unknown>(
        "PATCH",
        `/ocabra/mcp-servers/${encodeURIComponent(id)}`,
        serializeCreate(data),
      ),
    ),

  delete: async (id: string): Promise<void> => {
    await request<void>("DELETE", `/ocabra/mcp-servers/${encodeURIComponent(id)}`)
  },

  refresh: async (id: string): Promise<MCPServer> =>
    toMCPServer(
      await request<unknown>("POST", `/ocabra/mcp-servers/${encodeURIComponent(id)}/refresh`),
    ),

  test: async (id: string): Promise<MCPServerTestResult> =>
    toTestResult(
      await request<unknown>("POST", `/ocabra/mcp-servers/${encodeURIComponent(id)}/test`),
    ),

  tools: async (id: string): Promise<MCPToolSpec[]> =>
    (
      await request<unknown[]>(
        "GET",
        `/ocabra/mcp-servers/${encodeURIComponent(id)}/tools`,
      )
    ).map(toToolSpec),
}

export { toMCPServer, toToolSpec }
