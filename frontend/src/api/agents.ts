// Typed client for /ocabra/agents.
// Stream C (frontend). Backend (Stream A/B) not merged yet — UI falls back to
// mocks in the store when the API responds with 404.

import type {
  Agent,
  AgentCreate,
  AgentMCPBinding,
  AgentTestResult,
  AgentTestServerResult,
  AgentUpdate,
  ByAgentStats,
  ToolCallsData,
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

function buildQuery(params: Record<string, string | number | undefined>): string {
  const search = new URLSearchParams()
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== "") search.set(key, String(value))
  })
  const qs = search.toString()
  return qs ? `?${qs}` : ""
}

function toBinding(raw: unknown): AgentMCPBinding {
  const d = isRecord(raw) ? raw : {}
  return {
    mcpServerId: String(d.mcp_server_id ?? d.mcpServerId ?? ""),
    allowedTools: Array.isArray(d.allowed_tools ?? d.allowedTools)
      ? ((d.allowed_tools ?? d.allowedTools) as unknown[]).map(String)
      : null,
  }
}

function toAgent(raw: unknown): Agent {
  const d = isRecord(raw) ? raw : {}
  const rawMcp = d.mcp_servers ?? d.mcpServers
  return {
    id: String(d.id ?? ""),
    slug: String(d.slug ?? ""),
    displayName: String(d.display_name ?? d.displayName ?? d.slug ?? ""),
    description: (d.description ?? null) as string | null,
    baseModelId: (d.base_model_id ?? d.baseModelId ?? null) as string | null,
    profileId: (d.profile_id ?? d.profileId ?? null) as string | null,
    systemPrompt: String(d.system_prompt ?? d.systemPrompt ?? ""),
    toolChoiceDefault: (d.tool_choice_default ?? d.toolChoiceDefault ?? "auto") as Agent["toolChoiceDefault"],
    maxToolHops: Number(d.max_tool_hops ?? d.maxToolHops ?? 8),
    toolTimeoutSeconds: Number(d.tool_timeout_seconds ?? d.toolTimeoutSeconds ?? 60),
    requireApproval: (d.require_approval ?? d.requireApproval ?? "never") as Agent["requireApproval"],
    requestDefaults: isRecord(d.request_defaults ?? d.requestDefaults)
      ? ((d.request_defaults ?? d.requestDefaults) as Record<string, unknown>)
      : null,
    groupId: (d.group_id ?? d.groupId ?? null) as string | null,
    groupName: (d.group_name ?? d.groupName ?? null) as string | null,
    mcpServers: Array.isArray(rawMcp) ? (rawMcp as unknown[]).map(toBinding) : [],
    createdAt: String(d.created_at ?? d.createdAt ?? ""),
    updatedAt: String(d.updated_at ?? d.updatedAt ?? ""),
  }
}

function serializeAgent(data: AgentCreate | AgentUpdate): Record<string, unknown> {
  const out: Record<string, unknown> = {}
  if (data.slug !== undefined) out.slug = data.slug
  if (data.displayName !== undefined) out.display_name = data.displayName
  if (data.description !== undefined) out.description = data.description
  if (data.baseModelId !== undefined) out.base_model_id = data.baseModelId
  if (data.profileId !== undefined) out.profile_id = data.profileId
  if (data.systemPrompt !== undefined) out.system_prompt = data.systemPrompt
  if (data.toolChoiceDefault !== undefined) out.tool_choice_default = data.toolChoiceDefault
  if (data.maxToolHops !== undefined) out.max_tool_hops = data.maxToolHops
  if (data.toolTimeoutSeconds !== undefined) out.tool_timeout_seconds = data.toolTimeoutSeconds
  if (data.requireApproval !== undefined) out.require_approval = data.requireApproval
  if (data.requestDefaults !== undefined) out.request_defaults = data.requestDefaults
  if (data.groupId !== undefined) out.group_id = data.groupId
  if (data.mcpServers !== undefined) {
    out.mcp_servers = data.mcpServers.map((binding) => ({
      mcp_server_id: binding.mcpServerId,
      allowed_tools: binding.allowedTools,
    }))
  }
  return out
}

function toServerTest(raw: unknown): AgentTestServerResult {
  const d = isRecord(raw) ? raw : {}
  return {
    mcpServerId: String(d.mcp_server_id ?? d.mcpServerId ?? ""),
    alias: String(d.alias ?? ""),
    healthy: Boolean(d.healthy ?? false),
    toolsCount: Number(d.tools_count ?? d.toolsCount ?? 0),
    error: (d.error ?? null) as string | null,
  }
}

function toAgentTest(raw: unknown): AgentTestResult {
  const d = isRecord(raw) ? raw : {}
  return {
    healthy: Boolean(d.healthy ?? false),
    toolsCount: Number(d.tools_count ?? d.toolsCount ?? 0),
    servers: Array.isArray(d.servers) ? (d.servers as unknown[]).map(toServerTest) : [],
    baseModelReachable: Boolean(d.base_model_reachable ?? d.baseModelReachable ?? false),
    errors: Array.isArray(d.errors) ? (d.errors as unknown[]).map(String) : [],
  }
}

export const agentsApi = {
  list: async (): Promise<Agent[]> =>
    (await request<unknown[]>("GET", "/ocabra/agents")).map(toAgent),

  get: async (slug: string): Promise<Agent> =>
    toAgent(await request<unknown>("GET", `/ocabra/agents/${encodeURIComponent(slug)}`)),

  create: async (data: AgentCreate): Promise<Agent> =>
    toAgent(await request<unknown>("POST", "/ocabra/agents", serializeAgent(data))),

  update: async (slug: string, data: AgentUpdate): Promise<Agent> =>
    toAgent(
      await request<unknown>(
        "PATCH",
        `/ocabra/agents/${encodeURIComponent(slug)}`,
        serializeAgent(data),
      ),
    ),

  delete: async (slug: string): Promise<void> => {
    await request<void>("DELETE", `/ocabra/agents/${encodeURIComponent(slug)}`)
  },

  test: async (slug: string): Promise<AgentTestResult> =>
    toAgentTest(
      await request<unknown>("POST", `/ocabra/agents/${encodeURIComponent(slug)}/test`),
    ),
}

// TODO: remove these once backend exposes /ocabra/stats/by-agent and
// /ocabra/stats/tool-calls. See docs/tasks/agents-mcp-plan.md "Deudas abiertas".
export const agentStatsApi = {
  byAgent: async (params: { from?: string; to?: string }): Promise<ByAgentStats> => {
    const raw = await request<AnyRecord>("GET", `/ocabra/stats/by-agent${buildQuery(params)}`)
    const byAgent = Array.isArray(raw.by_agent ?? raw.byAgent)
      ? ((raw.by_agent ?? raw.byAgent) as AnyRecord[]).map((d) => ({
          agentId: String(d.agent_id ?? d.agentId ?? ""),
          slug: String(d.slug ?? ""),
          displayName: String(d.display_name ?? d.displayName ?? ""),
          requestCount: Number(d.request_count ?? d.requestCount ?? 0),
          toolCallCount: Number(d.tool_call_count ?? d.toolCallCount ?? 0),
          errorCount: Number(d.error_count ?? d.errorCount ?? 0),
          p50DurationMs: (d.p50_duration_ms ?? d.p50DurationMs ?? null) as number | null,
          p95DurationMs: (d.p95_duration_ms ?? d.p95DurationMs ?? null) as number | null,
          totalTokens: Number(d.total_tokens ?? d.totalTokens ?? 0),
        }))
      : []
    const byTool = Array.isArray(raw.by_tool ?? raw.byTool)
      ? ((raw.by_tool ?? raw.byTool) as AnyRecord[]).map((d) => ({
          mcpServerAlias: String(d.mcp_server_alias ?? d.mcpServerAlias ?? ""),
          toolName: String(d.tool_name ?? d.toolName ?? ""),
          invocations: Number(d.invocations ?? 0),
          errors: Number(d.errors ?? 0),
          p50DurationMs: (d.p50_duration_ms ?? d.p50DurationMs ?? null) as number | null,
          p95DurationMs: (d.p95_duration_ms ?? d.p95DurationMs ?? null) as number | null,
          errorRate: Number(d.error_rate ?? d.errorRate ?? 0),
        }))
      : []
    return { byAgent, byTool }
  },

  toolCalls: async (params: {
    agentId?: string
    limit?: number
  }): Promise<ToolCallsData> => {
    const raw = await request<AnyRecord>(
      "GET",
      `/ocabra/stats/tool-calls${buildQuery({
        agent_id: params.agentId,
        limit: params.limit,
      })}`,
    )
    const toolCalls = Array.isArray(raw.tool_calls ?? raw.toolCalls)
      ? ((raw.tool_calls ?? raw.toolCalls) as AnyRecord[]).map((d) => ({
          id: String(d.id ?? ""),
          createdAt: String(d.created_at ?? d.createdAt ?? ""),
          agentId: (d.agent_id ?? d.agentId ?? null) as string | null,
          agentSlug: (d.agent_slug ?? d.agentSlug ?? null) as string | null,
          mcpServerAlias: String(d.mcp_server_alias ?? d.mcpServerAlias ?? ""),
          toolName: String(d.tool_name ?? d.toolName ?? ""),
          status: (d.status ?? "ok") as "ok" | "timeout" | "schema_error" | "mcp_error",
          durationMs: Number(d.duration_ms ?? d.durationMs ?? 0),
          hopIndex: Number(d.hop_index ?? d.hopIndex ?? 0),
          error: (d.error ?? null) as string | null,
          argsRedacted: isRecord(d.args_redacted ?? d.argsRedacted)
            ? ((d.args_redacted ?? d.argsRedacted) as Record<string, unknown>)
            : {},
        }))
      : []
    return { toolCalls }
  },
}

export { toAgent }
