// Types for the Agents + MCP domain.
// See docs/tasks/agents-mcp-plan.md for the authoritative data model.

export type MCPTransport = "http" | "sse" | "stdio"
export type MCPAuthType = "none" | "api_key" | "bearer" | "basic" | "oauth2"
export type MCPHealthStatus = "unknown" | "healthy" | "unhealthy"

export type AgentToolChoice = "auto" | "required" | "none"
export type AgentRequireApproval = "never" | "always"

/** A tool exposed by an MCP server. `input_schema` is JSON Schema. */
export interface MCPToolSpec {
  name: string
  description: string
  inputSchema: Record<string, unknown>
}

export interface MCPServer {
  id: string
  alias: string
  displayName: string
  description: string | null
  transport: MCPTransport
  url: string | null
  command: string | null
  args: string[] | null
  env: Record<string, string> | null
  authType: MCPAuthType
  // We never return auth_value back; the form sends it on create/update only.
  allowedTools: string[] | null
  groupId: string | null
  groupName: string | null
  toolsCache: MCPToolSpec[] | null
  toolsCacheUpdatedAt: string | null
  healthStatus: MCPHealthStatus
  lastError: string | null
  createdAt: string
  updatedAt: string
}

export interface MCPServerCreate {
  alias: string
  displayName: string
  description?: string | null
  transport: MCPTransport
  url?: string | null
  command?: string | null
  args?: string[] | null
  env?: Record<string, string> | null
  authType: MCPAuthType
  authValue?: string | null
  allowedTools?: string[] | null
  groupId?: string | null
}

export interface MCPServerUpdate extends Partial<MCPServerCreate> {}

export interface MCPServerTestResult {
  healthy: boolean
  toolsCount: number
  error: string | null
}

export interface AgentMCPBinding {
  mcpServerId: string
  // `null` → inherit server-level allowed tools. A list restricts further.
  allowedTools: string[] | null
}

export interface Agent {
  id: string
  slug: string
  displayName: string
  description: string | null
  baseModelId: string | null
  profileId: string | null
  systemPrompt: string
  toolChoiceDefault: AgentToolChoice
  maxToolHops: number
  toolTimeoutSeconds: number
  requireApproval: AgentRequireApproval
  requestDefaults: Record<string, unknown> | null
  groupId: string | null
  groupName: string | null
  mcpServers: AgentMCPBinding[]
  createdAt: string
  updatedAt: string
}

export interface AgentCreate {
  slug: string
  displayName: string
  description?: string | null
  baseModelId?: string | null
  profileId?: string | null
  systemPrompt: string
  toolChoiceDefault: AgentToolChoice
  maxToolHops: number
  toolTimeoutSeconds: number
  requireApproval: AgentRequireApproval
  requestDefaults?: Record<string, unknown> | null
  groupId?: string | null
  mcpServers: AgentMCPBinding[]
}

export interface AgentUpdate extends Partial<AgentCreate> {}

export interface AgentTestServerResult {
  mcpServerId: string
  alias: string
  healthy: boolean
  toolsCount: number
  error: string | null
}

export interface AgentTestResult {
  healthy: boolean
  toolsCount: number
  servers: AgentTestServerResult[]
  baseModelReachable: boolean
  errors: string[]
}

// Stats types (Fase 5). Endpoints not yet implemented — see "Deudas abiertas" in plan.

export interface AgentStatRow {
  agentId: string
  slug: string
  displayName: string
  requestCount: number
  toolCallCount: number
  errorCount: number
  p50DurationMs: number | null
  p95DurationMs: number | null
  totalTokens: number
}

export interface ToolStatRow {
  mcpServerAlias: string
  toolName: string
  invocations: number
  errors: number
  p50DurationMs: number | null
  p95DurationMs: number | null
  errorRate: number
}

export interface ByAgentStats {
  byAgent: AgentStatRow[]
  byTool: ToolStatRow[]
}

export interface ToolCallRow {
  id: string
  createdAt: string
  agentId: string | null
  agentSlug: string | null
  mcpServerAlias: string
  toolName: string
  status: "ok" | "timeout" | "schema_error" | "mcp_error"
  durationMs: number
  hopIndex: number
  error: string | null
  argsRedacted: Record<string, unknown>
}

export interface ToolCallsData {
  toolCalls: ToolCallRow[]
}
