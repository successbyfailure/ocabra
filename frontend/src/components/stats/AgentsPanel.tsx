import { useEffect, useMemo, useState } from "react"
import { Sparkles } from "lucide-react"
import { agentStatsApi } from "@/api/agents"
import type { AgentStatRow, ByAgentStats, ToolCallRow, ToolStatRow } from "@/types/agents"

// TODO: remove mock once backend exposes /ocabra/stats/by-agent and
// /ocabra/stats/tool-calls. See docs/tasks/agents-mcp-plan.md "Deudas abiertas".
const MOCK_BY_AGENT: ByAgentStats = {
  byAgent: [
    {
      agentId: "mock-agent-1",
      slug: "research-bot",
      displayName: "Research bot",
      requestCount: 142,
      toolCallCount: 318,
      errorCount: 4,
      p50DurationMs: 1800,
      p95DurationMs: 5200,
      totalTokens: 284000,
    },
  ],
  byTool: [
    {
      mcpServerAlias: "filesystem",
      toolName: "read_file",
      invocations: 210,
      errors: 2,
      p50DurationMs: 35,
      p95DurationMs: 120,
      errorRate: 0.0095,
    },
    {
      mcpServerAlias: "filesystem",
      toolName: "list_dir",
      invocations: 88,
      errors: 0,
      p50DurationMs: 22,
      p95DurationMs: 65,
      errorRate: 0,
    },
    {
      mcpServerAlias: "github",
      toolName: "list_repos",
      invocations: 20,
      errors: 2,
      p50DurationMs: 180,
      p95DurationMs: 820,
      errorRate: 0.1,
    },
  ],
}

const MOCK_TOOL_CALLS: ToolCallRow[] = [
  {
    id: "tc-mock-1",
    createdAt: new Date(Date.now() - 3 * 60 * 1000).toISOString(),
    agentId: "mock-agent-1",
    agentSlug: "research-bot",
    mcpServerAlias: "filesystem",
    toolName: "read_file",
    status: "ok",
    durationMs: 42,
    hopIndex: 2,
    error: null,
    argsRedacted: { path: "/data/docs/readme.md" },
  },
  {
    id: "tc-mock-2",
    createdAt: new Date(Date.now() - 9 * 60 * 1000).toISOString(),
    agentId: "mock-agent-1",
    agentSlug: "research-bot",
    mcpServerAlias: "github",
    toolName: "list_repos",
    status: "mcp_error",
    durationMs: 412,
    hopIndex: 1,
    error: "401 Unauthorized",
    argsRedacted: {},
  },
]

function fmtMs(value: number | null): string {
  if (value == null) return "—"
  if (value < 1000) return `${Math.round(value)}ms`
  return `${(value / 1000).toFixed(2)}s`
}

function fmtPct(value: number): string {
  return `${(value * 100).toFixed(2)}%`
}

interface AgentsPanelProps {
  from: string
  to: string
}

export function AgentsPanel({ from, to }: AgentsPanelProps) {
  const [loading, setLoading] = useState(true)
  const [usingMock, setUsingMock] = useState(false)
  const [byAgent, setByAgent] = useState<AgentStatRow[]>([])
  const [byTool, setByTool] = useState<ToolStatRow[]>([])
  const [selectedAgentId, setSelectedAgentId] = useState<string | null>(null)
  const [toolCalls, setToolCalls] = useState<ToolCallRow[]>([])

  useEffect(() => {
    let active = true
    setLoading(true)
    agentStatsApi
      .byAgent({ from, to })
      .then((data) => {
        if (!active) return
        setByAgent(data.byAgent)
        setByTool(data.byTool)
        setUsingMock(false)
      })
      .catch(() => {
        // TODO: remove mock once backend exposes /ocabra/stats/by-agent.
        if (!active) return
        setByAgent(MOCK_BY_AGENT.byAgent)
        setByTool(MOCK_BY_AGENT.byTool)
        setUsingMock(true)
      })
      .finally(() => {
        if (active) setLoading(false)
      })
    return () => {
      active = false
    }
  }, [from, to])

  useEffect(() => {
    if (!selectedAgentId) {
      setToolCalls([])
      return
    }
    let active = true
    agentStatsApi
      .toolCalls({ agentId: selectedAgentId, limit: 30 })
      .then((data) => {
        if (active) setToolCalls(data.toolCalls)
      })
      .catch(() => {
        // TODO: remove mock once backend exposes /ocabra/stats/tool-calls.
        if (active) setToolCalls(MOCK_TOOL_CALLS.filter((tc) => tc.agentId === selectedAgentId))
      })
    return () => {
      active = false
    }
  }, [selectedAgentId])

  const sortedTools = useMemo(
    () => [...byTool].sort((a, b) => b.invocations - a.invocations),
    [byTool],
  )

  return (
    <div className="space-y-4">
      {usingMock && (
        <div className="rounded-md border border-amber-500/40 bg-amber-500/10 p-3 text-xs text-amber-100">
          Modo mock: los endpoints <code>/ocabra/stats/by-agent</code> y{" "}
          <code>/ocabra/stats/tool-calls</code> aún no existen. Ver deudas abiertas en
          <code> docs/tasks/agents-mcp-plan.md</code>.
        </div>
      )}

      {loading ? (
        <div className="h-40 animate-pulse rounded-lg bg-muted" />
      ) : (
        <>
          <div className="rounded-lg border border-border bg-card p-3">
            <h3 className="mb-3 flex items-center gap-2 text-sm font-semibold text-muted-foreground">
              <Sparkles size={14} />
              Top agents
            </h3>
            {byAgent.length === 0 ? (
              <p className="text-sm text-muted-foreground">Sin actividad.</p>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-xs uppercase tracking-wide text-muted-foreground">
                      <th className="px-2 py-1">Agent</th>
                      <th className="px-2 py-1 text-right">Requests</th>
                      <th className="px-2 py-1 text-right">Tool calls</th>
                      <th className="px-2 py-1 text-right">Errores</th>
                      <th className="px-2 py-1 text-right">P50</th>
                      <th className="px-2 py-1 text-right">P95</th>
                      <th className="px-2 py-1 text-right">Tokens</th>
                    </tr>
                  </thead>
                  <tbody>
                    {byAgent.map((row) => (
                      <tr
                        key={row.agentId}
                        onClick={() => setSelectedAgentId(row.agentId)}
                        className={`cursor-pointer border-t border-border hover:bg-muted/40 ${
                          selectedAgentId === row.agentId ? "bg-primary/10" : ""
                        }`}
                      >
                        <td className="px-2 py-1.5">
                          <div className="font-medium">{row.displayName}</div>
                          <div className="font-mono text-xs text-muted-foreground">
                            agent/{row.slug}
                          </div>
                        </td>
                        <td className="px-2 py-1.5 text-right">{row.requestCount}</td>
                        <td className="px-2 py-1.5 text-right">{row.toolCallCount}</td>
                        <td className="px-2 py-1.5 text-right">{row.errorCount}</td>
                        <td className="px-2 py-1.5 text-right">{fmtMs(row.p50DurationMs)}</td>
                        <td className="px-2 py-1.5 text-right">{fmtMs(row.p95DurationMs)}</td>
                        <td className="px-2 py-1.5 text-right">
                          {row.totalTokens.toLocaleString()}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          <div className="rounded-lg border border-border bg-card p-3">
            <h3 className="mb-3 text-sm font-semibold text-muted-foreground">Top tools</h3>
            {sortedTools.length === 0 ? (
              <p className="text-sm text-muted-foreground">Sin actividad.</p>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-xs uppercase tracking-wide text-muted-foreground">
                      <th className="px-2 py-1">Tool</th>
                      <th className="px-2 py-1 text-right">Invocaciones</th>
                      <th className="px-2 py-1 text-right">P50</th>
                      <th className="px-2 py-1 text-right">P95</th>
                      <th className="px-2 py-1 text-right">Error rate</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sortedTools.map((tool) => (
                      <tr
                        key={`${tool.mcpServerAlias}.${tool.toolName}`}
                        className="border-t border-border"
                      >
                        <td className="px-2 py-1.5 font-mono text-xs">
                          {tool.mcpServerAlias}.{tool.toolName}
                        </td>
                        <td className="px-2 py-1.5 text-right">{tool.invocations}</td>
                        <td className="px-2 py-1.5 text-right">{fmtMs(tool.p50DurationMs)}</td>
                        <td className="px-2 py-1.5 text-right">{fmtMs(tool.p95DurationMs)}</td>
                        <td className="px-2 py-1.5 text-right">{fmtPct(tool.errorRate)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {selectedAgentId && (
            <div className="rounded-lg border border-border bg-card p-3">
              <h3 className="mb-3 text-sm font-semibold text-muted-foreground">
                Tool calls recientes ·{" "}
                {byAgent.find((a) => a.agentId === selectedAgentId)?.slug ?? selectedAgentId}
              </h3>
              {toolCalls.length === 0 ? (
                <p className="text-sm text-muted-foreground">Sin tool calls en el rango.</p>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="text-left text-xs uppercase tracking-wide text-muted-foreground">
                        <th className="px-2 py-1">Fecha</th>
                        <th className="px-2 py-1">Tool</th>
                        <th className="px-2 py-1">Status</th>
                        <th className="px-2 py-1 text-right">Duración</th>
                        <th className="px-2 py-1 text-right">Hop</th>
                      </tr>
                    </thead>
                    <tbody>
                      {toolCalls.map((tc) => (
                        <tr key={tc.id} className="border-t border-border">
                          <td className="px-2 py-1.5 font-mono text-xs text-muted-foreground">
                            {new Date(tc.createdAt).toLocaleString()}
                          </td>
                          <td className="px-2 py-1.5 font-mono text-xs">
                            {tc.mcpServerAlias}.{tc.toolName}
                          </td>
                          <td className="px-2 py-1.5">
                            <span
                              className={`rounded-md px-2 py-0.5 text-xs ${
                                tc.status === "ok"
                                  ? "bg-emerald-500/20 text-emerald-200"
                                  : "bg-red-500/20 text-red-200"
                              }`}
                            >
                              {tc.status}
                            </span>
                            {tc.error && (
                              <span
                                className="ml-2 truncate font-mono text-xs text-muted-foreground"
                                title={tc.error}
                              >
                                {tc.error}
                              </span>
                            )}
                          </td>
                          <td className="px-2 py-1.5 text-right">{fmtMs(tc.durationMs)}</td>
                          <td className="px-2 py-1.5 text-right">{tc.hopIndex}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  )
}
