import { create } from "zustand"
import { agentsApi } from "@/api/agents"
import type { Agent, AgentCreate, AgentUpdate } from "@/types/agents"

// TODO: remove mock once Stream A merges /ocabra/agents.
const MOCK_AGENTS: Agent[] = [
  {
    id: "mock-agent-1",
    slug: "research-bot",
    displayName: "Research bot",
    description: "Example agent with filesystem access — seeded from mock fallback.",
    baseModelId: "mistral-7b",
    profileId: null,
    systemPrompt:
      "You are a research assistant. Use the filesystem tools to ground answers in local documentation.",
    toolChoiceDefault: "auto",
    maxToolHops: 8,
    toolTimeoutSeconds: 60,
    requireApproval: "never",
    requestDefaults: { temperature: 0.3 },
    groupId: null,
    groupName: null,
    mcpServers: [{ mcpServerId: "mock-mcp-1", allowedTools: ["list_dir", "read_file"] }],
    createdAt: "2026-04-22T12:00:00Z",
    updatedAt: "2026-04-24T09:00:00Z",
  },
]

function sortAgents(list: Agent[]): Agent[] {
  return [...list].sort((a, b) => a.slug.localeCompare(b.slug))
}

function isNotFoundError(err: unknown): boolean {
  const status = (err as { status?: number } | null)?.status
  return status === 404 || status === 501
}

interface AgentsStore {
  agents: Agent[]
  loading: boolean
  usingMock: boolean
  error: string | null
  fetchAll: () => Promise<void>
  create: (data: AgentCreate) => Promise<Agent>
  update: (slug: string, data: AgentUpdate) => Promise<Agent>
  remove: (slug: string) => Promise<void>
  upsert: (agent: Agent) => void
  handleWSEvent: (type: string, data: unknown) => void
}

export const useAgentsStore = create<AgentsStore>((set, get) => ({
  agents: [],
  loading: false,
  usingMock: false,
  error: null,

  fetchAll: async () => {
    set({ loading: true, error: null })
    const meta = import.meta as unknown as { env?: Record<string, string | undefined> }
    if (meta.env?.VITE_MOCK_AGENTS === "1") {
      set({ agents: sortAgents(MOCK_AGENTS), loading: false, usingMock: true, error: null })
      return
    }
    try {
      const data = await agentsApi.list()
      set({ agents: sortAgents(data), loading: false, usingMock: false })
    } catch (err) {
      if (isNotFoundError(err)) {
        // TODO: remove mock once Stream A merges /ocabra/agents.
        set({
          agents: sortAgents(MOCK_AGENTS),
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

  upsert: (agent) =>
    set((prev) => {
      const without = prev.agents.filter((a) => a.slug !== agent.slug)
      return { agents: sortAgents([...without, agent]) }
    }),

  create: async (data) => {
    if (get().usingMock) {
      const created: Agent = {
        id: `mock-${Date.now()}`,
        slug: data.slug,
        displayName: data.displayName,
        description: data.description ?? null,
        baseModelId: data.baseModelId ?? null,
        profileId: data.profileId ?? null,
        systemPrompt: data.systemPrompt,
        toolChoiceDefault: data.toolChoiceDefault,
        maxToolHops: data.maxToolHops,
        toolTimeoutSeconds: data.toolTimeoutSeconds,
        requireApproval: data.requireApproval,
        requestDefaults: data.requestDefaults ?? null,
        groupId: data.groupId ?? null,
        groupName: null,
        mcpServers: data.mcpServers,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      }
      get().upsert(created)
      return created
    }
    const created = await agentsApi.create(data)
    get().upsert(created)
    return created
  },

  update: async (slug, data) => {
    if (get().usingMock) {
      const existing = get().agents.find((a) => a.slug === slug)
      if (!existing) throw new Error("not found")
      const merged: Agent = {
        ...existing,
        ...(data.slug !== undefined && { slug: data.slug }),
        ...(data.displayName !== undefined && { displayName: data.displayName }),
        ...(data.description !== undefined && { description: data.description ?? null }),
        ...(data.baseModelId !== undefined && { baseModelId: data.baseModelId ?? null }),
        ...(data.profileId !== undefined && { profileId: data.profileId ?? null }),
        ...(data.systemPrompt !== undefined && { systemPrompt: data.systemPrompt }),
        ...(data.toolChoiceDefault !== undefined && { toolChoiceDefault: data.toolChoiceDefault }),
        ...(data.maxToolHops !== undefined && { maxToolHops: data.maxToolHops }),
        ...(data.toolTimeoutSeconds !== undefined && {
          toolTimeoutSeconds: data.toolTimeoutSeconds,
        }),
        ...(data.requireApproval !== undefined && { requireApproval: data.requireApproval }),
        ...(data.requestDefaults !== undefined && {
          requestDefaults: data.requestDefaults ?? null,
        }),
        ...(data.groupId !== undefined && { groupId: data.groupId ?? null }),
        ...(data.mcpServers !== undefined && { mcpServers: data.mcpServers }),
        updatedAt: new Date().toISOString(),
      }
      get().upsert(merged)
      return merged
    }
    const updated = await agentsApi.update(slug, data)
    get().upsert(updated)
    return updated
  },

  remove: async (slug) => {
    if (get().usingMock) {
      set((prev) => ({ agents: prev.agents.filter((a) => a.slug !== slug) }))
      return
    }
    await agentsApi.delete(slug)
    set((prev) => ({ agents: prev.agents.filter((a) => a.slug !== slug) }))
  },

  handleWSEvent: (type, data) => {
    if (type !== "agent_updated") return
    const payload = (data ?? {}) as Record<string, unknown>
    const slug = String(payload.slug ?? "")
    if (!slug) return
    // Lazy refresh: simpler than normalizing shapes here.
    void get().fetchAll()
  },
}))
