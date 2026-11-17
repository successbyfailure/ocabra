import { create } from "zustand"
import { agentsApi } from "@/api/agents"
import type { Agent, AgentCreate, AgentUpdate } from "@/types/agents"

function sortAgents(list: Agent[]): Agent[] {
  return [...list].sort((a, b) => a.slug.localeCompare(b.slug))
}

interface AgentsStore {
  agents: Agent[]
  loading: boolean
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
  error: null,

  fetchAll: async () => {
    set({ loading: true, error: null })
    try {
      const data = await agentsApi.list()
      set({ agents: sortAgents(data), loading: false })
    } catch (err) {
      // A failure (also a 404 from an older node in the remote panel) is
      // shown as an error. It used to swap in made-up example agents whose
      // edits only lived in memory.
      const message = err instanceof Error ? err.message : "unknown error"
      set({ loading: false, error: message })
    }
  },

  upsert: (agent) =>
    set((prev) => {
      const without = prev.agents.filter((a) => a.slug !== agent.slug)
      return { agents: sortAgents([...without, agent]) }
    }),

  create: async (data) => {
    const created = await agentsApi.create(data)
    get().upsert(created)
    return created
  },

  update: async (slug, data) => {
    const updated = await agentsApi.update(slug, data)
    get().upsert(updated)
    return updated
  },

  remove: async (slug) => {
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
