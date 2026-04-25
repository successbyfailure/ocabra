// @vitest-environment jsdom
import { render, screen, waitFor } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"
import { beforeEach, describe, expect, it, vi } from "vitest"

const { listAgents, listServers, listModels, listGroups } = vi.hoisted(() => ({
  listAgents: vi.fn(),
  listServers: vi.fn(),
  listModels: vi.fn(),
  listGroups: vi.fn(),
}))

vi.mock("@/api/agents", () => ({
  agentsApi: {
    list: listAgents,
    get: vi.fn(),
    create: vi.fn(),
    update: vi.fn(),
    delete: vi.fn(),
    test: vi.fn(),
  },
  agentStatsApi: {
    byAgent: vi.fn(),
    toolCalls: vi.fn(),
  },
}))

vi.mock("@/api/mcp", () => ({
  mcpApi: {
    list: listServers,
    get: vi.fn(),
    create: vi.fn(),
    update: vi.fn(),
    delete: vi.fn(),
    refresh: vi.fn(),
    test: vi.fn(),
    tools: vi.fn(),
  },
}))

vi.mock("@/api/client", () => ({
  api: {
    models: { list: listModels },
    groups: { list: listGroups },
  },
}))

// Default auth: model_manager
vi.mock("@/hooks/useAuth", () => ({
  useIsModelManager: () => true,
  useIsAdmin: () => false,
  useCurrentUser: () => ({ id: "u1", username: "admin", role: "model_manager" }),
}))

import { Agents } from "@/pages/Agents"
import { useAgentsStore } from "@/stores/agentsStore"
import { useMCPStore } from "@/stores/mcpStore"

describe("Agents page", () => {
  beforeEach(() => {
    useAgentsStore.setState({ agents: [], loading: false, usingMock: false, error: null })
    useMCPStore.setState({ servers: [], loading: false, usingMock: false, error: null })

    listAgents.mockResolvedValue([
      {
        id: "a1",
        slug: "research-bot",
        displayName: "Research bot",
        description: "test agent",
        baseModelId: "mistral-7b",
        profileId: null,
        systemPrompt: "You help.",
        toolChoiceDefault: "auto",
        maxToolHops: 8,
        toolTimeoutSeconds: 60,
        requireApproval: "never",
        requestDefaults: null,
        groupId: null,
        groupName: null,
        mcpServers: [],
        createdAt: "2026-04-24T00:00:00Z",
        updatedAt: "2026-04-24T00:00:00Z",
      },
    ])
    listServers.mockResolvedValue([])
    listModels.mockResolvedValue([])
    listGroups.mockResolvedValue([])
  })

  it("renders without crashing and shows agents from the API", async () => {
    render(
      <MemoryRouter>
        <Agents />
      </MemoryRouter>,
    )
    // Heading uses the word "Agents" — so does the sidebar; match on the h1 role.
    expect(screen.getByRole("heading", { level: 1, name: /agents/i })).toBeTruthy()
    await waitFor(() => {
      expect(screen.getByText(/Research bot/)).toBeTruthy()
    })
    expect(screen.getByText(/agent\/research-bot/)).toBeTruthy()
  })

  it("falls back to mock when the API returns 404", async () => {
    const notFound = Object.assign(new Error("Not Found"), { status: 404 })
    listAgents.mockRejectedValueOnce(notFound)
    render(
      <MemoryRouter>
        <Agents />
      </MemoryRouter>,
    )
    await waitFor(() => {
      // Mock fixture contains "research-bot" as well — assert the mock banner shows up.
      expect(screen.getByText(/Modo mock/)).toBeTruthy()
    })
  })
})
