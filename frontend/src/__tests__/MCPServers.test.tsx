// @vitest-environment jsdom
import { render, screen, waitFor } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"
import { beforeEach, describe, expect, it, vi } from "vitest"

const { listServers, listGroups } = vi.hoisted(() => ({
  listServers: vi.fn(),
  listGroups: vi.fn(),
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
    groups: { list: listGroups },
  },
}))

vi.mock("@/hooks/useAuth", () => ({
  useIsModelManager: () => true,
  useIsAdmin: () => true,
  useCurrentUser: () => ({ id: "u1", username: "admin", role: "system_admin" }),
}))

import { MCPServers } from "@/pages/MCPServers"
import { useMCPStore } from "@/stores/mcpStore"

describe("MCPServers page", () => {
  beforeEach(() => {
    useMCPStore.setState({ servers: [], loading: false, usingMock: false, error: null })

    listServers.mockResolvedValue([
      {
        id: "s1",
        alias: "github",
        displayName: "GitHub",
        description: "github mcp",
        transport: "http",
        url: "https://mcp.example.com",
        command: null,
        args: null,
        env: null,
        authType: "bearer",
        allowedTools: null,
        groupId: null,
        groupName: null,
        toolsCache: [],
        toolsCacheUpdatedAt: null,
        healthStatus: "healthy",
        lastError: null,
        createdAt: "2026-04-24T00:00:00Z",
        updatedAt: "2026-04-24T00:00:00Z",
      },
    ])
    listGroups.mockResolvedValue([])
  })

  it("renders the MCP servers list", async () => {
    render(
      <MemoryRouter>
        <MCPServers />
      </MemoryRouter>,
    )
    expect(screen.getByRole("heading", { level: 1, name: /mcp servers/i })).toBeTruthy()
    await waitFor(() => {
      expect(screen.getByText(/GitHub/)).toBeTruthy()
    })
    expect(screen.getByText(/healthy/)).toBeTruthy()
  })

  it("falls back to mock data when the API returns 404", async () => {
    const notFound = Object.assign(new Error("Not Found"), { status: 404 })
    listServers.mockRejectedValueOnce(notFound)
    render(
      <MemoryRouter>
        <MCPServers />
      </MemoryRouter>,
    )
    await waitFor(() => {
      expect(screen.getByText(/Modo mock/)).toBeTruthy()
    })
  })
})
