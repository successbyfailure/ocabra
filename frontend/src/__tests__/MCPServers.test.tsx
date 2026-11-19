// @vitest-environment jsdom
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { MemoryRouter } from "react-router"
import { beforeEach, describe, expect, it, vi } from "vitest"

const { listServers, listGroups, updateServer, refreshServer } = vi.hoisted(() => ({
  listServers: vi.fn(),
  listGroups: vi.fn(),
  updateServer: vi.fn(),
  refreshServer: vi.fn(),
}))

vi.mock("@/api/mcp", () => ({
  mcpApi: {
    list: listServers,
    get: vi.fn(),
    create: vi.fn(),
    update: updateServer,
    delete: vi.fn(),
    refresh: refreshServer,
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
    useMCPStore.setState({ servers: [], loading: false, error: null })

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
        hasEnv: false,
        hasAuth: true,
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

  // BUG-47
  it("shows a 404 as an error, without made-up servers", async () => {
    const notFound = Object.assign(new Error("Not Found"), { status: 404 })
    listServers.mockRejectedValueOnce(notFound)
    render(
      <MemoryRouter>
        <MCPServers />
      </MemoryRouter>,
    )
    await waitFor(() => {
      expect(screen.getByText(/Not Found/)).toBeTruthy()
    })
    expect(screen.queryByText(/Modo mock/)).toBeNull()
  })

  // BUG-01: the API never returns the env values, so the field always opened
  // empty and saving sent env: null — renaming a server deleted its secrets.
  it("editing a stdio server keeps its saved env unless asked to clear it", async () => {
    const stdio = {
      id: "s2",
      alias: "files",
      displayName: "Files",
      description: null,
      transport: "stdio",
      url: null,
      command: "mcp-server-filesystem",
      args: ["/data"],
      hasEnv: true,
      hasAuth: false,
      authType: "none",
      allowedTools: null,
      groupId: null,
      groupName: null,
      toolsCache: [],
      toolsCacheUpdatedAt: null,
      healthStatus: "healthy",
      lastError: null,
      createdAt: "2026-04-24T00:00:00Z",
      updatedAt: "2026-04-24T00:00:00Z",
    }
    listServers.mockResolvedValue([stdio])
    updateServer.mockResolvedValue(stdio)
    refreshServer.mockResolvedValue(stdio)
    render(
      <MemoryRouter>
        <MCPServers />
      </MemoryRouter>,
    )
    fireEvent.click(await screen.findByTitle("Editar"))
    fireEvent.change(screen.getByLabelText(/Display name/i), { target: { value: "Files renamed" } })
    fireEvent.click(screen.getByRole("button", { name: "Guardar" }))

    await waitFor(() => expect(updateServer).toHaveBeenCalledTimes(1))
    const sent = updateServer.mock.calls[0][1]
    expect(sent.displayName).toBe("Files renamed")
    expect("env" in sent).toBe(false)

    // Ticking "Borrar" does send env: null.
    fireEvent.click(await screen.findByTitle("Editar"))
    fireEvent.click(screen.getByLabelText(/Borrar las variables guardadas/))
    fireEvent.click(screen.getByRole("button", { name: "Guardar" }))
    await waitFor(() => expect(updateServer).toHaveBeenCalledTimes(2))
    expect(updateServer.mock.calls[1][1].env).toBeNull()
  })
})
