import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import type { RecentRequest } from "@/types"

const row: RecentRequest = {
  id: "abc-123", modelId: "ravenx", requestedModel: "ravenx:normal", backendType: "llamacpp_mks", requestKind: "chat",
  endpointPath: "/v1/chat/completions", statusCode: 200, startedAt: new Date().toISOString(), durationMs: 2739,
  inputTokens: 5324, outputTokens: 347, error: null, userId: "u", username: "makespace", groupId: "g", groupName: "Makers",
  apiKeyName: "fulcrum-mks", clientAddr: "79.117.11.114", userAgent: "python-httpx/0.28.1", inFlight: false,
  generationMs: 2418, cachedTokens: 4759, runtimePreset: "normal", effectiveContext: 245760,
  promptTokensPerSecond: 2617, predictedTokensPerSecond: 143, runtimeTelemetry: { queue_ms: 0, prefill_ms: 215 },
}

const running: RecentRequest = {
  id: "run-1", modelId: "gemma4:26b", backendType: null, requestKind: "chat", endpointPath: null, statusCode: null,
  startedAt: new Date().toISOString(), durationMs: null, inputTokens: null, outputTokens: null, error: null, userId: null,
  username: "ana", groupId: null, groupName: null, apiKeyName: null, clientAddr: null, userAgent: null,
  inFlight: true, elapsedMs: 12_400, estimatedRemainingMs: 3000,
}

vi.mock("@/api/client", () => ({
  api: { stats: { recent: vi.fn().mockResolvedValue({ requests: [running, row] }), myRecent: vi.fn() } },
}))
vi.mock("@/hooks/useAuth", () => ({ useIsModelManager: () => true }))
vi.mock("@/stores/gpuStore", () => ({ useGpuStore: (sel: (s: { gpus: [] }) => unknown) => sel({ gpus: [] }) }))

describe("IP y user agent", () => {
  it("no salen en Últimas peticiones del dashboard", async () => {
    const { RecentRequestsTable } = await import("@/components/dashboard/RecentRequestsTable")
    const { container } = render(<RecentRequestsTable />)
    await screen.findByText("fulcrum-mks", { exact: false })
    expect(container.textContent).not.toContain("79.117.11.114")
    expect(container.textContent).not.toContain("python-httpx")
  })

  it("una petición en curso dice que está en curso, no muestra guiones vacíos", async () => {
    const { Logs } = await import("@/pages/Logs")
    render(<Logs />)
    expect((await screen.findByTestId("log-inflight")).textContent).toContain("en curso · 12 s")
    fireEvent.click(screen.getByText("gemma4:26b"))
    const detail = await screen.findByTestId("log-detail")
    expect(detail.textContent).toContain("Lleva: 12 s")
    expect(detail.textContent).toContain("Le queda: ~3.0 s")
    expect(detail.textContent).not.toContain("Endpoint")
  })

  it("salen en el detalle de Logs, con grupo, nombre pedido, peldaño e ID", async () => {
    const { Logs } = await import("@/pages/Logs")
    render(<Logs />)
    fireEvent.click(await screen.findByText("ravenx"))
    const detail = await screen.findByTestId("log-detail")
    for (const text of ["79.117.11.114", "python-httpx/0.28.1", "Makers", "ravenx:normal", "normal", "abc-123", "4759 de 5324"]) {
      expect(detail.textContent).toContain(text)
    }
  })
})
