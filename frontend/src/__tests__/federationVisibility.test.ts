import { afterEach, describe, expect, it, vi } from "vitest"
import { toModelState } from "@/api/client"
import { describeRoute } from "@/lib/federationRoute"
import { getPlaygroundNode, playgroundFetch, setPlaygroundNode } from "@/lib/playgroundNode"
import { minutesLeft } from "@/components/layout/FederationStatusChip"
import type { CloudNode } from "@/types"

describe("model federation info", () => {
  it("says where a local model is loaded (the badge never showed before)", () => {
    const m = toModelState({
      model_id: "diffusers/qwen-image-2.1",
      status: "configured",
      federation: {
        local: true,
        remote: false,
        nodes: [
          { node_name: "local", node_id: "me", loaded: false },
          { node_name: "pod-4090", node_id: "p1", loaded: true },
        ],
      },
    })
    expect(m.federation?.remote).toBe(false)
    expect(m.federation?.loadedOn).toEqual(["pod-4090"])
  })

  it("marks models that only exist on other nodes", () => {
    const m = toModelState({
      model_id: "x/y",
      status: "remote",
      federation: { local: false, nodes: [{ node_name: "Mapache", node_id: "p2", loaded: true }] },
    })
    expect(m.federation?.remote).toBe(true)
    expect(m.federation?.nodeName).toBe("Mapache")
  })

  it("stays quiet for a model loaded only here", () => {
    const m = toModelState({
      model_id: "a/b",
      status: "loaded",
      federation: { local: true, nodes: [{ node_name: "local", node_id: "me", loaded: true }] },
    })
    expect(m.federation).toBeNull()
  })
})

describe("cloud node shutdown", () => {
  const base = {
    terminates_at: null,
    idle_shutdown_minutes: null,
    last_active_at: null,
    started_at: null,
    created_at: null,
  } as unknown as CloudNode
  const now = Date.parse("2026-11-16T12:00:00Z")

  it("takes the earliest of TTL and idle", () => {
    const node = {
      ...base,
      terminates_at: "2026-11-16T13:00:00Z",
      idle_shutdown_minutes: 30,
      last_active_at: "2026-11-16T11:50:00Z",
    }
    expect(minutesLeft(node, now)).toBe(20)
  })

  it("is unknown without either", () => {
    expect(minutesLeft(base, now)).toBeNull()
  })
})

describe("playground node", () => {
  afterEach(() => {
    setPlaygroundNode("auto")
    vi.restoreAllMocks()
  })

  it("adds X-Ocabra-Node only when a node is chosen", async () => {
    const calls: Headers[] = []
    vi.spyOn(globalThis, "fetch").mockImplementation(async (_url, init) => {
      calls.push(new Headers(init?.headers))
      return new Response("{}")
    })
    await playgroundFetch("/v1/chat/completions", { headers: { "Content-Type": "application/json" } })
    setPlaygroundNode("pod-4090")
    expect(getPlaygroundNode()).toBe("pod-4090")
    await playgroundFetch("/v1/chat/completions", { headers: { "Content-Type": "application/json" } })
    expect(calls[0].get("X-Ocabra-Node")).toBeNull()
    expect(calls[1].get("X-Ocabra-Node")).toBe("pod-4090")
    expect(calls[1].get("Content-Type")).toBe("application/json")
  })
})

describe("route descriptions", () => {
  it("explains known reasons and passes unknown ones through", () => {
    expect(describeRoute("fallback_local")).toMatch(/falló/)
    expect(describeRoute("something_new")).toBe("something_new")
    expect(describeRoute(null)).toBe("")
  })
})
