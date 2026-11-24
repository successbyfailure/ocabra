import { afterEach, describe, expect, it, vi } from "vitest"
import { extractErrorMessage, httpFetch, request, sseUrl } from "@/api/http"
import { useAuthStore } from "@/stores/authStore"

const REMOTE_KEY = "ocabra.remoteNode"

function mockFetch(impl: (path: string, init?: RequestInit) => Response) {
  const spy = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
    const path = typeof input === "string" ? input : input.toString()
    return Promise.resolve(impl(path, init))
  })
  vi.stubGlobal("fetch", spy)
  return spy
}

afterEach(() => {
  vi.unstubAllGlobals()
  sessionStorage.clear()
  useAuthStore.setState({ user: null })
})

describe("extractErrorMessage", () => {
  it("unwraps oCabra and OpenAI error shapes (never '[object Object]')", () => {
    expect(extractErrorMessage({ detail: "nope" })).toBe("nope")
    expect(extractErrorMessage({ detail: { message: "bad body" } })).toBe("bad body")
    expect(extractErrorMessage({ detail: { error: { message: "openai style" } } })).toBe(
      "openai style",
    )
    expect(extractErrorMessage({ error: { message: "top-level" } })).toBe("top-level")
    expect(extractErrorMessage("plain")).toBeUndefined()
  })
})

describe("httpFetch", () => {
  it("routes the path through the remote-node proxy when one is selected", async () => {
    sessionStorage.setItem(REMOTE_KEY, JSON.stringify({ id: "peer1", name: "Mapache" }))
    const spy = mockFetch(() => new Response("{}"))
    await httpFetch("/ocabra/models")
    expect(spy.mock.calls[0][0]).toBe("/ocabra/federation/peers/peer1/proxy/ocabra/models")
  })

  it("clears the session on 401 (but not for /ocabra/auth/*)", async () => {
    useAuthStore.setState({
      user: { id: "u", username: "x", role: "admin", createdAt: "" } as never,
    })
    mockFetch(() => new Response("{}", { status: 401 }))
    await httpFetch("/ocabra/models")
    expect(useAuthStore.getState().user).toBeNull()

    useAuthStore.setState({
      user: { id: "u", username: "x", role: "admin", createdAt: "" } as never,
    })
    await httpFetch("/ocabra/auth/login")
    expect(useAuthStore.getState().user).not.toBeNull()
  })
})

describe("request", () => {
  it("throws a readable HttpError with .status on failure", async () => {
    mockFetch(() => new Response(JSON.stringify({ detail: "boom" }), { status: 400 }))
    await expect(request("GET", "/ocabra/x")).rejects.toMatchObject({
      message: "boom",
      status: 400,
    })
  })

  it("passes an AbortSignal through to fetch", async () => {
    const spy = mockFetch(() => new Response("{}"))
    const ctrl = new AbortController()
    await request("GET", "/ocabra/x", undefined, { signal: ctrl.signal })
    expect((spy.mock.calls[0][1] as RequestInit).signal).toBe(ctrl.signal)
  })

  it("returns undefined for 204", async () => {
    mockFetch(() => new Response(null, { status: 204 }))
    await expect(request("DELETE", "/ocabra/x")).resolves.toBeUndefined()
  })
})

describe("sseUrl", () => {
  it("applies the remote-node proxy", () => {
    expect(sseUrl("/ocabra/downloads/j1/stream")).toBe("/ocabra/downloads/j1/stream")
    sessionStorage.setItem(REMOTE_KEY, JSON.stringify({ id: "peer1", name: "Mapache" }))
    expect(sseUrl("/ocabra/downloads/j1/stream")).toBe(
      "/ocabra/federation/peers/peer1/proxy/ocabra/downloads/j1/stream",
    )
  })
})
