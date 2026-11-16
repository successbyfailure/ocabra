/**
 * Remote panel: this UI showing another node of the federation.
 *
 * While a remote node is selected, every API call of the UI goes to
 * ``/ocabra/federation/peers/{id}/proxy/<path>`` on *this* server, which
 * forwards it to that node through the federation link (so it works for nodes
 * without a reachable URL or without a web UI, like cloud pods). Login and the
 * federation/cloud pages always stay local.
 *
 * Kept in sessionStorage (per tab) and applied with a full reload, so no
 * store ever mixes data from two servers.
 */
const KEY = "ocabra.remoteNode"

export interface RemoteNode {
  id: string
  name: string
}

export function getRemoteNode(): RemoteNode | null {
  try {
    const raw = sessionStorage.getItem(KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw) as RemoteNode
    return parsed && typeof parsed.id === "string" ? parsed : null
  } catch {
    return null
  }
}

export function enterRemoteNode(node: RemoteNode, to = "/dashboard"): void {
  sessionStorage.setItem(KEY, JSON.stringify({ id: node.id, name: node.name }))
  window.location.assign(to)
}

export function leaveRemoteNode(to = "/federation"): void {
  sessionStorage.removeItem(KEY)
  window.location.assign(to)
}

// Always answered by this server, never by the remote one.
const LOCAL_ONLY = ["/ocabra/auth/", "/ocabra/federation", "/ocabra/cloud"]

/** Rewrite an API path to go through the remote panel proxy when active. */
export function remotePath(path: string, node: RemoteNode | null = getRemoteNode()): string {
  if (!node) return path
  if (!/^\/(ocabra|v1|api)\//.test(path)) return path
  if (LOCAL_ONLY.some((p) => path.startsWith(p))) return path
  return `/ocabra/federation/peers/${encodeURIComponent(node.id)}/proxy${path}`
}

/** UI sections that make no sense (or are refused) on a remote node. */
export const REMOTE_HIDDEN_ROUTES = new Set([
  "/federation",
  "/users",
  "/groups",
  "/api-keys",
  "/sessions",
  "/settings",
])

function rewriteUrl(url: string, node: RemoteNode): string {
  if (url.startsWith("/")) return remotePath(url, node)
  try {
    const u = new URL(url)
    if (u.origin !== window.location.origin) return url
    return `${u.origin}${remotePath(u.pathname, node)}${u.search}${u.hash}`
  } catch {
    return url
  }
}

let installed = false

/**
 * In remote mode, send *every* ``fetch`` and ``EventSource`` of the UI to the
 * remote node, not only the ones made through ``api/client.ts``: pages with
 * their own fetch helpers (MCP, agents), the Playground and the SSE streams
 * used to talk to this server instead, so e.g. the MCP page of a pod listed
 * ours. ``remotePath`` is idempotent, so calls already rewritten pass as is.
 */
export function installRemoteRouting(node: RemoteNode | null = getRemoteNode()): void {
  if (!node || installed || typeof window === "undefined") return
  installed = true
  const target: RemoteNode = node
  const nativeFetch = window.fetch.bind(window)
  window.fetch = (input: RequestInfo | URL, init?: RequestInit) => {
    if (typeof input === "string") return nativeFetch(rewriteUrl(input, node), init)
    if (input instanceof URL) return nativeFetch(rewriteUrl(input.toString(), node), init)
    const url = rewriteUrl(input.url, node)
    return nativeFetch(url === input.url ? input : new Request(url, input), init)
  }
  const NativeEventSource = window.EventSource
  if (NativeEventSource) {
    class RemoteEventSource extends NativeEventSource {
      constructor(url: string | URL, config?: EventSourceInit) {
        super(rewriteUrl(url.toString(), target), config)
      }
    }
    window.EventSource = RemoteEventSource as typeof EventSource
  }
}
