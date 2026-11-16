/**
 * Playground: run requests on a chosen federation node ("auto" = normal
 * routing, "local" = this server, or a peer name). Sent as the
 * ``X-Ocabra-Node`` header, which the server honours per request — handy to
 * compare the same model on two nodes. Kept per tab.
 */
import { useSyncExternalStore } from "react"

const KEY = "ocabra.playgroundNode"
const listeners = new Set<() => void>()

export function getPlaygroundNode(): string {
  try {
    return sessionStorage.getItem(KEY) || "auto"
  } catch {
    return "auto"
  }
}

export function setPlaygroundNode(node: string): void {
  try {
    if (node === "auto") sessionStorage.removeItem(KEY)
    else sessionStorage.setItem(KEY, node)
  } catch {
    // storage unavailable: the choice just doesn't stick
  }
  listeners.forEach((l) => l())
}

export function usePlaygroundNode(): string {
  return useSyncExternalStore(
    (l) => {
      listeners.add(l)
      return () => listeners.delete(l)
    },
    getPlaygroundNode,
    () => "auto",
  )
}

/** ``fetch`` that adds ``X-Ocabra-Node`` when a node is chosen. */
export function playgroundFetch(input: string, init: RequestInit = {}): Promise<Response> {
  const node = getPlaygroundNode()
  if (node === "auto") return fetch(input, init)
  const headers = new Headers(init.headers)
  headers.set("X-Ocabra-Node", node)
  return fetch(input, { ...init, headers })
}
