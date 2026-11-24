/**
 * Playground: run requests on a chosen federation node ("auto" = normal
 * routing, "local" = this server, or a peer name). Sent as the
 * ``X-Ocabra-Node`` header, which the server honours per request — handy to
 * compare the same model on two nodes. Kept per tab.
 */
import { useSyncExternalStore } from "react"
import { httpFetch } from "@/api/http"

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

/**
 * ``httpFetch`` (capa HTTP única: enrutado al nodo remoto + 401) que además
 * añade ``X-Ocabra-Node`` cuando se ha elegido un nodo del Playground. Acepta
 * ``init.signal`` para poder cancelar la petición (botón Detener).
 */
export function playgroundFetch(input: string, init: RequestInit = {}): Promise<Response> {
  const node = getPlaygroundNode()
  if (node === "auto") return httpFetch(input, init)
  const headers = new Headers(init.headers)
  headers.set("X-Ocabra-Node", node)
  return httpFetch(input, { ...init, headers })
}
