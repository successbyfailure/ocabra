/**
 * Capa HTTP única de la UI (REF-14). Antes convivían tres helpers ``request()``
 * casi idénticos (``api/client.ts``, ``api/mcp.ts``, ``api/agents.ts``), ~16
 * ``fetch`` crudos y un parche global de ``window.fetch`` en ``lib/remoteNode``
 * que reescribía *todas* las peticiones al nodo remoto. Esto unifica:
 *
 *  - **Enrutado al nodo remoto**: cada ruta pasa por ``remotePath`` aquí, sin
 *    parchear ``window.fetch`` ni ``EventSource`` (ver ``sseUrl``).
 *  - **401 → cierre de sesión**: una sesión caducada limpia el usuario para que
 *    ``ProtectedRoute`` redirija a /login (salvo en ``/ocabra/auth/*``).
 *  - **Forma única de error**: ``extractErrorMessage`` desenvuelve tanto el
 *    ``{detail: "..."}`` de oCabra como el ``{error:{message}}`` de la API
 *    OpenAI, de modo que ningún ``toast`` vuelve a mostrar "[object Object]".
 *  - **AbortSignal**: ``request`` acepta ``signal`` para poder cancelar.
 */
import { remotePath } from "@/lib/remoteNode"
import { useAuthStore } from "@/stores/authStore"

const BASE = ""

export interface HttpError extends Error {
  status?: number
}

export interface RequestOptions {
  signal?: AbortSignal
  headers?: Record<string, string>
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null
}

/**
 * FastAPI's HTTPException(detail=...) always wraps the response as
 * {"detail": <value>}. oCabra's own /ocabra/* endpoints raise with a plain
 * string detail, but the OpenAI-compatible /v1/* endpoints (batches, files,
 * chat, ...) raise with detail={"error": {"message": ...}} to match the
 * OpenAI error shape — unwrap both so callers never do `new Error(object)`
 * (which renders as the literal string "[object Object]" in a toast).
 */
export function extractErrorMessage(err: unknown): string | undefined {
  if (!isRecord(err)) return undefined
  const detail = err.detail
  if (typeof detail === "string") return detail
  // /ocabra/* validation errors: {"detail": {"message": ..., "errors": [...]}}
  if (isRecord(detail) && typeof detail.message === "string") return detail.message
  if (isRecord(detail) && isRecord(detail.error) && typeof detail.error.message === "string") {
    return detail.error.message
  }
  if (isRecord(err.error) && typeof err.error.message === "string") return err.error.message
  if (typeof err.message === "string") return err.message
  return undefined
}

/**
 * Low-level fetch that routes the path through the remote-node proxy and clears
 * the session on 401. Returns the raw ``Response`` — use it for streaming,
 * file uploads/downloads and anything that is not a JSON body. For JSON use
 * ``request`` instead.
 */
export async function httpFetch(path: string, init: RequestInit = {}): Promise<Response> {
  const res = await fetch(`${BASE}${remotePath(path)}`, init)
  // Session gone → clear user so ProtectedRoute redirects to /login and the
  // dashboard's polls unmount instead of looping on 401. Skip /ocabra/auth/*
  // so a failed login doesn't nuke an already-null user and the login form
  // keeps its own error flow.
  if (res.status === 401 && !path.startsWith("/ocabra/auth/")) {
    useAuthStore.getState().setUser(null)
  }
  return res
}

/** Throw a single, readable ``HttpError`` (with ``.status``) from a bad response. */
export async function raiseForStatus(res: Response): Promise<never> {
  const body = await res.json().catch(() => ({ detail: res.statusText }))
  const error = new Error(extractErrorMessage(body) ?? res.statusText) as HttpError
  error.status = res.status
  throw error
}

/** JSON request helper used by every typed client. */
export async function request<T>(
  method: string,
  path: string,
  body?: unknown,
  opts: RequestOptions = {},
): Promise<T> {
  const headers: Record<string, string> = { ...opts.headers }
  if (body !== undefined) headers["Content-Type"] = "application/json"
  const res = await httpFetch(path, {
    method,
    headers,
    body: body !== undefined ? JSON.stringify(body) : undefined,
    signal: opts.signal,
  })
  if (!res.ok) await raiseForStatus(res)
  if (res.status === 204) return undefined as T
  return res.json() as Promise<T>
}

/** URL for an ``EventSource``, routed through the remote-node proxy when active
 *  (``EventSource`` can only carry cookies, so the proxy path is all it needs). */
export function sseUrl(path: string): string {
  return `${BASE}${remotePath(path)}`
}
