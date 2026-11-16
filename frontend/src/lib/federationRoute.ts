/** Human-readable reasons for where a request was served (federation). */
const ROUTES: Record<string, string> = {
  local_loaded: "Cargado aquí",
  local_loaded_peer_lower_load: "Cargado aquí, pero otro nodo estaba más libre",
  local_cold_peer_warm: "No cargado aquí; otro nodo lo tenía cargado",
  no_peer_has_model: "Nadie lo tenía cargado: se cargó aquí",
  forced: "Nodo elegido a mano (X-Ocabra-Node)",
  forced_local: "Forzado a este servidor (X-Ocabra-Node)",
  fallback_local: "El nodo elegido falló: se sirvió aquí",
}

export function describeRoute(route: string | null | undefined): string {
  if (!route) return ""
  return ROUTES[route] ?? route
}
