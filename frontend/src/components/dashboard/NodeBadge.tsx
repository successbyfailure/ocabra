import { describeRoute } from "@/lib/federationRoute"

/**
 * Where a request ran, for federated traffic: "→ pod" (served by another
 * node), "← Mapache" (another node sent it to us), or "respaldo local" (the
 * chosen node failed and it was served here). Nothing for plain local traffic.
 * The tooltip says why it went there.
 */
export function NodeBadge({
  node,
  inboundFrom,
  route,
}: {
  node?: string | null
  inboundFrom?: string | null
  route?: string | null
}) {
  const why = describeRoute(route)
  if (route === "fallback_local") {
    return (
      <span
        className="shrink-0 rounded border border-amber-500/40 px-1 text-[10px] text-amber-700 dark:text-amber-300"
        title={why}
      >
        respaldo local
      </span>
    )
  }
  if (node) {
    return (
      <span
        className="shrink-0 truncate rounded border border-sky-500/40 px-1 text-[10px] text-sky-700 dark:text-sky-300"
        title={why ? `Servida por ${node}: ${why}` : `Servida por ${node}`}
      >
        → {node}
      </span>
    )
  }
  if (inboundFrom) {
    return (
      <span
        className="shrink-0 truncate rounded border border-violet-500/40 px-1 text-[10px] text-violet-700 dark:text-violet-300"
        title={`Petición de ${inboundFrom}, servida aquí`}
      >
        ← {inboundFrom}
      </span>
    )
  }
  return null
}
