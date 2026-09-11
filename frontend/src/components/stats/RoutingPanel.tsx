import { useEffect, useState } from "react"
import { ArrowRight, GitBranch } from "lucide-react"
import type { RoutingDecisionsResponse } from "@/api/client"
import { api } from "@/api/client"

interface Props {
  from: string
  to: string
}

/**
 * Aggregated view of how Router profiles decided over the selected window.
 *
 * A row per router shows the fanout: how many requests reached each
 * target profile. Useful for spotting "gemma4:26b is meant to prefer
 * vLLM but 90% of traffic went to the Ollama fallback — the vLLM
 * backend must have been unhealthy or evicted."
 */
export function RoutingPanel({ from, to }: Props) {
  const [data, setData] = useState<RoutingDecisionsResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let active = true
    setLoading(true)
    setError(null)
    api
      .routingStats({ from, to })
      .then((result) => {
        if (active) setData(result)
      })
      .catch((e) => {
        if (active) setError(e instanceof Error ? e.message : String(e))
      })
      .finally(() => {
        if (active) setLoading(false)
      })
    return () => {
      active = false
    }
  }, [from, to])

  if (loading && !data) {
    return <div className="text-sm text-muted-foreground">Cargando decisiones de routing...</div>
  }
  if (error) {
    return <div className="text-sm text-destructive">Error: {error}</div>
  }
  const routers = data?.routers ?? []
  if (routers.length === 0) {
    return (
      <div className="rounded-md border border-border bg-muted/20 px-4 py-8 text-center text-sm text-muted-foreground">
        <GitBranch size={20} className="mx-auto mb-2 opacity-60" />
        No hay decisiones de router en el rango seleccionado.
        <p className="mx-auto mt-1 max-w-md text-xs">
          Cuando un cliente use un profile con <code>routing_targets</code>, aparecerá aquí
          el desglose de qué destino sirvió cada petición.
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {routers.map((r) => (
        <div key={r.routerProfileId} className="rounded-lg border border-border bg-card p-4">
          <div className="mb-3 flex items-center justify-between">
            <div>
              <div className="flex items-center gap-2">
                <span className="rounded border border-primary/40 bg-primary/10 px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide text-primary">
                  router
                </span>
                <code className="font-mono text-sm">{r.routerProfileId}</code>
              </div>
            </div>
            <span className="text-xs text-muted-foreground">{r.total.toLocaleString()} peticiones</span>
          </div>
          <table className="w-full text-sm">
            <thead className="text-xs uppercase text-muted-foreground">
              <tr>
                <th className="py-1 text-left">Destino</th>
                <th className="py-1 text-right">Peticiones</th>
                <th className="py-1 text-right">OK</th>
                <th className="py-1 text-right">Duración media</th>
                <th className="py-1 text-right">Cuota</th>
              </tr>
            </thead>
            <tbody>
              {r.targets.map((t) => {
                const sharePct = Math.round(t.share * 1000) / 10
                const errPct = t.count > 0 ? Math.round(((t.count - t.okCount) / t.count) * 1000) / 10 : 0
                return (
                  <tr key={t.modelId} className="border-t border-border/50">
                    <td className="py-2">
                      <div className="flex items-center gap-1.5">
                        <ArrowRight size={12} className="text-muted-foreground" />
                        <code className="font-mono text-xs">{t.modelId}</code>
                      </div>
                    </td>
                    <td className="py-2 text-right tabular-nums">{t.count.toLocaleString()}</td>
                    <td className="py-2 text-right tabular-nums">
                      {t.okCount.toLocaleString()}
                      {errPct > 0 && (
                        <span className="ml-1 text-xs text-destructive">({errPct}% err)</span>
                      )}
                    </td>
                    <td className="py-2 text-right tabular-nums text-xs text-muted-foreground">
                      {t.avgDurationMs > 0 ? `${Math.round(t.avgDurationMs)} ms` : "—"}
                    </td>
                    <td className="py-2 text-right tabular-nums">
                      <div className="inline-flex items-center gap-2">
                        <div className="h-1.5 w-16 rounded-full bg-muted">
                          <div
                            className="h-full rounded-full bg-primary"
                            style={{ width: `${Math.min(100, sharePct)}%` }}
                          />
                        </div>
                        <span className="text-xs">{sharePct}%</span>
                      </div>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      ))}
    </div>
  )
}
