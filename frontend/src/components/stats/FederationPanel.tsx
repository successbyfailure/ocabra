import type { FederationNodeTraffic, FederationStats } from "@/types"
import { describeRoute } from "@/lib/federationRoute"

interface FederationPanelProps {
  data: FederationStats
}

const nf = new Intl.NumberFormat("es-ES")

function ms(v: number): string {
  return v >= 10_000 ? `${(v / 1000).toFixed(1)} s` : `${nf.format(Math.round(v))} ms`
}

function usd(v: number | null | undefined, digits = 2): string {
  return v == null ? "—" : `$${v.toFixed(digits)}`
}

function NodeTable({
  title,
  hint,
  rows,
  withCost,
}: {
  title: string
  hint: string
  rows: FederationNodeTraffic[]
  withCost?: boolean
}) {
  return (
    <div className="rounded-md border border-border bg-background/40 p-3">
      <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">{title}</p>
      <p className="mb-2 text-[11px] text-muted-foreground">{hint}</p>
      {rows.length === 0 ? (
        <p className="py-2 text-xs text-muted-foreground">Nada en este periodo.</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead className="text-left text-muted-foreground">
              <tr>
                <th className="py-1 pr-3 font-normal">Nodo</th>
                <th className="py-1 pr-3 text-right font-normal">Peticiones</th>
                <th className="py-1 pr-3 text-right font-normal">Errores</th>
                <th className="py-1 pr-3 text-right font-normal">Tokens (entr. / sal.)</th>
                <th className="py-1 pr-3 text-right font-normal">p50 / p95</th>
                {withCost && <th className="py-1 pr-3 text-right font-normal">Coste</th>}
                {withCost && <th className="py-1 pr-3 text-right font-normal">$/petición · $/1k tok</th>}
                <th className="py-1 font-normal">Modelos</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.node} className="border-t border-border/50 align-top">
                  <td className="py-1.5 pr-3 font-medium">{r.node}</td>
                  <td className="py-1.5 pr-3 text-right tabular-nums">{nf.format(r.count)}</td>
                  <td
                    className={`py-1.5 pr-3 text-right tabular-nums ${r.errors ? "text-red-600 dark:text-red-300" : ""}`}
                  >
                    {nf.format(r.errors)}
                  </td>
                  <td className="py-1.5 pr-3 text-right tabular-nums">
                    {nf.format(r.inputTokens)} / {nf.format(r.outputTokens)}
                  </td>
                  <td className="py-1.5 pr-3 text-right tabular-nums">
                    {ms(r.p50DurationMs)} / {ms(r.p95DurationMs)}
                  </td>
                  {withCost && <td className="py-1.5 pr-3 text-right tabular-nums">{usd(r.costUsd)}</td>}
                  {withCost && (
                    <td className="py-1.5 pr-3 text-right tabular-nums">
                      {usd(r.usdPerRequest, 4)} · {usd(r.usdPer1kTokens, 4)}
                    </td>
                  )}
                  <td className="py-1.5 text-muted-foreground">
                    {r.topModels.map((m) => `${m.model} (${m.count})`).join(", ")}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

export function FederationPanel({ data }: FederationPanelProps) {
  const inbound = data.inboundCount ?? 0
  const total = data.localCount + data.remoteCount
  const localPct = total > 0 ? (data.localCount / total) * 100 : 0
  const remotePct = total > 0 ? (data.remoteCount / total) * 100 : 0
  const outbound =
    data.outbound ??
    data.byNode.map((n) => ({
      node: n.nodeName || n.nodeId,
      count: n.count,
      errors: 0,
      inputTokens: 0,
      outputTokens: 0,
      avgDurationMs: n.avgDurationMs,
      p50DurationMs: n.avgDurationMs,
      p95DurationMs: n.avgDurationMs,
      topModels: [],
    }))

  return (
    <div className="rounded-lg border border-border bg-card p-3">
      <h3 className="mb-3 text-sm font-semibold text-muted-foreground">Federación</h3>
      {data.remoteCount === 0 && inbound === 0 && !data.fallbackCount ? (
        <p className="py-4 text-center text-sm text-muted-foreground">Sin peticiones federadas en este periodo</p>
      ) : (
        <div className="space-y-4">
          <div className="space-y-2">
            <div className="flex flex-wrap justify-between gap-2 text-sm">
              <span>
                Aquí: <span className="font-semibold">{nf.format(data.localCount)}</span> ({localPct.toFixed(1)}%)
              </span>
              <span>
                Enviadas a otros nodos: <span className="font-semibold">{nf.format(data.remoteCount)}</span> (
                {remotePct.toFixed(1)}%)
              </span>
              <span>
                Recibidas de otros nodos: <span className="font-semibold">{nf.format(inbound)}</span>
              </span>
            </div>
            <div className="flex h-3 w-full overflow-hidden rounded-full bg-muted">
              <div className="bg-emerald-500/70" style={{ width: `${localPct}%` }} title={`Aquí: ${data.localCount}`} />
              <div className="bg-blue-500/70" style={{ width: `${remotePct}%` }} title={`Otros nodos: ${data.remoteCount}`} />
            </div>
          </div>

          <NodeTable
            title="Enviadas"
            hint="Peticiones nuestras que sirvió otro nodo. En los nodos en la nube, lo que costaron en este periodo."
            rows={outbound}
            withCost
          />
          <NodeTable
            title="Recibidas"
            hint="Peticiones de otros nodos que servimos aquí."
            rows={data.inbound ?? []}
          />

          {(data.routes?.length || data.fallbackCount) ? (
            <div className="grid gap-3 md:grid-cols-2">
              <div className="rounded-md border border-border bg-background/40 p-3">
                <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">Por qué fueron donde fueron</p>
                <ul className="space-y-1 text-xs">
                  {(data.routes ?? []).map((r) => (
                    <li key={r.route} className="flex justify-between gap-3">
                      <span title={r.route}>{describeRoute(r.route)}</span>
                      <span className="tabular-nums text-muted-foreground">{nf.format(r.count)}</span>
                    </li>
                  ))}
                </ul>
              </div>
              <div className="rounded-md border border-border bg-background/40 p-3">
                <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">Respaldo local</p>
                <p className="mb-2 text-[11px] text-muted-foreground">
                  Se eligió otro nodo pero falló, y se sirvieron aquí.
                </p>
                {data.fallbacksByModel?.length ? (
                  <ul className="space-y-1 text-xs">
                    {data.fallbacksByModel.map((m) => (
                      <li key={m.model} className="flex justify-between gap-3">
                        <span>{m.model}</span>
                        <span className="tabular-nums text-amber-600 dark:text-amber-300">{nf.format(m.count)}</span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-xs text-muted-foreground">Ninguno.</p>
                )}
              </div>
            </div>
          ) : null}
        </div>
      )}
    </div>
  )
}
