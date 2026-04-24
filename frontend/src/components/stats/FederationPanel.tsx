import type { FederationStats } from "@/types"

interface FederationPanelProps {
  data: FederationStats
}

export function FederationPanel({ data }: FederationPanelProps) {
  const total = data.localCount + data.remoteCount
  const localPct = total > 0 ? (data.localCount / total) * 100 : 0
  const remotePct = total > 0 ? (data.remoteCount / total) * 100 : 0

  return (
    <div className="rounded-lg border border-border bg-card p-3">
      <h3 className="mb-3 text-sm font-semibold text-muted-foreground">Federacion</h3>
      {total === 0 ? (
        <p className="text-sm text-muted-foreground py-4 text-center">Sin peticiones federadas</p>
      ) : (
        <div className="space-y-4">
          {/* Local vs Remote bar */}
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span>Local: <span className="font-semibold">{data.localCount}</span> ({localPct.toFixed(1)}%)</span>
              <span>Remoto: <span className="font-semibold">{data.remoteCount}</span> ({remotePct.toFixed(1)}%)</span>
            </div>
            <div className="flex h-3 w-full overflow-hidden rounded-full bg-muted">
              <div
                className="bg-emerald-500/70 transition-all"
                style={{ width: `${localPct}%` }}
                title={`Local: ${data.localCount}`}
              />
              <div
                className="bg-blue-500/70 transition-all"
                style={{ width: `${remotePct}%` }}
                title={`Remoto: ${data.remoteCount}`}
              />
            </div>
          </div>

          {/* Per-node breakdown */}
          {data.byNode.length > 0 && (
            <div className="rounded-md border border-border bg-background/40 p-3">
              <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">Por nodo remoto</p>
              <div className="space-y-2">
                {data.byNode.map((node) => (
                  <div key={node.nodeId} className="flex justify-between text-xs">
                    <span className="font-mono">{node.nodeId}</span>
                    <span className="text-muted-foreground">{node.count} req · {node.avgDurationMs} ms</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
