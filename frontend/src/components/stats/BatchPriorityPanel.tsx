import { useEffect, useState } from "react"
import { Clock, ListChecks, ShieldAlert } from "lucide-react"
import { api } from "@/api/client"
import type { BatchPriorityStatus, BatchVsInteractiveStats } from "@/types"
import { BatchVsInteractiveChart } from "./BatchVsInteractiveChart"

interface Props {
  from: string
  to: string
}

/**
 * Batches API usage vs. direct interactive traffic, plus live visibility
 * into the batch-vs-interactive priority mechanism (2026-10-30): whether
 * it has preempted anything, and which workers are currently in a resume
 * cooldown after doing so.
 */
export function BatchPriorityPanel({ from, to }: Props) {
  const [stats, setStats] = useState<BatchVsInteractiveStats | null>(null)
  const [status, setStatus] = useState<BatchPriorityStatus | null>(null)
  const [error, setError] = useState<string | null>(null)

  const refresh = () => {
    api
      .stats
      .batchVsInteractive({ from, to })
      .then(setStats)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
    api.stats.batchPriorityStatus().then(setStatus).catch(() => setStatus(null))
  }

  useEffect(() => {
    refresh()
    const handle = window.setInterval(refresh, 15_000)
    return () => window.clearInterval(handle)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [from, to])

  if (error) {
    return <div className="text-sm text-destructive">Error: {error}</div>
  }
  if (!stats) {
    return <div className="text-sm text-muted-foreground">Cargando estadísticas de lotes...</div>
  }

  return (
    <div className="space-y-4">
      <BatchVsInteractiveChart data={stats} />

      <div className="grid gap-3 lg:grid-cols-2">
        <div className="rounded-lg border border-border bg-card p-4">
          <h3 className="mb-2 flex items-center gap-1.5 text-sm font-semibold text-muted-foreground">
            <ShieldAlert size={14} />
            Preempciones
          </h3>
          {!status ? (
            <p className="text-xs text-muted-foreground">Sin datos (coordinador no disponible).</p>
          ) : (
            <>
              <p className="text-sm">
                <span className="text-xl font-semibold tabular-nums">{status.totalPreemptions}</span>{" "}
                <span className="text-muted-foreground">desde el arranque del servidor</span>
              </p>
              {status.recentEvents.length === 0 ? (
                <p className="mt-2 text-xs text-muted-foreground">Sin eventos de preempción recientes.</p>
              ) : (
                <ul className="mt-2 space-y-1">
                  {status.recentEvents.slice(0, 8).map((ev, i) => (
                    <li key={i} className="flex items-center justify-between gap-2 text-xs">
                      <code className="truncate font-mono text-[10px]">{ev.workerKey}</code>
                      <span className="text-muted-foreground">
                        {ev.elapsedSeconds}s invertidos · hace {Math.round(ev.secondsAgo)}s
                      </span>
                    </li>
                  ))}
                </ul>
              )}
            </>
          )}
        </div>

        <div className="rounded-lg border border-border bg-card p-4">
          <h3 className="mb-2 flex items-center gap-1.5 text-sm font-semibold text-muted-foreground">
            <Clock size={14} />
            Cooldowns activos
          </h3>
          {!status || status.activeCooldowns.length === 0 ? (
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <ListChecks size={14} className="opacity-60" />
              Ningún worker en cooldown — los lotes pueden retomar a toda potencia.
            </div>
          ) : (
            <ul className="space-y-1">
              {status.activeCooldowns.map((cd) => (
                <li key={cd.workerKey} className="flex items-center justify-between gap-2 text-xs">
                  <code className="truncate font-mono text-[10px]">{cd.workerKey}</code>
                  <span className="text-muted-foreground">
                    {cd.kind} · {cd.remainingSeconds}s restantes
                  </span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  )
}
