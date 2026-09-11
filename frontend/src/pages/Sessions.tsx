import { useEffect, useState } from "react"
import { AlertTriangle, RefreshCw, Skull, Users2 } from "lucide-react"
import { toast } from "sonner"
import { api, type RealtimeSessionInfo } from "@/api/client"

/**
 * Live view of Realtime sessions currently holding worker models.
 *
 * A session appears here from the moment its Realtime WebSocket is
 * accepted until the client disconnects (or the zombie sweeper drops it
 * for exceeding ``session_max_idle_s``). While a session is listed, the
 * ModelManager will refuse to evict the workers it holds — so this page
 * doubles as a "why can't my model load?" debugging surface.
 */
function formatRelative(iso: string): string {
  if (!iso) return "—"
  const then = new Date(iso).getTime()
  if (!Number.isFinite(then)) return iso
  const diffSec = Math.max(0, Math.round((Date.now() - then) / 1000))
  if (diffSec < 60) return `hace ${diffSec}s`
  if (diffSec < 3600) return `hace ${Math.floor(diffSec / 60)}m`
  const h = Math.floor(diffSec / 3600)
  const m = Math.floor((diffSec % 3600) / 60)
  return m > 0 ? `hace ${h}h${m}m` : `hace ${h}h`
}

export function Sessions() {
  const [sessions, setSessions] = useState<RealtimeSessionInfo[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [killingId, setKillingId] = useState<string | null>(null)

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      const list = await api.sessions.list()
      setSessions(list)
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e)
      setError(msg)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void load()
    const handle = window.setInterval(() => void load(), 10_000)
    return () => window.clearInterval(handle)
  }, [])

  const handleKill = async (sessionId: string) => {
    if (!window.confirm(`¿Terminar la sesión ${sessionId}? Los workers quedarán liberados para eviction.`)) return
    setKillingId(sessionId)
    try {
      await api.sessions.kill(sessionId)
      toast.success("Sesión terminada")
      await load()
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e)
      toast.error(`No se pudo terminar: ${msg}`)
    } finally {
      setKillingId(null)
    }
  }

  return (
    <div className="space-y-4 p-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold">Sesiones Realtime</h1>
          <p className="text-sm text-muted-foreground">
            Sesiones activas que están reservando workers. El ModelManager no
            desalojará estos workers mientras la sesión esté viva.
          </p>
        </div>
        <button
          type="button"
          onClick={() => void load()}
          disabled={loading}
          className="inline-flex items-center gap-1 rounded-md border border-border px-3 py-1.5 text-sm hover:bg-muted disabled:opacity-50"
        >
          <RefreshCw size={14} className={loading ? "animate-spin" : ""} />
          Actualizar
        </button>
      </div>

      {error && (
        <div className="flex items-start gap-2 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
          <AlertTriangle size={16} className="mt-0.5" />
          {error}
        </div>
      )}

      {sessions.length === 0 && !loading && !error && (
        <div className="rounded-md border border-border bg-muted/20 px-4 py-8 text-center text-sm text-muted-foreground">
          <Users2 size={20} className="mx-auto mb-2 opacity-60" />
          No hay sesiones Realtime activas ahora mismo.
        </div>
      )}

      {sessions.length > 0 && (
        <div className="overflow-x-auto rounded-md border border-border">
          <table className="min-w-full text-sm">
            <thead className="bg-muted/40 text-xs uppercase text-muted-foreground">
              <tr>
                <th className="px-3 py-2 text-left">ID</th>
                <th className="px-3 py-2 text-left">Usuario</th>
                <th className="px-3 py-2 text-left">API Key</th>
                <th className="px-3 py-2 text-left">Workers</th>
                <th className="px-3 py-2 text-left">Inicio</th>
                <th className="px-3 py-2 text-left">Última actividad</th>
                <th className="px-3 py-2 text-left">Estado</th>
                <th className="px-3 py-2" />
              </tr>
            </thead>
            <tbody>
              {sessions.map((s) => (
                <tr key={s.sessionId} className="border-t border-border">
                  <td className="px-3 py-2 font-mono text-xs">{s.sessionId.slice(0, 12)}…</td>
                  <td className="px-3 py-2">{s.userId ?? "—"}</td>
                  <td className="px-3 py-2">{s.apiKeyName ?? "—"}</td>
                  <td className="px-3 py-2">
                    <div className="flex flex-wrap gap-1">
                      {s.workersHeld.map((w) => (
                        <span
                          key={w}
                          className="rounded bg-primary/10 px-1.5 py-0.5 text-xs font-mono text-primary"
                        >
                          {w}
                        </span>
                      ))}
                    </div>
                  </td>
                  <td className="px-3 py-2 text-xs text-muted-foreground">{formatRelative(s.startedAt)}</td>
                  <td className="px-3 py-2 text-xs text-muted-foreground">{formatRelative(s.lastActivityAt)}</td>
                  <td className="px-3 py-2">
                    {s.isZombie ? (
                      <span className="rounded bg-destructive/10 px-2 py-0.5 text-xs text-destructive">Zombie</span>
                    ) : s.isPaused ? (
                      <span className="rounded bg-amber-500/10 px-2 py-0.5 text-xs text-amber-600 dark:text-amber-400">
                        Pausada
                      </span>
                    ) : (
                      <span className="rounded bg-emerald-500/10 px-2 py-0.5 text-xs text-emerald-600 dark:text-emerald-400">
                        Activa
                      </span>
                    )}
                  </td>
                  <td className="px-3 py-2 text-right">
                    <button
                      type="button"
                      onClick={() => void handleKill(s.sessionId)}
                      disabled={killingId === s.sessionId}
                      className="inline-flex items-center gap-1 rounded-md border border-destructive/40 px-2 py-1 text-xs text-destructive hover:bg-destructive/10 disabled:opacity-50"
                    >
                      <Skull size={12} />
                      Kill
                    </button>
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
