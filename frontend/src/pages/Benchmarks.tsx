import { useEffect, useMemo, useState } from "react"
import {
  Activity,
  AlertTriangle,
  Ban,
  BarChart3,
  Loader2,
  Pause,
  Play,
  Plus,
  RefreshCw,
  Terminal,
} from "lucide-react"
import { toast } from "sonner"
import { api } from "@/api/client"
import type {
  BenchmarkAppCatalog,
  BenchmarkRun,
  BenchmarkRunStatus,
  ModelProfile,
} from "@/types"

const ACTIVE_STATUSES: BenchmarkRunStatus[] = [
  "queued",
  "running",
  "pausing",
  "paused",
  "cancelling",
]

const ACTION_BUTTON = "inline-flex items-center gap-1 rounded-md border border-border px-2.5 py-1.5 text-xs hover:bg-muted disabled:opacity-50"

const STATUS_LABEL: Record<BenchmarkRunStatus, string> = {
  queued: "En cola",
  running: "Ejecutando",
  pausing: "Pausando",
  paused: "Pausado",
  cancelling: "Cancelando",
  completed: "Completado",
  failed: "Fallido",
  cancelled: "Cancelado",
}

function statusClass(status: BenchmarkRunStatus): string {
  if (status === "completed") return "border-emerald-500/40 bg-emerald-500/10 text-emerald-400"
  if (status === "failed") return "border-destructive/40 bg-destructive/10 text-destructive"
  if (status === "paused" || status === "pausing") return "border-amber-500/40 bg-amber-500/10 text-amber-400"
  if (status === "cancelled" || status === "cancelling") return "border-border bg-muted text-muted-foreground"
  return "border-sky-500/40 bg-sky-500/10 text-sky-400"
}

function formatDate(value: string | null): string {
  return value ? new Date(value).toLocaleString() : "—"
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null
    ? value as Record<string, unknown>
    : null
}

function metricRows(result: Record<string, unknown>): Array<Record<string, unknown>> {
  const metrics = result.metrics
  return Array.isArray(metrics)
    ? metrics.map(asRecord).filter((row): row is Record<string, unknown> => row !== null)
    : []
}

function pct(value: unknown): string {
  return typeof value === "number" ? `${Math.round(value * 100)}%` : "—"
}

function number(value: unknown, suffix = ""): string {
  return typeof value === "number" ? `${value.toFixed(1)}${suffix}` : "—"
}

export function Benchmarks() {
  const [catalog, setCatalog] = useState<BenchmarkAppCatalog[]>([])
  const [profiles, setProfiles] = useState<ModelProfile[]>([])
  const [runs, setRuns] = useState<BenchmarkRun[]>([])
  const [loading, setLoading] = useState(true)
  const [creating, setCreating] = useState(false)
  const [appName, setAppName] = useState("")
  const [suiteName, setSuiteName] = useState("")
  const [models, setModels] = useState<string[]>([])
  const [revision, setRevision] = useState("")
  const [profile, setProfile] = useState("standard")
  const [autoPause, setAutoPause] = useState(true)
  const [busyId, setBusyId] = useState<string | null>(null)
  const [logs, setLogs] = useState<Record<string, string[]>>({})

  const refresh = async () => {
    const next = await api.benchmarks.list()
    setRuns(next)
  }

  useEffect(() => {
    void Promise.all([
      api.benchmarks.catalog(),
      api.profiles.listAll(),
      api.benchmarks.list(),
    ])
      .then(([nextCatalog, nextProfiles, nextRuns]) => {
        setCatalog(nextCatalog)
        setProfiles(nextProfiles.filter((item) => item.enabled && item.profileKind === "execution"))
        setRuns(nextRuns)
        if (nextCatalog.length > 0) {
          setAppName(nextCatalog[0].app)
          setSuiteName(nextCatalog[0].suites[0]?.name ?? "")
          setRevision(nextCatalog[0].defaultRevision)
        }
      })
      .catch((error) => toast.error(error instanceof Error ? error.message : "No se pudo cargar benchmarks"))
      .finally(() => setLoading(false))
  }, [])

  const hasActive = runs.some((run) => ACTIVE_STATUSES.includes(run.status))
  useEffect(() => {
    const timer = window.setInterval(
      () => void refresh().catch(() => undefined),
      hasActive ? 2_000 : 10_000,
    )
    return () => window.clearInterval(timer)
  }, [hasActive])

  const selectedApp = catalog.find((item) => item.app === appName)
  const selectedSuite = selectedApp?.suites.find((item) => item.name === suiteName)
  const modelOptions = useMemo(
    () => [...profiles].sort((a, b) => a.profileId.localeCompare(b.profileId)),
    [profiles],
  )

  const changeApp = (value: string) => {
    const app = catalog.find((item) => item.app === value)
    setAppName(value)
    setSuiteName(app?.suites[0]?.name ?? "")
    setRevision(app?.defaultRevision ?? "")
  }

  const createRun = async () => {
    if (!appName || !suiteName || models.length === 0) {
      toast.error("Selecciona aplicación, suite y al menos un modelo")
      return
    }
    setCreating(true)
    try {
      const suiteArgs = appName === "corrala" ? ["--profile", profile] : []
      await api.benchmarks.create({
        app: appName,
        suite: suiteName,
        models,
        revision,
        suiteArgs,
        autoPause,
      })
      toast.success("Benchmark añadido a la cola")
      setModels([])
      await refresh()
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "No se pudo encolar")
    } finally {
      setCreating(false)
    }
  }

  const control = async (run: BenchmarkRun, action: "pause" | "resume" | "cancel") => {
    setBusyId(run.id)
    try {
      await api.benchmarks[action](run.id)
      await refresh()
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "No se pudo cambiar la ejecución")
    } finally {
      setBusyId(null)
    }
  }

  const loadLog = async (runId: string) => {
    try {
      const response = await api.benchmarks.log(runId)
      setLogs((current) => ({ ...current, [runId]: response.lines }))
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "No se pudo leer el log")
    }
  }

  return (
    <div className="space-y-5">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="flex items-center gap-2 text-2xl font-semibold">
            <BarChart3 size={22} className="text-primary" /> Benchmarks
          </h1>
          <p className="mt-1 max-w-3xl text-sm text-muted-foreground">
            Ejecuta datasets reales por modelo. Si entra tráfico interactivo, oCabra pausa el
            runner, descarta los intentos contaminados y los reanuda tras un periodo sin actividad.
          </p>
        </div>
        <button
          type="button"
          onClick={() => void refresh()}
          className="rounded-md border border-border p-2 text-muted-foreground hover:bg-muted"
          title="Actualizar"
        >
          <RefreshCw size={16} />
        </button>
      </div>

      <section className="rounded-lg border border-border bg-card p-4">
        <div className="mb-4 flex items-center gap-2">
          <Plus size={17} className="text-primary" />
          <h2 className="font-medium">Nueva ejecución</h2>
        </div>
        <div className="grid gap-4 lg:grid-cols-2">
          <label className="space-y-1 text-sm">
            <span className="text-muted-foreground">Aplicación</span>
            <select
              value={appName}
              onChange={(event) => changeApp(event.target.value)}
              className="w-full rounded-md border border-border bg-background px-3 py-2"
            >
              {catalog.map((item) => <option key={item.app} value={item.app}>{item.app}</option>)}
            </select>
            <p className="text-xs text-muted-foreground">{selectedApp?.description}</p>
          </label>
          <label className="space-y-1 text-sm">
            <span className="text-muted-foreground">Suite</span>
            <select
              value={suiteName}
              onChange={(event) => setSuiteName(event.target.value)}
              className="w-full rounded-md border border-border bg-background px-3 py-2"
            >
              {selectedApp?.suites.map((item) => <option key={item.name} value={item.name}>{item.name}</option>)}
            </select>
            <p className="text-xs text-muted-foreground">
              {selectedSuite?.description} {selectedSuite?.resumable ? "Reanudable." : ""}
            </p>
          </label>
          <label className="space-y-1 text-sm">
            <span className="text-muted-foreground">Modelos (Ctrl/Cmd para varios)</span>
            <select
              multiple
              size={Math.min(9, Math.max(5, modelOptions.length))}
              value={models}
              onChange={(event) => setModels(
                Array.from(event.currentTarget.selectedOptions, (option) => option.value),
              )}
              className="w-full rounded-md border border-border bg-background px-2 py-1 font-mono text-xs"
            >
              {modelOptions.map((item) => (
                <option key={item.profileId} value={item.profileId}>
                  {item.profileId}{item.displayName ? ` — ${item.displayName}` : ""}
                </option>
              ))}
            </select>
          </label>
          <div className="space-y-3">
            <label className="block space-y-1 text-sm">
              <span className="text-muted-foreground">Revisión Git</span>
              <input
                value={revision}
                onChange={(event) => setRevision(event.target.value)}
                className="w-full rounded-md border border-border bg-background px-3 py-2 font-mono text-xs"
              />
            </label>
            {appName === "corrala" && (
              <label className="block space-y-1 text-sm">
                <span className="text-muted-foreground">Perfil de coste</span>
                <select
                  value={profile}
                  onChange={(event) => setProfile(event.target.value)}
                  className="w-full rounded-md border border-border bg-background px-3 py-2"
                >
                  <option value="quick">Quick (34 casos)</option>
                  <option value="standard">Standard (54 casos)</option>
                  <option value="full">Full (83 casos)</option>
                  <option value="final">Final (83 casos × 3)</option>
                </select>
              </label>
            )}
            <label className="flex items-start gap-2 text-sm">
              <input
                type="checkbox"
                checked={autoPause}
                onChange={(event) => setAutoPause(event.target.checked)}
                className="mt-1"
              />
              <span>
                Pausar ante tráfico real
                <span className="block text-xs text-muted-foreground">
                  Los intentos solapados se repiten y no entran en el resultado limpio.
                </span>
              </span>
            </label>
          </div>
        </div>
        <div className="mt-4 flex justify-end">
          <button
            type="button"
            onClick={() => void createRun()}
            disabled={creating || loading || models.length === 0}
            className="inline-flex items-center gap-2 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground disabled:opacity-50"
          >
            {creating ? <Loader2 size={15} className="animate-spin" /> : <Play size={15} />}
            Encolar {models.length > 0 ? `${models.length} modelo${models.length === 1 ? "" : "s"}` : ""}
          </button>
        </div>
      </section>

      <section className="space-y-3">
        <h2 className="font-medium">Ejecuciones</h2>
        {!loading && runs.length === 0 && (
          <div className="rounded-lg border border-dashed border-border p-10 text-center text-sm text-muted-foreground">
            No hay benchmarks todavía.
          </div>
        )}
        {runs.map((run) => (
          <RunCard
            key={run.id}
            run={run}
            busy={busyId === run.id}
            log={logs[run.id]}
            onPause={() => void control(run, "pause")}
            onResume={() => void control(run, "resume")}
            onCancel={() => void control(run, "cancel")}
            onLog={() => void loadLog(run.id)}
          />
        ))}
      </section>
    </div>
  )
}

function RunCard({
  run,
  busy,
  log,
  onPause,
  onResume,
  onCancel,
  onLog,
}: {
  run: BenchmarkRun
  busy: boolean
  log?: string[]
  onPause: () => void
  onResume: () => void
  onCancel: () => void
  onLog: () => void
}) {
  const progress = run.models.length > 0
    ? Math.round((run.currentModelIndex / run.models.length) * 100)
    : 0

  return (
    <article className="rounded-lg border border-border bg-card p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="font-medium">{run.app} · {run.suite}</h3>
            <span className={`rounded border px-2 py-0.5 text-xs ${statusClass(run.status)}`}>
              {STATUS_LABEL[run.status]}
            </span>
            {run.contaminatedAttempts > 0 && (
              <span className="inline-flex items-center gap-1 text-xs text-amber-400">
                <AlertTriangle size={12} /> {run.contaminatedAttempts} contaminado(s)
              </span>
            )}
          </div>
          <p className="mt-1 text-xs text-muted-foreground">
            {formatDate(run.createdAt)} · revisión <code>{run.revision}</code>
          </p>
        </div>
        <div className="flex items-center gap-2">
          {run.status === "paused" ? (
            <button type="button" onClick={onResume} disabled={busy} className={ACTION_BUTTON}>
              <Play size={14} /> Reanudar
            </button>
          ) : ACTIVE_STATUSES.includes(run.status) && run.status !== "cancelling" ? (
            <button type="button" onClick={onPause} disabled={busy} className={ACTION_BUTTON}>
              <Pause size={14} /> Pausar
            </button>
          ) : null}
          {ACTIVE_STATUSES.includes(run.status) && run.status !== "cancelling" && (
            <button type="button" onClick={onCancel} disabled={busy} className={`${ACTION_BUTTON} text-destructive`}>
              <Ban size={14} /> Cancelar
            </button>
          )}
          {run.attempts.length > 0 && (
            <button type="button" onClick={onLog} className={ACTION_BUTTON}>
              <Terminal size={14} /> Log
            </button>
          )}
        </div>
      </div>

      <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-muted">
        <div className="h-full bg-primary transition-all" style={{ width: `${progress}%` }} />
      </div>
      <div className="mt-2 flex flex-wrap justify-between gap-2 text-xs text-muted-foreground">
        <span>{run.currentModelIndex}/{run.models.length} modelos aceptados</span>
        <span>{run.currentModel ? `Ahora: ${run.currentModel}` : run.models.join(", ")}</span>
      </div>

      {run.pauseReason === "external_traffic" && (
        <div className="mt-3 flex items-start gap-2 rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-300">
          <Activity size={14} className="mt-0.5" />
          Pausado por tráfico externo. Se reanudará automáticamente cuando el servidor quede libre.
        </div>
      )}
      {run.error && (
        <div className="mt-3 rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
          {run.error}
        </div>
      )}

      {Object.keys(run.result).length > 0 && <Results results={run.result} />}

      {run.attempts.length > 0 && (
        <details className="mt-3 text-xs">
          <summary className="cursor-pointer text-muted-foreground">
            Intentos ({run.attempts.length})
          </summary>
          <div className="mt-2 overflow-x-auto">
            <table className="w-full text-left">
              <thead className="text-muted-foreground">
                <tr><th className="py-1">#</th><th>Modelo</th><th>Estado</th><th>Inicio</th><th>Tráfico externo</th></tr>
              </thead>
              <tbody>
                {run.attempts.map((attempt) => (
                  <tr key={attempt.id} className="border-t border-border/60">
                    <td className="py-1">{attempt.attemptNumber}</td>
                    <td className="font-mono">{attempt.model}</td>
                    <td>{attempt.status}</td>
                    <td>{formatDate(attempt.startedAt)}</td>
                    <td>{attempt.externalRequests.length}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      )}

      {log && (
        <pre className="mt-3 max-h-72 overflow-auto rounded-md bg-black/60 p-3 text-[11px] text-zinc-300">
          {log.join("\n") || "Sin salida todavía"}
        </pre>
      )}
    </article>
  )
}

function Results({ results }: { results: Record<string, Record<string, unknown>> }) {
  const rows: Array<{ model: string; metric: Record<string, unknown> | null }> = []
  Object.entries(results).forEach(([model, result]) => {
    const metrics = metricRows(result)
    if (metrics.length > 0) {
      rows.push(...metrics.map((metric) => ({ model, metric })))
    } else {
      rows.push({ model, metric: null })
    }
  })
  return (
    <details className="mt-4" open>
      <summary className="cursor-pointer text-sm font-medium">Resultados aceptados</summary>
      <div className="mt-2 overflow-x-auto rounded-md border border-border">
        <table className="w-full text-left text-xs">
          <thead className="bg-muted/40 text-muted-foreground">
            <tr>
              <th className="px-2 py-2">Modelo</th><th>Suite</th><th>Calidad</th>
              <th>Primaria</th><th>Latencia p50</th><th>Decode</th><th>Errores</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(({ model, metric }, index) => {
              const quality = asRecord(metric?.quality)
              const speed = asRecord(metric?.speed)
              return (
                <tr key={`${model}-${index}`} className="border-t border-border/60">
                  <td className="px-2 py-2 font-mono">{String(metric?.model ?? model)}</td>
                  <td>{String(metric?.suite ?? "resultado")}</td>
                  <td>{pct(quality?.pass_rate)}</td>
                  <td>{pct(quality?.primary_rate)}</td>
                  <td>{number(speed?.latency_s_p50, "s")}</td>
                  <td>{number(speed?.decode_tok_s_p50, " tok/s")}</td>
                  <td>{String(quality?.errors ?? "—")}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      <details className="mt-2">
        <summary className="cursor-pointer text-xs text-muted-foreground">JSON bruto</summary>
        <pre className="mt-2 max-h-96 overflow-auto rounded-md bg-black/60 p-3 text-[11px] text-zinc-300">
          {JSON.stringify(results, null, 2)}
        </pre>
      </details>
    </details>
  )
}
