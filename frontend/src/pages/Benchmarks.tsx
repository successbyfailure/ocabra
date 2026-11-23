import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react"
import {
  Activity, AlertTriangle, Ban, BarChart3, CheckCircle2, ChevronDown, CircleDot,
  Clipboard, Clock3, Copy, Cpu, CalendarDays, GitCommit, ListChecks, Loader2, Pause, Play, Plus,
  RefreshCw, RotateCcw, Save, Search, Server, Star, Terminal, Trash2, X, XCircle,
} from "lucide-react"
import { toast } from "sonner"
import { api } from "@/api/client"
import type {
  BenchmarkAppCatalog, BenchmarkAttempt, BenchmarkRun, BenchmarkRunStatus, BenchmarkSchedule,
  BenchmarkScheduleInput,
  ModelProfile,
} from "@/types"
import { BenchmarkResultsDashboard } from "./benchmarks/BenchmarkResultsDashboard"
import { BenchmarkTimeEstimate } from "./benchmarks/BenchmarkTimeEstimate"
import { estimateBenchmarkTime } from "./benchmarks/benchmarkEstimate"

const ACTIVE_STATUSES: BenchmarkRunStatus[] = [
  "queued", "running", "pausing", "paused", "cancelling",
]

const ACTION_BUTTON = "inline-flex items-center gap-1.5 rounded-md border border-border px-2.5 py-1.5 text-xs hover:bg-muted disabled:cursor-not-allowed disabled:opacity-50"

const STATUS_LABEL: Record<BenchmarkRunStatus, string> = {
  queued: "En cola", running: "Ejecutando", pausing: "Pausando", paused: "Pausado",
  cancelling: "Cancelando", completed: "Completado", failed: "Fallido", cancelled: "Cancelado",
}

const ATTEMPT_LABEL: Record<BenchmarkAttempt["status"], string> = {
  running: "Ejecutando", interrupted: "Interrumpido", contaminated: "Contaminado",
  completed: "Completado", failed: "Fallido",
}

const PHASE_LABEL: Record<string, string> = {
  queued: "Esperando turno",
  checkout: "Preparando checkout",
  preparing: "Preparando dataset",
  running: "Ejecutando runner",
  finalizing: "Consolidando resultados",
  pausing: "Deteniendo runner",
  paused: "Pausa manual",
  paused_external_traffic: "Esperando tráfico cero",
  cancelling: "Cancelando",
  cancelled: "Cancelado",
  completed: "Finalizado",
  failed: "Fallido",
}

const WEEKDAYS = ["L", "M", "X", "J", "V", "S", "D"]

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

function formatDuration(start: string | null, end: string | null): string {
  if (!start) return "—"
  const seconds = Math.max(0, Math.round((new Date(end ?? Date.now()).getTime() - new Date(start).getTime()) / 1_000))
  if (seconds < 60) return `${seconds} s`
  const minutes = Math.floor(seconds / 60)
  if (minutes < 60) return `${minutes} min ${seconds % 60} s`
  return `${Math.floor(minutes / 60)} h ${minutes % 60} min`
}

function suiteProfileArgs(app: string, profile: string): string[] {
  return app === "corrala" ? ["--profile", profile] : []
}

function benchmarkCategories(app: string, suite: string): Set<ModelProfile["category"]> {
  if (app === "escriba" && suite === "embeddings") return new Set(["embedding", "rerank"])
  if (app === "escriba" && (suite === "stt" || suite === "stt_diarization")) return new Set(["stt"])
  return new Set(["llm"])
}

function profileBackend(profile: ModelProfile): string {
  return profile.baseModelId?.split("/", 1)[0] ?? "sin backend"
}

function logicalModelId(profile: ModelProfile): string {
  const value = profile.baseModelId ?? profile.profileId
  return value.includes("/") ? value.slice(value.indexOf("/") + 1) : value
}

export function backendVersionLabel(backend: {
  type: string
  installedVersion?: string | null
  sourceCommit?: string | null
  ociDigest?: string | null
  gitRef?: string | null
}): string {
  const version = backend.installedVersion
    ?? backend.sourceCommit?.slice(0, 12)
    ?? backend.ociDigest?.slice(0, 19)
    ?? backend.gitRef
    ?? "versión no informada"
  return `${backend.type} ${version}`
}

export function benchmarkModelGroups(
  profiles: ModelProfile[], app: string, suite: string, search = "",
) {
  const needle = search.trim().toLocaleLowerCase()
  const categories = benchmarkCategories(app, suite)
  const groups = new Map<string, ModelProfile[]>()
  profiles.filter((item) => categories.has(item.category)).forEach((item) => {
    const key = logicalModelId(item)
    groups.set(key, [...(groups.get(key) ?? []), item])
  })
  return [...groups.entries()]
    .map(([key, options]) => ({
      key,
      name: options.find((item) => item.isDefault)?.displayName ?? options[0].displayName ?? key,
      options: options.sort((a, b) => profileBackend(a).localeCompare(profileBackend(b)) || a.profileId.localeCompare(b.profileId)),
    }))
    .filter((group) => !needle || `${group.key} ${group.name} ${group.options.map((item) => `${item.profileId} ${profileBackend(item)}`).join(" ")}`.toLocaleLowerCase().includes(needle))
    .sort((a, b) => a.name.localeCompare(b.name))
}

export function Benchmarks() {
  const [view, setView] = useState<"results" | "runs" | "launch">("results")
  const [catalog, setCatalog] = useState<BenchmarkAppCatalog[]>([])
  const [profiles, setProfiles] = useState<ModelProfile[]>([])
  const [runs, setRuns] = useState<BenchmarkRun[]>([])
  const [schedules, setSchedules] = useState<BenchmarkSchedule[]>([])
  const [loading, setLoading] = useState(true)
  const [creating, setCreating] = useState(false)
  const [appName, setAppName] = useState("")
  const [suiteName, setSuiteName] = useState("")
  const [models, setModels] = useState<string[]>([])
  const [modelSearch, setModelSearch] = useState("")
  const [revision, setRevision] = useState("")
  const [profile, setProfile] = useState("standard")
  const [autoPause, setAutoPause] = useState(true)
  const [scheduleName, setScheduleName] = useState("")
  const [scheduleEnabled, setScheduleEnabled] = useState(false)
  const [scheduleDays, setScheduleDays] = useState<number[]>([0])
  const [scheduleTime, setScheduleTime] = useState("02:00")
  const [scheduleTimezone, setScheduleTimezone] = useState(() => Intl.DateTimeFormat().resolvedOptions().timeZone || "Europe/Madrid")
  const [editingScheduleId, setEditingScheduleId] = useState<string | null>(null)
  const [busyId, setBusyId] = useState<string | null>(null)
  const [logs, setLogs] = useState<Record<string, string[]>>({})
  const [openLogs, setOpenLogs] = useState<Set<string>>(new Set())
  const [runSearch, setRunSearch] = useState("")
  const [statusFilter, setStatusFilter] = useState<"all" | BenchmarkRunStatus>("all")
  const [appFilter, setAppFilter] = useState("all")

  const refresh = useCallback(async () => {
    const [nextRuns, nextSchedules] = await Promise.all([
      api.benchmarks.list(), api.benchmarks.schedules.list(),
    ])
    setRuns(nextRuns)
    setSchedules(nextSchedules)
  }, [])

  useEffect(() => {
    void Promise.all([api.benchmarks.catalog(), api.profiles.listAll(), api.benchmarks.list(), api.benchmarks.schedules.list()])
      .then(([nextCatalog, nextProfiles, nextRuns, nextSchedules]) => {
        setCatalog(nextCatalog)
        setProfiles(nextProfiles.filter((item) => item.enabled && item.profileKind === "execution"))
        setRuns(nextRuns)
        setSchedules(nextSchedules)
        if (nextCatalog.length > 0) {
          setAppName(nextCatalog[0].app)
          setSuiteName(nextCatalog[0].suites[0]?.name ?? "")
          setRevision(nextCatalog[0].defaultRevision)
        }
      })
      .catch((error) => toast.error(error instanceof Error ? error.message : "No se pudieron cargar los benchmarks"))
      .finally(() => setLoading(false))
  }, [])

  const hasActive = runs.some((run) => ACTIVE_STATUSES.includes(run.status))
  useEffect(() => {
    const timer = window.setInterval(() => void refresh().catch(() => undefined), hasActive ? 2_000 : 10_000)
    return () => window.clearInterval(timer)
  }, [hasActive, refresh])

  const selectedApp = catalog.find((item) => item.app === appName)
  const selectedSuite = selectedApp?.suites.find((item) => item.name === suiteName)
  const timeEstimate = useMemo(() => estimateBenchmarkTime(
    runs, appName, suiteName, suiteProfileArgs(appName, profile), models,
  ), [appName, models, profile, runs, suiteName])
  const modelGroups = useMemo(() => {
    return benchmarkModelGroups(profiles, appName, suiteName, modelSearch)
  }, [appName, modelSearch, profiles, suiteName])
  const visibleProfileIds = useMemo(() => modelGroups.flatMap((group) => group.options.map((item) => item.profileId)), [modelGroups])

  const visibleRuns = useMemo(() => {
    const needle = runSearch.trim().toLocaleLowerCase()
    return runs.filter((run) => (
      (statusFilter === "all" || run.status === statusFilter)
      && (appFilter === "all" || run.app === appFilter)
      && (!needle || `${run.app} ${run.suite} ${run.models.join(" ")} ${run.revision}`.toLocaleLowerCase().includes(needle))
    ))
  }, [appFilter, runSearch, runs, statusFilter])

  const changeApp = (value: string) => {
    const app = catalog.find((item) => item.app === value)
    setAppName(value)
    setSuiteName(app?.suites[0]?.name ?? "")
    setRevision(app?.defaultRevision ?? "")
    setProfile("standard")
    setModels([])
  }

  const toggleModel = (model: string) => {
    setModels((current) => current.includes(model)
      ? current.filter((item) => item !== model)
      : [...current, model])
  }

  const createRun = async () => {
    if (!appName || !suiteName || models.length === 0) {
      toast.error("Selecciona una batería, una suite y al menos un modelo")
      return
    }
    setCreating(true)
    try {
      await api.benchmarks.create({
        app: appName, suite: suiteName, models, revision,
        suiteArgs: suiteProfileArgs(appName, profile), autoPause,
      })
      toast.success("Benchmark añadido a la cola")
      setModels([])
      await refresh()
      setView("runs")
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "No se pudo encolar")
    } finally {
      setCreating(false)
    }
  }

  const schedulePayload = (overrides: Partial<BenchmarkScheduleInput> = {}): BenchmarkScheduleInput => ({
    name: scheduleName.trim(), app: appName, suite: suiteName, models, revision,
    suiteArgs: suiteProfileArgs(appName, profile), autoPause, enabled: scheduleEnabled,
    weekdays: scheduleDays, timeOfDay: scheduleTime, timezone: scheduleTimezone,
    ...overrides,
  })

  const saveSchedule = async () => {
    if (!scheduleName.trim() || !appName || !suiteName || models.length === 0) {
      toast.error("Pon nombre al plan y selecciona al menos un modelo")
      return
    }
    setCreating(true)
    try {
      if (editingScheduleId) {
        await api.benchmarks.schedules.update(editingScheduleId, schedulePayload())
        toast.success("Plan actualizado")
      } else {
        await api.benchmarks.schedules.create(schedulePayload())
        toast.success(scheduleEnabled ? "Plan programado" : "Plantilla guardada")
      }
      setEditingScheduleId(null)
      setScheduleName("")
      await refresh()
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "No se pudo guardar el plan")
    } finally {
      setCreating(false)
    }
  }

  const editSchedule = (schedule: BenchmarkSchedule) => {
    setEditingScheduleId(schedule.id)
    setScheduleName(schedule.name)
    setAppName(schedule.app)
    setSuiteName(schedule.suite)
    setModels(schedule.models)
    setRevision(schedule.revision)
    setAutoPause(schedule.autoPause)
    setScheduleEnabled(schedule.enabled)
    setScheduleDays(schedule.weekdays)
    setScheduleTime(schedule.timeOfDay)
    setScheduleTimezone(schedule.timezone)
    const profileIndex = schedule.suiteArgs.indexOf("--profile")
    setProfile(profileIndex >= 0 ? schedule.suiteArgs[profileIndex + 1] ?? "standard" : "standard")
    window.scrollTo({ top: 0, behavior: "smooth" })
  }

  const updateSchedule = async (schedule: BenchmarkSchedule, changes: Partial<BenchmarkScheduleInput>) => {
    setBusyId(schedule.id)
    try {
      await api.benchmarks.schedules.update(schedule.id, {
        name: schedule.name, app: schedule.app, suite: schedule.suite,
        revision: schedule.revision, models: schedule.models, suiteArgs: schedule.suiteArgs,
        autoPause: schedule.autoPause, enabled: schedule.enabled,
        weekdays: schedule.weekdays, timeOfDay: schedule.timeOfDay,
        timezone: schedule.timezone, ...changes,
      })
      await refresh()
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "No se pudo actualizar el plan")
    } finally { setBusyId(null) }
  }

  const runSchedule = async (schedule: BenchmarkSchedule) => {
    setBusyId(schedule.id)
    try {
      await api.benchmarks.schedules.run(schedule.id)
      toast.success("Plan añadido a la cola")
      await refresh()
      setView("runs")
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "No se pudo ejecutar el plan")
    } finally { setBusyId(null) }
  }

  const removeSchedule = async (schedule: BenchmarkSchedule) => {
    if (!window.confirm(`¿Eliminar el plan «${schedule.name}»? El histórico se conserva.`)) return
    setBusyId(schedule.id)
    try {
      await api.benchmarks.schedules.remove(schedule.id)
      await refresh()
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "No se pudo eliminar el plan")
    } finally { setBusyId(null) }
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

  const rerun = async (run: BenchmarkRun) => {
    setBusyId(run.id)
    try {
      await api.benchmarks.create({
        app: run.app, suite: run.suite, models: run.models, revision: run.revision,
        suiteArgs: run.suiteArgs, autoPause: run.autoPause,
      })
      toast.success("Configuración añadida de nuevo a la cola")
      await refresh()
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "No se pudo repetir la ejecución")
    } finally {
      setBusyId(null)
    }
  }

  const retry = async (run: BenchmarkRun) => {
    setBusyId(run.id)
    try {
      await api.benchmarks.retry(run.id)
      toast.success(`Reintentando desde ${run.models[run.currentModelIndex] ?? "el modelo fallido"}`)
      await refresh()
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "No se pudo reintentar")
    } finally {
      setBusyId(null)
    }
  }

  const setBaseline = async (run: BenchmarkRun) => {
    setBusyId(run.id)
    try {
      await api.benchmarks.baseline(run.id)
      toast.success("Referencia histórica actualizada")
      await refresh()
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "No se pudo fijar la referencia")
    } finally { setBusyId(null) }
  }

  const loadLog = async (runId: string, open = true) => {
    try {
      const response = await api.benchmarks.log(runId, 2_000)
      setLogs((current) => ({ ...current, [runId]: response.lines }))
      if (open) setOpenLogs((current) => new Set(current).add(runId))
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "No se pudo leer el log")
    }
  }

  const toggleLog = (runId: string) => {
    if (openLogs.has(runId)) {
      setOpenLogs((current) => {
        const next = new Set(current)
        next.delete(runId)
        return next
      })
    } else {
      void loadLog(runId)
    }
  }

  const summary = {
    active: runs.filter((run) => ACTIVE_STATUSES.includes(run.status)).length,
    completed: runs.filter((run) => run.status === "completed").length,
    failed: runs.filter((run) => run.status === "failed").length,
    contaminated: runs.reduce((total, run) => total + run.contaminatedAttempts, 0),
  }

  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="flex items-center gap-2 text-2xl font-semibold">
            <BarChart3 size={22} className="text-primary" /> Benchmarks
          </h1>
          <p className="mt-1 max-w-3xl text-sm text-muted-foreground">
            Ejecuta, compara y audita modelos con datasets reales. El tráfico interactivo pausa
            automáticamente el runner y los intentos solapados se descartan.
          </p>
        </div>
        <button type="button" onClick={() => void refresh()} className={ACTION_BUTTON}>
          <RefreshCw size={14} /> Actualizar
        </button>
      </header>

      <nav className="flex w-full gap-1 overflow-x-auto rounded-xl border border-border bg-card p-1" aria-label="Secciones de benchmarks">
        {([
          ["results", "Resultados", <BarChart3 size={15} />, runs.filter((run) => run.status === "completed" && Object.keys(run.result).length > 0).length],
          ["runs", "Ejecuciones", <ListChecks size={15} />, summary.active],
          ["launch", "Lanzar pruebas", <Plus size={15} />, schedules.filter((schedule) => schedule.enabled).length],
        ] as const).map(([key, label, icon, count]) => (
          <button key={key} type="button" onClick={() => setView(key)} className={`inline-flex min-w-max flex-1 items-center justify-center gap-2 rounded-lg px-4 py-2.5 text-sm font-medium transition-colors ${view === key ? "bg-primary text-primary-foreground shadow-sm" : "text-muted-foreground hover:bg-muted hover:text-foreground"}`}>
            {icon}{label}{count > 0 && <span className={`rounded-full px-1.5 py-0.5 text-[10px] ${view === key ? "bg-primary-foreground/15" : "bg-muted"}`}>{count}</span>}
          </button>
        ))}
      </nav>

      {view === "results" && <BenchmarkResultsDashboard runs={runs} />}

      {view === "runs" && <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <SummaryCard icon={<CircleDot size={16} />} label="En curso" value={summary.active} tone="sky" />
        <SummaryCard icon={<CheckCircle2 size={16} />} label="Completadas" value={summary.completed} tone="emerald" />
        <SummaryCard icon={<XCircle size={16} />} label="Fallidas" value={summary.failed} tone="red" />
        <SummaryCard icon={<AlertTriangle size={16} />} label="Intentos descartados" value={summary.contaminated} tone="amber" />
      </div>}

      {view === "launch" && <>
      <section className="rounded-xl border border-border bg-card">
        <div className="flex items-center gap-2 border-b border-border px-4 py-3">
          <Plus size={17} className="text-primary" />
          <div>
            <h2 className="font-medium">Nueva ejecución</h2>
            <p className="text-xs text-muted-foreground">Cada modelo se ejecuta de forma secuencial y aislada.</p>
          </div>
        </div>

        <div className="space-y-5 p-4">
          <div>
            <p className="mb-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">Batería</p>
            <div className="grid gap-2 md:grid-cols-2">
              {catalog.map((item) => (
                <button
                  key={item.app}
                  type="button"
                  onClick={() => changeApp(item.app)}
                  className={`rounded-lg border p-3 text-left transition-colors ${appName === item.app ? "border-primary bg-primary/10" : "border-border hover:bg-muted/50"}`}
                >
                  <span className="font-mono text-sm font-semibold">{item.app}</span>
                  <span className="mt-1 block text-xs text-muted-foreground">{item.description}</span>
                </button>
              ))}
            </div>
          </div>

          <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(320px,0.8fr)]">
            <div className="space-y-3">
              <div className="flex items-center justify-between gap-2">
                <label className="text-sm font-medium">Modelos</label>
                <div className="flex gap-2 text-xs">
                  <button type="button" onClick={() => setModels((current) => Array.from(new Set([...current, ...visibleProfileIds])))} className="text-primary hover:underline">Seleccionar visibles</button>
                  <button type="button" onClick={() => setModels([])} className="text-muted-foreground hover:underline">Limpiar</button>
                </div>
              </div>
              <div className="relative">
                <Search size={14} className="absolute left-3 top-2.5 text-muted-foreground" />
                <input value={modelSearch} onChange={(event) => setModelSearch(event.target.value)} placeholder="Buscar perfil de ejecución…" className="w-full rounded-md border border-border bg-background py-2 pl-9 pr-3 text-sm" />
              </div>
              <div className="max-h-72 space-y-1 overflow-y-auto rounded-lg border border-border bg-background p-1.5">
                {modelGroups.map((group) => (
                  <div key={group.key} className="rounded-md border border-transparent px-2 py-2 hover:border-border hover:bg-muted/40">
                    <div className="min-w-0"><span className="block truncate text-sm font-medium">{group.name}</span><span className="block truncate font-mono text-[11px] text-muted-foreground">{group.key}</span></div>
                    <div className="mt-2 flex flex-wrap gap-1.5">
                      {group.options.map((item) => {
                        const backend = profileBackend(item)
                        const sameBackend = group.options.filter((candidate) => profileBackend(candidate) === backend).length > 1
                        const selected = models.includes(item.profileId)
                        return <button key={item.profileId} type="button" onClick={() => toggleModel(item.profileId)} title={`${item.profileId} · ${item.baseModelId ?? ""}`} className={`rounded-full border px-2 py-1 text-[11px] transition-colors ${selected ? "border-primary bg-primary text-primary-foreground" : "border-border bg-card text-muted-foreground hover:border-primary/60 hover:text-foreground"}`}>{backend}{sameBackend ? ` · ${item.profileId}` : ""}</button>
                      })}
                    </div>
                  </div>
                ))}
                {modelGroups.length === 0 && <p className="p-4 text-center text-xs text-muted-foreground">No hay modelos compatibles con esta suite.</p>}
              </div>
              {models.length > 0 && (
                <div className="flex flex-wrap gap-1.5">
                  {models.map((model) => (
                    <button key={model} type="button" onClick={() => toggleModel(model)} className="inline-flex items-center gap-1 rounded-full bg-primary/10 px-2 py-1 font-mono text-[11px] text-primary">
                      {model} <X size={11} />
                    </button>
                  ))}
                </div>
              )}
            </div>

            <div className="space-y-4">
              <label className="block space-y-1 text-sm">
                <span className="font-medium">Suite</span>
                <select value={suiteName} onChange={(event) => { setSuiteName(event.target.value); setModels([]) }} className="w-full rounded-md border border-border bg-background px-3 py-2">
                  {selectedApp?.suites.map((item) => <option key={item.name} value={item.name}>{item.name}</option>)}
                </select>
                <span className="block text-xs text-muted-foreground">{selectedSuite?.description} {selectedSuite?.resumable ? "Admite reanudación." : ""}</span>
              </label>

              {appName === "corrala" && (
                <label className="block space-y-1 text-sm">
                  <span className="font-medium">Perfil de coste</span>
                  <select value={profile} onChange={(event) => setProfile(event.target.value)} className="w-full rounded-md border border-border bg-background px-3 py-2">
                    <option value="quick">Quick · 34 casos</option><option value="standard">Standard · 54 casos</option>
                    <option value="full">Full · 83 casos</option><option value="final">Final · 83 casos × 3</option>
                  </select>
                </label>
              )}

              <label className="flex items-start gap-2 rounded-lg border border-border p-3 text-sm">
                <input type="checkbox" checked={autoPause} onChange={(event) => setAutoPause(event.target.checked)} className="mt-1" />
                <span><span className="font-medium">Proteger el tráfico real</span><span className="block text-xs text-muted-foreground">Pausa, espera a que el servidor quede libre y repite el intento afectado.</span></span>
              </label>

              <BenchmarkTimeEstimate
                estimate={timeEstimate}
                needsPreparation={selectedSuite?.needsPreparation ?? false}
                autoPause={autoPause}
              />

              <details className="rounded-lg border border-border p-3 text-sm">
                <summary className="flex cursor-pointer list-none items-center justify-between font-medium">Opciones avanzadas <ChevronDown size={14} /></summary>
                <label className="mt-3 block space-y-1"><span className="text-xs text-muted-foreground">Revisión Git</span><input value={revision} onChange={(event) => setRevision(event.target.value)} className="w-full rounded-md border border-border bg-background px-3 py-2 font-mono text-xs" /></label>
              </details>

              <details className="rounded-lg border border-border p-3 text-sm" open={Boolean(editingScheduleId)}>
                <summary className="flex cursor-pointer list-none items-center justify-between font-medium"><span className="inline-flex items-center gap-1.5"><CalendarDays size={14} /> Guardar como plan</span><ChevronDown size={14} /></summary>
                <div className="mt-3 space-y-3">
                  <label className="block space-y-1"><span className="text-xs text-muted-foreground">Nombre</span><input value={scheduleName} onChange={(event) => setScheduleName(event.target.value)} placeholder="IFEval nocturno" className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm" /></label>
                  <label className="flex items-center gap-2"><input type="checkbox" checked={scheduleEnabled} onChange={(event) => setScheduleEnabled(event.target.checked)} /><span>Programación activa</span></label>
                  <div className="flex flex-wrap gap-1.5">{WEEKDAYS.map((day, index) => <button key={day} type="button" onClick={() => setScheduleDays((current) => current.includes(index) ? current.length > 1 ? current.filter((value) => value !== index) : current : [...current, index].sort())} className={`h-8 w-8 rounded-full border text-xs ${scheduleDays.includes(index) ? "border-primary bg-primary text-primary-foreground" : "border-border text-muted-foreground"}`}>{day}</button>)}</div>
                  <div className="grid grid-cols-2 gap-2"><label className="space-y-1"><span className="text-xs text-muted-foreground">Hora</span><input type="time" value={scheduleTime} onChange={(event) => setScheduleTime(event.target.value)} className="w-full rounded-md border border-border bg-background px-2 py-2" /></label><label className="space-y-1"><span className="text-xs text-muted-foreground">Zona</span><input value={scheduleTimezone} onChange={(event) => setScheduleTimezone(event.target.value)} className="w-full rounded-md border border-border bg-background px-2 py-2 text-xs" /></label></div>
                  <button type="button" onClick={() => void saveSchedule()} disabled={creating || models.length === 0 || !scheduleName.trim()} className={ACTION_BUTTON}><Save size={13} /> {editingScheduleId ? "Actualizar plan" : "Guardar plan"}</button>
                  {editingScheduleId && <button type="button" onClick={() => { setEditingScheduleId(null); setScheduleName("") }} className="ml-2 text-xs text-muted-foreground hover:underline">Cancelar edición</button>}
                </div>
              </details>

              <button type="button" onClick={() => void createRun()} disabled={creating || loading || models.length === 0} className="inline-flex w-full items-center justify-center gap-2 rounded-md bg-primary px-4 py-2.5 text-sm font-medium text-primary-foreground disabled:opacity-50">
                {creating ? <Loader2 size={15} className="animate-spin" /> : <Play size={15} />}
                {models.length > 0 ? `Encolar ${models.length} modelo${models.length === 1 ? "" : "s"}` : "Selecciona modelos"}
              </button>
            </div>
          </div>
        </div>
      </section>

      <section className="rounded-xl border border-border bg-card">
        <div className="flex items-center gap-2 border-b border-border px-4 py-3"><CalendarDays size={16} className="text-primary" /><div><h2 className="font-medium">Planes y horarios</h2><p className="text-xs text-muted-foreground">Las plantillas desactivadas se conservan sin lanzarse solas.</p></div></div>
        {schedules.length === 0 ? <p className="p-6 text-center text-sm text-muted-foreground">Todavía no hay planes guardados.</p> : <div className="grid gap-3 p-4 lg:grid-cols-2">{schedules.map((schedule) => (
          <article key={schedule.id} className="rounded-lg border border-border bg-background p-3">
            <div className="flex items-start justify-between gap-3"><div><h3 className="font-medium">{schedule.name}</h3><p className="font-mono text-xs text-muted-foreground">{schedule.app}/{schedule.suite} · {schedule.models.length} modelo{schedule.models.length === 1 ? "" : "s"}</p></div><button type="button" disabled={busyId === schedule.id} onClick={() => void updateSchedule(schedule, { enabled: !schedule.enabled })} className={`rounded-full border px-2 py-1 text-xs ${schedule.enabled ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-400" : "border-border text-muted-foreground"}`}>{schedule.enabled ? "Activo" : "Plantilla"}</button></div>
            <div className="mt-3 grid grid-cols-2 gap-2 text-xs text-muted-foreground"><div><span className="block">Próxima</span><strong className="font-normal text-foreground">{schedule.nextRunAt ? formatDate(schedule.nextRunAt) : "No programada"}</strong></div><div><span className="block">Última</span><strong className="font-normal text-foreground">{formatDate(schedule.lastRunAt)}</strong></div><div><span className="block">Horario</span><strong className="font-normal text-foreground">{schedule.weekdays.map((day) => WEEKDAYS[day]).join(" ")} · {schedule.timeOfDay}</strong></div><div><span className="block">Zona</span><strong className="font-normal text-foreground">{schedule.timezone}</strong></div></div>
            <div className="mt-3 flex flex-wrap gap-2"><button type="button" onClick={() => void runSchedule(schedule)} disabled={busyId === schedule.id} className={ACTION_BUTTON}><Play size={12} /> Ejecutar ahora</button><button type="button" onClick={() => editSchedule(schedule)} className={ACTION_BUTTON}>Editar</button><button type="button" onClick={() => void removeSchedule(schedule)} disabled={busyId === schedule.id} className={`${ACTION_BUTTON} text-destructive`}><Trash2 size={12} /> Eliminar</button></div>
          </article>
        ))}</div>}
      </section>
      </>}

      {view === "runs" && (
      <section className="space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div><h2 className="font-medium">Ejecuciones</h2><p className="text-xs text-muted-foreground">{visibleRuns.length} de {runs.length}</p></div>
          <div className="flex flex-1 flex-wrap justify-end gap-2">
            <div className="relative min-w-52"><Search size={13} className="absolute left-2.5 top-2.5 text-muted-foreground" /><input value={runSearch} onChange={(event) => setRunSearch(event.target.value)} placeholder="Modelo, suite o revisión…" className="w-full rounded-md border border-border bg-background py-2 pl-8 pr-2 text-xs" /></div>
            <select value={appFilter} onChange={(event) => setAppFilter(event.target.value)} className="rounded-md border border-border bg-background px-2 py-2 text-xs"><option value="all">Todas las baterías</option>{catalog.map((app) => <option key={app.app} value={app.app}>{app.app}</option>)}</select>
            <select value={statusFilter} onChange={(event) => setStatusFilter(event.target.value as typeof statusFilter)} className="rounded-md border border-border bg-background px-2 py-2 text-xs"><option value="all">Todos los estados</option>{Object.entries(STATUS_LABEL).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select>
          </div>
        </div>

        {!loading && visibleRuns.length === 0 && <div className="rounded-xl border border-dashed border-border p-10 text-center text-sm text-muted-foreground">{runs.length === 0 ? "No hay benchmarks todavía." : "Ninguna ejecución coincide con los filtros."}</div>}
        {visibleRuns.map((run) => (
          <RunCard key={run.id} run={run} busy={busyId === run.id} log={logs[run.id]} logOpen={openLogs.has(run.id)}
            onPause={() => void control(run, "pause")} onResume={() => void control(run, "resume")} onCancel={() => void control(run, "cancel")}
            onRetry={() => void retry(run)} onRerun={() => void rerun(run)} onBaseline={() => void setBaseline(run)} onToggleLog={() => toggleLog(run.id)} onRefreshLog={() => void loadLog(run.id, false)} />
        ))}
      </section>
      )}
    </div>
  )
}

function SummaryCard({ icon, label, value, tone }: { icon: ReactNode; label: string; value: number; tone: "sky" | "emerald" | "red" | "amber" }) {
  const colors = { sky: "text-sky-400 bg-sky-500/10", emerald: "text-emerald-400 bg-emerald-500/10", red: "text-red-400 bg-red-500/10", amber: "text-amber-400 bg-amber-500/10" }
  return <div className="flex items-center gap-3 rounded-xl border border-border bg-card p-3"><span className={`rounded-lg p-2 ${colors[tone]}`}>{icon}</span><div><p className="text-xl font-semibold tabular-nums">{value}</p><p className="text-xs text-muted-foreground">{label}</p></div></div>
}

function RunCard({ run, busy, log, logOpen, onPause, onResume, onCancel, onRetry, onRerun, onBaseline, onToggleLog, onRefreshLog }: {
  run: BenchmarkRun; busy: boolean; log?: string[]; logOpen: boolean; onPause: () => void; onResume: () => void;
  onCancel: () => void; onRetry: () => void; onRerun: () => void; onBaseline: () => void; onToggleLog: () => void; onRefreshLog: () => void
}) {
  const progress = run.models.length > 0 ? Math.round((run.currentModelIndex / run.models.length) * 100) : 0
  const terminal = ["completed", "failed", "cancelled"].includes(run.status)
  const copyLog = async () => {
    try {
      await navigator.clipboard.writeText((log ?? []).join("\n"))
      toast.success("Log copiado")
    } catch {
      toast.error("El navegador no permitió copiar el log")
    }
  }

  return (
    <article className="overflow-hidden rounded-xl border border-border bg-card">
      <div className="p-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <h3 className="font-medium">{run.app} <span className="text-muted-foreground">·</span> {run.suite}</h3>
              <span className={`rounded border px-2 py-0.5 text-xs ${statusClass(run.status)}`}>{STATUS_LABEL[run.status]}</span>
              {run.isBaseline && <span className="inline-flex items-center gap-1 text-xs text-amber-400"><Star size={12} className="fill-current" /> Referencia</span>}
              {run.contaminatedAttempts > 0 && <span className="inline-flex items-center gap-1 text-xs text-amber-400"><AlertTriangle size={12} /> {run.contaminatedAttempts} descartado{run.contaminatedAttempts === 1 ? "" : "s"}</span>}
            </div>
            <div className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
              <span>{formatDate(run.createdAt)}</span><span className="inline-flex items-center gap-1"><Clock3 size={11} /> {formatDuration(run.startedAt, run.finishedAt)}</span>
              <span>revisión <code>{run.revision.slice(0, 12)}</code></span>{run.scheduleId && <span className="inline-flex items-center gap-1"><CalendarDays size={11} /> plan</span>}<span className="font-mono" title={run.id}>{run.id.slice(0, 8)}</span>
            </div>
          </div>
          <div className="flex flex-wrap items-center justify-end gap-2">
            {run.status === "paused" ? <button type="button" onClick={onResume} disabled={busy} className={ACTION_BUTTON}><Play size={13} /> Reanudar</button>
              : ACTIVE_STATUSES.includes(run.status) && run.status !== "cancelling" ? <button type="button" onClick={onPause} disabled={busy} className={ACTION_BUTTON}><Pause size={13} /> Pausar</button> : null}
            {ACTIVE_STATUSES.includes(run.status) && run.status !== "cancelling" && <button type="button" onClick={onCancel} disabled={busy} className={`${ACTION_BUTTON} text-destructive`}><Ban size={13} /> Cancelar</button>}
            {run.status === "failed" && <button type="button" onClick={onRetry} disabled={busy} className={ACTION_BUTTON}><RotateCcw size={13} /> Reintentar modelo</button>}
            {run.status === "completed" && !run.isBaseline && <button type="button" onClick={onBaseline} disabled={busy} className={ACTION_BUTTON}><Star size={13} /> Usar como referencia</button>}
            {terminal && <button type="button" onClick={onRerun} disabled={busy} className={ACTION_BUTTON}><RotateCcw size={13} /> Repetir todo</button>}
            {run.attempts.length > 0 && <button type="button" onClick={onToggleLog} className={ACTION_BUTTON}><Terminal size={13} /> {logOpen ? "Ocultar log" : "Log"}</button>}
          </div>
        </div>

        <div className="mt-4 h-1.5 overflow-hidden rounded-full bg-muted"><div className={`h-full transition-all ${run.status === "failed" ? "bg-destructive" : "bg-primary"}`} style={{ width: `${progress}%` }} /></div>
        <div className="mt-2 flex flex-wrap justify-between gap-2 text-xs text-muted-foreground"><span>{run.currentModelIndex}/{run.models.length} modelos aceptados · {progress}%</span><span className="font-mono">{run.currentModel ? `Ahora: ${run.currentModel}` : run.models.join(", ")}</span></div>
        {run.phase && <div className="mt-2 inline-flex items-center gap-1.5 rounded-full bg-sky-500/10 px-2 py-1 text-xs text-sky-300"><Loader2 size={11} className={ACTIVE_STATUSES.includes(run.status) && run.status !== "paused" ? "animate-spin" : ""} />{PHASE_LABEL[run.phase] ?? run.phase}</div>}

        <ExecutionEnvironment attempt={[...run.attempts].reverse().find((attempt) => Object.keys(attempt.environment ?? {}).length > 0)} />

        {run.pauseReason && <div className="mt-3 flex items-start gap-2 rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-300"><Activity size={14} className="mt-0.5 shrink-0" />{run.pauseReason === "external_traffic" ? "Pausado por tráfico externo. Se reanudará automáticamente cuando el servidor quede libre." : "Pausado manualmente. Usa Reanudar cuando quieras continuar."}</div>}
        {run.error && <div className="mt-3 rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">{run.error}</div>}
        {run.attempts.length > 0 && <Attempts attempts={run.attempts} />}
      </div>

      {logOpen && <div className="border-t border-border bg-black/40"><div className="flex items-center justify-between px-3 py-2"><span className="text-xs font-medium text-zinc-300">Salida del último intento</span><div className="flex gap-1"><button type="button" onClick={onRefreshLog} className="rounded p-1.5 text-zinc-400 hover:bg-white/10" title="Actualizar log"><RefreshCw size={13} /></button><button type="button" onClick={() => void copyLog()} className="rounded p-1.5 text-zinc-400 hover:bg-white/10" title="Copiar log"><Copy size={13} /></button></div></div><pre className="max-h-96 overflow-auto border-t border-white/10 p-3 text-[11px] leading-relaxed text-zinc-300">{log?.join("\n") || "Sin salida todavía"}</pre></div>}
    </article>
  )
}

function Attempts({ attempts }: { attempts: BenchmarkAttempt[] }) {
  return (
    <details className="mt-3 rounded-lg border border-border px-3 py-2 text-xs">
      <summary className="flex cursor-pointer list-none items-center justify-between text-muted-foreground"><span className="inline-flex items-center gap-1.5"><Clipboard size={13} /> Historial de intentos ({attempts.length})</span><ChevronDown size={13} /></summary>
      <div className="mt-2 overflow-x-auto">
        <table className="w-full min-w-[760px] text-left"><thead className="text-muted-foreground"><tr><th className="py-2">Modelo</th><th>Intento</th><th>Estado</th><th>Duración</th><th>GPU</th><th>Peticiones</th><th>Tráfico externo</th><th>Error</th></tr></thead>
          <tbody>{[...attempts].reverse().map((attempt) => {
            const gpus = attempt.environment?.hardware?.gpus ?? []
            return <tr key={attempt.id} className="border-t border-border/60"><td className="py-2 font-mono">{attempt.model}</td><td>#{attempt.attemptNumber}</td><td className={attempt.status === "completed" ? "text-emerald-400" : attempt.status === "failed" ? "text-destructive" : attempt.status === "interrupted" || attempt.status === "contaminated" ? "text-amber-400" : ""}>{ATTEMPT_LABEL[attempt.status]}</td><td>{formatDuration(attempt.startedAt, attempt.finishedAt)}</td><td>{gpus.length ? gpus.map((gpu) => `#${gpu.index}`).join(" + ") : "—"}</td><td>{attempt.environment?.requests?.count ?? "—"}</td><td>{attempt.externalRequests.length || "—"}</td><td className="max-w-72 truncate text-muted-foreground" title={attempt.error ?? undefined}>{attempt.error ?? "—"}</td></tr>
          })}</tbody>
        </table>
      </div>
    </details>
  )
}

function ExecutionEnvironment({ attempt }: { attempt?: BenchmarkAttempt }) {
  if (!attempt) return null
  const environment = attempt.environment
  const software = environment.software ?? {}
  const benchmark = environment.benchmark ?? {}
  const model = environment.model ?? {}
  const hardware = environment.hardware ?? {}
  const requests = environment.requests ?? {}
  const gpus = hardware.gpus ?? []
  return (
    <details className="mt-3 rounded-lg border border-border bg-muted/20 px-3 py-2 text-xs">
      <summary className="flex cursor-pointer list-none items-center justify-between text-muted-foreground"><span className="inline-flex items-center gap-1.5"><Cpu size={13} /> Entorno reproducible · intento #{attempt.attemptNumber}</span><ChevronDown size={13} /></summary>
      <div className="mt-3 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <EnvironmentBlock icon={<GitCommit size={13} />} title="Versiones" lines={[
          software.ocabra ? `oCabra ${software.ocabra}` : null,
          software.scorer ? `Scorer ${software.scorer}` : null,
          software.python ? `Python ${software.python}` : null,
          ...(software.backends ?? []).map(backendVersionLabel),
          benchmark.commit ? `Dataset ${benchmark.commit.slice(0, 12)}` : null,
        ]} />
        <EnvironmentBlock icon={<Cpu size={13} />} title="Modelo" lines={[
          model.profile ?? attempt.model,
          ...(model.backends ?? []).map((backend) => `Backend ${backend}`),
          ...(model.runtimePresets ?? []).map((preset) => `Preset ${preset}`),
          ...(model.effectiveContexts ?? []).map((context) => `Contexto ${context.toLocaleString()}`),
        ]} />
        <EnvironmentBlock icon={<Server size={13} />} title="Hardware" lines={[
          ...gpus.map((gpu) => `GPU ${gpu.index}: ${gpu.name ?? "desconocida"}${gpu.totalVramMb ? ` · ${(gpu.totalVramMb / 1024).toFixed(0)} GB` : ""}`),
          ...(hardware.nodes ?? []).map((node) => `Nodo ${node}`),
        ]} />
        <EnvironmentBlock icon={<Activity size={13} />} title="Carga medida" lines={[
          `${requests.count ?? 0} peticiones`,
          `${(requests.inputTokens ?? 0).toLocaleString()} tokens entrada`,
          `${(requests.outputTokens ?? 0).toLocaleString()} tokens salida`,
          requests.errors ? `${requests.errors} errores` : "Sin errores HTTP",
        ]} />
      </div>
    </details>
  )
}

function EnvironmentBlock({ icon, title, lines }: { icon: ReactNode; title: string; lines: Array<string | null> }) {
  const visible = lines.filter((line): line is string => Boolean(line))
  return <div><p className="mb-1 inline-flex items-center gap-1 font-medium">{icon}{title}</p>{visible.length > 0 ? visible.map((line) => <p key={line} className="truncate text-muted-foreground" title={line}>{line}</p>) : <p className="text-muted-foreground">Sin datos</p>}</div>
}
