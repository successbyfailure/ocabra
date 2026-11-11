import { useEffect, useMemo, useState } from "react"
import * as Tooltip from "@radix-ui/react-tooltip"
import { Cpu, X } from "lucide-react"
import { api } from "@/api/client"
import { EmptyState } from "@/components/common/EmptyState"
import { CpuWidget, GpuWidget, type HostHistoryPoint } from "@/components/dashboard/DeviceWidget"
import { FederationSection } from "@/components/dashboard/FederationSection"
import { KpiStrip } from "@/components/dashboard/KpiStrip"
import { ModelLiveCard } from "@/components/dashboard/ModelLiveCard"
import { QueueSection } from "@/components/dashboard/QueueSection"
import { RecentRequestsTable } from "@/components/dashboard/RecentRequestsTable"
import { Section } from "@/components/dashboard/Section"
import { GpuBadge } from "@/components/dashboard/GpuBadge"
import { useLiveActivity } from "@/components/dashboard/useLiveActivity"
import { useCurrentUser, useIsAdmin, useIsModelManager } from "@/hooks/useAuth"
import { useGpuStore } from "@/stores/gpuStore"
import { useModelStore } from "@/stores/modelStore"
import { useServiceStore } from "@/stores/serviceStore"
import type { ActiveRequestProgress, HostStats, ModelState, OllamaRuntimeInfo, ServerPower } from "@/types"

function CountBadge({ n, tone = "emerald" }: { n: number; tone?: "emerald" | "amber" }) {
  if (n <= 0) return null
  const cls = tone === "amber" ? "bg-amber-500/15 text-amber-300" : "bg-emerald-500/15 text-emerald-300"
  return <span className={`ml-1 rounded-full px-2 py-0.5 text-xs ${cls}`}>{n}</span>
}

function ollamaRuntimeFor(model: ModelState, runtime: Record<string, OllamaRuntimeInfo>) {
  return model.backendType === "ollama" ? runtime[model.modelId.replace(/^ollama\//, "")] : undefined
}

/**
 * Dashboard: lo que está pasando ahora.
 *
 * Arriba los KPIs (en vivo y última hora frente a 24 h); después cada modelo
 * activo con sus peticiones en curso como barras de progreso y su barra de
 * contexto; la cola en el orden real de turno; GPUs y CPU con lo que tienen
 * cargado; federación; y las últimas peticiones terminadas. El histórico está
 * en /stats. La actividad llega por WebSocket cada segundo (con polling REST
 * de respaldo); el resto, por REST a ritmo más lento.
 */
export function Dashboard() {
  const isAdmin = useIsAdmin()
  const isModelManager = useIsModelManager()
  const currentUser = useCurrentUser()
  const [error, setError] = useState<string | null>(null)
  const [nowMs, setNowMs] = useState(() => Date.now())
  const [hostStats, setHostStats] = useState<HostStats | null>(null)
  const [serverPower, setServerPower] = useState<ServerPower | null>(null)
  const [hostHistory, setHostHistory] = useState<HostHistoryPoint[]>([])
  const [ollamaRuntime, setOllamaRuntime] = useState<Record<string, OllamaRuntimeInfo>>({})

  const { data: live } = useLiveActivity()
  const activity = live.activity
  const contexts = live.contexts ?? {}
  const stuckThreshold = live.stuckThresholdSeconds ?? 300

  const gpus = useGpuStore((s) => s.gpus)
  const setGpus = useGpuStore((s) => s.setGpus)
  const models = useModelStore((s) => s.models)
  const setModels = useModelStore((s) => s.setModels)
  const unloadModel = useModelStore((s) => s.unloadModel)
  const services = useServiceStore((s) => s.services)
  const setServices = useServiceStore((s) => s.setServices)
  const unloadService = useServiceStore((s) => s.unloadService)

  useEffect(() => {
    const id = window.setInterval(() => setNowMs(Date.now()), 1000)
    return () => window.clearInterval(id)
  }, [])

  useEffect(() => {
    let cancelled = false
    async function bootstrap() {
      try {
        const [gpuList, modelList, serviceList] = await Promise.all([api.gpus.list(), api.models.list(), api.services.list()])
        if (cancelled) return
        setGpus(gpuList)
        setModels(modelList)
        setServices(serviceList)
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : "No se pudieron cargar los datos del dashboard")
      }
    }
    void bootstrap()
    // El WebSocket avisa de cambios de estado pero no de currentGpu ni de la
    // VRAM medida tras cargar: hace falta la lista completa de vez en cuando.
    const id = window.setInterval(() => {
      void api.models.list().then((list) => !cancelled && setModels(list)).catch(() => undefined)
    }, 10_000)
    return () => {
      cancelled = true
      window.clearInterval(id)
    }
  }, [setGpus, setModels, setServices])

  useEffect(() => {
    let cancelled = false
    async function poll() {
      let stats: HostStats | null = null
      let power: ServerPower | null = null
      try {
        stats = await api.host.stats()
      } catch {
        /* se conserva lo último */
      }
      try {
        power = await api.stats.serverPower()
      } catch {
        /* se conserva lo último */
      }
      try {
        const rt = await api.models.ollamaRuntime()
        if (!cancelled) setOllamaRuntime(rt)
      } catch {
        /* sin Ollama */
      }
      if (cancelled) return
      if (stats) setHostStats(stats)
      if (power) setServerPower(power)
      if (stats) {
        const point: HostHistoryPoint = { t: Date.now(), cpuPct: stats.cpuPct, memPct: stats.memPct, powerW: power?.cpuPowerW ?? null }
        setHostHistory((prev) => [...prev.slice(-119), point])
      }
    }
    void poll()
    const id = window.setInterval(() => void poll(), 5000)
    return () => {
      cancelled = true
      window.clearInterval(id)
    }
  }, [])

  const modelList = useMemo(() => Object.values(models), [models])
  const activeModels = useMemo(
    () =>
      modelList
        .filter((m) => m.status === "loaded" || m.status === "loading")
        .sort((a, b) => (activity[b.modelId]?.inFlight ?? 0) - (activity[a.modelId]?.inFlight ?? 0) || a.displayName.localeCompare(b.displayName)),
    [modelList, activity],
  )
  const loadedModels = useMemo(() => modelList.filter((m) => m.status === "loaded"), [modelList])
  // CPU: lo que corre de verdad en la CPU local. Ollama no informa de índice
  // de GPU, así que sus modelos sin índice cuentan como CPU solo si Ollama
  // dice que tienen parte en CPU; si están enteros en GPU van aparte.
  const cpuModels = useMemo(
    () =>
      loadedModels.filter((m) => {
        // Usa RAM del host a propósito aunque corra en GPU (Strata): también
        // es carga del host, sale en su widget con lo que ocupa.
        if ((live.ramByModel?.[m.modelId] ?? 0) > 0) return true
        if ((m.currentGpu ?? []).length > 0) return false
        const rt = ollamaRuntimeFor(m, ollamaRuntime)
        return rt ? rt.cpuPct > 0 : true
      }),
    [loadedModels, ollamaRuntime, live.ramByModel],
  )
  const ollamaOnUnknownGpu = useMemo(
    () =>
      loadedModels.filter((m) => {
        if ((m.currentGpu ?? []).length > 0) return false
        const rt = ollamaRuntimeFor(m, ollamaRuntime)
        return rt != null && rt.cpuPct === 0
      }),
    [loadedModels, ollamaRuntime],
  )
  const serviceList = useMemo(() => Object.values(services), [services])
  const activeServices = useMemo(() => serviceList.filter((s) => s.runtimeLoaded), [serviceList])
  const externalRuntimeServices = useMemo(() => serviceList.filter((s) => !s.enabled && s.serviceAlive), [serviceList])
  const queuedTotal = useMemo(
    () => Object.values(activity).reduce((acc, a) => acc + (a.queuedCount ?? 0), 0),
    [activity],
  )

  // El backend solo deja cancelar las propias a quien no es admin: el botón
  // sale solo donde va a funcionar.
  const canCancel = (req: ActiveRequestProgress) =>
    isAdmin || (currentUser != null && req.username != null && req.username === currentUser.username)

  return (
    <Tooltip.Provider delayDuration={200}>
      <div className="space-y-6">
        <KpiStrip activity={activity} serverPower={serverPower} />

        <Section title="Modelos activos" badge={<CountBadge n={activeModels.length + activeServices.length} />}>
          <div className="space-y-3">
            {activeModels.map((model) => (
              <ModelLiveCard
                key={model.modelId}
                model={model}
                act={activity[model.modelId]}
                ctx={contexts[model.modelId]}
                phase={live.modelStates?.[model.modelId]}
                gpus={gpus}
                vramByGpu={live.vramByGpu?.[model.modelId]}
                hostRamMb={live.ramByModel?.[model.modelId]}
                hostTotalMb={hostStats?.memTotalMb ?? null}
                nowMs={nowMs}
                stuckThreshold={stuckThreshold}
                idleTimeoutSeconds={live.idleTimeoutSeconds}
                onUnload={(id) => void unloadModel(id)}
                canCancel={canCancel}
              />
            ))}

            {/* Servicios con runtime cargado: ComfyUI, A1111, ACE-Step, TRELLIS.2 */}
            {activeServices.map((service) => (
              <div
                key={service.serviceId}
                className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-border bg-card px-4 py-3 shadow-sm"
              >
                <div className="min-w-0 space-y-1">
                  <p className="truncate text-[14.5px] font-semibold tracking-tight">{service.displayName}</p>
                  <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                    <span className="inline-flex items-center rounded-full border border-sky-500/40 bg-sky-500/20 px-2.5 py-0.5 text-xs font-medium text-sky-200">
                      Servicio
                    </span>
                    {service.isGenerating && (
                      <span className="inline-flex animate-pulse items-center rounded-full border border-amber-500/40 bg-amber-500/20 px-2.5 py-0.5 text-xs font-medium text-amber-100">
                        Generando{service.queueDepth > 0 && ` +${service.queueDepth}`}
                      </span>
                    )}
                    {service.preferredGpu != null ? (
                      <GpuBadge index={service.preferredGpu} gpus={gpus} />
                    ) : (
                      <span className="inline-flex items-center gap-1 rounded-md bg-muted px-2 py-0.5">
                        <Cpu size={11} />
                        GPU —
                      </span>
                    )}
                    <span>
                      {service.vramUsedMb != null
                        ? `${service.vramUsedMb.toLocaleString()} MB`
                        : service.gpuUtilPct != null
                          ? `${service.gpuUtilPct.toFixed(0)}% util`
                          : "—"}
                    </span>
                  </div>
                </div>
                {isModelManager && (
                  <button
                    type="button"
                    onClick={() => void unloadService(service.serviceId)}
                    className="inline-flex items-center gap-1.5 rounded-md border border-red-500/30 px-3 py-1 text-xs font-medium text-red-300 transition-colors hover:border-red-500/50 hover:bg-red-500/15"
                  >
                    <X size={13} />
                    Descargar
                  </button>
                )}
              </div>
            ))}

            {externalRuntimeServices.length > 0 && (
              <div className="space-y-2 rounded-lg border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-sm text-amber-100">
                <p>Hay runtimes externos ocupando GPU en servicios de generación.</p>
                <ul className="space-y-0.5 text-xs text-amber-200/90">
                  {externalRuntimeServices.map((service) => (
                    <li key={service.serviceId} className="flex flex-wrap items-center gap-1.5">
                      <span className="font-medium text-amber-100">{service.displayName}</span>
                      {service.preferredGpu != null && <GpuBadge index={service.preferredGpu} gpus={gpus} />}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {activeModels.length === 0 && activeServices.length === 0 && externalRuntimeServices.length === 0 && (
              <EmptyState title="Sin modelos cargados" description="Carga un modelo desde Models o lanza una petición." />
            )}
          </div>
        </Section>

        <Section title="Cola" badge={<CountBadge n={queuedTotal} tone="amber" />}>
          <QueueSection activity={activity} models={models} />
        </Section>

        <Section title="GPUs y CPU">
          <div className="grid gap-4 xl:grid-cols-2">
            {gpus.map((gpu) => (
              <GpuWidget
                key={gpu.index}
                gpu={gpu}
                models={loadedModels}
                activity={activity}
                vramByGpu={live.vramByGpu}
                modelStates={live.modelStates}
                stuckThreshold={stuckThreshold}
              />
            ))}
            {hostStats && (
              <CpuWidget
                stats={hostStats}
                serverPower={serverPower}
                history={hostHistory}
                models={cpuModels}
                activity={activity}
                ollamaRuntime={ollamaRuntime}
                ramByModel={live.ramByModel}
                modelStates={live.modelStates}
                stuckThreshold={stuckThreshold}
              />
            )}
            {gpus.length === 0 && !hostStats && (
              <EmptyState title="Sin datos de GPU" description="No se detectaron GPUs o el monitor no está disponible." />
            )}
          </div>
          {ollamaOnUnknownGpu.length > 0 && (
            <p className="text-xs text-muted-foreground">
              En GPU vía Ollama, sin índice conocido: {ollamaOnUnknownGpu.map((m) => m.displayName).join(", ")}
            </p>
          )}
        </Section>

        <FederationSection />

        <Section title="Últimas peticiones">
          <RecentRequestsTable />
        </Section>

        {error && <div className="rounded-md border border-red-500/40 bg-red-500/10 px-4 py-3 text-sm text-red-200">{error}</div>}
      </div>
    </Tooltip.Provider>
  )
}
