import * as DropdownMenu from "@radix-ui/react-dropdown-menu"
import {
  AlertTriangle,
  Copy,
  Cpu,
  FlaskConical,
  Globe,
  HardDrive,
  Loader2,
  Minimize2,
  MoreHorizontal,
  Pencil,
  Play,
  Plus,
  RefreshCw,
  Square,
  Star,
  Trash2,
  Wrench,
  X,
} from "lucide-react"
import { toast } from "sonner"
import { BackendIcons, backendLabel, mergeBackendItems } from "@/components/models/BackendIcons"
import { CapabilityIcons } from "@/components/models/CapabilityIcons"
import { LoadPolicyBadge } from "@/components/models/LoadPolicyBadge"
import { ContextBar } from "@/components/dashboard/ContextBar"
import { GpuBadge } from "@/components/dashboard/GpuBadge"
import { cardState, requestColorOrder, StatePill, type CardState } from "@/components/dashboard/ModelLiveCard"
import { VISUAL, fmtAgo, fmtContext, fmtDuration, fmtTps } from "@/components/dashboard/format"
import { getVllmConfig } from "@/lib/modelContext"
import type {
  BackendType,
  GPUState,
  LogicalModel,
  ModelActivityMap,
  ModelDeployment,
  ModelState,
  ModelStatus,
  RuntimePreset,
} from "@/types"

/** What the card can ask the page to do. Every id is a deployment id. */
export interface ModelCardActions {
  onLoad: (modelId: string) => void
  onUnload: (modelId: string) => void
  onCancelLoad: (modelId: string) => void
  onDownsize: (modelId: string) => void
  /** Opens the editor page; ``query`` without the leading "?". */
  onEdit: (modelId: string, query?: string) => void
  onPlayground: (modelId: string) => void
  onUpdate: (model: ModelState) => void
  onCompile: (model: ModelState) => void
  onDelete: (model: ModelState) => void
  /** Logical view only. */
  onSetDefault?: (backend: BackendType) => void
  onAddDeployment?: (backend: BackendType) => void
  onRemoveDeployment?: (deployment: ModelDeployment) => void
}

interface ModelCardProps {
  /** Logical view: one card per public name, with all its deployments. */
  logical?: LogicalModel
  /** Deployment view: the deployment to show. Ignored when ``logical`` is set. */
  deploymentId?: string
  states: Record<string, ModelState | undefined>
  live: ModelActivityMap
  gpus: GPUState[]
  nowMs: number
  busy?: boolean
  actions: ModelCardActions
}

export function formatPreferredGpu(gpu: number | null | undefined): string {
  return gpu == null ? "GPU auto" : `GPU ${gpu}`
}

export function formatDiskSize(bytes: number | null | undefined): string | null {
  if (bytes == null || bytes <= 0) return null
  const gb = bytes / 1024 ** 3
  if (gb >= 1) return `${gb.toFixed(gb >= 100 ? 0 : 1)} GB`
  return `${(bytes / 1024 ** 2).toFixed(0)} MB`
}

const gb = (mb: number) => `${(mb / 1024).toFixed(1)} GB`

const ACTIVE_STATUSES: ModelStatus[] = ["loaded", "loading", "unloading"]

/** The deployment whose state the card shows: one that is running, else the
 *  one that serves the public name. */
export function featuredDeployment(
  model: LogicalModel,
  states: Record<string, ModelState | undefined>,
): ModelDeployment | undefined {
  const statusOf = (dep: ModelDeployment) => states[dep.modelId]?.status ?? dep.status
  const sorted = [...model.deployments].sort((a, b) => Number(b.isDefault) - Number(a.isDefault))
  return (
    sorted.find((dep) => statusOf(dep) === "loaded") ??
    sorted.find((dep) => ACTIVE_STATUSES.includes(statusOf(dep) as ModelStatus)) ??
    sorted[0]
  )
}

export function summarizeError(model: ModelState): string | null {
  const error = model.errorMessage?.trim()
  if (!error) return null
  const needsTrustRemote =
    error.includes("trust_remote_code") ||
    error.includes("--trust-remote-code") ||
    error.includes("custom tokenizer") ||
    error.includes("Tokenizer class")
  if (needsTrustRemote) {
    return getVllmConfig(model)?.trustRemoteCode
      ? "Sigue fallando con «trust remote code» activado: parece una incompatibilidad real del tokenizer con esta versión de vLLM o transformers."
      : "Parece que necesita trust_remote_code: actívalo en el editor."
  }
  const first = error.split("\n")[0]?.trim() || "Error al cargar el modelo."
  return first.length > 200 ? `${first.slice(0, 197)}…` : first
}

/** Runtime presets of the default profile, the ones a load can pick. */
function rungsOf(model: ModelState | undefined): RuntimePreset[] {
  const profiles = model?.profiles ?? []
  const profile = profiles.find((p) => p.isDefault && p.enabled) ?? profiles.find((p) => p.enabled) ?? profiles[0]
  return [...(profile?.runtimePresets ?? [])]
    .filter((preset) => preset.enabled)
    .sort((a, b) => a.priority - b.priority || a.presetKey.localeCompare(b.presetKey))
}

/** API name a client sends in ``model``: the public name in the logical view,
 *  the default profile otherwise. */
function apiNameOf(logical: LogicalModel | undefined, model: ModelState | undefined, modelId: string): string {
  if (logical) return logical.modelKey
  const profiles = model?.profiles ?? []
  return (profiles.find((p) => p.isDefault) ?? profiles[0])?.profileId ?? modelId
}

function stateOf(model: ModelState | undefined, status: ModelStatus | null, live: CardState): CardState & { loadedIdle?: boolean } {
  if (model?.runtimePhase === "upsizing") return { visual: "loading", label: "Ampliando", detail: live.detail }
  if (model?.runtimePhase === "downsizing") return { visual: "loading", label: "Reduciendo", detail: live.detail }
  if (status === "loading" || live.visual === "loading" || live.visual === "waiting_space") return live
  if (status === "unloading") return { visual: "processing", label: "Descargando" }
  if (status === "error") return { visual: "stuck", label: "Error" }
  if (status === "loaded") return live.visual === "idle" ? { visual: "idle", label: "Cargado", loadedIdle: true } : live
  return { visual: "idle", label: status === "unloaded" ? "Descargado" : "Apagado" }
}

function Pill({ state }: { state: CardState & { loadedIdle?: boolean } }) {
  if (state.loadedIdle) {
    return (
      <span className="inline-flex shrink-0 items-center gap-1.5 rounded-full border border-emerald-500/40 bg-emerald-500/10 px-2.5 py-0.5 text-xs font-semibold text-emerald-300">
        <span className="h-2 w-2 rounded-full bg-emerald-400" />
        {state.label}
      </span>
    )
  }
  // Sin el detalle: en la cabecera le quitaría sitio al nombre. Va debajo.
  return <StatePill state={{ visual: state.visual, label: state.label }} />
}

function Stat({ label, value, title }: { label: string; value: React.ReactNode; title?: string }) {
  return (
    <div className="min-w-[3.75rem] flex-1 rounded-lg bg-muted/45 px-2 py-1" title={title}>
      <span className="block whitespace-nowrap text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">{label}</span>
      <span className="block whitespace-nowrap text-[13px] font-semibold tabular-nums">{value}</span>
    </div>
  )
}

const menuItem =
  "flex cursor-pointer items-center gap-2 rounded px-2 py-1.5 text-sm outline-none data-[highlighted]:bg-muted data-[disabled]:cursor-not-allowed data-[disabled]:opacity-50"
const menuLabel = "px-2 pb-0.5 pt-1.5 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground"
const btn =
  "inline-flex items-center gap-1.5 rounded-md border px-2 py-1 text-xs font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50"

/**
 * Tarjeta de un modelo en /models. Con ``logical`` resume un nombre público y
 * todos sus despliegues (el estado en vivo es el del que corre, o el que sirve
 * el nombre); sin él, un único despliegue. El estado en vivo sale del mismo
 * canal que el dashboard (``models_activity``).
 */
export function ModelCard({ logical, deploymentId, states, live, gpus, nowMs, busy = false, actions }: ModelCardProps) {
  const featured = logical ? featuredDeployment(logical, states) : undefined
  const modelId = featured?.modelId ?? deploymentId ?? (logical ? `${logical.defaultBackend}/${logical.modelKey}` : "")
  const model = states[modelId]
  const defaultDeployment = logical
    ? (logical.deployments.find((dep) => dep.isDefault) ?? logical.deployments[0])
    : undefined
  const editId = defaultDeployment?.modelId ?? modelId
  const title = logical ? logical.displayName || logical.modelKey : (model?.displayName ?? modelId)
  const apiName = apiNameOf(logical, model, modelId)
  const status: ModelStatus | null = model?.status ?? featured?.status ?? null
  const loadPolicy = model?.loadPolicy ?? featured?.loadPolicy ?? null
  const preferredGpu = model?.preferredGpu ?? featured?.preferredGpu ?? null
  const backend = model?.backendType ?? featured?.backend

  const act = live.activity[modelId]
  const ctx = live.contexts?.[modelId]
  const phase = live.modelStates?.[modelId]
  const vramByGpu = live.vramByGpu?.[modelId]
  const liveState = model ? cardState(model, act, phase, live.stuckThresholdSeconds) : { visual: "idle" as const, label: "Inactivo" }
  const state = stateOf(model, status, liveState)
  const visual = VISUAL[state.visual]
  const accent = state.loadedIdle ? "#10b981" : state.visual === "idle" ? null : visual.color

  const loaded = status === "loaded"
  const loading = status === "loading" || model?.runtimePhase === "upsizing" || phase?.phase === "loading"
  const inFlight = act?.inFlight ?? 0
  const queued = act?.queuedCount ?? act?.admission?.queuedRequests ?? 0
  const genTps = act?.generationTokensPerSecondTotal ?? act?.tokensPerSecondTotal ?? null
  const prefillTps = act?.prefillTokensPerSecondTotal ?? null
  const activePreset = loaded ? (ctx?.activePresetKey ?? act?.runtimePreset ?? null) : null
  const rungs = rungsOf(model)
  const errorText = status === "error" && model ? summarizeError(model) : null
  const disk = formatDiskSize(model?.diskSizeBytes)

  let idleLeft: number | null = null
  if (live.idleTimeoutSeconds && loadPolicy === "on_demand" && loaded && inFlight === 0 && model?.lastRequestAt) {
    const since = (nowMs - new Date(model.lastRequestAt).getTime()) / 1000
    if (Number.isFinite(since)) idleLeft = Math.max(0, Math.round(live.idleTimeoutSeconds - since))
  }
  let lastUsed: string | null = null
  if (model?.lastRequestAt) {
    const since = Math.round((nowMs - new Date(model.lastRequestAt).getTime()) / 1000)
    if (Number.isFinite(since)) lastUsed = `usado hace ${fmtAgo(Math.max(0, since))}`
  }

  const loadPct =
    phase?.loadingForMs != null && phase.typicalLoadMs ? Math.min(97, (phase.loadingForMs / phase.typicalLoadMs) * 100) : null
  const gpuIdx = model?.currentGpu ?? []

  const backendItems = logical
    ? mergeBackendItems(logical.deployments)
    : backend
      ? [{ backend, deployed: true, isDefault: false, modelId }]
      : []
  const addable = logical
    ? logical.compatibleBackends.filter((entry) => !entry.deployed && entry.installed && entry.status === "compatible")
    : []
  const multi = (logical?.deployments.length ?? 0) > 1

  const copyApiName = async () => {
    try {
      await navigator.clipboard.writeText(apiName)
      toast.success(`Copiado: ${apiName}`)
    } catch {
      toast.error("No se pudo copiar al portapapeles")
    }
  }

  return (
    <article
      className="flex min-w-0 flex-col gap-3 rounded-2xl border border-l-4 border-border bg-card p-4 shadow-sm"
      style={accent ? { borderLeftColor: accent } : undefined}
      data-testid={logical ? `logical-model-${logical.modelKey}` : `deployment-${modelId}`}
      data-model-id={modelId}
      data-visual={state.loadedIdle ? "loaded" : state.visual}
    >
      {/* Cabecera: nombre, backends, capacidades y política */}
      <header className="min-w-0 space-y-2">
        <div className="flex min-w-0 items-start gap-2">
          <div className="min-w-0 flex-1">
            <div className="flex min-w-0 items-start gap-1">
              <h3 className="line-clamp-2 min-w-0 break-words text-[15px] font-semibold leading-snug tracking-tight" title={title}>
                {title}
              </h3>
              <button
                type="button"
                onClick={() => void copyApiName()}
                className="shrink-0 rounded p-1 text-muted-foreground hover:bg-muted hover:text-foreground"
                aria-label={`Copiar nombre de API ${apiName}`}
                title={`Copiar el nombre de API: ${apiName}`}
              >
                <Copy size={13} />
              </button>
            </div>
            {apiName !== title && <p className="truncate font-mono text-[11px] text-muted-foreground">{apiName}</p>}
          </div>
          <Pill state={state} />
        </div>

        <div className="flex flex-wrap items-center gap-1.5">
          <BackendIcons
            items={backendItems}
            onSelect={(item) => item.modelId && actions.onEdit(item.modelId)}
          />
          {model && <CapabilityIcons model={model} />}
        </div>

        <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-muted-foreground">
          {loadPolicy && <LoadPolicyBadge policy={loadPolicy} />}
          {model?.federation?.remote && (
            <span className="inline-flex items-center gap-1 rounded-md bg-sky-500/15 px-1.5 py-0.5 text-sky-200" title="Modelo federado">
              <Globe size={11} />
              {model.federation.nodeName || model.federation.nodeId}
            </span>
          )}
          {disk && (
            <span className="inline-flex items-center gap-1" title="Tamaño en disco">
              <HardDrive size={11} />
              {disk}
            </span>
          )}
          <span>{lastUsed ?? "sin uso registrado"}</span>
        </div>
      </header>

      {/* Estado en vivo */}
      <section className="min-w-0 space-y-2" aria-label="Estado">
        {loading && (
          <div className="space-y-1">
            {state.detail && <p className={`text-xs ${visual.text}`}>{state.detail}</p>}
            <div className="h-1.5 overflow-hidden rounded-full bg-muted">
              <div
                className={`h-full rounded-full ${loadPct == null ? "w-1/3 animate-pulse" : ""}`}
                style={{ background: VISUAL.loading.color, width: loadPct == null ? undefined : `${loadPct}%` }}
              />
            </div>
          </div>
        )}

        {errorText && (
          <div className="rounded-lg border border-red-500/40 bg-red-500/10 px-3 py-2 text-xs text-red-200">
            <p className="flex items-start gap-1.5 break-words">
              <AlertTriangle size={13} className="mt-0.5 shrink-0" />
              <span className="min-w-0">{errorText}</span>
            </p>
            <button
              type="button"
              onClick={() => actions.onEdit(modelId, "s=summary")}
              className="mt-1 font-medium text-red-100 underline underline-offset-2 hover:text-white"
            >
              Ver en el editor
            </button>
          </div>
        )}

        {(loaded || loading) && (gpuIdx.length > 0 || (model?.ramUsedMb ?? 0) > 0 || (featured && !featured.isDefault)) && (
          <div className="flex flex-wrap items-center gap-1.5 text-xs">
            {featured && !featured.isDefault && multi && (
              <span className="text-muted-foreground" title="No es el backend que sirve el nombre público">
                con {backendLabel(featured.backend)}
              </span>
            )}
            {gpuIdx.map((index) => {
              const mb = vramByGpu?.[String(index)] ?? (gpuIdx.length === 1 ? model?.vramUsedMb : undefined)
              return (
                <span key={index} className="inline-flex items-center gap-1">
                  <GpuBadge index={index} gpus={gpus} />
                  {mb != null && mb > 0 && <span className="tabular-nums text-muted-foreground">{gb(mb)}</span>}
                </span>
              )
            })}
            {gpuIdx.length > 1 && !vramByGpu && (model?.vramUsedMb ?? 0) > 0 && (
              <span className="tabular-nums text-muted-foreground">{gb(model!.vramUsedMb)} en total</span>
            )}
            {(model?.ramUsedMb ?? 0) > 0 && (
              <span className="inline-flex items-center gap-1 tabular-nums text-muted-foreground">
                <Cpu size={11} />
                {gb(model!.ramUsedMb)} RAM
              </span>
            )}
          </div>
        )}

        {!loaded && !loading && !errorText && (
          <p className="text-xs text-muted-foreground">
            Sin cargar ·{" "}
            <span data-testid="preferred-gpu">{formatPreferredGpu(preferredGpu)}</span> preferida
          </p>
        )}

        {loaded && (
          <>
            {state.detail && <p className={`text-xs ${visual.text}`}>{state.detail}</p>}
            <div className="flex flex-wrap gap-1.5">
              <Stat label="En curso" value={Math.max(0, inFlight - queued)} />
              <Stat label="En cola" value={queued} />
              <Stat label="Gen." value={fmtTps(genTps)} title="Tokens por segundo generados, sumando las peticiones" />
              <Stat label="Prefill" value={fmtTps(prefillTps)} title="Tokens por segundo de prompt procesados" />
            </div>
            {ctx && <ContextBar ctx={ctx} requestOrder={requestColorOrder(act?.requests)} />}
            {idleLeft != null && (
              <p className="text-xs tabular-nums text-muted-foreground" title="on_demand: se descarga si no recibe peticiones">
                Se descarga por inactividad en {fmtDuration(idleLeft * 1000)}
              </p>
            )}
          </>
        )}
      </section>

      {/* Peldaños del perfil por defecto */}
      {rungs.length > 0 && (
        <section className="min-w-0" aria-label="Peldaños">
          <div className="flex flex-wrap gap-1.5">
            {rungs.map((preset) => {
              const active = activePreset === preset.presetKey
              return (
                <span
                  key={preset.id || preset.presetKey}
                  data-rung={preset.presetKey}
                  data-active={active || undefined}
                  className={`inline-flex max-w-full items-center gap-1 rounded-md border px-1.5 py-0.5 text-[11px] ${
                    active ? "border-emerald-500/60 bg-emerald-500/15 text-emerald-100" : "border-border bg-muted/40 text-muted-foreground"
                  }`}
                  title={`${preset.label || preset.presetKey}: contexto ${preset.contextMode === "max" ? "máximo" : fmtContext(preset.contextTokens)}, ${preset.parallelism} a la vez${active ? " · cargado ahora" : ""}`}
                >
                  {preset.isDefault && <Star size={10} className="shrink-0 fill-current" aria-label="por defecto" />}
                  <span className="whitespace-nowrap font-medium">{preset.label || preset.presetKey}</span>
                  <span className="shrink-0 whitespace-nowrap tabular-nums opacity-80">
                    · {preset.contextMode === "max" ? "máx" : fmtContext(preset.contextTokens)}
                  </span>
                  {(preset.preferredGpus ?? []).map((index) => (
                    <GpuBadge key={index} index={index} gpus={gpus} className="!text-[10px]" />
                  ))}
                </span>
              )
            })}
          </div>
        </section>
      )}

      {/* Controles */}
      <footer className="mt-auto flex flex-wrap items-center gap-1.5 border-t border-border pt-3">
        {loaded ? (
          <button
            type="button"
            disabled={busy}
            onClick={() => actions.onUnload(modelId)}
            className={`${btn} border-red-500/30 text-red-300 hover:border-red-500/50 hover:bg-red-500/15`}
          >
            <Square size={12} />
            Descargar
          </button>
        ) : loading ? (
          model?.canCancelLoad ? (
            <button
              type="button"
              disabled={busy}
              onClick={() => actions.onCancelLoad(modelId)}
              className={`${btn} border-amber-500/40 text-amber-200 hover:bg-amber-500/15`}
            >
              <X size={12} />
              Cancelar carga
            </button>
          ) : (
            <span className={`${btn} border-border text-muted-foreground`}>
              <Loader2 size={12} className="animate-spin" />
              Cargando
            </span>
          )
        ) : (
          <button
            type="button"
            disabled={busy || status === "unloading" || !model}
            onClick={() => actions.onLoad(modelId)}
            className={`${btn} border-primary/50 bg-primary/15 text-foreground hover:bg-primary/25`}
          >
            {busy ? <Loader2 size={12} className="animate-spin" /> : <Play size={12} />}
            Cargar
          </button>
        )}
        {loaded && model?.canDownsize && (
          <button
            type="button"
            disabled={busy}
            onClick={() => actions.onDownsize(modelId)}
            className={`${btn} border-border hover:bg-muted`}
            title="Recargar en una sola GPU"
          >
            <Minimize2 size={12} />1 GPU
          </button>
        )}
        <button
          type="button"
          onClick={() => actions.onEdit(editId)}
          className={`${btn} border-border hover:bg-muted`}
          aria-label={`Editar ${logical?.modelKey ?? title}`}
        >
          <Pencil size={12} />
          Editar
        </button>
        <button
          type="button"
          onClick={() => actions.onPlayground(modelId)}
          className={`${btn} border-border hover:bg-muted`}
          title={`Abrir ${modelId} en el Playground`}
        >
          <FlaskConical size={12} />
          Probar
        </button>

        <DropdownMenu.Root>
          <DropdownMenu.Trigger
            className="ml-auto rounded-md p-1.5 text-muted-foreground hover:bg-muted hover:text-foreground"
            aria-label={`Más acciones de ${title}`}
          >
            <MoreHorizontal size={15} />
          </DropdownMenu.Trigger>
          <DropdownMenu.Portal>
            <DropdownMenu.Content
              align="end"
              sideOffset={4}
              className="z-50 min-w-[14rem] rounded-md border border-border bg-card p-1 shadow-lg"
            >
              {model && (
                <DropdownMenu.Item className={menuItem} onSelect={() => actions.onUpdate(model)}>
                  <RefreshCw size={13} />
                  Actualizar (re-pull)
                </DropdownMenu.Item>
              )}
              {model?.backendType === "vllm" && (
                <DropdownMenu.Item className={menuItem} onSelect={() => actions.onCompile(model)}>
                  <Wrench size={13} />
                  Compilar TRT
                </DropdownMenu.Item>
              )}

              {logical && multi && (
                <>
                  <DropdownMenu.Separator className="my-1 h-px bg-border" />
                  <DropdownMenu.Label className={menuLabel}>Despliegues</DropdownMenu.Label>
                  {logical.deployments.map((dep) => (
                    <DropdownMenu.Item key={dep.modelId} className={menuItem} onSelect={() => actions.onEdit(dep.modelId)}>
                      <Pencil size={13} />
                      <span className="flex-1">Editar {backendLabel(dep.backend)}</span>
                      <span className="text-[11px] text-muted-foreground" data-testid={`preferred-gpu-${dep.backend}`}>
                        {formatPreferredGpu(states[dep.modelId]?.preferredGpu ?? dep.preferredGpu)}
                      </span>
                    </DropdownMenu.Item>
                  ))}
                  {actions.onSetDefault &&
                    logical.deployments
                      .filter((dep) => !dep.isDefault)
                      .map((dep) => (
                        <DropdownMenu.Item
                          key={`default-${dep.backend}`}
                          className={menuItem}
                          onSelect={() => actions.onSetDefault?.(dep.backend)}
                        >
                          <Star size={13} />
                          Servir {logical.modelKey} con {backendLabel(dep.backend)}
                        </DropdownMenu.Item>
                      ))}
                </>
              )}

              {logical && (addable.length > 0 || actions.onRemoveDeployment) && (
                <>
                  <DropdownMenu.Separator className="my-1 h-px bg-border" />
                  {actions.onAddDeployment &&
                    addable.map((entry) => (
                      <DropdownMenu.Item
                        key={`add-${entry.backend}`}
                        className={menuItem}
                        onSelect={() => actions.onAddDeployment?.(entry.backend as BackendType)}
                      >
                        <Plus size={13} />
                        Añadir backend {backendLabel(entry.backend)}
                      </DropdownMenu.Item>
                    ))}
                  {actions.onRemoveDeployment &&
                    logical.deployments.map((dep) => (
                      <DropdownMenu.Item
                        key={`remove-${dep.backend}`}
                        className={`${menuItem} text-red-300`}
                        onSelect={() => actions.onRemoveDeployment?.(dep)}
                      >
                        <X size={13} />
                        Quitar despliegue {dep.backend}
                      </DropdownMenu.Item>
                    ))}
                </>
              )}

              {model && (
                <>
                  <DropdownMenu.Separator className="my-1 h-px bg-border" />
                  <DropdownMenu.Item className={`${menuItem} text-red-300`} onSelect={() => actions.onDelete(model)}>
                    <Trash2 size={13} />
                    Borrar {logical && multi ? backendLabel(model.backendType) : "modelo"} y ficheros
                  </DropdownMenu.Item>
                </>
              )}
            </DropdownMenu.Content>
          </DropdownMenu.Portal>
        </DropdownMenu.Root>
      </footer>
    </article>
  )
}
