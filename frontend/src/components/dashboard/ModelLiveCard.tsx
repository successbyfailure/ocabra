import { Cpu, X } from "lucide-react"
import { CapabilityIcons } from "@/components/models/CapabilityIcons"
import { LoadPolicyBadge } from "@/components/models/LoadPolicyBadge"
import {
  getLlamaCppConfig,
  getModelContextSummary,
  getVllmConfig,
  getWhisperConfig,
} from "@/lib/modelContext"
import type {
  ActiveRequestProgress,
  GPUState,
  ModelActivity,
  ModelContextInfo,
  ModelPhaseInfo,
  ModelState,
} from "@/types"
import { BackendIcon } from "./BackendIcon"
import { ContextBar, requestHue } from "./ContextBar"
import { VISUAL, fmtContext, fmtDuration, fmtTps, loadedMeta, type VisualState } from "./format"
import { RequestProgress } from "./RequestProgress"

/** Índice de color estable por petición: orden de llegada dentro del modelo. */
export function requestColorOrder(requests: ActiveRequestProgress[] = []): Map<string, number> {
  const sorted = [...requests].sort((a, b) => (a.startedAt ?? 0) - (b.startedAt ?? 0))
  return new Map(sorted.map((r, i) => [r.requestId, i]))
}

const gb = (mb: number) => (mb / 1024).toFixed(1)

function Chip({ children, className = "", title }: { children: React.ReactNode; className?: string; title?: string }) {
  return (
    <span className={`inline-flex items-center gap-1 rounded-md bg-muted px-2 py-0.5 ${className}`} title={title}>
      {children}
    </span>
  )
}

function Total({ label, value, title }: { label: string; value: React.ReactNode; title?: string }) {
  return (
    <div className="min-w-0 rounded-lg bg-muted/45 px-2.5 py-1.5" title={title}>
      <span className="block truncate text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">{label}</span>
      <span className="block truncate text-[13px] font-semibold tabular-nums">{value}</span>
    </div>
  )
}

export interface CardState {
  visual: VisualState
  label: string
  detail?: string
}

/**
 * Estado que resume la card, por prioridad: atascada > esperando espacio >
 * cargando > sin actividad > generando > prefill > procesando > en espera >
 * inactivo.
 */
export function cardState(
  model: ModelState,
  act: ModelActivity | undefined,
  phase: ModelPhaseInfo | undefined,
  stuckThreshold: number,
): CardState {
  const requests = act?.requests ?? []
  const count = (stage: string) => requests.filter((r) => r.stage === stage).length
  const plural = (n: number, one: string, many: string) => `${n} ${n === 1 ? one : many}`
  if ((act?.inFlight ?? 0) > 0 && (act?.oldestSeconds ?? 0) > stuckThreshold) {
    return { visual: "stuck", label: "Atascada", detail: `una petición lleva ${fmtDuration((act?.oldestSeconds ?? 0) * 1000)}` }
  }
  if (phase?.phase === "waiting_for_gpu") {
    return {
      visual: "waiting_space",
      label: "Esperando espacio en GPU",
      detail: phase.waitingOn ? `desalojando ${phase.waitingOn}` : "liberando VRAM",
    }
  }
  if (phase?.phase === "waiting_for_service_gpu") {
    return { visual: "waiting_space", label: "Esperando GPU", detail: `retenida por ${phase.waitingOn ?? "un servicio"}` }
  }
  if (phase?.phase === "loading" || model.status === "loading") {
    const elapsed = phase?.loadingForMs
    const typical = phase?.typicalLoadMs
    return {
      visual: "loading",
      label: "Cargando",
      detail:
        elapsed != null ? `${fmtDuration(elapsed)}${typical ? ` de ~${fmtDuration(typical)} habituales` : ""}` : undefined,
    }
  }
  const stalled = requests.filter((r) => r.stalled).length
  const running = requests.filter((r) => r.stage !== "queued" && r.stage !== "backend_wait").length
  if (stalled > 0 && stalled === running) {
    return { visual: "stuck", label: "Sin actividad", detail: plural(stalled, "petición parada", "peticiones paradas") }
  }
  if (count("generating") > 0) return { visual: "generating", label: "Generando", detail: plural(count("generating"), "petición", "peticiones") }
  if (count("prefill") > 0) return { visual: "prefill", label: "Procesando prompt", detail: plural(count("prefill"), "petición", "peticiones") }
  if (count("processing") > 0) {
    return { visual: "processing", label: "Procesando", detail: `${plural(count("processing"), "petición", "peticiones")} sin streaming` }
  }
  const waiting = count("queued") + count("backend_wait")
  if (waiting > 0) return { visual: "waiting", label: "En espera", detail: plural(waiting, "petición", "peticiones") }
  return { visual: "idle", label: "Inactivo" }
}

/** Píldora de estado: color y animación según `VISUAL`. ``compact`` para
 *  listas (widgets de GPU/CPU): sin detalle, el detalle va en el tooltip. */
export function StatePill({ state, compact = false }: { state: CardState; compact?: boolean }) {
  const v = VISUAL[state.visual]
  return (
    <span
      className={`inline-flex shrink-0 items-center gap-1.5 rounded-full border font-semibold ${v.bg} ${v.border} ${v.text} ${
        compact ? "px-1.5 py-0 text-[10.5px]" : "px-2.5 py-0.5 text-xs"
      }`}
      data-visual={state.visual}
      title={compact && state.detail ? `${state.label} · ${state.detail}` : undefined}
    >
      <span className={`${compact ? "h-1.5 w-1.5" : "h-2 w-2"} rounded-full ${v.anim}`} style={{ background: v.color }} />
      {state.label}
      {!compact && state.detail && <span className="font-normal opacity-80">· {state.detail}</span>}
    </span>
  )
}

/**
 * Memoria del modelo en cada GPU que ocupa: lo suyo (medido por NVML si se
 * conoce), lo del resto de la tarjeta y una marca en lo que reservó el
 * scheduler. Si lo medido supera la reserva, en ámbar: es el patrón previo a
 * un CUDA OOM.
 */
function ModelMemory({
  model,
  gpus,
  vramByGpu,
  visual,
}: {
  model: ModelState
  gpus: GPUState[]
  vramByGpu?: Record<string, number>
  visual: VisualState
}) {
  const indices = model.currentGpu ?? []
  if (indices.length === 0) return null
  const n = indices.length
  const reserved = model.vramEstimateMb ?? 0
  const mineOf = (idx: number) => vramByGpu?.[String(idx)] ?? model.vramUsedMb / n
  const totalMine = indices.reduce((acc, idx) => acc + mineOf(idx), 0)
  // La reserva del scheduler es un total. Con tensor-split el reparto real es
  // desigual (nemotron: 9,3 + 20,9 GB), así que la marca de cada GPU va en
  // proporción a lo que ocupa en ella, y el aviso de OOM compara totales: con
  // un reparto a partes iguales una GPU salía "por encima" sin estarlo.
  const over = reserved > 0 && totalMine > reserved * 1.02
  return (
    <div className="space-y-1" data-testid="model-memory">
      <p className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
        Memoria
        <span className={`ml-1.5 font-normal normal-case tracking-normal ${over ? "text-amber-400" : ""}`}>
          {gb(totalMine)} GB{reserved > 0 ? ` de ${gb(reserved)} reservados` : ""}
          {over ? " · por encima de la reserva: riesgo de OOM" : ""}
        </span>
      </p>
      {indices.map((idx) => {
        const gpu = gpus.find((g) => g.index === idx)
        if (!gpu || gpu.totalVramMb <= 0) return null
        const measured = vramByGpu?.[String(idx)]
        const mine = mineOf(idx)
        const others = Math.max(0, gpu.usedVramMb - mine)
        const pct = (mb: number) => Math.max(0, Math.min(100, (mb / gpu.totalVramMb) * 100))
        const reservedPerGpu = reserved > 0 && totalMine > 0 ? (reserved * mine) / totalMine : 0
        const color = over ? "#f59e0b" : visual === "loading" ? VISUAL.loading.color : "#2a78d6"
        return (
          <div key={idx} className="grid grid-cols-[52px_1fr_auto] items-center gap-2 text-[11px]">
            <span className="text-muted-foreground">GPU {idx}</span>
            <div
              className="relative h-2.5 overflow-hidden rounded-full bg-muted"
              title={
                `${model.displayName}: ${gb(mine)} GB${measured != null ? " (medido)" : " (estimado: reparto a partes iguales)"} · ` +
                `otros ${gb(others)} GB · ${gb(gpu.totalVramMb)} GB en total` +
                (reservedPerGpu > 0 ? ` · parte de la reserva del scheduler en esta GPU: ${gb(reservedPerGpu)} GB` : "") +
                (over ? " — usa más de lo reservado: riesgo de OOM" : "")
              }
            >
              <span
                className={`absolute inset-y-0 left-0 ${visual === "loading" ? "animate-flow" : ""}`}
                style={{
                  width: `${pct(mine)}%`,
                  background:
                    visual === "loading" ? `repeating-linear-gradient(90deg, ${color} 0 8px, ${color}aa 8px 12px)` : color,
                }}
              />
              <span className="absolute inset-y-0 bg-slate-500/50" style={{ left: `${pct(mine)}%`, width: `${pct(others)}%` }} />
              {reservedPerGpu > 0 && (
                <span className="absolute inset-y-0 w-0.5 bg-foreground/80" style={{ left: `calc(${pct(reservedPerGpu)}% - 1px)` }} />
              )}
            </div>
            <span className={`tabular-nums ${over ? "font-medium text-amber-400" : "text-muted-foreground"}`}>
              {gb(mine)} / {gb(gpu.totalVramMb)} GB
            </span>
          </div>
        )
      })}
    </div>
  )
}

/** Progreso de carga frente a lo que suele tardar este modelo. */
function LoadProgress({ phase }: { phase?: ModelPhaseInfo }) {
  if (!phase || phase.phase !== "loading" || phase.loadingForMs == null) return null
  const typical = phase.typicalLoadMs
  const pct = typical ? Math.min(100, (phase.loadingForMs / typical) * 100) : null
  const slow = typical != null && phase.loadingForMs > typical * 1.5
  return (
    <div className="space-y-0.5" title={typical ? `Suele tardar ~${fmtDuration(typical)} en cargar` : "Sin historial de cargas"}>
      <div className="relative h-1.5 overflow-hidden rounded-full bg-muted">
        <span
          className="absolute inset-y-0 left-0 animate-flow"
          style={{
            width: `${pct ?? 100}%`,
            background: `repeating-linear-gradient(90deg, ${slow ? "#f59e0b" : VISUAL.loading.color} 0 8px, transparent 8px 12px)`,
          }}
        />
      </div>
      {slow && <p className="text-[10.5px] text-amber-300">Tarda más de lo habitual en cargar</p>}
    </div>
  )
}

export function ModelLiveCard({
  model,
  act,
  ctx,
  phase,
  gpus = [],
  vramByGpu,
  nowMs,
  stuckThreshold,
  idleTimeoutSeconds,
  onUnload,
  canCancel,
}: {
  model: ModelState
  act?: ModelActivity
  ctx?: ModelContextInfo
  phase?: ModelPhaseInfo
  gpus?: GPUState[]
  vramByGpu?: Record<string, number>
  nowMs: number
  stuckThreshold: number
  idleTimeoutSeconds?: number
  onUnload: (modelId: string) => void
  canCancel: (req: ActiveRequestProgress) => boolean
}) {
  // Slots del preset ACTIVO (lo calcula el backend desde el worker derivado
  // cargado) antes que el parallelSlots configurado del modelo base.
  const slots = act?.slots ?? getLlamaCppConfig(model)?.parallelSlots ?? getVllmConfig(model)?.maxNumSeqs ?? null
  const whisperWorkers = slots == null ? getWhisperConfig(model)?.numWorkers ?? null : null
  // El contexto CARGADO (el de la barra) antes que el configurado en el modelo
  // base: con presets pueden diferir (nemotron: 512k base, 1024k cargado).
  const configuredCtx = ctx?.activeMaxTokens ?? act?.effectiveContext ?? getModelContextSummary(model).configuredContext
  const adm = act?.admission
  const inFlight = act?.inFlight ?? 0
  const queued = act?.queuedCount ?? adm?.queuedRequests ?? 0
  const running = Math.max(0, inFlight - queued)
  const state = cardState(model, act, phase, stuckThreshold)
  const v = VISUAL[state.visual]
  const presetKey = ctx?.activePresetKey ?? act?.runtimePreset

  let idleLeft: number | null = null
  if (idleTimeoutSeconds && model.loadPolicy === "on_demand" && model.status === "loaded" && inFlight === 0 && model.lastRequestAt) {
    const since = (nowMs - new Date(model.lastRequestAt).getTime()) / 1000
    if (Number.isFinite(since)) idleLeft = Math.max(0, Math.round(idleTimeoutSeconds - since))
  }
  const loaded = loadedMeta(model.loadedAt, nowMs)
  const order = requestColorOrder(act?.requests)
  const runningRequests = (act?.requests ?? []).filter((r) => r.stage !== "queued")
  const gpuIdx = model.currentGpu ?? []

  return (
    <article
      className={`rounded-2xl border border-l-4 bg-card p-4 shadow-sm ${state.visual === "idle" ? "border-border" : v.border}`}
      style={state.visual === "idle" ? undefined : { borderLeftColor: v.color }}
      data-model-id={model.modelId}
      data-visual={state.visual}
    >
      <header className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0 space-y-1.5">
          <div className="flex flex-wrap items-center gap-2">
            <BackendIcon backendType={model.backendType} />
            <h3 className="truncate text-[15px] font-semibold tracking-tight">{model.displayName}</h3>
            <CapabilityIcons model={model} />
            <StatePill state={state} />
          </div>
          <div className="flex flex-wrap items-center gap-1.5 text-xs text-muted-foreground">
            <LoadPolicyBadge policy={model.loadPolicy} />
            <Chip>
              <Cpu size={11} />
              {gpuIdx.length > 0 ? `GPU ${gpuIdx.join(", ")}` : model.ramUsedMb > 0 ? "CPU" : "—"}
            </Chip>
            {model.ramUsedMb > 0 && <span>{model.ramUsedMb.toLocaleString()} MB RAM</span>}
            {slots != null && (
              <Chip title={presetKey ? `Preset de runtime activo: ${presetKey}` : "Slots concurrentes y contexto cargado"}>
                {slots} slot{slots === 1 ? "" : "s"} · {fmtContext(configuredCtx)} ctx
                {presetKey ? ` · ${presetKey}` : ""}
              </Chip>
            )}
            {whisperWorkers != null && (
              <Chip title="num_workers de CTranslate2">
                {whisperWorkers} worker{whisperWorkers === 1 ? "" : "s"}
              </Chip>
            )}
            {(adm?.slotSaturated || adm?.computeSaturated) && (
              <Chip
                className="bg-amber-500/15 text-amber-300"
                title={
                  adm?.computeSaturated && !adm?.slotSaturated
                    ? `Límite de cómputo alcanzado (${adm?.admissionLimit} de ${adm?.slots} slots): un prefill grande limita la concurrencia`
                    : "Todos los slots ocupados: las siguientes peticiones esperarán en cola"
                }
              >
                {adm?.computeSaturated && !adm?.slotSaturated ? "límite de cómputo" : "slots llenos"}
              </Chip>
            )}
            {idleLeft != null && (
              <Chip className="tabular-nums" title="Se descargará por inactividad si no recibe peticiones">
                descarga en {idleLeft >= 60 ? `${Math.floor(idleLeft / 60)}m${idleLeft % 60}s` : `${idleLeft}s`}
              </Chip>
            )}
            {loaded && (
              <Chip>
                Cargado {loaded.at} · {loaded.ago}
              </Chip>
            )}
          </div>
        </div>
        {model.status === "loaded" && (
          <button
            type="button"
            onClick={() => onUnload(model.modelId)}
            className="inline-flex items-center gap-1.5 rounded-md border border-red-500/30 px-3 py-1 text-xs font-medium text-red-300 transition-colors hover:border-red-500/50 hover:bg-red-500/15"
          >
            <X size={13} />
            Descargar
          </button>
        )}
      </header>

      <div className="mt-3 space-y-2">
        <LoadProgress phase={phase} />
        <ModelMemory model={model} gpus={gpus} vramByGpu={vramByGpu} visual={state.visual} />
        {ctx && (
          <div className="space-y-1">
            <p className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">Contexto</p>
            <ContextBar ctx={ctx} requestOrder={order} />
          </div>
        )}
      </div>

      {inFlight > 0 && (
        <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Total
            label="En curso"
            value={slots != null ? `${running}/${slots}` : running}
            title={slots != null ? `${running} admitidas de ${slots} slots` : "Backend sin pool de slots gestionado"}
          />
          <Total label="En cola" value={queued} />
          <Total label="Prefill" value={fmtTps(act?.prefillTokensPerSecondTotal)} title="Suma de las peticiones en prefill, medida en su propia fase" />
          <Total label="Generación" value={fmtTps(act?.generationTokensPerSecondTotal)} title="Suma de las peticiones generando, medida desde su primer token" />
        </div>
      )}

      {runningRequests.length > 0 && (
        <div className="mt-3 space-y-1.5">
          {runningRequests.map((r) => (
            <RequestProgress
              key={r.requestId}
              req={r}
              hue={requestHue(order.get(r.requestId) ?? 0)}
              stuckThreshold={stuckThreshold}
              canCancel={canCancel(r)}
            />
          ))}
        </div>
      )}
    </article>
  )
}
