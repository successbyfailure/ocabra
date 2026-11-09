import { Cpu, X } from "lucide-react"
import { CapabilityIcons } from "@/components/models/CapabilityIcons"
import { LoadPolicyBadge } from "@/components/models/LoadPolicyBadge"
import { ModelStatusBadge } from "@/components/models/ModelStatusBadge"
import {
  getLlamaCppConfig,
  getModelContextSummary,
  getVllmConfig,
  getWhisperConfig,
} from "@/lib/modelContext"
import type { ActiveRequestProgress, ModelActivity, ModelContextInfo, ModelState } from "@/types"
import { BackendIcon } from "./BackendIcon"
import { ContextBar, requestHue } from "./ContextBar"
import { fmtContext, fmtDuration, fmtTps, loadedMeta } from "./format"
import { RequestProgress } from "./RequestProgress"

/** Índice de color estable por petición: orden de llegada dentro del modelo. */
export function requestColorOrder(requests: ActiveRequestProgress[] = []): Map<string, number> {
  const sorted = [...requests].sort((a, b) => (a.startedAt ?? 0) - (b.startedAt ?? 0))
  return new Map(sorted.map((r, i) => [r.requestId, i]))
}

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

export function ModelLiveCard({
  model,
  act,
  ctx,
  nowMs,
  stuckThreshold,
  idleTimeoutSeconds,
  onUnload,
  canCancel,
}: {
  model: ModelState
  act?: ModelActivity
  ctx?: ModelContextInfo
  nowMs: number
  stuckThreshold: number
  idleTimeoutSeconds?: number
  onUnload: (modelId: string) => void
  canCancel: (req: ActiveRequestProgress) => boolean
}) {
  // Slots del preset ACTIVO (lo calcula el backend desde el worker derivado
  // cargado) antes que el parallelSlots configurado del modelo base: con
  // presets de runtime no tienen por qué coincidir.
  const slots = act?.slots ?? getLlamaCppConfig(model)?.parallelSlots ?? getVllmConfig(model)?.maxNumSeqs ?? null
  const whisperWorkers = slots == null ? getWhisperConfig(model)?.numWorkers ?? null : null
  // El contexto CARGADO (el de la barra) antes que el configurado en el
  // modelo base: con presets pueden diferir (nemotron: 512k base, 1024k
  // cargado con el preset dual-gpu).
  const configuredCtx = ctx?.activeMaxTokens ?? act?.effectiveContext ?? getModelContextSummary(model).configuredContext
  const adm = act?.admission
  const inFlight = act?.inFlight ?? 0
  const queued = act?.queuedCount ?? adm?.queuedRequests ?? 0
  const running = Math.max(0, inFlight - queued)
  const stuck = inFlight > 0 && (act?.oldestSeconds ?? 0) > stuckThreshold

  // vramUsedMb es max(medido, reservado): solo divergen cuando lo medido
  // supera la reserva, que es el patrón previo a un CUDA OOM.
  const reservedMb = model.vramEstimateMb ?? 0
  const overReserved = reservedMb > 0 && model.vramUsedMb > reservedMb * 1.02

  // Solo los on_demand se descargan por inactividad.
  let idleLeft: number | null = null
  if (idleTimeoutSeconds && model.loadPolicy === "on_demand" && model.status === "loaded" && inFlight === 0 && model.lastRequestAt) {
    const since = (nowMs - new Date(model.lastRequestAt).getTime()) / 1000
    if (Number.isFinite(since)) idleLeft = Math.max(0, Math.round(idleTimeoutSeconds - since))
  }
  const loaded = loadedMeta(model.loadedAt, nowMs)
  const order = requestColorOrder(act?.requests)
  const runningRequests = (act?.requests ?? []).filter((r) => r.stage !== "queued")
  const gpus = model.currentGpu ?? []

  return (
    <article
      className={`rounded-2xl border bg-card p-4 shadow-sm ${stuck ? "border-red-500/60" : "border-border"}`}
      data-model-id={model.modelId}
    >
      <header className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0 space-y-1.5">
          <div className="flex items-center gap-2">
            <BackendIcon backendType={model.backendType} />
            <h3 className="truncate text-[15px] font-semibold tracking-tight">{model.displayName}</h3>
            <CapabilityIcons model={model} />
          </div>
          <div className="flex flex-wrap items-center gap-1.5 text-xs text-muted-foreground">
            <ModelStatusBadge status={model.runtimePhase ?? model.status} />
            <LoadPolicyBadge policy={model.loadPolicy} />
            <Chip>
              <Cpu size={11} />
              {gpus.length > 0 ? `GPU ${gpus.join(", ")}` : model.ramUsedMb > 0 ? "CPU" : "—"}
            </Chip>
            {model.ramUsedMb > 0 ? (
              <span>{model.ramUsedMb.toLocaleString()} MB RAM</span>
            ) : (
              <span
                className={overReserved ? "font-medium text-amber-400" : undefined}
                title={
                  reservedMb > 0
                    ? `${model.vramUsedMb.toLocaleString()} MB en uso frente a ${reservedMb.toLocaleString()} MB reservados por el scheduler` +
                      (overReserved ? " — el uso supera la reserva, riesgo de OOM" : "")
                    : "VRAM en uso"
                }
              >
                {model.vramUsedMb.toLocaleString()}
                {overReserved ? ` / ${reservedMb.toLocaleString()} reservados` : ""} MB
              </span>
            )}
            {slots != null && (
              <Chip title={act?.runtimePreset ? `Preset de runtime activo: ${act.runtimePreset}` : "Slots concurrentes y contexto configurado"}>
                {slots} slot{slots === 1 ? "" : "s"} · {fmtContext(configuredCtx)} ctx
                {act?.runtimePreset ? ` · ${act.runtimePreset}` : ""}
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
            {stuck && (
              <Chip className="bg-red-500/15 text-red-300" title={`Una petición lleva más de ${stuckThreshold} s`}>
                atascada · {fmtDuration((act?.oldestSeconds ?? 0) * 1000)}
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

      {inFlight > 0 && (
      <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Total
          label="En curso"
          value={slots != null ? `${running}/${slots}` : running}
          title={slots != null ? `${running} procesando de ${slots} slots` : "Backend sin pool de slots gestionado"}
        />
        <Total label="En cola" value={queued} />
        <Total
          label="Prefill"
          value={fmtTps(act?.prefillTokensPerSecondTotal)}
          title="Suma de las peticiones en prefill ahora, medida en su propia fase"
        />
        <Total
          label="Generación"
          value={fmtTps(act?.generationTokensPerSecondTotal)}
          title="Suma de las peticiones generando ahora, medida desde su primer token"
        />
      </div>
      )}

      {ctx && (
        <div className="mt-3">
          <ContextBar ctx={ctx} requestOrder={order} />
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
