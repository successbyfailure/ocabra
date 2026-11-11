import { useMemo } from "react"
import { useGpuStore } from "@/stores/gpuStore"
import type {
  ActiveRequestProgress,
  FinishedRequest,
  ModelActivity,
  ModelPhaseInfo,
  ModelState,
} from "@/types"
import { BackendIcon } from "./BackendIcon"
import { GpuBadge } from "./GpuBadge"
import { STAGE_META, VISUAL, audioProgressText, fmtClock, fmtDuration, fmtRealtime, fmtTokens, fmtTps } from "./format"
import { RequestKindBadge } from "./RequestKindBadge"

const ACTIVE_STAGES = new Set(["prefill", "generating", "processing"])

interface Row {
  req: ActiveRequestProgress
  modelId: string
  model?: ModelState
  phase?: ModelPhaseInfo
}

function Who({ username, apiKeyName }: { username?: string | null; apiKeyName?: string | null }) {
  return (
    <span className="flex min-w-0 items-center gap-1.5">
      <span className={`truncate ${username ? "font-medium" : "italic text-muted-foreground"}`}>{username ?? "anónimo"}</span>
      {apiKeyName && (
        <span className="truncate font-mono text-[10.5px] text-muted-foreground" title={`API key: ${apiKeyName}`}>
          🔑 {apiKeyName}
        </span>
      )}
    </span>
  )
}

function ModelTag({ modelId, model }: { modelId: string; model?: ModelState }) {
  const gpus = useGpuStore((s) => s.gpus)
  return (
    <span className="flex min-w-0 items-center gap-1">
      {model && <BackendIcon backendType={model.backendType} showLabel={false} />}
      <span className="truncate" title={modelId}>
        {model?.displayName ?? modelId}
      </span>
      {(model?.currentGpu ?? []).map((i) => (
        <GpuBadge key={i} index={i} gpus={gpus} />
      ))}
    </span>
  )
}

function Stat({ label, value, title }: { label: string; value: React.ReactNode; title?: string }) {
  return (
    <span className="whitespace-nowrap tabular-nums" title={title}>
      <span className="text-muted-foreground/70">{label} </span>
      {value}
    </span>
  )
}

/** Tokens de entrada: "~" mientras sean estimados; con la parte de cache. */
function InputTokens({ req }: { req: ActiveRequestProgress }) {
  const p = req.prompt
  const exact = p?.source === "live" && p.totalKnown !== false
  // Sin recuento (audio, imagen...): no se pinta "entrada ~—".
  if ((p?.totalTokens ?? req.inputTokensEstimate) == null) return null
  return (
    <Stat
      label="entrada"
      title={exact ? "Tokens del prompt según el backend" : "Estimación hasta que el backend informe del total"}
      value={
        <>
          {exact ? "" : "~"}
          {fmtTokens(p?.totalTokens ?? req.inputTokensEstimate)}
          {(p?.cachedTokens ?? 0) > 0 ? <span className="text-muted-foreground"> ({fmtTokens(p?.cachedTokens)} cache)</span> : null}
        </>
      }
    />
  )
}

function ActiveRow({ row }: { row: Row }) {
  const { req } = row
  const stage = req.stage ?? "processing"
  const visual = VISUAL[req.stalled ? "stuck" : STAGE_META[stage].visual]
  const elapsed = req.elapsedMs ?? req.elapsedSeconds * 1000
  return (
    <li className="space-y-1 rounded-lg bg-muted/40 px-2.5 py-1.5 text-[11.5px]" data-request-row="active" data-stage={stage}>
      <div className="flex items-center justify-between gap-2">
        <span className="flex min-w-0 items-center gap-2">
          <span className={`h-2 w-2 shrink-0 rounded-full ${visual.anim}`} style={{ background: visual.color }} />
          <Who username={req.username} apiKeyName={req.apiKeyName} />
          <RequestKindBadge kind={req.requestKind} />
        </span>
        <span className={`shrink-0 font-medium ${visual.text}`}>
          {req.stalled
            ? "sin actividad"
            : req.audio
              ? audioProgressText(req.audio)
              : STAGE_META[stage].label}
          {!req.audio && stage === "prefill" && req.prefillPct != null ? ` ${req.prefillPct.toFixed(0)}%` : ""}
        </span>
      </div>
      <div className="flex items-center justify-between gap-2">
        <ModelTag modelId={row.modelId} model={row.model} />
      </div>
      {req.audio?.phase === "transcribing" && req.audio.pct != null && (
        <div className="h-1 overflow-hidden rounded-full bg-muted" title="Audio transcrito">
          <span className="block h-full animate-flow" style={{ width: `${req.audio.pct}%`, background: `repeating-linear-gradient(90deg, ${VISUAL.prefill.color} 0 8px, ${VISUAL.prefill.color}99 8px 12px)` }} />
        </div>
      )}
      <div className="flex flex-wrap items-center gap-x-3 gap-y-0.5 text-[11px]">
        <InputTokens req={req} />
        {req.outputTokens != null && (
          <Stat
            label="salida"
            value={`${fmtTokens(req.outputTokens)}${req.maxOutputTokens ? `/${fmtTokens(req.maxOutputTokens)}` : ""}`}
          />
        )}
        {req.prefillTokensPerSecond != null && (
          <Stat
            label="prefill"
            value={`${fmtTps(req.prefillTokensPerSecond)}${req.prefillSource !== "live" ? "~" : ""}`}
            title={req.prefillSource === "live" ? "Medido en esta petición" : "Ritmo estimado del modelo"}
          />
        )}
        {req.generationTokensPerSecond != null && <Stat label="generación" value={fmtTps(req.generationTokensPerSecond)} />}
        {req.firstTokenAfterMs != null && (
          <Stat label="TTFT" value={fmtDuration(req.firstTokenAfterMs)} title="Tiempo hasta el primer token" />
        )}
        {req.kvReservedTokens != null && <Stat label="KV" value={fmtTokens(req.kvReservedTokens)} title="Contexto reservado en el pool" />}
        <Stat
          label="lleva"
          value={
            <>
              {fmtDuration(elapsed)}
              {req.estimatedRemainingMs != null && req.estimatedRemainingMs > 0
                ? ` · ${req.remainingIsUpperBound ? "≤" : "~"}${fmtDuration(req.estimatedRemainingMs)}`
                : ""}
            </>
          }
        />
      </div>
    </li>
  )
}

function waitingReason(row: Row): { text: string; cls: string } {
  const { req, phase } = row
  switch (req.stage) {
    case "queued": {
      const turn = req.queuePosition ?? 0
      const start = req.predictedStartOrder
      const note =
        start == null
          ? req.queueWaitEstimateMs == null && req.kvReservedTokens
            ? " · no cabe en el KV libre"
            : ""
          : start < turn
            ? " · se adelanta: cabe en el KV libre"
            : start > turn
              ? " · espera KV: la adelantan otras"
              : ""
      return { text: `en cola #${turn + 1}${note}`, cls: VISUAL.waiting.text }
    }
    case "backend_wait":
      return { text: "admitida · esperando a que el backend empiece", cls: VISUAL.waiting.text }
    case "loading":
      return {
        text: `cargando el modelo${phase?.loadingForMs != null ? ` · ${fmtDuration(phase.loadingForMs)}${phase.typicalLoadMs ? ` de ~${fmtDuration(phase.typicalLoadMs)}` : ""}` : ""}`,
        cls: VISUAL.loading.text,
      }
    case "waiting_for_gpu":
      return {
        text: `esperando espacio en GPU${req.waitingOnModelId ?? phase?.waitingOn ? ` · desalojando ${req.waitingOnModelId ?? phase?.waitingOn}` : ""}`,
        cls: VISUAL.waiting_space.text,
      }
    default:
      return { text: STAGE_META[req.stage ?? "queued"]?.label ?? "esperando", cls: VISUAL.waiting.text }
  }
}

function WaitingRow({ row }: { row: Row }) {
  const { req } = row
  const reason = waitingReason(row)
  const visual = VISUAL[STAGE_META[req.stage ?? "queued"].visual]
  const waited = req.queuedMs ?? req.elapsedMs ?? req.elapsedSeconds * 1000
  return (
    <li className="space-y-1 rounded-lg bg-amber-500/5 px-2.5 py-1.5 text-[11.5px]" data-request-row="waiting" data-stage={req.stage}>
      <div className="flex items-center justify-between gap-2">
        <span className="flex min-w-0 items-center gap-2">
          <span className={`h-2 w-2 shrink-0 rounded-full ${visual.anim}`} style={{ background: visual.color }} />
          <Who username={req.username} apiKeyName={req.apiKeyName} />
          <RequestKindBadge kind={req.requestKind} />
        </span>
        <span className={`truncate text-right font-medium ${reason.cls}`}>{reason.text}</span>
      </div>
      <ModelTag modelId={row.modelId} model={row.model} />
      <div className="flex flex-wrap items-center gap-x-3 gap-y-0.5 text-[11px]">
        <InputTokens req={req} />
        {req.kvReservedTokens != null && <Stat label="reserva" value={fmtTokens(req.kvReservedTokens)} />}
        <Stat label="espera" value={fmtDuration(waited)} />
        {req.stage === "queued" && (
          <Stat
            label="entra"
            value={
              req.queueWaitEstimateMs == null ? "?" : req.queueWaitEstimateMs < 1000 ? "ya" : `en ~${fmtDuration(req.queueWaitEstimateMs)}`
            }
          />
        )}
        {req.expectedDurationMs != null && <Stat label="dura" value={`~${fmtDuration(req.expectedDurationMs)}`} />}
      </div>
    </li>
  )
}

function FinishedRow({ item, models }: { item: FinishedRequest; models: Record<string, ModelState> }) {
  const ok = item.statusCode != null && item.statusCode < 400 && !item.error
  const rejected = item.statusCode === 429
  const cls = ok
    ? "border-emerald-500/40 bg-emerald-500/15 text-emerald-200"
    : rejected
      ? "border-amber-500/40 bg-amber-500/15 text-amber-200"
      : "border-red-500/40 bg-red-500/15 text-red-200"
  // El feed da el nombre lógico ("gemma4:12b"); el modelo, el id con backend.
  // Mismo nombre puede existir en varios backends (gemma4:26b en llama.cpp y
  // en Ollama): se casa también por el backend de la petición.
  const candidates = Object.values(models).filter(
    (m) => m.modelId === item.model || m.modelId.endsWith(`/${item.model}`),
  )
  const model = candidates.find((m) => m.backendType === item.backendType) ?? candidates[0]
  // Se desvanece en sus 20 s de vida.
  const opacity = Math.max(0.45, 1 - (item.endedAgoMs ?? 0) / 20_000)
  return (
    <li className="flex flex-wrap items-center gap-x-3 gap-y-0.5 rounded-lg px-2.5 py-1 text-[11px]" style={{ opacity }} data-request-row="finished">
      <span className={`rounded-full border px-1.5 py-0 font-medium ${cls}`} title={item.error ?? undefined}>
        {rejected ? "429 rechazada" : item.statusCode ?? "error"}
      </span>
      <Who username={item.username} apiKeyName={item.apiKeyName} />
      <RequestKindBadge kind={item.requestKind} />
      <span className="truncate text-muted-foreground">{model?.displayName ?? item.model}</span>
      {item.inputTokens != null && (
        <Stat
          label="entrada"
          value={
            <>
              {fmtTokens(item.inputTokens)}
              {(item.cachedTokens ?? 0) > 0 ? <span className="text-muted-foreground"> ({fmtTokens(item.cachedTokens)} cache)</span> : null}
            </>
          }
        />
      )}
      {item.outputTokens != null && item.outputTokens > 0 && <Stat label="salida" value={fmtTokens(item.outputTokens)} />}
      {item.promptTokensPerSecond != null && <Stat label="prefill" value={fmtTps(item.promptTokensPerSecond)} />}
      {item.predictedTokensPerSecond != null && <Stat label="generación" value={fmtTps(item.predictedTokensPerSecond)} />}
      {item.ttftMs != null && <Stat label="TTFT" value={fmtDuration(item.ttftMs)} />}
      {item.audioSeconds ? (
        <Stat
          label="audio"
          value={`${fmtClock(item.audioSeconds)} · ${fmtRealtime(item.audioSeconds / Math.max(0.001, item.durationMs / 1000))} tiempo real`}
          title="Duración del audio y velocidad respecto al tiempo real"
        />
      ) : null}
      {item.queueMs ? <Stat label="cola" value={fmtDuration(item.queueMs)} /> : null}
      <Stat label="duró" value={fmtDuration(item.durationMs)} />
      <span className="ml-auto text-muted-foreground">hace {Math.round((item.endedAgoMs ?? 0) / 1000)} s</span>
      {item.error && !rejected && <span className="w-full truncate font-mono text-[10.5px] text-red-300/90">{item.error}</span>}
    </li>
  )
}

/**
 * Peticiones de todos los modelos en el momento en que entran: activas (en
 * una GPU ahora mismo) y esperando (cola de admisión, esperando al backend,
 * a que cargue el modelo o a que haya espacio en GPU), con usuario/llave,
 * tokens y velocidad. Debajo, las terminadas en los últimos 20 s: las muy
 * cortas no llegarían a verse entre dos refrescos.
 */
export function RequestsSection({
  activity,
  modelStates = {},
  models,
  finished = [],
}: {
  activity: Record<string, ModelActivity>
  modelStates?: Record<string, ModelPhaseInfo>
  models: Record<string, ModelState>
  finished?: FinishedRequest[]
}) {
  const { active, waiting } = useMemo(() => {
    const rows: Row[] = Object.entries(activity).flatMap(([modelId, act]) =>
      (act.requests ?? []).map((req) => ({ req, modelId, model: models[modelId], phase: modelStates[modelId] })),
    )
    const activeRows = rows
      .filter((r) => ACTIVE_STAGES.has(r.req.stage ?? ""))
      .sort((a, b) => (a.req.startedAt ?? 0) - (b.req.startedAt ?? 0))
    const order = (r: Row) =>
      r.req.stage === "queued" ? 2 : r.req.stage === "backend_wait" ? 1 : 0 // cargando/espacio primero
    const waitingRows = rows
      .filter((r) => !ACTIVE_STAGES.has(r.req.stage ?? ""))
      .sort(
        (a, b) =>
          order(a) - order(b) ||
          (a.req.predictedStartOrder ?? 1e9) - (b.req.predictedStartOrder ?? 1e9) ||
          (a.req.queuePosition ?? 0) - (b.req.queuePosition ?? 0) ||
          (a.req.startedAt ?? 0) - (b.req.startedAt ?? 0),
      )
    return { active: activeRows, waiting: waitingRows }
  }, [activity, models, modelStates])

  return (
    <div className="space-y-3">
      <div className="grid gap-3 lg:grid-cols-2">
        <div className="rounded-2xl border border-border bg-card p-3 shadow-sm" data-testid="requests-active">
          <h3 className="mb-2 flex items-center gap-2 text-sm font-semibold">
            <span className="h-2 w-2 animate-pulse rounded-full" style={{ background: VISUAL.generating.color }} />
            Activas <span className="text-xs font-normal text-muted-foreground">{active.length}</span>
          </h3>
          {active.length === 0 ? (
            <p className="text-xs text-muted-foreground">Ninguna petición en una GPU ahora mismo.</p>
          ) : (
            <ul className="space-y-1.5">
              {active.map((row) => (
                <ActiveRow key={row.req.requestId} row={row} />
              ))}
            </ul>
          )}
        </div>
        <div className="rounded-2xl border border-border bg-card p-3 shadow-sm" data-testid="requests-waiting">
          <h3 className="mb-2 flex items-center gap-2 text-sm font-semibold">
            <span className="h-2 w-2 animate-blink-slow rounded-full" style={{ background: VISUAL.waiting.color }} />
            Esperando <span className="text-xs font-normal text-muted-foreground">{waiting.length}</span>
          </h3>
          {waiting.length === 0 ? (
            <p className="text-xs text-muted-foreground">Nada esperando: todo lo que llega entra directamente.</p>
          ) : (
            <ul className="space-y-1.5">
              {waiting.map((row) => (
                <WaitingRow key={row.req.requestId} row={row} />
              ))}
            </ul>
          )}
        </div>
      </div>
      {finished.length > 0 && (
        <div className="rounded-2xl border border-border bg-card/60 p-2" data-testid="requests-finished">
          <p className="mb-1 px-1 text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
            Recién terminadas · últimos 20 s
          </p>
          <ul className="space-y-0.5">
            {finished.map((item, i) => (
              <FinishedRow key={`${item.requestId ?? "x"}-${item.endedAt}-${i}`} item={item} models={models} />
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
