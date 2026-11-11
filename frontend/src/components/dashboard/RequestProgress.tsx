import { X } from "lucide-react"
import { toast } from "sonner"
import { api } from "@/api/client"
import type { ActiveRequestProgress } from "@/types"
import { STALL_SECONDS, STAGE_META, VISUAL, audioProgressText, fmtDuration, fmtTokens, fmtTps } from "./format"
import { RequestKindBadge } from "./RequestKindBadge"

const STRIPES =
  "repeating-linear-gradient(135deg, rgba(148,163,184,0.22) 0 5px, transparent 5px 10px)"

/**
 * Una petición en curso como barra de progreso en el tiempo.
 *
 * El largo total es la duración estimada (transcurrido + lo que falta según
 * el estimador). Dentro se pintan los tramos REALES ya ocurridos: espera en
 * cola, prefill y generación, con sus límites en los instantes reales de
 * admisión y primer token. Lo que falta va rayado. Sin estimación, la barra
 * es indeterminada y solo muestra los tramos reales.
 *
 * El prefill de llama.cpp lleva además su progreso de verdad (tokens
 * procesados del prompt), en la línea fina de debajo.
 */
export function RequestProgress({
  req,
  hue,
  stuckThreshold,
  canCancel,
}: {
  req: ActiveRequestProgress
  hue: string
  stuckThreshold: number
  canCancel: boolean
}) {
  const stage = req.stage ?? "prefill"
  const meta = STAGE_META[stage]
  const elapsed = req.elapsedMs ?? req.elapsedSeconds * 1000
  const remaining = req.estimatedRemainingMs ?? null
  const total = remaining != null ? elapsed + remaining : null
  // Ya pasó de lo estimado: decirlo en vez de "~0 ms" restantes.
  const running = elapsed - (req.admittedAfterMs ?? 0)
  const overrun =
    req.expectedDurationMs != null && remaining === 0 ? Math.max(0, running - req.expectedDurationMs) : 0
  // Indeterminada: el tramo real ocupa el 80 % para que se vea que sigue.
  const scale = total != null && total > 0 ? total : elapsed / 0.8 || 1
  const at = (ms: number | null | undefined) => Math.max(0, Math.min(100, ((ms ?? 0) / scale) * 100))

  const admitted = req.admittedAfterMs ?? (stage === "queued" ? null : 0)
  const firstToken = req.firstTokenAfterMs ?? null
  // En cola o esperando dentro del backend, todo lo transcurrido es espera.
  const queueEnd = stage === "queued" || stage === "backend_wait" ? elapsed : admitted ?? 0
  const prefillEnd = firstToken ?? (stage === "prefill" || stage === "processing" ? elapsed : queueEnd)
  // Generando sin instante de primer token (backend antiguo o sin dato): la
  // generación ocupa igualmente hasta ahora, en vez de dejar la barra vacía.
  const genEnd = firstToken != null || stage === "generating" ? elapsed : prefillEnd

  // El backend sabe si hay una señal de vida que vigilar (tokens o progreso
  // real de prefill); sin ella nunca marca "parada". Con versiones antiguas
  // del backend, sin el campo, se mantiene la regla anterior.
  const stalled =
    req.stalled ??
    ((stage === "prefill" || stage === "generating") && req.lastActivitySecondsAgo >= STALL_SECONDS)
  const visual = VISUAL[stalled ? "stuck" : meta.visual]
  // Tramo que está avanzando ahora: rayas en movimiento.
  const flowing = (color: string) =>
    `repeating-linear-gradient(90deg, ${color} 0 8px, ${color}99 8px 12px)`
  const stuck = req.elapsedSeconds >= stuckThreshold
  const prompt = req.prompt
  const promptLive = prompt?.source === "live"

  const statusText = (() => {
    if (stage === "queued") {
      const pos = (req.queuePosition ?? 0) + 1
      const wait = req.queueWaitEstimateMs != null ? ` · entra en ~${fmtDuration(req.queueWaitEstimateMs)}` : ""
      return `#${pos} en cola${wait}`
    }
    if (stage === "waiting_for_gpu" && req.waitingOnModelId) return `esperando a que libere GPU ${req.waitingOnModelId}`
    if (req.audio) return audioProgressText(req.audio)
    if (stage === "backend_wait") return "admitida · esperando a que el backend empiece"
    if (stage === "processing") return "procesando · sin streaming, sin progreso del backend"
    if (stage === "prefill" && promptLive && prompt?.totalKnown === false) {
      // Sin streaming: /slots da lo procesado pero no el total del prompt.
      return `prefill · ${fmtTokens(prompt.processedTokens)} tok procesados`
    }
    if (stage === "prefill" && promptLive && prompt?.totalTokens) {
      return `prefill ${req.prefillPct?.toFixed(0) ?? 0}% · ${fmtTokens(prompt.processedTokens)}/${fmtTokens(prompt.totalTokens)}`
    }
    if (stage === "prefill" && req.prefillPct != null) {
      return `prefill ~${req.prefillPct.toFixed(0)}% · ~${fmtTokens(prompt?.totalTokens)} tok`
    }
    if (stage === "generating") {
      return `${fmtTokens(req.outputTokens)}${req.maxOutputTokens ? `/${fmtTokens(req.maxOutputTokens)}` : ""} tok`
    }
    return meta.label
  })()

  const cancel = async () => {
    if (!window.confirm("¿Cancelar esta petición en curso?")) return
    try {
      await api.stats.cancelRequest(req.requestId)
      toast.success("Petición cancelada")
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "No se pudo cancelar")
    }
  }

  return (
    <div
      className={`rounded-lg px-2.5 py-1.5 text-[11.5px] ${stuck ? "bg-red-500/10" : stalled ? "bg-amber-500/10" : "bg-muted/40"}`}
      data-stage={stage}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="flex min-w-0 items-center gap-1.5">
          <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: hue }} />
          <span className={`truncate ${req.username ? "font-medium" : "italic text-muted-foreground"}`}>
            {req.username ?? "anónimo"}
          </span>
          {req.apiKeyName && (
            <span className="truncate font-mono text-[10.5px] text-muted-foreground" title={`API key: ${req.apiKeyName}`}>
              🔑 {req.apiKeyName}
            </span>
          )}
          <RequestKindBadge kind={req.requestKind} />
          {req.profileId && (
            <span className="truncate text-[10.5px] text-muted-foreground" title="Nombre pedido por el cliente">
              {req.profileId}
            </span>
          )}
        </span>
        <span className="flex shrink-0 items-center gap-2 tabular-nums">
          <span className={`inline-flex items-center gap-1 ${visual.text}`}>
            <span className={`h-1.5 w-1.5 rounded-full ${visual.anim}`} style={{ background: visual.color }} />
            {statusText}
          </span>
          {req.prefillTokensPerSecond != null && (
            <span
              className="text-muted-foreground"
              title={
                req.prefillSource === "live"
                  ? "Prefill medido por llama.cpp"
                  : req.prefillSource === "historical"
                    ? "Ritmo de prefill medido en las peticiones recientes de este modelo"
                    : req.prefillSource === "regression"
                      ? "Ritmo de prefill según el estimador de duración"
                      : "Prefill estimado: tokens del prompt aproximados"
              }
            >
              ⇢ {fmtTps(req.prefillTokensPerSecond)}
              {req.prefillSource !== "live" ? "~" : ""}
            </span>
          )}
          {req.generationTokensPerSecond != null && (
            <span className="text-muted-foreground" title="Generación, medida desde el primer token">
              ✎ {fmtTps(req.generationTokensPerSecond)}
            </span>
          )}
          <span
            className={stuck ? "font-medium text-red-400" : "text-muted-foreground"}
            title={req.remainingIsUpperBound ? "Transcurrido · restante como mucho (hasta su max_tokens)" : "Transcurrido · restante estimado"}
          >
            {fmtDuration(elapsed)}
            {stage !== "queued" && remaining != null && remaining > 0
              ? ` · ${req.remainingIsUpperBound ? "≤" : "~"}${fmtDuration(remaining)}`
              : ""}
            {stage !== "queued" && overrun > 0 ? (
              <span className="text-amber-300"> · +{fmtDuration(overrun)} sobre lo estimado</span>
            ) : null}
          </span>
          {canCancel && (
            <button
              type="button"
              onClick={() => void cancel()}
              className="rounded p-0.5 text-red-300/80 transition-colors hover:bg-red-500/20 hover:text-red-200"
              title="Cancelar esta petición"
              aria-label="Cancelar esta petición"
            >
              <X size={12} />
            </button>
          )}
        </span>
      </div>

      <div
        className="relative mt-1 h-2 overflow-hidden rounded-full bg-muted"
        title={
          `Cola ${fmtDuration(queueEnd)} · prefill ${fmtDuration(Math.max(0, prefillEnd - queueEnd))}` +
          (firstToken != null ? ` · generación ${fmtDuration(Math.max(0, genEnd - prefillEnd))}` : "") +
          (total != null ? ` · total estimado ${fmtDuration(total)}` : " · sin estimación de duración")
        }
      >
        {total != null && (
          <span className="absolute inset-y-0" style={{ left: `${at(elapsed)}%`, right: 0, background: STRIPES }} />
        )}
        <span
          className={`absolute inset-y-0 left-0 ${stage === "queued" || stage === "backend_wait" ? "animate-flow" : ""}`}
          style={{
            width: `${at(queueEnd)}%`,
            background: stage === "queued" || stage === "backend_wait" ? flowing(VISUAL.waiting.color) : `${VISUAL.waiting.color}b3`,
          }}
        />
        <span
          className={`absolute inset-y-0 ${stage === "prefill" || stage === "processing" ? "animate-flow" : ""}`}
          style={{
            left: `${at(queueEnd)}%`,
            width: `${Math.max(0, at(prefillEnd) - at(queueEnd))}%`,
            background:
              stage === "processing"
                ? flowing(VISUAL.processing.color)
                : stage === "prefill"
                  ? flowing(VISUAL.prefill.color)
                  : `${VISUAL.prefill.color}cc`,
          }}
        />
        <span
          className={`absolute inset-y-0 ${stage === "generating" ? "animate-flow" : ""}`}
          style={{
            left: `${at(prefillEnd)}%`,
            width: `${Math.max(0, at(genEnd) - at(prefillEnd))}%`,
            background: stage === "generating" ? flowing(VISUAL.generating.color) : VISUAL.generating.color,
          }}
        />
        {/* Ahora */}
        <span className="absolute inset-y-0 w-px animate-pulse bg-foreground/80" style={{ left: `${at(elapsed)}%` }} />
      </div>

      {req.audio?.phase === "transcribing" && req.audio.pct != null && (
        <div className="mt-0.5 h-0.5 overflow-hidden rounded-full bg-muted" title="Audio transcrito (avance real)">
          <span className="block h-full bg-sky-400" style={{ width: `${req.audio.pct}%` }} />
        </div>
      )}

      {stage === "prefill" && req.prefillPct != null && (
        <div
          className="mt-0.5 h-0.5 overflow-hidden rounded-full bg-muted"
          title={
            promptLive
              ? "Progreso real del prefill (llama.cpp)"
              : `Progreso estimado del prefill al ritmo ${req.prefillSource === "historical" ? "medido" : "estimado"} del modelo; el backend no lo informa`
          }
        >
          <span
            className={`block h-full ${promptLive ? "bg-sky-400" : "bg-sky-400/50"}`}
            style={{ width: `${req.prefillPct}%` }}
          />
        </div>
      )}

      {(stalled || stuck || (prompt?.cachedTokens ?? 0) > 0) && (
        <p className="mt-0.5 text-[10px] text-muted-foreground">
          {stuck && <span className="text-red-400">atascada (más de {fmtDuration(stuckThreshold * 1000)}) · </span>}
          {stalled && <span className="text-amber-300">sin actividad del worker hace {Math.round(req.lastActivitySecondsAgo)} s · </span>}
          {(prompt?.cachedTokens ?? 0) > 0 && <span>{fmtTokens(prompt?.cachedTokens)} tokens de cache</span>}
        </p>
      )}
    </div>
  )
}
