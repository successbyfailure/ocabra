import type { ModelState, ModelActivity, OllamaRuntimeInfo } from "@/types"
import { BackendBadge } from "@/components/models/BackendBadge"

const gb = (mb: number) => (mb / 1024).toFixed(1)
// Spanish labels for the phases the backend reports (ModelManager.get_load_phase).
const PHASE_LABEL: Record<string, string> = {
  generating: "generando",
  prompt_eval: "prompt",
  loading: "cargando",
  waiting_for_gpu: "esperando GPU",
}
const fmtAge = (s: number) => (s >= 60 ? `${Math.floor(s / 60)}m${Math.round(s % 60)}s` : `${Math.round(s)}s`)
// A request stalls much faster than the overall "stuck" threshold (5min by
// default) would catch — no new output tokens for 15s is already a strong
// hung signal while total elapsed time is still well within "normal".
const STALL_SECONDS = 15

function ollamaName(modelId: string): string {
  return modelId.startsWith("ollama/") ? modelId.slice("ollama/".length) : modelId
}

// Compact list of loaded models for a GPU / host card. Shows a live "processing"
// chip (with a stuck warning past the busy timeout) and, for Ollama models, the
// GPU/CPU placement split that oCabra can't record as an index.
export function LoadedModelList({
  models,
  emptyLabel,
  activity = {},
  stuckThreshold = 300,
  ollamaRuntime = {},
}: {
  models: ModelState[]
  emptyLabel?: string
  activity?: Record<string, ModelActivity>
  stuckThreshold?: number
  ollamaRuntime?: Record<string, OllamaRuntimeInfo>
}) {
  if (models.length === 0) {
    return emptyLabel ? <p className="text-[11.5px] text-muted-foreground">{emptyLabel}</p> : null
  }
  return (
    <div className="flex flex-col gap-1.5">
      {models.map((m) => {
        const act = activity[m.modelId]
        const inFlight = act?.inFlight ?? 0
        const stuck = inFlight > 0 && (act?.oldestSeconds ?? 0) >= stuckThreshold
        const rt = m.backendType === "ollama" ? ollamaRuntime[ollamaName(m.modelId)] : undefined
        // Deliberately lighter than the "Modelos activos" card: this one is
        // nested inside a GPU card and is about what is happening *right now*
        // on it. Configuration-flavoured data (KV pool bar, runtime preset,
        // VRAM reservation, idle countdown) lives only in the big card, which
        // has full-width rows. Seven fields duplicated across both was the
        // actual overload.
        const adm = act?.admission
        const queued = adm?.queuedRequests ?? 0
        const saturated = Boolean(adm?.slotSaturated || adm?.computeSaturated)
        const tpsTotal = act?.tokensPerSecondTotal ?? null
        const genCount = act?.generatingCount ?? 0
        const phaseMix = Object.entries(act?.phases ?? {}).filter(([, n]) => n > 0)
        return (
          <div key={m.modelId} className="rounded-lg bg-muted/50 px-2.5 py-1.5 text-[11.5px]">
            <div className="flex items-center justify-between gap-2">
              <span className="flex min-w-0 items-center gap-1.5">
                <BackendBadge backendType={m.backendType} />
                <span className="min-w-0 truncate font-medium text-foreground/90">{m.displayName || m.modelId}</span>
              </span>
              <span className="flex shrink-0 items-center gap-2">
                {inFlight > 0 &&
                  (stuck ? (
                    <span className="inline-flex items-center gap-1 rounded-full bg-red-500/15 px-1.5 py-0.5 text-[10.5px] font-medium text-red-500">
                      <span className="h-1.5 w-1.5 rounded-full bg-red-500" />
                      atascada {fmtAge(act?.oldestSeconds ?? 0)}
                    </span>
                  ) : (
                    <span className="inline-flex items-center gap-1 rounded-full bg-emerald-500/15 px-1.5 py-0.5 text-[10.5px] font-medium text-emerald-500">
                      <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-500" />
                      {inFlight} en curso
                    </span>
                  ))}
                {queued > 0 && (
                  <span
                    className="inline-flex items-center rounded-full bg-amber-500/15 px-1.5 py-0.5 text-[10.5px] font-medium text-amber-500"
                    title="Peticiones esperando un slot libre en este worker"
                  >
                    {queued} en cola
                  </span>
                )}
                {saturated && queued === 0 && (
                  <span
                    className="inline-flex items-center rounded-full bg-amber-500/15 px-1.5 py-0.5 text-[10.5px] font-medium text-amber-500"
                    title={
                      adm?.computeSaturated && !adm?.slotSaturated
                        ? `Límite de cómputo alcanzado (${adm?.admissionLimit} de ${adm?.slots} slots): las siguientes peticiones esperarán`
                        : "Todos los slots ocupados: las siguientes peticiones esperarán"
                    }
                  >
                    al límite
                  </span>
                )}
                {tpsTotal != null && tpsTotal > 0 && (
                  <span
                    className="tabular-nums text-muted-foreground"
                    title={`Suma de ${genCount} petici${genCount === 1 ? "ón" : "ones"} generando. Las que están procesando prompt no aportan t/s.`}
                  >
                    {tpsTotal} t/s{genCount > 1 ? `×${genCount}` : ""}
                  </span>
                )}
                {m.vramUsedMb > 0 && (
                  <span className="tabular-nums text-muted-foreground">{gb(m.vramUsedMb)} GB</span>
                )}
                {m.ramUsedMb > 0 && (
                  <span className="tabular-nums text-muted-foreground">{gb(m.ramUsedMb)} GB RAM</span>
                )}
              </span>
            </div>
            {phaseMix.length > 0 && (
              <div className="mt-1 flex items-center gap-2 text-[10px] text-muted-foreground">
                {phaseMix.length > 0 && (
                  <span className="truncate">
                    {phaseMix
                      .map(([phase, n]) => `${n} ${PHASE_LABEL[phase] ?? phase}`)
                      .join(" · ")}
                    {adm?.slots ? ` de ${adm.slots} slots` : ""}
                  </span>
                )}
              </div>
            )}
            {rt && (
              <div className="mt-1 flex items-center gap-2 text-[10px] text-muted-foreground">
                <span className="flex-1 overflow-hidden rounded-full bg-muted">
                  <span className="flex h-1.5">
                    <span style={{ width: `${rt.gpuPct}%`, background: "#16a34a" }} className="h-full" />
                    {rt.cpuPct > 0 && <span style={{ width: `${rt.cpuPct}%`, background: "#eda100" }} className="h-full" />}
                  </span>
                </span>
                <span className="shrink-0 tabular-nums">
                  {rt.gpuPct}% GPU{rt.cpuPct > 0 ? ` · ${rt.cpuPct}% CPU` : ""}
                  {rt.contextLength ? ` · ctx ${rt.contextLength >= 1000 ? `${Math.round(rt.contextLength / 1024)}k` : rt.contextLength}` : ""}
                </span>
              </div>
            )}
            {act?.requests?.map((r) => {
              // Phase comes straight from the backend (ModelManager.get_load_phase)
              // instead of being inferred from outputTokensEstimate — a huge
              // prompt (tool definitions the input estimate doesn't count —
              // see _count_input_tokens) can legitimately sit in prompt-eval
              // for a long time with no output yet, and *that* phase is
              // exactly where "hung or just slow?" is hardest to tell from
              // elapsed time alone. Reported live 2026-10-29: "esperando
              // salida" showed no timing at all and didn't distinguish a
              // cold load, an eviction wait, or a huge prompt — all three
              // looked identical.
              const stalled =
                (r.phase === "prompt_eval" || r.phase === "generating") &&
                r.lastActivitySecondsAgo >= STALL_SECONDS
              const neverActive = r.lastActivitySecondsAgo >= r.elapsedSeconds - 0.05
              let statusLabel: string
              if (r.phase === "loading") {
                statusLabel = "cargando modelo…"
              } else if (r.phase === "waiting_for_gpu") {
                statusLabel = r.waitingOnModelId
                  ? `esperando a que libere GPU ${r.waitingOnModelId}`
                  : "esperando a que se libere la GPU"
              } else if (r.phase === "prompt_eval") {
                statusLabel = "procesando prompt…"
              } else {
                statusLabel = r.outputTokensEstimate != null ? `${r.outputTokensEstimate} tok` : "generando…"
              }
              const showStallSuffix = r.phase === "prompt_eval" || r.phase === "generating"
              return (
                <div
                  key={r.requestId}
                  className={`mt-1 flex items-center gap-1.5 truncate text-[10px] ${stalled ? "text-red-500" : "text-muted-foreground"}`}
                >
                  <span className={`h-1 w-1 shrink-0 rounded-full ${stalled ? "bg-red-500" : "bg-emerald-500 animate-pulse"}`} />
                  <span className="truncate tabular-nums">
                    {r.username ? `${r.username} · ` : ""}
                    {r.inputTokensEstimate != null ? `~${r.inputTokensEstimate} tok entrada · ` : ""}
                    {statusLabel}
                    {r.phase === "generating" && r.tokensPerSecond != null ? ` · ${r.tokensPerSecond} t/s` : ""}
                    {showStallSuffix &&
                      (stalled
                        ? ` · sin actividad hace ${fmtAge(r.lastActivitySecondsAgo)}`
                        : neverActive
                          ? ` · sin respuesta del worker aún`
                          : ` · actividad hace ${fmtAge(r.lastActivitySecondsAgo)}`)}
                  </span>
                </div>
              )
            })}
          </div>
        )
      })}
    </div>
  )
}
