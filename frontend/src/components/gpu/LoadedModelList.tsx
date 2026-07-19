import type { ModelState, ModelActivity, OllamaRuntimeInfo } from "@/types"
import { BackendBadge } from "@/components/models/BackendBadge"

const gb = (mb: number) => (mb / 1024).toFixed(1)
const fmtAge = (s: number) => (s >= 60 ? `${Math.floor(s / 60)}m${Math.round(s % 60)}s` : `${Math.round(s)}s`)

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
                {m.vramUsedMb > 0 && <span className="tabular-nums text-muted-foreground">{gb(m.vramUsedMb)} GB</span>}
              </span>
            </div>
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
          </div>
        )
      })}
    </div>
  )
}
