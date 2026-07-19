import type { ModelState } from "@/types"
import { BackendBadge } from "@/components/models/BackendBadge"

const gb = (mb: number) => (mb / 1024).toFixed(1)

// Compact list of loaded models for a GPU / host card. `activity` maps modelId →
// in-flight request count so an active model shows a pulsing "procesando" chip.
export function LoadedModelList({
  models,
  emptyLabel,
  activity = {},
}: {
  models: ModelState[]
  emptyLabel?: string
  activity?: Record<string, number>
}) {
  if (models.length === 0) {
    return emptyLabel ? <p className="text-[11.5px] text-muted-foreground">{emptyLabel}</p> : null
  }
  return (
    <div className="flex flex-col gap-1.5">
      {models.map((m) => {
        const inFlight = activity[m.modelId] ?? 0
        return (
          <div
            key={m.modelId}
            className="flex items-center justify-between gap-2 rounded-lg bg-muted/50 px-2.5 py-1.5 text-[11.5px]"
          >
            <span className="flex min-w-0 items-center gap-1.5">
              <BackendBadge backendType={m.backendType} />
              <span className="min-w-0 truncate font-medium text-foreground/90">{m.displayName || m.modelId}</span>
            </span>
            <span className="flex shrink-0 items-center gap-2">
              {inFlight > 0 && (
                <span className="inline-flex items-center gap-1 rounded-full bg-emerald-500/15 px-1.5 py-0.5 text-[10.5px] font-medium text-emerald-500">
                  <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-500" />
                  {inFlight} en curso
                </span>
              )}
              {m.vramUsedMb > 0 && (
                <span className="tabular-nums text-muted-foreground">{gb(m.vramUsedMb)} GB</span>
              )}
            </span>
          </div>
        )
      })}
    </div>
  )
}
