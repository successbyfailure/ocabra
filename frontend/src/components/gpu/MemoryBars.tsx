import { METRIC, statusColor } from "./metrics"

interface MemoryBarsProps {
  name: string // "VRAM" | "RAM"
  usedLabel: string // "20.4"
  totalLabel: string // "24.0"
  unit?: string // "GB"
  usedPct: number
  secondaryName: string // "bloqueada" | "swap"
  secondaryPct: number
  // How the secondary relates to the primary:
  //  - "component": it's a sub-region of what's in use (VRAM reserved/locked) →
  //    drawn stacked inside the fill in violet.
  //  - "aside": it's a separate memory space (host swap) → NOT stacked into the
  //    bar (that would misread as "part of RAM"); shown only as a legend stat.
  secondaryMode?: "component" | "aside"
}

const clamp = (n: number) => Math.max(0, Math.min(100, n))

// Occupation as ONE stacked bar. The occupied span reads as "in use" and fills
// with a status hue (green -> amber -> red) so how-full is legible at a glance.
export function MemoryBars({
  name,
  usedLabel,
  totalLabel,
  unit = "GB",
  usedPct,
  secondaryName,
  secondaryPct,
  secondaryMode = "component",
}: MemoryBarsProps) {
  const used = clamp(usedPct)
  const secondary = clamp(secondaryPct)
  const status = statusColor(used)

  // component mode: split the fill into reserved (violet) + rest-in-use (status),
  // plus a faded violet tail when a reservation isn't resident yet (locked > used).
  const reservedResident = secondaryMode === "component" ? Math.min(secondary, used) : 0
  const usedUnreserved = secondaryMode === "component" ? Math.max(0, used - secondary) : used
  const reservedPending = secondaryMode === "component" ? Math.max(0, secondary - used) : 0

  return (
    <div className="min-w-0">
      <div className="mb-[7px] flex items-baseline justify-between">
        <span className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">{name}</span>
        <span className="text-[13px] font-semibold tabular-nums">
          <b className="font-bold">{usedLabel}</b>
          <span className="font-medium text-muted-foreground"> / {totalLabel} {unit}</span>
        </span>
      </div>

      <div className="flex h-[11px] w-full overflow-hidden rounded-[6px] bg-muted">
        {reservedResident > 0 && (
          <div className="h-full transition-all" style={{ width: `${reservedResident}%`, background: METRIC.memLocked }} />
        )}
        {usedUnreserved > 0 && (
          <div className="h-full transition-all" style={{ width: `${usedUnreserved}%`, background: status }} />
        )}
        {reservedPending > 0 && (
          <div className="h-full transition-all" style={{ width: `${reservedPending}%`, background: METRIC.memLocked, opacity: 0.4 }} />
        )}
      </div>

      <div className="mt-[7px] flex flex-wrap items-center gap-x-4 gap-y-1 text-[10.5px] text-muted-foreground">
        <span className="flex items-center gap-[5px]">
          <i className="inline-block h-2 w-2 rounded-sm" style={{ background: status }} />
          en uso <span className="tabular-nums text-foreground/80">{Math.round(used)}%</span>
        </span>
        {secondary > 0 && (
          <span className="flex items-center gap-[5px]">
            <i className="inline-block h-2 w-2 rounded-sm" style={{ background: METRIC.memLocked }} />
            {secondaryName} <span className="tabular-nums text-foreground/80">{Math.round(secondary)}%</span>
          </span>
        )}
      </div>
    </div>
  )
}
