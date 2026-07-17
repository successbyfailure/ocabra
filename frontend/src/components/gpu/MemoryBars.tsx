import { METRIC, statusColor } from "./metrics"

interface MemoryBarsProps {
  name: string // "VRAM" | "RAM"
  usedLabel: string // "20.4"
  totalLabel: string // "24.0"
  unit?: string // "GB"
  usedPct: number
  secondaryName: string // "bloqueada" | "caché"
  secondaryPct: number
}

function Bar({ pct, color }: { pct: number; color: string }) {
  return (
    <div className="h-[9px] flex-1 overflow-hidden rounded-[5px] bg-muted">
      <div className="h-full rounded-[5px] transition-all" style={{ width: `${Math.max(0, Math.min(100, pct))}%`, background: color }} />
    </div>
  )
}

// Memory occupation reads as "filling a fixed container" → labeled bars, one for
// what's in use (status-colored as it fills) and a separate one for the
// locked / cached share (its own hue), each with its percentage.
export function MemoryBars({ name, usedLabel, totalLabel, unit = "GB", usedPct, secondaryName, secondaryPct }: MemoryBarsProps) {
  return (
    <div className="min-w-0">
      <div className="mb-[7px] flex items-baseline justify-between">
        <span className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">{name}</span>
        <span className="text-[13px] font-semibold tabular-nums">
          <b className="font-bold">{usedLabel}</b>
          <span className="font-medium text-muted-foreground"> / {totalLabel} {unit}</span>
        </span>
      </div>
      <div className="flex items-center gap-[9px]">
        <span className="flex flex-[0_0_64px] items-center gap-[5px] text-[10.5px] text-muted-foreground">
          <i className="inline-block h-2 w-2 rounded-sm" style={{ background: statusColor(usedPct) }} />
          en uso
        </span>
        <Bar pct={usedPct} color={statusColor(usedPct)} />
        <span className="flex-[0_0_34px] text-right text-[11px] tabular-nums text-muted-foreground">{Math.round(usedPct)}%</span>
      </div>
      <div className="mt-[7px] flex items-center gap-[9px]">
        <span className="flex flex-[0_0_64px] items-center gap-[5px] text-[10.5px] text-muted-foreground">
          <i className="inline-block h-2 w-2 rounded-sm" style={{ background: METRIC.memLocked }} />
          {secondaryName}
        </span>
        <Bar pct={secondaryPct} color={METRIC.memLocked} />
        <span className="flex-[0_0_34px] text-right text-[11px] tabular-nums text-muted-foreground">{Math.round(secondaryPct)}%</span>
      </div>
    </div>
  )
}
