import { MetricGauge } from "./MetricGauge"
import { AreaSpark } from "./AreaSpark"
import { METRIC } from "./metrics"

interface PowerBlockProps {
  label: string
  powerW: number
  powerLimitW?: number // when present, an instantaneous draw/limit gauge is shown
  history: number[]
  subtitle: string
}

// Electrical consumption = a quantity over time → historical area, plus an
// instantaneous gauge (draw vs limit) when a limit exists.
export function PowerBlock({ label, powerW, powerLimitW, history, subtitle }: PowerBlockProps) {
  const hasLimit = typeof powerLimitW === "number" && powerLimitW > 0
  const powPct = hasLimit ? (powerW / (powerLimitW as number)) * 100 : 0
  return (
    <div className="mt-[18px] border-t border-border pt-4">
      <div className="mb-[10px] text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="grid grid-cols-[auto_1fr] items-center gap-4">
        {hasLimit ? (
          <MetricGauge
            pct={powPct}
            color={METRIC.power}
            label="Consumo"
            value={String(Math.round(powerW))}
            unit="W"
            caption={`de ${Math.round(powerLimitW as number)}`}
            size="sm"
          />
        ) : (
          <div className="w-[82px] shrink-0 text-center">
            <div className="text-[23px] font-bold leading-none tabular-nums tracking-tight">
              {Math.round(powerW)}
              <span className="text-[13px] font-semibold text-muted-foreground"> W</span>
            </div>
            <div className="mt-1 text-[8.5px] font-semibold uppercase tracking-wider text-muted-foreground">actual</div>
          </div>
        )}
        <div className="min-w-0">
          <div className="text-[11.5px] tabular-nums text-muted-foreground">{subtitle}</div>
          <div className="mt-0.5">
            <AreaSpark data={history} color={METRIC.power} />
          </div>
        </div>
      </div>
    </div>
  )
}
