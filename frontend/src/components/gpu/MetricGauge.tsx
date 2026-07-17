interface MetricGaugeProps {
  /** 0–100 fill of the arc */
  pct: number
  /** stroke color of the filled arc (CSS color) */
  color: string
  /** accessible label */
  label: string
  /** big centered value + small unit */
  value: string
  unit?: string
  caption?: string
  size?: "lg" | "sm"
}

// A ~270° radial gauge. Used for "how busy / how full right now" magnitudes —
// GPU/CPU utilization and instantaneous power draw vs limit.
export function MetricGauge({ pct, color, label, value, unit, caption, size = "lg" }: MetricGaugeProps) {
  const clamped = Math.max(0, Math.min(100, pct))
  const R = 52, C = 60, START = 135, SWEEP = 270
  const rad = (d: number) => ((d - 90) * Math.PI) / 180
  const pt = (d: number): [number, number] => [C + R * Math.cos(rad(d)), C + R * Math.sin(rad(d))]
  const arc = (a0: number, a1: number) => {
    const [ax, ay] = pt(a0)
    const [bx, by] = pt(a1)
    const large = (a1 - a0) % 360 > 180 ? 1 : 0
    return `M ${ax} ${ay} A ${R} ${R} 0 ${large} 1 ${bx} ${by}`
  }
  const px = size === "lg" ? 118 : 82
  return (
    <div className="relative shrink-0" style={{ width: px, height: px }}>
      <svg viewBox="0 0 120 120" className="block h-full w-full overflow-visible" role="img" aria-label={`${label} ${Math.round(clamped)}%`}>
        <path d={arc(START, START + SWEEP)} fill="none" stroke="hsl(var(--muted))" strokeWidth={10} strokeLinecap="round" />
        <path d={arc(START, START + (SWEEP * clamped) / 100)} fill="none" stroke={color} strokeWidth={10} strokeLinecap="round" />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <div className={`font-semibold leading-none tabular-nums tracking-tight ${size === "lg" ? "text-[26px]" : "text-[15px]"}`}>
          {value}
          {unit && <span className={`font-semibold text-muted-foreground ${size === "lg" ? "text-sm" : "text-[10px]"}`}> {unit}</span>}
        </div>
        {caption && (
          <div className={`mt-1 font-semibold uppercase tracking-wider text-muted-foreground ${size === "lg" ? "text-[10px]" : "text-[8.5px]"}`}>
            {caption}
          </div>
        )}
      </div>
    </div>
  )
}
