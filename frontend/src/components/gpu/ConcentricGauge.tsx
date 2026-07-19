interface Ring {
  value: number // 0-100
  label: string
  color: string
}

interface ConcentricGaugeProps {
  outer: Ring
  inner: Ring
  centerValue: string
  centerUnit?: string
  centerSub?: string
  size?: number
}

const clamp = (n: number) => Math.max(0, Math.min(100, n))

function Arc({ r, pct, color, width }: { r: number; pct: number; color: string; width: number }) {
  const circ = 2 * Math.PI * r
  return (
    <>
      <circle cx="50" cy="50" r={r} fill="none" stroke="currentColor" strokeWidth={width} className="text-muted-foreground/20" />
      <circle
        cx="50"
        cy="50"
        r={r}
        fill="none"
        stroke={color}
        strokeWidth={width}
        strokeLinecap="round"
        strokeDasharray={`${(clamp(pct) / 100) * circ} ${circ}`}
        transform="rotate(-90 50 50)"
        className="transition-all"
      />
    </>
  )
}

// Two concentric rings: outer = utilisation/load, inner = power. Combines what
// used to be two separate gauges into one compact dial, freeing room for trends.
export function ConcentricGauge({ outer, inner, centerValue, centerUnit, centerSub, size = 128 }: ConcentricGaugeProps) {
  return (
    <div className="flex flex-col items-center gap-2" style={{ width: size }}>
      <div className="relative" style={{ width: size, height: size }}>
        <svg viewBox="0 0 100 100" className="h-full w-full -rotate-0">
          <Arc r={43} pct={outer.value} color={outer.color} width={9} />
          <Arc r={30} pct={inner.value} color={inner.color} width={9} />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-[22px] font-bold leading-none tabular-nums text-foreground">
            {centerValue}
            {centerUnit && <span className="text-[13px] font-semibold text-muted-foreground">{centerUnit}</span>}
          </span>
          {centerSub && <span className="mt-0.5 text-[11px] tabular-nums text-muted-foreground">{centerSub}</span>}
        </div>
      </div>
      <div className="flex w-full items-center justify-center gap-3 text-[10.5px]">
        <span className="flex items-center gap-1 text-muted-foreground">
          <i className="inline-block h-2 w-2 rounded-full" style={{ background: outer.color }} />
          {outer.label} <b className="font-semibold tabular-nums text-foreground/90">{Math.round(outer.value)}%</b>
        </span>
        <span className="flex items-center gap-1 text-muted-foreground">
          <i className="inline-block h-2 w-2 rounded-full" style={{ background: inner.color }} />
          {inner.label} <b className="font-semibold tabular-nums text-foreground/90">{Math.round(inner.value)}%</b>
        </span>
      </div>
    </div>
  )
}
