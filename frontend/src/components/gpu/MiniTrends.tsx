import { useId } from "react"

export interface TrendMetric {
  label: string
  data: number[]
  color: string
  value: string
}

function Sparkline({ data, color, max = 100 }: { data: number[]; color: string; max?: number }) {
  const id = useId().replace(/:/g, "")
  const W = 100
  const H = 28
  if (data.length < 2) {
    return <div className="h-[28px] w-full rounded bg-muted/40" />
  }
  const peak = Math.max(max, ...data)
  const step = W / (data.length - 1)
  const pts = data.map((v, i) => [i * step, H - (Math.max(0, v) / peak) * (H - 2) - 1])
  const line = pts.map(([x, y], i) => `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`).join(" ")
  const area = `${line} L${W},${H} L0,${H} Z`
  return (
    <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="h-[28px] w-full">
      <defs>
        <linearGradient id={id} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.28" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <path d={area} fill={`url(#${id})`} />
      <path d={line} fill="none" stroke={color} strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
    </svg>
  )
}

// Compact row of labelled sparklines placed under the memory bar — more history
// at a glance without taking much vertical space.
export function MiniTrends({ metrics }: { metrics: TrendMetric[] }) {
  return (
    <div className="grid gap-2.5" style={{ gridTemplateColumns: `repeat(${metrics.length}, minmax(0, 1fr))` }}>
      {metrics.map((m) => (
        <div key={m.label} className="min-w-0">
          <div className="mb-0.5 flex items-baseline justify-between gap-1">
            <span className="truncate text-[10px] uppercase tracking-wide text-muted-foreground">{m.label}</span>
            <span className="shrink-0 text-[11px] font-semibold tabular-nums" style={{ color: m.color }}>
              {m.value}
            </span>
          </div>
          <Sparkline data={m.data} color={m.color} />
        </div>
      ))}
    </div>
  )
}
