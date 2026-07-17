import { useId } from "react"

interface AreaSparkProps {
  data: number[]
  color: string
  height?: number
}

// Filled-area sparkline for a quantity accumulating over time (power draw).
// The area fill reads as "amount consumed"; the endpoint dot marks "now".
export function AreaSpark({ data, color, height = 56 }: AreaSparkProps) {
  const id = useId().replace(/:/g, "")
  if (data.length < 2) return <div style={{ height }} />
  const w = 300, h = height, pad = 2
  const max = Math.max(...data) * 1.12
  const min = Math.min(...data) * 0.9
  const range = max - min || 1
  const x = (i: number) => pad + (i * (w - 2 * pad)) / (data.length - 1)
  const y = (v: number) => h - pad - ((v - min) / range) * (h - 2 * pad)
  const line = data.map((v, i) => `${i ? "L" : "M"} ${x(i).toFixed(1)} ${y(v).toFixed(1)}`).join(" ")
  const lx = x(data.length - 1)
  const ly = y(data[data.length - 1])
  return (
    <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" className="block h-full w-full overflow-visible" style={{ height }} aria-hidden="true">
      <defs>
        <linearGradient id={id} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor={color} stopOpacity={0.34} />
          <stop offset="1" stopColor={color} stopOpacity={0} />
        </linearGradient>
      </defs>
      <path d={`${line} L ${lx} ${h} L ${x(0)} ${h} Z`} fill={`url(#${id})`} />
      <path d={line} fill="none" stroke={color} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" vectorEffect="non-scaling-stroke" />
      <circle cx={lx} cy={ly} r={3.2} fill={color} stroke="hsl(var(--card))" strokeWidth={1.5} />
    </svg>
  )
}
