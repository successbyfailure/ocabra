interface PowerGaugeProps {
  powerDrawW: number
  powerLimitW: number
}

function polarToCartesian(cx: number, cy: number, radius: number, angleDeg: number) {
  const angleRad = ((angleDeg - 180) * Math.PI) / 180
  return {
    x: cx + radius * Math.cos(angleRad),
    y: cy + radius * Math.sin(angleRad),
  }
}

function describeArc(cx: number, cy: number, radius: number, progressPct: number) {
  const clamped = Math.max(0, Math.min(100, progressPct))
  const endAngle = clamped * 1.8
  const start = polarToCartesian(cx, cy, radius, 0)
  const end = polarToCartesian(cx, cy, radius, endAngle)
  const largeArcFlag = clamped > 50 ? 1 : 0
  return `M ${start.x} ${start.y} A ${radius} ${radius} 0 ${largeArcFlag} 1 ${end.x} ${end.y}`
}

export function PowerGauge({ powerDrawW, powerLimitW }: PowerGaugeProps) {
  const pct = powerLimitW > 0 ? (powerDrawW / powerLimitW) * 100 : 0
  const clampedPct = Math.max(0, Math.min(100, pct))

  let color = "stroke-emerald-500"
  if (clampedPct > 80) {
    color = "stroke-red-500"
  } else if (clampedPct >= 50) {
    color = "stroke-amber-400"
  }

  return (
    <div className="flex flex-col items-center gap-1">
      <svg viewBox="0 0 120 70" className="h-16 w-28" role="img" aria-label="Power gauge">
        <path d="M 10 60 A 50 50 0 0 1 110 60" className="fill-none stroke-muted stroke-[8]" />
        <path d={describeArc(60, 60, 50, clampedPct)} className={`fill-none ${color} stroke-[8]`} />
      </svg>
      <p className="text-xs text-muted-foreground">
        {Math.round(powerDrawW)}W / {Math.round(powerLimitW)}W
      </p>
    </div>
  )
}
