interface VramBarProps {
  used: number
  total: number
  locked: number
}

const percentage = (value: number, total: number) =>
  total > 0 ? Math.min(100, Math.max(0, (value / total) * 100)) : 0

export function VramBar({ used, total, locked }: VramBarProps) {
  const usedPct = percentage(used, total)
  const lockedPct = percentage(locked, total)

  let usedClass = "bg-emerald-500"
  if (usedPct >= 90) {
    usedClass = "bg-red-500"
  } else if (usedPct >= 70) {
    usedClass = "bg-amber-400"
  }

  return (
    <div className="space-y-2">
      <div className="h-2.5 w-full overflow-hidden rounded-full bg-muted">
        <div className={`h-full transition-all ${usedClass}`} style={{ width: `${usedPct}%` }} />
      </div>

      <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted/70">
        <div
          className="h-full bg-cyan-400 transition-all"
          style={{ width: `${lockedPct}%` }}
          title="Locked VRAM"
        />
      </div>

      <div className="flex items-center justify-between text-xs text-muted-foreground">
        <span>
          {used.toLocaleString()} / {total.toLocaleString()} MB
        </span>
        <span>locked {locked.toLocaleString()} MB</span>
      </div>
    </div>
  )
}
