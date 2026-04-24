interface InstallProgressBarProps {
  progress: number | null
  detail: string | null
}

export function InstallProgressBar({ progress, detail }: InstallProgressBarProps) {
  const pct = progress == null ? null : Math.min(100, Math.max(0, progress * 100))

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between text-xs text-muted-foreground">
        <span className="truncate">{detail ?? "Procesando..."}</span>
        {pct != null && <span className="tabular-nums">{pct.toFixed(0)}%</span>}
      </div>
      <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
        {pct == null ? (
          <div className="h-full w-1/3 animate-pulse rounded-full bg-violet-500/60" />
        ) : (
          <div
            className="h-full rounded-full bg-violet-500 transition-[width] duration-300"
            style={{ width: `${pct}%` }}
          />
        )}
      </div>
    </div>
  )
}
