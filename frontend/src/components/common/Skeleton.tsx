interface SkeletonProps {
  className?: string
}

export function Skeleton({ className = "h-12" }: SkeletonProps) {
  return <div className={`animate-pulse rounded-md bg-muted ${className}`} />
}

export function SkeletonList({ count = 5, className = "h-12" }: { count?: number; className?: string }) {
  return (
    <div className="space-y-2" role="status" aria-label="Cargando">
      {Array.from({ length: count }).map((_, i) => (
        <Skeleton key={`sk-${i}`} className={className} />
      ))}
    </div>
  )
}

export function SkeletonCard() {
  return (
    <div className="animate-pulse rounded-lg border border-border bg-card p-4 space-y-3">
      <div className="h-4 w-1/3 rounded bg-muted" />
      <div className="h-3 w-2/3 rounded bg-muted" />
      <div className="h-8 w-full rounded bg-muted" />
    </div>
  )
}
