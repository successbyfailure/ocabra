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
