import { Clock, Loader2 } from "lucide-react"
import type { DurationEstimateResult } from "@/api/client"

export function formatEtaSeconds(seconds: number): string {
  if (seconds < 1) return "<1s"
  if (seconds < 60) return `${Math.round(seconds)}s`
  const m = Math.floor(seconds / 60)
  const s = Math.round(seconds - m * 60)
  return s === 0 ? `${m}m` : `${m}m${s.toString().padStart(2, "0")}s`
}

export function EtaBadge({
  estimate,
  loading,
}: {
  estimate: DurationEstimateResult | null
  loading: boolean
}) {
  if (loading && !estimate) {
    return (
      <span
        className="inline-flex items-center gap-1 rounded-md border border-border px-2 py-1 text-xs text-muted-foreground"
        title="Estimando duración..."
      >
        <Loader2 size={12} className="animate-spin" />
        Estimando
      </span>
    )
  }
  if (!estimate) return null
  const eta = formatEtaSeconds(estimate.expectedSeconds)
  const p95 = formatEtaSeconds(estimate.p95Seconds)
  const confidencePct = Math.round(estimate.confidence * 100)
  // Only show provenance when it's meaningful — a family-default with no
  // samples is basically a wild guess, so we flag it.
  const isWeak = estimate.source === "family_default" || estimate.confidence < 0.3
  const tone = isWeak
    ? "border-border text-muted-foreground"
    : "border-primary/40 text-primary"
  const title = [
    `Estimación: ${eta} (p95 ${p95})`,
    `Confianza: ${confidencePct}%`,
    `Fuente: ${estimate.source}`,
    estimate.sampleCount > 0 ? `${estimate.sampleCount} muestras previas` : null,
    estimate.coldStartSeconds > 0
      ? `Añade ~${formatEtaSeconds(estimate.coldStartSeconds)} si el modelo está frío`
      : null,
  ]
    .filter(Boolean)
    .join(" · ")
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-md border px-2 py-1 text-xs ${tone}`}
      title={title}
    >
      <Clock size={12} />
      ~{eta}
      {isWeak && <span className="opacity-60">?</span>}
    </span>
  )
}
