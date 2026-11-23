import { Clock3, Database, Info } from "lucide-react"
import {
  formatEstimatedDuration,
  type BenchmarkTimeEstimate as TimeEstimate,
  type EstimateConfidence,
  type EstimateSource,
} from "./benchmarkEstimate"

const CONFIDENCE_LABEL: Record<EstimateConfidence, string> = {
  high: "Confianza alta",
  medium: "Confianza media",
  low: "Confianza baja",
}

const SOURCE_LABEL: Record<EstimateSource, string> = {
  model_history: "histórico del modelo",
  scaled_model_history: "histórico escalado del modelo",
  suite_history: "mediana de la suite",
  scaled_suite_history: "suite similar escalada",
  fallback: "estimación inicial",
}

export function BenchmarkTimeEstimate({
  estimate,
  needsPreparation,
  autoPause,
}: {
  estimate: TimeEstimate | null
  needsPreparation: boolean
  autoPause: boolean
}) {
  if (!estimate) return (
    <div className="rounded-lg border border-dashed border-border p-3 text-xs text-muted-foreground">
      <Clock3 size={14} className="mr-1.5 inline" />Selecciona modelos para calcular el tiempo aproximado.
    </div>
  )

  const confidenceClass = estimate.confidence === "high"
    ? "bg-emerald-500/10 text-emerald-300"
    : estimate.confidence === "medium"
      ? "bg-sky-500/10 text-sky-300"
      : "bg-amber-500/10 text-amber-300"

  return (
    <div className="rounded-lg border border-primary/25 bg-primary/5 p-3">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="flex items-center gap-1.5 text-xs font-medium uppercase tracking-wide text-muted-foreground"><Clock3 size={14} /> Tiempo estimado</p>
          <p className="mt-1 text-2xl font-semibold">≈ {formatEstimatedDuration(estimate.seconds)}</p>
          <p className="text-xs text-muted-foreground">
            Rango probable: {formatEstimatedDuration(estimate.lowerSeconds)}–{formatEstimatedDuration(estimate.upperSeconds)}
          </p>
        </div>
        <span className={`rounded-full px-2 py-1 text-[10px] font-medium ${confidenceClass}`}>{CONFIDENCE_LABEL[estimate.confidence]}</span>
      </div>

      <div className="mt-3 flex items-center gap-1.5 text-xs text-muted-foreground">
        <Database size={12} /> Histórico para {estimate.historicalModels}/{estimate.models.length} modelos
        <span>· preparación ≈ {formatEstimatedDuration(estimate.setupSeconds)}</span>
      </div>

      <details className="mt-2 text-xs">
        <summary className="cursor-pointer text-primary hover:underline">Ver desglose por modelo</summary>
        <div className="mt-2 max-h-48 space-y-1 overflow-y-auto rounded-md border border-border/60 bg-background/70 p-2">
          {estimate.models.map((item) => (
            <div key={item.model} className="flex items-center justify-between gap-3">
              <span className="min-w-0 truncate font-mono" title={item.model}>{item.model}</span>
              <span className="shrink-0 text-right">
                <strong className="font-medium">{formatEstimatedDuration(item.seconds)}</strong>
                <span className="ml-1 text-[10px] text-muted-foreground">{SOURCE_LABEL[item.source]}{item.samples > 0 ? ` · ${item.samples} muestra${item.samples === 1 ? "" : "s"}` : ""}</span>
              </span>
            </div>
          ))}
        </div>
      </details>

      {(needsPreparation || autoPause) && <p className="mt-2 text-[10px] leading-relaxed text-muted-foreground"><Info size={11} className="mr-1 inline" />
        {needsPreparation ? "La primera descarga o preparación del dataset puede ampliar este tiempo. " : ""}
        {autoPause ? "Las pausas por tráfico real no están incluidas." : ""}
      </p>}
    </div>
  )
}
