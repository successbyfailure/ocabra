import type { HFModelCard } from "@/types"
import { getProbeOverrideHint, getProbeStatusLabel } from "@/lib/vllmProbe"

interface HFModelCardProps {
  model: HFModelCard
  onInstall: (model: HFModelCard) => void
}

export function HFModelCard({ model, onInstall }: HFModelCardProps) {
  const sizeLabel = model.sizeGb == null ? "unknown" : `${model.sizeGb.toFixed(1)} GB`
  const isUnsupported = model.compatibility === "unsupported"
  const support = model.vllmSupport
  const probe = support?.runtimeProbe
  const supportLabel = support ? `${support.label}${support.runner ? ` / ${support.runner}` : ""}` : null
  const probeRecommendation =
    probe && (probe.recommendedModelImpl || probe.recommendedRunner)
      ? [
          probe.recommendedModelImpl ? `model_impl=${probe.recommendedModelImpl}` : null,
          probe.recommendedRunner ? `runner=${probe.recommendedRunner}` : null,
        ]
          .filter(Boolean)
          .join(", ")
      : null
  const probeStatusLabel = getProbeStatusLabel(probe?.status)
  const probeOverrideHint = getProbeOverrideHint(probe?.status)

  return (
    <article className="rounded-lg border border-border bg-card p-4">
      <div className="mb-2 flex items-start justify-between gap-3">
        <div>
          <h3 className="font-medium">{model.modelName}</h3>
          <p className="text-xs text-muted-foreground">{model.repoId}</p>
        </div>
        <button
          type="button"
          onClick={() => onInstall(model)}
          disabled={isUnsupported}
          className="rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground"
        >
          Instalar
        </button>
      </div>

      <div className="flex flex-wrap gap-2 text-xs text-muted-foreground">
        <span className="rounded bg-muted px-2 py-0.5">downloads {model.downloads.toLocaleString()}</span>
        <span className="rounded bg-muted px-2 py-0.5">task {model.task ?? "unknown"}</span>
        <span className="rounded bg-muted px-2 py-0.5">size {sizeLabel}</span>
        <span className="rounded bg-muted px-2 py-0.5">backend {model.suggestedBackend}</span>
        {supportLabel && <span className="rounded bg-muted px-2 py-0.5">vllm {supportLabel}</span>}
        {model.gated && <span className="rounded bg-amber-500/20 px-2 py-0.5 text-amber-200">gated</span>}
        {model.compatibility === "warning" && (
          <span className="rounded bg-amber-500/20 px-2 py-0.5 text-amber-200">compat warning</span>
        )}
        {model.compatibility === "unsupported" && (
          <span className="rounded bg-red-500/20 px-2 py-0.5 text-red-200">not installable</span>
        )}
        {probeOverrideHint && (
          <span className="rounded bg-amber-500/20 px-2 py-0.5 text-amber-200">{probeOverrideHint}</span>
        )}
      </div>
      {support?.requiredOverrides && support.requiredOverrides.length > 0 && (
        <p className="mt-3 text-xs text-muted-foreground">
          requiere: {support.requiredOverrides.join(", ")}
        </p>
      )}
      {support?.recipeId && (
        <p className="mt-2 text-xs text-muted-foreground">
          recipe: {support.recipeId}
          {support.recipeNotes.length > 0 ? ` · ${support.recipeNotes[0]}` : ""}
        </p>
      )}
      {support?.suggestedConfig && Object.keys(support.suggestedConfig).length > 0 && (
        <p className="mt-2 text-xs text-muted-foreground">
          config sugerida:{" "}
          {Object.entries(support.suggestedConfig)
            .map(([key, value]) => `${key}=${typeof value === "string" ? value : JSON.stringify(value)}`)
            .join(", ")}
        </p>
      )}
      {support?.suggestedTuning && Object.keys(support.suggestedTuning).length > 0 && (
        <p className="mt-2 text-xs text-muted-foreground">
          tuning recomendado:{" "}
          {Object.entries(support.suggestedTuning)
            .map(([key, value]) => `${key}=${typeof value === "string" ? value : JSON.stringify(value)}`)
            .join(", ")}
        </p>
      )}
      {probeStatusLabel && (
        <p className="mt-2 text-xs text-muted-foreground">probe: {probeStatusLabel}</p>
      )}
      {probeRecommendation && (
        <p className="mt-2 text-xs text-muted-foreground">probe verificado: {probeRecommendation}</p>
      )}
      {probeOverrideHint && <p className="mt-2 text-xs text-muted-foreground">{probeOverrideHint}</p>}
      {probe?.reason && <p className="mt-2 text-xs text-muted-foreground">{probe.reason}</p>}
      {model.compatibilityReason && (
        <p className="mt-3 text-xs text-muted-foreground">{model.compatibilityReason}</p>
      )}
    </article>
  )
}
