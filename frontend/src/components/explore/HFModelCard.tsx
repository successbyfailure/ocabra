import type { HFModelCard } from "@/types"

interface HFModelCardProps {
  model: HFModelCard
  onInstall: (model: HFModelCard) => void
}

export function HFModelCard({ model, onInstall }: HFModelCardProps) {
  const sizeLabel = model.sizeGb == null ? "unknown" : `${model.sizeGb.toFixed(1)} GB`
  const isUnsupported = model.compatibility === "unsupported"

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
        {model.gated && <span className="rounded bg-amber-500/20 px-2 py-0.5 text-amber-200">gated</span>}
        {model.compatibility === "warning" && (
          <span className="rounded bg-amber-500/20 px-2 py-0.5 text-amber-200">compat warning</span>
        )}
        {model.compatibility === "unsupported" && (
          <span className="rounded bg-red-500/20 px-2 py-0.5 text-red-200">not installable</span>
        )}
      </div>
      {model.compatibilityReason && (
        <p className="mt-3 text-xs text-muted-foreground">{model.compatibilityReason}</p>
      )}
    </article>
  )
}
