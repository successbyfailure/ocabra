import type { OllamaModelCard } from "@/types"

interface OllamaModelCardProps {
  model: OllamaModelCard
  onInstall: (model: OllamaModelCard) => void
}

export function OllamaModelCard({ model, onInstall }: OllamaModelCardProps) {
  return (
    <article className="rounded-lg border border-border bg-card p-4">
      <div className="mb-2 flex items-start justify-between gap-3">
        <div>
          <h3 className="font-medium">{model.name}</h3>
          <p className="text-xs text-muted-foreground line-clamp-2">{model.description}</p>
        </div>
        <button
          type="button"
          onClick={() => onInstall(model)}
          className="rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground"
        >
          Instalar
        </button>
      </div>

      <div className="flex flex-wrap gap-2 text-xs text-muted-foreground">
        <span className="rounded bg-muted px-2 py-0.5">pulls {model.pulls.toLocaleString()}</span>
        <span className="rounded bg-muted px-2 py-0.5">size {model.sizeGb ? `${model.sizeGb} GB` : "?"}</span>
        {model.tags.slice(0, 3).map((tag) => (
          <span key={`${model.name}-${tag}`} className="rounded bg-muted px-2 py-0.5">{tag}</span>
        ))}
      </div>
    </article>
  )
}
