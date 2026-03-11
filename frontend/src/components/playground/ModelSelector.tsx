import { CapabilityBadge } from "@/components/common/CapabilityBadge"
import type { ModelState } from "@/types"

interface ModelSelectorProps {
  models: ModelState[]
  selectedModelId: string
  onSelect: (modelId: string) => void
}

export function ModelSelector({ models, selectedModelId, onSelect }: ModelSelectorProps) {
  const selected = models.find((model) => model.modelId === selectedModelId) ?? null
  const loaded = models.filter((model) => model.status === "loaded")

  return (
    <div className="space-y-2 rounded-lg border border-border bg-card p-3">
      <label className="block text-sm text-muted-foreground">
        Modelo
        <select
          value={selectedModelId}
          onChange={(event) => onSelect(event.target.value)}
          className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2"
        >
          {models.map((model) => (
            <option key={model.modelId} value={model.modelId}>
              {model.displayName} {model.status === "loaded" ? "• loaded" : ""}
            </option>
          ))}
        </select>
      </label>

      <div className="flex flex-wrap items-center gap-2 text-xs">
        <span className="text-muted-foreground">Estado:</span>
        {selected ? (
          <span
            className={`rounded-md px-2 py-0.5 ${
              selected.status === "loaded"
                ? "bg-emerald-500/20 text-emerald-200"
                : "bg-amber-500/20 text-amber-200"
            }`}
          >
            {selected.status === "loaded" ? "Cargado" : selected.status}
          </span>
        ) : (
          <span className="rounded-md bg-muted px-2 py-0.5 text-muted-foreground">Sin modelo</span>
        )}
      </div>

      <div className="flex flex-wrap gap-1 text-xs">
        <span className="text-muted-foreground">Modelos cargados:</span>
        {loaded.length > 0 ? (
          loaded.map((model) => (
            <button
              key={model.modelId}
              type="button"
              onClick={() => onSelect(model.modelId)}
              className="rounded-md border border-emerald-500/40 bg-emerald-500/10 px-2 py-0.5 text-emerald-100 hover:bg-emerald-500/20"
            >
              {model.displayName}
            </button>
          ))
        ) : (
          <span className="text-muted-foreground">ninguno</span>
        )}
      </div>

      {selected && (
        <div className="flex flex-wrap gap-1">
          {(Object.keys(selected.capabilities) as Array<keyof typeof selected.capabilities>)
            .filter((key) => typeof selected.capabilities[key] === "boolean" && selected.capabilities[key])
            .map((capability) => (
              <CapabilityBadge key={capability} capability={capability} />
            ))}
        </div>
      )}
    </div>
  )
}
