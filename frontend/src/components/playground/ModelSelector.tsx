import { CapabilityBadge } from "@/components/common/CapabilityBadge"
import type { ModelState } from "@/types"

interface ModelSelectorProps {
  models: ModelState[]
  selectedModelId: string
  onSelect: (modelId: string) => void
}

export function ModelSelector({ models, selectedModelId, onSelect }: ModelSelectorProps) {
  const selected = models.find((model) => model.modelId === selectedModelId) ?? null

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
              {model.displayName}
            </option>
          ))}
        </select>
      </label>

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
