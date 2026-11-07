import { useState } from "react"
import { Loader2, Plus, Settings2, Trash2 } from "lucide-react"
import { BackendIcons, backendLabel, mergeBackendItems } from "@/components/models/BackendIcons"
import { ModelStatusBadge } from "@/components/models/ModelStatusBadge"
import type { BackendType, LogicalModel, ModelDeployment, ModelState } from "@/types"

interface LogicalModelCardProps {
  model: LogicalModel
  /** Category label shown on the card (already resolved by the page). */
  categoryLabel?: string
  /** Live deployment state from the model store, keyed by deployment id. */
  deploymentStates: Record<string, ModelState | undefined>
  busy?: boolean
  onConfigure: (modelId: string) => void
  onLoad: (modelId: string) => void
  onUnload: (modelId: string) => void
  onSetDefault: (model: LogicalModel, backend: BackendType) => void
  onAddDeployment: (model: LogicalModel, backend: BackendType) => void
  onRemoveDeployment: (model: LogicalModel, deployment: ModelDeployment) => void
}

/**
 * One public model with its per-backend deployments (Models → "Por modelo").
 * Clicking a deployment opens that deployment's config modal; the default
 * backend, adding and removing deployments are managed here.
 */
export function LogicalModelCard({
  model,
  categoryLabel,
  deploymentStates,
  busy = false,
  onConfigure,
  onLoad,
  onUnload,
  onSetDefault,
  onAddDeployment,
  onRemoveDeployment,
}: LogicalModelCardProps) {
  const [backendToAdd, setBackendToAdd] = useState("")
  const items = mergeBackendItems(model.deployments, model.compatibleBackends)
  const addable = model.compatibleBackends.filter((entry) => !entry.deployed && entry.installed)
  const deployedBackends = new Set(model.deployments.map((dep) => dep.backend))
  // A compatible entry flagged deployed=false may still exist as a deployment
  // if the list is stale; never offer to add a backend twice.
  const addOptions = addable.filter((entry) => !deployedBackends.has(entry.backend))
  const showName = model.displayName && model.displayName !== model.modelKey

  return (
    <article
      className="rounded-lg border border-border bg-card p-4"
      data-testid={`logical-model-${model.modelKey}`}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <h3 className="truncate font-medium" title={model.modelKey}>
            {showName ? model.displayName : model.modelKey}
          </h3>
          {showName && <p className="truncate font-mono text-xs text-muted-foreground">{model.modelKey}</p>}
        </div>
        <div className="flex shrink-0 items-center gap-2">
          {busy && <Loader2 size={14} className="animate-spin text-muted-foreground" />}
          {categoryLabel && (
            <span className="rounded bg-muted px-2 py-0.5 text-xs text-muted-foreground">{categoryLabel}</span>
          )}
        </div>
      </div>

      <BackendIcons
        className="mt-2"
        items={items}
        onSelect={(item) => item.modelId && onConfigure(item.modelId)}
      />

      <ul className="mt-3 space-y-1.5">
        {model.deployments.map((dep) => {
          const state = deploymentStates[dep.modelId]
          const status = state?.runtimePhase ?? state?.status ?? dep.status ?? "configured"
          const isLoaded = (state?.status ?? dep.status) === "loaded"
          const isTransient = status === "loading" || status === "unloading"
          return (
            <li
              key={dep.modelId}
              className="flex items-center justify-between gap-2 rounded-md border border-border/60 bg-background/40 px-2 py-1.5 text-sm"
            >
              <button
                type="button"
                onClick={() => onConfigure(dep.modelId)}
                className="flex min-w-0 items-center gap-1.5 text-left hover:underline"
                title={`Configurar ${dep.modelId}`}
              >
                {dep.isDefault && (
                  <span className="text-amber-300" aria-label="backend por defecto">
                    ★
                  </span>
                )}
                <span className={dep.installed ? "" : "text-muted-foreground line-through"}>
                  {backendLabel(dep.backend)}
                </span>
                {!dep.installed && <span className="text-xs text-muted-foreground">(no instalado)</span>}
              </button>
              <div className="flex shrink-0 items-center gap-1.5">
                <ModelStatusBadge status={status} />
                <button
                  type="button"
                  disabled={busy || isTransient || !dep.installed}
                  onClick={() => (isLoaded ? onUnload(dep.modelId) : onLoad(dep.modelId))}
                  className="rounded-md border border-border px-2 py-0.5 text-xs hover:bg-muted disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {isLoaded ? "Descargar" : "Cargar"}
                </button>
                <button
                  type="button"
                  onClick={() => onConfigure(dep.modelId)}
                  className="rounded-md border border-border p-1 text-muted-foreground hover:bg-muted hover:text-foreground"
                  aria-label={`Configurar ${dep.backend}`}
                  title="Configurar despliegue"
                >
                  <Settings2 size={12} />
                </button>
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => onRemoveDeployment(model, dep)}
                  className="rounded-md border border-red-500/30 p-1 text-red-300 hover:bg-red-500/15 disabled:opacity-50"
                  aria-label={`Quitar despliegue ${dep.backend}`}
                  title="Quitar despliegue (los ficheros se conservan)"
                >
                  <Trash2 size={12} />
                </button>
              </div>
            </li>
          )
        })}
      </ul>

      <div className="mt-3 grid gap-2 sm:grid-cols-2">
        {model.deployments.length > 1 && (
          <label className="block text-xs text-muted-foreground">
            Backend por defecto
            <select
              aria-label="Backend por defecto"
              value={model.defaultBackend}
              disabled={busy}
              onChange={(event) => {
                const next = event.target.value as BackendType
                if (next && next !== model.defaultBackend) onSetDefault(model, next)
              }}
              className="mt-1 w-full rounded-md border border-border bg-background px-2 py-1.5 text-sm text-foreground"
            >
              {model.deployments.map((dep) => (
                <option key={dep.backend} value={dep.backend} disabled={!dep.installed}>
                  {backendLabel(dep.backend)}
                  {dep.installed ? "" : " (no instalado)"}
                </option>
              ))}
            </select>
          </label>
        )}
        {addOptions.length > 0 && (
          <div className="block text-xs text-muted-foreground">
            Añadir backend
            <div className="mt-1 flex gap-1.5">
              <select
                aria-label="Añadir backend"
                value={backendToAdd}
                disabled={busy}
                onChange={(event) => setBackendToAdd(event.target.value)}
                className="w-full min-w-0 rounded-md border border-border bg-background px-2 py-1.5 text-sm text-foreground"
              >
                <option value="">Elegir…</option>
                {addOptions.map((entry) => (
                  <option key={entry.backend} value={entry.backend} title={entry.reason}>
                    {backendLabel(entry.backend)}
                    {entry.status === "probable" ? " (probable)" : ""}
                  </option>
                ))}
              </select>
              <button
                type="button"
                disabled={busy || !backendToAdd}
                onClick={() => {
                  if (!backendToAdd) return
                  onAddDeployment(model, backendToAdd as BackendType)
                  setBackendToAdd("")
                }}
                className="inline-flex shrink-0 items-center gap-1 rounded-md border border-border px-2 py-1.5 text-sm text-foreground hover:bg-muted disabled:cursor-not-allowed disabled:opacity-50"
              >
                <Plus size={12} />
                Añadir
              </button>
            </div>
          </div>
        )}
      </div>
    </article>
  )
}
