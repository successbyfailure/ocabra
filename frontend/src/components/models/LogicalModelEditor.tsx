import { useEffect, useState } from "react"
import * as Dialog from "@radix-ui/react-dialog"
import * as Tabs from "@radix-ui/react-tabs"
import { Loader2, Plus, Trash2, X } from "lucide-react"
import { api } from "@/api/client"
import { BackendIcons, backendLabel, mergeBackendItems } from "@/components/models/BackendIcons"
import { formatPreferredGpu } from "@/components/models/LogicalModelCard"
import { ModelConfigForm, type ModelConfigSaveHandler } from "@/components/models/ModelConfigModal"
import { ModelStatusBadge } from "@/components/models/ModelStatusBadge"
import type { BackendType, GPUState, LogicalModel, ModelDeployment, ModelState } from "@/types"

const GENERAL_TAB = "__general__"

interface LogicalModelEditorProps {
  /** Model being edited; null = closed. Pass the fresh entry after each refresh. */
  model: LogicalModel | null
  open: boolean
  onOpenChange: (open: boolean) => void
  gpus: GPUState[]
  /** Live deployment state from the model store, keyed by deployment id. */
  deploymentStates: Record<string, ModelState | undefined>
  busy?: boolean
  onRename: (model: LogicalModel, displayName: string) => Promise<void> | void
  onSetDefault: (model: LogicalModel, backend: BackendType) => Promise<void> | void
  onAddDeployment: (model: LogicalModel, backend: BackendType) => Promise<void> | void
  /** Asks for confirmation (the page owns the dialog) and removes. */
  onRemoveDeployment: (model: LogicalModel, deployment: ModelDeployment) => void
  /** Saves one deployment's config (same handler as the per-deployment modal). */
  onSaveDeployment: ModelConfigSaveHandler
}

const tabClass =
  "px-3 py-2 text-sm font-medium text-muted-foreground transition-colors hover:text-foreground data-[state=active]:border-b-2 data-[state=active]:border-primary data-[state=active]:text-foreground"

/**
 * "Editar modelo": a General tab (display name, default backend, add a
 * backend) plus one tab per deployment embedding that deployment's config form.
 */
export function LogicalModelEditor({
  model,
  open,
  onOpenChange,
  gpus,
  deploymentStates,
  busy = false,
  onRename,
  onSetDefault,
  onAddDeployment,
  onRemoveDeployment,
  onSaveDeployment,
}: LogicalModelEditorProps) {
  const [tab, setTab] = useState(GENERAL_TAB)
  const [displayName, setDisplayName] = useState("")
  const [backendToAdd, setBackendToAdd] = useState("")
  // Deployments not in the model store yet (e.g. just added) are fetched once.
  const [fetched, setFetched] = useState<Record<string, ModelState>>({})

  const modelKey = model?.modelKey ?? null
  useEffect(() => {
    if (!open) return
    setTab(GENERAL_TAB)
    setBackendToAdd("")
    setFetched({})
  }, [open, modelKey])

  useEffect(() => {
    setDisplayName(model?.displayName ?? "")
  }, [model?.displayName, modelKey])

  // A removed deployment takes its tab with it.
  useEffect(() => {
    if (tab !== GENERAL_TAB && model && !model.deployments.some((dep) => dep.modelId === tab)) {
      setTab(GENERAL_TAB)
    }
  }, [model, tab])

  useEffect(() => {
    if (!open || tab === GENERAL_TAB || deploymentStates[tab] || fetched[tab]) return
    let active = true
    void api.models
      .get(tab)
      .then((state) => {
        if (active) setFetched((prev) => ({ ...prev, [tab]: state }))
      })
      .catch(() => {
        // the tab shows a hint; the model view still works
      })
    return () => {
      active = false
    }
  }, [deploymentStates, fetched, open, tab])

  if (!model) return null

  const deployedBackends = new Set(model.deployments.map((dep) => dep.backend))
  const addOptions = model.compatibleBackends.filter(
    (entry) => entry.installed && !entry.deployed && !deployedBackends.has(entry.backend),
  )
  const nameChanged = displayName.trim() !== "" && displayName.trim() !== model.displayName
  const deployments = [...model.deployments].sort((a, b) => Number(b.isDefault) - Number(a.isDefault))

  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-40 bg-black/60" />
        <Dialog.Content className="fixed left-1/2 top-1/2 z-50 max-h-[90vh] w-[95vw] max-w-3xl -translate-x-1/2 -translate-y-1/2 overflow-y-auto rounded-lg border border-border bg-card p-5">
          <div className="mb-3 flex items-start justify-between gap-3">
            <div className="min-w-0">
              <Dialog.Title className="text-lg font-semibold">Editar modelo</Dialog.Title>
              <Dialog.Description className="truncate font-mono text-sm text-muted-foreground">
                {model.modelKey}
              </Dialog.Description>
            </div>
            <div className="flex items-center gap-2">
              {busy && <Loader2 size={14} className="animate-spin text-muted-foreground" />}
              <Dialog.Close className="rounded-md p-1 text-muted-foreground hover:bg-muted" aria-label="Cerrar">
                <X size={16} />
              </Dialog.Close>
            </div>
          </div>

          <Tabs.Root value={tab} onValueChange={setTab}>
            <Tabs.List className="flex flex-wrap gap-1 border-b border-border" aria-label="Secciones del modelo">
              <Tabs.Trigger value={GENERAL_TAB} className={tabClass}>
                General
              </Tabs.Trigger>
              {deployments.map((dep) => (
                <Tabs.Trigger key={dep.modelId} value={dep.modelId} className={tabClass}>
                  {dep.isDefault ? "★ " : ""}
                  {backendLabel(dep.backend)}
                </Tabs.Trigger>
              ))}
            </Tabs.List>

            <Tabs.Content value={GENERAL_TAB} className="mt-4 space-y-4">
              <BackendIcons
                items={mergeBackendItems(model.deployments, model.compatibleBackends)}
                onSelect={(item) => item.modelId && setTab(item.modelId)}
              />

              <div className="block text-sm">
                <label htmlFor="logical-display-name" className="mb-1 block text-muted-foreground">
                  Nombre visible
                </label>
                <div className="flex gap-2">
                  <input
                    id="logical-display-name"
                    value={displayName}
                    onChange={(event) => setDisplayName(event.target.value)}
                    className="w-full rounded-md border border-border bg-background px-3 py-2"
                  />
                  <button
                    type="button"
                    disabled={busy || !nameChanged}
                    onClick={() => void onRename(model, displayName.trim())}
                    className="shrink-0 rounded-md border border-border px-3 py-2 text-sm hover:bg-muted disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    Guardar nombre
                  </button>
                </div>
                <span className="mt-1 block text-xs text-muted-foreground">
                  Solo cambia cómo se muestra; el nombre público de la API sigue siendo {model.modelKey}.
                </span>
              </div>

              <label className="block text-sm">
                <span className="mb-1 block text-muted-foreground">Backend por defecto</span>
                <select
                  aria-label="Backend por defecto"
                  value={model.defaultBackend}
                  disabled={busy || model.deployments.length < 2}
                  onChange={(event) => {
                    const next = event.target.value as BackendType
                    if (next && next !== model.defaultBackend) void onSetDefault(model, next)
                  }}
                  className="w-full rounded-md border border-border bg-background px-3 py-2"
                >
                  {model.deployments.map((dep) => (
                    <option key={dep.backend} value={dep.backend} disabled={!dep.installed}>
                      {backendLabel(dep.backend)}
                      {dep.installed ? "" : " (no instalado)"}
                    </option>
                  ))}
                </select>
                <span className="mt-1 block text-xs text-muted-foreground">
                  Sirve el nombre {model.modelKey}. Los demás despliegues siguen accesibles como
                  «backend/{model.modelKey}».
                </span>
              </label>

              <div className="block text-sm">
                <label htmlFor="logical-add-backend" className="mb-1 block text-muted-foreground">
                  Añadir backend
                </label>
                {addOptions.length === 0 ? (
                  <p className="text-xs text-muted-foreground">
                    No hay más backends instalados que puedan servir este fichero.
                  </p>
                ) : (
                  <div className="flex gap-2">
                    <select
                      id="logical-add-backend"
                      value={backendToAdd}
                      disabled={busy}
                      onChange={(event) => setBackendToAdd(event.target.value)}
                      className="w-full rounded-md border border-border bg-background px-3 py-2"
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
                        void onAddDeployment(model, backendToAdd as BackendType)
                        setBackendToAdd("")
                      }}
                      className="inline-flex shrink-0 items-center gap-1 rounded-md border border-border px-3 py-2 text-sm hover:bg-muted disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      <Plus size={14} />
                      Añadir
                    </button>
                  </div>
                )}
              </div>

              <ul className="space-y-1.5 text-sm">
                {deployments.map((dep) => {
                  const state = deploymentStates[dep.modelId]
                  return (
                    <li
                      key={dep.modelId}
                      className="flex items-center justify-between gap-2 rounded-md border border-border/60 px-2 py-1.5"
                    >
                      <button
                        type="button"
                        className="font-mono text-xs hover:underline"
                        onClick={() => setTab(dep.modelId)}
                      >
                        {dep.isDefault ? "★ " : ""}
                        {dep.modelId}
                      </button>
                      <span className="flex items-center gap-2 text-xs text-muted-foreground">
                        {formatPreferredGpu(state?.preferredGpu ?? dep.preferredGpu)}
                        <ModelStatusBadge status={state?.runtimePhase ?? state?.status ?? dep.status ?? "configured"} />
                      </span>
                    </li>
                  )
                })}
              </ul>
            </Tabs.Content>

            {deployments.map((dep) => {
              const state = deploymentStates[dep.modelId] ?? fetched[dep.modelId]
              const removeButton = (
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => onRemoveDeployment(model, dep)}
                  className="mr-auto inline-flex items-center gap-1 rounded-md border border-red-500/40 px-3 py-2 text-sm text-red-200 hover:bg-red-500/20 disabled:opacity-50"
                >
                  <Trash2 size={14} />
                  Quitar despliegue
                </button>
              )
              return (
                <Tabs.Content key={dep.modelId} value={dep.modelId} className="mt-4 space-y-3">
                  <p className="text-xs text-muted-foreground">
                    <span className="font-mono">{dep.modelId}</span>
                    {dep.isDefault ? " · sirve el nombre público" : " · accesible solo por su nombre oculto"}
                    {dep.installed ? "" : " · backend no instalado en este nodo"}
                  </p>
                  {state ? (
                    <ModelConfigForm
                      model={state}
                      gpus={gpus}
                      active={open && tab === dep.modelId}
                      onSave={onSaveDeployment}
                      footerStart={removeButton}
                    />
                  ) : (
                    <div className="space-y-3">
                      <p className="text-sm text-muted-foreground">Cargando la configuración del despliegue…</p>
                      <div className="flex">{removeButton}</div>
                    </div>
                  )}
                </Tabs.Content>
              )
            })}
          </Tabs.Root>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  )
}
