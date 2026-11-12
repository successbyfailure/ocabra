import { useState } from "react"
import { Plus, Star, Trash2, Wand2 } from "lucide-react"
import { toast } from "sonner"
import { api } from "@/api/client"
import { BackendIcon } from "@/components/dashboard/BackendIcon"
import { fmtContext } from "@/components/dashboard/format"
import { ConfirmDialog } from "@/components/common/ConfirmDialog"
import { CapabilityIcons } from "@/components/models/CapabilityIcons"
import { getProbeOverrideHint, getProbeStatusLabel } from "@/lib/vllmProbe"
import { cn } from "@/lib/cn"
import type { BackendType, ModelState } from "@/types"
import { getPath, isObject } from "../draft"
import { fmtMb } from "../PreviewPanel"
import type { JsonObject } from "../types"
import { Callout, Chip, Hint, Panel, inputCls } from "../ui"
import type { SectionProps } from "./types"

function fmtDate(value: string | null | undefined): string | null {
  if (!value) return null
  const d = new Date(value)
  return Number.isNaN(d.getTime()) ? null : d.toLocaleDateString("es-ES", { year: "numeric", month: "short", day: "numeric" })
}

export function SummarySection(props: SectionProps & { modelState: ModelState | null; onNavigateDeployment: (modelId: string) => void; onGoTo: (section: string) => void }) {
  const { editor, readOnly, logical, refreshLogical, modelState, onNavigateDeployment, onGoTo } = props
  const m = editor.model
  const [name, setName] = useState<string | null>(null)
  const [removeTarget, setRemoveTarget] = useState<BackendType | null>(null)
  if (!m) return null
  const caps = m.capabilities ?? {}
  const extra = editor.base?.extraConfig ?? {}
  const vllm = (isObject(extra.vllm) ? extra.vllm : {}) as JsonObject
  const errorMsg = m.error_message
  const deployments = logical?.deployments ?? []
  const addable = (logical?.compatibleBackends ?? []).filter((b) => b.installed && !deployments.some((d) => d.backend === b.backend))

  const saveName = async () => {
    if (!logical || name == null || name === logical.displayName) return setName(null)
    try {
      await api.logicalModels.patch(logical.modelKey, { displayName: name })
      toast.success("Nombre guardado")
      await refreshLogical()
    } catch (err) {
      toast.error(err instanceof Error ? err.message : String(err))
    }
    setName(null)
  }

  const removeText = (() => {
    if (!removeTarget || !logical) return ""
    if (deployments.length <= 1) return "Es el único despliegue: se borra el modelo de oCabra (los ficheros se conservan)."
    const isDefault = logical.defaultBackend === removeTarget
    return isDefault
      ? "Es el despliegue por defecto: el nombre público pasará a servirlo otro backend. Los ficheros se conservan."
      : "Se borra la configuración de este despliegue. Los ficheros se conservan."
  })()

  return (
    <div className="space-y-4">
      {errorMsg && (
        <Callout tone="error">
          <p className="font-medium">Último error de carga</p>
          <p className="mt-1 font-mono text-[11px]">{errorMsg}</p>
          {/trust_remote_code/i.test(errorMsg) && editor.fieldByKey.has("vllm.trust_remote_code") && (
            <button type="button" className="mt-1 underline" onClick={() => onGoTo("runtime")}>
              Ir a «Confiar en código remoto»
            </button>
          )}
        </Callout>
      )}

      <Panel title="Modelo" subtitle="El nombre público es el que piden los clientes; cada despliegue lo sirve con un backend.">
        <div className="grid gap-4 md:grid-cols-2">
          <div className="space-y-1">
            <p className="text-xs font-medium">Nombre visible</p>
            <input
              className={inputCls}
              disabled={readOnly || !logical}
              value={name ?? logical?.displayName ?? m.display_name}
              onChange={(e) => setName(e.target.value)}
              onBlur={() => void saveName()}
              onKeyDown={(e) => e.key === "Enter" && void saveName()}
            />
            <Hint>
              El nombre de la API sigue siendo <span className="font-mono">{logical?.modelKey ?? m.model_id}</span>. Se guarda al salir del campo.
            </Hint>
          </div>
          <div className="space-y-1">
            <p className="text-xs font-medium">Capacidades</p>
            {modelState ? <CapabilityIcons model={modelState} /> : null}
            <div className="flex flex-wrap gap-1.5 pt-1">
              {Number(caps.context_length) > 0 && <Chip>contexto {fmtContext(Number(caps.context_length))}</Chip>}
              {Number(caps.native_context_length) > 0 && <Chip>nativo {fmtContext(Number(caps.native_context_length))}</Chip>}
              {m.disk_size_bytes ? <Chip>{fmtMb(m.disk_size_bytes / 1024 / 1024)} en disco</Chip> : null}
            </div>
          </div>
        </div>

        <div className="mt-4 space-y-2" data-testid="deployments">
          <p className="text-xs font-medium">Despliegues</p>
          {deployments.map((dep) => {
            const isDefault = logical?.defaultBackend === dep.backend
            const current = dep.modelId === m.model_id
            return (
              <div
                key={dep.modelId}
                className={cn("flex flex-wrap items-center gap-2 rounded-lg border px-3 py-2", current ? "border-primary/60 bg-primary/5" : "border-border")}
              >
                <BackendIcon backendType={dep.backend} />
                <button type="button" className="font-mono text-xs hover:underline" onClick={() => onNavigateDeployment(dep.modelId)}>
                  {dep.modelId}
                </button>
                {isDefault ? (
                  <Chip tone="info">
                    <Star size={10} fill="currentColor" /> sirve el nombre público
                  </Chip>
                ) : (
                  <Chip>solo por su nombre completo</Chip>
                )}
                {!dep.installed && <Chip tone="warn">backend no instalado</Chip>}
                {dep.status && <Chip>{dep.status}</Chip>}
                <span className="ml-auto flex items-center gap-1">
                  {!isDefault && !readOnly && dep.installed && (
                    <button
                      type="button"
                      className="rounded border border-border px-2 py-0.5 text-[11px] hover:bg-muted"
                      onClick={async () => {
                        try {
                          await api.logicalModels.patch(logical!.modelKey, { defaultBackend: dep.backend })
                          toast.success(`Ahora el nombre público lo sirve ${dep.backend}`)
                          await refreshLogical()
                        } catch (err) {
                          toast.error(err instanceof Error ? err.message : String(err))
                        }
                      }}
                    >
                      Hacer por defecto
                    </button>
                  )}
                  {!readOnly && (
                    <button type="button" title="Quitar despliegue" className="rounded p-1 text-muted-foreground hover:bg-red-500/10 hover:text-red-300" onClick={() => setRemoveTarget(dep.backend)}>
                      <Trash2 size={13} />
                    </button>
                  )}
                </span>
              </div>
            )
          })}
          {!readOnly && addable.length > 0 && (
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-[11px] text-muted-foreground">Añadir con:</span>
              {addable.map((b) => (
                <button
                  key={b.backend}
                  type="button"
                  title={b.reason}
                  className="inline-flex items-center gap-1 rounded-md border border-dashed border-border px-2 py-1 text-xs hover:bg-muted"
                  onClick={async () => {
                    try {
                      await api.logicalModels.addDeployment(logical!.modelKey, { backend: b.backend })
                      toast.success(`Despliegue ${b.backend} añadido`)
                      await refreshLogical()
                    } catch (err) {
                      toast.error(err instanceof Error ? err.message : String(err))
                    }
                  }}
                >
                  <Plus size={12} /> {b.backend}
                  {b.status === "probable" && <span className="text-[10px] text-amber-300">(probable)</span>}
                </button>
              ))}
            </div>
          )}
        </div>
      </Panel>

      <Panel title="Origen">
        <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-xs">
          <dt className="text-muted-foreground">Id interno</dt>
          <dd className="font-mono">{m.backend_model_id}</dd>
          {typeof getPath(extra, ["llama_cpp", "model_path"]) === "string" && (
            <>
              <dt className="text-muted-foreground">Fichero</dt>
              <dd className="truncate font-mono" title={String(getPath(extra, ["llama_cpp", "model_path"]))}>
                {String(getPath(extra, ["llama_cpp", "model_path"])).split("/").pop()}
              </dd>
            </>
          )}
          {fmtDate(m.release_date) && (
            <>
              <dt className="text-muted-foreground">Publicado</dt>
              <dd>{fmtDate(m.release_date)}</dd>
            </>
          )}
          {fmtDate(m.last_updated) && (
            <>
              <dt className="text-muted-foreground">Actualizado</dt>
              <dd>{fmtDate(m.last_updated)}</dd>
            </>
          )}
          {fmtDate(m.metadata_fetched_at) && (
            <>
              <dt className="text-muted-foreground">Consultado</dt>
              <dd>{fmtDate(m.metadata_fetched_at)}</dd>
            </>
          )}
        </dl>
      </Panel>

      {(vllm.recipe_id || vllm.probe_status) && <VllmRecipeCard {...props} vllm={vllm} />}

      <ConfirmDialog
        open={removeTarget != null}
        onOpenChange={(open) => !open && setRemoveTarget(null)}
        title={`Quitar el despliegue ${removeTarget ?? ""}`}
        description={removeText}
        confirmLabel="Quitar"
        variant="danger"
        onConfirm={async () => {
          if (!logical || !removeTarget) return
          try {
            const res = await api.logicalModels.removeDeployment(logical.modelKey, removeTarget)
            toast.success(res.newDefaultBackend ? `Quitado; ahora lo sirve ${res.newDefaultBackend}` : "Despliegue quitado")
            setRemoveTarget(null)
            const next = logical.deployments.find((d) => d.backend !== removeTarget)
            if (next && removeTarget === m.backend_type) onNavigateDeployment(next.modelId)
            else await refreshLogical()
          } catch (err) {
            toast.error(err instanceof Error ? err.message : String(err))
          }
        }}
      />
    </div>
  )
}

/** Receta de vLLM y resultado del probe del motor, con sus botones de aplicar. */
function VllmRecipeCard({ editor, readOnly, vllm }: SectionProps & { vllm: JsonObject }) {
  const suggested = (isObject(vllm.suggested_config) ? vllm.suggested_config : {}) as JsonObject
  const tuning = (isObject(vllm.suggested_tuning) ? vllm.suggested_tuning : {}) as JsonObject
  const apply = (values: JsonObject) => {
    for (const [k, v] of Object.entries(values)) editor.setField(["vllm", k], v)
    toast.success("Aplicado al borrador: revisa los cambios y guarda")
  }
  const probeStatus = vllm.probe_status as string | undefined
  const recImpl = vllm.probe_recommended_model_impl as string | undefined
  const recRunner = vllm.probe_recommended_runner as string | undefined
  const curImpl = (getPath(editor.draft!.extraConfig, ["vllm", "model_impl"]) as string | undefined) ?? undefined
  const curRunner = (getPath(editor.draft!.extraConfig, ["vllm", "runner"]) as string | undefined) ?? undefined
  const hint = getProbeOverrideHint(probeStatus as never)
  return (
    <Panel title="Receta de vLLM" subtitle={vllm.recipe_id ? `Receta ${String(vllm.recipe_id)}` : undefined}>
      <div className="space-y-2 text-xs">
        {Array.isArray(vllm.recipe_notes) && vllm.recipe_notes.length > 0 && (
          <ul className="list-disc pl-4 text-muted-foreground">
            {vllm.recipe_notes.map((n, i) => (
              <li key={i}>{String(n)}</li>
            ))}
          </ul>
        )}
        <div className="flex flex-wrap gap-2">
          {Object.keys(suggested).length > 0 && (
            <button type="button" disabled={readOnly} className="inline-flex items-center gap-1 rounded-md border border-border px-2 py-1 hover:bg-muted" onClick={() => apply(suggested)}>
              <Wand2 size={12} /> Reaplicar la receta
            </button>
          )}
          {Object.keys(tuning).length > 0 && (
            <button type="button" disabled={readOnly} className="inline-flex items-center gap-1 rounded-md border border-border px-2 py-1 hover:bg-muted" onClick={() => apply(tuning)}>
              <Wand2 size={12} /> Aplicar el tuning recomendado
            </button>
          )}
        </div>
        {probeStatus && (
          <div className="rounded-lg border border-border p-2">
            <p>
              Probe del motor: <span className="font-medium">{getProbeStatusLabel(probeStatus as never)}</span>
              {vllm.probe_observed_at ? <span className="text-muted-foreground"> · {fmtDate(String(vllm.probe_observed_at))}</span> : null}
            </p>
            {(recImpl || recRunner) && (
              <p className="text-muted-foreground">
                Recomienda {recImpl ? `model_impl=${recImpl}` : ""} {recRunner ? `runner=${recRunner}` : ""}
                {(recImpl && recImpl !== curImpl) || (recRunner && recRunner !== curRunner) ? " — distinto de lo configurado" : ""}
              </p>
            )}
            {vllm.probe_reason ? <p className="text-muted-foreground">{String(vllm.probe_reason)}</p> : null}
            {typeof hint === "string" && hint && <Hint tone="info">{hint}</Hint>}
            {(recImpl || recRunner) && (
              <button
                type="button"
                disabled={readOnly}
                className="mt-1 inline-flex items-center gap-1 rounded-md border border-border px-2 py-1 hover:bg-muted"
                onClick={() => apply({ ...(recImpl ? { model_impl: recImpl } : {}), ...(recRunner ? { runner: recRunner } : {}) })}
              >
                <Wand2 size={12} /> Aplicar la recomendación
              </button>
            )}
          </div>
        )}
      </div>
    </Panel>
  )
}
