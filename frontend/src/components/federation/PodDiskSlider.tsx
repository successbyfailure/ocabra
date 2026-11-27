import { useEffect, useMemo, useState } from "react"
import { api } from "@/api/client"
import type { CloudReplicableModel } from "@/types"

/** Lo que ocupa en el pod lo que no son modelos (sistema, backends, cachés). */
export const POD_BASE_GB = 20
/** Mínimo libre que el pod exige para empezar una descarga (POD_MIN_FREE_GB). */
export const POD_MIN_FREE_GB = 5
const GB = 1024 ** 3
const MIN_DISK = 20
const MAX_DISK = 1000

function modelKey(modelId: string): string {
  return modelId.includes("/") ? modelId.slice(modelId.indexOf("/") + 1) : modelId
}

/** Tamaño de cada modelo; un backend sin desplegar (`ollama/x`) pesa lo del mismo fichero. */
export function modelSizes(modelIds: string[], catalog: CloudReplicableModel[]): number[] {
  const byId = new Map(catalog.map((item) => [item.model_id, item.size_bytes ?? 0]))
  const byKey = new Map<string, number>()
  catalog.forEach((item) => {
    const key = modelKey(item.model_id)
    byKey.set(key, Math.max(byKey.get(key) ?? 0, item.size_bytes ?? 0))
  })
  return modelIds.map((id) => byId.get(id) || byKey.get(modelKey(id)) || 0)
}

/**
 * Disco que se sugiere. Con borrado al terminar cada modelo (baterías de
 * benchmark) basta con que quepan a la vez los dos más grandes, que es lo que
 * el pod descarga en paralelo; si no, todos.
 */
export function suggestedDiskGb(sizes: number[], freesAsItGoes: boolean): number {
  const sorted = [...sizes].sort((a, b) => b - a)
  const needed = (freesAsItGoes ? sorted.slice(0, 2) : sorted).reduce((sum, size) => sum + size, 0)
  const total = POD_BASE_GB + POD_MIN_FREE_GB + (needed / GB) * 1.1
  return Math.min(MAX_DISK, Math.max(40, Math.ceil(total / 10) * 10))
}

/** Deslizable del disco del pod, con lo sugerido para los modelos elegidos. */
export function PodDiskSlider({
  value,
  defaultGb,
  modelIds,
  freesAsItGoes,
  onChange,
}: {
  value: number | null | undefined
  defaultGb: number | null
  modelIds: string[]
  freesAsItGoes: boolean
  onChange: (gb: number | null) => void
}) {
  const [catalog, setCatalog] = useState<CloudReplicableModel[]>([])
  useEffect(() => {
    // Los tamaños son una ayuda: sin ellos el deslizable funciona igual.
    Promise.resolve()
      .then(() => api.federation.cloud.replicableModels())
      .then(setCatalog)
      .catch(() => setCatalog([]))
  }, [])
  const sizes = useMemo(() => modelSizes(modelIds, catalog), [catalog, modelIds])
  const known = sizes.some((size) => size > 0)
  const suggested = known ? suggestedDiskGb(sizes, freesAsItGoes) : null
  const allAtOnce = sizes.reduce((sum, size) => sum + size, 0) / GB
  const current = value ?? defaultGb ?? 80
  const short = suggested != null && current < suggested

  return (
    <div className="text-xs" data-testid="pod-disk">
      <div className="flex items-center justify-between gap-2">
        <span className="font-medium">Disco del pod</span>
        <span className="font-mono">{current} GB{value == null ? " (ajustes)" : ""}</span>
      </div>
      <input
        type="range"
        min={MIN_DISK}
        max={MAX_DISK}
        step={10}
        value={current}
        onChange={(event) => onChange(Number(event.target.value))}
        className="mt-1 w-full"
        aria-label="Disco del pod en GB"
      />
      <div className={`mt-0.5 ${short ? "text-amber-500" : "text-muted-foreground"}`}>
        {suggested != null ? (
          <>
            Sugerido {suggested} GB
            {freesAsItGoes
              ? ` (los dos modelos más grandes a la vez; cada modelo se borra al terminar sus pruebas · todos juntos ocuparían ${allAtOnce.toFixed(0)} GB)`
              : ` (${allAtOnce.toFixed(0)} GB de modelos + sistema)`}
            {short ? " · con menos puede no caber" : ""}
            {current !== suggested && (
              <button type="button" onClick={() => onChange(suggested)} className="ml-2 text-primary hover:underline">Usar {suggested} GB</button>
            )}
          </>
        ) : (
          "Sin tamaño conocido de los modelos elegidos."
        )}
        {value != null && (
          <button type="button" onClick={() => onChange(null)} className="ml-2 text-primary hover:underline">Volver a los ajustes</button>
        )}
      </div>
    </div>
  )
}
