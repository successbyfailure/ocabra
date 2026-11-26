import { useEffect, useState } from "react"
import { api } from "@/api/client"
import type { BenchmarkCloudSpec, CloudGpuType } from "@/types"

/** Valor del selector de nodo que significa «lanzar un pod propio». */
export const CLOUD_NODE = "__cloud__"

export const DEFAULT_CLOUD_SPEC: BenchmarkCloudSpec = {
  gpu_type_ids: [],
  gpu_count: 1,
  cloud_type: null,
  backends: [],
  recipes: [],
  privacy_mode: "functional",
  wait_minutes: 360,
  ready_timeout_minutes: 90,
  max_runtime_minutes: 480,
  terminate_after: true,
}

/** Máquina para un benchmark con pod propio: GPUs por preferencia, backends y apagado. */
export function BenchmarkCloudFields({
  value,
  onChange,
}: {
  value: BenchmarkCloudSpec
  onChange: (next: BenchmarkCloudSpec) => void
}) {
  const [gpus, setGpus] = useState<CloudGpuType[] | null>(null)
  const [backends, setBackends] = useState<string[]>([])

  useEffect(() => {
    api.federation.cloud.gpuTypes().then(setGpus).catch(() => setGpus([]))
    api.backends.list()
      .then((items) => setBackends(items.map((item) => item.backendType).sort()))
      .catch(() => setBackends([]))
  }, [])

  const set = (patch: Partial<BenchmarkCloudSpec>) => onChange({ ...value, ...patch })
  const toggle = (list: string[], item: string) =>
    list.includes(item) ? list.filter((x) => x !== item) : [...list, item]

  return (
    <div className="space-y-3 rounded-lg border border-sky-500/30 bg-sky-500/5 p-3 text-sm" data-testid="benchmark-cloud">
      <p className="text-xs text-muted-foreground">
        Se pide una máquina (reintentando hasta que haya), se instalan los backends y los modelos a medir,
        se mide en ella y, si lo marcas, se apaga al terminar o si algo falla.
      </p>
      <div>
        <span className="font-medium">GPUs aceptables</span>
        <span className="ml-1 text-xs text-muted-foreground">(por orden: se prueba la primera y luego las demás)</span>
        <div className="mt-1 flex max-h-32 flex-wrap gap-1 overflow-y-auto">
          {gpus === null ? <span className="text-xs text-muted-foreground">Cargando…</span> : gpus.map((gpu) => {
            const position = value.gpu_type_ids.indexOf(gpu.id)
            return (
              <button
                key={gpu.id}
                type="button"
                onClick={() => set({ gpu_type_ids: toggle(value.gpu_type_ids, gpu.id) })}
                title={`Disponibilidad ahora: ${gpu.availability}${gpu.memory_gb ? ` · ${gpu.memory_gb} GB` : ""}`}
                className={`rounded-full border px-2 py-0.5 text-xs ${position >= 0 ? "border-primary bg-primary/10 text-primary" : "border-border text-muted-foreground"}`}
              >
                {position >= 0 ? `${position + 1}. ` : ""}{gpu.name}{gpu.memory_gb ? ` · ${gpu.memory_gb} GB` : ""}
              </button>
            )
          })}
        </div>
      </div>
      <div className="grid gap-2 sm:grid-cols-3">
        <label className="text-xs">GPUs por pod
          <input type="number" min={1} max={8} value={value.gpu_count} onChange={(e) => set({ gpu_count: Math.max(1, Math.min(8, Number(e.target.value) || 1)) })} className="mt-1 w-full rounded-md border border-border bg-background px-2 py-1" />
        </label>
        <label className="text-xs">Nube
          <select value={value.cloud_type ?? ""} onChange={(e) => set({ cloud_type: (e.target.value || null) as BenchmarkCloudSpec["cloud_type"] })} className="mt-1 w-full rounded-md border border-border bg-background px-2 py-1">
            <option value="">La de los ajustes</option>
            <option value="COMMUNITY">Community</option>
            <option value="SECURE">Secure</option>
          </select>
        </label>
        <label className="text-xs">Esperar máquina (h)
          <input type="number" min={1} max={168} value={Math.round(value.wait_minutes / 60)} onChange={(e) => set({ wait_minutes: Math.max(1, Number(e.target.value) || 1) * 60 })} className="mt-1 w-full rounded-md border border-border bg-background px-2 py-1" />
        </label>
        <label className="text-xs">Tope para estar listo (min)
          <input type="number" min={5} max={1440} value={value.ready_timeout_minutes} onChange={(e) => set({ ready_timeout_minutes: Math.max(5, Number(e.target.value) || 90) })} className="mt-1 w-full rounded-md border border-border bg-background px-2 py-1" />
        </label>
        <label className="text-xs">Vida máxima del pod (h)
          <input type="number" min={1} max={168} value={Math.round(value.max_runtime_minutes / 60)} onChange={(e) => set({ max_runtime_minutes: Math.max(1, Number(e.target.value) || 8) * 60 })} className="mt-1 w-full rounded-md border border-border bg-background px-2 py-1" />
        </label>
      </div>
      {backends.length > 0 && (
        <div>
          <span className="font-medium">Backends a instalar</span>
          <span className="ml-1 text-xs text-muted-foreground">(los que necesitan los modelos se añaden solos)</span>
          <div className="mt-1 flex flex-wrap gap-1">
            {backends.map((backend) => (
              <button
                key={backend}
                type="button"
                onClick={() => set({ backends: toggle(value.backends, backend) })}
                className={`rounded-full border px-2 py-0.5 font-mono text-xs ${value.backends.includes(backend) ? "border-primary bg-primary/10 text-primary" : "border-border text-muted-foreground"}`}
              >
                {backend}
              </button>
            ))}
          </div>
        </div>
      )}
      <label className="flex items-start gap-2">
        <input type="checkbox" className="mt-1" checked={value.terminate_after} onChange={(e) => set({ terminate_after: e.target.checked })} />
        <span><span className="font-medium">Apagar la instancia al terminar</span><span className="block text-xs text-muted-foreground">También si falla o la cancelas. Sin marcar, el pod queda en marcha (cuenta el tiempo de inactividad y la vida máxima).</span></span>
      </label>
    </div>
  )
}
