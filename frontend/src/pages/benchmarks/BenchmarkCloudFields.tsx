import { useEffect, useMemo, useState } from "react"
import { api } from "@/api/client"
import type { BenchmarkCloudSpec, CloudGpuType } from "@/types"

/** Valor del selector de nodo que significa «lanzar un pod propio». */
export const CLOUD_NODE = "__cloud__"

function offersFor(gpu: CloudGpuType, cloud: BenchmarkCloudSpec["cloud_type"]) {
  const offers: string[] = []
  if (cloud !== "SECURE" && gpu.community && gpu.price_community != null)
    offers.push(`${cloud === "BOTH" ? "Community " : ""}${gpu.price_community.toFixed(2)} $/h`)
  if (cloud !== "COMMUNITY" && gpu.secure && gpu.price_secure != null)
    offers.push(`${cloud === "BOTH" ? "Secure " : ""}${gpu.price_secure.toFixed(2)} $/h`)
  return offers
}

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
  const [minVram, setMinVram] = useState("")
  const [defaultCloud, setDefaultCloud] = useState<"COMMUNITY" | "SECURE" | null>(null)

  useEffect(() => {
    api.federation.cloud.gpuTypes().then(setGpus).catch(() => setGpus([]))
    api.federation.cloud.getRunpod()
      .then((info) => setDefaultCloud(info.settings.cloud_type))
      .catch(() => setDefaultCloud("COMMUNITY"))
    api.backends.list()
      .then((items) => setBackends(items.map((item) => item.backendType).sort()))
      .catch(() => setBackends([]))
  }, [])

  // Solo lo que RunPod ofrece ahora en esa nube (precio y stock), como al lanzar un nodo.
  const secureOnly = Boolean(value.volume_id) || value.privacy_mode === "zero_data"
  const cloud = secureOnly ? "SECURE" : value.cloud_type ?? defaultCloud
  const offered = useMemo(
    () =>
      (gpus ?? []).filter((gpu) =>
        offersFor(gpu, cloud).length > 0 &&
        (minVram === "" || (gpu.memory_gb != null && gpu.memory_gb > Number(minVram))) &&
        gpu.availability !== "NONE",
      ),
    [gpus, cloud, minVram],
  )
  // Al cambiar de nube se caen las elegidas que esa nube no ofrece.
  useEffect(() => {
    if (gpus === null || cloud === null) return
    const ids = new Set(offered.map((gpu) => gpu.id))
    const kept = value.gpu_type_ids.filter((id) => ids.has(id))
    if (kept.length !== value.gpu_type_ids.length) onChange({ ...value, gpu_type_ids: kept })
  }, [offered]) // eslint-disable-line react-hooks/exhaustive-deps

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
        <label className="mt-2 flex items-center gap-2 text-xs">
          VRAM por GPU mayor que (GB)
          <input type="number" min="0" step="any" placeholder="Sin filtro" value={minVram}
            onChange={(e) => setMinVram(e.target.value)}
            className="w-24 rounded-md border border-border bg-background px-2 py-1" />
        </label>
        <div className="mt-1 flex max-h-32 flex-wrap gap-1 overflow-y-auto">
          {gpus === null || cloud === null ? <span className="text-xs text-muted-foreground">Cargando…</span> : offered.length === 0 ? <span className="text-xs text-muted-foreground">Ninguna GPU disponible cumple los filtros.</span> : offered.map((gpu) => {
            const prices = offersFor(gpu, cloud).join(" / ")
            const position = value.gpu_type_ids.indexOf(gpu.id)
            return (
              <button
                key={gpu.id}
                type="button"
                onClick={() => set({ gpu_type_ids: toggle(value.gpu_type_ids, gpu.id) })}
                title={`Disponibilidad ahora: ${gpu.availability} · ${prices}`}
                className={`rounded-full border px-2 py-0.5 text-xs ${position >= 0 ? "border-primary bg-primary/10 text-primary" : "border-border text-muted-foreground"}`}
              >
                {position >= 0 ? `${position + 1}. ` : ""}{gpu.name}{gpu.memory_gb ? ` · ${gpu.memory_gb} GB` : ""} · {prices}
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
          <select disabled={secureOnly} value={secureOnly ? "SECURE" : value.cloud_type ?? ""} onChange={(e) => set({ cloud_type: (e.target.value || null) as BenchmarkCloudSpec["cloud_type"] })} className="mt-1 w-full rounded-md border border-border bg-background px-2 py-1">
            <option value="">La de los ajustes ({defaultCloud === "COMMUNITY" ? "Community" : "Secure"})</option>
            <option value="COMMUNITY">Community</option>
            <option value="SECURE">Secure</option>
            <option value="BOTH">Community y Secure</option>
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
