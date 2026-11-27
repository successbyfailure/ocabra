import { useCallback, useEffect, useMemo, useState } from "react"
import * as Dialog from "@radix-ui/react-dialog"
import { AlertTriangle, ArrowUpCircle, Ban, Clock, Cloud, Cpu, KeyRound, Loader2, PackagePlus, Power, RefreshCw, Rocket, Settings2, Trash2 } from "lucide-react"
import { toast } from "sonner"
import { api, request } from "@/api/client"
import { formatSeconds, podReadinessParts } from "@/lib/format"
import { VolumesPanel } from "./VolumesPanel"
import { EMPTY_PICK, ModelPicker, pickCount } from "./ModelPicker"
import { PodDiskSlider } from "./PodDiskSlider"
import type {
  CloudDataCenter,
  CloudLaunchRequest,
  CloudGpuType,
  CloudVolume,
  CloudNode,
  CloudPreloadItem,
  CloudPreloadPick,
  CloudProviderInfo,
  CloudSettings,
} from "@/types"


function AddModelsDialog({
  node,
  onOpenChange,
  onDone,
}: {
  node: CloudNode | null
  onOpenChange: (v: boolean) => void
  onDone: () => void
}) {
  const [picked, setPicked] = useState<CloudPreloadPick>(EMPTY_PICK)
  const [busy, setBusy] = useState(false)
  useEffect(() => {
    if (node) setPicked(EMPTY_PICK)
  }, [node])
  return (
    <Dialog.Root open={node !== null} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-40 bg-black/60" />
        <Dialog.Content className="fixed left-1/2 top-1/2 z-50 max-h-[90vh] w-[95vw] max-w-xl -translate-x-1/2 -translate-y-1/2 overflow-y-auto rounded-lg border border-border bg-card p-5">
          <Dialog.Title className="text-lg font-semibold">Llevar modelos a {node?.name}</Dialog.Title>
          <Dialog.Description className="mt-1 text-sm text-muted-foreground">
            Se descargan en el nodo y se cargan (o solo se descargan), con la configuración que tenían.
          </Dialog.Description>
          <div className="mt-4">
            <ModelPicker
              value={picked}
              onChange={setPicked}
              exclude={(node?.preload_models ?? []).filter((p) => p.status !== "error").map((p) => p.model_id)}
            />
          </div>
          <div className="mt-4 flex justify-end gap-2">
            <Dialog.Close className={btnCls}>Cancelar</Dialog.Close>
            <button
              type="button"
              className={primaryBtnCls}
              disabled={busy || pickCount(picked) === 0 || !node}
              onClick={async () => {
                if (!node) return
                setBusy(true)
                try {
                  await api.federation.cloud.addModels(node.id, picked.models, {
                    recipes: picked.recipes,
                    download_only: picked.downloadOnly,
                    backends: picked.backends,
                  })
                  toast.success("Modelos en cola en el nodo")
                  onOpenChange(false)
                  onDone()
                } catch (err) {
                  toast.error(errMsg(err))
                } finally {
                  setBusy(false)
                }
              }}
            >
              {busy ? <Loader2 size={14} className="animate-spin" /> : <PackagePlus size={14} />} Llevar
            </button>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  )
}

/**
 * Nube · RunPod. Un nodo en la nube es un pod con la imagen ocabra-node que se
 * une a esta federación con una invitación de un solo uso: no abre puertos.
 * El dinero es el riesgo, así que todo lo que se lanza tiene TTL, apagado por
 * inactividad y tope de gasto mensual (los aplica el backend, aquí se ven).
 */

const POLL_MS = 10_000
const inputCls = "mt-1 w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
const btnCls =
  "inline-flex items-center gap-1.5 rounded-md border border-border px-2.5 py-1.5 text-xs hover:bg-muted disabled:opacity-50"
const primaryBtnCls =
  "inline-flex items-center gap-1.5 rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground disabled:opacity-50"

const usd = (v: number | null | undefined, digits = 2) => (v == null ? "—" : `$${v.toFixed(digits)}`)

function errMsg(err: unknown): string {
  return err instanceof Error ? err.message : "Error"
}

function minutesLabel(m: number | null | undefined): string {
  if (!m) return "sin límite"
  if (m < 60) return `${m} min`
  const h = m / 60
  return Number.isInteger(h) ? `${h} h` : `${h.toFixed(1)} h`
}

function untilLabel(iso: string | null): string {
  if (!iso) return "—"
  const ms = Date.parse(iso) - Date.now()
  if (Number.isNaN(ms)) return "—"
  if (ms <= 0) return "ya"
  const m = Math.round(ms / 60000)
  return m < 60 ? `en ${m} min` : `en ${Math.floor(m / 60)} h ${m % 60} min`
}

const REASONS: Record<string, string> = {
  manual: "a mano",
  ttl: "tiempo máximo",
  idle: "inactividad",
  budget: "presupuesto",
  join_timeout: "no llegó a unirse",
  exited: "se paró en RunPod",
  error: "error en RunPod",
  gone: "ya no existía",
  provider: "lo terminó RunPod",
  gpu_broken: "GPU averiada en el pod",
}

const STATUS: Record<CloudNode["status"], { cls: string; label: string }> = {
  creating: { cls: "border-sky-500/30 bg-sky-500/15 text-sky-700 dark:text-sky-300", label: "Creando" },
  starting: { cls: "border-sky-500/30 bg-sky-500/15 text-sky-700 dark:text-sky-300", label: "Arrancando" },
  running: { cls: "border-emerald-500/30 bg-emerald-500/15 text-emerald-600 dark:text-emerald-300", label: "En marcha" },
  stopping: { cls: "border-amber-500/30 bg-amber-500/15 text-amber-700 dark:text-amber-300", label: "Parando" },
  terminated: { cls: "border-border bg-muted text-muted-foreground", label: "Terminado" },
  error: { cls: "border-red-500/30 bg-red-500/15 text-red-600 dark:text-red-300", label: "Error" },
}

function ApiKeyForm({ onSaved }: { onSaved: () => void }) {
  const [key, setKey] = useState("")
  const [busy, setBusy] = useState(false)
  return (
    <div className="flex flex-wrap items-end gap-2">
      <label className="block min-w-64 flex-1 text-sm text-muted-foreground">
        API key de RunPod
        <input
          className={inputCls}
          type="password"
          placeholder="rpa_…"
          value={key}
          onChange={(e) => setKey(e.target.value)}
        />
      </label>
      <button
        type="button"
        className={primaryBtnCls}
        disabled={busy || !key.trim()}
        onClick={async () => {
          setBusy(true)
          try {
            await api.federation.cloud.updateRunpod({ api_key: key.trim() })
            toast.success("API key comprobada y guardada")
            setKey("")
            onSaved()
          } catch (err) {
            toast.error(errMsg(err))
          } finally {
            setBusy(false)
          }
        }}
      >
        {busy ? <Loader2 size={14} className="animate-spin" /> : <KeyRound size={14} />} Guardar
      </button>
      <p className="w-full text-xs text-muted-foreground">
        Se comprueba contra RunPod antes de guardarla y se guarda cifrada. Créala en RunPod → Settings → API Keys.
      </p>
    </div>
  )
}

function SettingsDialog({
  open,
  onOpenChange,
  info,
  onSaved,
}: {
  open: boolean
  onOpenChange: (v: boolean) => void
  info: CloudProviderInfo
  onSaved: () => void
}) {
  const [form, setForm] = useState<CloudSettings>(info.settings)
  const [busy, setBusy] = useState(false)
  useEffect(() => {
    if (open) setForm(info.settings)
  }, [open, info.settings])

  const num = (k: keyof CloudSettings) => (e: React.ChangeEvent<HTMLInputElement>) =>
    setForm({ ...form, [k]: Number(e.target.value) })

  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-40 bg-black/60" />
        <Dialog.Content className="fixed left-1/2 top-1/2 z-50 max-h-[90vh] w-[95vw] max-w-lg -translate-x-1/2 -translate-y-1/2 overflow-y-auto rounded-lg border border-border bg-card p-5">
          <Dialog.Title className="text-lg font-semibold">Ajustes de RunPod</Dialog.Title>
          <div className="mt-4 grid grid-cols-2 gap-3">
            <label className="col-span-2 block text-sm text-muted-foreground">
              Imagen
              <input className={inputCls} value={form.image} onChange={(e) => setForm({ ...form, image: e.target.value })} />
            </label>
            <label className="block text-sm text-muted-foreground">
              Tope de gasto al mes ($)
              <input className={inputCls} type="number" min={0} step={1} value={form.monthly_budget_usd} onChange={num("monthly_budget_usd")} />
            </label>
            <label className="block text-sm text-muted-foreground">
              Nube por defecto
              <select
                className={inputCls}
                value={form.cloud_type}
                onChange={(e) => setForm({ ...form, cloud_type: e.target.value as CloudSettings["cloud_type"] })}
              >
                <option value="COMMUNITY">Community (barata, hosts de terceros)</option>
                <option value="SECURE">Secure (centros de datos)</option>
              </select>
            </label>
            <label className="block text-sm text-muted-foreground">
              Tiempo máximo (min)
              <input className={inputCls} type="number" min={0} value={form.default_ttl_minutes} onChange={num("default_ttl_minutes")} />
            </label>
            <label className="block text-sm text-muted-foreground">
              Apagar si inactivo (min)
              <input className={inputCls} type="number" min={0} value={form.default_idle_minutes} onChange={num("default_idle_minutes")} />
            </label>
            <label className="block text-sm text-muted-foreground">
              Disco del contenedor (GB)
              <input className={inputCls} type="number" min={20} value={form.container_disk_gb} onChange={num("container_disk_gb")} />
            </label>
            <label className="block text-sm text-muted-foreground">
              URL a la que se conectan
              <input
                className={inputCls}
                placeholder="(PUBLIC_URL)"
                value={form.hub_url}
                onChange={(e) => setForm({ ...form, hub_url: e.target.value })}
              />
            </label>
            <label className="col-span-2 flex items-start gap-2 text-sm">
              <input
                type="checkbox"
                className="mt-1"
                checked={form.pass_hf_token}
                onChange={(e) => setForm({ ...form, pass_hf_token: e.target.checked })}
              />
              <span>
                Pasar el HF_TOKEN a los pods
                <span className="block text-xs text-muted-foreground">
                  Necesario para modelos con acceso restringido en Hugging Face. El token viaja a una máquina de terceros.
                </span>
              </span>
            </label>
          </div>
          <div className="mt-4 flex justify-end gap-2">
            <Dialog.Close className={btnCls}>Cancelar</Dialog.Close>
            <button
              type="button"
              className={primaryBtnCls}
              disabled={busy}
              onClick={async () => {
                setBusy(true)
                try {
                  await api.federation.cloud.updateRunpod({ settings: form })
                  toast.success("Ajustes guardados")
                  onOpenChange(false)
                  onSaved()
                } catch (err) {
                  toast.error(errMsg(err))
                } finally {
                  setBusy(false)
                }
              }}
            >
              Guardar
            </button>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  )
}

/** «Relanzar»: an earlier node's setup, to launch the same again. */
export interface LaunchPreset {
  nodeName: string
  gpuId: string
  count: number
  cloudType: "COMMUNITY" | "SECURE"
  volumeId: string
  zeroData: boolean
  pick: CloudPreloadPick
}

export function presetFromNode(node: CloudNode): LaunchPreset {
  const items = node.preload_models ?? []
  const key = (i: CloudPreloadItem) => i.recipe_id ?? i.model_id
  return {
    nodeName: node.name,
    gpuId: node.gpu_type_id,
    count: node.gpu_count,
    cloudType: node.cloud_type === "SECURE" ? "SECURE" : "COMMUNITY",
    volumeId: node.network_volume_id ?? "",
    zeroData: node.privacy_mode === "zero_data",
    pick: {
      models: items.filter((i) => !i.recipe_id && i.kind !== "backend").map((i) => i.model_id),
      recipes: items.filter((i) => i.recipe_id && i.kind !== "backend").map((i) => i.recipe_id as string),
      downloadOnly: items.filter((i) => i.load === false && i.kind !== "backend").map(key),
      backends: items.filter((i) => i.kind === "backend").map((i) => i.model_id.replace(/^backend:/, "")),
    },
  }
}

function LaunchDialog({
  open,
  onOpenChange,
  info,
  onLaunched,
  preset,
}: {
  open: boolean
  onOpenChange: (v: boolean) => void
  info: CloudProviderInfo
  onLaunched: () => void
  preset?: LaunchPreset | null
}) {
  const [gpus, setGpus] = useState<CloudGpuType[] | null>(null)
  const [gpuId, setGpuId] = useState("")
  const [cloudType, setCloudType] = useState<"COMMUNITY" | "SECURE">(info.settings.cloud_type)
  const [count, setCount] = useState(1)
  const [ttl, setTtl] = useState(info.settings.default_ttl_minutes)
  const [idle, setIdle] = useState(info.settings.default_idle_minutes)
  const [name, setName] = useState("")
  const [busy, setBusy] = useState(false)
  const [picked, setPicked] = useState<CloudPreloadPick>(EMPTY_PICK)
  const [volumes, setVolumes] = useState<CloudVolume[]>([])
  const [dataCenters, setDataCenters] = useState<CloudDataCenter[]>([])
  const [volumeId, setVolumeId] = useState("")
  const [zeroData, setZeroData] = useState(false)
  // Si no hay máquina libre: dejar una petición que se reintenta sola.
  const [queueIfBusy, setQueueIfBusy] = useState(false)
  const [alternatives, setAlternatives] = useState<string[]>([])
  const [waitHours, setWaitHours] = useState(6)
  const [diskGb, setDiskGb] = useState<number | null>(null)

  useEffect(() => {
    if (!open) return
    setPicked(preset?.pick ?? EMPTY_PICK)
    setVolumeId(preset?.volumeId ?? "")
    setZeroData(preset?.zeroData ?? false)
    api.federation.cloud.volumes().then(setVolumes).catch(() => setVolumes([]))
    api.federation.cloud.dataCenters().then(setDataCenters).catch(() => setDataCenters([]))
    setCloudType(preset?.cloudType ?? info.settings.cloud_type)
    setTtl(info.settings.default_ttl_minutes)
    setDiskGb(null)
    setIdle(info.settings.default_idle_minutes)
    setCount(preset?.count ?? 1)
    setGpuId(preset?.gpuId ?? "")
    setName("")
    setGpus(null)
    api.federation.cloud
      .gpuTypes()
      .then(setGpus)
      .catch((err) => {
        toast.error(errMsg(err))
        setGpus([])
      })
    // Only when the dialog opens: the section polls every 5 s and hands a new
    // `info` object each time, which used to reset the form under the user.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open])

  // A volume forces Secure Cloud (the only one with network volumes).
  const priceOf = useCallback(
    (g: CloudGpuType) =>
      !volumeId && !zeroData && cloudType === "COMMUNITY" ? g.price_community : g.price_secure,
    [cloudType, volumeId, zeroData],
  )
  const volume = volumes.find((v) => v.id === volumeId)
  // A volume pins the pod to its data center: offer only what is in stock there.
  const dcGpus = useMemo(() => {
    if (!volume) return null
    const dc = dataCenters.find((d) => d.id === volume.data_center)
    return new Set((dc?.gpus ?? []).map((g) => g.id))
  }, [volume, dataCenters])
  // A volume or zero data forces Secure Cloud.
  const effectiveCloud = volume || zeroData ? "SECURE" : cloudType
  const offered = useMemo(
    () =>
      (gpus ?? []).filter(
        (g) =>
          (effectiveCloud === "COMMUNITY" ? g.community : g.secure) &&
          (effectiveCloud === "COMMUNITY" ? g.price_community : g.price_secure) != null &&
          // Encolando se puede pedir una GPU agotada ahora mismo: se esperará.
          (queueIfBusy || g.availability !== "NONE") &&
          (!dcGpus || dcGpus.has(g.id)),
      ),
    [gpus, effectiveCloud, dcGpus, queueIfBusy],
  )
  const chosen = offered.find((g) => g.id === gpuId)
  const hourly = chosen ? (priceOf(chosen) ?? 0) * count : 0
  const budgetLeft = info.settings.monthly_budget_usd - (info.month_spend_usd ?? 0)

  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-40 bg-black/60" />
        <Dialog.Content className="fixed left-1/2 top-1/2 z-50 max-h-[90vh] w-[95vw] max-w-2xl -translate-x-1/2 -translate-y-1/2 overflow-y-auto rounded-lg border border-border bg-card p-5">
          <Dialog.Title className="text-lg font-semibold">
            {preset ? `Relanzar como ${preset.nodeName}` : "Lanzar nodo en RunPod"}
          </Dialog.Title>
          <Dialog.Description className="mt-1 text-sm text-muted-foreground">
            Se une solo a esta federación en modo enjambre y descarga y carga los modelos que elijas.
          </Dialog.Description>

          {volumes.length > 0 && (
            <label className="mt-4 block text-sm">
              <span className="text-xs text-muted-foreground">Disco de modelos</span>
              <select
                className="mt-0.5 w-full rounded-md border border-border bg-background px-2 py-1.5 text-sm"
                value={volumeId}
                onChange={(e) => {
                  setVolumeId(e.target.value)
                  setGpuId("")
                }}
              >
                <option value="">Ninguno: el nodo descarga todo en su disco temporal</option>
                {volumes.map((v) => (
                  <option key={v.id} value={v.id} disabled={Boolean(v.in_use_by)}>
                    {v.name} · {v.size_gb} GB · {v.data_center}
                    {v.in_use_by ? ` (en uso por ${v.in_use_by})` : ""}
                  </option>
                ))}
              </select>
              {volume && (
                <span className="text-xs text-muted-foreground">
                  El pod irá a {volume.data_center} en Secure Cloud; solo se ofrecen las GPUs con stock allí.
                </span>
              )}
            </label>
          )}

          <label className="mt-4 flex items-start gap-2 text-sm">
            <input
              type="checkbox"
              className="mt-0.5 accent-primary"
              checked={zeroData}
              onChange={(e) => setZeroData(e.target.checked)}
            />
            <span>
              <span className="font-medium">Modo 0 data</span>
              <span className="block text-xs text-muted-foreground">
                El nodo no guarda nada de las peticiones ni de quién las hace, ni recibe tu token de Hugging
                Face. Va a Secure Cloud. No protege frente al dueño de la máquina.
              </span>
            </span>
          </label>

          <div className="mt-4 flex gap-2">
            {(["COMMUNITY", "SECURE"] as const).map((t) => (
              <button
                key={t}
                type="button"
                className={`${btnCls} ${effectiveCloud === t ? "border-primary bg-primary/10" : ""}`}
                disabled={(Boolean(volume) || zeroData) && t === "COMMUNITY"}
                onClick={() => setCloudType(t)}
              >
                {t === "COMMUNITY" ? "Community" : "Secure"}
              </button>
            ))}
            {effectiveCloud === "COMMUNITY" && (
              <span className="self-center text-xs text-muted-foreground">
                Hosts de particulares: tus peticiones pasan por máquinas de terceros.
              </span>
            )}
          </div>

          <div className="mt-3 max-h-64 overflow-y-auto rounded-md border border-border">
            {gpus == null ? (
              <div className="flex items-center gap-2 p-4 text-sm text-muted-foreground">
                <Loader2 size={14} className="animate-spin" /> Consultando el catálogo…
              </div>
            ) : offered.length === 0 ? (
              <div className="p-4 text-sm text-muted-foreground">No hay GPUs disponibles ahora mismo en esta nube.</div>
            ) : (
              <table className="w-full text-sm">
                <thead className="sticky top-0 bg-card text-xs text-muted-foreground">
                  <tr>
                    <th className="px-3 py-1.5 text-left font-medium">GPU</th>
                    <th className="px-3 py-1.5 text-right font-medium">VRAM</th>
                    <th className="px-3 py-1.5 text-right font-medium">$/h</th>
                    <th className="px-3 py-1.5 text-left font-medium">Stock</th>
                  </tr>
                </thead>
                <tbody>
                  {offered.map((g) => (
                    <tr
                      key={g.id}
                      className={`cursor-pointer border-t border-border hover:bg-muted ${gpuId === g.id ? "bg-primary/10" : ""}`}
                      onClick={() => setGpuId(g.id)}
                    >
                      <td className="px-3 py-1.5">{g.name}</td>
                      <td className="px-3 py-1.5 text-right tabular-nums">{g.memory_gb ?? "?"} GB</td>
                      <td className="px-3 py-1.5 text-right tabular-nums">{usd(priceOf(g))}</td>
                      <td className="px-3 py-1.5 text-xs text-muted-foreground">{g.availability.toLowerCase()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>

          <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
            <label className="block text-sm text-muted-foreground">
              GPUs
              <input className={inputCls} type="number" min={1} max={8} value={count} onChange={(e) => setCount(Math.max(1, Number(e.target.value) || 1))} />
            </label>
            <label className="block text-sm text-muted-foreground">
              Máximo (min)
              <input className={inputCls} type="number" min={0} value={ttl} onChange={(e) => setTtl(Number(e.target.value))} />
            </label>
            <label className="block text-sm text-muted-foreground">
              Inactivo (min)
              <input className={inputCls} type="number" min={0} value={idle} onChange={(e) => setIdle(Number(e.target.value))} />
            </label>
            <label className="block text-sm text-muted-foreground">
              Nombre
              <input className={inputCls} placeholder="opcional" value={name} onChange={(e) => setName(e.target.value)} />
            </label>
          </div>

          <div className="mt-3">
            <ModelPicker
              value={picked}
              onChange={setPicked}
              gpuMemoryGb={chosen?.memory_gb != null ? chosen.memory_gb * count : null}
              autoRecipesFrom={open ? (preset?.nodeName ?? null) : null}
            />
          </div>

          <div className="mt-3 rounded-md border border-border px-3 py-2 text-sm">
            <label className="flex items-start gap-2">
              <input type="checkbox" className="mt-1" checked={queueIfBusy} onChange={(e) => setQueueIfBusy(e.target.checked)} />
              <span>
                <span className="font-medium">Si no hay máquina libre, seguir intentándolo</span>
                <span className="block text-xs text-muted-foreground">Deja una petición que se reintenta cada 2 minutos con esta GPU y las alternativas, por orden, hasta que haya una o pase el plazo.</span>
              </span>
            </label>
            {queueIfBusy && (
              <div className="mt-2 space-y-2 pl-6">
                <div className="flex flex-wrap gap-1">
                  {offered.filter((g) => g.id !== gpuId).map((g) => {
                    const on = alternatives.includes(g.id)
                    return (
                      <button
                        key={g.id}
                        type="button"
                        onClick={() => setAlternatives((cur) => (on ? cur.filter((x) => x !== g.id) : [...cur, g.id]))}
                        className={`rounded-full border px-2 py-0.5 text-xs ${on ? "border-primary bg-primary/10 text-primary" : "border-border text-muted-foreground"}`}
                        title={`Disponibilidad ahora: ${g.availability}`}
                      >
                        {on ? `${alternatives.indexOf(g.id) + 2}. ` : ""}{g.name}
                      </button>
                    )
                  })}
                </div>
                <label className="flex items-center gap-2 text-xs">
                  Esperar como mucho
                  <input type="number" min={1} max={168} value={waitHours} onChange={(e) => setWaitHours(Math.max(1, Math.min(168, Number(e.target.value) || 1)))} className="w-16 rounded border border-border bg-background px-2 py-0.5" />
                  horas
                </label>
              </div>
            )}
          </div>

          <div className="mt-3">
            <PodDiskSlider
              value={diskGb}
              defaultGb={info.settings.container_disk_gb}
              modelIds={picked.models}
              freesAsItGoes={false}
              onChange={setDiskGb}
            />
          </div>

          <div className="mt-3 rounded-md border border-border bg-muted/30 px-3 py-2 text-sm">
            {chosen ? (
              <>
                <span className="font-medium">{usd(hourly, 3)}/h</span>
                {ttl > 0 && <span className="text-muted-foreground"> · como mucho {usd(hourly * (ttl / 60))} si dura {minutesLabel(ttl)}</span>}
                {idle > 0 && <span className="text-muted-foreground"> · se apaga tras {minutesLabel(idle)} sin uso</span>}
              </>
            ) : (
              <span className="text-muted-foreground">Elige una GPU.</span>
            )}
            <div className="text-xs text-muted-foreground">
              Te quedan {usd(Math.max(0, budgetLeft))} de {usd(info.settings.monthly_budget_usd)} este mes.
            </div>
          </div>

          <div className="mt-4 flex justify-end gap-2">
            <Dialog.Close className={btnCls}>Cancelar</Dialog.Close>
            <button
              type="button"
              className={primaryBtnCls}
              disabled={busy || !chosen}
              onClick={async () => {
                setBusy(true)
                try {
                  const spec = {
                    gpu_type_id: gpuId,
                    gpu_count: count,
                    cloud_type: effectiveCloud,
                    volume_id: volumeId || undefined,
                    privacy_mode: zeroData ? "zero_data" : "functional",
                    max_runtime_minutes: ttl,
                    idle_shutdown_minutes: idle,
                    name: name.trim() || undefined,
                    models: picked.models,
                    recipes: picked.recipes,
                    download_only: picked.downloadOnly,
                    backends: picked.backends,
                    container_disk_gb: diskGb ?? undefined,
                  } as const
                  if (queueIfBusy) {
                    const req = await api.federation.cloud.requests.create({
                      ...spec,
                      alternatives: alternatives.filter((g) => g !== gpuId),
                      wait_minutes: waitHours * 60,
                    })
                    if (req.status === "launched") toast.success(`Lanzado con ${req.launchedGpuTypeId}: tardará unos minutos en unirse`)
                    else if (req.status === "waiting") toast.info("No hay máquina libre: la petición queda en espera y se reintentará sola")
                    else toast.error(req.lastError ?? "La petición no salió")
                  } else {
                    const node = await api.federation.cloud.launch(spec)
                    toast.success(`${node.name} lanzado: tardará unos minutos en unirse`)
                  }
                  onOpenChange(false)
                  onLaunched()
                } catch (err) {
                  toast.error(errMsg(err))
                } finally {
                  setBusy(false)
                }
              }}
            >
              {busy ? <Loader2 size={14} className="animate-spin" /> : <Rocket size={14} />} {queueIfBusy ? "Pedir máquina" : "Lanzar"}
            </button>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  )
}

function PreloadChip({ item, seconds }: { item: CloudPreloadItem; seconds?: number | null }) {
  const pct = item.progress != null ? ` ${Math.round(item.progress)} %` : ""
  const took = seconds != null && (item.status === "loaded" || item.status === "ready" || item.status === "error") ? ` en ${formatSeconds(seconds)}` : ""
  const steps = item.steps?.length ? ` (paso ${(item.step ?? 0) + 1}/${item.steps.length})` : ""
  const label = {
    pending: item.waiting_disk ? "esperando disco" : "en cola",
    installing: `instalando backend${pct}`,
    registering: "registrando",
    downloading: `descargando${pct}${steps}`,
    loading: "cargando",
    loaded: `listo${took}`,
    ready: `${item.kind === "backend" ? "instalado" : "descargado"}${took}`,
    error: `error${took}`,
    removed: "medido y borrado",
  }[item.status]
  const cls = {
    pending: "border-border text-muted-foreground",
    installing: "border-sky-500/40 text-sky-700 dark:text-sky-300",
    registering: "border-sky-500/40 text-sky-700 dark:text-sky-300",
    downloading: "border-sky-500/40 text-sky-700 dark:text-sky-300",
    loading: "border-sky-500/40 text-sky-700 dark:text-sky-300",
    loaded: "border-emerald-500/40 text-emerald-600 dark:text-emerald-300",
    ready: "border-emerald-500/30 text-muted-foreground",
    error: "border-red-500/40 text-red-600 dark:text-red-300",
    removed: "border-border text-muted-foreground line-through",
  }[item.status]
  return (
    <span title={item.error ?? item.phase ?? undefined} className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] ${cls}`}>
      {item.status !== "pending" && item.status !== "loaded" && item.status !== "ready" && item.status !== "error" && item.status !== "removed" && (
        <Loader2 size={10} className="animate-spin" />
      )}
      <span className="font-mono">
        {item.kind === "backend" ? `backend ${item.model_id.replace(/^backend:/, "")}` : item.model_id.split("/").pop()}
      </span>{" "}
      · {label}
    </span>
  )
}

function NodeRow({
  node,
  onChanged,
  onAddModels,
  onRelaunch,
}: {
  node: CloudNode
  onChanged: () => void
  onAddModels?: (node: CloudNode) => void
  onRelaunch?: (node: CloudNode) => void
}) {
  const [busy, setBusy] = useState(false)
  const st = STATUS[node.status] ?? STATUS.error
  const live = node.status !== "terminated"
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 px-4 py-2.5 text-sm">
      <Cpu size={14} className="text-muted-foreground" />
      <span className="font-medium">{node.name}</span>
      <span className={`inline-flex rounded-full border px-2 py-0.5 text-[11px] font-medium ${st.cls}`}>{st.label}</span>
      {live && (
        <span className={`text-xs ${node.peer_online ? "text-emerald-600 dark:text-emerald-300" : "text-muted-foreground"}`}>
          {node.peer_id ? (node.peer_online ? "en la federación" : "unido, sin conexión") : "esperando a que se una"}
        </span>
      )}
      <span className="text-xs text-muted-foreground">
        {node.gpu_count > 1 ? `${node.gpu_count}× ` : ""}
        {node.gpu_type_id} · {node.cloud_type.toLowerCase()} · {usd(node.cost_per_hr, 3)}/h · gastado {usd(node.cost_so_far)}
        {live && node.terminates_at ? ` · se apaga ${untilLabel(node.terminates_at)}` : ""}
        {live && node.idle_shutdown_minutes ? ` · o tras ${minutesLabel(node.idle_shutdown_minutes)} inactivo` : ""}
        {node.requests_proxied ? ` · ${node.requests_proxied} peticiones` : ""}
        {!live && node.terminate_reason ? ` · terminado por ${REASONS[node.terminate_reason] ?? node.terminate_reason}` : ""}
      </span>
      {podReadinessParts(node.durations).length > 0 && (
        <span className="w-full text-xs text-muted-foreground" title="Tiempos medidos cada ~30 s">
          ⏱ {podReadinessParts(node.durations).join(" · ")}
        </span>
      )}
      {node.preload_models.length > 0 && (
        <div className="flex w-full flex-wrap gap-1.5">
          {node.preload_models.map((p) => (
            <PreloadChip key={p.model_id} item={p} seconds={node.durations?.items?.[p.model_id]} />
          ))}
        </div>
      )}
      {node.error && <span className="w-full text-xs text-red-600 dark:text-red-300">{node.error}</span>}
      {!live && onRelaunch && (
        <button
          type="button"
          className={`${btnCls} ml-auto`}
          title="Misma GPU, disco, privacidad y modelos (también los que se le pusieron a mano)"
          onClick={() => onRelaunch(node)}
        >
          <RefreshCw size={12} /> Relanzar
        </button>
      )}
      {live && onAddModels && (
        <button type="button" className={`${btnCls} ml-auto`} onClick={() => onAddModels(node)}>
          <PackagePlus size={12} /> Llevar modelos
        </button>
      )}
      {live && (
        <button
          type="button"
          className={`${btnCls} ${onAddModels ? "" : "ml-auto"}`}
          disabled={busy}
          title={`Imagen actual: ${node.image}`}
          data-testid="cloud-node-update"
          onClick={async () => {
            const version = window.prompt(
              `Reiniciar ${node.name} con otra imagen de ocabra-node.\n\nMisma máquina y GPU, pero se borra el disco del contenedor: vuelve a unirse como un nodo nuevo y se recargan sus modelos (los del disco de red se conservan).\n\nVersión (X.Y.Z), o vacío para la imagen configurada:`,
              "",
            )
            if (version === null) return
            if (version && !/^\d+\.\d+\.\d+$/.test(version.trim())) {
              toast.error("La versión tiene que ser X.Y.Z")
              return
            }
            setBusy(true)
            try {
              await request("POST", `/ocabra/cloud/nodes/${encodeURIComponent(node.id)}/update-image`, {
                version: version.trim() || null,
              })
              toast.success("Reiniciando con la imagen nueva; volverá a unirse en unos minutos")
              onChanged()
            } catch (err) {
              toast.error(errMsg(err))
            } finally {
              setBusy(false)
            }
          }}
        >
          <ArrowUpCircle size={12} /> Actualizar imagen
        </button>
      )}
      {live && (
        <button
          type="button"
          className={`${btnCls} text-red-600 dark:text-red-300`}
          disabled={busy}
          onClick={async () => {
            if (!window.confirm(`¿Terminar ${node.name}? Deja de cobrarse y sale de la federación.`)) return
            setBusy(true)
            try {
              await api.federation.cloud.terminate(node.id)
              toast.success("Nodo terminado")
              onChanged()
            } catch (err) {
              toast.error(errMsg(err))
            } finally {
              setBusy(false)
            }
          }}
        >
          {busy ? <Loader2 size={12} className="animate-spin" /> : <Power size={12} />} Terminar
        </button>
      )}
    </div>
  )
}

export function CloudSection() {
  const [info, setInfo] = useState<CloudProviderInfo | null>(null)
  const [nodes, setNodes] = useState<CloudNode[]>([])
  const [launchOpen, setLaunchOpen] = useState(false)
  const [launchPreset, setLaunchPreset] = useState<LaunchPreset | null>(null)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [showOld, setShowOld] = useState(false)
  const [addingTo, setAddingTo] = useState<CloudNode | null>(null)
  const [requests, setRequests] = useState<CloudLaunchRequest[]>([])

  const load = useCallback(async () => {
    try {
      const [i, n, r] = await Promise.all([
        api.federation.cloud.getRunpod(),
        api.federation.cloud.nodes(),
        api.federation.cloud.requests.list().catch(() => [] as CloudLaunchRequest[]),
      ])
      setInfo(i)
      setNodes(n)
      setRequests(r)
    } catch {
      // Federation off or cloud not reachable: the page already says so.
    }
  }, [])

  useEffect(() => {
    void load()
    const id = window.setInterval(() => void load(), POLL_MS)
    return () => window.clearInterval(id)
  }, [load])

  if (!info) return null
  const live = nodes.filter((n) => n.status !== "terminated")
  const old = nodes.filter((n) => n.status === "terminated")
  const spend = info.month_spend_usd ?? 0
  const budget = info.settings.monthly_budget_usd
  const pct = budget > 0 ? Math.min(100, (spend / budget) * 100) : 0
  const hourlyNow = live.reduce((acc, n) => acc + (n.cost_per_hr ?? 0), 0)

  return (
    <section className="space-y-2">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-muted-foreground">
          <Cloud size={15} /> Nube · RunPod
        </h2>
        {info.has_api_key && (
          <div className="flex gap-2">
            <button type="button" className={btnCls} onClick={() => setSettingsOpen(true)}>
              <Settings2 size={12} /> Ajustes
            </button>
            <button type="button" className={primaryBtnCls} onClick={() => setLaunchOpen(true)}>
              <Rocket size={14} /> Lanzar nodo
            </button>
          </div>
        )}
      </div>

      <div className="rounded-lg border border-border bg-card">
        {!info.has_api_key ? (
          <div className="p-4">
            <ApiKeyForm onSaved={() => void load()} />
          </div>
        ) : (
          <>
            <div className="flex flex-wrap items-center gap-x-6 gap-y-2 px-4 py-3 text-sm">
              <div className="min-w-56 flex-1">
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>Gastado este mes</span>
                  <span className="tabular-nums">
                    {usd(spend)} / {budget > 0 ? usd(budget) : "sin tope"}
                  </span>
                </div>
                <div className="mt-1 h-1.5 rounded bg-muted">
                  <div
                    className={`h-1.5 rounded ${pct >= 90 ? "bg-red-500" : pct >= 70 ? "bg-amber-500" : "bg-primary"}`}
                    style={{ width: `${pct}%` }}
                  />
                </div>
              </div>
              <span className="text-xs text-muted-foreground">
                Ahora: {live.length} {live.length === 1 ? "nodo" : "nodos"} · {usd(hourlyNow, 3)}/h
              </span>
              <span className="text-xs text-muted-foreground">
                Por defecto: máx {minutesLabel(info.settings.default_ttl_minutes)} · inactivo{" "}
                {minutesLabel(info.settings.default_idle_minutes)}
              </span>
            </div>
            {info.last_error && (
              <div className="border-t border-border px-4 py-2 text-xs text-red-600 dark:text-red-300">
                La vigilancia de nodos falló: {info.last_error}
              </div>
            )}
            {(info.orphans ?? []).length > 0 && (
              <div className="border-t border-amber-500/40 bg-amber-500/5 px-4 py-2 text-sm">
                <div className="flex items-center gap-1.5 font-medium text-amber-700 dark:text-amber-300">
                  <AlertTriangle size={14} /> Pods huérfanos en RunPod (cobrando y sin seguimiento)
                </div>
                {info.orphans!.map((o) => (
                  <div key={o.pod_id} className="mt-1 flex items-center gap-3 text-xs">
                    <span className="font-mono">{o.name}</span>
                    <span className="text-muted-foreground">
                      {o.status} · {usd(o.cost_per_hr, 3)}/h
                    </span>
                    <button
                      type="button"
                      className={`${btnCls} ml-auto text-red-600 dark:text-red-300`}
                      onClick={async () => {
                        if (!window.confirm(`¿Terminar el pod ${o.name}?`)) return
                        try {
                          await api.federation.cloud.terminateOrphan(o.pod_id)
                          void load()
                        } catch (err) {
                          toast.error(errMsg(err))
                        }
                      }}
                    >
                      <Trash2 size={12} /> Terminar
                    </button>
                  </div>
                ))}
              </div>
            )}
            <PendingRequests requests={requests} onChanged={() => void load()} />
            <div className="divide-y divide-border border-t border-border">
              {live.length === 0 ? (
                <div className="px-4 py-4 text-sm text-muted-foreground">No hay nodos en la nube en marcha.</div>
              ) : (
                live.map((n) => (
                  <NodeRow key={n.id} node={n} onChanged={() => void load()} onAddModels={setAddingTo} />
                ))
              )}
              {old.length > 0 && (
                <button
                  type="button"
                  className="w-full px-4 py-2 text-left text-xs text-muted-foreground hover:bg-muted"
                  onClick={() => setShowOld((v) => !v)}
                >
                  {showOld ? "Ocultar" : "Ver"} terminados ({old.length})
                </button>
              )}
              {showOld &&
                old.map((n) => (
                  <NodeRow
                    key={n.id}
                    node={n}
                    onChanged={() => void load()}
                    onRelaunch={(node) => {
                      setLaunchPreset(presetFromNode(node))
                      setLaunchOpen(true)
                    }}
                  />
                ))}
            </div>
          </>
        )}
      </div>

      {info.has_api_key && <VolumesPanel />}

      {info.has_api_key && (
        <>
          <LaunchDialog
            open={launchOpen}
            onOpenChange={(v) => {
              setLaunchOpen(v)
              if (!v) setLaunchPreset(null)
            }}
            info={info}
            onLaunched={() => void load()}
            preset={launchPreset}
          />
          <SettingsDialog open={settingsOpen} onOpenChange={setSettingsOpen} info={info} onSaved={() => void load()} />
          <AddModelsDialog node={addingTo} onOpenChange={(v) => !v && setAddingTo(null)} onDone={() => void load()} />
        </>
      )}
    </section>
  )
}


/** Peticiones de máquina en espera de capacidad, y las que acabaron sin salir. */
function PendingRequests({ requests, onChanged }: { requests: CloudLaunchRequest[]; onChanged: () => void }) {
  const waiting = requests.filter((r) => r.status === "waiting")
  const recent = requests.filter((r) => r.status === "expired" || r.status === "failed").slice(0, 3)
  if (waiting.length === 0 && recent.length === 0) return null
  return (
    <div className="divide-y divide-border border-t border-border">
      {[...waiting, ...recent].map((r) => (
        <div key={r.id} className="flex flex-wrap items-center justify-between gap-2 px-4 py-2 text-sm">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <Clock size={13} className={r.status === "waiting" ? "text-sky-400" : "text-destructive"} />
              <span className="font-medium">{r.name || "Petición de máquina"}</span>
              <span className="text-xs text-muted-foreground">
                {r.gpuCount}× {r.gpuTypeIds.join(" → ")}
                {r.purpose === "benchmark" ? " · para un benchmark" : ""}
              </span>
            </div>
            <div className="text-xs text-muted-foreground">
              {r.status === "waiting"
                ? `Esperando capacidad · ${r.attempts} intentos · caduca ${r.expiresAt ? new Date(r.expiresAt).toLocaleString() : "—"}`
                : r.status === "expired" ? "Caducó sin máquina libre" : `Falló: ${r.lastError ?? ""}`}
              {r.status === "waiting" && r.lastError ? ` · ${r.lastError}` : ""}
            </div>
          </div>
          {r.status === "waiting" && (
            <button
              type="button"
              className={btnCls}
              onClick={async () => {
                try {
                  await api.federation.cloud.requests.cancel(r.id)
                  onChanged()
                } catch (err) {
                  toast.error(errMsg(err))
                }
              }}
            >
              <Ban size={12} /> Cancelar
            </button>
          )}
        </div>
      ))}
    </div>
  )
}
