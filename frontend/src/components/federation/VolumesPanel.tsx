import { useCallback, useEffect, useState } from "react"
import * as Dialog from "@radix-ui/react-dialog"
import { HardDrive, Loader2, Plus, Trash2 } from "lucide-react"
import { toast } from "sonner"
import { api } from "@/api/client"
import type { CloudDataCenter, CloudVolume } from "@/types"

const btnCls =
  "inline-flex items-center gap-1.5 rounded-md border border-border px-2.5 py-1 text-xs hover:bg-muted disabled:opacity-50"
const inputCls = "w-full rounded-md border border-border bg-background px-2 py-1.5 text-sm"

function errMsg(err: unknown): string {
  return err instanceof Error ? err.message : String(err)
}

/** "4090 ×3 alta, A40 baja…" — what a data center has in stock. */
function stockSummary(dc: CloudDataCenter): string {
  if (!dc.gpus.length) return "sin GPUs libres ahora"
  const short = dc.gpus.slice(0, 4).map((g) => `${g.name.replace(/^NVIDIA (GeForce )?/, "")} (${g.availability.toLowerCase()})`)
  return short.join(", ") + (dc.gpus.length > 4 ? ` y ${dc.gpus.length - 4} más` : "")
}

function CreateVolumeDialog({
  open,
  onOpenChange,
  onCreated,
}: {
  open: boolean
  onOpenChange: (v: boolean) => void
  onCreated: () => void
}) {
  const [dcs, setDcs] = useState<CloudDataCenter[] | null>(null)
  const [name, setName] = useState("ocabra-models")
  const [size, setSize] = useState(100)
  const [dc, setDc] = useState("")
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    if (!open) return
    setDcs(null)
    api.federation.cloud
      .dataCenters()
      .then((list) => {
        setDcs(list)
        setDc((cur) => cur || list[0]?.id || "")
      })
      .catch((err) => {
        toast.error(errMsg(err))
        setDcs([])
      })
  }, [open])

  const chosen = dcs?.find((d) => d.id === dc)
  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-40 bg-black/60" />
        <Dialog.Content className="fixed left-1/2 top-1/2 z-50 max-h-[90vh] w-[95vw] max-w-lg -translate-x-1/2 -translate-y-1/2 overflow-y-auto rounded-lg border border-border bg-card p-5">
          <Dialog.Title className="text-lg font-semibold">Nuevo disco de modelos</Dialog.Title>
          <Dialog.Description className="mt-1 text-sm text-muted-foreground">
            Guarda modelos y backends entre pods: el siguiente nodo arranca con ellos ya descargados. Solo
            guarda eso: ni datos ni peticiones. Ata los pods a su centro de datos y a Secure Cloud, así que
            elige uno con las GPUs que vayas a usar.
          </Dialog.Description>
          <div className="mt-4 space-y-3 text-sm">
            <label className="block">
              <span className="text-xs text-muted-foreground">Nombre</span>
              <input className={inputCls} value={name} onChange={(e) => setName(e.target.value)} />
            </label>
            <label className="block">
              <span className="text-xs text-muted-foreground">Tamaño (GB) — solo puede crecer después</span>
              <input
                className={inputCls}
                type="number"
                min={10}
                max={4096}
                value={size}
                onChange={(e) => setSize(Math.max(10, Number(e.target.value) || 10))}
              />
              <span className="text-xs text-muted-foreground">
                ${(size * 0.07).toFixed(2)}/mes mientras exista, se use o no.
              </span>
            </label>
            <label className="block">
              <span className="text-xs text-muted-foreground">Centro de datos</span>
              {dcs == null ? (
                <div className="flex items-center gap-2 py-1.5 text-muted-foreground">
                  <Loader2 size={14} className="animate-spin" /> Cargando…
                </div>
              ) : (
                <select className={inputCls} value={dc} onChange={(e) => setDc(e.target.value)}>
                  {dcs.map((d) => (
                    <option key={d.id} value={d.id}>
                      {d.id}
                      {d.region ? ` · ${d.region}` : ""} — {d.gpus.length} tipos de GPU libres
                    </option>
                  ))}
                </select>
              )}
              {chosen && <span className="text-xs text-muted-foreground">{stockSummary(chosen)}</span>}
            </label>
          </div>
          <div className="mt-4 flex justify-end gap-2">
            <Dialog.Close className={btnCls}>Cancelar</Dialog.Close>
            <button
              type="button"
              className="inline-flex items-center gap-1.5 rounded-md bg-primary px-3 py-1.5 text-sm text-primary-foreground disabled:opacity-50"
              disabled={busy || !dc || !name.trim()}
              onClick={async () => {
                setBusy(true)
                try {
                  await api.federation.cloud.createVolume({ name: name.trim(), size_gb: size, data_center: dc })
                  toast.success("Disco creado")
                  onOpenChange(false)
                  onCreated()
                } catch (err) {
                  toast.error(errMsg(err))
                } finally {
                  setBusy(false)
                }
              }}
            >
              {busy ? <Loader2 size={14} className="animate-spin" /> : <Plus size={14} />} Crear
            </button>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  )
}

/**
 * RunPod network volumes: the model/backend cache cloud nodes can mount.
 */
export function VolumesPanel() {
  const [vols, setVols] = useState<CloudVolume[] | null>(null)
  const [creating, setCreating] = useState(false)
  const [busy, setBusy] = useState<string | null>(null)

  const load = useCallback(async () => {
    try {
      setVols(await api.federation.cloud.volumes())
    } catch {
      setVols([])
    }
  }, [])

  useEffect(() => {
    void load()
  }, [load])

  const monthly = (vols ?? []).reduce((acc, v) => acc + v.usd_per_month, 0)
  return (
    <div className="rounded-lg border border-border">
      <div className="flex flex-wrap items-center gap-2 border-b border-border px-4 py-2">
        <HardDrive size={14} className="text-muted-foreground" />
        <span className="text-sm font-medium">Discos de modelos</span>
        <span className="text-xs text-muted-foreground">
          {vols?.length ? `${vols.length} · $${monthly.toFixed(2)}/mes` : "ninguno: cada pod descarga sus modelos"}
        </span>
        <button type="button" className={`${btnCls} ml-auto`} onClick={() => setCreating(true)}>
          <Plus size={12} /> Nuevo disco
        </button>
      </div>
      {vols == null ? (
        <div className="flex items-center gap-2 px-4 py-3 text-sm text-muted-foreground">
          <Loader2 size={14} className="animate-spin" /> Cargando discos…
        </div>
      ) : (
        vols.map((v) => (
          <div key={v.id} className="flex flex-wrap items-center gap-x-3 gap-y-1 border-b border-border/50 px-4 py-2 text-sm last:border-b-0">
            <span className="font-medium">{v.name}</span>
            <span className="text-xs text-muted-foreground">
              {v.size_gb} GB · {v.data_center} · ${v.usd_per_month.toFixed(2)}/mes
            </span>
            {v.in_use_by ? (
              <span className="rounded-full border border-sky-500/40 px-2 text-[11px] text-sky-700 dark:text-sky-300">
                en uso por {v.in_use_by}
              </span>
            ) : (
              <span className="text-[11px] text-muted-foreground">libre</span>
            )}
            <span className="ml-auto flex gap-2">
              <button
                type="button"
                className={btnCls}
                disabled={busy === v.id}
                onClick={async () => {
                  const raw = window.prompt(`Nuevo tamaño en GB (ahora ${v.size_gb}; solo puede crecer)`, String(v.size_gb + 50))
                  const size = Number(raw)
                  if (!raw || !Number.isFinite(size) || size <= v.size_gb) return
                  setBusy(v.id)
                  try {
                    await api.federation.cloud.resizeVolume(v.id, size)
                    toast.success("Disco ampliado")
                    await load()
                  } catch (err) {
                    toast.error(errMsg(err))
                  } finally {
                    setBusy(null)
                  }
                }}
              >
                Ampliar
              </button>
              <button
                type="button"
                className={`${btnCls} text-red-600 dark:text-red-300`}
                disabled={busy === v.id || Boolean(v.in_use_by)}
                title={v.in_use_by ? "Termina antes el nodo que lo usa" : undefined}
                onClick={async () => {
                  if (!window.confirm(`¿Borrar ${v.name}? Se pierden los modelos y backends que guarda.`)) return
                  setBusy(v.id)
                  try {
                    await api.federation.cloud.deleteVolume(v.id)
                    toast.success("Disco borrado")
                    await load()
                  } catch (err) {
                    toast.error(errMsg(err))
                  } finally {
                    setBusy(null)
                  }
                }}
              >
                <Trash2 size={12} /> Borrar
              </button>
            </span>
          </div>
        ))
      )}
      <CreateVolumeDialog open={creating} onOpenChange={setCreating} onCreated={() => void load()} />
    </div>
  )
}
