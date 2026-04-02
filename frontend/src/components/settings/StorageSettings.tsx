import { useEffect, useMemo, useState } from "react"
import * as Dialog from "@radix-ui/react-dialog"
import { toast } from "sonner"
import type { LocalModel, ServerConfig } from "@/types"

interface StorageSettingsProps {
  localModels: LocalModel[]
  config: ServerConfig
  onSave: (patch: Partial<ServerConfig>) => Promise<void>
}

export function StorageSettings({ localModels, config, onSave }: StorageSettingsProps) {
  const [downloadDir, setDownloadDir] = useState(config.downloadDir)
  const [confirmOpen, setConfirmOpen] = useState(false)

  useEffect(() => {
    setDownloadDir(config.downloadDir)
  }, [config.downloadDir])

  const maxSize = useMemo(
    () => Math.max(1, ...localModels.map((model) => model.sizeGb)),
    [localModels],
  )

  const save = async () => {
    try {
      await onSave({ downloadDir })
      toast.success("Storage settings guardadas")
    } catch {
      // page-level toast is shown in Settings
    }
  }

  const clearCache = () => {
    localStorage.removeItem("ocabra.hfCache")
    toast.success("Cache local de UI limpiada")
    setConfirmOpen(false)
  }

  return (
    <section className="space-y-3 rounded-lg border border-border bg-card p-4">
      <h2 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">Storage</h2>

      <div className="space-y-2">
        {localModels.map((model) => {
          const ratio = Math.max(8, (model.sizeGb / maxSize) * 100)
          return (
            <div key={model.modelId} className="space-y-1">
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>{model.modelId}</span>
                <span>{model.sizeGb.toFixed(2)} GB</span>
              </div>
              <div className="h-2 rounded-full bg-muted">
                <div className="h-full rounded-full bg-sky-500" style={{ width: `${ratio}%` }} />
              </div>
            </div>
          )
        })}
      </div>

      <label className="block text-sm text-muted-foreground">
        MODELS_DIR
        <input
          value={config.modelsDir}
          readOnly
          className="mt-1 w-full rounded-md border border-border bg-muted/30 px-3 py-2 text-muted-foreground"
        />
        <p className="mt-1 text-xs text-muted-foreground">
          Gestionado por el entorno del contenedor. No se cambia en runtime desde la UI.
        </p>
      </label>

      <label className="block text-sm text-muted-foreground">
        Carpeta de descarga de modelos
        <input
          value={downloadDir}
          onChange={(event) => setDownloadDir(event.target.value)}
          className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2"
        />
      </label>

      <div className="flex flex-wrap items-center gap-2">
        <button
          type="button"
          onClick={() => void save()}
          className="rounded-md bg-primary px-3 py-2 text-sm font-medium text-primary-foreground"
        >
          Guardar Storage
        </button>
        <button
          type="button"
          onClick={() => setConfirmOpen(true)}
          className="rounded-md border border-red-500/40 px-3 py-2 text-sm text-red-200 hover:bg-red-500/20"
        >
          Limpiar cache HuggingFace
        </button>
      </div>

      <Dialog.Root open={confirmOpen} onOpenChange={setConfirmOpen}>
        <Dialog.Portal>
          <Dialog.Overlay className="fixed inset-0 z-40 bg-black/60" />
          <Dialog.Content className="fixed left-1/2 top-1/2 z-50 w-[95vw] max-w-md -translate-x-1/2 -translate-y-1/2 rounded-lg border border-border bg-card p-5">
            <Dialog.Title className="text-lg font-semibold">Limpiar cache HF</Dialog.Title>
            <Dialog.Description className="mt-1 text-sm text-muted-foreground">
              Esta accion limpia el estado local de cache en UI. La limpieza del cache real en backend aun no esta cableada.
            </Dialog.Description>
            <div className="mt-4 flex justify-end gap-2">
              <Dialog.Close className="rounded-md border border-border px-3 py-2 text-sm hover:bg-muted">
                Cancelar
              </Dialog.Close>
              <button
                type="button"
                onClick={clearCache}
                className="rounded-md border border-red-500/40 px-3 py-2 text-sm text-red-200 hover:bg-red-500/20"
              >
                Limpiar
              </button>
            </div>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
    </section>
  )
}
