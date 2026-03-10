import { useEffect, useState } from "react"
import * as Dialog from "@radix-ui/react-dialog"
import { X } from "lucide-react"
import { ScheduleEditor } from "@/components/models/ScheduleEditor"
import type { EvictionSchedule, GPUState, LoadPolicy, ModelState } from "@/types"

interface ModelConfigModalProps {
  model: ModelState | null
  gpus: GPUState[]
  open: boolean
  onOpenChange: (open: boolean) => void
  onSave: (modelId: string, patch: {
    loadPolicy: LoadPolicy
    preferredGpu: number | null
    autoReload: boolean
    schedules: EvictionSchedule[]
  }) => Promise<void>
}

export function ModelConfigModal({ model, gpus, open, onOpenChange, onSave }: ModelConfigModalProps) {
  const [loadPolicy, setLoadPolicy] = useState<LoadPolicy>("on_demand")
  const [preferredGpu, setPreferredGpu] = useState<number | null>(null)
  const [autoReload, setAutoReload] = useState(false)
  const [schedules, setSchedules] = useState<EvictionSchedule[]>([])
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    if (!model) return
    setLoadPolicy(model.loadPolicy)
    setPreferredGpu(model.preferredGpu)
    setAutoReload(model.autoReload)
    setSchedules(model.schedules ?? [])
  }, [model])

  const handleSave = async () => {
    if (!model) return
    setSaving(true)
    try {
      await onSave(model.modelId, { loadPolicy, preferredGpu, autoReload, schedules })
      onOpenChange(false)
    } finally {
      setSaving(false)
    }
  }

  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-40 bg-black/60" />
        <Dialog.Content className="fixed left-1/2 top-1/2 z-50 max-h-[88vh] w-[95vw] max-w-2xl -translate-x-1/2 -translate-y-1/2 overflow-y-auto rounded-lg border border-border bg-card p-5">
          <div className="mb-4 flex items-start justify-between">
            <div>
              <Dialog.Title className="text-lg font-semibold">Configurar modelo</Dialog.Title>
              <Dialog.Description className="text-sm text-muted-foreground">
                {model?.displayName ?? ""}
              </Dialog.Description>
            </div>
            <Dialog.Close className="rounded-md p-1 text-muted-foreground hover:bg-muted" aria-label="Close">
              <X size={16} />
            </Dialog.Close>
          </div>

          <div className="space-y-4">
            <label className="block text-sm">
              <span className="mb-1 block text-muted-foreground">Load policy</span>
              <select
                value={loadPolicy}
                onChange={(event) => setLoadPolicy(event.target.value as LoadPolicy)}
                className="w-full rounded-md border border-border bg-background px-3 py-2"
              >
                <option value="pin">pin</option>
                <option value="warm">warm</option>
                <option value="on_demand">on_demand</option>
              </select>
            </label>

            <label className="block text-sm">
              <span className="mb-1 block text-muted-foreground">GPU preferida</span>
              <select
                value={preferredGpu === null ? "" : String(preferredGpu)}
                onChange={(event) => setPreferredGpu(event.target.value === "" ? null : Number(event.target.value))}
                className="w-full rounded-md border border-border bg-background px-3 py-2"
              >
                <option value="">Default del servidor</option>
                {gpus.map((gpu) => (
                  <option key={gpu.index} value={gpu.index}>
                    GPU {gpu.index} - {gpu.name}
                  </option>
                ))}
              </select>
            </label>

            <label className="inline-flex items-center gap-2 text-sm text-muted-foreground">
              <input
                type="checkbox"
                checked={autoReload}
                onChange={(event) => setAutoReload(event.target.checked)}
              />
              auto_reload
            </label>

            <ScheduleEditor value={schedules} onChange={setSchedules} />
          </div>

          <div className="mt-5 flex justify-end gap-2">
            <Dialog.Close className="rounded-md border border-border px-3 py-2 text-sm hover:bg-muted">
              Cancelar
            </Dialog.Close>
            <button
              type="button"
              onClick={() => void handleSave()}
              disabled={saving}
              className="rounded-md bg-primary px-3 py-2 text-sm font-medium text-primary-foreground disabled:opacity-50"
            >
              {saving ? "Guardando..." : "Guardar"}
            </button>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  )
}
