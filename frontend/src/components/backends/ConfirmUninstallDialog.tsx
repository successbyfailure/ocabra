import * as Dialog from "@radix-ui/react-dialog"
import { AlertTriangle } from "lucide-react"
import type { BackendModuleState } from "@/types"

interface ConfirmUninstallDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  backend: BackendModuleState | null
  onConfirm: () => void
}

export function ConfirmUninstallDialog({
  open,
  onOpenChange,
  backend,
  onConfirm,
}: ConfirmUninstallDialogProps) {
  const hasModelsLoaded = (backend?.modelsLoaded ?? 0) > 0

  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-40 bg-black/60" />
        <Dialog.Content className="fixed left-1/2 top-1/2 z-50 w-[95vw] max-w-md -translate-x-1/2 -translate-y-1/2 rounded-lg border border-border bg-card p-5">
          <Dialog.Title className="text-lg font-semibold">
            Desinstalar {backend?.displayName ?? "backend"}
          </Dialog.Title>
          <Dialog.Description className="mt-1 text-sm text-muted-foreground">
            Se eliminaran los binarios/venv del backend. Los modelos que use este backend
            dejaran de estar disponibles hasta que vuelvas a instalarlo.
          </Dialog.Description>

          {hasModelsLoaded && (
            <div className="mt-4 flex items-start gap-2 rounded-md border border-amber-500/40 bg-amber-500/10 p-3 text-sm text-amber-100">
              <AlertTriangle size={16} className="mt-0.5 shrink-0" />
              <div>
                Este backend tiene{" "}
                <span className="font-semibold">{backend?.modelsLoaded}</span> modelo(s)
                cargados. Descargalos antes de desinstalar el backend.
              </div>
            </div>
          )}

          <div className="mt-5 flex justify-end gap-2">
            <Dialog.Close className="rounded-md border border-border px-4 py-2 text-sm hover:bg-muted">
              Cancelar
            </Dialog.Close>
            <button
              type="button"
              disabled={hasModelsLoaded}
              className="rounded-md border border-red-500/40 px-4 py-2 text-sm font-medium text-red-200 hover:bg-red-500/20 disabled:cursor-not-allowed disabled:opacity-40"
              onClick={() => {
                onConfirm()
                onOpenChange(false)
              }}
            >
              Desinstalar
            </button>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  )
}
