import { useEffect, useState } from "react"
import * as Dialog from "@radix-ui/react-dialog"
import { RefreshCw, AlertCircle } from "lucide-react"
import { toast } from "sonner"
import { api } from "@/api/client"
import { BackendCard } from "@/components/backends/BackendCard"
import { ConfirmUninstallDialog } from "@/components/backends/ConfirmUninstallDialog"
import { useBackendsStore } from "@/stores/backendsStore"
import type { BackendInstallMethod, BackendModuleState } from "@/types"

export function Backends() {
  const backends = useBackendsStore((s) => s.backends)
  const loading = useBackendsStore((s) => s.loading)
  const usingMock = useBackendsStore((s) => s.usingMock)
  const error = useBackendsStore((s) => s.error)
  const fetchAll = useBackendsStore((s) => s.fetchAll)
  const install = useBackendsStore((s) => s.install)
  const cancelInstall = useBackendsStore((s) => s.cancelInstall)
  const uninstall = useBackendsStore((s) => s.uninstall)

  const [uninstallTarget, setUninstallTarget] = useState<BackendModuleState | null>(null)
  const [logsTarget, setLogsTarget] = useState<BackendModuleState | null>(null)
  const [logsContent, setLogsContent] = useState<string>("")
  const [logsLoading, setLogsLoading] = useState(false)

  useEffect(() => {
    void fetchAll()
  }, [fetchAll])

  const handleInstall = (backendType: string, method: BackendInstallMethod) => {
    install(backendType, method)
    toast.success(`Instalando ${backendType} desde ${method}`)
  }

  const handleCancel = (backendType: string) => {
    cancelInstall(backendType)
    toast("Instalación cancelada")
  }

  const handleRequestUninstall = (backend: BackendModuleState) => {
    setUninstallTarget(backend)
  }

  const handleConfirmUninstall = async () => {
    if (!uninstallTarget) return
    const name = uninstallTarget.displayName
    try {
      await uninstall(uninstallTarget.backendType)
      toast.success(`${name} desinstalado`)
    } catch (err) {
      toast.error(err instanceof Error ? err.message : `Error al desinstalar ${name}`)
    }
  }

  const handleViewLogs = async (backend: BackendModuleState) => {
    setLogsTarget(backend)
    setLogsContent("")
    if (usingMock) {
      setLogsContent(
        `[mock] Logs de instalación para ${backend.backendType}.\nNo hay datos reales disponibles en modo mock.`,
      )
      return
    }
    setLogsLoading(true)
    try {
      const text = await api.backends.logs(backend.backendType)
      setLogsContent(text || "(sin logs)")
    } catch (err) {
      setLogsContent(
        `Error al obtener logs: ${err instanceof Error ? err.message : "desconocido"}`,
      )
    } finally {
      setLogsLoading(false)
    }
  }

  return (
    <div className="space-y-4 pb-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold">Backends</h1>
          <p className="text-muted-foreground">
            Instala, actualiza y desinstala los backends de inferencia.
          </p>
        </div>
        <button
          type="button"
          onClick={() => void fetchAll()}
          className="inline-flex items-center gap-1.5 rounded-md border border-border px-3 py-1.5 text-sm text-muted-foreground hover:bg-muted hover:text-foreground"
          title="Recargar"
        >
          <RefreshCw size={14} />
          Recargar
        </button>
      </div>

      {usingMock && (
        <div className="flex items-start gap-2 rounded-md border border-amber-500/40 bg-amber-500/10 p-3 text-sm text-amber-100">
          <AlertCircle size={16} className="mt-0.5 shrink-0" />
          <div>
            <span className="font-semibold">Modo mock:</span> la API{" "}
            <code className="font-mono">/ocabra/backends</code> todavía no está disponible.
            Esta vista muestra datos de ejemplo hasta que el backend se fusione.
          </div>
        </div>
      )}

      {error && !usingMock && (
        <div className="flex items-start gap-2 rounded-md border border-red-500/40 bg-red-500/10 p-3 text-sm text-red-100">
          <AlertCircle size={16} className="mt-0.5 shrink-0" />
          <div>
            <span className="font-semibold">Error:</span> {error}
          </div>
        </div>
      )}

      {loading && backends.length === 0 ? (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          {Array.from({ length: 4 }).map((_, idx) => (
            <div
              key={idx}
              className="h-60 animate-pulse rounded-lg border border-border bg-muted/30"
            />
          ))}
        </div>
      ) : backends.length === 0 ? (
        <div className="rounded-md border border-dashed border-border p-10 text-center text-sm text-muted-foreground">
          No hay backends disponibles.
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          {backends.map((backend) => (
            <BackendCard
              key={backend.backendType}
              backend={backend}
              onInstall={handleInstall}
              onCancelInstall={handleCancel}
              onUninstall={handleRequestUninstall}
              onUpdate={(b) =>
                handleInstall(b.backendType, b.installSource === "source" ? "source" : "oci")
              }
              onViewLogs={handleViewLogs}
            />
          ))}
        </div>
      )}

      <ConfirmUninstallDialog
        open={uninstallTarget !== null}
        onOpenChange={(open) => {
          if (!open) setUninstallTarget(null)
        }}
        backend={uninstallTarget}
        onConfirm={() => void handleConfirmUninstall()}
      />

      <Dialog.Root
        open={logsTarget !== null}
        onOpenChange={(open) => {
          if (!open) {
            setLogsTarget(null)
            setLogsContent("")
          }
        }}
      >
        <Dialog.Portal>
          <Dialog.Overlay className="fixed inset-0 z-40 bg-black/60" />
          <Dialog.Content className="fixed left-1/2 top-1/2 z-50 flex h-[80vh] w-[95vw] max-w-3xl -translate-x-1/2 -translate-y-1/2 flex-col rounded-lg border border-border bg-card p-5">
            <Dialog.Title className="text-lg font-semibold">
              Logs · {logsTarget?.displayName}
            </Dialog.Title>
            <Dialog.Description className="mt-1 text-sm text-muted-foreground">
              Ultima instalación o error del backend.
            </Dialog.Description>
            <div className="mt-3 flex-1 overflow-auto rounded-md border border-border bg-black/40 p-3 font-mono text-xs text-muted-foreground">
              {logsLoading ? (
                <span className="text-muted-foreground/70">Cargando logs...</span>
              ) : (
                <pre className="whitespace-pre-wrap break-words">{logsContent}</pre>
              )}
            </div>
            <div className="mt-3 flex justify-end">
              <Dialog.Close className="rounded-md border border-border px-4 py-2 text-sm hover:bg-muted">
                Cerrar
              </Dialog.Close>
            </div>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
    </div>
  )
}
