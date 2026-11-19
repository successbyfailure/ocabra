import { useEffect, useState } from "react"
import * as Dialog from "@radix-ui/react-dialog"
import { AlertCircle, Plus, RefreshCw } from "lucide-react"
import { toast } from "sonner"
import { api } from "@/api/client"
import { AddBackendDialog } from "@/components/backends/AddBackendDialog"
import { BackendCard } from "@/components/backends/BackendCard"
import { ConfirmUninstallDialog } from "@/components/backends/ConfirmUninstallDialog"
import { OllamaControls } from "@/components/backends/OllamaServerCard"
import { BackendDefaults, hasBackendDefaults } from "@/components/backends/BackendDefaults"
import { useBackendsStore } from "@/stores/backendsStore"
import { cn } from "@/lib/cn"
import type { BackendInstallMethod, BackendModuleState, BackendPublishedInfo, ServerConfig } from "@/types"

/** Para qué sirve cada backend, a partir de sus etiquetas. */
const USES: { key: string; label: string; tags: string[] }[] = [
  { key: "llm", label: "LLM", tags: ["llm"] },
  { key: "audio", label: "Audio", tags: ["tts", "stt"] },
  { key: "image", label: "Imagen", tags: ["image"] },
  { key: "video", label: "Vídeo", tags: ["video"] },
  { key: "music", label: "Música", tags: ["music"] },
]
const usesOf = (b: BackendModuleState) => {
  const tags = b.tags.map((t) => t.toLowerCase())
  return USES.filter((u) => u.tags.some((t) => tags.includes(t))).map((u) => u.key)
}

function fmtSize(mb: number): string {
  return mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${Math.round(mb)} MB`
}

export function Backends() {
  const backends = useBackendsStore((s) => s.backends)
  const loading = useBackendsStore((s) => s.loading)
  const error = useBackendsStore((s) => s.error)
  const fetchAll = useBackendsStore((s) => s.fetchAll)
  const install = useBackendsStore((s) => s.install)
  const cancelInstall = useBackendsStore((s) => s.cancelInstall)
  const uninstall = useBackendsStore((s) => s.uninstall)

  const [uninstallTarget, setUninstallTarget] = useState<BackendModuleState | null>(null)
  const [logsTarget, setLogsTarget] = useState<BackendModuleState | null>(null)
  const [logsContent, setLogsContent] = useState<string>("")
  const [logsLoading, setLogsLoading] = useState(false)
  const [adding, setAdding] = useState(false)
  // En la página solo lo que está (o está entrando o saliendo, o falló al
  // instalarse); el resto se añade desde el catálogo.
  const present = backends.filter((b) => b.installStatus !== "not_installed")
  const available = backends.filter((b) => b.installStatus === "not_installed")
  const [use, setUse] = useState<string | null>(null)
  // La configuración del servidor la cargan una vez la página y la comparten
  // las tarjetas (valores por defecto de cada backend, ajustes de Ollama).
  const [config, setConfig] = useState<ServerConfig | null>(null)
  useEffect(() => {
    api.config.get().then(setConfig).catch(() => setConfig(null))
  }, [])
  const saveConfig = async (patch: Partial<ServerConfig>) => {
    setConfig(await api.config.patch(patch))
  }
  const [offers, setOffers] = useState<Record<string, BackendPublishedInfo | null>>({})
  const onOffer = (backendType: string, info: BackendPublishedInfo | null) =>
    setOffers((prev) => (prev[backendType] === info ? prev : { ...prev, [backendType]: info }))
  const shown = use ? present.filter((b) => usesOf(b).includes(use)) : present
  const diskMb = present.reduce((a, b) => a + (b.installStatus === "installed" ? (b.actualSizeMb ?? 0) : 0), 0)
  const loadedTotal = present.reduce((a, b) => a + (b.modelsLoaded ?? 0), 0)
  const updates = present.filter(
    (b) => b.installStatus === "installed" && (offers[b.backendType]?.updateAvailable || b.hasUpdate),
  )

  useEffect(() => {
    void fetchAll()
  }, [fetchAll])

  const handleInstall = (backendType: string, method: BackendInstallMethod) => {
    install(backendType, method)
    const how = { auto: "", oci: " (precompilado)", source: " desde el código" }[method]
    toast.success(`Instalando ${backendType}${how}`)
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
            Los motores de inferencia instalados en este servidor. Añade más desde el catálogo.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => void fetchAll()}
            className="inline-flex items-center gap-1.5 rounded-md border border-border px-3 py-1.5 text-sm text-muted-foreground hover:bg-muted hover:text-foreground"
            title="Recargar"
          >
            <RefreshCw size={14} />
            Recargar
          </button>
          <button
            type="button"
            onClick={() => setAdding(true)}
            disabled={loading && backends.length === 0}
            className="inline-flex items-center gap-1.5 rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-40"
          >
            <Plus size={14} />
            Añadir backend{available.length ? ` (${available.length})` : ""}
          </button>
        </div>
      </div>

      {error && (
        <div className="flex items-start gap-2 rounded-md border border-red-500/40 bg-red-500/10 p-3 text-sm text-red-100">
          <AlertCircle size={16} className="mt-0.5 shrink-0" />
          <div>
            <span className="font-semibold">Error:</span> {error}
          </div>
        </div>
      )}

      {present.length > 0 && (
        <div className="flex flex-wrap items-center justify-between gap-3" data-testid="backends-summary">
          <div className="flex flex-wrap items-center gap-1.5">
            <button
              type="button"
              onClick={() => setUse(null)}
              className={cn("rounded-full border px-3 py-1 text-xs", use === null ? "border-primary bg-primary/15 text-foreground" : "border-border text-muted-foreground hover:bg-muted")}
            >
              Todos ({present.length})
            </button>
            {USES.map((u) => {
              const n = present.filter((b) => usesOf(b).includes(u.key)).length
              if (!n) return null
              return (
                <button
                  key={u.key}
                  type="button"
                  onClick={() => setUse(use === u.key ? null : u.key)}
                  className={cn("rounded-full border px-3 py-1 text-xs", use === u.key ? "border-primary bg-primary/15 text-foreground" : "border-border text-muted-foreground hover:bg-muted")}
                  data-use={u.key}
                >
                  {u.label} ({n})
                </button>
              )
            })}
          </div>
          <p className="text-xs text-muted-foreground">
            {fmtSize(diskMb)} en disco · {loadedTotal} {loadedTotal === 1 ? "modelo cargado" : "modelos cargados"}
          </p>
        </div>
      )}

      {updates.length > 0 && (
        <div className="flex flex-wrap items-center gap-2 rounded-md border border-emerald-500/40 bg-emerald-500/10 px-3 py-2 text-sm text-emerald-100" data-testid="backends-updates">
          <span className="font-medium">
            {updates.length === 1 ? "Hay una actualización disponible:" : `Hay ${updates.length} actualizaciones disponibles:`}
          </span>
          {updates.map((b) => (
            <a key={b.backendType} href={`#backend-${b.backendType}`} className="rounded border border-emerald-500/40 px-2 py-0.5 text-xs hover:bg-emerald-500/20">
              {b.displayName}
            </a>
          ))}
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
      ) : present.length === 0 ? (
        <div className="rounded-md border border-dashed border-border p-10 text-center text-sm text-muted-foreground">
          No hay ningún backend instalado.{" "}
          {available.length > 0 && (
            <button type="button" className="text-primary underline" onClick={() => setAdding(true)}>
              Añade uno
            </button>
          )}
        </div>
      ) : (
        <div className="grid grid-cols-[repeat(auto-fill,minmax(min(100%,380px),1fr))] gap-4">
          {shown.map((backend) => (
            <BackendCard
              key={backend.backendType}
              backend={backend}
              onOffer={onOffer}
              onInstall={handleInstall}
              onCancelInstall={handleCancel}
              onUninstall={handleRequestUninstall}
              onUpdate={(b) => handleInstall(b.backendType, "oci")}
              onViewLogs={handleViewLogs}
              extra={
                backend.installStatus !== "installed" || !config ? undefined : backend.backendType === "ollama" ? (
                  <OllamaControls config={config} onSave={saveConfig} />
                ) : hasBackendDefaults(backend.backendType) ? (
                  <BackendDefaults backendType={backend.backendType} config={config} onSave={saveConfig} />
                ) : undefined
              }
            />
          ))}
        </div>
      )}

      <AddBackendDialog open={adding} onOpenChange={setAdding} available={available} onInstall={handleInstall} />

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
