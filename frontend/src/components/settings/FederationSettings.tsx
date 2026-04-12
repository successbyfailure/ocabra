import { useCallback, useEffect, useState } from "react"
import * as Dialog from "@radix-ui/react-dialog"
import * as Tooltip from "@radix-ui/react-tooltip"
import { Globe, Loader2, Network, Plus, Trash2, Pencil, Wifi, WifiOff, Zap } from "lucide-react"
import { toast } from "sonner"
import { api } from "@/api/client"
import type { FederationPeer, FederationPeerCreate, FederationPeerUpdate, FederationTestResult, ServerConfig } from "@/types"

interface FederationSettingsProps {
  config: ServerConfig
  onSave: (patch: Partial<ServerConfig>) => Promise<void>
}

function formatTimeAgo(isoDate: string | null): string {
  if (!isoDate) return "Nunca"
  const diff = Date.now() - Date.parse(isoDate)
  if (Number.isNaN(diff)) return "Nunca"
  const seconds = Math.floor(diff / 1000)
  if (seconds < 60) return `${seconds}s`
  const minutes = Math.floor(seconds / 60)
  if (minutes < 60) return `${minutes}m`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h`
  const days = Math.floor(hours / 24)
  return `${days}d`
}

function PeerStatusBadge({ peer }: { peer: FederationPeer }) {
  if (!peer.enabled) {
    return (
      <span className="inline-flex items-center gap-1 rounded-full border border-border bg-muted px-2 py-0.5 text-xs font-medium text-muted-foreground">
        Desactivado
      </span>
    )
  }
  if (peer.online) {
    return (
      <span className="inline-flex items-center gap-1 rounded-full border border-emerald-500/30 bg-emerald-500/20 px-2 py-0.5 text-xs font-medium text-emerald-200">
        <Wifi size={10} />
        Online
      </span>
    )
  }
  return (
    <span className="inline-flex items-center gap-1 rounded-full border border-red-500/30 bg-red-500/20 px-2 py-0.5 text-xs font-medium text-red-200">
      <WifiOff size={10} />
      Offline
    </span>
  )
}

function AddPeerDialog({
  open,
  onOpenChange,
  onAdd,
  editPeer,
  onUpdate,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  onAdd: (data: FederationPeerCreate) => Promise<void>
  editPeer: FederationPeer | null
  onUpdate: (id: string, data: FederationPeerUpdate) => Promise<void>
}) {
  const [name, setName] = useState("")
  const [url, setUrl] = useState("")
  const [apiKey, setApiKey] = useState("")
  const [accessLevel, setAccessLevel] = useState<"inference" | "full">("inference")
  const [submitting, setSubmitting] = useState(false)

  useEffect(() => {
    if (editPeer) {
      setName(editPeer.name)
      setUrl(editPeer.url)
      setApiKey("")
      setAccessLevel(editPeer.access_level)
    } else {
      setName("")
      setUrl("")
      setApiKey("")
      setAccessLevel("inference")
    }
  }, [editPeer, open])

  const handleSubmit = async () => {
    if (!name.trim() || !url.trim()) {
      toast.error("Nombre y URL son obligatorios")
      return
    }
    if (!editPeer && !apiKey.trim()) {
      toast.error("API key es obligatoria")
      return
    }
    setSubmitting(true)
    try {
      if (editPeer) {
        const patch: FederationPeerUpdate = {
          name: name.trim(),
          url: url.trim(),
          access_level: accessLevel,
        }
        if (apiKey.trim()) {
          patch.api_key = apiKey.trim()
        }
        await onUpdate(editPeer.peer_id, patch)
        toast.success("Peer actualizado")
      } else {
        await onAdd({
          name: name.trim(),
          url: url.trim(),
          api_key: apiKey.trim(),
          access_level: accessLevel,
        })
        toast.success("Peer agregado")
      }
      onOpenChange(false)
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Error")
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-40 bg-black/60" />
        <Dialog.Content className="fixed left-1/2 top-1/2 z-50 w-[95vw] max-w-md -translate-x-1/2 -translate-y-1/2 rounded-lg border border-border bg-card p-5">
          <Dialog.Title className="text-lg font-semibold">
            {editPeer ? "Editar peer" : "Agregar peer"}
          </Dialog.Title>
          <Dialog.Description className="mt-1 text-sm text-muted-foreground">
            {editPeer
              ? "Edita la configuracion del peer federado."
              : "Conecta un nuevo nodo oCabra a la federacion."}
          </Dialog.Description>

          <div className="mt-4 space-y-3">
            <label className="block text-sm text-muted-foreground">
              Nombre
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="nodo-B"
                className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
              />
            </label>

            <label className="block text-sm text-muted-foreground">
              URL
              <input
                type="url"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="https://nodo-b.local:8000"
                className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
              />
            </label>

            <label className="block text-sm text-muted-foreground">
              API Key {editPeer && <span className="text-xs">(dejar vacio para mantener la actual)</span>}
              <input
                type="password"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder={editPeer ? "••••••••" : "sk-ocabra-..."}
                className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
              />
            </label>

            <label className="block text-sm text-muted-foreground">
              Nivel de acceso
              <select
                value={accessLevel}
                onChange={(e) => setAccessLevel(e.target.value as "inference" | "full")}
                className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
              >
                <option value="inference">inference — solo inferencia</option>
                <option value="full">full — inferencia + gestion</option>
              </select>
            </label>
          </div>

          <div className="mt-4 flex justify-end gap-2">
            <Dialog.Close className="rounded-md border border-border px-3 py-2 text-sm hover:bg-muted">
              Cancelar
            </Dialog.Close>
            <button
              type="button"
              disabled={submitting}
              onClick={() => void handleSubmit()}
              className="rounded-md bg-primary px-3 py-2 text-sm font-medium text-primary-foreground disabled:opacity-50"
            >
              {submitting ? (
                <Loader2 size={14} className="animate-spin" />
              ) : editPeer ? (
                "Guardar"
              ) : (
                "Agregar"
              )}
            </button>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  )
}

export function FederationSettings({ config, onSave }: FederationSettingsProps) {
  const [peers, setPeers] = useState<FederationPeer[]>([])
  const [loading, setLoading] = useState(true)
  const [togglingFederation, setTogglingFederation] = useState(false)
  const federationEnabled = config.federationEnabled ?? false
  const [addDialogOpen, setAddDialogOpen] = useState(false)
  const [editPeer, setEditPeer] = useState<FederationPeer | null>(null)
  const [testingPeerId, setTestingPeerId] = useState<string | null>(null)
  const [testResults, setTestResults] = useState<Record<string, FederationTestResult>>({})
  const [deletingPeerId, setDeletingPeerId] = useState<string | null>(null)

  const loadPeers = useCallback(async () => {
    try {
      const data = await api.federation.getPeers()
      setPeers(data)
    } catch {
      // Federation may not be enabled; show empty state
      setPeers([])
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void loadPeers()
    const timer = window.setInterval(() => { void loadPeers() }, 15_000)
    return () => window.clearInterval(timer)
  }, [loadPeers])

  const handleAdd = async (data: FederationPeerCreate) => {
    await api.federation.addPeer(data)
    await loadPeers()
  }

  const handleUpdate = async (id: string, data: FederationPeerUpdate) => {
    await api.federation.updatePeer(id, data)
    await loadPeers()
  }

  const handleToggleEnabled = async (peer: FederationPeer) => {
    try {
      await api.federation.updatePeer(peer.peer_id, { enabled: !peer.enabled })
      toast.success(peer.enabled ? "Peer desactivado" : "Peer activado")
      await loadPeers()
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Error")
    }
  }

  const handleTest = async (peerId: string) => {
    setTestingPeerId(peerId)
    try {
      const result = await api.federation.testPeer(peerId)
      setTestResults((prev) => ({ ...prev, [peerId]: result }))
      if (result.success) {
        toast.success(`Conexion exitosa — ${result.latency_ms ?? "?"}ms`)
      } else {
        toast.error(`Conexion fallida: ${result.error ?? "error desconocido"}`)
      }
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Error al testear conexion")
    } finally {
      setTestingPeerId(null)
    }
  }

  const handleDelete = async (peerId: string) => {
    setDeletingPeerId(peerId)
    try {
      await api.federation.deletePeer(peerId)
      toast.success("Peer eliminado")
      await loadPeers()
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Error al eliminar peer")
    } finally {
      setDeletingPeerId(null)
    }
  }

  const openEdit = (peer: FederationPeer) => {
    setEditPeer(peer)
    setAddDialogOpen(true)
  }

  const closeDialog = (open: boolean) => {
    setAddDialogOpen(open)
    if (!open) setEditPeer(null)
  }

  const onlineCount = peers.filter((p) => p.enabled && p.online).length
  const totalModels = peers.reduce((acc, p) => acc + (p.online ? p.models.length : 0), 0)
  const totalGpus = peers.reduce((acc, p) => acc + (p.online ? p.gpus.length : 0), 0)

  return (
    <Tooltip.Provider delayDuration={300}>
      <section className="space-y-4">
        <div className="rounded-lg border border-border bg-card p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Network size={20} className="text-muted-foreground" />
              <div>
                <h2 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">
                  Federacion
                </h2>
                <p className="text-xs text-muted-foreground">
                  Conecta multiples nodos oCabra para compartir modelos y GPUs.
                </p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <button
                type="button"
                disabled={togglingFederation}
                onClick={async () => {
                  setTogglingFederation(true)
                  try {
                    await onSave({ federationEnabled: !federationEnabled } as Partial<ServerConfig>)
                    toast.success(federationEnabled ? "Federacion desactivada" : "Federacion activada")
                    if (!federationEnabled) {
                      // Just enabled — reload peers after a short delay for manager to start
                      setTimeout(() => { void loadPeers() }, 1000)
                    }
                  } catch (err) {
                    toast.error(err instanceof Error ? err.message : "Error")
                  } finally {
                    setTogglingFederation(false)
                  }
                }}
                className={`relative inline-flex h-6 w-11 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none disabled:opacity-50 ${
                  federationEnabled ? "bg-emerald-500" : "bg-muted-foreground/30"
                }`}
                role="switch"
                aria-checked={federationEnabled}
              >
                <span
                  className={`pointer-events-none inline-block h-5 w-5 rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
                    federationEnabled ? "translate-x-5" : "translate-x-0"
                  }`}
                />
              </button>
              {federationEnabled && (
                <button
                  type="button"
                  onClick={() => {
                    setEditPeer(null)
                    setAddDialogOpen(true)
                  }}
                  className="inline-flex items-center gap-1.5 rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground"
                >
                  <Plus size={14} />
                  Agregar peer
                </button>
              )}
            </div>
          </div>

          {peers.length > 0 && (
            <div className="mt-3 flex items-center gap-4 text-xs text-muted-foreground">
              <span className="inline-flex items-center gap-1">
                <Globe size={12} />
                {onlineCount} nodo{onlineCount !== 1 ? "s" : ""} online
              </span>
              <span>{totalModels} modelo{totalModels !== 1 ? "s" : ""} remotos</span>
              <span>{totalGpus} GPU{totalGpus !== 1 ? "s" : ""} remota{totalGpus !== 1 ? "s" : ""}</span>
            </div>
          )}
        </div>

        {!federationEnabled ? (
          <div className="rounded-lg border border-dashed border-border bg-card/50 px-6 py-8 text-center">
            <Network size={32} className="mx-auto text-muted-foreground/50" />
            <p className="mt-2 text-sm font-medium text-muted-foreground">Federacion desactivada</p>
            <p className="mt-1 text-xs text-muted-foreground">
              Activa el modo federado para conectar multiples nodos oCabra y compartir modelos entre ellos.
            </p>
          </div>
        ) : loading ? (
          <div className="space-y-2">
            {Array.from({ length: 2 }).map((_, idx) => (
              <div key={`fed-skel-${idx}`} className="h-20 animate-pulse rounded-lg bg-muted" />
            ))}
          </div>
        ) : peers.length === 0 ? (
          <div className="rounded-lg border border-dashed border-border bg-card/50 px-6 py-8 text-center">
            <Network size={32} className="mx-auto text-muted-foreground/50" />
            <p className="mt-2 text-sm font-medium text-muted-foreground">Sin peers federados</p>
            <p className="mt-1 text-xs text-muted-foreground">
              Agrega un peer para conectar otro nodo oCabra y compartir modelos entre ellos.
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            {peers.map((peer) => {
              const testResult = testResults[peer.peer_id]
              return (
                <div
                  key={peer.peer_id}
                  className={`rounded-lg border bg-card px-4 py-3 ${
                    !peer.enabled
                      ? "border-border opacity-60"
                      : peer.online
                        ? "border-emerald-500/20"
                        : "border-red-500/20"
                  }`}
                >
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div className="space-y-1.5">
                      <div className="flex items-center gap-2">
                        <p className="font-medium">{peer.name}</p>
                        <PeerStatusBadge peer={peer} />
                        <span className="rounded-full border border-border bg-muted px-2 py-0.5 text-xs text-muted-foreground">
                          {peer.access_level}
                        </span>
                      </div>
                      <div className="flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
                        <span className="font-mono">{peer.url}</span>
                        {peer.last_heartbeat && (
                          <Tooltip.Root>
                            <Tooltip.Trigger asChild>
                              <span className="cursor-default">
                                Heartbeat: {formatTimeAgo(peer.last_heartbeat)} ago
                              </span>
                            </Tooltip.Trigger>
                            <Tooltip.Portal>
                              <Tooltip.Content
                                className="z-50 rounded-md border border-border bg-popover px-3 py-1.5 text-xs shadow-md"
                                sideOffset={4}
                              >
                                {new Date(peer.last_heartbeat).toLocaleString()}
                                <Tooltip.Arrow className="fill-border" />
                              </Tooltip.Content>
                            </Tooltip.Portal>
                          </Tooltip.Root>
                        )}
                        {peer.latency_ms != null && (
                          <span className="inline-flex items-center gap-1">
                            <Zap size={10} />
                            {Math.round(peer.latency_ms)}ms
                          </span>
                        )}
                        {peer.online && (
                          <>
                            <span>{peer.models.length} modelo{peer.models.length !== 1 ? "s" : ""}</span>
                            <span>{peer.gpus.length} GPU{peer.gpus.length !== 1 ? "s" : ""}</span>
                          </>
                        )}
                      </div>
                      {testResult && (
                        <div
                          className={`mt-1 rounded-md px-2 py-1 text-xs ${
                            testResult.success
                              ? "bg-emerald-500/10 text-emerald-300"
                              : "bg-red-500/10 text-red-300"
                          }`}
                        >
                          {testResult.success
                            ? `Test OK — latencia ${testResult.latency_ms ?? "?"}ms`
                            : `Test fallido: ${testResult.error ?? "error desconocido"}`}
                        </div>
                      )}
                    </div>

                    <div className="flex items-center gap-1.5">
                      <button
                        type="button"
                        onClick={() => void handleToggleEnabled(peer)}
                        className={`rounded-md border px-2.5 py-1 text-xs ${
                          peer.enabled
                            ? "border-amber-500/40 text-amber-200 hover:bg-amber-500/20"
                            : "border-emerald-500/40 text-emerald-200 hover:bg-emerald-500/20"
                        }`}
                      >
                        {peer.enabled ? "Desactivar" : "Activar"}
                      </button>
                      <button
                        type="button"
                        onClick={() => void handleTest(peer.peer_id)}
                        disabled={testingPeerId === peer.peer_id}
                        className="rounded-md border border-border px-2.5 py-1 text-xs hover:bg-muted disabled:opacity-50"
                      >
                        {testingPeerId === peer.peer_id ? (
                          <Loader2 size={12} className="animate-spin" />
                        ) : (
                          "Test"
                        )}
                      </button>
                      <button
                        type="button"
                        onClick={() => openEdit(peer)}
                        className="rounded-md border border-border p-1.5 text-muted-foreground hover:bg-muted hover:text-foreground"
                        aria-label="Editar peer"
                      >
                        <Pencil size={12} />
                      </button>
                      <button
                        type="button"
                        onClick={() => void handleDelete(peer.peer_id)}
                        disabled={deletingPeerId === peer.peer_id}
                        className="rounded-md border border-red-500/40 p-1.5 text-red-300 hover:bg-red-500/20 disabled:opacity-50"
                        aria-label="Eliminar peer"
                      >
                        {deletingPeerId === peer.peer_id ? (
                          <Loader2 size={12} className="animate-spin" />
                        ) : (
                          <Trash2 size={12} />
                        )}
                      </button>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        )}

        <AddPeerDialog
          open={addDialogOpen}
          onOpenChange={closeDialog}
          onAdd={handleAdd}
          editPeer={editPeer}
          onUpdate={handleUpdate}
        />
      </section>
    </Tooltip.Provider>
  )
}
