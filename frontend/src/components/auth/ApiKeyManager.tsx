import { useEffect, useState } from "react"
import { toast } from "sonner"
import { api } from "@/api/client"
import type { ApiKey } from "@/types"

function keyStatus(key: ApiKey): "active" | "revoked" | "expired" {
  if (key.isRevoked) return "revoked"
  if (key.expiresAt && new Date(key.expiresAt) < new Date()) return "expired"
  return "active"
}

function StatusBadge({ status }: { status: "active" | "revoked" | "expired" }) {
  if (status === "active") {
    return (
      <span className="inline-flex items-center rounded-full bg-green-500/15 px-2 py-0.5 text-xs font-medium text-green-600 dark:text-green-400">
        Activa
      </span>
    )
  }
  if (status === "expired") {
    return (
      <span className="inline-flex items-center rounded-full bg-orange-500/15 px-2 py-0.5 text-xs font-medium text-orange-600 dark:text-orange-400">
        Expirada
      </span>
    )
  }
  return (
    <span className="inline-flex items-center rounded-full bg-red-500/15 px-2 py-0.5 text-xs font-medium text-red-600 dark:text-red-400">
      Revocada
    </span>
  )
}

function formatDate(dateStr: string | null): string {
  if (!dateStr) return "—"
  return new Date(dateStr).toLocaleDateString("es-ES", {
    year: "numeric",
    month: "short",
    day: "numeric",
  })
}

const EXPIRY_OPTIONS: { label: string; days: number | null }[] = [
  { label: "30 días", days: 30 },
  { label: "3 meses", days: 90 },
  { label: "6 meses (por defecto)", days: 180 },
  { label: "1 año", days: 365 },
  { label: "Sin caducidad", days: null },
]

interface NewKeyDialogProps {
  onClose: () => void
  onCreated: (key: ApiKey & { key: string }) => void
}

function NewKeyDialog({ onClose, onCreated }: NewKeyDialogProps) {
  const [name, setName] = useState("")
  const [expiryDays, setExpiryDays] = useState<number | null>(180)
  const [saving, setSaving] = useState(false)

  const handleCreate = async () => {
    const trimmed = name.trim()
    if (!trimmed) {
      toast.error("El nombre de la key es obligatorio")
      return
    }
    setSaving(true)
    try {
      const result = await api.auth.createKey(trimmed, expiryDays)
      onCreated(result)
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "No se pudo crear la API key")
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
      <div className="w-full max-w-md rounded-lg border border-border bg-card p-5 shadow-lg">
        <h2 className="mb-4 text-base font-semibold">Crear nueva API Key</h2>

        <div className="space-y-4">
          <label className="block text-sm">
            <span className="mb-1 block text-muted-foreground">Nombre</span>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="ej: Mi app local"
              autoFocus
              className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-ring"
            />
          </label>

          <fieldset className="block text-sm">
            <legend className="mb-2 text-muted-foreground">Caducidad</legend>
            <div className="space-y-1.5">
              {EXPIRY_OPTIONS.map((opt) => (
                <label key={String(opt.days)} className="flex items-center gap-2">
                  <input
                    type="radio"
                    name="expiry"
                    checked={expiryDays === opt.days}
                    onChange={() => setExpiryDays(opt.days)}
                    className="accent-primary"
                  />
                  <span>{opt.label}</span>
                </label>
              ))}
            </div>
          </fieldset>
        </div>

        <div className="mt-5 flex justify-end gap-2">
          <button
            type="button"
            onClick={onClose}
            disabled={saving}
            className="rounded-md border border-border px-3 py-1.5 text-sm hover:bg-muted"
          >
            Cancelar
          </button>
          <button
            type="button"
            onClick={() => void handleCreate()}
            disabled={saving}
            className="rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-60"
          >
            {saving ? "Creando..." : "Crear"}
          </button>
        </div>
      </div>
    </div>
  )
}

interface CreatedKeyDialogProps {
  keyValue: string
  onClose: () => void
}

function CreatedKeyDialog({ keyValue, onClose }: CreatedKeyDialogProps) {
  const [copied, setCopied] = useState(false)

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(keyValue)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch {
      toast.error("No se pudo copiar al portapapeles")
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
      <div className="w-full max-w-md rounded-lg border border-border bg-card p-5 shadow-lg">
        <h2 className="mb-3 text-base font-semibold">API Key creada</h2>

        <div className="mb-4 rounded-md border border-orange-400 bg-orange-500/10 p-3">
          <p className="mb-2 text-sm font-medium text-orange-600 dark:text-orange-400">
            Copia esta key ahora. No se volverá a mostrar.
          </p>
          <code className="block break-all rounded bg-background px-2 py-1.5 text-xs font-mono">
            {keyValue}
          </code>
        </div>

        <div className="flex justify-end gap-2">
          <button
            type="button"
            onClick={() => void handleCopy()}
            className="rounded-md border border-border px-3 py-1.5 text-sm hover:bg-muted"
          >
            {copied ? "Copiado!" : "Copiar al portapapeles"}
          </button>
          <button
            type="button"
            onClick={onClose}
            className="rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground hover:bg-primary/90"
          >
            Cerrar
          </button>
        </div>
      </div>
    </div>
  )
}

interface RevokeConfirmDialogProps {
  keyName: string
  onConfirm: () => void
  onCancel: () => void
  revoking: boolean
}

function RevokeConfirmDialog({ keyName, onConfirm, onCancel, revoking }: RevokeConfirmDialogProps) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
      <div className="w-full max-w-sm rounded-lg border border-border bg-card p-5 shadow-lg">
        <h2 className="mb-2 text-base font-semibold">Revocar API Key</h2>
        <p className="mb-4 text-sm text-muted-foreground">
          ¿Seguro que quieres revocar la key <strong>{keyName}</strong>? Esta acción no se puede deshacer.
        </p>
        <div className="flex justify-end gap-2">
          <button
            type="button"
            onClick={onCancel}
            disabled={revoking}
            className="rounded-md border border-border px-3 py-1.5 text-sm hover:bg-muted"
          >
            Cancelar
          </button>
          <button
            type="button"
            onClick={onConfirm}
            disabled={revoking}
            className="rounded-md bg-destructive px-3 py-1.5 text-sm font-medium text-destructive-foreground hover:bg-destructive/90 disabled:opacity-60"
          >
            {revoking ? "Revocando..." : "Revocar"}
          </button>
        </div>
      </div>
    </div>
  )
}

export function ApiKeyManager() {
  const [keys, setKeys] = useState<ApiKey[]>([])
  const [loading, setLoading] = useState(true)
  const [showNewDialog, setShowNewDialog] = useState(false)
  const [createdKey, setCreatedKey] = useState<string | null>(null)
  const [revokingId, setRevokingId] = useState<string | null>(null)
  const [confirmRevokeKey, setConfirmRevokeKey] = useState<ApiKey | null>(null)

  const loadKeys = async () => {
    try {
      const data = await api.auth.listKeys()
      setKeys(data)
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "No se pudieron cargar las API keys")
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void loadKeys()
  }, [])

  const handleCreated = (result: ApiKey & { key: string }) => {
    setShowNewDialog(false)
    setKeys((prev) => [result, ...prev])
    setCreatedKey(result.key)
  }

  const handleRevoke = async () => {
    if (!confirmRevokeKey) return
    setRevokingId(confirmRevokeKey.id)
    try {
      await api.auth.revokeKey(confirmRevokeKey.id)
      setKeys((prev) =>
        prev.map((k) => (k.id === confirmRevokeKey.id ? { ...k, isRevoked: true } : k)),
      )
      toast.success("API key revocada")
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "No se pudo revocar la key")
    } finally {
      setRevokingId(null)
      setConfirmRevokeKey(null)
    }
  }

  return (
    <div>
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-base font-semibold">API Keys</h2>
        <button
          type="button"
          onClick={() => setShowNewDialog(true)}
          className="rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground hover:bg-primary/90"
        >
          Crear nueva key
        </button>
      </div>

      {loading ? (
        <div className="space-y-2">
          {Array.from({ length: 3 }).map((_, i) => (
            <div key={`api-key-skel-${i}`} className="h-12 animate-pulse rounded-lg bg-muted" />
          ))}
        </div>
      ) : keys.length === 0 ? (
        <p className="text-sm text-muted-foreground">No tienes API keys. Crea una para empezar.</p>
      ) : (
        <div className="overflow-x-auto rounded-lg border border-border">
          <table className="w-full text-sm">
            <thead className="bg-muted/50">
              <tr>
                <th className="px-3 py-2 text-left font-medium text-muted-foreground">Nombre</th>
                <th className="px-3 py-2 text-left font-medium text-muted-foreground">Prefijo</th>
                <th className="px-3 py-2 text-left font-medium text-muted-foreground">Expira</th>
                <th className="px-3 py-2 text-left font-medium text-muted-foreground">Último uso</th>
                <th className="px-3 py-2 text-left font-medium text-muted-foreground">Estado</th>
                <th className="px-3 py-2 text-left font-medium text-muted-foreground">Acciones</th>
              </tr>
            </thead>
            <tbody>
              {keys.map((key) => {
                const status = keyStatus(key)
                return (
                  <tr key={key.id} className="border-t border-border">
                    <td className="px-3 py-2 font-medium">{key.name}</td>
                    <td className="px-3 py-2 font-mono text-xs text-muted-foreground">
                      {key.keyPrefix}…
                    </td>
                    <td className="px-3 py-2 text-muted-foreground">{formatDate(key.expiresAt)}</td>
                    <td className="px-3 py-2 text-muted-foreground">{formatDate(key.lastUsedAt)}</td>
                    <td className="px-3 py-2">
                      <StatusBadge status={status} />
                    </td>
                    <td className="px-3 py-2">
                      {status === "active" ? (
                        <button
                          type="button"
                          onClick={() => setConfirmRevokeKey(key)}
                          disabled={revokingId === key.id}
                          className="rounded-md border border-destructive/40 px-2 py-0.5 text-xs text-destructive hover:bg-destructive/10 disabled:opacity-60"
                        >
                          Revocar
                        </button>
                      ) : (
                        <span className="text-xs text-muted-foreground">—</span>
                      )}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}

      {showNewDialog && (
        <NewKeyDialog
          onClose={() => setShowNewDialog(false)}
          onCreated={handleCreated}
        />
      )}

      {createdKey !== null && (
        <CreatedKeyDialog
          keyValue={createdKey}
          onClose={() => setCreatedKey(null)}
        />
      )}

      {confirmRevokeKey !== null && (
        <RevokeConfirmDialog
          keyName={confirmRevokeKey.name}
          onConfirm={() => void handleRevoke()}
          onCancel={() => setConfirmRevokeKey(null)}
          revoking={revokingId === confirmRevokeKey.id}
        />
      )}
    </div>
  )
}
