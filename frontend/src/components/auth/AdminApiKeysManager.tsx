import { useEffect, useMemo, useState } from "react"
import { toast } from "sonner"
import { api } from "@/api/client"
import type { AdminApiKey, AdminUser } from "@/types"

function keyStatus(key: AdminApiKey): "active" | "revoked" | "expired" {
  if (key.isRevoked) return "revoked"
  if (key.expiresAt && new Date(key.expiresAt) < new Date()) return "expired"
  return "active"
}

function StatusBadge({ status }: { status: "active" | "revoked" | "expired" }) {
  const map = {
    active: { cls: "bg-green-500/15 text-green-600 dark:text-green-400", label: "Activa" },
    expired: { cls: "bg-orange-500/15 text-orange-600 dark:text-orange-400", label: "Expirada" },
    revoked: { cls: "bg-red-500/15 text-red-600 dark:text-red-400", label: "Revocada" },
  }[status]
  return (
    <span className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${map.cls}`}>
      {map.label}
    </span>
  )
}

function formatDate(dateStr: string | null): string {
  if (!dateStr) return "—"
  return new Date(dateStr).toLocaleDateString("es-ES", { year: "numeric", month: "short", day: "numeric" })
}

interface ReassignDialogProps {
  count: number
  users: AdminUser[]
  saving: boolean
  onConfirm: (targetUserId: string) => void
  onCancel: () => void
}

function ReassignDialog({ count, users, saving, onConfirm, onCancel }: ReassignDialogProps) {
  const [targetUserId, setTargetUserId] = useState("")

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
      <div className="w-full max-w-md rounded-lg border border-border bg-card p-5 shadow-lg">
        <h2 className="mb-2 text-base font-semibold">Reasignar {count} API key{count === 1 ? "" : "s"}</h2>
        <p className="mb-4 text-sm text-muted-foreground">
          Las keys seleccionadas pasarán a pertenecer al usuario destino, que heredará su acceso (rol y grupos).
        </p>
        <label className="block text-sm">
          <span className="mb-1 block text-muted-foreground">Usuario destino</span>
          <select
            value={targetUserId}
            onChange={(e) => setTargetUserId(e.target.value)}
            autoFocus
            className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-ring"
          >
            <option value="" disabled>
              Selecciona un usuario…
            </option>
            {users.map((u) => (
              <option key={u.id} value={u.id}>
                {u.username}
                {u.email ? ` · ${u.email}` : ""} ({u.role})
                {u.isActive ? "" : " · inactivo"}
              </option>
            ))}
          </select>
        </label>
        <div className="mt-5 flex justify-end gap-2">
          <button
            type="button"
            onClick={onCancel}
            disabled={saving}
            className="rounded-md border border-border px-3 py-1.5 text-sm hover:bg-muted"
          >
            Cancelar
          </button>
          <button
            type="button"
            onClick={() => onConfirm(targetUserId)}
            disabled={saving || !targetUserId}
            className="rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-60"
          >
            {saving ? "Reasignando…" : "Reasignar"}
          </button>
        </div>
      </div>
    </div>
  )
}

export function AdminApiKeysManager() {
  const [keys, setKeys] = useState<AdminApiKey[]>([])
  const [users, setUsers] = useState<AdminUser[]>([])
  const [loading, setLoading] = useState(true)
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [showReassign, setShowReassign] = useState(false)
  const [saving, setSaving] = useState(false)
  const [showPurge, setShowPurge] = useState(false)
  const [purging, setPurging] = useState(false)

  const revokedCount = keys.filter((k) => keyStatus(k) === "revoked").length

  const load = async () => {
    setLoading(true)
    try {
      const [keysData, usersData] = await Promise.all([api.users.listAllKeys(), api.users.list()])
      setKeys(keysData)
      setUsers(usersData)
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "No se pudieron cargar las API keys")
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void load()
  }, [])

  const toggle = (id: string) => {
    setSelected((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const allSelected = keys.length > 0 && selected.size === keys.length
  const toggleAll = () => {
    setSelected(allSelected ? new Set() : new Set(keys.map((k) => k.id)))
  }

  const usersById = useMemo(() => new Map(users.map((u) => [u.id, u])), [users])

  const handlePurge = async () => {
    setPurging(true)
    try {
      const res = await api.users.purgeRevokedKeys()
      toast.success(`${res.deleted} key(s) revocada(s) eliminada(s)`)
      setSelected(new Set())
      setShowPurge(false)
      await load()
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "No se pudieron limpiar las keys revocadas")
    } finally {
      setPurging(false)
    }
  }

  const handleReassign = async (targetUserId: string) => {
    const ids = [...selected]
    setSaving(true)
    try {
      const res = await api.users.reassignKeys(ids, targetUserId)
      const target = usersById.get(targetUserId)
      const parts = [`${res.reassigned} reasignada(s) a ${target?.username ?? "usuario"}`]
      if (res.skipped) parts.push(`${res.skipped} ya pertenecían`)
      if (res.failed.length) parts.push(`${res.failed.length} fallida(s)`)
      if (res.failed.length) toast.warning(parts.join(" · "))
      else toast.success(parts.join(" · "))
      setSelected(new Set())
      setShowReassign(false)
      await load()
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "No se pudo reasignar")
    } finally {
      setSaving(false)
    }
  }

  return (
    <div>
      <div className="mb-3 flex items-center justify-between gap-3">
        <div>
          <h2 className="text-base font-semibold">Todas las API keys (admin)</h2>
          <p className="text-xs text-muted-foreground">
            Selecciona varias keys para reasignarlas a otro usuario.
          </p>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          {revokedCount > 0 && (
            <button
              type="button"
              onClick={() => setShowPurge(true)}
              className="rounded-md border border-destructive/40 px-3 py-1.5 text-sm text-destructive hover:bg-destructive/10"
            >
              Limpiar revocadas ({revokedCount})
            </button>
          )}
          {selected.size > 0 && (
            <button
              type="button"
              onClick={() => setShowReassign(true)}
              className="rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground hover:bg-primary/90"
            >
              Reasignar {selected.size} seleccionada{selected.size === 1 ? "" : "s"}
            </button>
          )}
        </div>
      </div>

      {loading ? (
        <div className="space-y-2">
          {Array.from({ length: 3 }).map((_, i) => (
            <div key={`admin-key-skel-${i}`} className="h-12 animate-pulse rounded-lg bg-muted" />
          ))}
        </div>
      ) : keys.length === 0 ? (
        <p className="text-sm text-muted-foreground">No hay API keys.</p>
      ) : (
        <div className="overflow-x-auto rounded-lg border border-border">
          <table className="w-full text-sm">
            <thead className="bg-muted/50">
              <tr>
                <th className="px-3 py-2 text-left">
                  <input
                    type="checkbox"
                    checked={allSelected}
                    onChange={toggleAll}
                    aria-label="Seleccionar todas"
                    className="accent-primary"
                  />
                </th>
                <th className="px-3 py-2 text-left font-medium text-muted-foreground">Propietario</th>
                <th className="px-3 py-2 text-left font-medium text-muted-foreground">Nombre</th>
                <th className="px-3 py-2 text-left font-medium text-muted-foreground">Prefijo</th>
                <th className="px-3 py-2 text-left font-medium text-muted-foreground">Estado</th>
                <th className="px-3 py-2 text-left font-medium text-muted-foreground">Creada</th>
              </tr>
            </thead>
            <tbody>
              {keys.map((key) => (
                <tr
                  key={key.id}
                  className={`border-t border-border ${selected.has(key.id) ? "bg-primary/5" : ""}`}
                >
                  <td className="px-3 py-2">
                    <input
                      type="checkbox"
                      checked={selected.has(key.id)}
                      onChange={() => toggle(key.id)}
                      aria-label={`Seleccionar ${key.name}`}
                      className="accent-primary"
                    />
                  </td>
                  <td className="px-3 py-2 font-medium">{key.username}</td>
                  <td className="px-3 py-2">{key.name}</td>
                  <td className="px-3 py-2 font-mono text-xs text-muted-foreground">{key.keyPrefix}…</td>
                  <td className="px-3 py-2">
                    <StatusBadge status={keyStatus(key)} />
                  </td>
                  <td className="px-3 py-2 text-muted-foreground">{formatDate(key.createdAt)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {showReassign && (
        <ReassignDialog
          count={selected.size}
          users={users}
          saving={saving}
          onConfirm={(targetUserId) => void handleReassign(targetUserId)}
          onCancel={() => setShowReassign(false)}
        />
      )}

      {showPurge && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
          <div className="w-full max-w-sm rounded-lg border border-border bg-card p-5 shadow-lg">
            <h2 className="mb-2 text-base font-semibold">Limpiar API keys revocadas</h2>
            <p className="mb-4 text-sm text-muted-foreground">
              Se eliminarán permanentemente <strong>{revokedCount}</strong> key(s) revocada(s) de todos los
              usuarios. Esta acción no se puede deshacer.
            </p>
            <div className="flex justify-end gap-2">
              <button
                type="button"
                onClick={() => setShowPurge(false)}
                disabled={purging}
                className="rounded-md border border-border px-3 py-1.5 text-sm hover:bg-muted"
              >
                Cancelar
              </button>
              <button
                type="button"
                onClick={() => void handlePurge()}
                disabled={purging}
                className="rounded-md bg-destructive px-3 py-1.5 text-sm font-medium text-destructive-foreground hover:bg-destructive/90 disabled:opacity-60"
              >
                {purging ? "Limpiando…" : "Eliminar revocadas"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
