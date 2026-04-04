import { useEffect, useState, type FormEvent } from "react"
import * as Dialog from "@radix-ui/react-dialog"
import * as Select from "@radix-ui/react-select"
import { ChevronDown, ChevronUp, Plus, X, KeyRound, RotateCcw, Pencil, Trash2, ShieldOff, ShieldCheck } from "lucide-react"
import { toast } from "sonner"
import { api } from "@/api/client"
import type { AdminUser, ApiKey, UserRole } from "@/types"

// ─── helpers ────────────────────────────────────────────────────────────────

function roleBadge(role: UserRole) {
  const cls: Record<UserRole, string> = {
    user: "bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300",
    model_manager: "bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300",
    system_admin: "bg-purple-100 text-purple-700 dark:bg-purple-900 dark:text-purple-300",
  }
  const label: Record<UserRole, string> = {
    user: "Usuario",
    model_manager: "Manager",
    system_admin: "Admin",
  }
  return (
    <span className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${cls[role]}`}>
      {label[role]}
    </span>
  )
}

function formatDate(iso: string) {
  if (!iso) return "—"
  try {
    return new Intl.DateTimeFormat("es-ES", { dateStyle: "medium" }).format(new Date(iso))
  } catch {
    return iso
  }
}

// ─── sub-components ──────────────────────────────────────────────────────────

interface SelectRoleProps {
  value: UserRole
  onChange: (v: UserRole) => void
  id?: string
}

function SelectRole({ value, onChange, id }: SelectRoleProps) {
  return (
    <Select.Root value={value} onValueChange={(v) => onChange(v as UserRole)}>
      <Select.Trigger
        id={id}
        className="flex w-full items-center justify-between rounded-md border border-input bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
      >
        <Select.Value />
        <Select.Icon><ChevronDown size={14} /></Select.Icon>
      </Select.Trigger>
      <Select.Portal>
        <Select.Content className="z-50 rounded-md border border-border bg-popover shadow-md">
          <Select.ScrollUpButton className="flex items-center justify-center py-1">
            <ChevronUp size={14} />
          </Select.ScrollUpButton>
          <Select.Viewport className="p-1">
            {(["user", "model_manager", "system_admin"] as UserRole[]).map((r) => (
              <Select.Item
                key={r}
                value={r}
                className="relative cursor-pointer select-none rounded px-3 py-1.5 text-sm outline-none hover:bg-muted"
              >
                <Select.ItemText>{r === "user" ? "Usuario" : r === "model_manager" ? "Manager" : "Administrador"}</Select.ItemText>
              </Select.Item>
            ))}
          </Select.Viewport>
          <Select.ScrollDownButton className="flex items-center justify-center py-1">
            <ChevronDown size={14} />
          </Select.ScrollDownButton>
        </Select.Content>
      </Select.Portal>
    </Select.Root>
  )
}

// ─── modals ──────────────────────────────────────────────────────────────────

interface CreateUserModalProps {
  open: boolean
  onClose: () => void
  onCreated: (user: AdminUser) => void
}

function CreateUserModal({ open, onClose, onCreated }: CreateUserModalProps) {
  const [username, setUsername] = useState("")
  const [password, setPassword] = useState("")
  const [role, setRole] = useState<UserRole>("user")
  const [email, setEmail] = useState("")
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)

  function reset() {
    setUsername(""); setPassword(""); setRole("user"); setEmail(""); setErr(null)
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    setErr(null)
    if (!username.trim()) { setErr("El nombre de usuario es requerido."); return }
    if (!password.trim()) { setErr("La contraseña es requerida."); return }
    setBusy(true)
    try {
      const user = await api.users.create({
        username: username.trim(),
        password,
        role,
        email: email.trim() || null,
      })
      onCreated(user)
      reset()
      onClose()
      toast.success(`Usuario "${user.username}" creado correctamente.`)
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Error al crear el usuario.")
    } finally {
      setBusy(false)
    }
  }

  return (
    <Dialog.Root open={open} onOpenChange={(v) => { if (!v) { reset(); onClose() } }}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-40 bg-black/50" />
        <Dialog.Content className="fixed left-1/2 top-1/2 z-50 w-full max-w-md -translate-x-1/2 -translate-y-1/2 rounded-lg border border-border bg-card p-6 shadow-xl">
          <div className="mb-4 flex items-center justify-between">
            <Dialog.Title className="text-lg font-semibold">Nuevo usuario</Dialog.Title>
            <Dialog.Close asChild>
              <button type="button" className="rounded p-1 hover:bg-muted" onClick={() => { reset(); onClose() }}>
                <X size={16} />
              </button>
            </Dialog.Close>
          </div>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label htmlFor="cu-username" className="block text-sm font-medium mb-1">Username *</label>
              <input
                id="cu-username"
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
                placeholder="username"
                autoFocus
              />
            </div>
            <div>
              <label htmlFor="cu-password" className="block text-sm font-medium mb-1">Contraseña *</label>
              <input
                id="cu-password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
                placeholder="••••••••"
              />
            </div>
            <div>
              <label htmlFor="cu-role" className="block text-sm font-medium mb-1">Rol</label>
              <SelectRole id="cu-role" value={role} onChange={setRole} />
            </div>
            <div>
              <label htmlFor="cu-email" className="block text-sm font-medium mb-1">Email (opcional)</label>
              <input
                id="cu-email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
                placeholder="usuario@ejemplo.com"
              />
            </div>
            {err && <p className="text-sm text-destructive">{err}</p>}
            <div className="flex justify-end gap-2 pt-2">
              <button
                type="button"
                onClick={() => { reset(); onClose() }}
                className="rounded-md px-4 py-2 text-sm border border-border hover:bg-muted"
              >
                Cancelar
              </button>
              <button
                type="submit"
                disabled={busy}
                className="rounded-md bg-primary px-4 py-2 text-sm text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
              >
                {busy ? "Creando…" : "Crear usuario"}
              </button>
            </div>
          </form>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  )
}

interface EditUserModalProps {
  user: AdminUser | null
  onClose: () => void
  onUpdated: (user: AdminUser) => void
}

function EditUserModal({ user, onClose, onUpdated }: EditUserModalProps) {
  const [role, setRole] = useState<UserRole>("user")
  const [email, setEmail] = useState("")
  const [isActive, setIsActive] = useState(true)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    if (user) {
      setRole(user.role)
      setEmail(user.email ?? "")
      setIsActive(user.isActive)
      setErr(null)
    }
  }, [user])

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    if (!user) return
    setBusy(true)
    setErr(null)
    try {
      const updated = await api.users.update(user.id, {
        role,
        isActive,
        email: email.trim() || null,
      })
      onUpdated(updated)
      onClose()
      toast.success(`Usuario "${updated.username}" actualizado.`)
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Error al actualizar.")
    } finally {
      setBusy(false)
    }
  }

  return (
    <Dialog.Root open={!!user} onOpenChange={(v) => { if (!v) onClose() }}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-40 bg-black/50" />
        <Dialog.Content className="fixed left-1/2 top-1/2 z-50 w-full max-w-md -translate-x-1/2 -translate-y-1/2 rounded-lg border border-border bg-card p-6 shadow-xl">
          <div className="mb-4 flex items-center justify-between">
            <Dialog.Title className="text-lg font-semibold">
              Editar usuario — <span className="font-mono">{user?.username}</span>
            </Dialog.Title>
            <Dialog.Close asChild>
              <button type="button" className="rounded p-1 hover:bg-muted" onClick={onClose}>
                <X size={16} />
              </button>
            </Dialog.Close>
          </div>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label htmlFor="eu-role" className="block text-sm font-medium mb-1">Rol</label>
              <SelectRole id="eu-role" value={role} onChange={setRole} />
            </div>
            <div>
              <label htmlFor="eu-email" className="block text-sm font-medium mb-1">Email</label>
              <input
                id="eu-email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
                placeholder="usuario@ejemplo.com"
              />
            </div>
            <div className="flex items-center gap-3">
              <input
                id="eu-active"
                type="checkbox"
                checked={isActive}
                onChange={(e) => setIsActive(e.target.checked)}
                className="h-4 w-4 rounded border-border"
              />
              <label htmlFor="eu-active" className="text-sm font-medium">Cuenta activa</label>
            </div>
            {err && <p className="text-sm text-destructive">{err}</p>}
            <div className="flex justify-end gap-2 pt-2">
              <button type="button" onClick={onClose} className="rounded-md px-4 py-2 text-sm border border-border hover:bg-muted">
                Cancelar
              </button>
              <button type="submit" disabled={busy} className="rounded-md bg-primary px-4 py-2 text-sm text-primary-foreground hover:bg-primary/90 disabled:opacity-50">
                {busy ? "Guardando…" : "Guardar cambios"}
              </button>
            </div>
          </form>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  )
}

interface ResetPasswordModalProps {
  user: AdminUser | null
  onClose: () => void
}

function ResetPasswordModal({ user, onClose }: ResetPasswordModalProps) {
  const [password, setPassword] = useState("")
  const [confirm, setConfirm] = useState("")
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)

  function reset() { setPassword(""); setConfirm(""); setErr(null) }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    if (!user) return
    if (!password) { setErr("Introduce la nueva contraseña."); return }
    if (password !== confirm) { setErr("Las contraseñas no coinciden."); return }
    setBusy(true)
    setErr(null)
    try {
      await api.users.resetPassword(user.id, password)
      toast.success(`Contraseña de "${user.username}" restablecida.`)
      reset()
      onClose()
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Error al restablecer la contraseña.")
    } finally {
      setBusy(false)
    }
  }

  return (
    <Dialog.Root open={!!user} onOpenChange={(v) => { if (!v) { reset(); onClose() } }}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-40 bg-black/50" />
        <Dialog.Content className="fixed left-1/2 top-1/2 z-50 w-full max-w-md -translate-x-1/2 -translate-y-1/2 rounded-lg border border-border bg-card p-6 shadow-xl">
          <div className="mb-4 flex items-center justify-between">
            <Dialog.Title className="text-lg font-semibold">
              Restablecer contraseña — <span className="font-mono">{user?.username}</span>
            </Dialog.Title>
            <Dialog.Close asChild>
              <button type="button" className="rounded p-1 hover:bg-muted" onClick={() => { reset(); onClose() }}>
                <X size={16} />
              </button>
            </Dialog.Close>
          </div>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label htmlFor="rp-password" className="block text-sm font-medium mb-1">Nueva contraseña</label>
              <input
                id="rp-password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
                placeholder="••••••••"
                autoFocus
              />
            </div>
            <div>
              <label htmlFor="rp-confirm" className="block text-sm font-medium mb-1">Confirmar contraseña</label>
              <input
                id="rp-confirm"
                type="password"
                value={confirm}
                onChange={(e) => setConfirm(e.target.value)}
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
                placeholder="••••••••"
              />
            </div>
            {err && <p className="text-sm text-destructive">{err}</p>}
            <div className="flex justify-end gap-2 pt-2">
              <button type="button" onClick={() => { reset(); onClose() }} className="rounded-md px-4 py-2 text-sm border border-border hover:bg-muted">
                Cancelar
              </button>
              <button type="submit" disabled={busy} className="rounded-md bg-primary px-4 py-2 text-sm text-primary-foreground hover:bg-primary/90 disabled:opacity-50">
                {busy ? "Guardando…" : "Restablecer"}
              </button>
            </div>
          </form>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  )
}

interface ApiKeysModalProps {
  user: AdminUser | null
  onClose: () => void
}

function ApiKeysModal({ user, onClose }: ApiKeysModalProps) {
  const [keys, setKeys] = useState<ApiKey[]>([])
  const [loading, setLoading] = useState(false)
  const [revoking, setRevoking] = useState<string | null>(null)

  useEffect(() => {
    if (!user) return
    setLoading(true)
    api.users.listKeys(user.id)
      .then(setKeys)
      .catch((e: unknown) => toast.error(e instanceof Error ? e.message : "Error cargando keys."))
      .finally(() => setLoading(false))
  }, [user])

  async function handleRevoke(keyId: string) {
    if (!user) return
    if (!confirm("¿Revocar esta API key?")) return
    setRevoking(keyId)
    try {
      await api.users.revokeKey(user.id, keyId)
      setKeys((prev) => prev.filter((k) => k.id !== keyId))
      toast.success("API key revocada.")
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Error al revocar.")
    } finally {
      setRevoking(null)
    }
  }

  return (
    <Dialog.Root open={!!user} onOpenChange={(v) => { if (!v) onClose() }}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-40 bg-black/50" />
        <Dialog.Content className="fixed left-1/2 top-1/2 z-50 w-full max-w-lg -translate-x-1/2 -translate-y-1/2 rounded-lg border border-border bg-card p-6 shadow-xl">
          <div className="mb-4 flex items-center justify-between">
            <Dialog.Title className="text-lg font-semibold">
              API Keys — <span className="font-mono">{user?.username}</span>
            </Dialog.Title>
            <Dialog.Close asChild>
              <button type="button" className="rounded p-1 hover:bg-muted" onClick={onClose}>
                <X size={16} />
              </button>
            </Dialog.Close>
          </div>
          {loading ? (
            <div className="flex justify-center py-8">
              <div className="h-6 w-6 animate-spin rounded-full border-4 border-primary border-t-transparent" />
            </div>
          ) : keys.length === 0 ? (
            <p className="text-sm text-muted-foreground text-center py-6">No hay API keys.</p>
          ) : (
            <div className="space-y-2 max-h-80 overflow-y-auto">
              {keys.map((k) => (
                <div key={k.id} className={`flex items-start justify-between rounded-md border border-border p-3 text-sm gap-3 ${k.isRevoked ? "opacity-50" : ""}`}>
                  <div className="min-w-0">
                    <p className="font-medium truncate">{k.name || "(sin nombre)"}</p>
                    <p className="text-xs text-muted-foreground font-mono">{k.keyPrefix}…</p>
                    <p className="text-xs text-muted-foreground">
                      Expira: {k.expiresAt ? formatDate(k.expiresAt) : "Nunca"} ·
                      Último uso: {k.lastUsedAt ? formatDate(k.lastUsedAt) : "Nunca"}
                    </p>
                    {k.isRevoked && <span className="text-xs text-destructive font-medium">Revocada</span>}
                  </div>
                  {!k.isRevoked && (
                    <button
                      type="button"
                      onClick={() => handleRevoke(k.id)}
                      disabled={revoking === k.id}
                      className="shrink-0 rounded px-2 py-1 text-xs border border-destructive text-destructive hover:bg-destructive/10 disabled:opacity-50"
                    >
                      {revoking === k.id ? "…" : "Revocar"}
                    </button>
                  )}
                </div>
              ))}
            </div>
          )}
          <div className="mt-4 flex justify-end">
            <button type="button" onClick={onClose} className="rounded-md px-4 py-2 text-sm border border-border hover:bg-muted">
              Cerrar
            </button>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  )
}

interface ConfirmDialogProps {
  open: boolean
  title: string
  description: string
  confirmLabel?: string
  destructive?: boolean
  onConfirm: () => void
  onCancel: () => void
}

function ConfirmDialog({ open, title, description, confirmLabel = "Confirmar", destructive = false, onConfirm, onCancel }: ConfirmDialogProps) {
  return (
    <Dialog.Root open={open} onOpenChange={(v) => { if (!v) onCancel() }}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-50 bg-black/50" />
        <Dialog.Content className="fixed left-1/2 top-1/2 z-50 w-full max-w-sm -translate-x-1/2 -translate-y-1/2 rounded-lg border border-border bg-card p-6 shadow-xl">
          <Dialog.Title className="text-base font-semibold mb-2">{title}</Dialog.Title>
          <Dialog.Description className="text-sm text-muted-foreground mb-4">{description}</Dialog.Description>
          <div className="flex justify-end gap-2">
            <button type="button" onClick={onCancel} className="rounded-md px-4 py-2 text-sm border border-border hover:bg-muted">
              Cancelar
            </button>
            <button
              type="button"
              onClick={onConfirm}
              className={`rounded-md px-4 py-2 text-sm text-white ${destructive ? "bg-destructive hover:bg-destructive/90" : "bg-primary hover:bg-primary/90"}`}
            >
              {confirmLabel}
            </button>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  )
}

// ─── main page ───────────────────────────────────────────────────────────────

export function Users() {
  const [users, setUsers] = useState<AdminUser[]>([])
  const [loading, setLoading] = useState(true)
  const [createOpen, setCreateOpen] = useState(false)
  const [editUser, setEditUser] = useState<AdminUser | null>(null)
  const [resetUser, setResetUser] = useState<AdminUser | null>(null)
  const [keysUser, setKeysUser] = useState<AdminUser | null>(null)
  const [confirmDelete, setConfirmDelete] = useState<AdminUser | null>(null)
  const [confirmToggle, setConfirmToggle] = useState<AdminUser | null>(null)
  const [busy, setBusy] = useState<string | null>(null)

  const loadUsers = () => {
    setLoading(true)
    api.users.list()
      .then(setUsers)
      .catch((e: unknown) => toast.error(e instanceof Error ? e.message : "Error cargando usuarios."))
      .finally(() => setLoading(false))
  }

  useEffect(() => { loadUsers() }, [])

  async function handleDelete(user: AdminUser) {
    setBusy(user.id)
    try {
      await api.users.remove(user.id)
      setUsers((prev) => prev.filter((u) => u.id !== user.id))
      toast.success(`Usuario "${user.username}" eliminado.`)
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Error al eliminar.")
    } finally {
      setBusy(null)
      setConfirmDelete(null)
    }
  }

  async function handleToggleActive(user: AdminUser) {
    setBusy(user.id)
    try {
      const updated = await api.users.update(user.id, { isActive: !user.isActive })
      setUsers((prev) => prev.map((u) => u.id === updated.id ? updated : u))
      toast.success(`Usuario "${updated.username}" ${updated.isActive ? "activado" : "desactivado"}.`)
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Error al cambiar estado.")
    } finally {
      setBusy(null)
      setConfirmToggle(null)
    }
  }

  return (
    <div className="space-y-6 p-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground">Usuarios</h1>
          <p className="text-sm text-muted-foreground mt-0.5">Gestiona las cuentas de usuario del sistema.</p>
        </div>
        <button
          type="button"
          onClick={() => setCreateOpen(true)}
          className="flex items-center gap-2 rounded-md bg-primary px-4 py-2 text-sm text-primary-foreground hover:bg-primary/90"
        >
          <Plus size={16} />
          Nuevo usuario
        </button>
      </div>

      {/* Table */}
      {loading ? (
        <div className="space-y-2">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="h-12 animate-pulse rounded-md bg-muted" />
          ))}
        </div>
      ) : users.length === 0 ? (
        <div className="rounded-lg border border-border p-8 text-center text-muted-foreground">
          No hay usuarios registrados.
        </div>
      ) : (
        <div className="overflow-auto rounded-lg border border-border">
          <table className="min-w-full divide-y divide-border text-sm">
            <thead className="bg-muted/50">
              <tr>
                <th className="px-4 py-3 text-left font-medium text-muted-foreground">Username</th>
                <th className="px-4 py-3 text-left font-medium text-muted-foreground">Email</th>
                <th className="px-4 py-3 text-left font-medium text-muted-foreground">Rol</th>
                <th className="px-4 py-3 text-left font-medium text-muted-foreground">Estado</th>
                <th className="px-4 py-3 text-left font-medium text-muted-foreground">Creado</th>
                <th className="px-4 py-3 text-right font-medium text-muted-foreground">Acciones</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border bg-card">
              {users.map((user) => (
                <tr key={user.id} className="hover:bg-muted/30 transition-colors">
                  <td className="px-4 py-3 font-mono font-medium">{user.username}</td>
                  <td className="px-4 py-3 text-muted-foreground">{user.email ?? "—"}</td>
                  <td className="px-4 py-3">{roleBadge(user.role)}</td>
                  <td className="px-4 py-3">
                    <span className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${user.isActive ? "bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300" : "bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300"}`}>
                      {user.isActive ? "Activo" : "Inactivo"}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-muted-foreground">{formatDate(user.createdAt)}</td>
                  <td className="px-4 py-3">
                    <div className="flex items-center justify-end gap-1">
                      <button
                        type="button"
                        title="Editar"
                        onClick={() => setEditUser(user)}
                        className="rounded p-1.5 hover:bg-muted text-muted-foreground hover:text-foreground"
                      >
                        <Pencil size={14} />
                      </button>
                      <button
                        type="button"
                        title="Restablecer contraseña"
                        onClick={() => setResetUser(user)}
                        className="rounded p-1.5 hover:bg-muted text-muted-foreground hover:text-foreground"
                      >
                        <RotateCcw size={14} />
                      </button>
                      <button
                        type="button"
                        title="Ver API Keys"
                        onClick={() => setKeysUser(user)}
                        className="rounded p-1.5 hover:bg-muted text-muted-foreground hover:text-foreground"
                      >
                        <KeyRound size={14} />
                      </button>
                      <button
                        type="button"
                        title={user.isActive ? "Desactivar" : "Activar"}
                        onClick={() => setConfirmToggle(user)}
                        disabled={busy === user.id}
                        className="rounded p-1.5 hover:bg-muted text-muted-foreground hover:text-foreground disabled:opacity-50"
                      >
                        {user.isActive ? <ShieldOff size={14} /> : <ShieldCheck size={14} />}
                      </button>
                      <button
                        type="button"
                        title="Eliminar"
                        onClick={() => setConfirmDelete(user)}
                        disabled={busy === user.id}
                        className="rounded p-1.5 hover:bg-muted text-destructive hover:text-destructive disabled:opacity-50"
                      >
                        <Trash2 size={14} />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Modals */}
      <CreateUserModal
        open={createOpen}
        onClose={() => setCreateOpen(false)}
        onCreated={(u) => setUsers((prev) => [...prev, u])}
      />
      <EditUserModal
        user={editUser}
        onClose={() => setEditUser(null)}
        onUpdated={(u) => setUsers((prev) => prev.map((x) => x.id === u.id ? u : x))}
      />
      <ResetPasswordModal
        user={resetUser}
        onClose={() => setResetUser(null)}
      />
      <ApiKeysModal
        user={keysUser}
        onClose={() => setKeysUser(null)}
      />
      <ConfirmDialog
        open={!!confirmDelete}
        title={`Eliminar usuario "${confirmDelete?.username}"`}
        description="Esta acción no se puede deshacer. Se eliminará la cuenta y todas sus API keys."
        confirmLabel="Eliminar"
        destructive
        onConfirm={() => confirmDelete && handleDelete(confirmDelete)}
        onCancel={() => setConfirmDelete(null)}
      />
      <ConfirmDialog
        open={!!confirmToggle}
        title={confirmToggle?.isActive ? `Desactivar "${confirmToggle.username}"` : `Activar "${confirmToggle?.username}"`}
        description={confirmToggle?.isActive
          ? "El usuario no podrá iniciar sesión hasta que sea reactivado."
          : "El usuario podrá volver a iniciar sesión."}
        confirmLabel={confirmToggle?.isActive ? "Desactivar" : "Activar"}
        destructive={confirmToggle?.isActive}
        onConfirm={() => confirmToggle && handleToggleActive(confirmToggle)}
        onCancel={() => setConfirmToggle(null)}
      />
    </div>
  )
}
