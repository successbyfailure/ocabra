import { useEffect, useState, type FormEvent } from "react"
import * as Dialog from "@radix-ui/react-dialog"
import * as Tabs from "@radix-ui/react-tabs"
import { Plus, X, Trash2, Users as UsersIcon, Layers } from "lucide-react"
import { toast } from "sonner"
import { api } from "@/api/client"
import type { AdminUser, Group, GroupMember } from "@/types"

// ─── helpers ────────────────────────────────────────────────────────────────

function formatDate(iso: string) {
  if (!iso) return "—"
  try {
    return new Intl.DateTimeFormat("es-ES", { dateStyle: "medium" }).format(new Date(iso))
  } catch {
    return iso
  }
}

// ─── Confirm dialog ──────────────────────────────────────────────────────────

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
        <Dialog.Overlay className="fixed inset-0 z-[60] bg-black/50" />
        <Dialog.Content className="fixed left-1/2 top-1/2 z-[70] w-full max-w-sm -translate-x-1/2 -translate-y-1/2 rounded-lg border border-border bg-card p-6 shadow-xl">
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

// ─── Create group modal ───────────────────────────────────────────────────────

interface CreateGroupModalProps {
  open: boolean
  onClose: () => void
  onCreated: (group: Group) => void
}

function CreateGroupModal({ open, onClose, onCreated }: CreateGroupModalProps) {
  const [name, setName] = useState("")
  const [description, setDescription] = useState("")
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)

  function reset() { setName(""); setDescription(""); setErr(null) }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    if (!name.trim()) { setErr("El nombre es requerido."); return }
    setBusy(true)
    setErr(null)
    try {
      const group = await api.groups.create({ name: name.trim(), description: description.trim() || null })
      onCreated(group)
      reset()
      onClose()
      toast.success(`Grupo "${group.name}" creado correctamente.`)
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Error al crear el grupo.")
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
            <Dialog.Title className="text-lg font-semibold">Nuevo grupo</Dialog.Title>
            <Dialog.Close asChild>
              <button type="button" className="rounded p-1 hover:bg-muted" onClick={() => { reset(); onClose() }}>
                <X size={16} />
              </button>
            </Dialog.Close>
          </div>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label htmlFor="cg-name" className="block text-sm font-medium mb-1">Nombre *</label>
              <input
                id="cg-name"
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
                placeholder="mi-grupo"
                autoFocus
              />
            </div>
            <div>
              <label htmlFor="cg-desc" className="block text-sm font-medium mb-1">Descripción (opcional)</label>
              <textarea
                id="cg-desc"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                rows={2}
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary resize-none"
                placeholder="Descripción del grupo…"
              />
            </div>
            {err && <p className="text-sm text-destructive">{err}</p>}
            <div className="flex justify-end gap-2 pt-2">
              <button type="button" onClick={() => { reset(); onClose() }} className="rounded-md px-4 py-2 text-sm border border-border hover:bg-muted">
                Cancelar
              </button>
              <button type="submit" disabled={busy} className="rounded-md bg-primary px-4 py-2 text-sm text-primary-foreground hover:bg-primary/90 disabled:opacity-50">
                {busy ? "Creando…" : "Crear grupo"}
              </button>
            </div>
          </form>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  )
}

// ─── Group detail panel ───────────────────────────────────────────────────────

interface GroupDetailPanelProps {
  group: Group
  onClose: () => void
  onGroupUpdated: (g: Group) => void
}

function GroupDetailPanel({ group, onClose, onGroupUpdated }: GroupDetailPanelProps) {
  const [models, setModels] = useState<string[]>([])
  const [members, setMembers] = useState<GroupMember[]>([])
  const [allUsers, setAllUsers] = useState<AdminUser[]>([])
  const [loadingModels, setLoadingModels] = useState(true)
  const [loadingMembers, setLoadingMembers] = useState(true)
  const [newModelId, setNewModelId] = useState("")
  const [addingModel, setAddingModel] = useState(false)
  const [selectedUserId, setSelectedUserId] = useState("")
  const [addingMember, setAddingMember] = useState(false)
  const [confirmRemoveModel, setConfirmRemoveModel] = useState<string | null>(null)
  const [confirmRemoveMember, setConfirmRemoveMember] = useState<GroupMember | null>(null)

  useEffect(() => {
    setLoadingModels(true)
    api.groups.listModels(group.id)
      .then(setModels)
      .catch((e: unknown) => toast.error(e instanceof Error ? e.message : "Error cargando modelos."))
      .finally(() => setLoadingModels(false))

    setLoadingMembers(true)
    api.groups.listMembers(group.id)
      .then(setMembers)
      .catch((e: unknown) => toast.error(e instanceof Error ? e.message : "Error cargando miembros."))
      .finally(() => setLoadingMembers(false))

    api.users.list().then(setAllUsers).catch(() => {/* best effort */})
  }, [group.id])

  async function handleAddModel() {
    const modelId = newModelId.trim()
    if (!modelId) return
    setAddingModel(true)
    try {
      await api.groups.addModel(group.id, modelId)
      setModels((prev) => [...prev, modelId])
      setNewModelId("")
      const updated = { ...group, modelCount: group.modelCount + 1 }
      onGroupUpdated(updated)
      toast.success("Modelo añadido al grupo.")
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Error al añadir modelo.")
    } finally {
      setAddingModel(false)
    }
  }

  async function handleRemoveModel(modelId: string) {
    try {
      await api.groups.removeModel(group.id, modelId)
      setModels((prev) => prev.filter((m) => m !== modelId))
      onGroupUpdated({ ...group, modelCount: Math.max(0, group.modelCount - 1) })
      toast.success("Modelo eliminado del grupo.")
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Error al eliminar modelo.")
    } finally {
      setConfirmRemoveModel(null)
    }
  }

  async function handleAddMember() {
    if (!selectedUserId) return
    setAddingMember(true)
    try {
      await api.groups.addMember(group.id, selectedUserId)
      const user = allUsers.find((u) => u.id === selectedUserId)
      if (user) {
        setMembers((prev) => [...prev, { userId: user.id, username: user.username, role: user.role }])
        onGroupUpdated({ ...group, memberCount: group.memberCount + 1 })
      }
      setSelectedUserId("")
      toast.success("Miembro añadido al grupo.")
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Error al añadir miembro.")
    } finally {
      setAddingMember(false)
    }
  }

  async function handleRemoveMember(member: GroupMember) {
    try {
      await api.groups.removeMember(group.id, member.userId)
      setMembers((prev) => prev.filter((m) => m.userId !== member.userId))
      onGroupUpdated({ ...group, memberCount: Math.max(0, group.memberCount - 1) })
      toast.success("Miembro eliminado del grupo.")
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Error al eliminar miembro.")
    } finally {
      setConfirmRemoveMember(null)
    }
  }

  const memberIds = new Set(members.map((m) => m.userId))
  const availableUsers = allUsers.filter((u) => !memberIds.has(u.id))

  return (
    <>
      <div className="rounded-lg border border-border bg-card overflow-hidden">
        <div className="flex items-center justify-between px-4 py-3 bg-muted/50 border-b border-border">
          <div>
            <span className="font-semibold text-sm">{group.name}</span>
            {group.isDefault && (
              <span className="ml-2 inline-flex items-center rounded-full bg-yellow-100 text-yellow-700 dark:bg-yellow-900 dark:text-yellow-300 px-2 py-0.5 text-xs font-medium">
                Default
              </span>
            )}
            {group.description && (
              <p className="text-xs text-muted-foreground mt-0.5">{group.description}</p>
            )}
          </div>
          <button type="button" onClick={onClose} className="rounded p-1 hover:bg-muted">
            <X size={14} />
          </button>
        </div>

        <Tabs.Root defaultValue="models" className="p-4">
          <Tabs.List className="flex gap-1 border-b border-border mb-4">
            <Tabs.Trigger
              value="models"
              className="flex items-center gap-1.5 px-3 py-2 text-sm font-medium text-muted-foreground data-[state=active]:text-foreground data-[state=active]:border-b-2 data-[state=active]:border-primary -mb-px"
            >
              <Layers size={14} />
              Modelos ({group.modelCount})
            </Tabs.Trigger>
            <Tabs.Trigger
              value="members"
              className="flex items-center gap-1.5 px-3 py-2 text-sm font-medium text-muted-foreground data-[state=active]:text-foreground data-[state=active]:border-b-2 data-[state=active]:border-primary -mb-px"
            >
              <UsersIcon size={14} />
              Miembros ({group.memberCount})
            </Tabs.Trigger>
          </Tabs.List>

          {/* Models tab */}
          <Tabs.Content value="models" className="space-y-3">
            <div className="flex gap-2">
              <input
                type="text"
                value={newModelId}
                onChange={(e) => setNewModelId(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); handleAddModel() } }}
                className="flex-1 rounded-md border border-input bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
                placeholder="model-id o org/model"
              />
              <button
                type="button"
                onClick={handleAddModel}
                disabled={addingModel || !newModelId.trim()}
                className="flex items-center gap-1 rounded-md bg-primary px-3 py-2 text-sm text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
              >
                <Plus size={14} />
                Añadir
              </button>
            </div>

            {loadingModels ? (
              <div className="space-y-1">
                {[...Array(3)].map((_, i) => <div key={i} className="h-8 animate-pulse rounded bg-muted" />)}
              </div>
            ) : models.length === 0 ? (
              <p className="text-sm text-muted-foreground text-center py-4">No hay modelos asignados.</p>
            ) : (
              <div className="space-y-1 max-h-64 overflow-y-auto">
                {models.map((modelId) => (
                  <div key={modelId} className="flex items-center justify-between rounded-md border border-border px-3 py-2 text-sm">
                    <span className="font-mono text-xs truncate">{modelId}</span>
                    <button
                      type="button"
                      onClick={() => setConfirmRemoveModel(modelId)}
                      className="ml-2 shrink-0 rounded p-1 text-muted-foreground hover:text-destructive hover:bg-muted"
                    >
                      <X size={12} />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </Tabs.Content>

          {/* Members tab */}
          <Tabs.Content value="members" className="space-y-3">
            {!group.isDefault && (
              <div className="flex gap-2">
                <select
                  value={selectedUserId}
                  onChange={(e) => setSelectedUserId(e.target.value)}
                  className="flex-1 rounded-md border border-input bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
                >
                  <option value="">Selecciona un usuario…</option>
                  {availableUsers.map((u) => (
                    <option key={u.id} value={u.id}>{u.username} ({u.role})</option>
                  ))}
                </select>
                <button
                  type="button"
                  onClick={handleAddMember}
                  disabled={addingMember || !selectedUserId}
                  className="flex items-center gap-1 rounded-md bg-primary px-3 py-2 text-sm text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
                >
                  <Plus size={14} />
                  Añadir
                </button>
              </div>
            )}

            {loadingMembers ? (
              <div className="space-y-1">
                {[...Array(3)].map((_, i) => <div key={i} className="h-8 animate-pulse rounded bg-muted" />)}
              </div>
            ) : members.length === 0 ? (
              <p className="text-sm text-muted-foreground text-center py-4">No hay miembros.</p>
            ) : (
              <div className="space-y-1 max-h-64 overflow-y-auto">
                {members.map((member) => (
                  <div key={member.userId} className="flex items-center justify-between rounded-md border border-border px-3 py-2 text-sm">
                    <div>
                      <span className="font-medium">{member.username}</span>
                      <span className="ml-2 text-xs text-muted-foreground">{member.role}</span>
                    </div>
                    {!group.isDefault && (
                      <button
                        type="button"
                        onClick={() => setConfirmRemoveMember(member)}
                        className="ml-2 shrink-0 rounded p-1 text-muted-foreground hover:text-destructive hover:bg-muted"
                      >
                        <X size={12} />
                      </button>
                    )}
                  </div>
                ))}
              </div>
            )}
          </Tabs.Content>
        </Tabs.Root>
      </div>

      <ConfirmDialog
        open={!!confirmRemoveModel}
        title="Quitar modelo del grupo"
        description={`¿Quitar "${confirmRemoveModel}" del grupo "${group.name}"?`}
        confirmLabel="Quitar"
        destructive
        onConfirm={() => confirmRemoveModel && handleRemoveModel(confirmRemoveModel)}
        onCancel={() => setConfirmRemoveModel(null)}
      />
      <ConfirmDialog
        open={!!confirmRemoveMember}
        title="Quitar miembro del grupo"
        description={`¿Quitar a "${confirmRemoveMember?.username}" del grupo "${group.name}"?`}
        confirmLabel="Quitar"
        destructive
        onConfirm={() => confirmRemoveMember && handleRemoveMember(confirmRemoveMember)}
        onCancel={() => setConfirmRemoveMember(null)}
      />
    </>
  )
}

// ─── main page ───────────────────────────────────────────────────────────────

export function Groups() {
  const [groups, setGroups] = useState<Group[]>([])
  const [loading, setLoading] = useState(true)
  const [createOpen, setCreateOpen] = useState(false)
  const [expandedId, setExpandedId] = useState<string | null>(null)
  const [confirmDelete, setConfirmDelete] = useState<Group | null>(null)
  const [deletingId, setDeletingId] = useState<string | null>(null)

  const loadGroups = () => {
    setLoading(true)
    api.groups.list()
      .then(setGroups)
      .catch((e: unknown) => toast.error(e instanceof Error ? e.message : "Error cargando grupos."))
      .finally(() => setLoading(false))
  }

  useEffect(() => { loadGroups() }, [])

  async function handleDelete(group: Group) {
    setDeletingId(group.id)
    try {
      await api.groups.remove(group.id)
      setGroups((prev) => prev.filter((g) => g.id !== group.id))
      if (expandedId === group.id) setExpandedId(null)
      toast.success(`Grupo "${group.name}" eliminado.`)
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Error al eliminar el grupo.")
    } finally {
      setDeletingId(null)
      setConfirmDelete(null)
    }
  }

  return (
    <div className="space-y-6 p-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground">Grupos</h1>
          <p className="text-sm text-muted-foreground mt-0.5">Gestiona los grupos de acceso a modelos.</p>
        </div>
        <button
          type="button"
          onClick={() => setCreateOpen(true)}
          className="flex items-center gap-2 rounded-md bg-primary px-4 py-2 text-sm text-primary-foreground hover:bg-primary/90"
        >
          <Plus size={16} />
          Nuevo grupo
        </button>
      </div>

      {/* Group list */}
      {loading ? (
        <div className="space-y-3">
          {[...Array(3)].map((_, i) => (
            <div key={i} className="h-20 animate-pulse rounded-lg bg-muted" />
          ))}
        </div>
      ) : groups.length === 0 ? (
        <div className="rounded-lg border border-border p-8 text-center text-muted-foreground">
          No hay grupos configurados.
        </div>
      ) : (
        <div className="space-y-3">
          {groups.map((group) => (
            <div key={group.id}>
              <div className="rounded-lg border border-border bg-card hover:border-primary/50 transition-colors">
                <div className="flex items-center gap-4 px-4 py-4">
                  {/* Info */}
                  <button
                    type="button"
                    className="flex-1 text-left min-w-0"
                    onClick={() => setExpandedId(expandedId === group.id ? null : group.id)}
                  >
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="font-semibold text-sm">{group.name}</span>
                      {group.isDefault && (
                        <span className="inline-flex items-center rounded-full bg-yellow-100 text-yellow-700 dark:bg-yellow-900 dark:text-yellow-300 px-2 py-0.5 text-xs font-medium">
                          Default
                        </span>
                      )}
                    </div>
                    {group.description && (
                      <p className="text-xs text-muted-foreground mt-0.5 truncate">{group.description}</p>
                    )}
                  </button>

                  {/* Counters */}
                  <div className="hidden sm:flex items-center gap-4 text-sm text-muted-foreground shrink-0">
                    <span className="flex items-center gap-1">
                      <Layers size={14} />
                      {group.modelCount} modelo{group.modelCount !== 1 ? "s" : ""}
                    </span>
                    <span className="flex items-center gap-1">
                      <UsersIcon size={14} />
                      {group.memberCount} miembro{group.memberCount !== 1 ? "s" : ""}
                    </span>
                    <span className="text-xs">{formatDate(group.createdAt)}</span>
                  </div>

                  {/* Actions */}
                  {!group.isDefault && (
                    <button
                      type="button"
                      title="Eliminar grupo"
                      onClick={() => setConfirmDelete(group)}
                      disabled={deletingId === group.id}
                      className="shrink-0 rounded p-1.5 text-muted-foreground hover:text-destructive hover:bg-muted disabled:opacity-50"
                    >
                      <Trash2 size={14} />
                    </button>
                  )}
                </div>
              </div>

              {/* Expanded detail */}
              {expandedId === group.id && (
                <div className="mt-2">
                  <GroupDetailPanel
                    group={group}
                    onClose={() => setExpandedId(null)}
                    onGroupUpdated={(updated) => setGroups((prev) => prev.map((g) => g.id === updated.id ? updated : g))}
                  />
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      <CreateGroupModal
        open={createOpen}
        onClose={() => setCreateOpen(false)}
        onCreated={(g) => setGroups((prev) => [...prev, g])}
      />

      <ConfirmDialog
        open={!!confirmDelete}
        title={`Eliminar grupo "${confirmDelete?.name}"`}
        description="Esta acción no se puede deshacer. Se eliminará el grupo y todas sus asignaciones."
        confirmLabel="Eliminar"
        destructive
        onConfirm={() => confirmDelete && handleDelete(confirmDelete)}
        onCancel={() => setConfirmDelete(null)}
      />
    </div>
  )
}
