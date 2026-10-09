import { useEffect, useMemo, useRef, useState } from "react"
import { AlertCircle, ArrowDown, ArrowUp, Copy, GitBranch, Loader2, Pencil, Plus, RefreshCw, Save, Search, Trash2, X } from "lucide-react"
import { toast } from "sonner"
import { api, type RoutingDecisionsResponse } from "@/api/client"
import type { ModelProfile, ModelState, ProfileCategory } from "@/types"

const ROUTER_CATEGORIES: { value: ProfileCategory; label: string }[] = [
  { value: "llm", label: "LLM" },
  { value: "tts", label: "TTS" },
  { value: "stt", label: "STT" },
  { value: "image", label: "Image" },
  { value: "music", label: "Music" },
]

/**
 * Bloque 20 — dedicated admin surface for router profiles.
 *
 * A "router" is a profile whose ``routing_targets`` is a non-empty list.
 * When a client hits its ``profile_id`` the resolver walks that list in
 * order and returns the first target that is loaded (or loadable without
 * disrupting a busy neighbour).
 *
 * This page consolidates what used to require jumping between Models →
 * expand → edit profile: browse every router in one list, see live
 * status of each target, reorder the priorities and remove/add targets
 * inline, and review the last-7-days routing distribution per router.
 */
type WorkerStatus = "loaded" | "loading" | "unloaded" | "error" | "configured" | "unknown"

interface RouterTarget {
  profileId: string
  status: WorkerStatus
  errorMessage?: string | null
}

interface RouterCardData {
  router: ModelProfile
  targets: RouterTarget[]
  /** True when the ``targets`` order differs from ``router.routingTargets``. */
  dirty: boolean
}

function statusPill(status: WorkerStatus) {
  const map: Record<WorkerStatus, { cls: string; label: string }> = {
    loaded: { cls: "border-emerald-500/40 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400", label: "loaded" },
    loading: { cls: "border-sky-500/40 bg-sky-500/10 text-sky-600 dark:text-sky-400", label: "loading" },
    unloaded: { cls: "border-border bg-muted/40 text-muted-foreground", label: "unloaded" },
    configured: { cls: "border-border bg-muted/40 text-muted-foreground", label: "configured" },
    error: { cls: "border-destructive/40 bg-destructive/10 text-destructive", label: "error" },
    unknown: { cls: "border-border bg-muted/20 text-muted-foreground", label: "—" },
  }
  const { cls, label } = map[status]
  return (
    <span className={`inline-flex items-center rounded border px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide ${cls}`}>
      {label}
    </span>
  )
}

function coerceStatus(raw: string | undefined): WorkerStatus {
  if (!raw) return "unknown"
  const v = raw.toLowerCase()
  if (v === "loaded" || v === "loading" || v === "unloaded" || v === "error" || v === "configured") {
    return v
  }
  return "unknown"
}

export function Routers() {
  const [cards, setCards] = useState<RouterCardData[]>([])
  const [profilesById, setProfilesById] = useState<Map<string, ModelProfile>>(new Map())
  const [statesByModelId, setStatesByModelId] = useState<Map<string, ModelState>>(new Map())
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [routingStats, setRoutingStats] = useState<RoutingDecisionsResponse | null>(null)
  const [saving, setSaving] = useState<string | null>(null)
  const [addingTo, setAddingTo] = useState<string | null>(null)
  const [addTargetValue, setAddTargetValue] = useState("")
  const [showCreate, setShowCreate] = useState(false)
  const [editingRouter, setEditingRouter] = useState<ModelProfile | null>(null)
  const [cloningRouter, setCloningRouter] = useState<ModelProfile | null>(null)

  const refresh = async () => {
    setLoading(true)
    setError(null)
    try {
      const [profiles, models] = await Promise.all([
        api.profiles.listAll(),
        api.models.list(),
      ])
      const byId = new Map(profiles.map((p) => [p.profileId, p]))
      setProfilesById(byId)
      const stateMap = new Map(models.map((m) => [m.modelId, m]))
      setStatesByModelId(stateMap)

      // A profile counts as a router whenever its ``routingTargets`` is a
      // list, even an empty one — ``null`` means "regular profile, never
      // configured as a router". This distinction lets a router that just
      // had its last target removed stay visible in /routers so the user can
      // repopulate it or delete it; previously an empty list hid the card
      // and left an orphan profile behind.
      const routerProfiles = profiles.filter((p) => Array.isArray(p.routingTargets))
      const built: RouterCardData[] = routerProfiles.map((router) => {
        const targets = (router.routingTargets ?? []).map((targetId) => {
          const target = byId.get(targetId)
          if (!target) {
            return { profileId: targetId, status: "unknown" as WorkerStatus, errorMessage: null }
          }
          const modelState = stateMap.get(target.baseModelId)
          return {
            profileId: targetId,
            status: coerceStatus(modelState?.status),
            errorMessage: modelState?.errorMessage ?? null,
          }
        })
        return { router, targets, dirty: false }
      })
      // Preserve any locally-dirty cards (user reordered but has not saved yet)
      // so the periodic refresh doesn't wipe their pending moves.
      setCards((prev) => {
        const dirtyById = new Map(
          prev.filter((c) => c.dirty).map((c) => [c.router.profileId, c]),
        )
        return built.map((b) => dirtyById.get(b.router.profileId) ?? b)
      })

      // Best-effort: routing stats last 7 days
      const to = new Date().toISOString()
      const from = new Date(Date.now() - 7 * 24 * 3600 * 1000).toISOString()
      try {
        const stats = await api.routingStats({ from, to })
        setRoutingStats(stats)
      } catch {
        setRoutingStats(null)
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void refresh()
    const handle = window.setInterval(() => void refresh(), 20_000)
    return () => window.clearInterval(handle)
  }, [])

  const candidatesForAdd = useMemo(() => {
    // Any enabled non-router profile that isn't already in the router's list
    const forRouter = cards.find((c) => c.router.profileId === addingTo)
    if (!forRouter) return [] as ModelProfile[]
    const already = new Set(forRouter.targets.map((t) => t.profileId))
    already.add(forRouter.router.profileId)
    return Array.from(profilesById.values())
      .filter((p) => p.enabled && !already.has(p.profileId))
      // Only non-router profiles are eligible — ``routingTargets === null``
      // means "regular profile", a list (even empty) marks it as a router
      // and routers can't be nested to avoid resolver loops.
      .filter((p) => p.routingTargets === null)
      .sort((a, b) => a.profileId.localeCompare(b.profileId))
  }, [cards, addingTo, profilesById])

  const moveTarget = (routerId: string, from: number, direction: -1 | 1) => {
    setCards((prev) =>
      prev.map((c) => {
        if (c.router.profileId !== routerId) return c
        const to = from + direction
        if (to < 0 || to >= c.targets.length) return c
        const targets = [...c.targets]
        ;[targets[from], targets[to]] = [targets[to], targets[from]]
        return { ...c, targets, dirty: true }
      }),
    )
  }

  const persistTargets = async (routerId: string, targetIds: string[]) => {
    setSaving(routerId)
    try {
      await api.profiles.update(routerId, { routingTargets: targetIds })
      await refresh()
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Error guardando router")
      await refresh()
    } finally {
      setSaving(null)
    }
  }

  const removeTarget = (routerId: string, index: number) => {
    const card = cards.find((c) => c.router.profileId === routerId)
    if (!card) return
    if (card.targets.length === 1) {
      if (
        !window.confirm(
          `Vas a quitar el último destino del router "${routerId}". Mientras esté vacío, las peticiones resolverán directo al base model (${card.router.baseModelId}). Para eliminar el router del todo usa el botón de la papelera. ¿Continuar?`,
        )
      ) {
        return
      }
    }
    const newTargets = card.targets.filter((_, i) => i !== index).map((t) => t.profileId)
    void persistTargets(routerId, newTargets)
  }

  const addTarget = (routerId: string, targetId: string) => {
    if (!targetId) return
    const card = cards.find((c) => c.router.profileId === routerId)
    if (!card) return
    if (card.targets.some((t) => t.profileId === targetId)) {
      toast.error("Ese destino ya está en el router")
      return
    }
    const newTargets = [...card.targets.map((t) => t.profileId), targetId]
    setAddingTo(null)
    setAddTargetValue("")
    void persistTargets(routerId, newTargets)
  }

  const deleteRouter = async (routerId: string) => {
    if (!window.confirm(`¿Eliminar el router "${routerId}"? Esto borra también el perfil.`)) {
      return
    }
    setSaving(routerId)
    try {
      await api.profiles.delete(routerId)
      toast.success(`Router ${routerId} eliminado`)
      await refresh()
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Error eliminando router")
    } finally {
      setSaving(null)
    }
  }

  const save = async (routerId: string) => {
    const card = cards.find((c) => c.router.profileId === routerId)
    if (!card) return
    setSaving(routerId)
    try {
      await api.profiles.update(routerId, {
        routingTargets: card.targets.map((t) => t.profileId),
      })
      toast.success(`Router ${routerId} actualizado`)
      await refresh()
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Error guardando router")
    } finally {
      setSaving(null)
    }
  }

  const statsForRouter = (routerId: string) => routingStats?.routers.find((r) => r.routerProfileId === routerId)

  return (
    <div className="space-y-4 p-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold">Routers</h1>
          <p className="text-sm text-muted-foreground">
            Perfiles con <code>routing_targets</code>. Cuando un cliente llama a este
            perfil, el resolver recorre la lista en orden y devuelve el primer
            objetivo cargado (o cargable sin desalojar workers ocupados).
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setShowCreate(true)}
            className="inline-flex items-center gap-1 rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground hover:bg-primary/90"
          >
            <Plus size={14} />
            Nuevo router
          </button>
          <button
            type="button"
            onClick={() => void refresh()}
            disabled={loading}
            className="inline-flex items-center gap-1 rounded-md border border-border px-3 py-1.5 text-sm hover:bg-muted disabled:opacity-50"
          >
            <RefreshCw size={14} className={loading ? "animate-spin" : ""} />
            Actualizar
          </button>
        </div>
      </div>

      {showCreate && (
        <CreateRouterModal
          models={Array.from(statesByModelId.values())}
          profiles={Array.from(profilesById.values())}
          onClose={() => setShowCreate(false)}
          onCreated={async () => {
            setShowCreate(false)
            await refresh()
          }}
        />
      )}

      {editingRouter && (
        <EditRouterModal
          router={editingRouter}
          existingIds={Array.from(profilesById.keys())}
          onClose={() => setEditingRouter(null)}
          onSaved={async () => {
            setEditingRouter(null)
            await refresh()
          }}
        />
      )}

      {cloningRouter && (
        <CloneRouterModal
          source={cloningRouter}
          existingIds={Array.from(profilesById.keys())}
          onClose={() => setCloningRouter(null)}
          onCloned={async () => {
            setCloningRouter(null)
            await refresh()
          }}
        />
      )}

      {error && (
        <div className="flex items-start gap-2 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
          <AlertCircle size={16} className="mt-0.5" />
          {error}
        </div>
      )}

      {cards.length === 0 && !loading && !error && (
        <div className="rounded-md border border-border bg-muted/20 px-4 py-10 text-center text-sm text-muted-foreground">
          <GitBranch size={20} className="mx-auto mb-2 opacity-60" />
          No hay routers configurados. Usa el botón <strong>Nuevo router</strong> para crear uno.
        </div>
      )}

      <div className="grid gap-4">
        {cards.map((card) => {
          const stats = statsForRouter(card.router.profileId)
          return (
            <div key={card.router.profileId} className="rounded-lg border border-border bg-card p-4">
              <div className="mb-3 flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="rounded border border-primary/40 bg-primary/10 px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide text-primary">
                      router
                    </span>
                    <code className="font-mono text-sm font-semibold">{card.router.profileId}</code>
                    {card.router.displayName && card.router.displayName !== card.router.profileId && (
                      <span className="text-sm text-muted-foreground">— {card.router.displayName}</span>
                    )}
                  </div>
                  <div className="mt-1 text-xs text-muted-foreground">
                    base: <code className="font-mono">{card.router.baseModelId}</code>
                  </div>
                  {card.router.description && (
                    <p className="mt-1 text-xs text-muted-foreground">{card.router.description}</p>
                  )}
                </div>
                <div className="flex items-center gap-2">
                  {card.dirty && (
                    <button
                      type="button"
                      onClick={() => void save(card.router.profileId)}
                      disabled={saving === card.router.profileId || card.targets.length === 0}
                      className="inline-flex items-center gap-1 rounded-md bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
                    >
                      {saving === card.router.profileId ? <Loader2 size={12} className="animate-spin" /> : <Save size={12} />}
                      Guardar orden
                    </button>
                  )}
                  <button
                    type="button"
                    onClick={() => setEditingRouter(card.router)}
                    disabled={saving === card.router.profileId}
                    className="rounded p-1.5 text-muted-foreground hover:bg-muted disabled:opacity-30"
                    title="Editar nombre y descripción"
                  >
                    <Pencil size={14} />
                  </button>
                  <button
                    type="button"
                    onClick={() => setCloningRouter(card.router)}
                    disabled={saving === card.router.profileId}
                    className="rounded p-1.5 text-muted-foreground hover:bg-muted disabled:opacity-30"
                    title="Clonar router"
                  >
                    <Copy size={14} />
                  </button>
                  <button
                    type="button"
                    onClick={() => void deleteRouter(card.router.profileId)}
                    disabled={saving === card.router.profileId}
                    className="rounded p-1.5 text-destructive hover:bg-destructive/10 disabled:opacity-30"
                    title="Eliminar router"
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
              </div>

              {card.targets.length === 0 && (
                <div className="mb-2 flex items-start gap-2 rounded-md border border-amber-500/40 bg-amber-500/10 px-2 py-1.5 text-xs text-amber-700 dark:text-amber-400">
                  <AlertCircle size={14} className="mt-0.5" />
                  <span>
                    Router sin destinos. Mientras esté vacío las peticiones a
                    {" "}<code>{card.router.profileId}</code> resuelven directo al
                    base model <code>{card.router.baseModelId}</code>. Añade destinos
                    para que vuelva a actuar como router.
                  </span>
                </div>
              )}

              <ol className="space-y-1">
                {card.targets.map((target, idx) => {
                  const targetProfile = profilesById.get(target.profileId)
                  const share = stats?.targets.find((t) => t.modelId === target.profileId)?.share
                  return (
                    <li
                      key={`${target.profileId}-${idx}`}
                      className="flex items-center gap-2 rounded-md border border-border/60 bg-muted/10 px-2 py-1.5 text-sm"
                    >
                      <span className="w-6 text-center text-xs text-muted-foreground">{idx + 1}</span>
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-2">
                          <code className="truncate font-mono text-sm">{target.profileId}</code>
                          {statusPill(target.status)}
                          {!targetProfile && (
                            <span className="rounded border border-amber-500/40 bg-amber-500/10 px-1.5 py-0.5 text-[10px] text-amber-600 dark:text-amber-400">
                              perfil no encontrado
                            </span>
                          )}
                        </div>
                        {targetProfile && (
                          <div className="mt-0.5 truncate text-[10px] text-muted-foreground">
                            {targetProfile.baseModelId}
                          </div>
                        )}
                        {target.errorMessage && (
                          <div className="mt-0.5 truncate text-[10px] text-destructive" title={target.errorMessage}>
                            {target.errorMessage.slice(0, 90)}
                          </div>
                        )}
                      </div>
                      {typeof share === "number" && (
                        <span className="text-[10px] text-muted-foreground tabular-nums">
                          {Math.round(share * 100)}% 7d
                        </span>
                      )}
                      <div className="flex items-center gap-0.5">
                        <button
                          type="button"
                          disabled={idx === 0 || saving === card.router.profileId}
                          onClick={() => moveTarget(card.router.profileId, idx, -1)}
                          className="rounded p-1 hover:bg-muted disabled:opacity-30"
                          title="Subir"
                        >
                          <ArrowUp size={12} />
                        </button>
                        <button
                          type="button"
                          disabled={idx === card.targets.length - 1 || saving === card.router.profileId}
                          onClick={() => moveTarget(card.router.profileId, idx, 1)}
                          className="rounded p-1 hover:bg-muted disabled:opacity-30"
                          title="Bajar"
                        >
                          <ArrowDown size={12} />
                        </button>
                        <button
                          type="button"
                          disabled={saving === card.router.profileId}
                          onClick={() => removeTarget(card.router.profileId, idx)}
                          className="rounded p-1 text-destructive hover:bg-destructive/10 disabled:opacity-30"
                          title="Quitar del router"
                        >
                          <Trash2 size={12} />
                        </button>
                      </div>
                    </li>
                  )
                })}
              </ol>

              {addingTo === card.router.profileId ? (
                <div className="mt-2 space-y-1">
                  <div className="flex items-start gap-2">
                    <div className="flex-1">
                      <ProfilePicker
                        options={candidatesForAdd.map((p) => ({
                          profileId: p.profileId,
                          baseModelId: p.baseModelId,
                          displayName: p.displayName,
                          category: p.category,
                        }))}
                        value={addTargetValue ? [addTargetValue] : []}
                        onChange={(ids) => setAddTargetValue(ids[0] ?? "")}
                        placeholder="Filtrar perfiles disponibles…"
                        disabled={candidatesForAdd.length === 0 || saving === card.router.profileId}
                      />
                    </div>
                    <button
                      type="button"
                      onClick={() => addTarget(card.router.profileId, addTargetValue)}
                      disabled={!addTargetValue || saving === card.router.profileId}
                      className="inline-flex items-center gap-1 rounded-md bg-primary px-2 py-1.5 text-xs text-primary-foreground disabled:opacity-50"
                    >
                      {saving === card.router.profileId ? (
                        <Loader2 size={12} className="animate-spin" />
                      ) : (
                        <Plus size={12} />
                      )}
                      Añadir
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setAddingTo(null)
                        setAddTargetValue("")
                      }}
                      className="rounded p-1.5 text-muted-foreground hover:bg-muted"
                    >
                      <X size={12} />
                    </button>
                  </div>
                  {candidatesForAdd.length === 0 && (
                    <p className="text-[10px] text-muted-foreground">
                      No hay perfiles enabled no-router disponibles. Un router no puede
                      apuntar a otro router (para evitar loops).
                    </p>
                  )}
                </div>
              ) : (
                <button
                  type="button"
                  onClick={() => {
                    setAddingTo(card.router.profileId)
                    setAddTargetValue("")
                  }}
                  className="mt-2 inline-flex items-center gap-1 rounded-md border border-dashed border-border px-2 py-1 text-xs text-muted-foreground hover:bg-muted"
                >
                  <Plus size={12} />
                  Añadir destino
                </button>
              )}

              {stats && (
                <div className="mt-3 border-t border-border/60 pt-2 text-[10px] text-muted-foreground">
                  Últimos 7d: {stats.total.toLocaleString()} peticiones,{" "}
                  {stats.targets.length} destinos servidos
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}

interface CreateRouterModalProps {
  models: ModelState[]
  profiles: ModelProfile[]
  onClose: () => void
  onCreated: () => Promise<void>
}

function CreateRouterModal({ models, profiles, onClose, onCreated }: CreateRouterModalProps) {
  const sortedModels = useMemo(
    () => [...models].sort((a, b) => a.modelId.localeCompare(b.modelId)),
    [models],
  )
  const [baseModelId, setBaseModelId] = useState(sortedModels[0]?.modelId ?? "")
  const [profileId, setProfileId] = useState("")
  const [displayName, setDisplayName] = useState("")
  const [description, setDescription] = useState("")
  const [category, setCategory] = useState<ProfileCategory>("llm")
  const [parsedTargets, setParsedTargets] = useState<string[]>([])
  const [submitting, setSubmitting] = useState(false)

  const selectableProfiles = useMemo(
    () =>
      profiles
        .filter((p) => p.enabled && p.routingTargets === null)
        .map((p) => ({
          profileId: p.profileId,
          baseModelId: p.baseModelId,
          displayName: p.displayName,
          category: p.category,
        })),
    [profiles],
  )

  const selfReference = parsedTargets.includes(profileId.trim())

  const handleSubmit = async () => {
    const trimmedId = profileId.trim()
    if (!baseModelId) {
      toast.error("Elige un modelo base")
      return
    }
    if (!trimmedId) {
      toast.error("profile_id es obligatorio")
      return
    }
    if (parsedTargets.length === 0) {
      toast.error("Un router necesita al menos un destino")
      return
    }
    if (selfReference) {
      toast.error("Un router no puede referenciarse a sí mismo")
      return
    }
    setSubmitting(true)
    try {
      await api.profiles.create(baseModelId, {
        profileId: trimmedId,
        displayName: displayName.trim() || undefined,
        description: description.trim() || undefined,
        category,
        enabled: true,
        isDefault: false,
        routingTargets: parsedTargets,
      })
      toast.success(`Router ${trimmedId} creado`)
      await onCreated()
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Error creando el router")
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
      onClick={onClose}
    >
      <div
        className="w-full max-w-lg rounded-lg border border-border bg-card p-5 shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-4 flex items-start justify-between gap-3">
          <div>
            <h2 className="text-lg font-semibold">Nuevo router</h2>
            <p className="mt-1 text-xs text-muted-foreground">
              Un router es un perfil con <code>routing_targets</code>: cuando un cliente
              lo invoca, se resuelve al primer destino disponible de la lista.
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded p-1 text-muted-foreground hover:bg-muted"
          >
            <X size={16} />
          </button>
        </div>

        <div className="space-y-3">
          <div>
            <label className="mb-1 block text-xs font-medium">Modelo base</label>
            <select
              value={baseModelId}
              onChange={(e) => setBaseModelId(e.target.value)}
              className="w-full rounded-md border border-input bg-background px-2 py-1.5 text-sm"
            >
              {sortedModels.map((m) => (
                <option key={m.modelId} value={m.modelId}>{m.modelId}</option>
              ))}
            </select>
            <p className="mt-1 text-[10px] text-muted-foreground">
              Fallback último del resolver. Si ninguno de los destinos está disponible,
              se cae al perfil default de este modelo.
            </p>
          </div>

          <div>
            <label className="mb-1 block text-xs font-medium">profile_id</label>
            <input
              type="text"
              value={profileId}
              onChange={(e) => setProfileId(e.target.value)}
              placeholder="chat-router"
              className="w-full rounded-md border border-input bg-background px-2 py-1.5 font-mono text-sm"
            />
          </div>

          <div>
            <label className="mb-1 block text-xs font-medium">Display name (opcional)</label>
            <input
              type="text"
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
              placeholder="Chat Router"
              className="w-full rounded-md border border-input bg-background px-2 py-1.5 text-sm"
            />
          </div>

          <div>
            <label className="mb-1 block text-xs font-medium">Categoría</label>
            <select
              value={category}
              onChange={(e) => setCategory(e.target.value as ProfileCategory)}
              className="w-full rounded-md border border-input bg-background px-2 py-1.5 text-sm"
            >
              {ROUTER_CATEGORIES.map((c) => (
                <option key={c.value} value={c.value}>{c.label}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="mb-1 block text-xs font-medium">
              Destinos (en orden de prioridad)
            </label>
            <ProfilePicker
              multi
              options={selectableProfiles}
              value={parsedTargets}
              onChange={setParsedTargets}
              placeholder="Filtrar perfiles disponibles…"
              excludeIds={profileId.trim() ? [profileId.trim()] : []}
            />
            {parsedTargets.length > 0 && (
              <p className="mt-1 text-[10px] text-muted-foreground">
                {parsedTargets.length} destino{parsedTargets.length === 1 ? "" : "s"} —
                el resolver los recorrerá en este orden.
              </p>
            )}
          </div>

          <div>
            <label className="mb-1 block text-xs font-medium">Descripción (opcional)</label>
            <input
              type="text"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              className="w-full rounded-md border border-input bg-background px-2 py-1.5 text-sm"
            />
          </div>
        </div>

        <div className="mt-5 flex items-center justify-end gap-2">
          <button
            type="button"
            onClick={onClose}
            className="rounded-md border border-border px-3 py-1.5 text-sm hover:bg-muted"
          >
            Cancelar
          </button>
          <button
            type="button"
            disabled={submitting}
            onClick={() => void handleSubmit()}
            className="inline-flex items-center gap-1 rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
          >
            {submitting ? <Loader2 size={14} className="animate-spin" /> : <Plus size={14} />}
            Crear router
          </button>
        </div>
      </div>
    </div>
  )
}

interface EditRouterModalProps {
  router: ModelProfile
  existingIds: string[]
  onClose: () => void
  onSaved: () => Promise<void>
}

function EditRouterModal({ router: r, existingIds, onClose, onSaved }: EditRouterModalProps) {
  const [profileId, setProfileId] = useState(r.profileId)
  const [displayName, setDisplayName] = useState(r.displayName ?? "")
  const [description, setDescription] = useState(r.description ?? "")
  const [submitting, setSubmitting] = useState(false)

  const idChanged = profileId.trim() !== r.profileId
  const duplicate =
    idChanged && existingIds.includes(profileId.trim()) && profileId.trim() !== r.profileId

  const handleSubmit = async () => {
    const newId = profileId.trim()
    if (!newId) {
      toast.error("profile_id es obligatorio")
      return
    }
    if (duplicate) {
      toast.error("Ya existe un perfil con ese profile_id")
      return
    }
    setSubmitting(true)
    try {
      if (idChanged) {
        await api.profiles.rename(r.profileId, newId)
      }
      const patchChanged =
        displayName.trim() !== (r.displayName ?? "") ||
        description.trim() !== (r.description ?? "")
      if (patchChanged) {
        await api.profiles.update(idChanged ? newId : r.profileId, {
          displayName: displayName.trim(),
          description: description.trim(),
        })
      }
      toast.success(idChanged ? `Router renombrado a ${newId}` : "Router actualizado")
      await onSaved()
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Error guardando")
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
      onClick={onClose}
    >
      <div
        className="w-full max-w-lg rounded-lg border border-border bg-card p-5 shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-4 flex items-start justify-between gap-3">
          <h2 className="text-lg font-semibold">Editar router</h2>
          <button type="button" onClick={onClose} className="rounded p-1 text-muted-foreground hover:bg-muted">
            <X size={16} />
          </button>
        </div>

        <div className="space-y-3">
          <div>
            <label className="mb-1 block text-xs font-medium">profile_id</label>
            <input
              type="text"
              value={profileId}
              onChange={(e) => setProfileId(e.target.value)}
              className="w-full rounded-md border border-input bg-background px-2 py-1.5 font-mono text-sm"
            />
            {idChanged && (
              <p className="mt-1 text-[10px] text-amber-500">
                Renombrar actualiza en cascada las referencias en routing_targets de otros routers.
                Los clientes que invoquen el id antiguo recibirán 404 — avisa antes de aplicar.
              </p>
            )}
            {duplicate && (
              <p className="mt-1 text-[10px] text-destructive">Ya existe un perfil con ese profile_id.</p>
            )}
          </div>

          <div>
            <label className="mb-1 block text-xs font-medium">Display name</label>
            <input
              type="text"
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
              className="w-full rounded-md border border-input bg-background px-2 py-1.5 text-sm"
            />
          </div>

          <div>
            <label className="mb-1 block text-xs font-medium">Descripción</label>
            <textarea
              rows={3}
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              className="w-full rounded-md border border-input bg-background px-2 py-1.5 text-sm"
            />
          </div>
        </div>

        <div className="mt-5 flex items-center justify-end gap-2">
          <button
            type="button"
            onClick={onClose}
            className="rounded-md border border-border px-3 py-1.5 text-sm hover:bg-muted"
          >
            Cancelar
          </button>
          <button
            type="button"
            disabled={submitting || duplicate}
            onClick={() => void handleSubmit()}
            className="inline-flex items-center gap-1 rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
          >
            {submitting ? <Loader2 size={14} className="animate-spin" /> : <Save size={14} />}
            Guardar cambios
          </button>
        </div>
      </div>
    </div>
  )
}

interface CloneRouterModalProps {
  source: ModelProfile
  existingIds: string[]
  onClose: () => void
  onCloned: () => Promise<void>
}

function CloneRouterModal({ source, existingIds, onClose, onCloned }: CloneRouterModalProps) {
  const [profileId, setProfileId] = useState(`${source.profileId}-copy`)
  const [displayName, setDisplayName] = useState(source.displayName ?? "")
  const [submitting, setSubmitting] = useState(false)
  const duplicate = existingIds.includes(profileId.trim())

  const handleSubmit = async () => {
    const newId = profileId.trim()
    if (!newId) {
      toast.error("profile_id es obligatorio")
      return
    }
    if (duplicate) {
      toast.error("Ya existe un perfil con ese profile_id")
      return
    }
    setSubmitting(true)
    try {
      await api.profiles.clone(source.profileId, newId, displayName.trim() || undefined)
      toast.success(`Router clonado como ${newId}`)
      await onCloned()
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Error clonando")
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
      onClick={onClose}
    >
      <div
        className="w-full max-w-lg rounded-lg border border-border bg-card p-5 shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-4 flex items-start justify-between gap-3">
          <div>
            <h2 className="text-lg font-semibold">Clonar router</h2>
            <p className="mt-1 text-xs text-muted-foreground">
              Crea un nuevo router copiando <code>{source.profileId}</code>. Los destinos
              y configuración se replican; <code>is_default</code> se desactiva en la copia.
            </p>
          </div>
          <button type="button" onClick={onClose} className="rounded p-1 text-muted-foreground hover:bg-muted">
            <X size={16} />
          </button>
        </div>

        <div className="space-y-3">
          <div>
            <label className="mb-1 block text-xs font-medium">Nuevo profile_id</label>
            <input
              type="text"
              value={profileId}
              onChange={(e) => setProfileId(e.target.value)}
              className="w-full rounded-md border border-input bg-background px-2 py-1.5 font-mono text-sm"
            />
            {duplicate && (
              <p className="mt-1 text-[10px] text-destructive">Ya existe un perfil con ese profile_id.</p>
            )}
          </div>

          <div>
            <label className="mb-1 block text-xs font-medium">Display name (opcional)</label>
            <input
              type="text"
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
              className="w-full rounded-md border border-input bg-background px-2 py-1.5 text-sm"
            />
          </div>
        </div>

        <div className="mt-5 flex items-center justify-end gap-2">
          <button
            type="button"
            onClick={onClose}
            className="rounded-md border border-border px-3 py-1.5 text-sm hover:bg-muted"
          >
            Cancelar
          </button>
          <button
            type="button"
            disabled={submitting || duplicate}
            onClick={() => void handleSubmit()}
            className="inline-flex items-center gap-1 rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
          >
            {submitting ? <Loader2 size={14} className="animate-spin" /> : <Copy size={14} />}
            Clonar
          </button>
        </div>
      </div>
    </div>
  )
}

interface ProfileOption {
  profileId: string
  baseModelId: string
  displayName?: string | null
  category: string
}

interface ProfilePickerProps {
  options: ProfileOption[]
  value: string[]
  onChange: (next: string[]) => void
  multi?: boolean
  placeholder?: string
  disabled?: boolean
  excludeIds?: string[]
}

/** Combobox con filtro por texto. Si ``multi`` muestra chips para los
 * elementos ya elegidos y un input que añade al array; en modo single
 * se comporta como un ``<select>`` con búsqueda. Filtra sobre profile_id,
 * base_model_id y display_name a la vez. */
function ProfilePicker({
  options,
  value,
  onChange,
  multi = false,
  placeholder,
  disabled,
  excludeIds = [],
}: ProfilePickerProps) {
  const [query, setQuery] = useState("")
  const [open, setOpen] = useState(false)
  const [activeIndex, setActiveIndex] = useState(0)
  const wrapRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    if (!open) return
    const onDocClick = (e: MouseEvent) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    document.addEventListener("mousedown", onDocClick)
    return () => document.removeEventListener("mousedown", onDocClick)
  }, [open])

  const excluded = useMemo(
    () => new Set([...excludeIds, ...(multi ? value : [])]),
    [excludeIds, value, multi],
  )

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    return options
      .filter((o) => !excluded.has(o.profileId))
      .filter((o) => {
        if (!q) return true
        return (
          o.profileId.toLowerCase().includes(q) ||
          o.baseModelId.toLowerCase().includes(q) ||
          (o.displayName ?? "").toLowerCase().includes(q)
        )
      })
      .sort((a, b) => a.profileId.localeCompare(b.profileId))
  }, [options, excluded, query])

  useEffect(() => {
    if (activeIndex >= filtered.length) setActiveIndex(0)
  }, [filtered.length, activeIndex])

  const selectOption = (opt: ProfileOption) => {
    if (multi) {
      onChange([...value, opt.profileId])
      setQuery("")
    } else {
      onChange([opt.profileId])
      setQuery(opt.profileId)
      setOpen(false)
    }
  }

  const removeChip = (id: string) => {
    onChange(value.filter((v) => v !== id))
  }

  const onKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (!open) {
      if (["ArrowDown", "ArrowUp", "Enter"].includes(e.key)) setOpen(true)
    }
    if (e.key === "ArrowDown") {
      e.preventDefault()
      setActiveIndex((i) => Math.min(i + 1, filtered.length - 1))
    } else if (e.key === "ArrowUp") {
      e.preventDefault()
      setActiveIndex((i) => Math.max(i - 1, 0))
    } else if (e.key === "Enter") {
      e.preventDefault()
      const opt = filtered[activeIndex]
      if (opt) selectOption(opt)
    } else if (e.key === "Escape") {
      setOpen(false)
    } else if (e.key === "Backspace" && multi && !query && value.length > 0) {
      // Backspace sobre input vacío: quita el último chip (patrón habitual
      // de tag-inputs; evita que el usuario tenga que ir al botón ×).
      removeChip(value[value.length - 1])
    }
  }

  return (
    <div ref={wrapRef} className="relative">
      {multi && value.length > 0 && (
        <div className="mb-1 flex flex-wrap gap-1">
          {value.map((id) => (
            <span
              key={id}
              className="inline-flex items-center gap-1 rounded-md border border-primary/40 bg-primary/10 px-1.5 py-0.5 font-mono text-xs text-primary"
            >
              {id}
              <button
                type="button"
                disabled={disabled}
                onClick={() => removeChip(id)}
                className="rounded hover:bg-primary/20 disabled:opacity-50"
                aria-label={`Quitar ${id}`}
              >
                <X size={10} />
              </button>
            </span>
          ))}
        </div>
      )}
      <div className="relative">
        <Search
          size={12}
          className="pointer-events-none absolute left-2 top-1/2 -translate-y-1/2 text-muted-foreground"
        />
        <input
          type="text"
          disabled={disabled}
          value={query}
          onChange={(e) => {
            setQuery(e.target.value)
            setOpen(true)
          }}
          onFocus={() => setOpen(true)}
          onKeyDown={onKeyDown}
          placeholder={placeholder ?? "Filtrar…"}
          className="w-full rounded-md border border-input bg-background pl-7 pr-2 py-1.5 text-sm disabled:opacity-50"
        />
      </div>
      {open && !disabled && (
        <ul className="absolute left-0 right-0 top-full z-20 mt-1 max-h-60 overflow-y-auto rounded-md border border-border bg-popover shadow-lg">
          {filtered.length === 0 ? (
            <li className="px-2 py-1.5 text-xs text-muted-foreground">Sin coincidencias.</li>
          ) : (
            filtered.map((opt, idx) => (
              <li
                key={opt.profileId}
                onMouseDown={(e) => {
                  e.preventDefault()
                  selectOption(opt)
                }}
                onMouseEnter={() => setActiveIndex(idx)}
                className={`cursor-pointer px-2 py-1.5 text-xs ${
                  idx === activeIndex ? "bg-primary/10" : "hover:bg-muted"
                }`}
              >
                <div className="flex items-baseline justify-between gap-2">
                  <code className="truncate font-mono">{opt.profileId}</code>
                  <span className="shrink-0 rounded border border-border/60 bg-muted/40 px-1 text-[10px] text-muted-foreground">
                    {opt.category}
                  </span>
                </div>
                <div className="truncate text-[10px] text-muted-foreground">
                  {opt.baseModelId}
                  {opt.displayName && opt.displayName !== opt.profileId && ` — ${opt.displayName}`}
                </div>
              </li>
            ))
          )}
        </ul>
      )}
    </div>
  )
}
