import { useEffect, useMemo, useState } from "react"
import { AlertCircle, ArrowDown, ArrowUp, GitBranch, Loader2, Plus, RefreshCw, Save, Trash2, X } from "lucide-react"
import { toast } from "sonner"
import { api, type RoutingDecisionsResponse } from "@/api/client"
import type { ModelProfile, ModelState } from "@/types"

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

      const routerProfiles = profiles.filter((p) => Array.isArray(p.routingTargets) && p.routingTargets.length > 0)
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
      setCards(built)

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
      .filter((p) => !p.routingTargets || p.routingTargets.length === 0)
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

  const removeTarget = (routerId: string, index: number) => {
    setCards((prev) =>
      prev.map((c) => {
        if (c.router.profileId !== routerId) return c
        return { ...c, targets: c.targets.filter((_, i) => i !== index), dirty: true }
      }),
    )
  }

  const addTarget = (routerId: string, targetId: string) => {
    if (!targetId) return
    setCards((prev) =>
      prev.map((c) => {
        if (c.router.profileId !== routerId) return c
        const target = profilesById.get(targetId)
        if (!target) return c
        const modelState = statesByModelId.get(target.baseModelId)
        return {
          ...c,
          targets: [
            ...c.targets,
            {
              profileId: targetId,
              status: coerceStatus(modelState?.status),
              errorMessage: modelState?.errorMessage ?? null,
            },
          ],
          dirty: true,
        }
      }),
    )
    setAddingTo(null)
    setAddTargetValue("")
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

      {error && (
        <div className="flex items-start gap-2 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
          <AlertCircle size={16} className="mt-0.5" />
          {error}
        </div>
      )}

      {cards.length === 0 && !loading && !error && (
        <div className="rounded-md border border-border bg-muted/20 px-4 py-10 text-center text-sm text-muted-foreground">
          <GitBranch size={20} className="mx-auto mb-2 opacity-60" />
          No hay routers configurados. Cualquier perfil se convierte en router
          al añadirle <code>routing_targets</code> desde la página Models.
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
                  </div>
                  <div className="mt-1 text-xs text-muted-foreground">
                    base: <code className="font-mono">{card.router.baseModelId}</code>
                  </div>
                  {card.router.description && (
                    <p className="mt-1 text-xs text-muted-foreground">{card.router.description}</p>
                  )}
                </div>
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
              </div>

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
                <div className="mt-2 flex items-center gap-2">
                  <select
                    value={addTargetValue}
                    onChange={(e) => setAddTargetValue(e.target.value)}
                    className="flex-1 rounded-md border border-border bg-background px-2 py-1 text-xs"
                  >
                    <option value="">Seleccionar perfil…</option>
                    {candidatesForAdd.map((p) => (
                      <option key={p.profileId} value={p.profileId}>
                        {p.profileId} — {p.baseModelId}
                      </option>
                    ))}
                  </select>
                  <button
                    type="button"
                    onClick={() => addTarget(card.router.profileId, addTargetValue)}
                    disabled={!addTargetValue}
                    className="rounded-md bg-primary px-2 py-1 text-xs text-primary-foreground disabled:opacity-50"
                  >
                    Añadir
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setAddingTo(null)
                      setAddTargetValue("")
                    }}
                    className="rounded p-1 text-muted-foreground hover:bg-muted"
                  >
                    <X size={12} />
                  </button>
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
