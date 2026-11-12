import { useCallback, useEffect, useState } from "react"
import { toast } from "sonner"
import { api } from "@/api/client"
import { cn } from "@/lib/cn"
import type { Group, ModelProfile } from "@/types"
import { Callout, Chip, Panel } from "../ui"
import type { SectionProps } from "./types"

/**
 * Qué grupos de usuarios pueden usar este despliegue. Los grupos conceden
 * nombres de la API (ids de perfil), no despliegues: por eso la tabla cruza
 * cada grupo con cada perfil. Solo admins; los cambios se guardan al momento.
 */
export function AccessSection(props: SectionProps) {
  const { editor, isAdmin, readOnly } = props
  const modelId = editor.model?.model_id ?? null
  const [groups, setGroups] = useState<Group[]>([])
  const [profiles, setProfiles] = useState<ModelProfile[]>([])
  const [membership, setMembership] = useState<Record<string, Set<string>>>({})
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState<string | null>(null)

  const load = useCallback(async () => {
    if (!modelId || !isAdmin) return
    try {
      const [gs, ps] = await Promise.all([api.groups.list(), api.profiles.listByModel(modelId)])
      const lists = await Promise.all(gs.map((g) => api.groups.listModels(g.id).catch(() => [] as string[])))
      setGroups(gs)
      setProfiles(ps)
      setMembership(Object.fromEntries(gs.map((g, i) => [g.id, new Set(lists[i])])))
    } catch (err) {
      toast.error(err instanceof Error ? err.message : String(err))
    } finally {
      setLoading(false)
    }
  }, [modelId, isAdmin])

  useEffect(() => {
    void load()
  }, [load])

  if (!isAdmin) {
    return <Callout tone="info">Solo los administradores pueden ver y cambiar qué grupos tienen acceso a este modelo.</Callout>
  }

  const toggle = async (group: Group, profileId: string) => {
    const has = membership[group.id]?.has(profileId)
    const cell = `${group.id}:${profileId}`
    setBusy(cell)
    try {
      if (has) await api.groups.removeModel(group.id, profileId)
      else await api.groups.addModel(group.id, profileId)
      setMembership((m) => {
        const next = new Set(m[group.id] ?? [])
        if (has) next.delete(profileId)
        else next.add(profileId)
        return { ...m, [group.id]: next }
      })
    } catch (err) {
      toast.error(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(null)
    }
  }

  return (
    <div className="space-y-4" data-testid="access-section">
      <Callout tone="info">
        Los grupos conceden nombres de la API (los ids de perfil). Un perfil sin grupo solo lo ven los grupos que
        dan acceso a todo. Los cambios se guardan al momento.
      </Callout>
      <Panel title="Grupos con acceso">
        {loading ? (
          <p className="text-xs text-muted-foreground">Cargando…</p>
        ) : groups.length === 0 ? (
          <p className="text-xs text-muted-foreground">No hay grupos.</p>
        ) : profiles.length === 0 ? (
          <p className="text-xs text-muted-foreground">Este despliegue no tiene perfiles: no hay nombre que conceder.</p>
        ) : (
          <div className="overflow-auto">
            <table className="min-w-full text-xs">
              <thead>
                <tr className="text-left text-muted-foreground">
                  <th className="py-1.5 pr-4 font-medium">Grupo</th>
                  {profiles.map((p) => (
                    <th key={p.profileId} className="px-2 py-1.5 text-center font-mono font-medium">
                      {p.profileId}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-border/50">
                {groups.map((g) => (
                  <tr key={g.id}>
                    <td className="py-1.5 pr-4">
                      <div className="flex items-center gap-1.5">
                        <span className="font-medium text-foreground">{g.name}</span>
                        {g.isDefault && <Chip>por defecto</Chip>}
                        {g.grantsAllModels && <Chip tone="info">todos los modelos</Chip>}
                      </div>
                      {g.description && <div className="text-[11px] text-muted-foreground">{g.description}</div>}
                    </td>
                    {profiles.map((p) => {
                      const has = g.grantsAllModels || membership[g.id]?.has(p.profileId)
                      const cell = `${g.id}:${p.profileId}`
                      return (
                        <td key={p.profileId} className="px-2 py-1.5 text-center">
                          <input
                            type="checkbox"
                            aria-label={`${g.name} · ${p.profileId}`}
                            checked={Boolean(has)}
                            disabled={readOnly || g.grantsAllModels || busy === cell}
                            onChange={() => void toggle(g, p.profileId)}
                            className={cn("h-4 w-4", g.grantsAllModels && "opacity-50")}
                            title={g.grantsAllModels ? "Este grupo ya da acceso a todos los modelos" : undefined}
                          />
                        </td>
                      )
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>
    </div>
  )
}
