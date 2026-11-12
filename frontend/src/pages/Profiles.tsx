import { useEffect, useMemo, useState } from "react"
import { Gauge, Layers3, Pencil, Plus, RefreshCw, Search } from "lucide-react"
import { useNavigate } from "react-router-dom"
import { toast } from "sonner"
import { api } from "@/api/client"
import type { ModelProfile, ModelState, ProfileCategory } from "@/types"

const CATEGORY_LABELS: Record<ProfileCategory, string> = {
  llm: "LLM",
  embedding: "Embeddings",
  rerank: "Reranking",
  tts: "TTS",
  stt: "STT",
  image: "Imagen",
  music: "Música",
  video_upscaling: "Escalado de vídeo",
}

export function Profiles() {
  const [profiles, setProfiles] = useState<ModelProfile[]>([])
  const [models, setModels] = useState<ModelState[]>([])
  const [query, setQuery] = useState("")
  const [category, setCategory] = useState<ProfileCategory | "all">("all")
  const [loading, setLoading] = useState(true)
  const [createModelId, setCreateModelId] = useState("")
  const navigate = useNavigate()
  // Perfiles y presets se editan en el editor del modelo base; esta página es
  // el índice para buscarlos entre todos los modelos.
  const editorUrl = (baseModelId: string, section: "profiles" | "presets", extra: Record<string, string> = {}) =>
    `/models/edit/${encodeURIComponent(baseModelId)}?${new URLSearchParams({ s: section, ...extra }).toString()}`

  const refresh = async () => {
    try {
      const [allProfiles, allModels] = await Promise.all([
        api.profiles.listAll(),
        api.models.list(),
      ])
      setProfiles(allProfiles.filter((profile) => profile.profileKind === "execution"))
      setModels(allModels)
      setCreateModelId((current) => current || allModels[0]?.modelId || "")
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "No se pudieron cargar los perfiles")
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void refresh()
  }, [])

  const modelsById = useMemo(
    () => new Map(models.map((model) => [model.modelId, model])),
    [models],
  )

  const visible = useMemo(() => {
    const needle = query.trim().toLowerCase()
    return profiles
      .filter((profile) => category === "all" || profile.category === category)
      .filter((profile) => {
        if (!needle) return true
        return [profile.profileId, profile.displayName, profile.baseModelId]
          .filter(Boolean)
          .some((value) => String(value).toLowerCase().includes(needle))
      })
      .sort((left, right) => left.profileId.localeCompare(right.profileId))
  }, [category, profiles, query])

  const openCreate = () => {
    const model = modelsById.get(createModelId)
    if (!model) {
      toast.error("No hay modelos físicos disponibles")
      return
    }
    navigate(editorUrl(model.modelId, "profiles", { new: "1" }))
  }

  const openEdit = (profile: ModelProfile, section: "profiles" | "presets" = "profiles") => {
    if (!profile.baseModelId) {
      toast.error("El perfil no tiene modelo base")
      return
    }
    navigate(editorUrl(profile.baseModelId, section, { p: profile.profileId }))
  }

  const toggleEnabled = async (profile: ModelProfile) => {
    try {
      await api.profiles.update(profile.profileId, { enabled: !profile.enabled })
      await refresh()
      toast.success(profile.enabled ? "Perfil desactivado" : "Perfil activado")
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "No se pudo actualizar el perfil")
    }
  }

  return (
    <div className="space-y-4 p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold">Perfiles de ejecución</h1>
          <p className="text-sm text-muted-foreground">
            Un identificador público estable por modelo y uso. El contexto, paralelismo y política de carga se gestionan como presets internos.
          </p>
        </div>
        <div className="flex gap-2">
          <select value={createModelId} onChange={(event) => setCreateModelId(event.target.value)} className="max-w-72 rounded-md border border-border bg-background px-2 py-1.5 text-xs" aria-label="Modelo base del nuevo perfil">
            {models.map((model) => <option key={model.modelId} value={model.modelId}>{model.displayName || model.modelId}</option>)}
          </select>
          <button type="button" onClick={openCreate} disabled={!createModelId} className="inline-flex items-center gap-1 rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground disabled:opacity-50">
            <Plus size={14} /> Nuevo perfil
          </button>
          <button type="button" onClick={() => void refresh()} disabled={loading} className="inline-flex items-center gap-1 rounded-md border border-border px-3 py-1.5 text-sm hover:bg-muted disabled:opacity-50">
            <RefreshCw size={14} className={loading ? "animate-spin" : ""} /> Actualizar
          </button>
        </div>
      </div>

      <div className="flex flex-wrap gap-2 rounded-lg border border-border bg-card p-3">
        <label className="relative min-w-64 flex-1">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
          <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Buscar perfil o modelo base…" className="w-full rounded-md border border-border bg-background py-2 pl-9 pr-3 text-sm" />
        </label>
        <select value={category} onChange={(event) => setCategory(event.target.value as ProfileCategory | "all")} className="rounded-md border border-border bg-background px-3 py-2 text-sm">
          <option value="all">Todas las categorías</option>
          {Object.entries(CATEGORY_LABELS).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
        </select>
      </div>

      {!loading && visible.length === 0 && (
        <div className="rounded-lg border border-dashed border-border py-12 text-center text-sm text-muted-foreground">
          <Layers3 className="mx-auto mb-2 opacity-60" size={22} /> No hay perfiles que coincidan.
        </div>
      )}

      <div className="space-y-3">
        {visible.map((profile) => {
          const defaultPreset = profile.runtimePresets.find((preset) => preset.isDefault)
          return (
            <section key={profile.profileId} className={`rounded-lg border bg-card ${profile.enabled ? "border-border" : "border-border/50 opacity-70"}`}>
              <div className="flex flex-wrap items-center justify-between gap-3 p-4">
                <button type="button" onClick={() => openEdit(profile)} className="flex min-w-0 flex-1 items-center gap-3 text-left">
                  <div className="min-w-0">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="font-medium">{profile.displayName || profile.profileId}</span>
                      <code className="rounded bg-muted px-1.5 py-0.5 text-[11px]">{profile.profileId}</code>
                      <span className="rounded border border-border px-1.5 py-0.5 text-[10px]">{CATEGORY_LABELS[profile.category]}</span>
                      {profile.isDefault && <span className="rounded border border-blue-500/40 bg-blue-500/10 px-1.5 py-0.5 text-[10px] text-blue-300">Perfil por defecto</span>}
                    </div>
                    <div className="mt-1 truncate text-xs text-muted-foreground">
                      {profile.baseModelId} · {profile.runtimePresets.length} presets
                      {defaultPreset ? ` · default ${defaultPreset.label}` : ""}
                      {profile.aliases.length ? ` · ${profile.aliases.length} aliases` : ""}
                    </div>
                  </div>
                </button>
                <div className="flex items-center gap-2">
                  <button type="button" onClick={() => void toggleEnabled(profile)} className="rounded border border-border px-2.5 py-1.5 text-xs hover:bg-muted">{profile.enabled ? "Desactivar" : "Activar"}</button>
                  {profile.category === "llm" && (
                    <button type="button" onClick={() => openEdit(profile, "presets")} className="inline-flex items-center gap-1 rounded border border-border px-2.5 py-1.5 text-xs hover:bg-muted"><Gauge size={13} /> Presets</button>
                  )}
                  <button type="button" onClick={() => openEdit(profile)} className="inline-flex items-center gap-1 rounded border border-border px-2.5 py-1.5 text-xs hover:bg-muted"><Pencil size={13} /> Editar</button>
                </div>
              </div>
              {profile.aliases.length > 0 && (
                <div className="border-t border-border px-4 py-2 text-[11px] text-muted-foreground">
                  Alias: {profile.aliases.map((alias) => `${alias.aliasId}${alias.presetKey ? ` → ${alias.presetKey}` : ""}`).join(" · ")}
                </div>
              )}
            </section>
          )
        })}
      </div>

    </div>
  )
}
