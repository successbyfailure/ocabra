import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { Link, useSearchParams } from "react-router-dom"
import { Copy, Pause, Play, Plus, Star, Trash2, Upload } from "lucide-react"
import { toast } from "sonner"
import { api } from "@/api/client"
import { ConfirmDialog } from "@/components/common/ConfirmDialog"
import { cn } from "@/lib/cn"
import type { ModelProfile, ProfileCategory } from "@/types"
import { JsonInput } from "../SchemaField"
import type { Json } from "../types"
import { Callout, Chip, Hint, Panel, Segmented, Toggle, inputCls } from "../ui"
import { DefaultsForm } from "./profiles/DefaultsForms"
import {
  ASSET_ACCEPT,
  CATEGORIES,
  CATEGORY_TONE,
  type Defaults,
  fmt,
  inferCategory,
  isAudioFile,
  isDedicatedWorker,
  isValidSlug,
  overridesDiff,
} from "./profiles/logic"
import type { SectionProps } from "./types"

function errMsg(err: unknown): string {
  return err instanceof Error ? err.message : String(err)
}

function copy(text: string) {
  void navigator.clipboard?.writeText(text).then(
    () => toast.success(`Copiado: ${text}`),
    () => toast.error("No se pudo copiar"),
  )
}

/**
 * Perfiles del despliegue: el nombre con el que lo piden los clientes, sus
 * valores por defecto, sus sobrescrituras de carga y sus ficheros. Cada
 * perfil es una entidad aparte: sus cambios se guardan al momento, no con el
 * botón Guardar de la página.
 */
export function ProfilesSection(props: SectionProps) {
  const { editor, readOnly, refreshLogical } = props
  const modelId = editor.model?.model_id ?? null
  const [profiles, setProfiles] = useState<ModelProfile[]>([])
  const [routers, setRouters] = useState<ModelProfile[]>([])
  const [loading, setLoading] = useState(true)
  // ?p=<perfil> abre ese perfil; ?new=1 abre el formulario de perfil nuevo
  // (enlaces desde /profiles y desde las tarjetas de Modelos).
  const [search] = useSearchParams()
  const [selected, setSelected] = useState<string | null>(() => search.get("p"))
  const [creating, setCreating] = useState(() => search.get("new") === "1")

  const load = useCallback(async () => {
    if (!modelId) return
    try {
      const [mine, all] = await Promise.all([api.profiles.listByModel(modelId), api.profiles.listAll().catch(() => [])])
      setProfiles(mine)
      const ids = new Set(mine.map((p) => p.profileId))
      setRouters(all.filter((p) => (p.routingTargets ?? []).some((t) => ids.has(t))))
    } catch (err) {
      toast.error(`No se pudieron cargar los perfiles: ${errMsg(err)}`)
    } finally {
      setLoading(false)
    }
  }, [modelId])

  useEffect(() => {
    void load()
  }, [load])

  const afterWrite = useCallback(async () => {
    await load()
    await refreshLogical().catch(() => undefined)
  }, [load, refreshLogical])

  const current = profiles.find((p) => p.profileId === selected) ?? null

  const toggleEnabled = async (p: ModelProfile) => {
    try {
      await api.profiles.update(p.profileId, { enabled: !p.enabled })
      await afterWrite()
    } catch (err) {
      toast.error(errMsg(err))
    }
  }

  return (
    <div className="space-y-4" data-testid="profiles-section">
      <Callout tone="info">
        Un perfil es el nombre que los clientes ponen en <code>model</code>. Los cambios de esta sección se
        guardan al momento, sin pasar por «Guardar».
      </Callout>

      <Panel
        title="Perfiles de este despliegue"
        subtitle="El perfil por defecto (★) es el que recibe el nombre público del modelo."
        actions={
          !readOnly && (
            <button
              type="button"
              onClick={() => {
                setCreating(true)
                setSelected(null)
              }}
              className="inline-flex items-center gap-1 rounded-md border border-border px-2.5 py-1 text-xs hover:bg-muted"
            >
              <Plus size={13} /> Nuevo perfil
            </button>
          )
        }
      >
        {loading ? (
          <p className="text-xs text-muted-foreground">Cargando…</p>
        ) : profiles.length === 0 ? (
          <p className="text-xs text-muted-foreground">Este despliegue no tiene perfiles.</p>
        ) : (
          <div className="grid gap-2 md:grid-cols-2">
            {profiles.map((p) => (
              <ProfileCard
                key={p.profileId}
                profile={p}
                active={p.profileId === selected}
                readOnly={readOnly}
                onSelect={() => {
                  setSelected(p.profileId === selected ? null : p.profileId)
                  setCreating(false)
                }}
                onToggle={() => void toggleEnabled(p)}
              />
            ))}
          </div>
        )}
      </Panel>

      {creating && modelId && (
        <CreateProfile
          modelId={modelId}
          defaultCategory={inferCategory(editor.model?.capabilities)}
          existing={profiles.map((p) => p.profileId)}
          onCancel={() => setCreating(false)}
          onCreated={async (id) => {
            setCreating(false)
            await afterWrite()
            setSelected(id)
          }}
        />
      )}

      {current && (
        <ProfileEditor
          key={`${current.profileId}:${current.updatedAt ?? ""}`}
          profile={current}
          baseConfig={(editor.base?.extraConfig ?? {}) as Record<string, unknown>}
          readOnly={readOnly}
          routers={routers.filter((r) => (r.routingTargets ?? []).includes(current.profileId))}
          onChanged={afterWrite}
          onRenamed={async (id) => {
            await afterWrite()
            setSelected(id)
          }}
          onDeleted={async () => {
            setSelected(null)
            await afterWrite()
          }}
        />
      )}

      <Panel
        title="Routers que apuntan aquí"
        subtitle="Un router prueba sus destinos en orden (primero el que ya esté cargado). Con ‹ › cambias la posición de este modelo; sus valores por defecto se aplican entre los del perfil elegido y los del cliente."
        actions={
          <Link to="/routers" className="text-xs text-sky-300 hover:underline">
            Gestionar routers
          </Link>
        }
      >
        {routers.length === 0 ? (
          <p className="text-xs text-muted-foreground">Ningún router usa los perfiles de este despliegue.</p>
        ) : (
          <ul className="space-y-1 text-xs" data-testid="routers-list">
            {routers.map((r) => (
              <li key={r.profileId} className="flex flex-wrap items-center gap-2">
                <span className="font-mono text-foreground">{r.profileId}</span>
                <span className="text-muted-foreground">→</span>
                {(r.routingTargets ?? []).map((t, i, all) => {
                  const ours = profiles.some((p) => p.profileId === t)
                  const move = async (dir: -1 | 1) => {
                    const next = [...all]
                    const j = i + dir
                    if (j < 0 || j >= next.length) return
                    ;[next[i], next[j]] = [next[j], next[i]]
                    try {
                      await api.profiles.update(r.profileId, { routingTargets: next })
                      toast.success(`${t} pasa a la posición ${j + 1} en ${r.profileId}`)
                      await afterWrite()
                    } catch (err) {
                      toast.error(errMsg(err))
                    }
                  }
                  return (
                    <span
                      key={t}
                      className={cn(
                        "inline-flex items-center gap-0.5 rounded border px-1.5 py-px font-mono",
                        ours ? "border-sky-500/40 text-sky-200" : "border-border text-muted-foreground",
                      )}
                      title={`Posición ${i + 1}`}
                    >
                      {ours && !readOnly && i > 0 && (
                        <button type="button" aria-label={`Subir ${t} en ${r.profileId}`} className="hover:text-white" onClick={() => void move(-1)}>
                          ‹
                        </button>
                      )}
                      {t}
                      {ours && !readOnly && i < all.length - 1 && (
                        <button type="button" aria-label={`Bajar ${t} en ${r.profileId}`} className="hover:text-white" onClick={() => void move(1)}>
                          ›
                        </button>
                      )}
                    </span>
                  )
                })}
              </li>
            ))}
          </ul>
        )}
      </Panel>
    </div>
  )
}

function ProfileCard({
  profile,
  active,
  readOnly,
  onSelect,
  onToggle,
}: {
  profile: ModelProfile
  active: boolean
  readOnly: boolean
  onSelect: () => void
  onToggle: () => void
}) {
  const dedicated = isDedicatedWorker(profile.loadOverrides)
  const defaultPreset = profile.runtimePresets.find((r) => r.isDefault)
  return (
    <div
      className={cn(
        "rounded-xl border bg-background/40 p-3 transition-colors",
        active ? "border-primary" : "border-border hover:border-muted-foreground/50",
        !profile.enabled && "opacity-60",
      )}
      data-testid={`profile-card-${profile.profileId}`}
    >
      <div className="flex items-start justify-between gap-2">
        <button type="button" onClick={onSelect} className="min-w-0 flex-1 text-left">
          <div className="flex items-center gap-1.5">
            {profile.isDefault && <Star size={13} className="shrink-0 fill-amber-400 text-amber-400" />}
            <span className="truncate text-sm font-medium">{profile.displayName || profile.profileId}</span>
          </div>
        </button>
        <Toggle checked={profile.enabled} disabled={readOnly} onChange={onToggle} />
      </div>
      <div className="mt-1 flex items-center gap-1">
        <code className="truncate font-mono text-xs text-muted-foreground">{profile.profileId}</code>
        <button type="button" title="Copiar el nombre de la API" onClick={() => copy(profile.profileId)} className="text-muted-foreground hover:text-foreground">
          <Copy size={12} />
        </button>
      </div>
      <div className="mt-2 flex flex-wrap gap-1">
        <span className={cn("rounded-full border px-2 py-0.5 text-[11px]", CATEGORY_TONE[profile.category] ?? CATEGORY_TONE.llm)}>
          {CATEGORIES.find((c) => c.value === profile.category)?.label ?? profile.category}
        </span>
        <Chip tone={dedicated ? "warn" : "muted"} title={dedicated ? "Sus sobrescrituras de carga arrancan un proceso propio" : "Comparte el proceso del despliegue"}>
          {dedicated ? "Worker dedicado" : "Worker compartido"}
        </Chip>
        {profile.runtimePresets.length > 0 && (
          <Chip>
            {profile.runtimePresets.length} presets{defaultPreset ? ` · ${defaultPreset.label}` : ""}
          </Chip>
        )}
        {profile.aliases.length > 0 && <Chip>{profile.aliases.length} alias</Chip>}
        {profile.requestDefaults && Object.keys(profile.requestDefaults).length > 0 && <Chip tone="info">valores por defecto</Chip>}
      </div>
    </div>
  )
}

function CreateProfile({
  modelId,
  defaultCategory,
  existing,
  onCancel,
  onCreated,
}: {
  modelId: string
  defaultCategory: ProfileCategory
  existing: string[]
  onCancel: () => void
  onCreated: (id: string) => Promise<void>
}) {
  const [id, setId] = useState("")
  const [name, setName] = useState("")
  const [category, setCategory] = useState<ProfileCategory>(defaultCategory)
  const [busy, setBusy] = useState(false)
  const slugOk = isValidSlug(id)
  const taken = existing.includes(id)
  const submit = async () => {
    setBusy(true)
    try {
      await api.profiles.create(modelId, {
        profileId: id,
        displayName: name || undefined,
        category,
        enabled: true,
        isDefault: false,
      })
      toast.success(`Perfil ${id} creado`)
      await onCreated(id)
    } catch (err) {
      toast.error(errMsg(err))
    } finally {
      setBusy(false)
    }
  }
  return (
    <Panel title="Nuevo perfil">
      <div className="grid gap-3 md:grid-cols-2">
        <div>
          <label className="mb-0.5 block text-xs font-medium text-muted-foreground">Id (nombre en la API)</label>
          <input className={inputCls} value={id} onChange={(e) => setId(e.target.value.trim())} placeholder="mi-modelo-creativo" data-testid="new-profile-id" />
          {id && !slugOk && <Hint tone="error">Minúsculas, números, «.», «_», «-» o «:»; sin «/» y empezando por letra o número.</Hint>}
          {id && taken && <Hint tone="error">Ya existe un perfil con ese id.</Hint>}
          {id && slugOk && !taken && (
            <Hint tone="info">
              Los clientes usarán <code>model=&apos;{id}&apos;</code>.
            </Hint>
          )}
        </div>
        <div>
          <label className="mb-0.5 block text-xs font-medium text-muted-foreground">Nombre visible</label>
          <input className={inputCls} value={name} onChange={(e) => setName(e.target.value)} />
        </div>
        <div className="md:col-span-2">
          <label className="mb-0.5 block text-xs font-medium text-muted-foreground">Categoría</label>
          <Segmented size="sm" value={category} options={CATEGORIES.map((c) => ({ value: c.value, label: c.label }))} onChange={setCategory} />
        </div>
      </div>
      <div className="mt-3 flex justify-end gap-2">
        <button type="button" onClick={onCancel} className="rounded-md border border-border px-3 py-1.5 text-xs hover:bg-muted">
          Cancelar
        </button>
        <button
          type="button"
          disabled={!slugOk || taken || busy}
          onClick={() => void submit()}
          className="rounded-md bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground disabled:opacity-50"
        >
          {busy ? "Creando…" : "Crear perfil"}
        </button>
      </div>
    </Panel>
  )
}

function ProfileEditor({
  profile,
  baseConfig,
  readOnly,
  routers,
  onChanged,
  onRenamed,
  onDeleted,
}: {
  profile: ModelProfile
  baseConfig: Record<string, unknown>
  readOnly: boolean
  routers: ModelProfile[]
  onChanged: () => Promise<void>
  onRenamed: (id: string) => Promise<void>
  onDeleted: () => Promise<void>
}) {
  const [displayName, setDisplayName] = useState(profile.displayName ?? "")
  const [description, setDescription] = useState(profile.description ?? "")
  const [category, setCategory] = useState<ProfileCategory>(profile.category)
  const [isDefault, setIsDefault] = useState(profile.isDefault)
  const [defaults, setDefaults] = useState<Defaults>(profile.requestDefaults ?? {})
  const [overrides, setOverrides] = useState<Record<string, unknown>>(profile.loadOverrides ?? {})
  const [rawDefaults, setRawDefaults] = useState(false)
  const [busy, setBusy] = useState(false)
  const [action, setAction] = useState<"rename" | "clone" | null>(null)
  const [newId, setNewId] = useState("")
  const [confirmDelete, setConfirmDelete] = useState(false)

  const dirty =
    displayName !== (profile.displayName ?? "") ||
    description !== (profile.description ?? "") ||
    category !== profile.category ||
    isDefault !== profile.isDefault ||
    JSON.stringify(defaults) !== JSON.stringify(profile.requestDefaults ?? {}) ||
    JSON.stringify(overrides) !== JSON.stringify(profile.loadOverrides ?? {})

  const diff = useMemo(() => overridesDiff(baseConfig, overrides), [baseConfig, overrides])
  const dedicated = isDedicatedWorker(overrides)

  const save = async () => {
    setBusy(true)
    try {
      await api.profiles.update(profile.profileId, {
        displayName,
        description,
        category,
        isDefault,
        // Vacío = null: el perfil queda sin valores por defecto / sin sobrescrituras.
        requestDefaults: Object.keys(defaults).length ? defaults : null,
        loadOverrides: Object.keys(overrides).length ? overrides : null,
      })
      toast.success("Perfil guardado")
      await onChanged()
    } catch (err) {
      toast.error(errMsg(err))
    } finally {
      setBusy(false)
    }
  }

  const runAction = async () => {
    if (!isValidSlug(newId)) return
    setBusy(true)
    try {
      if (action === "rename") {
        await api.profiles.rename(profile.profileId, newId)
        toast.success(`Renombrado a ${newId}`)
        setAction(null)
        await onRenamed(newId)
      } else {
        await api.profiles.clone(profile.profileId, newId)
        toast.success(`Clonado como ${newId}`)
        setAction(null)
        await onRenamed(newId)
      }
    } catch (err) {
      toast.error(errMsg(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="space-y-4" data-testid="profile-editor">
      <Panel
        title={
          <span className="flex items-center gap-2">
            Perfil <code className="font-mono text-sm">{profile.profileId}</code>
          </span>
        }
        subtitle="Se guarda al pulsar «Guardar perfil»; independiente del resto de la página."
        actions={
          !readOnly && (
            <>
              <button type="button" onClick={() => { setAction("clone"); setNewId(`${profile.profileId}-copia`) }} className="rounded-md border border-border px-2 py-1 text-xs hover:bg-muted">
                Clonar
              </button>
              <button type="button" onClick={() => { setAction("rename"); setNewId(profile.profileId) }} className="rounded-md border border-border px-2 py-1 text-xs hover:bg-muted">
                Renombrar
              </button>
              <button type="button" onClick={() => setConfirmDelete(true)} className="rounded-md border border-red-500/40 px-2 py-1 text-xs text-red-300 hover:bg-red-500/10">
                Borrar
              </button>
            </>
          )
        }
      >
        {action && (
          <div className="mb-3 rounded-lg border border-border bg-muted/20 p-3">
            <label className="mb-0.5 block text-xs font-medium text-muted-foreground">
              {action === "rename" ? "Nuevo id" : "Id de la copia"}
            </label>
            <div className="flex gap-2">
              <input className={inputCls} value={newId} onChange={(e) => setNewId(e.target.value.trim())} />
              <button type="button" disabled={!isValidSlug(newId) || newId === profile.profileId || busy} onClick={() => void runAction()} className="rounded-md bg-primary px-3 py-1 text-xs text-primary-foreground disabled:opacity-50">
                {action === "rename" ? "Renombrar" : "Clonar"}
              </button>
              <button type="button" onClick={() => setAction(null)} className="rounded-md border border-border px-3 py-1 text-xs">
                Cancelar
              </button>
            </div>
            {action === "rename" && (
              <Hint tone="warn">
                Los routers que apuntan a este perfil se actualizan solos, pero los clientes que pidan «{profile.profileId}» recibirán un 404.
              </Hint>
            )}
          </div>
        )}
        <div className="grid gap-3 md:grid-cols-2">
          <div>
            <label className="mb-0.5 block text-xs font-medium text-muted-foreground">Nombre visible</label>
            <input className={inputCls} disabled={readOnly} value={displayName} onChange={(e) => setDisplayName(e.target.value)} />
          </div>
          <div>
            <label className="mb-0.5 block text-xs font-medium text-muted-foreground">Descripción</label>
            <input className={inputCls} disabled={readOnly} value={description} onChange={(e) => setDescription(e.target.value)} />
          </div>
          <div className="md:col-span-2">
            <label className="mb-0.5 block text-xs font-medium text-muted-foreground">Categoría</label>
            <Segmented size="sm" disabled={readOnly} value={category} options={CATEGORIES.map((c) => ({ value: c.value, label: c.label }))} onChange={setCategory} />
            <Hint>Decide qué formulario de valores por defecto se muestra.</Hint>
          </div>
          <Toggle checked={isDefault} disabled={readOnly} onChange={setIsDefault} label="Perfil por defecto del despliegue" />
        </div>
      </Panel>

      <Panel
        title="Valores por defecto de las peticiones"
        subtitle="Se aplican cuando el cliente no manda el parámetro: el cliente siempre gana. Si el cliente pide un router, los valores del router van entre los de este perfil y los del cliente."
        actions={
          <Segmented size="sm" value={rawDefaults ? "json" : "form"} options={[{ value: "form", label: "Formulario" }, { value: "json", label: "JSON" }]} onChange={(v) => setRawDefaults(v === "json")} />
        }
      >
        {rawDefaults ? (
          <JsonInput
            value={defaults as Json}
            readOnly={readOnly}
            placeholder="{}"
            onChange={(v) => setDefaults(v && typeof v === "object" && !Array.isArray(v) ? (v as Defaults) : {})}
          />
        ) : (
          <DefaultsForm category={category} defaults={defaults} onChange={setDefaults} readOnly={readOnly} />
        )}
      </Panel>

      <Panel
        title="Sobrescrituras de carga"
        subtitle="Cambian la configuración del despliegue solo para este perfil. Unas sobrescrituras distintas arrancan un worker aparte, con su propia VRAM."
        actions={
          <Chip tone={dedicated ? "warn" : "muted"} title="El interruptor de diarización no cuenta: se decide por petición">
            {dedicated ? "Worker dedicado" : "Worker compartido"}
          </Chip>
        }
      >
        {diff.length > 0 ? (
          <table className="mb-3 w-full text-xs" data-testid="overrides-diff">
            <thead className="text-left text-muted-foreground">
              <tr>
                <th className="py-1 font-medium">Opción</th>
                <th className="py-1 font-medium">Despliegue</th>
                <th className="py-1 font-medium">Este perfil</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/50">
              {diff.map((row) => (
                <tr key={row.key}>
                  <td className="py-1 font-mono">{row.key}</td>
                  <td className="py-1 font-mono text-muted-foreground">{fmt(row.base)}</td>
                  <td className="py-1 font-mono text-sky-200">{fmt(row.override)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="mb-3 text-xs text-muted-foreground">Sin sobrescrituras: usa la configuración del despliegue tal cual.</p>
        )}
        <JsonInput
          value={overrides as Json}
          readOnly={readOnly}
          placeholder='{"llama_cpp": {"ctx_size": 32768}}'
          onChange={(v) => setOverrides(v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : {})}
        />
        <Hint>Se fusionan con la configuración del despliegue clave a clave; un null quita la opción del despliegue.</Hint>
      </Panel>

      <AssetsPanel profile={profile} category={category} readOnly={readOnly} onChanged={onChanged} />

      {routers.length > 0 && (
        <Hint tone="info">
          Lo usan como destino: {routers.map((r) => r.profileId).join(", ")}.
        </Hint>
      )}

      {!readOnly && (
        <div className="sticky bottom-2 flex justify-end gap-2">
          <button
            type="button"
            disabled={!dirty || busy}
            onClick={() => {
              setDisplayName(profile.displayName ?? "")
              setDescription(profile.description ?? "")
              setCategory(profile.category)
              setIsDefault(profile.isDefault)
              setDefaults(profile.requestDefaults ?? {})
              setOverrides(profile.loadOverrides ?? {})
            }}
            className="rounded-md border border-border bg-card px-3 py-1.5 text-xs disabled:opacity-50"
          >
            Descartar
          </button>
          <button type="button" disabled={!dirty || busy} onClick={() => void save()} className="rounded-md bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground disabled:opacity-50" data-testid="save-profile">
            {busy ? "Guardando…" : "Guardar perfil"}
          </button>
        </div>
      )}

      <ConfirmDialog
        open={confirmDelete}
        onOpenChange={setConfirmDelete}
        title={`Borrar el perfil ${profile.profileId}`}
        description={`Los clientes que pidan «${profile.profileId}» recibirán un 404, y los routers que lo usan lo perderán como destino. Los ficheros del modelo no se tocan.`}
        confirmLabel="Borrar"
        variant="danger"
        onConfirm={async () => {
          try {
            await api.profiles.delete(profile.profileId)
            toast.success("Perfil borrado")
            await onDeleted()
          } catch (err) {
            toast.error(errMsg(err))
          }
        }}
      />
    </div>
  )
}

function AssetsPanel({
  profile,
  category,
  readOnly,
  onChanged,
}: {
  profile: ModelProfile
  category: ProfileCategory
  readOnly: boolean
  onChanged: () => Promise<void>
}) {
  const fileRef = useRef<HTMLInputElement>(null)
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const [playing, setPlaying] = useState<string | null>(null)
  const [uploading, setUploading] = useState(false)
  const assets = profile.assets ?? {}

  useEffect(() => () => audioRef.current?.pause(), [])

  const upload = async (file: File) => {
    setUploading(true)
    try {
      await api.profiles.uploadAsset(profile.profileId, file)
      toast.success(`${file.name} subido`)
      await onChanged()
    } catch (err) {
      toast.error(errMsg(err))
    } finally {
      setUploading(false)
    }
  }

  const remove = async (key: string) => {
    try {
      await api.profiles.deleteAsset(profile.profileId, key)
      await onChanged()
    } catch (err) {
      toast.error(errMsg(err))
    }
  }

  const play = (key: string) => {
    if (playing === key) {
      audioRef.current?.pause()
      setPlaying(null)
      return
    }
    audioRef.current?.pause()
    const audio = new Audio(`/ocabra/profiles/${encodeURIComponent(profile.profileId)}/assets/${encodeURIComponent(key)}/file`)
    audio.onended = () => setPlaying(null)
    audio.play().catch(() => toast.error("No se pudo reproducir"))
    audioRef.current = audio
    setPlaying(key)
  }

  return (
    <Panel
      title="Ficheros del perfil"
      subtitle="Audio de referencia de voz, pesos LoRA… La clave es el nombre del fichero sin extensión: la voz de referencia tiene que llamarse voice_ref.* y el cliente no puede sustituirla."
      actions={
        !readOnly && (
          <>
            <input
              ref={fileRef}
              type="file"
              className="hidden"
              accept={ASSET_ACCEPT}
              onChange={(e) => {
                const f = e.target.files?.[0]
                if (f) void upload(f)
                e.target.value = ""
              }}
            />
            <button type="button" disabled={uploading} onClick={() => fileRef.current?.click()} className="inline-flex items-center gap-1 rounded-md border border-border px-2.5 py-1 text-xs hover:bg-muted disabled:opacity-50">
              <Upload size={13} /> {uploading ? "Subiendo…" : "Subir fichero"}
            </button>
          </>
        )
      }
    >
      {Object.keys(assets).length === 0 ? (
        <p className="text-xs text-muted-foreground">Sin ficheros.</p>
      ) : (
        <ul className="space-y-1.5">
          {Object.entries(assets).map(([key, value]) => {
            const info = value && typeof value === "object" ? (value as Record<string, unknown>) : {}
            const filename = String(info.filename ?? key)
            const size = Number(info.size_bytes ?? 0)
            const canPlay = (category === "tts" || key === "voice_ref") && isAudioFile(filename)
            return (
              <li key={key} className="flex items-center justify-between gap-2 rounded-lg border border-border/60 bg-muted/20 px-3 py-1.5 text-xs">
                <span className="min-w-0 truncate">
                  <span className="font-mono text-muted-foreground">{key}</span> — {filename}
                  {size > 0 && <span className="ml-2 text-muted-foreground">({(size / 1024).toFixed(0)} KB)</span>}
                  {key === "voice_ref" && <Chip tone="info">voz de referencia</Chip>}
                </span>
                <span className="flex items-center gap-1">
                  {canPlay && (
                    <button type="button" onClick={() => play(key)} className="rounded border border-border p-1 hover:bg-muted" title={playing === key ? "Pausar" : "Escuchar"}>
                      {playing === key ? <Pause size={12} /> : <Play size={12} />}
                    </button>
                  )}
                  {!readOnly && (
                    <button type="button" onClick={() => void remove(key)} className="rounded border border-red-500/40 p-1 text-red-300 hover:bg-red-500/10" title="Borrar">
                      <Trash2 size={12} />
                    </button>
                  )}
                </span>
              </li>
            )
          })}
        </ul>
      )}
    </Panel>
  )
}
