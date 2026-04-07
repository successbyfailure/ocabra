import { useEffect, useRef, useState } from "react"
import * as Dialog from "@radix-ui/react-dialog"
import {
  X,
  Upload,
  Trash2,
  Share2,
  Server,
  Play,
  ChevronDown,
  ChevronRight,
} from "lucide-react"
import { toast } from "sonner"
import { api } from "@/api/client"
import type { ModelProfile, ModelState, ProfileCategory } from "@/types"

interface ProfileModalProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  model: ModelState
  profile: ModelProfile | null // null = create mode
  onSaved: () => void
}

const CATEGORIES: { value: ProfileCategory; label: string }[] = [
  { value: "llm", label: "LLM" },
  { value: "tts", label: "TTS" },
  { value: "stt", label: "STT" },
  { value: "image", label: "Image" },
  { value: "music", label: "Music" },
]

const CATEGORY_COLORS: Record<ProfileCategory, string> = {
  llm: "bg-blue-500/20 text-blue-300 border-blue-500/40",
  tts: "bg-emerald-500/20 text-emerald-300 border-emerald-500/40",
  stt: "bg-amber-500/20 text-amber-300 border-amber-500/40",
  image: "bg-purple-500/20 text-purple-300 border-purple-500/40",
  music: "bg-pink-500/20 text-pink-300 border-pink-500/40",
}

function inferDefaultCategory(model: ModelState): ProfileCategory {
  if (model.capabilities.tts) return "tts"
  if (model.capabilities.audioTranscription) return "stt"
  if (model.capabilities.imageGeneration) return "image"
  if (model.capabilities.musicGeneration) return "music"
  return "llm"
}

function isValidSlug(value: string): boolean {
  return /^[a-z0-9][a-z0-9._-]*$/.test(value) && !value.includes("/")
}

function parseJsonSafe(text: string): Record<string, unknown> | null {
  if (!text.trim()) return {}
  try {
    const parsed = JSON.parse(text)
    if (typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>
    }
    return null
  } catch {
    return null
  }
}

/* ── Helpers: merge UI fields → JSON and back ───────────────────── */

type Defaults = Record<string, unknown>

function extractField(obj: Defaults, key: string): string {
  const v = obj[key]
  if (v === undefined || v === null) return ""
  return String(v)
}

function extractBool(obj: Defaults, key: string): boolean | null {
  const v = obj[key]
  if (v === true) return true
  if (v === false) return false
  return null
}

function mergeField(obj: Defaults, key: string, value: string): Defaults {
  const next = { ...obj }
  if (value === "") {
    delete next[key]
  } else {
    // try numeric
    const n = Number(value)
    next[key] = isNaN(n) ? value : n
  }
  return next
}

function mergeBool(obj: Defaults, key: string, value: boolean | null): Defaults {
  const next = { ...obj }
  if (value === null) {
    delete next[key]
  } else {
    next[key] = value
  }
  return next
}

/* ── Compact form field components ──────────────────────────────── */

function Field({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="mb-0.5 block text-xs font-medium text-muted-foreground">{label}</label>
      {children}
      {hint && <p className="mt-0.5 text-[10px] text-muted-foreground/70">{hint}</p>}
    </div>
  )
}

function SmallInput({ value, onChange, placeholder, type = "text" }: {
  value: string; onChange: (v: string) => void; placeholder?: string; type?: string
}) {
  return (
    <input
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      type={type}
      className="w-full rounded-md border border-border bg-background px-2.5 py-1.5 text-sm"
    />
  )
}

function TriStateToggle({ value, onChange, labelOn, labelOff }: {
  value: boolean | null; onChange: (v: boolean | null) => void; labelOn: string; labelOff: string
}) {
  return (
    <div className="flex gap-1">
      {([
        { v: null, label: "Auto" },
        { v: true, label: labelOn },
        { v: false, label: labelOff },
      ] as const).map(({ v, label }) => (
        <button
          key={label}
          type="button"
          onClick={() => onChange(v)}
          className={`rounded-md border px-2 py-1 text-xs transition-colors ${
            value === v
              ? "border-primary bg-primary/20 text-primary-foreground"
              : "border-border text-muted-foreground hover:bg-muted"
          }`}
        >
          {label}
        </button>
      ))}
    </div>
  )
}

/* ── Category-specific request defaults panels ──────────────────── */

function TTSDefaults({ defaults, onChange }: { defaults: Defaults; onChange: (d: Defaults) => void }) {
  return (
    <div className="grid grid-cols-2 gap-3">
      <Field label="Voz por defecto" hint="alloy, echo, fable, nova, onyx, shimmer o speaker name">
        <SmallInput
          value={extractField(defaults, "voice")}
          onChange={(v) => onChange(mergeField(defaults, "voice", v))}
          placeholder="alloy"
        />
      </Field>
      <Field label="Velocidad" hint="0.25 – 4.0">
        <SmallInput
          value={extractField(defaults, "speed")}
          onChange={(v) => onChange(mergeField(defaults, "speed", v))}
          placeholder="1.0"
          type="number"
        />
      </Field>
      <Field label="Idioma">
        <SmallInput
          value={extractField(defaults, "language")}
          onChange={(v) => onChange(mergeField(defaults, "language", v))}
          placeholder="Auto"
        />
      </Field>
      <Field label="Speaker" hint="Para modelos CustomVoice (ryan, vivian, etc.)">
        <SmallInput
          value={extractField(defaults, "speaker")}
          onChange={(v) => onChange(mergeField(defaults, "speaker", v))}
          placeholder=""
        />
      </Field>
      <Field label="Formato de audio">
        <select
          value={extractField(defaults, "response_format") || ""}
          onChange={(e) => onChange(mergeField(defaults, "response_format", e.target.value))}
          className="w-full rounded-md border border-border bg-background px-2.5 py-1.5 text-sm"
        >
          <option value="">Sin preferencia</option>
          <option value="mp3">MP3</option>
          <option value="wav">WAV</option>
          <option value="opus">Opus</option>
          <option value="flac">FLAC</option>
        </select>
      </Field>
      <Field label="Instruccion de estilo" hint="Para modelos que soporten instrucciones de estilo">
        <SmallInput
          value={extractField(defaults, "instruct")}
          onChange={(v) => onChange(mergeField(defaults, "instruct", v))}
          placeholder="Speak calmly and slowly"
        />
      </Field>
    </div>
  )
}

function STTDefaults({ defaults, onChange }: { defaults: Defaults; onChange: (d: Defaults) => void }) {
  return (
    <div className="grid grid-cols-2 gap-3">
      <Field label="Idioma" hint="es, en, fr... o vacio para auto-deteccion">
        <SmallInput
          value={extractField(defaults, "language")}
          onChange={(v) => onChange(mergeField(defaults, "language", v))}
          placeholder="Auto"
        />
      </Field>
      <Field label="Diarizacion">
        <TriStateToggle
          value={extractBool(defaults, "diarize")}
          onChange={(v) => onChange(mergeBool(defaults, "diarize", v))}
          labelOn="Activar"
          labelOff="Desactivar"
        />
      </Field>
      <Field label="Temperatura" hint="0.0 – 1.0, menor = mas preciso">
        <SmallInput
          value={extractField(defaults, "temperature")}
          onChange={(v) => onChange(mergeField(defaults, "temperature", v))}
          placeholder="0.0"
          type="number"
        />
      </Field>
      <Field label="Formato respuesta">
        <select
          value={extractField(defaults, "response_format") || ""}
          onChange={(e) => onChange(mergeField(defaults, "response_format", e.target.value))}
          className="w-full rounded-md border border-border bg-background px-2.5 py-1.5 text-sm"
        >
          <option value="">Sin preferencia</option>
          <option value="json">JSON</option>
          <option value="verbose_json">JSON Verbose</option>
          <option value="text">Texto plano</option>
          <option value="srt">SRT</option>
          <option value="vtt">VTT</option>
        </select>
      </Field>
    </div>
  )
}

function LLMDefaults({ defaults, onChange }: { defaults: Defaults; onChange: (d: Defaults) => void }) {
  return (
    <div className="grid grid-cols-2 gap-3">
      <Field label="Temperatura" hint="0.0 – 2.0, mayor = mas creativo">
        <SmallInput
          value={extractField(defaults, "temperature")}
          onChange={(v) => onChange(mergeField(defaults, "temperature", v))}
          placeholder="0.7"
          type="number"
        />
      </Field>
      <Field label="Max tokens">
        <SmallInput
          value={extractField(defaults, "max_tokens")}
          onChange={(v) => onChange(mergeField(defaults, "max_tokens", v))}
          placeholder=""
          type="number"
        />
      </Field>
      <Field label="Top P" hint="0.0 – 1.0">
        <SmallInput
          value={extractField(defaults, "top_p")}
          onChange={(v) => onChange(mergeField(defaults, "top_p", v))}
          placeholder="1.0"
          type="number"
        />
      </Field>
      <Field label="System prompt">
        <SmallInput
          value={extractField(defaults, "system_prompt")}
          onChange={(v) => onChange(mergeField(defaults, "system_prompt", v))}
          placeholder=""
        />
      </Field>
    </div>
  )
}

function ImageDefaults({ defaults, onChange }: { defaults: Defaults; onChange: (d: Defaults) => void }) {
  return (
    <div className="grid grid-cols-2 gap-3">
      <Field label="Tamano">
        <select
          value={extractField(defaults, "size") || ""}
          onChange={(e) => onChange(mergeField(defaults, "size", e.target.value))}
          className="w-full rounded-md border border-border bg-background px-2.5 py-1.5 text-sm"
        >
          <option value="">Sin preferencia</option>
          <option value="256x256">256x256</option>
          <option value="512x512">512x512</option>
          <option value="1024x1024">1024x1024</option>
        </select>
      </Field>
      <Field label="Steps" hint="Pasos de inferencia">
        <SmallInput
          value={extractField(defaults, "num_inference_steps")}
          onChange={(v) => onChange(mergeField(defaults, "num_inference_steps", v))}
          placeholder="30"
          type="number"
        />
      </Field>
      <Field label="Guidance scale">
        <SmallInput
          value={extractField(defaults, "guidance_scale")}
          onChange={(v) => onChange(mergeField(defaults, "guidance_scale", v))}
          placeholder="7.5"
          type="number"
        />
      </Field>
      <Field label="Negative prompt">
        <SmallInput
          value={extractField(defaults, "negative_prompt")}
          onChange={(v) => onChange(mergeField(defaults, "negative_prompt", v))}
          placeholder=""
        />
      </Field>
    </div>
  )
}

/* ── Main component ────────────────────────────────────────────── */

export function ProfileModal({ open, onOpenChange, model, profile, onSaved }: ProfileModalProps) {
  const isEdit = profile !== null

  const [profileId, setProfileId] = useState("")
  const [displayName, setDisplayName] = useState("")
  const [description, setDescription] = useState("")
  const [category, setCategory] = useState<ProfileCategory>("llm")
  const [loadOverridesText, setLoadOverridesText] = useState("")
  const [requestDefaults, setRequestDefaults] = useState<Defaults>({})
  const [requestDefaultsRawMode, setRequestDefaultsRawMode] = useState(false)
  const [requestDefaultsRaw, setRequestDefaultsRaw] = useState("")
  const [enabled, setEnabled] = useState(true)
  const [isDefault, setIsDefault] = useState(false)
  const [saving, setSaving] = useState(false)
  const [deleting, setDeleting] = useState(false)
  const [confirmDelete, setConfirmDelete] = useState(false)
  const [showAdvanced, setShowAdvanced] = useState(false)

  // Assets
  const [assets, setAssets] = useState<Record<string, unknown>>({})
  const [uploadingAsset, setUploadingAsset] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [playingAsset, setPlayingAsset] = useState<string | null>(null)
  const audioRef = useRef<HTMLAudioElement | null>(null)

  useEffect(() => {
    if (!open) {
      setConfirmDelete(false)
      setShowAdvanced(false)
      setRequestDefaultsRawMode(false)
      return
    }
    if (profile) {
      setProfileId(profile.profileId)
      setDisplayName(profile.displayName ?? "")
      setDescription(profile.description ?? "")
      setCategory(profile.category)
      setLoadOverridesText(
        profile.loadOverrides && Object.keys(profile.loadOverrides).length > 0
          ? JSON.stringify(profile.loadOverrides, null, 2)
          : "",
      )
      const rd = (profile.requestDefaults ?? {}) as Defaults
      setRequestDefaults(rd)
      setRequestDefaultsRaw(Object.keys(rd).length > 0 ? JSON.stringify(rd, null, 2) : "")
      setEnabled(profile.enabled)
      setIsDefault(profile.isDefault)
      setAssets(profile.assets ?? {})
    } else {
      setProfileId("")
      setDisplayName("")
      setDescription("")
      setCategory(inferDefaultCategory(model))
      setLoadOverridesText("")
      setRequestDefaults({})
      setRequestDefaultsRaw("")
      setEnabled(true)
      setIsDefault(false)
      setAssets({})
    }
  }, [open, profile, model])

  // Sync raw ↔ structured when toggling mode
  const toggleRawMode = () => {
    if (requestDefaultsRawMode) {
      // raw → structured
      const parsed = parseJsonSafe(requestDefaultsRaw)
      if (parsed !== null) setRequestDefaults(parsed)
    } else {
      // structured → raw
      setRequestDefaultsRaw(
        Object.keys(requestDefaults).length > 0
          ? JSON.stringify(requestDefaults, null, 2)
          : "",
      )
    }
    setRequestDefaultsRawMode(!requestDefaultsRawMode)
  }

  const handleSave = async () => {
    if (!isEdit && !isValidSlug(profileId)) {
      toast.error("El profile_id debe ser un slug valido (minusculas, sin espacios, sin /)")
      return
    }

    const overrides = parseJsonSafe(loadOverridesText)
    if (overrides === null) {
      toast.error("load_overrides no es JSON valido")
      return
    }

    let defaults: Defaults
    if (requestDefaultsRawMode) {
      const parsed = parseJsonSafe(requestDefaultsRaw)
      if (parsed === null) {
        toast.error("request_defaults no es JSON valido")
        return
      }
      defaults = parsed
    } else {
      defaults = requestDefaults
    }

    setSaving(true)
    try {
      if (isEdit) {
        await api.profiles.update(profile.profileId, {
          displayName: displayName || undefined,
          description: description || undefined,
          category,
          loadOverrides: Object.keys(overrides).length > 0 ? overrides : undefined,
          requestDefaults: Object.keys(defaults).length > 0 ? defaults : undefined,
          enabled,
          isDefault,
        })
        toast.success("Perfil actualizado")
      } else {
        await api.profiles.create(model.modelId, {
          profileId,
          displayName: displayName || undefined,
          description: description || undefined,
          category,
          loadOverrides: Object.keys(overrides).length > 0 ? overrides : undefined,
          requestDefaults: Object.keys(defaults).length > 0 ? defaults : undefined,
          enabled,
          isDefault,
        })
        toast.success("Perfil creado")
      }
      onSaved()
      onOpenChange(false)
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Error al guardar perfil")
    } finally {
      setSaving(false)
    }
  }

  const handleDelete = async () => {
    if (!profile) return
    if (!confirmDelete) {
      setConfirmDelete(true)
      return
    }
    setDeleting(true)
    try {
      await api.profiles.delete(profile.profileId)
      toast.success("Perfil eliminado")
      onSaved()
      onOpenChange(false)
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Error al eliminar perfil")
    } finally {
      setDeleting(false)
      setConfirmDelete(false)
    }
  }

  const handleAssetUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file || !profile) return
    setUploadingAsset(true)
    try {
      const updated = await api.profiles.uploadAsset(profile.profileId, file)
      setAssets(updated.assets ?? {})
      toast.success(`Asset "${file.name}" subido`)
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Error al subir asset")
    } finally {
      setUploadingAsset(false)
      if (fileInputRef.current) fileInputRef.current.value = ""
    }
  }

  const handleAssetDelete = async (assetKey: string) => {
    if (!profile) return
    try {
      const updated = await api.profiles.deleteAsset(profile.profileId, assetKey)
      setAssets(updated.assets ?? {})
      toast.success(`Asset "${assetKey}" eliminado`)
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Error al eliminar asset")
    }
  }

  const isAudioFile = (filename: string) =>
    /\.(wav|mp3|flac|ogg)$/i.test(filename)

  const hasOverrides = loadOverridesText.trim().length > 0 && loadOverridesText.trim() !== "{}"

  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-40 bg-black/60" />
        <Dialog.Content className="fixed left-1/2 top-1/2 z-50 max-h-[90vh] w-[95vw] max-w-2xl -translate-x-1/2 -translate-y-1/2 overflow-y-auto rounded-lg border border-border bg-card p-5">
          <div className="mb-4 flex items-center justify-between">
            <Dialog.Title className="text-lg font-semibold">
              {isEdit ? "Editar perfil" : "Nuevo perfil"}
            </Dialog.Title>
            <Dialog.Close className="rounded-md p-1 hover:bg-muted">
              <X size={18} />
            </Dialog.Close>
          </div>

          <div className="space-y-4">
            {/* Profile ID */}
            <div>
              <label className="mb-1 block text-sm font-medium">Profile ID</label>
              {isEdit ? (
                <div className="rounded-md border border-border bg-muted/40 px-3 py-2 text-sm text-muted-foreground">
                  {profileId}
                </div>
              ) : (
                <>
                  <input
                    value={profileId}
                    onChange={(e) => setProfileId(e.target.value.toLowerCase().replace(/\s/g, "-"))}
                    placeholder="mi-perfil-custom"
                    className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                  />
                  {profileId && (
                    <p className="mt-1 text-xs text-muted-foreground">
                      Los clientes usaran: <code className="rounded bg-muted px-1 py-0.5 font-mono text-xs">model=&apos;{profileId}&apos;</code>
                    </p>
                  )}
                  {profileId && !isValidSlug(profileId) && (
                    <p className="mt-1 text-xs text-red-400">
                      Solo minusculas, numeros, puntos, guiones. Sin espacios ni /.
                    </p>
                  )}
                </>
              )}
            </div>

            {/* Display Name + Description */}
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="mb-1 block text-sm font-medium">Nombre visible</label>
                <input
                  value={displayName}
                  onChange={(e) => setDisplayName(e.target.value)}
                  placeholder="Nombre para la UI"
                  className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                />
              </div>
              <div>
                <label className="mb-1 block text-sm font-medium">Categoria</label>
                <select
                  value={category}
                  onChange={(e) => setCategory(e.target.value as ProfileCategory)}
                  className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                >
                  {CATEGORIES.map((c) => (
                    <option key={c.value} value={c.value}>
                      {c.label}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div>
              <label className="mb-1 block text-sm font-medium">Descripcion</label>
              <textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Descripcion breve del perfil"
                rows={2}
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
              />
            </div>

            {/* ── Request Defaults (category-specific UI) ────────── */}
            <div className="rounded-md border border-border/60 bg-muted/10 p-3 space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-medium">Parametros por defecto</h3>
                <button
                  type="button"
                  onClick={toggleRawMode}
                  className="rounded-md border border-border px-2 py-0.5 text-[10px] text-muted-foreground hover:bg-muted"
                >
                  {requestDefaultsRawMode ? "Vista formulario" : "Editar JSON"}
                </button>
              </div>
              <p className="text-[10px] text-muted-foreground -mt-1">
                Valores que se inyectan como defaults en cada request. El cliente puede sobreescribirlos.
              </p>

              {requestDefaultsRawMode ? (
                <textarea
                  value={requestDefaultsRaw}
                  onChange={(e) => setRequestDefaultsRaw(e.target.value)}
                  placeholder='{"temperature": 0.7}'
                  rows={5}
                  className="w-full rounded-md border border-border bg-background px-3 py-2 font-mono text-xs"
                  spellCheck={false}
                />
              ) : (
                <>
                  {category === "tts" && (
                    <TTSDefaults defaults={requestDefaults} onChange={setRequestDefaults} />
                  )}
                  {category === "stt" && (
                    <STTDefaults defaults={requestDefaults} onChange={setRequestDefaults} />
                  )}
                  {category === "llm" && (
                    <LLMDefaults defaults={requestDefaults} onChange={setRequestDefaults} />
                  )}
                  {category === "image" && (
                    <ImageDefaults defaults={requestDefaults} onChange={setRequestDefaults} />
                  )}
                  {category === "music" && (
                    <LLMDefaults defaults={requestDefaults} onChange={setRequestDefaults} />
                  )}
                </>
              )}
            </div>

            {/* Toggles */}
            <div className="flex flex-wrap items-center gap-6">
              <label className="flex items-center gap-2 text-sm">
                <button
                  type="button"
                  role="switch"
                  aria-checked={enabled}
                  onClick={() => setEnabled(!enabled)}
                  className={`relative h-5 w-9 rounded-full transition-colors ${
                    enabled ? "bg-emerald-500" : "bg-muted"
                  }`}
                >
                  <span
                    className={`absolute top-0.5 block h-4 w-4 rounded-full bg-white transition-transform ${
                      enabled ? "translate-x-4" : "translate-x-0.5"
                    }`}
                  />
                </button>
                Habilitado
              </label>

              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={isDefault}
                  onChange={(e) => setIsDefault(e.target.checked)}
                  className="h-4 w-4 rounded border-border"
                />
                Perfil por defecto
              </label>
            </div>

            {/* Assets section (only in edit mode) */}
            {isEdit && (
              <div className="space-y-3 rounded-md border border-border p-3">
                <h3 className="text-sm font-medium">Assets</h3>
                <p className="text-[10px] text-muted-foreground -mt-1">
                  Archivos asociados al perfil: audios de referencia para voz, pesos LoRA, etc.
                </p>
                {Object.keys(assets).length > 0 ? (
                  <div className="space-y-2">
                    {Object.entries(assets).map(([key, value]) => {
                      const info = typeof value === "object" && value !== null ? (value as Record<string, unknown>) : {}
                      const filename = String(info.filename ?? key)
                      const sizeBytes = Number(info.size_bytes ?? 0)
                      const canPlay = category === "tts" && isAudioFile(filename)
                      return (
                        <div key={key} className="flex items-center justify-between gap-2 rounded border border-border/60 bg-muted/20 px-3 py-2 text-sm">
                          <div className="min-w-0 flex-1">
                            <span className="font-mono text-xs text-muted-foreground">{key}</span>
                            <span className="mx-2 text-muted-foreground">&mdash;</span>
                            <span>{filename}</span>
                            {sizeBytes > 0 && (
                              <span className="ml-2 text-xs text-muted-foreground">
                                ({(sizeBytes / 1024).toFixed(0)} KB)
                              </span>
                            )}
                          </div>
                          <div className="flex items-center gap-1">
                            {canPlay && (
                              <button
                                type="button"
                                onClick={() => {
                                  if (playingAsset === key) {
                                    audioRef.current?.pause()
                                    setPlayingAsset(null)
                                  } else {
                                    const path = String(info.path ?? "")
                                    if (path) {
                                      if (audioRef.current) audioRef.current.pause()
                                      const audio = new Audio(`/ocabra/profiles/${encodeURIComponent(profile.profileId)}/assets/${encodeURIComponent(key)}/file`)
                                      audio.onended = () => setPlayingAsset(null)
                                      audio.play().catch(() => toast.error("No se pudo reproducir"))
                                      audioRef.current = audio
                                      setPlayingAsset(key)
                                    }
                                  }
                                }}
                                className="rounded-md border border-border p-1 hover:bg-muted"
                                title={playingAsset === key ? "Pausar" : "Reproducir"}
                              >
                                <Play size={12} />
                              </button>
                            )}
                            <button
                              type="button"
                              onClick={() => handleAssetDelete(key)}
                              className="rounded-md border border-red-500/40 p-1 text-red-300 hover:bg-red-500/10"
                              title="Eliminar asset"
                            >
                              <Trash2 size={12} />
                            </button>
                          </div>
                        </div>
                      )
                    })}
                  </div>
                ) : (
                  <p className="text-xs text-muted-foreground">Sin assets.</p>
                )}

                <div>
                  <input
                    ref={fileInputRef}
                    type="file"
                    onChange={handleAssetUpload}
                    className="hidden"
                    accept=".wav,.mp3,.flac,.ogg,.safetensors,.bin,.pt,.gguf,.json"
                  />
                  <button
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    disabled={uploadingAsset}
                    className="flex items-center gap-2 rounded-md border border-border px-3 py-1.5 text-sm hover:bg-muted disabled:opacity-50"
                  >
                    <Upload size={14} />
                    {uploadingAsset ? "Subiendo..." : "Subir asset"}
                  </button>
                </div>
              </div>
            )}

            {/* ── Advanced: Load Overrides ────────────────────────── */}
            <div>
              <button
                type="button"
                onClick={() => setShowAdvanced(!showAdvanced)}
                className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
              >
                {showAdvanced ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                Avanzado: Load Overrides
                {hasOverrides && (
                  <span className="ml-1 inline-flex items-center gap-1 rounded border border-amber-500/40 bg-amber-500/10 px-1.5 py-0.5 text-[10px] text-amber-300">
                    <Server size={10} /> Worker dedicado
                  </span>
                )}
                {!hasOverrides && (
                  <span className="ml-1 inline-flex items-center gap-1 rounded border border-emerald-500/40 bg-emerald-500/10 px-1.5 py-0.5 text-[10px] text-emerald-300">
                    <Share2 size={10} /> Worker compartido
                  </span>
                )}
              </button>
              {showAdvanced && (
                <div className="mt-2 space-y-1">
                  <p className="text-[10px] text-muted-foreground">
                    Overrides de configuracion de carga del modelo (JSON). Si difieren entre perfiles, se crea un worker GPU separado.
                  </p>
                  <textarea
                    value={loadOverridesText}
                    onChange={(e) => setLoadOverridesText(e.target.value)}
                    placeholder='{"max_model_len": 8192}'
                    rows={4}
                    className="w-full rounded-md border border-border bg-background px-3 py-2 font-mono text-xs"
                    spellCheck={false}
                  />
                </div>
              )}
            </div>
          </div>

          {/* Actions */}
          <div className="mt-5 flex items-center justify-between">
            <div>
              {isEdit && (
                <button
                  type="button"
                  onClick={handleDelete}
                  disabled={deleting || saving}
                  className="rounded-md border border-red-500/40 px-3 py-2 text-sm text-red-200 hover:bg-red-500/20 disabled:opacity-50"
                >
                  {confirmDelete ? "Confirmar eliminacion" : "Eliminar perfil"}
                </button>
              )}
            </div>
            <div className="flex gap-2">
              <Dialog.Close className="rounded-md border border-border px-3 py-2 text-sm hover:bg-muted">
                Cancelar
              </Dialog.Close>
              <button
                type="button"
                onClick={handleSave}
                disabled={saving || (!isEdit && !profileId)}
                className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
              >
                {saving ? "Guardando..." : isEdit ? "Guardar" : "Crear"}
              </button>
            </div>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  )
}

export { CATEGORY_COLORS }
