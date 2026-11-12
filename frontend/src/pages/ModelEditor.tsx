import { useCallback, useEffect, useMemo, useState } from "react"
import { useLocation, useNavigate, useParams, useSearchParams } from "react-router-dom"
import {
  ArrowLeft,
  Box,
  Cpu,
  Hammer,
  Info,
  Layers,
  Loader2,
  Lock,
  MessageSquare,
  Mic,
  Power,
  PowerOff,
  Rocket,
  Save,
  Settings2,
  ShieldCheck,
  Undo2,
  Users,
  Volume2,
  XCircle,
  ImageIcon,
} from "lucide-react"
import { toast } from "sonner"
import { api } from "@/api/client"
import { BackendIcon } from "@/components/dashboard/BackendIcon"
import { useLiveActivity } from "@/components/dashboard/useLiveActivity"
import { ConfirmDialog } from "@/components/common/ConfirmDialog"
import { CompileModal } from "@/components/models/CompileModal"
import { ModelStatusBadge } from "@/components/models/ModelStatusBadge"
import { MobilePreviewBar, PreviewPanel } from "@/components/model-editor/PreviewPanel"
import { AccessSection } from "@/components/model-editor/sections/AccessSection"
import { CapacitySection } from "@/components/model-editor/sections/CapacitySection"
import { LoadSection } from "@/components/model-editor/sections/LoadSection"
import { AccelerationSection, AdvancedSection, GroupSection, ResponseSection } from "@/components/model-editor/sections/MiscSections"
import { ProfilesSection } from "@/components/model-editor/sections/ProfilesSection"
import { SummarySection } from "@/components/model-editor/sections/SummarySection"
import type { SectionProps } from "@/components/model-editor/sections/types"
import { Callout, Chip } from "@/components/model-editor/ui"
import { useModelEditor } from "@/components/model-editor/useModelEditor"
import { usePresetLadder } from "@/components/model-editor/usePresetLadder"
import { modelEditorApi } from "@/api/modelEditor"
import type { ModelInsights } from "@/components/model-editor/types"
import { useIsAdmin, useIsModelManager } from "@/hooks/useAuth"
import { cn } from "@/lib/cn"
import { useGpuStore } from "@/stores/gpuStore"
import type { LogicalModel, ModelState } from "@/types"

type SectionId =
  | "summary"
  | "load"
  | "capacity"
  | "response"
  | "acceleration"
  | "audio"
  | "image"
  | "voice"
  | "runtime"
  | "profiles"
  | "access"
  | "advanced"

const SECTIONS: { id: SectionId; label: string; icon: typeof Info; llmOnly?: boolean; group?: string }[] = [
  { id: "summary", label: "Resumen", icon: Info },
  { id: "load", label: "Carga", icon: Power },
  { id: "capacity", label: "Capacidad", icon: Layers },
  { id: "response", label: "Respuesta", icon: MessageSquare, llmOnly: true },
  { id: "acceleration", label: "Aceleración", icon: Rocket },
  { id: "audio", label: "Audio", icon: Mic, group: "audio" },
  { id: "image", label: "Imagen", icon: ImageIcon, group: "image" },
  { id: "voice", label: "Voz", icon: Volume2, group: "voice" },
  { id: "runtime", label: "Arranque", icon: Box, group: "runtime" },
  { id: "profiles", label: "Perfiles", icon: Users },
  { id: "access", label: "Acceso", icon: ShieldCheck },
  { id: "advanced", label: "Avanzado", icon: Settings2 },
]

/** ``llamacpp_mks/ornith-1.5:35b`` → ``ornith-1.5:35b``. */
function modelKeyOf(modelId: string): string {
  const i = modelId.indexOf("/")
  return i >= 0 ? modelId.slice(i + 1) : modelId
}

export function ModelEditor() {
  const params = useParams()
  const modelId = params["*"] ? decodeURIComponent(params["*"]) : null
  const navigate = useNavigate()
  const location = useLocation()
  const [search, setSearch] = useSearchParams()
  // GPUs, Contexto y Presets se fundieron en Capacidad (enlaces viejos incluidos).
  const rawSection = search.get("s")
  const section = (["gpus", "context", "presets"].includes(rawSection ?? "") ? "capacity" : rawSection) as SectionId | null ?? "summary"
  const setSection = (s: SectionId) => setSearch((q) => {
    q.set("s", s)
    return q
  }, { replace: true })

  const canEdit = useIsModelManager()
  const isAdmin = useIsAdmin()
  const readOnly = !canEdit
  const editor = useModelEditor(modelId)
  const gpus = useGpuStore((s) => s.gpus)
  const { data: activity } = useLiveActivity()
  const ladder = usePresetLadder(modelId)
  const [logical, setLogical] = useState<LogicalModel | null>(null)
  const [modelState, setModelState] = useState<ModelState | null>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const [confirm, setConfirm] = useState<null | "unload" | "reload" | "delete">(null)
  const [compileOpen, setCompileOpen] = useState(false)
  const [pendingNav, setPendingNav] = useState<string | null>(null)
  const [insights, setInsights] = useState<ModelInsights | null>(null)
  const reloadInsights = useCallback(async () => {
    if (!modelId) return
    try {
      setInsights(await modelEditorApi.insights(modelId))
    } catch {
      setInsights(null)
    }
  }, [modelId])
  useEffect(() => {
    void reloadInsights()
  }, [reloadInsights])

  const refreshLogical = useCallback(async () => {
    if (!modelId) return
    try {
      setLogical(await api.logicalModels.get(modelKeyOf(modelId)))
    } catch {
      setLogical(null)
    }
    try {
      setModelState(await api.models.get(modelId))
      await ladder.reload()
    } catch {
      /* el estado camelCase solo alimenta iconos */
    }
  }, [modelId])

  useEffect(() => {
    void refreshLogical()
  }, [refreshLogical])

  // Estado vivo (cargando, VRAM…) cada pocos segundos, sin tocar el borrador.
  useEffect(() => {
    const id = window.setInterval(() => void editor.refreshLive(), 5000)
    return () => window.clearInterval(id)
  }, [editor.refreshLive]) // eslint-disable-line react-hooks/exhaustive-deps

  const liveVramByGpu = useMemo(() => (modelId ? (activity.vramByGpu?.[modelId] ?? {}) : {}), [activity, modelId])

  // Avisar antes de salir con cambios sin guardar (BrowserRouter no tiene
  // useBlocker: se intercepta la navegación propia y el cierre de pestaña).
  const go = (to: string) => (editor.dirty ? setPendingNav(to) : navigate(to))
  useEffect(() => {
    const onBeforeUnload = (e: BeforeUnloadEvent) => {
      if (editor.dirty) e.preventDefault()
    }
    window.addEventListener("beforeunload", onBeforeUnload)
    return () => window.removeEventListener("beforeunload", onBeforeUnload)
  }, [editor.dirty])

  if (!modelId) return <Callout tone="error">Falta el modelo en la dirección.</Callout>
  if (editor.error) {
    return (
      <div className="space-y-3 p-4">
        <Callout tone="error">No se pudo abrir el modelo: {editor.error}</Callout>
        <button type="button" className="text-sm underline" onClick={() => navigate("/models")}>
          Volver a Modelos
        </button>
      </div>
    )
  }
  if (!editor.model || !editor.schema || !editor.draft) {
    return (
      <div className="flex items-center gap-2 p-6 text-sm text-muted-foreground">
        <Loader2 size={16} className="animate-spin" /> Cargando el editor…
      </div>
    )
  }

  const m = editor.model
  const schema = editor.schema
  const visible = SECTIONS.filter((s) => {
    if (s.llmOnly && !schema.isLlm) return false
    if (s.group && !editor.fields.some((f) => f.group === s.group)) return false
    if (s.id === "access" && !isAdmin) return false
    return true
  })
  const active = visible.some((s) => s.id === section) ? section : "summary"
  const dirtyGroups = new Set(
    editor.changes.map((c) => {
      if (c.key === "loadPolicy" || c.key === "pinPermanent" || c.key === "schedules") return "load"
      if (c.key === "preferredGpu") return "capacity"
      const group = editor.fieldByKey.get(c.key)?.group ?? "advanced"
      return group === "gpus" || group === "context" ? "capacity" : group
    }),
  )
  const errorGroups = new Set(
    editor.previewErrors.map((e) => {
      const group = editor.fieldByKey.get(e.key)?.group ?? "advanced"
      return group === "gpus" || group === "context" ? "capacity" : group
    }),
  )
  const status = (m.runtime_phase ?? m.status) as ModelState["status"]
  const loaded = m.status === "loaded"
  const loading = m.status === "loading"
  const needsReload = loaded && editor.changes.some((c) => c.reload)

  // Ollama no admite presets: aunque quedasen en la base de datos, se ignoran.
  const isOllama = m.backend_type === "ollama"
  const rungs = isOllama ? [] : ladder.rungs
  const capacityLadder = isOllama ? { ...ladder, rungs: [], profile: null } : ladder
  const props: SectionProps = { editor, readOnly, isAdmin, logical, refreshLogical, gpus, liveVramByGpu, presets: rungs, insights, reloadInsights }

  const run = async (label: string, fn: () => Promise<unknown>, ok: string) => {
    setBusy(label)
    try {
      await fn()
      toast.success(ok)
      await editor.refreshLive()
      await refreshLogical()
    } catch (err) {
      toast.error(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(null)
    }
  }

  const save = async (andReload: boolean) => {
    // Solo bloquea lo que se está cambiando: un valor guardado que el backend
    // ignora (p. ej. un resto del editor viejo) no impide guardar lo demás.
    const blocking = editor.previewErrors.filter((e) =>
      editor.changes.some((c) => c.key === e.key || e.key.startsWith(`${c.key}.`) || c.key.startsWith(`${e.key}.`)),
    )
    if (blocking.length) {
      toast.error(`Corrige antes de guardar: ${blocking.map((e) => e.label).join(", ")}`)
      return
    }
    setBusy(andReload ? "save-reload" : "save")
    try {
      await editor.save()
      if (andReload) {
        await api.models.unload(modelId)
        await api.models.load(modelId)
        toast.success("Guardado; recargando el modelo con la nueva configuración")
      } else {
        toast.success(needsReload ? "Guardado. Se aplicará en la próxima carga." : "Guardado")
      }
      await refreshLogical()
    } catch (err) {
      toast.error(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(null)
    }
  }

  return (
    <div className="mx-auto max-w-[1600px] space-y-4">
      {/* Cabecera */}
      <header className="sticky top-0 z-20 -mx-2 space-y-3 border-b border-border bg-background/95 px-2 pb-3 pt-1 backdrop-blur">
        <div className="flex flex-wrap items-center gap-3">
          <button type="button" onClick={() => go("/models")} className="rounded-md p-1 text-muted-foreground hover:bg-muted" title="Volver a Modelos">
            <ArrowLeft size={18} />
          </button>
          <BackendIcon backendType={m.backend_type} />
          <div className="min-w-0">
            <h1 className="truncate text-lg font-semibold">{logical?.displayName ?? m.display_name}</h1>
            <p className="truncate font-mono text-[11px] text-muted-foreground">{m.model_id}</p>
          </div>
          <ModelStatusBadge status={status} />
          {m.current_gpu?.length ? <Chip>en {m.current_gpu.map((g) => `#${g}`).join(" + ")}</Chip> : null}
          {readOnly && (
            <Chip tone="warn">
              <Lock size={11} /> solo lectura
            </Chip>
          )}
          <div className="ml-auto flex flex-wrap items-center gap-2">
            {!readOnly && (
              <>
                {loaded || loading ? (
                  m.can_cancel_load ? (
                    <HeaderButton icon={XCircle} label="Cancelar carga" busy={busy === "cancel"} onClick={() => run("cancel", () => api.models.cancelLoad(modelId), "Carga cancelada")} />
                  ) : (
                    <HeaderButton icon={PowerOff} label="Descargar" busy={busy === "unload"} onClick={() => setConfirm("unload")} />
                  )
                ) : (
                  <HeaderButton icon={Power} label="Cargar" busy={busy === "load"} onClick={() => run("load", () => api.models.load(modelId), "Cargando el modelo")} />
                )}
                {m.can_downsize && (
                  <HeaderButton icon={Cpu} label="Pasar a 1 GPU" busy={busy === "downsize"} onClick={() => run("downsize", () => api.models.downsize(modelId), "Reduciendo a 1 GPU")} />
                )}
                {m.backend_type === "vllm" && <HeaderButton icon={Hammer} label="Compilar TRT" onClick={() => setCompileOpen(true)} />}
                <span className="mx-1 h-6 w-px bg-border" />
                <HeaderButton icon={Undo2} label="Descartar" disabled={!editor.dirty || Boolean(busy)} onClick={editor.discard} />
                {needsReload ? (
                  <>
                    <HeaderButton icon={Save} label="Guardar" disabled={!editor.dirty || Boolean(busy)} busy={busy === "save"} onClick={() => void save(false)} />
                    <HeaderButton icon={Save} label="Guardar y recargar" primary disabled={!editor.dirty || Boolean(busy)} busy={busy === "save-reload"} onClick={() => setConfirm("reload")} />
                  </>
                ) : (
                  <HeaderButton icon={Save} label={editor.saving ? "Guardando…" : "Guardar"} primary disabled={!editor.dirty || Boolean(busy)} busy={busy === "save"} onClick={() => void save(false)} />
                )}
              </>
            )}
          </div>
        </div>
        {logical && logical.deployments.length > 1 && (
          <div className="flex flex-wrap items-center gap-1.5" data-testid="deployment-tabs">
            <span className="text-[11px] text-muted-foreground">Despliegues:</span>
            {logical.deployments.map((dep) => (
              <button
                key={dep.modelId}
                type="button"
                onClick={() => go(`/models/edit/${encodeURIComponent(dep.modelId)}${location.search}`)}
                className={cn(
                  "inline-flex items-center gap-1.5 rounded-md border px-2 py-0.5 text-xs",
                  dep.modelId === m.model_id ? "border-primary bg-primary/10 text-foreground" : "border-border text-muted-foreground hover:bg-muted",
                )}
              >
                <BackendIcon backendType={dep.backend} showLabel={false} />
                {dep.backend}
                {logical.defaultBackend === dep.backend && <span className="text-amber-300">★</span>}
              </button>
            ))}
          </div>
        )}
      </header>

      <div className="grid gap-4 lg:grid-cols-[190px_minmax(0,1fr)_320px]">
        {/* Navegación */}
        <nav className="lg:sticky lg:top-28 lg:self-start" aria-label="Secciones">
          <ul className="flex gap-1 overflow-x-auto lg:flex-col">
            {visible.map((s) => {
              const Icon = s.icon
              const dirty = dirtyGroups.has(s.id)
              const bad = errorGroups.has(s.id)
              return (
                <li key={s.id}>
                  <button
                    type="button"
                    onClick={() => setSection(s.id)}
                    aria-current={active === s.id}
                    className={cn(
                      "flex w-full items-center gap-2 whitespace-nowrap rounded-lg px-3 py-1.5 text-sm",
                      active === s.id ? "bg-primary/15 font-medium text-foreground" : "text-muted-foreground hover:bg-muted hover:text-foreground",
                    )}
                  >
                    <Icon size={15} />
                    {s.label}
                    {bad ? <span className="ml-auto h-2 w-2 rounded-full bg-red-400" title="Hay opciones no válidas" /> : dirty ? <span className="ml-auto h-2 w-2 rounded-full bg-sky-400" title="Con cambios" /> : null}
                  </button>
                </li>
              )
            })}
          </ul>
        </nav>

        {/* Sección activa */}
        <main className="min-w-0 space-y-4" data-testid={`section-${active}`}>
          {active === "summary" && (
            <SummarySection
              {...props}
              modelState={modelState}
              onNavigateDeployment={(id) => go(`/models/edit/${encodeURIComponent(id)}`)}
              onGoTo={(s) => setSection(s as SectionId)}
            />
          )}
          {active === "load" && <LoadSection {...props} />}
          {active === "capacity" && <CapacitySection {...props} ladder={capacityLadder} />}
          {active === "response" && <ResponseSection {...props} />}
          {active === "acceleration" && <AccelerationSection {...props} />}
          {active === "audio" && <GroupSection {...props} group="audio" title="Audio" />}
          {active === "image" && <GroupSection {...props} group="image" title="Imagen" />}
          {active === "voice" && <GroupSection {...props} group="voice" title="Voz" />}
          {active === "runtime" && <GroupSection {...props} group="runtime" title="Arranque del worker" />}
          {active === "profiles" && <ProfilesSection {...props} />}
          {active === "access" && <AccessSection {...props} />}
          {active === "advanced" && <AdvancedSection {...props} />}
        </main>

        {/* Vista previa */}
        <div className="hidden lg:sticky lg:top-28 lg:block lg:max-h-[calc(100vh-8rem)] lg:self-start lg:overflow-y-auto">
          <PreviewPanel editor={editor} gpus={gpus} liveVramByGpu={liveVramByGpu} presets={rungs} />
        </div>
      </div>
      <MobilePreviewBar editor={editor} gpus={gpus} liveVramByGpu={liveVramByGpu} presets={rungs} />
      <div className="h-12 lg:hidden" aria-hidden>
      </div>

      <ConfirmDialog
        open={confirm === "unload"}
        onOpenChange={(o) => !o && setConfirm(null)}
        title="Descargar el modelo"
        description="Se cortan las peticiones en curso de este modelo."
        confirmLabel="Descargar"
        variant="danger"
        onConfirm={() => {
          setConfirm(null)
          void run("unload", () => api.models.unload(modelId), "Modelo descargado")
        }}
      />
      <ConfirmDialog
        open={confirm === "reload"}
        onOpenChange={(o) => !o && setConfirm(null)}
        title="Guardar y recargar"
        description="El modelo se descarga y vuelve a cargarse con la nueva configuración. Las peticiones en curso se cortan."
        confirmLabel="Guardar y recargar"
        onConfirm={() => {
          setConfirm(null)
          void save(true)
        }}
      />
      <ConfirmDialog
        open={pendingNav != null}
        onOpenChange={(o) => !o && setPendingNav(null)}
        title="Hay cambios sin guardar"
        description={`${editor.changes.length} cambio(s) se perderán si sales.`}
        confirmLabel="Salir sin guardar"
        variant="danger"
        onConfirm={() => {
          const to = pendingNav
          setPendingNav(null)
          if (to) navigate(to)
        }}
      />
      {modelState && <CompileModal model={modelState} open={compileOpen} onOpenChange={setCompileOpen} />}
    </div>
  )
}

function HeaderButton({
  icon: Icon,
  label,
  onClick,
  disabled,
  busy,
  primary,
}: {
  icon: typeof Info
  label: string
  onClick: () => void
  disabled?: boolean
  busy?: boolean
  primary?: boolean
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled || busy}
      className={cn(
        "inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40",
        primary ? "bg-primary text-primary-foreground hover:bg-primary/90" : "border border-border hover:bg-muted",
      )}
    >
      {busy ? <Loader2 size={14} className="animate-spin" /> : <Icon size={14} />}
      {label}
    </button>
  )
}
