import { useEffect, useState } from "react"
import { Eye, EyeOff, ImageIcon, Sparkles } from "lucide-react"
import { api } from "@/api/client"
import { modelEditorApi } from "@/api/modelEditor"
import { toast } from "sonner"
import { cn } from "@/lib/cn"
import { getPath, isObject } from "../draft"
import { fmtMb } from "../PreviewPanel"
import { FieldRow, NumberInput, SchemaFields } from "../SchemaField"
import type { Json, JsonObject } from "../types"
import { Callout, Hint, Panel, Segmented, Toggle, inputCls } from "../ui"
import type { SectionProps } from "./types"

// ── Respuesta ───────────────────────────────────────────────────────────────

export function ResponseSection({ editor, readOnly, insights }: SectionProps) {
  const tpl = insights?.template ?? null
  const reasoningCapable = Boolean(editor.model?.capabilities?.reasoning)
  const effort = editor.fieldByKey.get("default_reasoning_effort")
  const reasoningKeys = ["llama_cpp.reasoning", "llama_cpp.reasoning_format", "llama_cpp.reasoning_budget"]
  const hasLlamaReasoning = reasoningKeys.some((k) => editor.fieldByKey.has(k))
  return (
    <div className="space-y-4">
      {effort && (
        <Panel
          title="Razonamiento"
          subtitle={
            reasoningCapable
              ? "Cuánto piensa el modelo cuando el cliente no lo pide."
              : "El modelo no declara capacidad de razonamiento: estos ajustes probablemente no tengan efecto."
          }
        >
          <FieldRow field={effort} editor={editor} readOnly={readOnly} />
          {hasLlamaReasoning && (
            <div className="mt-2 grid gap-x-6 md:grid-cols-3">
              <SchemaFields editor={editor} only={reasoningKeys} readOnly={readOnly} showAdvanced />
            </div>
          )}
          {tpl ? (
            tpl.effortLevels ? (
              <Callout tone="ok">La plantilla de chat del modelo lee el nivel de esfuerzo: los niveles cambian cuánto piensa.</Callout>
            ) : tpl.thinkingSwitch ? (
              <Callout tone="warn">
                La plantilla solo distingue pensar o no pensar: «Apagado» lo desactiva, pero bajo, medio y alto dan el mismo resultado.
              </Callout>
            ) : (
              <Callout tone="warn">La plantilla de chat de este modelo no lee el esfuerzo de razonamiento: este ajuste no tiene efecto.</Callout>
            )
          ) : (
            <Hint>
              Solo surte efecto si la plantilla de chat del modelo lee el esfuerzo (no todas lo hacen); los valores por petición del
              cliente siempre ganan.
            </Hint>
          )}
        </Panel>
      )}
      <SchemaFields
        title="Plantilla y formato"
        editor={editor}
        group="response"
        exclude={["default_reasoning_effort", ...reasoningKeys]}
        readOnly={readOnly}
        showAdvanced
      />
      <Callout tone="info">
        Los valores por defecto de las peticiones (temperatura, top_p, prompt de sistema, voz…) son de cada perfil: edítalos en
        Perfiles.
      </Callout>
    </div>
  )
}

// ── Aceleración ─────────────────────────────────────────────────────────────

const SPEC_TYPES: { value: string; label: string; hint: string; draft: boolean }[] = [
  { value: "draft-mtp", label: "MTP propio", hint: "El propio modelo predice varios tokens (si su GGUF trae cabezas MTP).", draft: false },
  { value: "draft-simple", label: "Modelo draft", hint: "Un modelo pequeño con el mismo tokenizer propone tokens.", draft: true },
  { value: "draft-eagle3", label: "EAGLE-3", hint: "Draft EAGLE-3 entrenado para este modelo.", draft: true },
  { value: "draft-dflash", label: "DFlash", hint: "Draft DFlash.", draft: true },
  { value: "ngram-simple", label: "N-gramas", hint: "Sin modelo extra: repite secuencias ya vistas en el contexto.", draft: false },
]

function SpeculativePanel({ editor, readOnly }: SectionProps) {
  const field = editor.fieldByKey.get("llama_cpp.speculative")
  const [candidates, setCandidates] = useState<{ modelId: string; displayName: string }[] | null>(null)
  const modelId = editor.model?.model_id
  useEffect(() => {
    if (!modelId) return
    api.models
      .speculativeCandidates(modelId)
      .then((rows) => setCandidates(rows))
      .catch(() => setCandidates([]))
  }, [modelId])
  if (!field) return null
  const spec = (getPath(editor.draft!.extraConfig, field.path) ?? undefined) as JsonObject | undefined
  const type = (spec?.spec_type as string | undefined) ?? (spec?.draft_model_id ? "draft-simple" : undefined)
  const meta = SPEC_TYPES.find((t) => t.value === type)
  const setSpec = (patch: Record<string, Json | undefined>) => {
    const next: JsonObject = { ...(spec ?? {}) }
    for (const [k, v] of Object.entries(patch)) {
      if (v === undefined) delete next[k]
      else next[k] = v
    }
    editor.setField(field.path, Object.keys(next).length ? next : undefined)
  }
  return (
    <Panel
      title={
        <span className="flex items-center gap-1.5">
          <Sparkles size={14} className="text-amber-300" /> Decodificación especulativa
        </span>
      }
      subtitle="Propone varios tokens de golpe y el modelo los valida: más t/s al generar, a cambio de algo de VRAM."
    >
      <div className="space-y-3">
        <Segmented
          size="sm"
          disabled={readOnly}
          value={type ?? "off"}
          options={[{ value: "off", label: "Apagada" }, ...SPEC_TYPES.map((t) => ({ value: t.value, label: t.label, title: t.hint }))]}
          onChange={(v) => (v === "off" ? editor.setField(field.path, undefined) : setSpec({ spec_type: v, ...(SPEC_TYPES.find((t) => t.value === v)?.draft ? {} : { draft_model_id: undefined }) }))}
          testId="spec-type"
        />
        {meta && <Hint>{meta.hint}</Hint>}
        {meta?.draft && (
          <div>
            <p className="mb-1 text-xs font-medium">Modelo draft</p>
            {candidates === null ? (
              <Hint>Buscando modelos compatibles…</Hint>
            ) : candidates.length === 0 ? (
              <Callout tone="warn">No hay ningún modelo con el mismo tokenizer. Indexa un GGUF pequeño de la misma familia para usarlo de draft.</Callout>
            ) : (
              <select
                className={inputCls}
                disabled={readOnly}
                value={(spec?.draft_model_id as string | undefined) ?? ""}
                onChange={(e) => setSpec({ draft_model_id: e.target.value || undefined })}
              >
                <option value="">Elige un draft…</option>
                {candidates.map((c) => (
                  <option key={c.modelId} value={c.modelId}>
                    {c.displayName || c.modelId}
                  </option>
                ))}
              </select>
            )}
          </div>
        )}
        {type && (
          <div className="grid gap-3 sm:grid-cols-3">
            {(
              [
                ["draft_n", "Tokens propuestos (máx.)", "int"],
                ["draft_min", "Tokens propuestos (mín.)", "int"],
                ["draft_p_min", "Probabilidad mínima", "float"],
              ] as const
            ).map(([key, label, t]) => (
              <div key={key}>
                <p className="mb-1 text-xs font-medium">{label}</p>
                <NumberInput
                  field={{ key, type: t, label, min: 0, max: t === "float" ? 1 : 64, step: t === "float" ? 0.05 : 1, default: null }}
                  value={spec?.[key]}
                  readOnly={readOnly}
                  onChange={(v) => setSpec({ [key]: v })}
                />
              </div>
            ))}
          </div>
        )}
      </div>
    </Panel>
  )
}

const VISION_OPTIONS = [
  { value: "enabled", label: "Siempre cargada", icon: Eye, hint: "El proyector se carga con el modelo: las imágenes no esperan, pero ocupa VRAM siempre." },
  { value: "on_demand", label: "Bajo demanda", icon: ImageIcon, hint: "Se carga con la primera imagen y se queda hasta descargar el modelo." },
  { value: "disabled", label: "Apagada", icon: EyeOff, hint: "Sin visión: el modelo rechaza imágenes y ahorra la VRAM del proyector." },
]

function VisionPanel({ editor, readOnly }: SectionProps) {
  const field = editor.fieldByKey.get("llama_cpp.vision_mode")
  if (!field) return null
  const extra = editor.draft!.extraConfig
  const mmproj = getPath(extra, ["llama_cpp", "mmproj_model_id"]) as string | undefined
  const raw = getPath(extra, field.path) as string | undefined
  const mode = raw ?? (mmproj ? "enabled" : "disabled")
  return (
    <Panel title="Visión" subtitle={mmproj ? `Proyector: ${mmproj.split("/").pop()}` : "Este modelo no tiene proyector de visión configurado."}>
      <div className="grid gap-2 sm:grid-cols-3" data-testid="vision-mode">
        {VISION_OPTIONS.map((o) => {
          const Icon = o.icon
          const active = mode === o.value
          const blocked = !mmproj && o.value !== "disabled"
          return (
            <button
              key={o.value}
              type="button"
              disabled={readOnly || blocked}
              onClick={() => editor.setField(field.path, o.value)}
              className={cn(
                "rounded-xl border p-3 text-left disabled:cursor-not-allowed disabled:opacity-40",
                active ? "border-primary bg-primary/10" : "border-border hover:bg-muted/30",
              )}
            >
              <div className="flex items-center gap-1.5 text-sm font-medium">
                <Icon size={14} /> {o.label}
              </div>
              <p className="mt-1 text-[11px] text-muted-foreground">{o.hint}</p>
            </button>
          )
        })}
      </div>
      {editor.model?.vision_state && <Hint>Ahora mismo: proyector {editor.model.vision_state === "active" ? "cargado" : "sin cargar"}.</Hint>}
    </Panel>
  )
}

const OFFLOAD = [
  { value: "none", label: "Todo en GPU", speed: 3, vram: 3 },
  { value: "model", label: "Por modelo", speed: 2, vram: 2 },
  { value: "sequential", label: "Secuencial", speed: 1, vram: 1 },
]

function OffloadPanel({ editor, readOnly }: SectionProps) {
  const field = editor.fieldByKey.get("diffusers.offload_mode")
  if (!field) return null
  const raw = getPath(editor.draft!.extraConfig, field.path) as string | undefined
  const value = raw ?? (field.default as string)
  const Bars = ({ n, tone }: { n: number; tone: string }) => (
    <span className="inline-flex gap-0.5">
      {[1, 2, 3].map((i) => (
        <span key={i} className={cn("h-2 w-3 rounded-sm", i <= n ? tone : "bg-muted")} />
      ))}
    </span>
  )
  return (
    <Panel title="Descarga a CPU" subtitle="Cuánto del pipeline vive en la GPU entre pasos.">
      <div className="grid gap-2 sm:grid-cols-3">
        {OFFLOAD.map((o) => (
          <button
            key={o.value}
            type="button"
            disabled={readOnly}
            onClick={() => editor.setField(field.path, o.value)}
            className={cn("rounded-xl border p-3 text-left", value === o.value ? "border-primary bg-primary/10" : "border-border hover:bg-muted/30")}
          >
            <div className="text-sm font-medium">
              {o.label}
              {raw === undefined && value === o.value && <span className="ml-1 text-[10px] text-muted-foreground">(por defecto)</span>}
            </div>
            <div className="mt-2 grid grid-cols-[auto_1fr] items-center gap-x-2 gap-y-1 text-[11px] text-muted-foreground">
              velocidad <Bars n={o.speed} tone="bg-emerald-400" />
              VRAM <Bars n={o.vram} tone="bg-amber-400" />
            </div>
          </button>
        ))}
      </div>
    </Panel>
  )
}

export function AccelerationSection(props: SectionProps) {
  const { editor, readOnly } = props
  return (
    <div className="space-y-4">
      <SpeculativePanel {...props} />
      <VisionPanel {...props} />
      <SchemaFields
        title="Rendimiento"
        editor={editor}
        group="acceleration"
        exclude={["llama_cpp.speculative", "llama_cpp.vision_mode"]}
        readOnly={readOnly}
        showAdvanced
      />
    </div>
  )
}

// ── Grupos sin control a medida (audio, voz, arranque…) ─────────────────────

export function GroupSection(props: SectionProps & { group: string; title: string }) {
  const { editor, readOnly, group, title } = props
  return (
    <div className="space-y-4">
      {group === "image" && <OffloadPanel {...props} />}
      <SchemaFields title={title} editor={editor} group={group} exclude={["diffusers.offload_mode"]} readOnly={readOnly} showAdvanced />
    </div>
  )
}

// ── Avanzado ────────────────────────────────────────────────────────────────

export function AdvancedSection({ editor, readOnly, insights, reloadInsights }: SectionProps) {
  const [maintBusy, setMaintBusy] = useState(false)
  const [raw, setRaw] = useState("")
  const [bad, setBad] = useState<string | null>(null)
  const extra = editor.draft?.extraConfig ?? {}
  useEffect(() => {
    setRaw(JSON.stringify(extra, null, 2))
    setBad(null)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [JSON.stringify(extra)])
  const unknown = editor.preview?.unknownKeys ?? editor.schema?.unknownKeys ?? []
  return (
    <div className="space-y-4">
      <Panel
        title="Mantenimiento"
        subtitle="Mientras está activo, las peticiones a este modelo reciben 503 con Retry-After. No se guarda: se pierde al reiniciar el servidor."
      >
        <Toggle
          checked={Boolean(insights?.maintenance)}
          disabled={readOnly || maintBusy || !editor.model}
          testId="maintenance"
          label={<span className="text-sm">{insights?.maintenance ? "En mantenimiento" : "Atendiendo peticiones"}</span>}
          onChange={async (on) => {
            if (!editor.model) return
            setMaintBusy(true)
            try {
              await modelEditorApi.setMaintenance(editor.model.model_id, on)
              toast.success(on ? "Modelo en mantenimiento" : "Mantenimiento desactivado")
              await reloadInsights?.()
            } catch (err) {
              toast.error(err instanceof Error ? err.message : String(err))
            } finally {
              setMaintBusy(false)
            }
          }}
        />
      </Panel>
      <SchemaFields title="Opciones avanzadas" editor={editor} group="advanced" readOnly={readOnly} showAdvanced />
      <Panel
        title="Configuración en crudo"
        subtitle="extra_config completo del despliegue. Lo que cambies aquí se suma al borrador y aparece en Cambios pendientes."
      >
        {unknown.length > 0 && (
          <Callout tone="warn">
            Claves que ningún backend lee (¿errata o restos?): <span className="font-mono">{unknown.join(", ")}</span>
          </Callout>
        )}
        <textarea
          className={cn(inputCls, "mt-2 min-h-[320px] font-mono text-[11.5px]", bad && "border-red-500")}
          disabled={readOnly}
          value={raw}
          spellCheck={false}
          onChange={(e) => {
            setRaw(e.target.value)
            try {
              const parsed = JSON.parse(e.target.value) as Json
              if (!isObject(parsed)) throw new Error("Tiene que ser un objeto JSON")
              setBad(null)
              editor.update({ extraConfig: parsed })
            } catch (err) {
              setBad(err instanceof Error ? err.message : "JSON no válido")
            }
          }}
          data-testid="raw-config"
        />
        {bad && <Hint tone="error">{bad}: no se aplica hasta que se corrija.</Hint>}
        <Hint>Las rutas del modelo y del proyector las gestiona el artefacto; las claves que empiezan por _ las escribe el sistema.</Hint>
      </Panel>
      {editor.model && (
        <Panel title="Memoria medida">
          <p className="text-xs text-muted-foreground">
            Reservada {fmtMb(editor.model.vram_estimate_mb ?? 0)} · en uso {fmtMb(editor.model.vram_used_mb)}
            {editor.model.ram_used_mb ? ` · RAM ${fmtMb(editor.model.ram_used_mb)}` : ""}
          </p>
        </Panel>
      )}
    </div>
  )
}
