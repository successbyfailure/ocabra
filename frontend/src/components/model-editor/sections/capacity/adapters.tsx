import { useEffect, useMemo, useState } from "react"
import { FlaskConical, Star, Trash2 } from "lucide-react"
import { toast } from "sonner"
import { api } from "@/api/client"
import { ConfirmDialog } from "@/components/common/ConfirmDialog"
import { ContextProbe } from "@/components/models/ContextProbe"
import { cn } from "@/lib/cn"
import type { GPUState, ModelProfile, RuntimePreset, RuntimePresetCreate, RuntimePresetResolution } from "@/types"
import { getPath } from "../../draft"
import type { Json } from "../../types"
import type { ModelEditorState } from "../../useModelEditor"
import { Callout, Hint, Segmented, Toggle, inputCls } from "../../ui"
import { GpuLayersPanel } from "../GpuSection"
import { AliasList } from "../presets/PresetParts"
import { EMPTY_PRESET, PRESET_KEY_RE, presetToForm } from "../presets/presetLogic"
import type { SectionProps } from "../types"
import { useCapacity } from "../useCapacity"
import { Workbench, type KvLabel, type MemoryShare, type WorkbenchState } from "./Workbench"

const errMsg = (e: unknown, fallback: string) => (e instanceof Error ? e.message : fallback)

function kvFromCacheType(ct: unknown): KvLabel {
  const v = String(ct ?? "").toLowerCase()
  if (!v || v === "f16") return "fp16"
  if (v.startsWith("q4")) return "q4"
  return "q8"
}
const CACHE_TYPE: Record<KvLabel, string | undefined> = { fp16: undefined, bf16: undefined, fp8: "q8_0", q8: "q8_0", q4: "q4_0" }

// ── Base de llama.cpp (modelos sin presets) ─────────────────────────────────

function readBase(editor: ModelEditorState, gpuIndices: number[]): WorkbenchState {
  const d = editor.draft!
  const sec = (getPath(d.extraConfig, ["llama_cpp"]) ?? {}) as Record<string, Json>
  const def = (k: string, fb: number) => {
    const v = sec[k]
    if (typeof v === "number") return v
    const f = editor.fieldByKey.get(`llama_cpp.${k}`)?.default
    return typeof f === "number" ? f : fb
  }
  const slots = Math.max(1, def("parallel_slots", 1))
  const pool = sec.kv_unified === true
  const total = def("ctx_size", 4096)
  const disabled = new Set((Array.isArray(sec.disabled_gpus) ? sec.disabled_gpus : []) as number[])
  const visible = gpuIndices.filter((g) => !disabled.has(g))
  let gpus: number[] | null = null
  let main: number | null = null
  if (sec.split_mode === "none") {
    const pref = d.preferredGpu != null && visible.includes(d.preferredGpu) ? d.preferredGpu : (visible[0] ?? d.preferredGpu)
    gpus = pref != null ? [pref] : null
    main = pref ?? null
  } else if (disabled.size || d.preferredGpu != null) {
    gpus = visible.length ? visible : null
    const mainPos = typeof sec.main_gpu === "number" ? sec.main_gpu : 0
    main = visible[mainPos] ?? visible[0] ?? null
  }
  return {
    context: pool ? total : Math.max(1, Math.floor(total / slots)),
    slots,
    pool,
    kv: kvFromCacheType(sec.cache_type_k),
    gpus,
    main,
    ratios: Array.isArray(sec.tensor_split) ? (sec.tensor_split as number[]) : null,
    splitMode: typeof sec.split_mode === "string" && sec.split_mode !== "none" ? sec.split_mode : undefined,
  }
}

function writeBase(editor: ModelEditorState, gpuIndices: number[], next: WorkbenchState) {
  const set = (k: string, v: Json | undefined) => editor.setField(["llama_cpp", k], v)
  set("ctx_size", next.pool ? next.context : next.context * next.slots)
  set("parallel_slots", next.slots)
  set("kv_unified", next.pool ? true : undefined)
  const ct = CACHE_TYPE[next.kv]
  set("cache_type_k", ct)
  set("cache_type_v", ct)
  if (ct) set("flash_attn", true)
  if (next.gpus == null) {
    editor.update({ preferredGpu: null })
    set("disabled_gpus", undefined)
    set("split_mode", undefined)
    set("main_gpu", undefined)
    set("tensor_split", undefined)
  } else if (next.gpus.length === 1) {
    editor.update({ preferredGpu: next.gpus[0] })
    const others = gpuIndices.filter((g) => g !== next.gpus![0])
    set("disabled_gpus", others.length ? others : undefined)
    set("split_mode", "none")
    set("main_gpu", undefined)
    set("tensor_split", undefined)
  } else {
    const others = gpuIndices.filter((g) => !next.gpus!.includes(g))
    const main = next.main ?? next.gpus[0]
    editor.update({ preferredGpu: main })
    set("disabled_gpus", others.length ? others : undefined)
    set("split_mode", next.splitMode)
    const pos = next.gpus.indexOf(main)
    set("main_gpu", pos > 0 ? pos : undefined)
    set("tensor_split", next.ratios && next.ratios.length === next.gpus.length ? next.ratios : undefined)
  }
}

/** Capacidad de un modelo llama.cpp sin presets: su base es lo que carga. */
export function BaseLlamaWorkbench(props: SectionProps & { onCreatePreset?: () => void }) {
  const { editor, readOnly, gpus, liveVramByGpu, onCreatePreset } = props
  const gpuIndices = gpus.map((g) => g.index)
  const value = readBase(editor, gpuIndices)
  const { data: cap } = useCapacity(editor.model?.model_id, { gpu: value.main ?? value.gpus?.[0] ?? null, slots: [1], kvDtype: value.kv === "q4" ? "q4" : value.kv === "fp16" ? "fp16" : "q8" })
  const memory: MemoryShare[] = (editor.preview?.gpus ?? []).map((g) => ({
    gpu: g.index,
    parts: Object.keys(g.breakdown).length ? g.breakdown : { model: g.mb },
  }))
  return (
    <Workbench
      title="Memoria y contexto"
      subtitle="Cambia un control y mira cómo se mueve la memoria a la derecha. Se guarda con Guardar."
      value={value}
      onChange={(patch) => writeBase(editor, gpuIndices, { ...value, ...patch })}
      readOnly={readOnly}
      gpus={gpus}
      liveVramByGpu={liveVramByGpu}
      memory={memory}
      cap={cap}
      native={cap?.arch?.native_context ?? (Number(editor.model?.capabilities?.native_context_length) || null)}
      allowRatios
      allowMax={false}
      poolEditable
      extraKnobs={<GpuLayersPanel {...props} />}
      footer={
        onCreatePreset && (
          <div className="flex flex-wrap items-center justify-between gap-3" data-testid="probe-needs-preset">
            <div className="text-xs">
              <p className="font-medium">Medir el contexto máximo real</p>
              <p className="text-muted-foreground">
                El sondeo carga el modelo y amplía el contexto hasta el límite, pero mide sobre un peldaño. Este modelo no tiene ninguno:
                crea uno en modo «lo que quepa» y mídelo desde él.
              </p>
            </div>
            {!readOnly && (
              <button type="button" onClick={onCreatePreset} className="inline-flex items-center gap-1 rounded-md border border-violet-500/50 px-3 py-1.5 text-xs hover:bg-violet-500/10">
                <FlaskConical size={13} /> Crear peldaño y medir
              </button>
            )}
          </div>
        )
      }
    />
  )
}

// ── Un preset ───────────────────────────────────────────────────────────────

function formToState(form: RuntimePresetCreate, pool: boolean, slotsFallback: number): WorkbenchState {
  const gpus = form.placementPolicy === "auto" ? null : (form.preferredGpus ?? null)
  return {
    context: form.contextTokens ?? 65536,
    maxMode: form.contextMode === "max",
    slots: form.parallelism || slotsFallback,
    pool,
    kv: (form.kvCacheDtype as KvLabel) ?? "q8",
    gpus,
    main: gpus?.[0] ?? null,
    ratios: null,
  }
}

/** Un peldaño: su contexto, peticiones, KV y GPUs con la memoria al lado,
 *  sus acciones (por defecto, medir, alias) y su guardado (al momento). */
export function PresetWorkbench({
  editor,
  readOnly,
  gpus,
  liveVramByGpu,
  profile,
  presetKey,
  resolutions,
  onChanged,
  onDone,
}: {
  editor: ModelEditorState
  readOnly: boolean
  gpus: GPUState[]
  liveVramByGpu: Record<string, number>
  profile: ModelProfile
  /** null = nuevo peldaño. */
  presetKey: string | null
  resolutions: Record<string, RuntimePresetResolution>
  onChanged: () => Promise<void>
  onDone: (key: string | null) => void
}) {
  const preset = presetKey ? profile.runtimePresets.find((p) => p.presetKey === presetKey) ?? null : null
  const initial = useMemo<RuntimePresetCreate>(
    () => (preset ? presetToForm(preset) : { ...EMPTY_PRESET, isDefault: profile.runtimePresets.length === 0 }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [presetKey, preset?.updatedAt],
  )
  const [form, setForm] = useState<RuntimePresetCreate>(initial)
  useEffect(() => setForm(initial), [initial])
  const [preview, setPreview] = useState<RuntimePresetResolution | null>(null)
  const [saving, setSaving] = useState(false)
  const [busy, setBusy] = useState<string | null>(null)
  const [confirm, setConfirm] = useState<"bench" | "delete" | null>(null)
  const pool = (getPath(editor.draft?.extraConfig, ["llama_cpp", "kv_unified"]) ?? getPath(editor.base?.extraConfig, ["llama_cpp", "kv_unified"])) === true
  const dirty = JSON.stringify(form) !== JSON.stringify(initial)
  const isNew = preset == null
  const keyValid = PRESET_KEY_RE.test(form.presetKey) && !(isNew && profile.runtimePresets.some((p) => p.presetKey === form.presetKey))
  const value = formToState(form, pool, 2)
  const { data: cap } = useCapacity(editor.model?.model_id, { gpu: value.gpus?.[0] ?? null, slots: [1], kvDtype: value.kv === "q4" ? "q4" : value.kv === "fp16" || value.kv === "bf16" ? "fp16" : "q8" })

  useEffect(() => {
    if (form.contextMode === "fixed" && !form.contextTokens) return
    let cancelled = false
    const t = window.setTimeout(() => {
      api.profiles
        .previewRuntimePreset(profile.profileId, form)
        .then((r) => !cancelled && setPreview(r))
        .catch(() => !cancelled && setPreview(null))
    }, 350)
    return () => {
      cancelled = true
      window.clearTimeout(t)
    }
  }, [form, profile.profileId])

  const shown = preview ?? (preset ? resolutions[preset.presetKey] : null) ?? null
  const breakdown = shown?.vramBreakdownByGpu ?? {}
  const memory: MemoryShare[] = Object.keys(breakdown).length
    ? Object.entries(breakdown).map(([g, parts]) => ({ gpu: Number(g), parts }))
    : Object.entries(shown?.vramByGpu ?? {}).map(([g, mb]) => ({ gpu: Number(g), parts: { model: mb } }))
  const measured = (profile.runtimePolicy?.validatedContextTokens ?? {})[presetKey ?? ""] ?? null

  const change = (patch: Partial<WorkbenchState>) => {
    const next = { ...value, ...patch }
    setForm((f) => ({
      ...f,
      contextMode: next.maxMode ? "max" : "fixed",
      contextTokens: next.maxMode ? null : next.context,
      parallelism: next.slots,
      kvCacheDtype: next.kv,
      placementPolicy: next.gpus ? "forced" : "auto",
      preferredGpus: next.gpus,
    }))
  }

  const run = async (label: string, fn: () => Promise<unknown>, ok: string) => {
    setBusy(label)
    try {
      await fn()
      toast.success(ok)
      await onChanged()
    } catch (e) {
      toast.error(errMsg(e, "No se pudo completar"))
    } finally {
      setBusy(null)
    }
  }

  const save = async () => {
    setSaving(true)
    try {
      const payload: RuntimePresetCreate = { ...form, contextTokens: form.contextMode === "max" ? null : form.contextTokens }
      if (isNew) await api.profiles.createRuntimePreset(profile.profileId, payload)
      else {
        const { presetKey: _k, ...update } = payload
        void _k
        await api.profiles.updateRuntimePreset(profile.profileId, preset!.presetKey, update)
      }
      toast.success("Peldaño guardado; se aplica en la siguiente carga")
      await onChanged()
      if (isNew) onDone(form.presetKey)
    } catch (e) {
      toast.error(errMsg(e, "No se pudo guardar el peldaño"))
    } finally {
      setSaving(false)
    }
  }

  const header = (
    <div className="mb-4 space-y-3">
      <div className="flex flex-wrap items-end gap-3">
        {isNew && (
          <label className="space-y-1 text-xs">
            <span className="font-medium">Clave</span>
            <input className={cn(inputCls, "w-36 font-mono", !keyValid && "border-red-500")} value={form.presetKey} disabled={readOnly} onChange={(e) => setForm({ ...form, presetKey: e.target.value })} placeholder="128k" />
          </label>
        )}
        <label className="space-y-1 text-xs">
          <span className="font-medium">Nombre</span>
          <input className={cn(inputCls, "w-44")} value={form.label} disabled={readOnly} onChange={(e) => setForm({ ...form, label: e.target.value })} />
        </label>
        <label className="space-y-1 text-xs">
          <span className="font-medium" title="Desempate entre peldaños que sirven igual">Prioridad</span>
          <input type="number" className={cn(inputCls, "w-20")} value={form.priority ?? 100} disabled={readOnly} onChange={(e) => setForm({ ...form, priority: Number(e.target.value) || 0 })} />
        </label>
        <div className="space-y-1 text-xs">
          <span className="font-medium">Ejecución</span>
          <Segmented
            size="sm"
            disabled={readOnly}
            value={form.executionPolicy}
            options={[
              { value: "auto", label: "Auto" },
              { value: "local", label: "Solo local" },
              { value: "remote", label: "Remota" },
            ]}
            onChange={(v) => setForm({ ...form, executionPolicy: v })}
          />
        </div>
        {!isNew && preset && (
          <div className="ml-auto flex flex-wrap items-center gap-2 text-xs">
            {preset.isDefault ? (
              <span className="inline-flex items-center gap-1 text-amber-300">
                <Star size={12} fill="currentColor" /> por defecto
              </span>
            ) : (
              !readOnly && (
                <button type="button" disabled={busy != null} className="inline-flex items-center gap-1 rounded border border-border px-2 py-1 hover:bg-muted" onClick={() => void run("default", () => api.profiles.updateRuntimePreset(profile.profileId, preset.presetKey, { isDefault: true }), "Ahora es el peldaño por defecto")}>
                  <Star size={12} /> Hacer por defecto
                </button>
              )
            )}
            {!readOnly && !preset.isDefault && (
              <Toggle
                checked={preset.enabled}
                onChange={(v) => void run("enable", () => api.profiles.updateRuntimePreset(profile.profileId, preset.presetKey, { enabled: v }), v ? "Peldaño activado" : "Peldaño desactivado")}
                label={<span>Activo</span>}
              />
            )}
            {!readOnly && (
              <button type="button" disabled={busy != null} className="rounded border border-border px-2 py-1 hover:bg-muted" onClick={() => setConfirm("bench")}>
                {busy === "bench" ? "Midiendo…" : "Medir en frío"}
              </button>
            )}
            {!readOnly && (
              <button
                type="button"
                disabled={busy != null || profile.runtimePresets.length <= 1 || preset.isDefault}
                title={profile.runtimePresets.length <= 1 ? "Un LLM necesita al menos un peldaño" : preset.isDefault ? "Haz otro por defecto antes" : "Borrar peldaño"}
                className="rounded border border-red-500/40 p-1 text-red-300 hover:bg-red-500/10 disabled:opacity-40"
                onClick={() => setConfirm("delete")}
              >
                <Trash2 size={13} />
              </button>
            )}
          </div>
        )}
      </div>
      {shown && !shown.fits && <Callout tone="error">No cabe: {shown.warnings.join(" · ") || "el planificador no encuentra sitio"}.</Callout>}
      {shown?.fits && shown.warnings.length > 0 && <Callout tone="warn">{shown.warnings.join(" · ")}</Callout>}
    </div>
  )

  const footer = (
    <div className="grid gap-4 lg:grid-cols-2">
      <div className="space-y-2">
        <p className="text-xs font-medium">¿Cuánto cabe de verdad?</p>
        <Hint>
          La gráfica es una estimación. El sondeo carga el modelo con este peldaño y amplía el contexto en caliente hasta el límite; lo medido
          queda como valor validado y aparece en la gráfica.
        </Hint>
        {preset && !isNew ? (
          <ContextProbe profileId={profile.profileId} preset={preset} onApplied={() => void onChanged()} />
        ) : (
          <Hint>Guarda el peldaño para poder medirlo.</Hint>
        )}
      </div>
      {preset && !isNew && (
        <div className="space-y-2">
          <p className="text-xs font-medium">Alias</p>
          <AliasList profile={profile} preset={preset} readOnly={readOnly} onChanged={onChanged} inset={false} />
        </div>
      )}
    </div>
  )

  return (
    <>
      <Workbench
        title={isNew ? "Nuevo peldaño" : `Peldaño «${preset!.label}»`}
        subtitle="Los peldaños se guardan al momento con su propio botón; se aplican en la siguiente carga."
        header={header}
        value={value}
        onChange={change}
        readOnly={readOnly}
        gpus={gpus}
        liveVramByGpu={liveVramByGpu}
        memory={memory.length ? memory : null}
        memoryNote={shown?.effectiveContext ? <Hint>Contexto efectivo: {Math.round(shown.effectiveContext / 1024)}k · {shown.effectiveParallelism} a la vez.</Hint> : null}
        cap={cap}
        native={cap?.arch?.native_context ?? null}
        allowRatios={false}
        allowMax
        poolEditable={false}
        measured={measured}
        footer={footer}
      />
      {!readOnly && (
        <div className="sticky bottom-0 z-10 -mt-2 flex items-center justify-end gap-2 rounded-b-2xl border-x border-b border-border bg-card/95 px-4 py-2 backdrop-blur">
          {dirty && <span className="mr-auto text-xs text-sky-300">Cambios sin guardar en este peldaño</span>}
          {isNew && (
            <button type="button" className="rounded-md border border-border px-3 py-1.5 text-sm hover:bg-muted" onClick={() => onDone(null)}>
              Cancelar
            </button>
          )}
          {dirty && !isNew && (
            <button type="button" className="rounded-md border border-border px-3 py-1.5 text-sm hover:bg-muted" onClick={() => setForm(initial)}>
              Descartar
            </button>
          )}
          <button
            type="button"
            disabled={saving || !dirty || !form.label || !keyValid}
            onClick={() => void save()}
            className="rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground disabled:opacity-40"
          >
            {saving ? "Guardando…" : isNew ? "Crear peldaño" : "Guardar peldaño"}
          </button>
        </div>
      )}
      <ConfirmDialog
        open={confirm != null}
        onOpenChange={(o) => !o && setConfirm(null)}
        title={confirm === "bench" ? "Medir en frío" : "Borrar peldaño"}
        description={
          confirm === "bench"
            ? "Descarga el modelo si está cargado y mide la carga en frío y una generación con este peldaño. Las peticiones en curso pueden fallar."
            : "Se borra el peldaño y sus alias dejarán de resolver."
        }
        confirmLabel={confirm === "bench" ? "Medir" : "Borrar"}
        variant={confirm === "bench" ? "default" : "danger"}
        onConfirm={() => {
          const c = confirm
          setConfirm(null)
          if (!preset) return
          if (c === "bench")
            void run(
              "bench",
              async () => {
                const r = await api.profiles.benchmarkRuntimePreset(profile.profileId, preset.presetKey, true)
                toast.info(`Carga ${Math.round(r.loadMs / 1000)} s · ${r.tokensPerSecond?.toFixed(1) ?? "?"} t/s`)
              },
              "Medición completada",
            )
          else void run("delete", () => api.profiles.deleteRuntimePreset(profile.profileId, preset.presetKey), "Peldaño borrado").then(() => onDone(null))
        }}
      />
    </>
  )
}

export type { RuntimePreset }
