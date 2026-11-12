import { useEffect, useMemo, useState } from "react"
import { FlaskConical, Play, Star, Trash2 } from "lucide-react"
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
import { modelEditorApi } from "@/api/modelEditor"
import { AliasList } from "../presets/PresetParts"
import { EMPTY_PRESET, PRESET_KEY_RE, presetToForm } from "../presets/presetLogic"
import type { SectionProps } from "../types"
import { useCapacity } from "../useCapacity"
import { DevicePlacement, type Placement } from "./DevicePlacement"
import {
  activeGpus,
  readBasePlacement,
  readBitnetPlacement,
  readPresetPlacement,
  writeBasePlacement,
  writeBitnetPlacement,
  writePresetPlacement,
} from "./placementMapping"
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

/** bitnet comparte el banco de trabajo pero lee su propia sección y llama
 *  `parallel` a los slots. */
const isBitnet = (editor: ModelEditorState) => editor.model?.backend_type === "bitnet"
const secName = (editor: ModelEditorState) => (isBitnet(editor) ? "bitnet" : "llama_cpp")
const slotsKey = (editor: ModelEditorState) => (isBitnet(editor) ? "parallel" : "parallel_slots")

function baseSection(editor: ModelEditorState) {
  return (getPath(editor.draft!.extraConfig, [secName(editor)]) ?? {}) as Record<string, Json>
}

function readBasePlacementOf(editor: ModelEditorState, gpuIndices: number[], layers: number | null): Placement {
  const f = editor.fieldByKey.get(`${secName(editor)}.gpu_layers`)?.default
  const read = isBitnet(editor) ? readBitnetPlacement : readBasePlacement
  return read(baseSection(editor), editor.draft!.preferredGpu ?? null, gpuIndices, layers, typeof f === "number" ? f : null)
}

function readBase(editor: ModelEditorState, placement: Placement): WorkbenchState {
  const sec = baseSection(editor)
  const def = (k: string, fb: number) => {
    const v = sec[k]
    if (typeof v === "number") return v
    const f = editor.fieldByKey.get(`${secName(editor)}.${k}`)?.default
    return typeof f === "number" ? f : fb
  }
  const slots = Math.max(1, def(slotsKey(editor), 1))
  const pool = sec.kv_unified === true
  const total = def("ctx_size", 4096)
  return {
    context: pool ? total : Math.max(1, Math.floor(total / slots)),
    slots,
    pool,
    kv: kvFromCacheType(sec.cache_type_k),
    gpus: activeGpus(placement),
  }
}

function writeBase(editor: ModelEditorState, next: WorkbenchState) {
  const set = (k: string, v: Json | undefined) => editor.setField([secName(editor), k], v)
  set("ctx_size", next.pool ? next.context : next.context * next.slots)
  set(slotsKey(editor), next.slots)
  set("kv_unified", next.pool ? true : undefined)
  const ct = CACHE_TYPE[next.kv]
  set("cache_type_k", ct)
  set("cache_type_v", ct)
  if (ct) set("flash_attn", true)
}

type SmokeResult = Awaited<ReturnType<typeof modelEditorApi.smokeTest>>

/** Las dos pruebas de un peldaño: si carga y responde tal cual, y cuánto
 *  contexto cabe de verdad (solo llama.cpp mks, que amplía en caliente). */
function RungTests({
  readOnly,
  testLabel,
  testBlocked,
  onTest,
  testing,
  result,
  probe,
  probeBlocked,
}: {
  readOnly: boolean
  testLabel: string
  testBlocked: string | null
  onTest: () => void
  testing: boolean
  result: string | null
  /** null = el backend no sabe medir el contexto en caliente. */
  probe: React.ReactNode | null
  probeBlocked: string | null
}) {
  return (
    <div className="grid grid-cols-[repeat(auto-fit,minmax(min(100%,280px),1fr))] gap-4" data-testid="rung-tests">
      <div className="space-y-2 rounded-xl border border-border p-3">
        <p className="text-xs font-medium">¿Carga y responde así?</p>
        <Hint>Carga el modelo en frío con esta configuración y le manda una petición mínima. Si estaba cargado se descarga antes.</Hint>
        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            disabled={readOnly || testing || testBlocked != null}
            title={testBlocked ?? undefined}
            onClick={onTest}
            className="inline-flex items-center gap-1 rounded-md border border-emerald-500/50 px-3 py-1.5 text-xs hover:bg-emerald-500/10 disabled:opacity-40"
            data-testid="smoke-test"
          >
            <Play size={13} /> {testing ? "Probando…" : testLabel}
          </button>
          {testBlocked && <Hint>{testBlocked}</Hint>}
        </div>
        {result && <p className="text-[11.5px] tabular-nums">{result}</p>}
      </div>
      {probe !== null && (
        <div className="space-y-2 rounded-xl border border-border p-3">
          <p className="text-xs font-medium">¿Cuánto contexto cabe de verdad?</p>
          <Hint>
            La gráfica es una estimación. El sondeo carga el modelo y amplía el contexto en caliente hasta el límite de sus GPUs; lo medido queda
            como valor validado y aparece en la gráfica.
          </Hint>
          {probeBlocked ? <Hint tone="warn">{probeBlocked}</Hint> : probe}
        </div>
      )}
    </div>
  )
}

function describeSmoke(r: SmokeResult): string {
  if (r.error) return `✗ ${r.error}`
  const parts = [r.loaded ? `cargó en ${r.load_ms != null ? `${(r.load_ms / 1000).toFixed(1)} s` : "?"}` : "no cargó"]
  if (r.request_kind) parts.push(r.request_ok ? `${r.request_kind} ✓ en ${r.request_ms ?? "?"} ms` : `${r.request_kind} ✗`)
  if (r.tokens_per_second) parts.push(`${r.tokens_per_second.toFixed(1)} t/s`)
  return `${r.request_ok === false || !r.loaded ? "✗" : "✓"} ${parts.join(" · ")}`
}

/** Capacidad de un modelo llama.cpp sin presets: su base es lo que carga. */
export function BaseLlamaWorkbench(props: SectionProps & { onCreatePreset?: () => void }) {
  const { editor, readOnly, gpus, liveVramByGpu, onCreatePreset } = props
  const gpuIndices = gpus.map((g) => g.index)
  const rough = readBasePlacementOf(editor, gpuIndices, null)
  const { data: cap } = useCapacity(editor.model?.model_id, { gpu: rough.star ?? rough.marked[0] ?? null, slots: [1], kvDtype: "q8" })
  const layers = cap?.arch?.layers ?? null
  const placement = readBasePlacementOf(editor, gpuIndices, layers)
  const value = readBase(editor, placement)
  const [testing, setTesting] = useState(false)
  const [result, setResult] = useState<string | null>(null)
  const memory: MemoryShare[] = (editor.preview?.gpus ?? []).map((g) => ({
    gpu: g.index,
    parts: Object.keys(g.breakdown).length ? g.breakdown : { model: g.mb },
  }))
  const mks = editor.model?.backend_type === "llamacpp_mks"
  const smoke = async () => {
    if (!editor.model) return
    setTesting(true)
    setResult(null)
    try {
      setResult(describeSmoke(await modelEditorApi.smokeTest(editor.model.model_id, true)))
    } catch (e) {
      setResult(`✗ ${errMsg(e, "No se pudo probar")}`)
    } finally {
      setTesting(false)
    }
  }
  return (
    <Workbench
      title="Memoria y contexto"
      subtitle="Cambia un control y mira cómo se mueve la memoria. Se guarda con Guardar."
      value={value}
      onChange={(patch) => writeBase(editor, { ...value, ...patch })}
      readOnly={readOnly}
      gpus={gpus}
      liveVramByGpu={liveVramByGpu}
      memory={memory}
      cap={cap}
      native={cap?.arch?.native_context ?? (Number(editor.model?.capabilities?.native_context_length) || null)}
      allowMax={false}
      poolEditable
      devices={
        <DevicePlacement
          value={placement}
          onChange={(patch) => {
            const setKey = (k: string, v: Json | undefined) => editor.setField([secName(editor), k], v)
            const setPreferred = (g: number | null) => editor.update({ preferredGpu: g })
            if (isBitnet(editor)) writeBitnetPlacement(setKey, setPreferred, { ...placement, ...patch }, patch)
            else writeBasePlacement(setKey, setPreferred, gpuIndices, { ...placement, ...patch }, patch)
          }}
          preferOnly={isBitnet(editor)}
          gpus={gpus}
          readOnly={readOnly}
          liveVramByGpu={liveVramByGpu}
          memory={memory}
          layers={layers}
        />
      }
      footer={
        <RungTests
          readOnly={readOnly}
          testLabel="Probar tal cual"
          testBlocked={editor.dirty ? "Guarda los cambios antes de probar." : null}
          onTest={() => void smoke()}
          testing={testing}
          result={result}
          probe={
            mks ? (
              <div className="flex flex-wrap items-center gap-2" data-testid="probe-needs-preset">
                <Hint>El sondeo mide sobre un peldaño y este modelo no tiene ninguno: crea uno en modo «lo que quepa» y mídelo desde él.</Hint>
                {!readOnly && onCreatePreset && (
                  <button type="button" onClick={onCreatePreset} className="inline-flex items-center gap-1 rounded-md border border-violet-500/50 px-3 py-1.5 text-xs hover:bg-violet-500/10">
                    <FlaskConical size={13} /> Crear peldaño y medir
                  </button>
                )}
              </div>
            ) : null
          }
          probeBlocked={null}
        />
      }
    />
  )
}

// ── Un preset ───────────────────────────────────────────────────────────────

function formToState(form: RuntimePresetCreate, pool: boolean, slotsFallback: number, placement: Placement): WorkbenchState {
  return {
    context: form.contextTokens ?? 65536,
    maxMode: form.contextMode === "max",
    slots: form.parallelism || slotsFallback,
    pool,
    kv: (form.kvCacheDtype as KvLabel) ?? "q8",
    gpus: activeGpus(placement),
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
  const sec = editor.model?.backend_type === "bitnet" ? "bitnet" : "llama_cpp"
  const pool = (getPath(editor.draft?.extraConfig, [sec, "kv_unified"]) ?? getPath(editor.base?.extraConfig, [sec, "kv_unified"])) === true
  const dirty = JSON.stringify(form) !== JSON.stringify(initial)
  const isNew = preset == null
  const keyValid = PRESET_KEY_RE.test(form.presetKey) && !(isNew && profile.runtimePresets.some((p) => p.presetKey === form.presetKey))
  const gpuIndices = gpus.map((g) => g.index)
  const rough = readPresetPlacement(form, gpuIndices, null)
  const kvDtype = form.kvCacheDtype === "q4" ? "q4" : form.kvCacheDtype === "fp16" || form.kvCacheDtype === "bf16" ? "fp16" : "q8"
  const { data: cap } = useCapacity(editor.model?.model_id, { gpu: activeGpus(rough)?.[0] ?? null, slots: [1], kvDtype })
  const layers = cap?.arch?.layers ?? null
  const placement = readPresetPlacement(form, gpuIndices, layers)
  const value = formToState(form, pool, 2, placement)
  const [result, setResult] = useState<string | null>(null)
  const mks = editor.model?.backend_type === "llamacpp_mks"

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

  const save = async (): Promise<boolean> => {
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
      return true
    } catch (e) {
      toast.error(errMsg(e, "No se pudo guardar el peldaño"))
      return false
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

  const probeGpus = form.placementPolicy !== "auto" && (form.preferredGpus?.length ?? 0) > 0
  const footer = (
    <div className="space-y-4">
      <RungTests
        readOnly={readOnly}
        testLabel={isNew ? "Crear y probar" : dirty ? "Guardar y probar" : "Probar tal cual"}
        testBlocked={!form.label || !keyValid ? "Falta el nombre o la clave." : null}
        onTest={() => setConfirm("bench")}
        testing={busy === "bench"}
        result={result}
        probe={mks ? preset && !isNew ? <ContextProbe profileId={profile.profileId} preset={preset} onApplied={() => void onChanged()} /> : <span /> : null}
        probeBlocked={
          isNew || !preset
            ? "Guarda el peldaño para poder medirlo."
            : dirty
              ? "Guarda los cambios antes de medir."
              : !probeGpus
                ? "Marca en «Dónde corre» las GPUs del peldaño (una, o varias a la vez): el sondeo mide sobre GPUs concretas."
                : null
        }
      />
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
        allowMax
        poolEditable={false}
        measured={measured}
        devices={
          <DevicePlacement
            value={placement}
            onChange={(patch) => setForm((f) => writePresetPlacement(f, { ...readPresetPlacement(f, gpuIndices, layers), ...patch }, gpuIndices))}
            gpus={gpus}
            readOnly={readOnly}
            liveVramByGpu={liveVramByGpu}
            memory={memory.length ? memory : null}
            layers={layers}
            // bitnet: el peldaño solo elige GPU; el reparto y la CPU son de llama.cpp.
            preferOnly={sec === "bitnet"}
            cpuEditable={sec !== "bitnet"}
          />
        }
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
        title={confirm === "bench" ? "Probar el peldaño tal cual" : "Borrar peldaño"}
        description={
          confirm === "bench"
            ? `${dirty || isNew ? "Guarda el peldaño, " : ""}descarga el modelo si está cargado y mide la carga en frío y una generación con este peldaño. Las peticiones en curso pueden fallar.`
            : "Se borra el peldaño y sus alias dejarán de resolver."
        }
        confirmLabel={confirm === "bench" ? "Probar" : "Borrar"}
        variant={confirm === "bench" ? "default" : "danger"}
        onConfirm={() => {
          const c = confirm
          setConfirm(null)
          if (c === "bench") {
            const key = preset?.presetKey ?? form.presetKey
            void (async () => {
              if ((dirty || isNew) && !(await save())) return
              setBusy("bench")
              setResult(null)
              try {
                const r = await api.profiles.benchmarkRuntimePreset(profile.profileId, key, true)
                setResult(`✓ cargó en ${(r.loadMs / 1000).toFixed(1)} s · generó ${r.outputTokens ?? "?"} tokens · ${r.tokensPerSecond?.toFixed(1) ?? "?"} t/s · contexto ${Math.round(r.effectiveContext / 1024)}k`)
                await onChanged()
              } catch (e) {
                setResult(`✗ ${errMsg(e, "No cargó o no respondió")}`)
              } finally {
                setBusy(null)
              }
            })()
            return
          }
          if (!preset) return
          void run("delete", () => api.profiles.deleteRuntimePreset(profile.profileId, preset.presetKey), "Peldaño borrado").then(() => onDone(null))
        }}
      />
    </>
  )
}

export type { RuntimePreset }
