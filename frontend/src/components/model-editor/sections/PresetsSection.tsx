import { useCallback, useEffect, useMemo, useState } from "react"
import { Check, Copy, Pencil, Plus, Star, Trash2, X } from "lucide-react"
import { useSearchParams } from "react-router-dom"
import { toast } from "sonner"
import { api } from "@/api/client"
import { modelEditorApi } from "@/api/modelEditor"
import { ConfirmDialog } from "@/components/common/ConfirmDialog"
import { GpuBadge } from "@/components/dashboard/GpuBadge"
import { fmtContext } from "@/components/dashboard/format"
import { ContextProbe } from "@/components/models/ContextProbe"
import { GpuMemoryBar } from "../PreviewPanel"
import { cn } from "@/lib/cn"
import type {
  GPUState,
  ModelProfile,
  RuntimePolicy,
  RuntimePreset,
  RuntimePresetBenchmark,
  RuntimePresetCreate,
  RuntimePresetResolution,
} from "@/types"
import { Callout, Chip, Hint, Panel, Segmented, Toggle, inputCls } from "../ui"
import {
  ALIAS_RE,
  AXIS_MARKS,
  DEFAULT_POLICY,
  EMPTY_PRESET,
  KV_HINT,
  PRESET_KEY_RE,
  axisPosition,
  limitingReason,
  presetToForm,
  sortRungs,
  suggestedLadder,
} from "./presets/presetLogic"
import type { SectionProps } from "./types"

const errMsg = (e: unknown, fallback: string) => (e instanceof Error ? e.message : fallback)

/** Perfiles de ejecución que admiten presets de contexto. */
function presetCapable(p: ModelProfile): boolean {
  return p.profileKind === "execution" && p.category === "llm" && !(p.routingTargets?.length)
}

/**
 * Sección "Presets": la escalera de contextos de cada perfil LLM del modelo,
 * su edición con vista previa, los alias que apuntan a cada peldaño y la
 * política con la que se elige y se cambia de preset. Son entidades propias:
 * todo se guarda al momento, no con el botón Guardar del editor.
 */
export function PresetsSection(props: SectionProps) {
  const { editor, readOnly } = props
  const modelId = editor.model?.model_id ?? null
  const [profiles, setProfiles] = useState<ModelProfile[] | null>(null)
  // ?p=<perfil> (desde /profiles) elige el perfil de entrada.
  const [search] = useSearchParams()
  const [selectedId, setSelectedId] = useState<string | null>(() => search.get("p"))
  const [loadError, setLoadError] = useState<string | null>(null)

  const loadProfiles = useCallback(async () => {
    if (!modelId) return
    try {
      const all = await api.profiles.listByModel(modelId)
      const usable = all.filter(presetCapable)
      setProfiles(usable)
      setSelectedId((cur) =>
        cur && usable.some((p) => p.profileId === cur)
          ? cur
          : (usable.find((p) => p.isDefault) ?? usable[0])?.profileId ?? null,
      )
      setLoadError(null)
    } catch (e) {
      setLoadError(errMsg(e, "No se pudieron cargar los perfiles"))
    }
  }, [modelId])

  useEffect(() => {
    void loadProfiles()
  }, [loadProfiles])

  const onChanged = useCallback(async () => {
    await loadProfiles()
    await Promise.allSettled([props.refreshLogical(), editor.refreshLive()])
  }, [loadProfiles, props, editor])

  if (loadError) return <Callout tone="error">{loadError}</Callout>
  if (!profiles) return <p className="text-sm text-muted-foreground">Cargando presets…</p>
  if (!profiles.length) {
    return (
      <Panel title="Presets de contexto">
        <Hint>
          Este modelo no tiene perfiles de chat (LLM) que no sean routers, así que no admite escalera de
          contextos. Los presets se crean sobre un perfil de categoría LLM.
        </Hint>
      </Panel>
    )
  }
  const profile = profiles.find((p) => p.profileId === selectedId) ?? profiles[0]
  return (
    <div className="space-y-4" data-testid="presets-section">
      {profiles.length > 1 && (
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs text-muted-foreground">Perfil:</span>
          <Segmented
            size="sm"
            value={profile.profileId}
            options={profiles.map((p) => ({
              value: p.profileId,
              label: (
                <span>
                  {p.displayName || p.profileId}
                  {p.isDefault ? " ★" : ""}
                </span>
              ),
            }))}
            onChange={setSelectedId}
          />
        </div>
      )}
      <ProfileLadder
        key={profile.profileId}
        profile={profile}
        gpus={props.gpus}
        readOnly={readOnly}
        nativeContext={Number(editor.model?.capabilities?.native_context_length || 0) || null}
        onChanged={onChanged}
      />
    </div>
  )
}

export function ProfileLadder({
  profile,
  gpus,
  readOnly,
  nativeContext,
  onChanged,
  focusKey,
  onFocusDone,
  liveVramByGpu,
}: {
  profile: ModelProfile
  gpus: GPUState[]
  readOnly: boolean
  nativeContext: number | null
  onChanged: () => Promise<void>
  /** Capacidad: solo el peldaño elegido, con su editor abierto ("new" = crear). */
  focusKey?: string
  /** Tras crear o borrar el peldaño enfocado: a qué peldaño pasar. */
  onFocusDone?: (key: string | null) => void
  liveVramByGpu?: Record<string, number>
}) {
  const presets = profile.runtimePresets
  const [resolutions, setResolutions] = useState<Record<string, RuntimePresetResolution>>({})
  const [resolving, setResolving] = useState(false)
  const [busy, setBusy] = useState(false)
  const [editing, setEditing] = useState<{ key: string | null; form: RuntimePresetCreate } | null>(null)
  // En modo enfocado el editor del peldaño elegido está siempre abierto.
  useEffect(() => {
    if (focusKey === undefined) return
    if (focusKey === "new") {
      setEditing({ key: null, form: { ...EMPTY_PRESET, isDefault: presets.length === 0 } })
      return
    }
    const target = presets.find((p) => p.presetKey === focusKey)
    setEditing(target ? { key: target.presetKey, form: presetToForm(target) } : null)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [focusKey, profile.profileId])
  const [benchmarks, setBenchmarks] = useState<Record<string, RuntimePresetBenchmark>>({})
  const [benchmarking, setBenchmarking] = useState<string | null>(null)
  const [confirm, setConfirm] = useState<{ kind: "bench" | "delete"; preset: RuntimePreset } | null>(null)

  const resolve = useCallback(async () => {
    setResolving(true)
    try {
      const rows = await api.profiles.resolveRuntimePresets(profile.profileId)
      setResolutions(Object.fromEntries(rows.map((r) => [r.presetKey, r])))
    } catch {
      /* la escalera se ve igualmente, sin lo efectivo */
    } finally {
      setResolving(false)
    }
  }, [profile.profileId])

  useEffect(() => {
    void resolve()
  }, [resolve, presets])

  const rungs = useMemo(() => sortRungs(presets, resolutions), [presets, resolutions])

  const axisMax = useMemo(() => {
    const values = [nativeContext ?? 0]
    for (const p of presets) {
      if (p.contextTokens) values.push(p.contextTokens)
      const r = resolutions[p.presetKey]
      if (r?.effectiveContext) values.push(r.effectiveContext)
      if (r?.modelMax) values.push(r.modelMax)
    }
    return Math.max(65536, ...values)
  }, [presets, resolutions, nativeContext])

  // Tope de una sola GPU: el mayor máximo de hardware entre los peldaños que
  // caben en una tarjeta, cuando otros necesitan dos.
  const singleGpuCeiling = useMemo(() => {
    const rows = Object.values(resolutions)
    if (!rows.some((r) => r.targetGpus.length > 1)) return null
    const single = rows.filter((r) => r.targetGpus.length === 1 && r.hardwareMax)
    return single.length ? Math.max(...single.map((r) => r.hardwareMax ?? 0)) : null
  }, [resolutions])

  const run = async (fn: () => Promise<unknown>, ok?: string, fail = "No se pudo completar la acción") => {
    setBusy(true)
    try {
      await fn()
      if (ok) toast.success(ok)
      await onChanged()
    } catch (e) {
      toast.error(errMsg(e, fail))
      await onChanged()
    } finally {
      setBusy(false)
    }
  }

  const createLadder = () =>
    run(async () => {
      for (const preset of suggestedLadder()) {
        await api.profiles.createRuntimePreset(profile.profileId, preset)
      }
    }, "Creados 32k, 64k (por defecto, p2), 128k y MAX", "No se pudieron crear todos los presets")

  const patchPreset = (preset: RuntimePreset, patch: Partial<RuntimePresetCreate>) =>
    run(() => api.profiles.updateRuntimePreset(profile.profileId, preset.presetKey, patch), undefined, "No se pudo actualizar el preset")

  const benchmark = async (preset: RuntimePreset) => {
    setBenchmarking(preset.presetKey)
    try {
      const result = await api.profiles.benchmarkRuntimePreset(profile.profileId, preset.presetKey, true)
      setBenchmarks((cur) => ({ ...cur, [preset.presetKey]: result }))
      toast.success(`Medición de ${preset.label} completada`)
    } catch (e) {
      toast.error(errMsg(e, "No se pudo medir el preset"))
    } finally {
      setBenchmarking(null)
    }
  }

  if (focusKey !== undefined) {
    const focused = presets.find((p) => p.presetKey === focusKey)
    return (
      <div className="space-y-3" data-testid="focused-rung">
        {focused && (
          <ol>
            <Rung
              preset={focused}
              resolution={resolutions[focused.presetKey]}
              benchmark={benchmarks[focused.presetKey]}
              axisMax={axisMax}
              gpus={gpus}
              profile={profile}
              readOnly={readOnly}
              busy={busy || benchmarking !== null}
              benchmarking={benchmarking === focused.presetKey}
              canDelete={!(focused.isDefault && presets.length > 1)}
              onEdit={() => setEditing({ key: focused.presetKey, form: presetToForm(focused) })}
              onToggle={() => void patchPreset(focused, { enabled: !focused.enabled })}
              onDefault={() => void patchPreset(focused, { isDefault: true })}
              onDelete={() => setConfirm({ kind: "delete", preset: focused })}
              onBenchmark={() => setConfirm({ kind: "bench", preset: focused })}
              onChanged={onChanged}
            />
          </ol>
        )}
        {editing && !readOnly && (
          <PresetEditor
            key={`${focusKey}-${editing.key ?? "new"}`}
            profile={profile}
            editingKey={editing.key}
            initial={editing.form}
            gpus={gpus}
            resolutions={resolutions}
            liveVramByGpu={liveVramByGpu}
            onCancel={() => (focusKey === "new" ? onFocusDone?.(null) : setEditing({ ...editing }))}
            onSaved={async (key) => {
              await onChanged()
              if (focusKey === "new") onFocusDone?.(key)
            }}
          />
        )}
        <ConfirmDialog
          open={confirm !== null}
          onOpenChange={(open) => !open && setConfirm(null)}
          title={confirm?.kind === "bench" ? "Medir en frío" : "Eliminar preset"}
          description={
            confirm?.kind === "bench"
              ? `Se descargará el modelo si está cargado y se medirá la carga en frío y una generación con «${confirm.preset.label}». Las peticiones en curso pueden fallar.`
              : `Se elimina el preset «${confirm?.preset.label ?? ""}» y sus alias dejarán de resolver.`
          }
          confirmLabel={confirm?.kind === "bench" ? "Medir" : "Eliminar"}
          variant={confirm?.kind === "bench" ? "default" : "danger"}
          onConfirm={() => {
            const c = confirm
            setConfirm(null)
            if (!c) return
            if (c.kind === "bench") void benchmark(c.preset)
            else
              void run(
                () => api.profiles.deleteRuntimePreset(profile.profileId, c.preset.presetKey),
                "Preset eliminado",
                "No se pudo eliminar el preset",
              ).then(() => onFocusDone?.(null))
          }}
        />
      </div>
    )
  }
  return (
    <>
      <Panel
        title="Escalera de contextos"
        subtitle={
          <>
            Cada peldaño es una variante del modelo con su contexto, concurrencia y colocación. El cliente elige
            uno con un alias o con <code>ocabra.context_preset</code>. <strong>Los presets se guardan al momento.</strong>
          </>
        }
        actions={
          <>
            <button
              type="button"
              onClick={() => void resolve()}
              disabled={resolving}
              className="rounded-md border border-border px-2 py-1 text-xs hover:bg-muted disabled:opacity-40"
            >
              {resolving ? "Calculando…" : "Recalcular capacidad"}
            </button>
            {!readOnly && (
              <button
                type="button"
                disabled={busy || editing !== null}
                onClick={() => setEditing({ key: null, form: { ...EMPTY_PRESET, isDefault: presets.length === 0 } })}
                className="inline-flex items-center gap-1 rounded-md bg-primary px-2 py-1 text-xs font-medium text-primary-foreground disabled:opacity-40"
              >
                <Plus size={12} /> Añadir peldaño
              </button>
            )}
          </>
        }
      >
        {presets.length === 0 ? (
          <div className="space-y-3">
            <Hint>
              Sin presets, el modelo se carga siempre con su configuración base. Una escalera deja que las
              peticiones pequeñas usen poca memoria y solo las grandes paguen el contexto alto.
            </Hint>
            {!readOnly && (
              <button
                type="button"
                disabled={busy}
                onClick={() => void createLadder()}
                className="rounded-md border border-primary/50 bg-primary/10 px-3 py-1.5 text-xs font-medium text-primary hover:bg-primary/20 disabled:opacity-40"
                data-testid="create-ladder"
              >
                Crear escalera 32k / 64k / 128k / MAX
              </button>
            )}
          </div>
        ) : (
          <div className="space-y-2">
            <Axis max={axisMax} singleGpuCeiling={singleGpuCeiling} native={nativeContext} />
            <ol className="space-y-2" data-testid="ladder">
              {rungs.map((preset) => (
                <Rung
                  key={preset.presetKey}
                  preset={preset}
                  resolution={resolutions[preset.presetKey]}
                  benchmark={benchmarks[preset.presetKey]}
                  axisMax={axisMax}
                  gpus={gpus}
                  profile={profile}
                  readOnly={readOnly}
                  busy={busy || benchmarking !== null}
                  benchmarking={benchmarking === preset.presetKey}
                  canDelete={!(preset.isDefault && presets.length > 1)}
                  onEdit={() => setEditing({ key: preset.presetKey, form: presetToForm(preset) })}
                  onToggle={() => void patchPreset(preset, { enabled: !preset.enabled })}
                  onDefault={() => void patchPreset(preset, { isDefault: true })}
                  onDelete={() => setConfirm({ kind: "delete", preset })}
                  onBenchmark={() => setConfirm({ kind: "bench", preset })}
                  onChanged={onChanged}
                />
              ))}
            </ol>
          </div>
        )}
      </Panel>

      {editing && (
        <PresetEditor
          profile={profile}
          editingKey={editing.key}
          initial={editing.form}
          gpus={gpus}
          resolutions={resolutions}
          onCancel={() => setEditing(null)}
          onSaved={async () => {
            setEditing(null)
            await onChanged()
          }}
        />
      )}

      <PolicyPanel profile={profile} readOnly={readOnly} onChanged={onChanged} />

      <ConfirmDialog
        open={confirm !== null}
        onOpenChange={(open) => !open && setConfirm(null)}
        title={confirm?.kind === "bench" ? "Medir en frío" : "Eliminar preset"}
        description={
          confirm?.kind === "bench"
            ? `Se descargará el modelo si está cargado y se medirá la carga en frío y una generación con «${confirm.preset.label}». Las peticiones en curso pueden fallar.`
            : `Se elimina el preset «${confirm?.preset.label ?? ""}» y sus alias dejarán de resolver.`
        }
        confirmLabel={confirm?.kind === "bench" ? "Medir" : "Eliminar"}
        variant={confirm?.kind === "bench" ? "default" : "danger"}
        onConfirm={() => {
          const c = confirm
          setConfirm(null)
          if (!c) return
          if (c.kind === "bench") void benchmark(c.preset)
          else
            void run(
              () => api.profiles.deleteRuntimePreset(profile.profileId, c.preset.presetKey),
              "Preset eliminado",
              "No se pudo eliminar el preset",
            )
        }}
      />
    </>
  )
}

function Axis({ max, singleGpuCeiling, native }: { max: number; singleGpuCeiling: number | null; native: number | null }) {
  const marks = AXIS_MARKS.filter((m) => m <= max * 1.01)
  return (
    <div className="relative ml-[13rem] mr-2 h-6 border-b border-border text-[10px] text-muted-foreground" aria-hidden>
      {marks.map((m) => (
        <span key={m} className="absolute -translate-x-1/2" style={{ left: `${axisPosition(m, max) * 100}%` }}>
          {fmtContext(m)}
        </span>
      ))}
      {singleGpuCeiling && (
        <span
          className="absolute bottom-0 top-3 border-l-2 border-dashed border-sky-400"
          style={{ left: `${axisPosition(singleGpuCeiling, max) * 100}%` }}
          title={`Tope con una sola GPU: ${fmtContext(singleGpuCeiling)}`}
        />
      )}
      {native && native > 0 && (
        <span
          className="absolute bottom-0 top-3 border-l border-muted-foreground/60"
          style={{ left: `${axisPosition(native, max) * 100}%` }}
          title={`Contexto nativo del modelo: ${fmtContext(native)}`}
        />
      )}
    </div>
  )
}

function Rung({
  preset,
  resolution,
  benchmark,
  axisMax,
  gpus,
  profile,
  readOnly,
  busy,
  benchmarking,
  canDelete,
  onEdit,
  onToggle,
  onDefault,
  onDelete,
  onBenchmark,
  onChanged,
}: {
  preset: RuntimePreset
  resolution?: RuntimePresetResolution
  benchmark?: RuntimePresetBenchmark
  axisMax: number
  gpus: GPUState[]
  profile: ModelProfile
  readOnly: boolean
  busy: boolean
  benchmarking: boolean
  canDelete: boolean
  onEdit: () => void
  onToggle: () => void
  onDefault: () => void
  onDelete: () => void
  onBenchmark: () => void
  onChanged: () => Promise<void>
}) {
  const requested = preset.contextMode === "fixed" ? preset.contextTokens ?? 0 : resolution?.effectiveContext ?? axisMax
  const effective = resolution?.effectiveContext ?? null
  const fits = resolution ? resolution.fits : null
  const reason = limitingReason(resolution)
  const targetGpus = resolution?.targetGpus?.length ? resolution.targetGpus : preset.preferredGpus ?? []
  const parallelText =
    resolution && resolution.effectiveParallelism !== preset.parallelism
      ? `p${preset.parallelism} → p${resolution.effectiveParallelism}`
      : `p${preset.parallelism}`
  const barColor = fits === false ? "bg-red-500/70" : preset.isDefault ? "bg-sky-500/80" : "bg-emerald-500/60"
  return (
    <li
      className={cn("rounded-xl border border-border bg-background/40 p-2", !preset.enabled && "opacity-55")}
      data-rung={preset.presetKey}
    >
      <div className="flex items-center gap-3">
        <div className="w-[12.5rem] shrink-0">
          <div className="flex items-center gap-1.5">
            {preset.isDefault && (
              <span title="Por defecto: hereda la política de carga del modelo; el resto carga siempre bajo demanda">
                <Star size={13} className="fill-sky-400 text-sky-400" aria-label="Preset por defecto" />
              </span>
            )}
            <span className="truncate text-sm font-medium">{preset.label}</span>
          </div>
          <div className="font-mono text-[10.5px] text-muted-foreground">
            {preset.presetKey} · {preset.contextMode === "max" ? "MAX" : fmtContext(preset.contextTokens)}
          </div>
        </div>
        <div className="relative h-5 flex-1 rounded bg-muted/40" title="Contexto pedido (claro) y efectivo (sólido)">
          <div
            className="absolute inset-y-0 left-0 rounded bg-muted-foreground/20"
            style={{ width: `${axisPosition(requested, axisMax) * 100}%` }}
          />
          {effective != null && (
            <div
              className={cn("absolute inset-y-0.5 left-0 rounded", barColor)}
              style={{ width: `${axisPosition(effective, axisMax) * 100}%` }}
            />
          )}
          <span className="absolute inset-y-0 right-1 flex items-center text-[10.5px] font-medium tabular-nums">
            {effective != null ? `${fmtContext(effective)} efectivos` : "—"}
          </span>
        </div>
      </div>

      <div className="mt-1.5 flex flex-wrap items-center gap-1.5 pl-[13.25rem]">
        {fits === true && <Chip tone="ok"><Check size={11} /> cabe</Chip>}
        {fits === false && <Chip tone="error"><X size={11} /> no cabe</Chip>}
        {reason && <Chip tone="warn" title="Lo que recorta el contexto pedido">{reason}</Chip>}
        <Chip title="Peticiones simultáneas pedidas → efectivas">{parallelText}</Chip>
        <Chip title={KV_HINT[preset.kvCacheDtype]}>KV {preset.kvCacheDtype}</Chip>
        <Chip title="Colocación">{preset.placementPolicy === "auto" ? "colocación auto" : preset.placementPolicy === "preferred" ? "preferida" : "forzada"}</Chip>
        {targetGpus.map((g) => (
          <GpuBadge key={g} index={g} gpus={gpus} />
        ))}
        {preset.executionPolicy !== "auto" && <Chip tone="info">{preset.executionPolicy === "local" ? "solo local" : "remota"}</Chip>}
        <Chip title="Prioridad (menor = antes)">prioridad {preset.priority}</Chip>
        {!preset.enabled && <Chip>desactivado</Chip>}
      </div>

      {resolution?.warnings.map((w) => (
        <p key={w} className="mt-1 pl-[13.25rem] text-[10.5px] text-amber-300">{w}</p>
      ))}
      {benchmark && (
        <p className="mt-1 pl-[13.25rem] text-[10.5px] text-emerald-300">
          Carga {benchmark.loadMs.toLocaleString("es-ES")} ms · generación {benchmark.generationMs.toLocaleString("es-ES")} ms
          {benchmark.tokensPerSecond != null ? ` · ${benchmark.tokensPerSecond} t/s` : ""}
        </p>
      )}

      <div className="mt-2 flex flex-wrap items-center gap-1.5 pl-[13.25rem]">
        {!readOnly && (
          <>
            <button type="button" disabled={busy} onClick={onEdit} className="inline-flex items-center gap-1 rounded border border-border px-2 py-0.5 text-[11px] hover:bg-muted disabled:opacity-40">
              <Pencil size={11} /> Editar
            </button>
            {!preset.isDefault && preset.enabled && (
              <button type="button" disabled={busy} onClick={onDefault} className="inline-flex items-center gap-1 rounded border border-border px-2 py-0.5 text-[11px] hover:bg-muted disabled:opacity-40">
                <Star size={11} /> Por defecto
              </button>
            )}
            <button
              type="button"
              disabled={busy || preset.isDefault}
              onClick={onToggle}
              title={preset.isDefault ? "El preset por defecto no se puede desactivar" : undefined}
              className="rounded border border-border px-2 py-0.5 text-[11px] hover:bg-muted disabled:opacity-40"
              data-testid={`toggle-${preset.presetKey}`}
            >
              {preset.enabled ? "Desactivar" : "Activar"}
            </button>
            {preset.enabled && (
              <button type="button" disabled={busy} onClick={onBenchmark} className="rounded border border-border px-2 py-0.5 text-[11px] hover:bg-muted disabled:opacity-40">
                {benchmarking ? "Midiendo…" : "Medir en frío"}
              </button>
            )}
            {preset.enabled && (
              <ContextProbe profileId={profile.profileId} preset={preset} onApplied={() => void onChanged()} />
            )}
            <button
              type="button"
              disabled={busy || !canDelete}
              onClick={onDelete}
              title={!canDelete ? "Elige otro preset por defecto antes de borrar este" : "Eliminar"}
              className="rounded border border-red-500/40 p-1 text-red-300 hover:bg-red-500/10 disabled:opacity-40"
              data-testid={`delete-${preset.presetKey}`}
            >
              <Trash2 size={11} />
            </button>
          </>
        )}
      </div>

      <AliasList profile={profile} preset={preset} readOnly={readOnly} onChanged={onChanged} />
    </li>
  )
}

function AliasList({
  profile,
  preset,
  readOnly,
  onChanged,
}: {
  profile: ModelProfile
  preset: RuntimePreset
  readOnly: boolean
  onChanged: () => Promise<void>
}) {
  const aliases = profile.aliases.filter((a) => a.presetKey === preset.presetKey)
  const [draft, setDraft] = useState("")
  const [hidden, setHidden] = useState(true)
  const [busy, setBusy] = useState(false)
  const valid = ALIAS_RE.test(draft)
  const add = async () => {
    setBusy(true)
    try {
      await modelEditorApi.createAlias(profile.profileId, { alias_id: draft, preset_key: preset.presetKey, hidden })
      setDraft("")
      toast.success(`Alias ${draft} creado`)
      await onChanged()
    } catch (e) {
      toast.error(errMsg(e, "No se pudo crear el alias"))
    } finally {
      setBusy(false)
    }
  }
  const remove = async (aliasId: string) => {
    setBusy(true)
    try {
      await modelEditorApi.deleteAlias(profile.profileId, aliasId)
      await onChanged()
    } catch (e) {
      toast.error(errMsg(e, "No se pudo borrar el alias"))
    } finally {
      setBusy(false)
    }
  }
  if (!aliases.length && readOnly) return null
  return (
    <div className="mt-2 flex flex-wrap items-center gap-1.5 pl-[13.25rem] text-[11px]">
      <span className="text-muted-foreground">Alias:</span>
      {aliases.map((a) => (
        <span key={a.aliasId} className="inline-flex items-center gap-1 rounded-md border border-border bg-muted/40 px-1.5 py-0.5 font-mono">
          {a.aliasId}
          {!a.hidden && <span className="font-sans text-[10px] text-sky-300">público</span>}
          <button
            type="button"
            title="Copiar"
            onClick={() => {
              void navigator.clipboard?.writeText(a.aliasId)
              toast.success("Copiado")
            }}
            className="text-muted-foreground hover:text-foreground"
          >
            <Copy size={10} />
          </button>
          {!readOnly && (
            <button type="button" title="Borrar alias" disabled={busy} onClick={() => void remove(a.aliasId)} className="text-red-300 hover:text-red-200">
              <X size={10} />
            </button>
          )}
        </span>
      ))}
      {!aliases.length && <span className="text-muted-foreground/70">ninguno</span>}
      {!readOnly && (
        <span className="inline-flex items-center gap-1">
          <input
            className={cn(inputCls, "h-6 w-40 py-0 font-mono text-[11px]", draft && !valid && "border-red-500")}
            placeholder={`${profile.profileId}:${preset.presetKey}`}
            value={draft}
            onChange={(e) => setDraft(e.target.value.trim())}
          />
          <label className="inline-flex items-center gap-1 text-muted-foreground" title="Oculto: resuelve pero no sale en /v1/models">
            <input type="checkbox" checked={hidden} onChange={(e) => setHidden(e.target.checked)} /> oculto
          </label>
          <button
            type="button"
            disabled={busy || !valid}
            onClick={() => void add()}
            className="rounded border border-border px-1.5 py-0.5 hover:bg-muted disabled:opacity-40"
          >
            Añadir
          </button>
        </span>
      )}
    </div>
  )
}

const LOG_MIN = 13 // 8k
const LOG_MAX = 20 // 1M

function ContextSlider({ value, onChange, max }: { value: number; onChange: (n: number) => void; max: number }) {
  const hi = Math.min(LOG_MAX, Math.max(18, Math.ceil(Math.log2(Math.max(max, 1)))))
  const pos = Math.log2(Math.max(value || 8192, 8192))
  const marks = AXIS_MARKS.filter((m) => Math.log2(m) <= hi)
  return (
    <div>
      <input
        type="range"
        min={LOG_MIN}
        max={hi}
        step={0.25}
        value={Math.min(pos, hi)}
        onChange={(e) => {
          const raw = 2 ** Number(e.target.value)
          onChange(Math.max(1024, Math.round(raw / 1024) * 1024))
        }}
        className="w-full accent-primary"
        aria-label="Contexto (escala logarítmica)"
      />
      <div className="relative h-3 text-[9.5px] text-muted-foreground">
        {marks.map((m) => (
          <span key={m} className="absolute -translate-x-1/2" style={{ left: `${((Math.log2(m) - LOG_MIN) / (hi - LOG_MIN)) * 100}%` }}>
            {fmtContext(m)}
          </span>
        ))}
      </div>
    </div>
  )
}

function PresetEditor({
  profile,
  editingKey,
  initial,
  gpus,
  resolutions,
  onCancel,
  onSaved,
  liveVramByGpu,
}: {
  profile: ModelProfile
  editingKey: string | null
  initial: RuntimePresetCreate
  gpus: GPUState[]
  resolutions: Record<string, RuntimePresetResolution>
  onCancel: () => void
  onSaved: (presetKey: string) => Promise<void>
  liveVramByGpu?: Record<string, number>
}) {
  const [form, setForm] = useState<RuntimePresetCreate>(initial)
  const [preview, setPreview] = useState<RuntimePresetResolution | null>(null)
  const [previewing, setPreviewing] = useState(false)
  const [saving, setSaving] = useState(false)
  const set = (patch: Partial<RuntimePresetCreate>) => setForm((f) => ({ ...f, ...patch }))
  const isNew = editingKey === null
  const keyValid = PRESET_KEY_RE.test(form.presetKey) && form.presetKey.length <= 64
  const keyTaken = isNew && profile.runtimePresets.some((p) => p.presetKey === form.presetKey)
  const ctxMissing = form.contextMode === "fixed" && (!form.contextTokens || form.contextTokens < 1)
  const gpuMissing = form.placementPolicy !== "auto" && !form.preferredGpus?.length

  useEffect(() => {
    if (ctxMissing || gpuMissing) {
      setPreview(null)
      return
    }
    let cancelled = false
    setPreviewing(true)
    const timer = window.setTimeout(() => {
      api.profiles
        .previewRuntimePreset(profile.profileId, form)
        .then((r) => !cancelled && setPreview(r))
        .catch(() => !cancelled && setPreview(null))
        .finally(() => !cancelled && setPreviewing(false))
    }, 400)
    return () => {
      cancelled = true
      window.clearTimeout(timer)
    }
  }, [form, profile.profileId, ctxMissing, gpuMissing])

  // ¿Quedaría al menos un preset activo que quepa? Si no, el backend devuelve
  // 400 al guardar; avisarlo antes.
  const othersFit = profile.runtimePresets.some(
    (p) => p.enabled && p.presetKey !== editingKey && resolutions[p.presetKey]?.fits,
  )
  const noneFit = preview != null && !preview.fits && !othersFit && form.enabled !== false

  const toggleGpu = (index: number) => {
    const cur = form.preferredGpus ?? []
    const next = cur.includes(index) ? cur.filter((g) => g !== index) : [...cur, index].sort((a, b) => a - b)
    set({ preferredGpus: next.length ? next : null })
  }

  const save = async () => {
    setSaving(true)
    try {
      const payload: RuntimePresetCreate = {
        ...form,
        contextTokens: form.contextMode === "max" ? null : form.contextTokens,
        preferredGpus: form.placementPolicy === "auto" ? null : form.preferredGpus,
      }
      if (isNew) await api.profiles.createRuntimePreset(profile.profileId, payload)
      else {
        const { presetKey: _k, ...update } = payload
        void _k
        await api.profiles.updateRuntimePreset(profile.profileId, editingKey, update)
      }
      toast.success("Preset guardado; se aplica en la siguiente carga")
      await onSaved(form.presetKey)
    } catch (e) {
      toast.error(errMsg(e, "No se pudo guardar el preset"))
    } finally {
      setSaving(false)
    }
  }

  const ceiling = preview?.hardwareMax ?? 0
  const requested = form.contextMode === "fixed" ? form.contextTokens ?? 0 : preview?.effectiveContext ?? 0
  const frac = ceiling > 0 ? Math.min(1, requested / ceiling) : 0

  return (
    <Panel
      title={isNew ? "Nuevo peldaño" : `Editar «${initial.label}»`}
      subtitle="La vista previa calcula si cabe con la memoria y las GPUs reales, sin guardar nada."
      className="border-primary/40"
      actions={
        <button type="button" onClick={onCancel} className="rounded-md border border-border px-2 py-1 text-xs hover:bg-muted">
          Cancelar
        </button>
      }
    >
      <div className="grid gap-4 lg:grid-cols-[1fr_18rem]">
        <div className="grid gap-3">
          <div className="grid gap-3 sm:grid-cols-[10rem_1fr_7rem]">
            <label className="grid gap-1 text-xs">
              <span className="font-medium">Clave</span>
              <input
                className={cn(inputCls, "font-mono", (!keyValid || keyTaken) && form.presetKey && "border-red-500")}
                value={form.presetKey}
                disabled={!isNew}
                onChange={(e) => set({ presetKey: e.target.value.trim().toLowerCase() })}
              />
              {!isNew && <Hint>No se puede cambiar: la usan los alias.</Hint>}
              {isNew && keyTaken && <Hint tone="error">Ya hay un preset con esa clave.</Hint>}
            </label>
            <label className="grid gap-1 text-xs">
              <span className="font-medium">Nombre</span>
              <input className={inputCls} value={form.label} onChange={(e) => set({ label: e.target.value })} />
            </label>
            <label className="grid gap-1 text-xs">
              <span className="font-medium">Prioridad</span>
              <input
                type="number"
                min={0}
                className={inputCls}
                value={form.priority ?? 100}
                onChange={(e) => set({ priority: Math.max(0, Number(e.target.value) || 0) })}
              />
            </label>
          </div>

          <div className="grid gap-1 text-xs">
            <span className="font-medium">Contexto</span>
            <div className="flex flex-wrap items-center gap-2">
              <Segmented
                size="sm"
                value={form.contextMode}
                options={[
                  { value: "fixed", label: "Fijo" },
                  { value: "max", label: "MAX (lo que quepa)" },
                ]}
                onChange={(v) => set({ contextMode: v, contextTokens: v === "max" ? null : form.contextTokens ?? 65536 })}
              />
              {form.contextMode === "fixed" && (
                <input
                  type="number"
                  min={1}
                  className={cn(inputCls, "w-32 tabular-nums")}
                  value={form.contextTokens ?? ""}
                  onChange={(e) => set({ contextTokens: e.target.value ? Number(e.target.value) : null })}
                  aria-label="Tokens de contexto"
                />
              )}
              {form.contextMode === "fixed" && <span className="text-muted-foreground">{fmtContext(form.contextTokens)}</span>}
            </div>
            {form.contextMode === "fixed" && (
              <ContextSlider value={form.contextTokens ?? 65536} max={preview?.modelMax ?? 262144} onChange={(n) => set({ contextTokens: n })} />
            )}
          </div>

          <div className="grid gap-3 sm:grid-cols-2">
            <label className="grid gap-1 text-xs">
              <span className="font-medium">Peticiones simultáneas: p{form.parallelism}</span>
              <input
                type="range"
                min={1}
                max={64}
                value={form.parallelism}
                onChange={(e) => set({ parallelism: Number(e.target.value) })}
                className="accent-primary"
                aria-label="Paralelismo"
              />
            </label>
            <div className="grid gap-1 text-xs">
              <span className="font-medium">Precisión de la KV</span>
              <Segmented
                size="sm"
                value={form.kvCacheDtype}
                options={(["fp16", "bf16", "fp8", "q8", "q4"] as const).map((v) => ({ value: v, label: v, title: KV_HINT[v] }))}
                onChange={(v) => set({ kvCacheDtype: v })}
              />
              <Hint>{KV_HINT[form.kvCacheDtype]}</Hint>
            </div>
          </div>

          <div className="grid gap-1 text-xs">
            <span className="font-medium">Colocación</span>
            <Segmented
              size="sm"
              value={form.placementPolicy}
              options={[
                { value: "auto", label: "Automática", title: "El planificador elige GPU" },
                { value: "preferred", label: "Preferida", title: "Intenta estas GPUs primero" },
                { value: "forced", label: "Forzada", title: "Solo en estas GPUs (varias = reparto)" },
              ]}
              onChange={(v) => set({ placementPolicy: v })}
            />
            {form.placementPolicy !== "auto" && (
              <div className="mt-1 flex flex-wrap gap-1.5">
                {gpus.map((g) => {
                  const on = form.preferredGpus?.includes(g.index) ?? false
                  return (
                    <button
                      key={g.index}
                      type="button"
                      aria-pressed={on}
                      onClick={() => toggleGpu(g.index)}
                      className={cn("rounded-md border p-0.5", on ? "border-primary bg-primary/10" : "border-transparent opacity-60 hover:opacity-100")}
                    >
                      <GpuBadge index={g.index} gpus={gpus} />
                    </button>
                  )
                })}
              </div>
            )}
            {gpuMissing && <Hint tone="error">Elige al menos una GPU para esta colocación.</Hint>}
          </div>

          <div className="flex flex-wrap items-center gap-4 text-xs">
            <div className="grid gap-1">
              <span className="font-medium">Ejecución</span>
              <Segmented
                size="sm"
                value={form.executionPolicy}
                options={[
                  { value: "auto", label: "Automática" },
                  { value: "local", label: "Solo local" },
                  { value: "remote", label: "Remota (federación)" },
                ]}
                onChange={(v) => set({ executionPolicy: v })}
              />
            </div>
            <Toggle
              checked={form.enabled !== false}
              disabled={initial.isDefault}
              onChange={(v) => set({ enabled: v })}
              label={<span className="text-xs">Activo{initial.isDefault ? " (el por defecto siempre)" : ""}</span>}
            />
          </div>
        </div>

        <div className="space-y-2">
          <div
            className={cn(
              "rounded-lg border p-3 text-xs",
              !preview ? "border-dashed border-border" : preview.fits ? "border-emerald-500/40 bg-emerald-500/5" : "border-red-500/40 bg-red-500/5",
            )}
            data-testid="preset-preview"
          >
            {!preview ? (
              <span className="text-muted-foreground">{previewing ? "Calculando…" : "Completa el formulario para ver si cabe."}</span>
            ) : (
              <>
                <div className="flex items-center justify-between gap-2 font-medium">
                  <span>{preview.fits ? "Cabe" : "No cabe"}</span>
                  <span className="text-muted-foreground">{previewing ? "actualizando…" : ""}</span>
                </div>
                <div className="mt-1 text-muted-foreground">
                  {fmtContext(preview.effectiveContext)} efectivos · p{preview.effectiveParallelism}
                  {ceiling ? ` · máx ${fmtContext(ceiling)} por slot` : ""}
                </div>
                <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-muted">
                  <div
                    className={cn("h-full", !preview.fits ? "bg-red-500" : frac > 0.85 ? "bg-amber-500" : "bg-emerald-500")}
                    style={{ width: `${Math.max(4, frac * 100)}%` }}
                  />
                </div>
                {preview.vramByGpu && Object.keys(preview.vramByGpu).length > 0 ? (
                  <div className="mt-2 space-y-1.5" data-testid="preset-memory">
                    {gpus
                      .filter((g) => preview.vramByGpu?.[String(g.index)] != null)
                      .map((g) => (
                        <GpuMemoryBar
                          key={g.index}
                          gpu={g}
                          compact
                          ownLiveMb={liveVramByGpu?.[String(g.index)] ?? 0}
                          share={{ index: g.index, mb: preview.vramByGpu![String(g.index)], breakdown: { preset: preview.vramByGpu![String(g.index)] } }}
                        />
                      ))}
                  </div>
                ) : preview.targetGpus.length > 0 ? (
                  <div className="mt-2 flex flex-wrap gap-1">
                    {preview.targetGpus.map((g) => (
                      <GpuBadge key={g} index={g} gpus={gpus} />
                    ))}
                  </div>
                ) : null}
                {preview.warnings.map((w) => (
                  <p key={w} className="mt-1 text-[10.5px] text-amber-300">{w}</p>
                ))}
              </>
            )}
          </div>
          {noneFit && (
            <Callout tone="error">
              Ningún preset activo cabría con este cambio y el servidor rechazará el guardado. Baja el contexto o el
              paralelismo, o deja otro preset que quepa.
            </Callout>
          )}
          <button
            type="button"
            disabled={saving || !form.label || !keyValid || keyTaken || ctxMissing || gpuMissing}
            onClick={() => void save()}
            className="w-full rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground disabled:opacity-40"
          >
            {saving ? "Guardando…" : isNew ? "Crear peldaño" : "Guardar peldaño"}
          </button>
        </div>
      </div>
    </Panel>
  )
}

function NumberField({
  label,
  value,
  onChange,
  disabled,
  placeholder,
  unit,
}: {
  label: string
  value: number | null
  onChange: (v: number | null) => void
  disabled?: boolean
  placeholder?: string
  unit?: string
}) {
  return (
    <label className="grid gap-1 text-xs">
      <span className="font-medium">{label}</span>
      <span className="flex items-center gap-1.5">
        <input
          type="number"
          min={0}
          aria-label={label}
          className={cn(inputCls, "tabular-nums")}
          disabled={disabled}
          placeholder={placeholder}
          value={value ?? ""}
          onChange={(e) => onChange(e.target.value === "" ? null : Math.max(0, Number(e.target.value)))}
        />
        {unit && <span className="text-muted-foreground">{unit}</span>}
      </span>
    </label>
  )
}

export function PolicyPanel({ profile, readOnly, onChanged }: { profile: ModelProfile; readOnly: boolean; onChanged: () => Promise<void> }) {
  const stored = profile.runtimePolicy ?? DEFAULT_POLICY
  const [policy, setPolicy] = useState<RuntimePolicy>(stored)
  const [saving, setSaving] = useState(false)
  useEffect(() => setPolicy(profile.runtimePolicy ?? DEFAULT_POLICY), [profile.runtimePolicy])
  const set = (patch: Partial<RuntimePolicy>) => setPolicy((p) => ({ ...p, ...patch }))
  const dirty = JSON.stringify(policy) !== JSON.stringify(stored)

  const save = async () => {
    setSaving(true)
    try {
      // Se manda la política completa: los campos calibrados (validados por la
      // sonda, reservas) viajan tal cual y no se pierden al guardar.
      await api.profiles.update(profile.profileId, { runtimePolicy: policy })
      toast.success("Política guardada")
      await onChanged()
    } catch (e) {
      toast.error(errMsg(e, "No se pudo guardar la política"))
    } finally {
      setSaving(false)
    }
  }

  const grace = policy.downgradeGraceSeconds
  const residency = policy.minimumResidencySeconds
  const total = Math.max(grace, residency, 1) * 1.15
  const validated = Object.entries(policy.validatedContextTokens ?? {})
  const vram = policy.validatedVramMb ?? {}

  return (
    <Panel
      title="Cómo se elige y se cambia de preset"
      subtitle="Se guarda con su propio botón, aparte del resto del editor."
      actions={
        !readOnly && (
          <button
            type="button"
            disabled={saving || !dirty}
            onClick={() => void save()}
            className="rounded-md bg-primary px-2.5 py-1 text-xs font-medium text-primary-foreground disabled:opacity-40"
          >
            {saving ? "Guardando…" : "Guardar política"}
          </button>
        )
      }
    >
      <div className="grid gap-4 lg:grid-cols-2">
        <div className="grid gap-3">
          <div className="grid gap-1 text-xs">
            <span className="font-medium">Al llegar una petición</span>
            <Segmented
              size="sm"
              disabled={readOnly}
              value={policy.selection}
              options={[
                { value: "prefer_loaded", label: "Usar el ya cargado si le basta" },
                { value: "prefer_default", label: "Preferir el por defecto" },
              ]}
              onChange={(v) => set({ selection: v })}
            />
          </div>
          <Toggle
            disabled={readOnly}
            checked={policy.restoreDefaultWhenIdle}
            onChange={(v) => set({ restoreDefaultWhenIdle: v })}
            label={<span className="text-xs">Volver al preset por defecto cuando quede ocioso</span>}
          />
          <Toggle
            disabled={readOnly}
            checked={policy.smallRequestsExtendLargeTtl}
            onChange={(v) => set({ smallRequestsExtendLargeTtl: v })}
            label={<span className="text-xs">Las peticiones pequeñas alargan la vida del preset grande</span>}
          />
          <div className="grid gap-3 sm:grid-cols-2">
            <NumberField label="Gracia antes de bajar" unit="s" disabled={readOnly} value={grace} onChange={(v) => set({ downgradeGraceSeconds: v ?? 0 })} />
            <NumberField label="Residencia mínima" unit="s" disabled={readOnly} value={residency} onChange={(v) => set({ minimumResidencySeconds: v ?? 0 })} />
            <NumberField label="Contexto máximo validado" unit="tokens" disabled={readOnly} placeholder="sin tope" value={policy.maxContextTokens} onChange={(v) => set({ maxContextTokens: v || null })} />
            <NumberField label="Máximo de salida (tokens)" disabled={readOnly} placeholder="sin tope" value={policy.maxOutputTokens} onChange={(v) => set({ maxOutputTokens: v || null })} />
            <NumberField
              label="Reserva de salida por petición"
              unit="tokens"
              disabled={readOnly}
              placeholder="la global"
              value={policy.defaultOutputReservationTokens}
              onChange={(v) => set({ defaultOutputReservationTokens: v || null })}
            />
          </div>
        </div>

        <div className="space-y-3">
          {/* Línea de tiempo: desde que el preset grande deja de hacer falta. */}
          <div className="rounded-lg border border-border bg-background/40 p-3 text-[11px]" aria-hidden>
            <div className="mb-2 font-medium text-foreground">Tras subir a un preset mayor</div>
            <div className="relative h-7 rounded bg-muted/40">
              <div className="absolute inset-y-0 left-0 rounded-l bg-violet-500/30" style={{ width: `${(residency / total) * 100}%` }} />
              <div
                className="absolute inset-y-1 rounded bg-amber-500/40"
                style={{ left: 0, width: `${(grace / total) * 100}%` }}
              />
              <span className="absolute top-0.5 border-l-2 border-sky-400" style={{ left: `${(grace / total) * 100}%`, height: "1.5rem" }} />
            </div>
            <div className="mt-1 flex flex-wrap gap-3 text-muted-foreground">
              <span><span className="inline-block h-2 w-2 rounded-sm bg-violet-500/50" /> residencia mínima {residency} s</span>
              <span><span className="inline-block h-2 w-2 rounded-sm bg-amber-500/60" /> gracia {grace} s</span>
              <span>
                <span className="inline-block h-2 w-0.5 bg-sky-400" />{" "}
                {policy.restoreDefaultWhenIdle ? "vuelve al por defecto si sigue ocioso" : "se queda en el grande"}
              </span>
            </div>
          </div>

          <div className="rounded-lg border border-border bg-background/40 p-3 text-[11px]">
            <div className="mb-1 font-medium text-foreground">Valores medidos (sonda de contexto)</div>
            {validated.length === 0 ? (
              <span className="text-muted-foreground">Ninguno todavía. Usa «Sondear contexto» en un peldaño.</span>
            ) : (
              <ul className="space-y-0.5">
                {validated.map(([key, tokens]) => (
                  <li key={key} className="flex justify-between gap-2 font-mono">
                    <span>{key}</span>
                    <span>
                      {fmtContext(tokens)}
                      {vram[key] ? ` · ${(vram[key] / 1024).toFixed(1)} GB` : ""}
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      </div>
    </Panel>
  )
}

