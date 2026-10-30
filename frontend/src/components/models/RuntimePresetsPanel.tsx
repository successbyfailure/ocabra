import { useEffect, useState } from "react"
import { Check, ChevronDown, ChevronUp, Pencil, Plus, Trash2, X } from "lucide-react"
import { toast } from "sonner"
import { api } from "@/api/client"
import { formatTokenCount } from "@/lib/modelContext"
import type {
  GPUState,
  ModelProfile,
  RuntimePolicy,
  RuntimePresetBenchmark,
  RuntimePreset,
  RuntimePresetCreate,
  RuntimePresetResolution,
} from "@/types"

interface Props {
  profile: ModelProfile
  onChanged: () => void
}

const DEFAULT_POLICY: RuntimePolicy = {
  selection: "prefer_loaded",
  restoreDefaultWhenIdle: true,
  downgradeGraceSeconds: 600,
  minimumResidencySeconds: 120,
  smallRequestsExtendLargeTtl: false,
  remoteWarmThresholdSeconds: 10,
  maxContextTokens: null,
}

const EMPTY_PRESET: RuntimePresetCreate = {
  presetKey: "64k",
  label: "Normal (64k)",
  contextMode: "fixed",
  contextTokens: 65536,
  parallelism: 2,
  kvCacheDtype: "q8",
  placementPolicy: "auto",
  preferredGpus: null,
  executionPolicy: "auto",
  enabled: true,
  isDefault: false,
  priority: 100,
}

function formatContext(preset: RuntimePreset): string {
  if (preset.contextMode === "max") return "MAX"
  const tokens = preset.contextTokens ?? 0
  return tokens >= 1024 && tokens % 1024 === 0 ? `${tokens / 1024}k` : tokens.toLocaleString()
}

function presetToForm(preset: RuntimePreset): RuntimePresetCreate {
  return {
    presetKey: preset.presetKey,
    label: preset.label,
    contextMode: preset.contextMode,
    contextTokens: preset.contextTokens,
    parallelism: preset.parallelism,
    kvCacheDtype: preset.kvCacheDtype,
    placementPolicy: preset.placementPolicy,
    preferredGpus: preset.preferredGpus,
    executionPolicy: preset.executionPolicy,
    enabled: preset.enabled,
    isDefault: preset.isDefault,
    priority: preset.priority,
  }
}

function LivePreviewPanel({
  loading,
  preview,
  gpus,
}: {
  loading: boolean
  preview: RuntimePresetResolution | null
  gpus: GPUState[]
}) {
  if (loading && !preview) {
    return (
      <div className="rounded-md border border-dashed border-border px-3 py-2 text-xs text-muted-foreground">
        Calculando capacidad…
      </div>
    )
  }
  if (!preview) return null

  const requested = preview.requestedContext ?? 0
  const ceiling = preview.hardwareMax ?? 0
  // How much of the per-slot budget the requested context would use — the
  // bar fills past 100% visually only in color (clamped width), since an
  // overflow is exactly the "doesn't fit" case the color already flags.
  const fraction = ceiling > 0 ? Math.min(1, requested / ceiling) : 0
  const overflow = ceiling > 0 && requested > ceiling
  const barColor = !preview.fits || overflow
    ? "bg-red-500"
    : fraction > 0.85
      ? "bg-amber-500"
      : "bg-emerald-500"
  const gpuNames = preview.targetGpus
    .map((index) => gpus.find((gpu) => gpu.index === index)?.name ?? `GPU ${index}`)
    .join(" + ")

  return (
    <div
      className={`rounded-md border px-3 py-2 text-xs ${
        preview.fits ? "border-emerald-500/30 bg-emerald-500/5" : "border-red-500/30 bg-red-500/5"
      }`}
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="font-medium">
          {preview.fits ? "Cabe" : "No cabe"} · {formatTokenCount(preview.effectiveContext)} efectivos
          {preview.targetGpus.length > 0 && ` en ${gpuNames}`}
          {preview.effectiveParallelism > 1 && ` · p${preview.effectiveParallelism}`}
        </span>
        <span className="text-muted-foreground">
          {loading ? "actualizando…" : `máx ${formatTokenCount(ceiling)}/slot`}
        </span>
      </div>
      <div className="mt-1.5 h-1.5 w-full overflow-hidden rounded-full bg-muted">
        <div
          className={`h-full transition-all ${barColor}`}
          style={{ width: `${Math.max(4, fraction * 100)}%` }}
        />
      </div>
      {preview.warnings.map((warning) => (
        <div key={warning} className="mt-1 text-[10px] text-amber-300">
          {warning}
        </div>
      ))}
    </div>
  )
}

export function RuntimePresetsPanel({ profile, onChanged }: Props) {
  const [presets, setPresets] = useState(profile.runtimePresets)
  const [policy, setPolicy] = useState(profile.runtimePolicy ?? DEFAULT_POLICY)
  const [editingKey, setEditingKey] = useState<string | null>(null)
  const [form, setForm] = useState<RuntimePresetCreate>(EMPTY_PRESET)
  const [busy, setBusy] = useState(false)
  const [policyOpen, setPolicyOpen] = useState(false)
  const [resolutions, setResolutions] = useState<Record<string, RuntimePresetResolution>>({})
  const [resolving, setResolving] = useState(false)
  const [benchmarking, setBenchmarking] = useState<string | null>(null)
  const [benchmarks, setBenchmarks] = useState<Record<string, RuntimePresetBenchmark>>({})
  const [gpus, setGpus] = useState<GPUState[]>([])
  const [livePreview, setLivePreview] = useState<RuntimePresetResolution | null>(null)
  const [livePreviewLoading, setLivePreviewLoading] = useState(false)

  const refresh = async () => {
    setPresets(await api.profiles.listRuntimePresets(profile.profileId))
  }

  const resolveCapacity = async () => {
    setResolving(true)
    try {
      const rows = await api.profiles.resolveRuntimePresets(profile.profileId)
      setResolutions(Object.fromEntries(rows.map((row) => [row.presetKey, row])))
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "No se pudo calcular la capacidad")
    } finally {
      setResolving(false)
    }
  }

  useEffect(() => {
    setPolicy(profile.runtimePolicy ?? DEFAULT_POLICY)
    void Promise.all([
      api.profiles.listRuntimePresets(profile.profileId),
      api.profiles.resolveRuntimePresets(profile.profileId),
      api.gpus.list(),
    ]).then(([nextPresets, rows, gpuList]) => {
      setPresets(nextPresets)
      setResolutions(Object.fromEntries(rows.map((row) => [row.presetKey, row])))
      setGpus(gpuList)
    }).catch((error) => {
      toast.error(error instanceof Error ? error.message : "No se pudieron cargar los presets")
    })
  }, [profile.profileId, profile.runtimePolicy])

  // Live estimation: resolve the in-progress draft against real GPU/model
  // capacity as the admin edits, instead of save -> "Calcular capacidad" ->
  // discover it doesn't fit -> edit -> save again.
  useEffect(() => {
    if (editingKey === null) {
      setLivePreview(null)
      return
    }
    if (form.contextMode === "fixed" && (!form.contextTokens || form.contextTokens < 1)) {
      setLivePreview(null)
      return
    }
    if (form.placementPolicy !== "auto" && !form.preferredGpus?.length) {
      setLivePreview(null)
      return
    }
    let cancelled = false
    setLivePreviewLoading(true)
    const timer = window.setTimeout(() => {
      void api.profiles
        .previewRuntimePreset(profile.profileId, form)
        .then((result) => {
          if (!cancelled) setLivePreview(result)
        })
        .catch(() => {
          if (!cancelled) setLivePreview(null)
        })
        .finally(() => {
          if (!cancelled) setLivePreviewLoading(false)
        })
    }, 400)
    return () => {
      cancelled = true
      window.clearTimeout(timer)
    }
  }, [editingKey, profile.profileId, form])

  const toggleGpu = (index: number) => {
    const current = form.preferredGpus ?? []
    const next = current.includes(index)
      ? current.filter((value) => value !== index)
      : [...current, index].sort((a, b) => a - b)
    setForm({ ...form, preferredGpus: next.length ? next : null })
  }

  const startCreate = () => {
    setEditingKey("")
    setForm({ ...EMPTY_PRESET, isDefault: presets.length === 0 })
  }

  const startEdit = (preset: RuntimePreset) => {
    setEditingKey(preset.presetKey)
    setForm(presetToForm(preset))
  }

  const createSuggested = async () => {
    const suggested: RuntimePresetCreate[] = [
      { ...EMPTY_PRESET, isDefault: true, priority: 20 },
      { ...EMPTY_PRESET, presetKey: "32k", label: "Ligero (32k)", contextTokens: 32768, priority: 10 },
      { ...EMPTY_PRESET, presetKey: "128k", label: "Alto (128k)", contextTokens: 131072, priority: 30 },
      { ...EMPTY_PRESET, presetKey: "max", label: "Máximo efectivo", contextMode: "max", contextTokens: null, priority: 40 },
    ]
    setBusy(true)
    try {
      for (const preset of suggested) {
        await api.profiles.createRuntimePreset(profile.profileId, preset)
      }
      await refresh()
      onChanged()
      toast.success("Creados 32k, 64k/p2, 128k y MAX")
    } catch (error) {
      await refresh()
      toast.error(error instanceof Error ? error.message : "No se pudieron crear todos los presets")
    } finally {
      setBusy(false)
    }
  }

  const savePreset = async () => {
    if (form.contextMode === "fixed" && (!form.contextTokens || form.contextTokens < 1)) {
      toast.error("El contexto fijo debe ser mayor que cero")
      return
    }
    if (form.placementPolicy !== "auto" && !form.preferredGpus?.length) {
      toast.error("Indica al menos una GPU para esta política de colocación")
      return
    }
    setBusy(true)
    try {
      const payload: RuntimePresetCreate = {
        ...form,
        contextTokens: form.contextMode === "max" ? null : form.contextTokens,
        preferredGpus: form.placementPolicy === "auto" ? null : form.preferredGpus,
      }
      if (editingKey) {
        const { presetKey: _ignored, ...update } = payload
        await api.profiles.updateRuntimePreset(profile.profileId, editingKey, update)
      } else {
        await api.profiles.createRuntimePreset(profile.profileId, payload)
      }
      await refresh()
      setEditingKey(null)
      onChanged()
      toast.success("Preset guardado; se aplicará en la siguiente carga")
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "No se pudo guardar el preset")
    } finally {
      setBusy(false)
    }
  }

  const patchPreset = async (preset: RuntimePreset, patch: Partial<RuntimePresetCreate>) => {
    setBusy(true)
    try {
      await api.profiles.updateRuntimePreset(profile.profileId, preset.presetKey, patch)
      await refresh()
      onChanged()
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "No se pudo actualizar el preset")
    } finally {
      setBusy(false)
    }
  }

  const deletePreset = async (preset: RuntimePreset) => {
    setBusy(true)
    try {
      await api.profiles.deleteRuntimePreset(profile.profileId, preset.presetKey)
      await refresh()
      onChanged()
      toast.success("Preset eliminado")
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "No se pudo eliminar el preset")
    } finally {
      setBusy(false)
    }
  }

  const benchmarkPreset = async (preset: RuntimePreset) => {
    setBenchmarking(preset.presetKey)
    try {
      const result = await api.profiles.benchmarkRuntimePreset(
        profile.profileId,
        preset.presetKey,
        true,
      )
      setBenchmarks((current) => ({ ...current, [preset.presetKey]: result }))
      toast.success(`Medición ${preset.label} completada`)
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "No se pudo medir el preset")
    } finally {
      setBenchmarking(null)
    }
  }

  const savePolicy = async () => {
    setBusy(true)
    try {
      await api.profiles.update(profile.profileId, { runtimePolicy: policy })
      onChanged()
      toast.success("Política de selección guardada")
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "No se pudo guardar la política")
    } finally {
      setBusy(false)
    }
  }

  return (
    <section className="space-y-3 rounded-md border border-border p-3">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <h3 className="text-sm font-medium">Contexto y ejecución</h3>
          <p className="mt-0.5 text-[11px] text-muted-foreground">
            Un modelo público con variantes internas. MAX se calcula para el modelo y el hardware disponible.
          </p>
        </div>
        <div className="flex gap-2">
          {presets.length > 0 && <button type="button" onClick={() => void resolveCapacity()} disabled={busy || resolving} className="rounded-md border border-border px-2.5 py-1.5 text-xs hover:bg-muted disabled:opacity-50">{resolving ? "Calculando…" : "Calcular capacidad"}</button>}
          {presets.length === 0 && <button type="button" onClick={() => void createSuggested()} disabled={busy} className="rounded-md border border-blue-500/40 bg-blue-500/10 px-2.5 py-1.5 text-xs text-blue-200 hover:bg-blue-500/20 disabled:opacity-50">Crear 32/64/128/MAX</button>}
          <button type="button" onClick={startCreate} disabled={busy || editingKey !== null} className="flex items-center gap-1 rounded-md border border-border px-2.5 py-1.5 text-xs hover:bg-muted disabled:opacity-50">
            <Plus size={13} /> Añadir preset
          </button>
        </div>
      </div>

      {presets.length === 0 && editingKey === null ? (
        <div className="rounded-md border border-dashed border-border px-3 py-4 text-center text-xs text-muted-foreground">
          Sin presets: el perfil conserva el comportamiento de carga actual.
        </div>
      ) : (
        <div className="space-y-2">
          {presets.map((preset) => (
            <div key={preset.presetKey} className={`rounded-md border px-3 py-2 ${preset.enabled ? "border-border" : "border-border/50 opacity-60"}`}>
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex min-w-0 flex-wrap items-center gap-2">
                  <span className="text-sm font-medium">{preset.label}</span>
                  <span className="rounded bg-muted px-1.5 py-0.5 font-mono text-[11px]">{formatContext(preset)}</span>
                  {resolutions[preset.presetKey]?.effectiveContext != null && (
                    <span className={`rounded px-1.5 py-0.5 font-mono text-[10px] ${resolutions[preset.presetKey].fits ? "bg-emerald-500/10 text-emerald-300" : "bg-amber-500/10 text-amber-300"}`}>
                      → {formatTokenCount(resolutions[preset.presetKey].effectiveContext)} · GPU {resolutions[preset.presetKey].targetGpus.join(", ") || "?"}
                    </span>
                  )}
                  <span className="text-[11px] text-muted-foreground">
                    {resolutions[preset.presetKey] && resolutions[preset.presetKey].effectiveParallelism !== preset.parallelism
                      ? `p${preset.parallelism} solicitado → p${resolutions[preset.presetKey].effectiveParallelism} efectivo`
                      : `p${preset.parallelism}`}
                  </span>
                  <span className="text-[11px] text-muted-foreground">KV {preset.kvCacheDtype}</span>
                  <span className="text-[11px] text-muted-foreground">{preset.executionPolicy}</span>
                  {preset.isDefault && (
                    <span
                      className="rounded border border-blue-500/40 bg-blue-500/10 px-1.5 py-0.5 text-[10px] text-blue-300"
                      title="El preset por defecto hereda la load policy del modelo base; cualquier otro preset siempre carga on_demand"
                    >
                      Predeterminado
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-1">
                  {preset.enabled && <button type="button" disabled={busy || benchmarking !== null} onClick={() => void benchmarkPreset(preset)} className="rounded border border-border px-2 py-1 text-[11px] hover:bg-muted disabled:opacity-40" title="Descarga este runtime, mide carga en frío y generación">{benchmarking === preset.presetKey ? "Midiendo…" : "Medir frío"}</button>}
                  {!preset.isDefault && preset.enabled && <button type="button" disabled={busy} onClick={() => void patchPreset(preset, { isDefault: true })} className="rounded border border-border p-1 hover:bg-muted" title="Usar por defecto"><Check size={13} /></button>}
                  <button type="button" disabled={busy || preset.isDefault} onClick={() => void patchPreset(preset, { enabled: !preset.enabled })} className="rounded border border-border px-2 py-1 text-[11px] hover:bg-muted disabled:opacity-40">{preset.enabled ? "Desactivar" : "Activar"}</button>
                  <button type="button" disabled={busy || editingKey !== null} onClick={() => startEdit(preset)} className="rounded border border-border p-1 hover:bg-muted disabled:opacity-40" title="Editar"><Pencil size={13} /></button>
                  <button type="button" disabled={busy || (preset.isDefault && presets.length > 1)} onClick={() => void deletePreset(preset)} className="rounded border border-red-500/40 p-1 text-red-300 hover:bg-red-500/10 disabled:opacity-40" title="Eliminar"><Trash2 size={13} /></button>
                </div>
              </div>
              <div className="mt-1 text-[10px] text-muted-foreground">
                {preset.presetKey} · prioridad {preset.priority} · colocación {preset.placementPolicy}{preset.preferredGpus?.length ? ` [GPU ${preset.preferredGpus.join(", ")}]` : ""}
              </div>
              {resolutions[preset.presetKey]?.warnings.map((warning) => (
                <div key={warning} className="mt-1 text-[10px] text-amber-300">{warning}</div>
              ))}
              {benchmarks[preset.presetKey] && (
                <div className="mt-1 text-[10px] text-emerald-300">
                  Carga {benchmarks[preset.presetKey].loadMs.toLocaleString()} ms · generación {benchmarks[preset.presetKey].generationMs.toLocaleString()} ms
                  {benchmarks[preset.presetKey].tokensPerSecond != null ? ` · ${benchmarks[preset.presetKey].tokensPerSecond} t/s` : ""}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {editingKey !== null && (
        <div className="space-y-3 rounded-md border border-blue-500/30 bg-blue-500/5 p-3">
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <label className="text-xs">Clave<input value={form.presetKey} disabled={editingKey !== ""} onChange={(event) => setForm({ ...form, presetKey: event.target.value.toLowerCase().replace(/[^a-z0-9-]/g, "") })} className="mt-1 w-full rounded border border-border bg-background px-2 py-1.5 disabled:opacity-50" /></label>
            <label className="text-xs sm:col-span-2">Nombre<input value={form.label} onChange={(event) => setForm({ ...form, label: event.target.value })} className="mt-1 w-full rounded border border-border bg-background px-2 py-1.5" /></label>
            <label className="text-xs">Prioridad<input type="number" min={0} value={form.priority} onChange={(event) => setForm({ ...form, priority: Number(event.target.value) })} className="mt-1 w-full rounded border border-border bg-background px-2 py-1.5" /></label>
            <label className="text-xs">Modo de contexto<select value={form.contextMode} onChange={(event) => setForm({ ...form, contextMode: event.target.value as "fixed" | "max", contextTokens: event.target.value === "max" ? null : (form.contextTokens ?? 65536) })} className="mt-1 w-full rounded border border-border bg-background px-2 py-1.5"><option value="fixed">Fijo</option><option value="max">MAX calculado</option></select></label>
            <label className="text-xs">Tokens<input type="number" min={1} disabled={form.contextMode === "max"} value={form.contextTokens ?? ""} onChange={(event) => setForm({ ...form, contextTokens: Number(event.target.value) })} className="mt-1 w-full rounded border border-border bg-background px-2 py-1.5 disabled:opacity-50" /></label>
            <label className="text-xs">Paralelismo<input type="number" min={1} max={64} value={form.parallelism} onChange={(event) => setForm({ ...form, parallelism: Number(event.target.value) })} className="mt-1 w-full rounded border border-border bg-background px-2 py-1.5" /></label>
            <label className="text-xs">KV cache<select value={form.kvCacheDtype} onChange={(event) => setForm({ ...form, kvCacheDtype: event.target.value as RuntimePresetCreate["kvCacheDtype"] })} className="mt-1 w-full rounded border border-border bg-background px-2 py-1.5">{(["q8", "fp8", "q4", "fp16", "bf16"] as const).map((value) => <option key={value}>{value}</option>)}</select></label>
            <label className="text-xs">Ejecución<select value={form.executionPolicy} onChange={(event) => setForm({ ...form, executionPolicy: event.target.value as RuntimePresetCreate["executionPolicy"] })} className="mt-1 w-full rounded border border-border bg-background px-2 py-1.5"><option value="auto">Auto</option><option value="local">Local</option><option value="remote">Remoto</option></select></label>
            <label className="text-xs">Colocación<select value={form.placementPolicy} onChange={(event) => setForm({ ...form, placementPolicy: event.target.value as RuntimePresetCreate["placementPolicy"] })} className="mt-1 w-full rounded border border-border bg-background px-2 py-1.5"><option value="auto">Auto</option><option value="preferred">Preferida</option><option value="forced">Forzada</option></select></label>
          </div>
          <p className="text-[11px] text-muted-foreground">
            La política de carga (pin/warm/on_demand) se configura en el modelo base, no por preset —
            el preset marcado "Predeterminado" la hereda; cualquier otro preset siempre carga on_demand.
          </p>

          <div className={form.placementPolicy === "auto" ? "opacity-50" : ""}>
            <span className="text-xs">GPU</span>
            <div className="mt-1 flex flex-wrap gap-2">
              {gpus.map((gpu) => {
                const selected = form.preferredGpus?.includes(gpu.index) ?? false
                return (
                  <button
                    key={gpu.index}
                    type="button"
                    disabled={form.placementPolicy === "auto"}
                    onClick={() => toggleGpu(gpu.index)}
                    className={`rounded-md border px-2.5 py-1.5 text-xs transition-colors disabled:cursor-not-allowed ${
                      selected
                        ? "border-blue-500/60 bg-blue-500/15 text-blue-200"
                        : "border-border hover:bg-muted"
                    }`}
                  >
                    GPU {gpu.index} · {gpu.name} ({(gpu.totalVramMb / 1024).toFixed(0)}GB)
                  </button>
                )
              })}
              {gpus.length === 0 && (
                <span className="text-[11px] text-muted-foreground">Sin GPUs detectadas</span>
              )}
            </div>
          </div>

          <LivePreviewPanel loading={livePreviewLoading} preview={livePreview} gpus={gpus} />

          <label className="flex items-center gap-2 text-xs"><input type="checkbox" checked={form.enabled} disabled={form.isDefault} onChange={(event) => setForm({ ...form, enabled: event.target.checked })} /> Activo</label>
          <div className="flex justify-end gap-2">
            <button type="button" onClick={() => setEditingKey(null)} className="flex items-center gap-1 rounded border border-border px-2 py-1.5 text-xs hover:bg-muted"><X size={13} /> Cancelar</button>
            <button type="button" disabled={busy || !form.presetKey || !form.label} onClick={() => void savePreset()} className="rounded bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground disabled:opacity-50">Guardar preset</button>
          </div>
        </div>
      )}

      <div className="border-t border-border pt-2">
        <button type="button" onClick={() => setPolicyOpen(!policyOpen)} className="flex w-full items-center justify-between text-xs font-medium">Política de cambio de contexto{policyOpen ? <ChevronUp size={14} /> : <ChevronDown size={14} />}</button>
        {policyOpen && (
          <div className="mt-3 space-y-3">
            <div className="grid gap-3 sm:grid-cols-4">
              <label className="text-xs">Selección<select value={policy.selection} onChange={(event) => setPolicy({ ...policy, selection: event.target.value as RuntimePolicy["selection"] })} className="mt-1 w-full rounded border border-border bg-background px-2 py-1.5"><option value="prefer_loaded">Reutilizar cargado</option><option value="prefer_default">Preferir predeterminado</option></select></label>
              <label className="text-xs">Espera antes de bajar (s)<input type="number" min={0} max={86400} value={policy.downgradeGraceSeconds} onChange={(event) => setPolicy({ ...policy, downgradeGraceSeconds: Number(event.target.value) })} className="mt-1 w-full rounded border border-border bg-background px-2 py-1.5" /></label>
              <label className="text-xs">Residencia mínima (s)<input type="number" min={0} max={86400} value={policy.minimumResidencySeconds} onChange={(event) => setPolicy({ ...policy, minimumResidencySeconds: Number(event.target.value) })} className="mt-1 w-full rounded border border-border bg-background px-2 py-1.5" /></label>
              <label className="text-xs">Máximo validado (tokens)<input type="number" min={1} placeholder="Sin límite adicional" value={policy.maxContextTokens ?? ""} onChange={(event) => setPolicy({ ...policy, maxContextTokens: event.target.value ? Number(event.target.value) : null })} className="mt-1 w-full rounded border border-border bg-background px-2 py-1.5" /></label>
            </div>
            <div className="flex flex-wrap gap-4 text-xs">
              <label className="flex items-center gap-2"><input type="checkbox" checked={policy.restoreDefaultWhenIdle} onChange={(event) => setPolicy({ ...policy, restoreDefaultWhenIdle: event.target.checked })} /> Volver al predeterminado al quedar inactivo</label>
              <label className="flex items-center gap-2"><input type="checkbox" checked={policy.smallRequestsExtendLargeTtl} onChange={(event) => setPolicy({ ...policy, smallRequestsExtendLargeTtl: event.target.checked })} /> Peticiones pequeñas prolongan el contexto alto</label>
            </div>
            <p className="text-[10px] text-muted-foreground">Una petición pequeña reutiliza el runner grande cargado, pero por defecto no prolonga su permanencia. El máximo validado limita MAX a una capacidad comprobada sin convertirlo en un preset fijo.</p>
            <div className="flex justify-end"><button type="button" disabled={busy} onClick={() => void savePolicy()} className="rounded border border-border px-3 py-1.5 text-xs hover:bg-muted disabled:opacity-50">Guardar política</button></div>
          </div>
        )}
      </div>
    </section>
  )
}
