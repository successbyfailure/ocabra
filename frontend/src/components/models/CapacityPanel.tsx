import { useCallback, useEffect, useMemo, useState } from "react"
import { api } from "@/api/client"
import { AreaSpark } from "@/components/gpu/AreaSpark"
import type { BackendExtraConfig, GPUState, ModelCapacity } from "@/types"

interface CapacityPanelProps {
  modelId: string
  gpus: GPUState[]
  extraConfig?: BackendExtraConfig
  onSaved?: () => void
  onUseCaseChange?: (active: boolean) => void
}

const KV_DTYPES = ["fp16", "fp8", "q4"] as const
const SLOT_CHOICES = [1, 2, 4, 8]

const fmtTokens = (n: number): string => {
  if (n >= 1000) return `${(n / 1000).toFixed(n >= 100000 ? 0 : 1)}k`
  return String(n)
}
const gb = (mb: number) => (mb / 1024).toFixed(1)

// Reads any use_case block already stored on the model.
function readUseCase(extra?: BackendExtraConfig): { context: string; slots: number; kvDtype: string } {
  const uc = (extra?.use_case ?? {}) as Record<string, unknown>
  const ctx = uc.context
  return {
    context: ctx === undefined || ctx === null || ctx === "max" ? "" : String(ctx),
    slots: typeof uc.slots === "number" ? uc.slots : 1,
    kvDtype: typeof uc.kv_dtype === "string" ? uc.kv_dtype : "fp16",
  }
}

function Chip({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-border bg-muted/40 px-2.5 py-1.5">
      <div className="text-[10px] uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="text-[13px] font-semibold tabular-nums text-foreground">{value}</div>
    </div>
  )
}

export function CapacityPanel({ modelId, gpus, extraConfig, onSaved, onUseCaseChange }: CapacityPanelProps) {
  const stored = useMemo(() => readUseCase(extraConfig), [extraConfig])
  const [active, setActive] = useState<boolean>(Boolean((extraConfig as Record<string, unknown> | undefined)?.use_case))
  const [cap, setCap] = useState<ModelCapacity | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [gpu, setGpu] = useState<number | undefined>(undefined)
  const [kvDtype, setKvDtype] = useState<string>(stored.kvDtype)
  const [slots, setSlots] = useState<number>(stored.slots)
  const [ctxMode, setCtxMode] = useState<"max" | "custom">(stored.context ? "custom" : "max")
  const [ctxValue, setCtxValue] = useState<string>(stored.context || "")
  const [validation, setValidation] = useState<ModelCapacity["validation"] | null>(null)
  const [saving, setSaving] = useState(false)
  const [savedMsg, setSavedMsg] = useState<string | null>(null)

  const loadCapacity = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await api.models.capacity(modelId, { gpu, slots: "1,2,4", kvDtype })
      setCap(data)
    } catch (e) {
      setError(e instanceof Error ? e.message : "Error")
    } finally {
      setLoading(false)
    }
  }, [modelId, gpu, kvDtype])

  useEffect(() => {
    void loadCapacity()
  }, [loadCapacity])

  // Live validation of the chosen use case (debounced).
  useEffect(() => {
    if (!cap?.applicable) return
    const context = ctxMode === "max" ? "max" : ctxValue
    if (ctxMode === "custom" && !ctxValue) {
      setValidation(null)
      return
    }
    const t = setTimeout(async () => {
      try {
        const data = await api.models.capacity(modelId, { gpu, slots: String(slots), kvDtype, context })
        setValidation(data.validation ?? null)
      } catch {
        setValidation(null)
      }
    }, 350)
    return () => clearTimeout(t)
  }, [modelId, gpu, slots, kvDtype, ctxMode, ctxValue, cap?.applicable])

  const save = async () => {
    setSaving(true)
    setSavedMsg(null)
    try {
      const nextExtra: Record<string, unknown> = { ...(extraConfig ?? {}) }
      nextExtra.use_case = {
        context: ctxMode === "max" ? "max" : Number(ctxValue),
        slots,
        kv_dtype: kvDtype,
      }
      await api.models.patch(modelId, { extraConfig: nextExtra as BackendExtraConfig })
      setSavedMsg("Caso de uso aplicado. Se usará en la próxima carga del modelo.")
      setActive(true)
      onUseCaseChange?.(true)
      onSaved?.()
    } catch (e) {
      setSavedMsg(e instanceof Error ? e.message : "Error al guardar")
    } finally {
      setSaving(false)
    }
  }

  const clear = async () => {
    setSaving(true)
    setSavedMsg(null)
    try {
      const nextExtra: Record<string, unknown> = { ...(extraConfig ?? {}) }
      delete nextExtra.use_case
      await api.models.patch(modelId, { extraConfig: nextExtra as BackendExtraConfig })
      setSavedMsg("Caso de uso eliminado. Vuelven a mandar los knobs manuales.")
      setActive(false)
      onUseCaseChange?.(false)
      onSaved?.()
    } catch (e) {
      setSavedMsg(e instanceof Error ? e.message : "Error al quitar")
    } finally {
      setSaving(false)
    }
  }

  const concurrencyLabel = cap?.concurrency_label === "concurrency" ? "secuencias" : "slots"
  const native = cap?.arch?.native_context ?? 0

  if (cap && !cap.applicable) {
    return (
      <div className="rounded-lg border border-border bg-muted/30 px-3 py-2.5 text-[12px] text-muted-foreground">
        Este backend no tiene KV cache que escale con el contexto, así que no aplica el planificador.
        {cap.note ? <div className="mt-1 text-[11px] opacity-80">{cap.note}</div> : null}
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-3.5">
      {/* Controls: GPU, KV dtype */}
      <div className="flex flex-wrap items-end gap-3">
        {gpus.length > 1 && (
          <label className="text-[12px]">
            <span className="mb-1 block text-muted-foreground">GPU</span>
            <select
              className="rounded-md border border-border bg-background px-2 py-1 text-[12px]"
              value={gpu ?? cap?.target_gpu ?? 0}
              onChange={(e) => setGpu(Number(e.target.value))}
            >
              {gpus.map((g) => (
                <option key={g.index} value={g.index}>
                  #{g.index} · {g.name} ({gb(g.totalVramMb)} GB)
                </option>
              ))}
            </select>
          </label>
        )}
        <label className="text-[12px]">
          <span className="mb-1 block text-muted-foreground">KV cache</span>
          <select
            className="rounded-md border border-border bg-background px-2 py-1 text-[12px]"
            value={kvDtype}
            onChange={(e) => setKvDtype(e.target.value)}
          >
            {KV_DTYPES.map((k) => (
              <option key={k} value={k}>
                {k}
              </option>
            ))}
          </select>
        </label>
      </div>

      {error && <div className="text-[12px] text-red-500">{error}</div>}
      {loading && !cap && <div className="text-[12px] text-muted-foreground">Calculando capacidad…</div>}

      {cap?.applicable && cap.arch && (
        <>
          {/* Architecture summary */}
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            <Chip label="Pesos" value={`${gb(cap.weights_mb)} GB`} />
            <Chip label="KV / 1k tok" value={`${cap.arch.kv_mb_per_1k_tokens} MB`} />
            <Chip label="Ctx nativo" value={fmtTokens(native)} />
            <Chip label="Capas · KV heads" value={`${cap.arch.layers} · ${cap.arch.n_kv_heads}`} />
          </div>

          {/* Capacity table */}
          <div>
            <div className="mb-1.5 text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
              Contexto máximo por {concurrencyLabel} · GPU #{cap.target_gpu}
            </div>
            <div className="flex flex-col gap-1.5">
              {cap.capacity?.map((r) => {
                const width = native ? Math.max(2, Math.min(100, (r.max_context_capped / native) * 100)) : 100
                return (
                  <div key={r.slots} className="flex items-center gap-2.5">
                    <span className="w-8 shrink-0 text-right text-[12px] tabular-nums text-muted-foreground">
                      {r.slots}×
                    </span>
                    <div className="h-[18px] flex-1 overflow-hidden rounded bg-muted">
                      <div
                        className="flex h-full items-center justify-end rounded bg-primary/80 px-1.5 transition-all"
                        style={{ width: `${width}%` }}
                      >
                        <span className="text-[10.5px] font-semibold tabular-nums text-primary-foreground">
                          {fmtTokens(r.max_context_capped)}
                        </span>
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          </div>

          {/* VRAM curve */}
          {cap.vram_curve && cap.vram_curve.length > 1 && (
            <div>
              <div className="mb-1 flex items-baseline justify-between">
                <span className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
                  VRAM vs contexto
                </span>
                <span className="text-[10.5px] tabular-nums text-muted-foreground">
                  {fmtTokens(cap.vram_curve[0].context)} → {fmtTokens(cap.vram_curve[cap.vram_curve.length - 1].context)} tok
                </span>
              </div>
              <AreaSpark data={cap.vram_curve.map((p) => p.vram_mb)} color="#2a78d6" height={44} />
              <div className="mt-0.5 flex justify-between text-[10px] tabular-nums text-muted-foreground">
                <span>{gb(cap.vram_curve[0].vram_mb)} GB</span>
                <span>{gb(cap.vram_curve[cap.vram_curve.length - 1].vram_mb)} GB</span>
              </div>
            </div>
          )}

          {/* Use-case configurator */}
          <div className="rounded-lg border border-border bg-muted/20 p-3">
            <div className="mb-2 text-[12px] font-semibold text-foreground">Caso de uso</div>
            <div className="flex flex-wrap items-end gap-3">
              <div className="text-[12px]">
                <span className="mb-1 block text-muted-foreground">Contexto</span>
                <div className="flex items-center gap-1.5">
                  <button
                    type="button"
                    onClick={() => setCtxMode("max")}
                    className={`rounded-md border px-2 py-1 text-[12px] ${ctxMode === "max" ? "border-primary bg-primary/10 text-foreground" : "border-border text-muted-foreground"}`}
                  >
                    Máximo
                  </button>
                  <button
                    type="button"
                    onClick={() => setCtxMode("custom")}
                    className={`rounded-md border px-2 py-1 text-[12px] ${ctxMode === "custom" ? "border-primary bg-primary/10 text-foreground" : "border-border text-muted-foreground"}`}
                  >
                    Fijo
                  </button>
                  {ctxMode === "custom" && (
                    <input
                      type="number"
                      value={ctxValue}
                      onChange={(e) => setCtxValue(e.target.value)}
                      placeholder="p. ej. 16384"
                      className="w-28 rounded-md border border-border bg-background px-2 py-1 text-[12px] tabular-nums"
                    />
                  )}
                </div>
              </div>
              <label className="text-[12px]">
                <span className="mb-1 block text-muted-foreground">{concurrencyLabel}</span>
                <select
                  className="rounded-md border border-border bg-background px-2 py-1 text-[12px]"
                  value={slots}
                  onChange={(e) => setSlots(Number(e.target.value))}
                >
                  {SLOT_CHOICES.map((s) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
                </select>
              </label>
            </div>

            {validation && (
              <div className="mt-2.5 text-[12px]">
                <span className="text-muted-foreground">Efectivo: </span>
                <span className="font-semibold tabular-nums text-foreground">
                  {fmtTokens(validation.effective_context)} tok
                </span>
                {validation.fits ? (
                  <span className="ml-2 text-green-600">✓ cabe</span>
                ) : (
                  <span className="ml-2 text-amber-500">ajustado al máximo ({fmtTokens(validation.max_context)})</span>
                )}
                {validation.warnings.map((w, i) => (
                  <div key={i} className="mt-0.5 text-[11px] text-amber-500">
                    {w}
                  </div>
                ))}
              </div>
            )}

            <div className="mt-3 flex items-center gap-3">
              <button
                type="button"
                onClick={save}
                disabled={saving}
                className="rounded-md bg-primary px-3 py-1.5 text-[12px] font-semibold text-primary-foreground disabled:opacity-50"
              >
                {saving ? "Guardando…" : active ? "Actualizar caso de uso" : "Aplicar caso de uso"}
              </button>
              {active && (
                <button
                  type="button"
                  onClick={clear}
                  disabled={saving}
                  className="rounded-md border border-border px-3 py-1.5 text-[12px] font-medium text-muted-foreground disabled:opacity-50"
                >
                  Quitar
                </button>
              )}
              {savedMsg && <span className="text-[11.5px] text-muted-foreground">{savedMsg}</span>}
            </div>
          </div>
        </>
      )}
    </div>
  )
}
