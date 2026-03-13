import type { ReactNode } from "react"
import { useEffect, useState } from "react"
import * as Dialog from "@radix-ui/react-dialog"
import { X } from "lucide-react"
import { ScheduleEditor } from "@/components/models/ScheduleEditor"
import type { EvictionSchedule, GPUState, LoadPolicy, ModelState, VLLMConfig } from "@/types"

interface ModelConfigModalProps {
  model: ModelState | null
  gpus: GPUState[]
  open: boolean
  onOpenChange: (open: boolean) => void
  onSave: (modelId: string, patch: {
    loadPolicy: LoadPolicy
    preferredGpu: number | null
    autoReload: boolean
    schedules: EvictionSchedule[]
    extraConfig?: {
      vllm?: VLLMConfig
    }
  }) => Promise<void>
}

function FieldHint({ children }: { children: string }) {
  return <p className="mt-1 text-xs text-muted-foreground">{children}</p>
}

function FieldSection({ title, description, children }: { title: string; description: string; children: ReactNode }) {
  return (
    <section className="space-y-4 rounded-lg border border-border/70 bg-background/40 p-4">
      <div>
        <h3 className="text-sm font-medium">{title}</h3>
        <p className="text-xs text-muted-foreground">{description}</p>
      </div>
      {children}
    </section>
  )
}

export function ModelConfigModal({ model, gpus, open, onOpenChange, onSave }: ModelConfigModalProps) {
  const [loadPolicy, setLoadPolicy] = useState<LoadPolicy>("on_demand")
  const [preferredGpu, setPreferredGpu] = useState<number | null>(null)
  const [autoReload, setAutoReload] = useState(false)
  const [schedules, setSchedules] = useState<EvictionSchedule[]>([])
  const [tensorParallelSize, setTensorParallelSize] = useState("")
  const [maxModelLen, setMaxModelLen] = useState("")
  const [maxNumSeqs, setMaxNumSeqs] = useState("")
  const [maxNumBatchedTokens, setMaxNumBatchedTokens] = useState("")
  const [gpuMemoryUtilization, setGpuMemoryUtilization] = useState("")
  const [enablePrefixCaching, setEnablePrefixCaching] = useState(true)
  const [trustRemoteCode, setTrustRemoteCode] = useState(false)
  const [enableChunkedPrefillMode, setEnableChunkedPrefillMode] = useState<"inherit" | "on" | "off">("inherit")
  const [kvCacheDtype, setKvCacheDtype] = useState("")
  const [swapSpace, setSwapSpace] = useState("")
  const [enforceEager, setEnforceEager] = useState(false)
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    if (!model) return
    setLoadPolicy(model.loadPolicy)
    setPreferredGpu(model.preferredGpu)
    setAutoReload(model.autoReload)
    setSchedules(model.schedules ?? [])
    const vllm = (model.extraConfig?.vllm as Record<string, unknown> | undefined) ?? {}
    setTensorParallelSize(vllm.tensorParallelSize == null ? "" : String(vllm.tensorParallelSize))
    setMaxModelLen(vllm.maxModelLen == null ? "" : String(vllm.maxModelLen))
    setMaxNumSeqs(vllm.maxNumSeqs == null ? "" : String(vllm.maxNumSeqs))
    setMaxNumBatchedTokens(vllm.maxNumBatchedTokens == null ? "" : String(vllm.maxNumBatchedTokens))
    setGpuMemoryUtilization(vllm.gpuMemoryUtilization == null ? "" : String(vllm.gpuMemoryUtilization))
    setEnablePrefixCaching(Boolean(vllm.enablePrefixCaching))
    setTrustRemoteCode(Boolean(vllm.trustRemoteCode))
    setEnableChunkedPrefillMode(
      vllm.enableChunkedPrefill == null ? "inherit" : vllm.enableChunkedPrefill ? "on" : "off",
    )
    setKvCacheDtype(vllm.kvCacheDtype == null ? "" : String(vllm.kvCacheDtype))
    setSwapSpace(vllm.swapSpace == null ? "" : String(vllm.swapSpace))
    setEnforceEager(Boolean(vllm.enforceEager))
  }, [model])

  const handleSave = async () => {
    if (!model) return
    setSaving(true)
    try {
      await onSave(model.modelId, {
        loadPolicy,
        preferredGpu,
        autoReload,
        schedules,
        extraConfig: model.backendType === "vllm"
          ? {
              vllm: {
                tensorParallelSize: tensorParallelSize === "" ? null : Number(tensorParallelSize),
                maxModelLen: maxModelLen === "" ? null : Number(maxModelLen),
                maxNumSeqs: maxNumSeqs === "" ? null : Number(maxNumSeqs),
                maxNumBatchedTokens: maxNumBatchedTokens === "" ? null : Number(maxNumBatchedTokens),
                gpuMemoryUtilization: gpuMemoryUtilization === "" ? null : Number(gpuMemoryUtilization),
                enablePrefixCaching,
                trustRemoteCode,
                enableChunkedPrefill:
                  enableChunkedPrefillMode === "inherit"
                    ? null
                    : enableChunkedPrefillMode === "on",
                kvCacheDtype: kvCacheDtype === "" ? null : kvCacheDtype,
                swapSpace: swapSpace === "" ? null : Number(swapSpace),
                enforceEager,
              },
            }
          : undefined,
      })
      onOpenChange(false)
    } finally {
      setSaving(false)
    }
  }

  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-40 bg-black/60" />
        <Dialog.Content className="fixed left-1/2 top-1/2 z-50 max-h-[88vh] w-[95vw] max-w-3xl -translate-x-1/2 -translate-y-1/2 overflow-y-auto rounded-lg border border-border bg-card p-5">
          <div className="mb-4 flex items-start justify-between">
            <div>
              <Dialog.Title className="text-lg font-semibold">Configurar modelo</Dialog.Title>
              <Dialog.Description className="text-sm text-muted-foreground">
                {model?.displayName ?? ""}
              </Dialog.Description>
            </div>
            <Dialog.Close className="rounded-md p-1 text-muted-foreground hover:bg-muted" aria-label="Close">
              <X size={16} />
            </Dialog.Close>
          </div>

          <div className="space-y-4">
            <FieldSection
              title="General"
              description="Politica de carga, afinidad de GPU y recarga automatica."
            >
              <div className="grid gap-4 md:grid-cols-2">
                <label className="block text-sm">
                  <span className="mb-1 block text-muted-foreground">Load policy</span>
                  <select
                    value={loadPolicy}
                    onChange={(event) => setLoadPolicy(event.target.value as LoadPolicy)}
                    className="w-full rounded-md border border-border bg-background px-3 py-2"
                  >
                    <option value="pin">pin</option>
                    <option value="warm">warm</option>
                    <option value="on_demand">on_demand</option>
                  </select>
                  <FieldHint>`pin` intenta dejarlo cargado; `on_demand` prioriza liberar recursos.</FieldHint>
                </label>

                <label className="block text-sm">
                  <span className="mb-1 block text-muted-foreground">GPU preferida</span>
                  <select
                    value={preferredGpu === null ? "" : String(preferredGpu)}
                    onChange={(event) => setPreferredGpu(event.target.value === "" ? null : Number(event.target.value))}
                    className="w-full rounded-md border border-border bg-background px-3 py-2"
                  >
                    <option value="">Default del servidor</option>
                    {gpus.map((gpu) => (
                      <option key={gpu.index} value={gpu.index}>
                        GPU {gpu.index} - {gpu.name}
                      </option>
                    ))}
                  </select>
                  <FieldHint>Fija una GPU preferida si quieres evitar que el scheduler elija otra.</FieldHint>
                </label>
              </div>

              <label className="inline-flex items-center gap-2 text-sm text-muted-foreground">
                <input
                  type="checkbox"
                  checked={autoReload}
                  onChange={(event) => setAutoReload(event.target.checked)}
                />
                auto reload tras eviction por presion
              </label>
            </FieldSection>

            {model?.backendType === "vllm" && (
              <>
                <FieldSection
                  title="vLLM Basico"
                  description="Parametros con impacto directo en compatibilidad, reparto entre GPUs y limite de contexto."
                >
                  <div className="grid gap-4 md:grid-cols-2">
                    <label className="block text-sm">
                      <span className="mb-1 block text-muted-foreground">Tensor parallel size</span>
                      <input
                        type="number"
                        min="1"
                        value={tensorParallelSize}
                        onChange={(event) => setTensorParallelSize(event.target.value)}
                        placeholder="heredar / auto"
                        className="w-full rounded-md border border-border bg-background px-3 py-2"
                      />
                      <FieldHint>Cuantas GPUs reparte el modelo. Si lo dejas vacio, `oCabra` usa lo que asigne el scheduler.</FieldHint>
                    </label>

                    <label className="block text-sm">
                      <span className="mb-1 block text-muted-foreground">Max model len</span>
                      <input
                        type="number"
                        min="1"
                        value={maxModelLen}
                        onChange={(event) => setMaxModelLen(event.target.value)}
                        placeholder="usar el del modelo"
                        className="w-full rounded-md border border-border bg-background px-3 py-2"
                      />
                      <FieldHint>Bajarlo ayuda a que un modelo quepa y reduce presion de KV cache.</FieldHint>
                    </label>
                  </div>

                  <div className="grid gap-3 md:grid-cols-2">
                    <label className="inline-flex items-center gap-2 text-sm text-muted-foreground">
                      <input
                        type="checkbox"
                        checked={enablePrefixCaching}
                        onChange={(event) => setEnablePrefixCaching(event.target.checked)}
                      />
                      prefix caching
                    </label>

                    <label className="inline-flex items-center gap-2 text-sm text-muted-foreground">
                      <input
                        type="checkbox"
                        checked={trustRemoteCode}
                        onChange={(event) => setTrustRemoteCode(event.target.checked)}
                      />
                      trust remote code
                    </label>
                  </div>
                </FieldSection>

                <FieldSection
                  title="Rendimiento"
                  description="Toca concurrencia, throughput y cuanta VRAM reserva vLLM."
                >
                  <div className="grid gap-4 md:grid-cols-2">
                    <label className="block text-sm">
                      <span className="mb-1 block text-muted-foreground">GPU memory utilization</span>
                      <input
                        type="number"
                        min="0"
                        max="1"
                        step="0.01"
                        value={gpuMemoryUtilization}
                        onChange={(event) => setGpuMemoryUtilization(event.target.value)}
                        placeholder="heredar global"
                        className="w-full rounded-md border border-border bg-background px-3 py-2"
                      />
                      <FieldHint>Normalmente `0.85-0.90`. Mas alto da mas KV cache, pero acerca el riesgo de OOM.</FieldHint>
                    </label>

                    <label className="block text-sm">
                      <span className="mb-1 block text-muted-foreground">Max concurrent sequences</span>
                      <input
                        type="number"
                        min="1"
                        value={maxNumSeqs}
                        onChange={(event) => setMaxNumSeqs(event.target.value)}
                        placeholder="heredar global"
                        className="w-full rounded-md border border-border bg-background px-3 py-2"
                      />
                      <FieldHint>Limita cuantas peticiones procesa por iteracion. Subirlo mejora concurrencia pero usa mas memoria.</FieldHint>
                    </label>

                    <label className="block text-sm">
                      <span className="mb-1 block text-muted-foreground">Max batched tokens</span>
                      <input
                        type="number"
                        min="1"
                        value={maxNumBatchedTokens}
                        onChange={(event) => setMaxNumBatchedTokens(event.target.value)}
                        placeholder="heredar global"
                        className="w-full rounded-md border border-border bg-background px-3 py-2"
                      />
                      <FieldHint>Es el knob principal de latencia vs throughput. Alto favorece throughput; bajo favorece respuesta mas reactiva.</FieldHint>
                    </label>

                    <label className="block text-sm">
                      <span className="mb-1 block text-muted-foreground">Chunked prefill</span>
                      <select
                        value={enableChunkedPrefillMode}
                        onChange={(event) => setEnableChunkedPrefillMode(event.target.value as "inherit" | "on" | "off")}
                        className="w-full rounded-md border border-border bg-background px-3 py-2"
                      >
                        <option value="inherit">heredar / auto</option>
                        <option value="on">forzar on</option>
                        <option value="off">forzar off</option>
                      </select>
                      <FieldHint>Ayuda con prompts largos y mezcla de prefills/decodes. Mejor dejarlo en auto salvo que estes midiendo.</FieldHint>
                    </label>
                  </div>
                </FieldSection>

                <FieldSection
                  title="Avanzado"
                  description="Escapes de compatibilidad y memoria. Tocarlos sin medir puede empeorar el servicio."
                >
                  <div className="grid gap-4 md:grid-cols-2">
                    <label className="block text-sm">
                      <span className="mb-1 block text-muted-foreground">KV cache dtype</span>
                      <select
                        value={kvCacheDtype}
                        onChange={(event) => setKvCacheDtype(event.target.value)}
                        className="w-full rounded-md border border-border bg-background px-3 py-2"
                      >
                        <option value="">heredar / auto</option>
                        <option value="fp8">fp8</option>
                      </select>
                      <FieldHint>`fp8` puede aumentar capacidad de contexto/KV cache, pero no siempre compensa en estabilidad.</FieldHint>
                    </label>

                    <label className="block text-sm">
                      <span className="mb-1 block text-muted-foreground">Swap space (GiB)</span>
                      <input
                        type="number"
                        min="0"
                        step="0.5"
                        value={swapSpace}
                        onChange={(event) => setSwapSpace(event.target.value)}
                        placeholder="default de vLLM"
                        className="w-full rounded-md border border-border bg-background px-3 py-2"
                      />
                      <FieldHint>CPU offload por GPU. Ultimo recurso si falta memoria; suele penalizar bastante el rendimiento.</FieldHint>
                    </label>
                  </div>

                  <label className="inline-flex items-center gap-2 text-sm text-muted-foreground">
                    <input
                      type="checkbox"
                      checked={enforceEager}
                      onChange={(event) => setEnforceEager(event.target.checked)}
                    />
                    enforce eager
                  </label>
                  <FieldHint>Desactiva CUDA graphs. Util solo como escape hatch de estabilidad.</FieldHint>
                </FieldSection>
              </>
            )}

            <ScheduleEditor value={schedules} onChange={setSchedules} />
          </div>

          <div className="mt-5 flex justify-end gap-2">
            <Dialog.Close className="rounded-md border border-border px-3 py-2 text-sm hover:bg-muted">
              Cancelar
            </Dialog.Close>
            <button
              type="button"
              onClick={() => void handleSave()}
              disabled={saving}
              className="rounded-md bg-primary px-3 py-2 text-sm font-medium text-primary-foreground disabled:opacity-50"
            >
              {saving ? "Guardando..." : "Guardar"}
            </button>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  )
}
