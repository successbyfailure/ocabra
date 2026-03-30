/**
 * CompileModal — formulario de compilación TensorRT-LLM + progreso SSE.
 *
 * Flujo:
 *  1. Pantalla "form"  — el usuario configura GPU, dtype, tamaños y nombre del engine.
 *  2. Pantalla "progress" — progreso en tiempo real via SSE al terminar de enviar.
 */
import { useCallback, useEffect, useRef, useState } from "react"
import * as Dialog from "@radix-ui/react-dialog"
import { AlertTriangle, Check, Cpu, Loader2, X } from "lucide-react"
import { toast } from "sonner"
import { api } from "@/api/client"
import { useSSE } from "@/hooks/useSSE"
import { useGpuStore } from "@/stores/gpuStore"
import type { CompileJob, GPUState, ModelState } from "@/types"

// ── VRAM estimation ──────────────────────────────────────────────

/** Bytes per parameter for each dtype */
const DTYPE_BYTES: Record<string, number> = {
  fp16: 2,
  bf16: 2,
  int8: 1,
  fp8: 1,
}

/**
 * Estimate engine VRAM in MB from model_id.
 * Looks for patterns like "7B", "27B", "35B-A3B" (MoE active params).
 * Returns null if it can't parse.
 */
function estimateVramMb(modelId: string, dtype: string): number | null {
  // MoE: prefer active-params like "35B-A3B" → use A3B
  const moeMatch = modelId.match(/A(\d+(?:\.\d+)?)B/i)
  if (moeMatch) {
    const activeBillions = parseFloat(moeMatch[1])
    const bytesPerParam = DTYPE_BYTES[dtype] ?? 2
    return Math.ceil(activeBillions * 1e9 * bytesPerParam / 1024 / 1024 * 1.15)
  }

  const bMatch = modelId.match(/(\d+(?:\.\d+)?)B/i)
  if (!bMatch) return null
  const billions = parseFloat(bMatch[1])
  const bytesPerParam = DTYPE_BYTES[dtype] ?? 2
  // +15% overhead
  return Math.ceil(billions * 1e9 * bytesPerParam / 1024 / 1024 * 1.15)
}

// ── Default engine name ──────────────────────────────────────────

function defaultEngineName(modelId: string, dtype: string): string {
  // Strip backend prefix
  let name = modelId
  const slash = name.indexOf("/")
  if (slash !== -1) {
    const prefix = name.slice(0, slash)
    if (["vllm", "tensorrt_llm", "llama_cpp", "sglang"].includes(prefix)) {
      name = name.slice(slash + 1)
    }
  }
  // Replace / with -- (like HF convention), strip unsafe chars
  name = name.replace(/\//g, "--").replace(/[^a-zA-Z0-9._-]/g, "-")
  return `${name}-${dtype}`
}

// ── Progress bar ─────────────────────────────────────────────────

function ProgressBar({ pct }: { pct: number }) {
  return (
    <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
      <div
        className="h-full rounded-full bg-emerald-500 transition-all duration-300"
        style={{ width: `${Math.min(100, Math.max(0, pct))}%` }}
      />
    </div>
  )
}

// ── Phase label ──────────────────────────────────────────────────

function PhaseLabel({ phase, status }: { phase: CompileJob["phase"]; status: CompileJob["status"] }) {
  if (status === "done") return <span className="text-emerald-400">Completado</span>
  if (status === "failed") return <span className="text-red-400">Fallido</span>
  if (status === "cancelled") return <span className="text-amber-400">Cancelado</span>
  if (phase === "convert") return <span className="text-blue-400">Paso 1/2: Convirtiendo pesos HF → TRT-LLM…</span>
  if (phase === "build") return <span className="text-purple-400">Paso 2/2: Compilando engine…</span>
  return <span className="text-muted-foreground">En cola…</span>
}

// ── Main component ───────────────────────────────────────────────

interface CompileModalProps {
  model: ModelState | null
  open: boolean
  onOpenChange: (open: boolean) => void
  /** Called when the job finishes with status 'done' and the user clicks "Cargar ahora" */
  onLoadNow?: (modelId: string) => void
}

export function CompileModal({ model, open, onOpenChange, onLoadNow }: CompileModalProps) {
  const gpus = useGpuStore((s) => s.gpus)

  // ── Form state ────────────────────────────────────────────────
  const [gpuSelection, setGpuSelection] = useState<"single0" | "single1" | "both">("single1")
  const [dtype, setDtype] = useState("fp16")
  const [maxBatchSize, setMaxBatchSize] = useState(1)
  const [maxInputLen, setMaxInputLen] = useState(2048)
  const [maxSeqLen, setMaxSeqLen] = useState(4096)
  const [engineName, setEngineName] = useState("")
  const [submitting, setSubmitting] = useState(false)

  // ── Progress state ────────────────────────────────────────────
  const [screen, setScreen] = useState<"form" | "progress">("form")
  const [job, setJob] = useState<CompileJob | null>(null)
  const [logLines, setLogLines] = useState<string[]>([])
  const logRef = useRef<HTMLDivElement>(null)

  // Reset when model changes or modal opens
  useEffect(() => {
    if (open && model) {
      setScreen("form")
      setJob(null)
      setLogLines([])
      setDtype("fp16")
      setMaxBatchSize(1)
      setMaxInputLen(2048)
      setMaxSeqLen(4096)
      setEngineName(defaultEngineName(model.modelId, "fp16"))
      // Pick best single GPU default: prefer GPU 1 (3090) if available
      const hasGpu1 = gpus.some((g) => g.index === 1)
      setGpuSelection(hasGpu1 ? "single1" : "single0")
    }
  }, [open, model, gpus])

  // Update engine name default when dtype changes (only if user hasn't edited it)
  useEffect(() => {
    if (model) {
      setEngineName((prev) => {
        const expected = defaultEngineName(model.modelId, dtype)
        // Only auto-update if it still looks like the auto-generated name
        const otherDtypes = ["fp16", "bf16", "int8", "fp8"].filter((d) => d !== dtype)
        const wasAuto = otherDtypes.some((d) => prev === defaultEngineName(model.modelId, d))
        return wasAuto ? expected : prev
      })
    }
  }, [dtype, model])

  // Auto-scroll log
  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight
    }
  }, [logLines])

  // ── GPU resolution ────────────────────────────────────────────
  const gpuIndices = gpuSelection === "both" ? [0, 1] : gpuSelection === "single0" ? [0] : [1]

  // ── VRAM estimation ───────────────────────────────────────────
  const vramEst = model ? estimateVramMb(model.modelId, dtype) : null

  function getGpuVram(idx: number): number {
    return gpus.find((g) => g.index === idx)?.totalVramMb ?? 0
  }

  function gpuLabel(g: GPUState) {
    const vramGb = (g.totalVramMb / 1024).toFixed(0)
    return `GPU ${g.index} — ${g.name} (${vramGb} GB)`
  }

  /** True if estimated VRAM exceeds available VRAM for selected GPUs */
  function vramWarning(): string | null {
    if (vramEst == null) return null
    const available = gpuIndices.reduce((acc, idx) => acc + getGpuVram(idx), 0)
    if (available > 0 && vramEst > available) {
      const estGb = (vramEst / 1024).toFixed(1)
      const availGb = (available / 1024).toFixed(0)
      return `Estimado ~${estGb} GB > VRAM disponible ${availGb} GB. El engine puede no compilar.`
    }
    return null
  }

  const asymmetricWarning =
    gpuSelection === "both"
      ? "La RTX 3060 es más lenta que la 3090. Usar ambas GPUs solo si el modelo no cabe en la 3090 sola."
      : null

  // ── SSE subscription ──────────────────────────────────────────
  const sseUrl = job && screen === "progress" ? api.trtllm.streamUrl(job.jobId) : ""

  const onSseMessage = useCallback(
    (data: Record<string, unknown>) => {
      if (data.type === "log") {
        const line = String(data.line ?? "")
        setLogLines((prev) => [...prev.slice(-499), line])
        return
      }
      // progress event — update job state
      const updated = data as unknown as CompileJob
      setJob((prev) => (prev ? { ...prev, ...updated } : prev))
    },
    [],
  )

  useSSE<Record<string, unknown>>(sseUrl, onSseMessage)

  // ── Submit ────────────────────────────────────────────────────
  const handleSubmit = async () => {
    if (!model) return
    setSubmitting(true)
    try {
      const created = await api.trtllm.compile({
        modelId: model.modelId,
        gpuIndices,
        dtype,
        maxBatchSize,
        maxInputLen,
        maxSeqLen,
        engineName: engineName.trim() || defaultEngineName(model.modelId, dtype),
      })
      setJob(created)
      setScreen("progress")
      toast.success("Compilación iniciada")
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Error al iniciar compilación")
    } finally {
      setSubmitting(false)
    }
  }

  const handleCancel = async () => {
    if (!job) return
    try {
      const updated = await api.trtllm.cancel(job.jobId)
      setJob(updated)
      toast("Compilación cancelada")
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Error al cancelar")
    }
  }

  const handleLoadNow = () => {
    if (!job?.engineDir) return
    const engineModelId = `tensorrt_llm/${job.engineName}`
    onLoadNow?.(engineModelId)
    onOpenChange(false)
  }

  if (!model) return null

  const isTerminal = job && ["done", "failed", "cancelled"].includes(job.status)
  const vramWarn = vramWarning()

  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-40 bg-black/60" />
        <Dialog.Content className="fixed left-1/2 top-1/2 z-50 w-[95vw] max-w-2xl -translate-x-1/2 -translate-y-1/2 rounded-lg border border-border bg-card p-6 shadow-xl">
          <div className="mb-4 flex items-center justify-between">
            <Dialog.Title className="flex items-center gap-2 text-lg font-semibold">
              <Cpu size={18} />
              Compilar engine TensorRT-LLM
            </Dialog.Title>
            <Dialog.Close className="rounded-md p-1 hover:bg-muted">
              <X size={16} />
            </Dialog.Close>
          </div>

          <p className="mb-4 text-sm text-muted-foreground">
            <span className="font-medium text-foreground">{model.displayName}</span>
            <span className="ml-2 text-xs opacity-60">{model.modelId}</span>
          </p>

          {/* ── AVISO: tiempo ──────────────────────────────────── */}
          <div className="mb-4 flex items-start gap-2 rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-200">
            <AlertTriangle size={14} className="mt-0.5 shrink-0" />
            <span>La compilación puede tardar <strong>15–60 minutos</strong> según el modelo y la GPU. La UI permanece usable durante el proceso.</span>
          </div>

          {screen === "form" && (
            <div className="space-y-4">
              {/* GPU target */}
              <div>
                <label className="mb-1.5 block text-sm font-medium">GPU destino</label>
                <div className="space-y-1.5">
                  {gpus.filter((g) => g.index === 1).map((g) => (
                    <label key={`single-${g.index}`} className="flex cursor-pointer items-center gap-2 rounded-md border border-border px-3 py-2 text-sm hover:bg-muted has-[:checked]:border-emerald-500/50 has-[:checked]:bg-emerald-500/5">
                      <input
                        type="radio"
                        name="gpu"
                        value="single1"
                        checked={gpuSelection === "single1"}
                        onChange={() => setGpuSelection("single1")}
                        className="accent-emerald-500"
                      />
                      <span>{gpuLabel(g)}</span>
                      <span className="ml-auto text-xs text-emerald-400">Recomendado</span>
                    </label>
                  ))}
                  {gpus.filter((g) => g.index === 0).map((g) => (
                    <label key={`single-${g.index}`} className="flex cursor-pointer items-center gap-2 rounded-md border border-border px-3 py-2 text-sm hover:bg-muted has-[:checked]:border-blue-500/50 has-[:checked]:bg-blue-500/5">
                      <input
                        type="radio"
                        name="gpu"
                        value="single0"
                        checked={gpuSelection === "single0"}
                        onChange={() => setGpuSelection("single0")}
                        className="accent-blue-500"
                      />
                      <span>{gpuLabel(g)}</span>
                    </label>
                  ))}
                  {gpus.length >= 2 && (
                    <label className="flex cursor-pointer items-center gap-2 rounded-md border border-border px-3 py-2 text-sm hover:bg-muted has-[:checked]:border-purple-500/50 has-[:checked]:bg-purple-500/5">
                      <input
                        type="radio"
                        name="gpu"
                        value="both"
                        checked={gpuSelection === "both"}
                        onChange={() => setGpuSelection("both")}
                        className="accent-purple-500"
                      />
                      <span>Ambas GPUs (Tensor Parallelism TP=2)</span>
                    </label>
                  )}
                </div>
                {asymmetricWarning && (
                  <p className="mt-1.5 flex items-start gap-1.5 text-xs text-amber-300">
                    <AlertTriangle size={12} className="mt-0.5 shrink-0" />
                    {asymmetricWarning}
                  </p>
                )}
              </div>

              {/* Dtype + batch size */}
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="mb-1 block text-sm font-medium">Dtype</label>
                  <select
                    value={dtype}
                    onChange={(e) => setDtype(e.target.value)}
                    className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                  >
                    <option value="fp16">fp16</option>
                    <option value="bf16">bf16</option>
                    <option value="int8">int8</option>
                    <option value="fp8">fp8</option>
                  </select>
                </div>
                <div>
                  <label className="mb-1 block text-sm font-medium">Max batch size</label>
                  <select
                    value={maxBatchSize}
                    onChange={(e) => setMaxBatchSize(Number(e.target.value))}
                    className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                  >
                    {[1, 4, 8, 16].map((v) => (
                      <option key={v} value={v}>{v}</option>
                    ))}
                  </select>
                </div>
              </div>

              {/* Input / seq len */}
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="mb-1 block text-sm font-medium">Max input len</label>
                  <select
                    value={maxInputLen}
                    onChange={(e) => setMaxInputLen(Number(e.target.value))}
                    className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                  >
                    {[512, 2048, 8192, 32768].map((v) => (
                      <option key={v} value={v}>{v.toLocaleString()}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="mb-1 block text-sm font-medium">Max seq len</label>
                  <select
                    value={maxSeqLen}
                    onChange={(e) => setMaxSeqLen(Number(e.target.value))}
                    className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                  >
                    {[1024, 4096, 16384, 65536].map((v) => (
                      <option key={v} value={v}>{v.toLocaleString()}</option>
                    ))}
                  </select>
                </div>
              </div>

              {/* Engine name */}
              <div>
                <label className="mb-1 block text-sm font-medium">Nombre del engine</label>
                <input
                  value={engineName}
                  onChange={(e) => setEngineName(e.target.value)}
                  placeholder="Nombre de la carpeta destino"
                  className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm font-mono"
                />
              </div>

              {/* VRAM estimate */}
              {vramEst != null && (
                <div className={`flex items-start gap-2 rounded-md border px-3 py-2 text-xs ${vramWarn ? "border-red-500/40 bg-red-500/10 text-red-300" : "border-border bg-muted/40 text-muted-foreground"}`}>
                  {vramWarn ? <AlertTriangle size={13} className="mt-0.5 shrink-0" /> : <Cpu size={13} className="mt-0.5 shrink-0" />}
                  <span>
                    VRAM estimada: <strong className="text-foreground">{(vramEst / 1024).toFixed(1)} GB</strong>
                    {vramWarn ? ` — ${vramWarn}` : " (estimación aproximada)"}
                  </span>
                </div>
              )}

              <div className="flex justify-end gap-2 pt-2">
                <Dialog.Close className="rounded-md border border-border px-4 py-2 text-sm hover:bg-muted">
                  Cancelar
                </Dialog.Close>
                <button
                  type="button"
                  onClick={() => void handleSubmit()}
                  disabled={submitting}
                  className="flex items-center gap-2 rounded-md bg-emerald-600 px-4 py-2 text-sm font-medium text-white hover:bg-emerald-500 disabled:opacity-60"
                >
                  {submitting && <Loader2 size={14} className="animate-spin" />}
                  Iniciar compilación
                </button>
              </div>
            </div>
          )}

          {screen === "progress" && job && (
            <div className="space-y-4">
              {/* Status + phase */}
              <div className="flex items-center justify-between text-sm">
                <PhaseLabel phase={job.phase} status={job.status} />
                <span className="text-muted-foreground">{job.progressPct}%</span>
              </div>

              <ProgressBar pct={job.progressPct} />

              {/* Log output */}
              <div
                ref={logRef}
                className="h-52 overflow-y-auto rounded-md border border-border bg-background p-2 font-mono text-xs text-muted-foreground"
              >
                {logLines.length === 0 ? (
                  <span className="opacity-50">Esperando salida de Docker…</span>
                ) : (
                  logLines.map((line, i) => (
                    <div key={i} className="whitespace-pre-wrap break-all leading-5">{line}</div>
                  ))
                )}
              </div>

              {/* Error detail */}
              {job.errorDetail && (
                <div className="rounded-md border border-red-500/40 bg-red-500/10 px-3 py-2 text-xs text-red-300">
                  {job.errorDetail}
                </div>
              )}

              {/* Engine dir */}
              {job.status === "done" && job.engineDir && (
                <div className="rounded-md border border-emerald-500/30 bg-emerald-500/10 px-3 py-2 text-xs text-emerald-300">
                  Engine listo en: <span className="font-mono">{job.engineDir}</span>
                </div>
              )}

              {/* Actions */}
              <div className="flex justify-between gap-2">
                <div>
                  {!isTerminal && (
                    <button
                      type="button"
                      onClick={() => void handleCancel()}
                      className="rounded-md border border-red-500/40 px-3 py-2 text-sm text-red-300 hover:bg-red-500/10"
                    >
                      Cancelar compilación
                    </button>
                  )}
                </div>
                <div className="flex gap-2">
                  <Dialog.Close className="rounded-md border border-border px-3 py-2 text-sm hover:bg-muted">
                    {isTerminal ? "Cerrar" : "Cerrar (sigue en curso)"}
                  </Dialog.Close>
                  {job.status === "done" && onLoadNow && (
                    <button
                      type="button"
                      onClick={handleLoadNow}
                      className="flex items-center gap-2 rounded-md bg-emerald-600 px-4 py-2 text-sm font-medium text-white hover:bg-emerald-500"
                    >
                      <Check size={14} />
                      Cargar modelo ahora
                    </button>
                  )}
                </div>
              </div>
            </div>
          )}
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  )
}
