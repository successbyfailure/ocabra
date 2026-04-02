/**
 * CompileModal — formulario de compilación TensorRT-LLM + progreso SSE.
 *
 * Flujo:
 *  1. Pantalla "form"  — el usuario configura GPU, dtype, tamaños y nombre del engine.
 *  2. Pantalla "progress" — progreso en tiempo real via SSE al terminar de enviar.
 */
import { useCallback, useEffect, useRef, useState } from "react"
import * as Dialog from "@radix-ui/react-dialog"
import { AlertTriangle, Check, CheckCircle2, Circle, Cpu, HardDrive, Loader2, X, XCircle } from "lucide-react"
import { toast } from "sonner"
import { api, type VramEstimate } from "@/api/client"
import { useSSE } from "@/hooks/useSSE"
import { useGpuStore } from "@/stores/gpuStore"
import type { CompileJob, GPUState, ModelState } from "@/types"

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

// ── Compile phase stepper ────────────────────────────────────────

type StepState = "waiting" | "running" | "done" | "failed"

const PHASES = [
  {
    id: "convert",
    label: "Convertir checkpoint",
    description: "Carga los pesos HuggingFace en CPU y los convierte al formato TRT-LLM. Para TP>1 se genera un shard por GPU.",
    eta: "5–20 min según tamaño",
  },
  {
    id: "build",
    label: "Compilar engine TensorRT",
    description: "TensorRT optimiza el grafo de operaciones: fusión de kernels, calibración de precisión y generación del engine binario.",
    eta: "3–30 min según modelo y GPU",
  },
] as const

function stepState(phaseId: string, currentPhase: CompileJob["phase"], status: CompileJob["status"]): StepState {
  const order = ["convert", "build"]
  const currentIdx = currentPhase ? order.indexOf(currentPhase) : -1
  const thisIdx = order.indexOf(phaseId)

  if (status === "failed" && currentPhase === phaseId) return "failed"
  if (status === "done") return "done"
  if (thisIdx < currentIdx) return "done"
  if (thisIdx === currentIdx) return "running"
  return "waiting"
}

function StepIcon({ state }: { state: StepState }) {
  if (state === "done") return <CheckCircle2 size={18} className="text-emerald-400 shrink-0" />
  if (state === "failed") return <XCircle size={18} className="text-red-400 shrink-0" />
  if (state === "running") return <Loader2 size={18} className="animate-spin text-blue-400 shrink-0" />
  return <Circle size={18} className="text-muted-foreground/40 shrink-0" />
}

function CompileStepper({ job }: { job: CompileJob }) {
  const allDone = job.status === "done"
  const allFailed = job.status === "failed"

  return (
    <div className="space-y-2">
      {PHASES.map((phase, i) => {
        const state = allDone ? "done" : stepState(phase.id, job.phase, job.status)
        const isActive = state === "running" || (allFailed && job.phase === phase.id)
        return (
          <div
            key={phase.id}
            className={`rounded-lg border px-3 py-2.5 transition-colors ${
              state === "running"
                ? "border-blue-500/40 bg-blue-500/5"
                : state === "failed"
                  ? "border-red-500/40 bg-red-500/5"
                  : state === "done"
                    ? "border-emerald-500/20 bg-emerald-500/5"
                    : "border-border bg-muted/20 opacity-50"
            }`}
          >
            <div className="flex items-center gap-2">
              <StepIcon state={state} />
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className={`text-sm font-medium ${state === "waiting" ? "text-muted-foreground" : "text-foreground"}`}>
                    Paso {i + 1}/2 — {phase.label}
                  </span>
                  {state === "running" && (
                    <span className="ml-auto text-xs text-blue-400 shrink-0">{job.progressPct}%</span>
                  )}
                  {state === "done" && !allDone && (
                    <span className="ml-auto text-xs text-emerald-400 shrink-0">listo</span>
                  )}
                </div>
                {isActive && (
                  <p className="mt-0.5 text-xs text-muted-foreground">{phase.description}</p>
                )}
                {state === "waiting" && (
                  <p className="mt-0.5 text-xs text-muted-foreground/60">ETA: {phase.eta}</p>
                )}
              </div>
            </div>
            {state === "running" && (
              <div className="mt-2 h-1 w-full overflow-hidden rounded-full bg-muted">
                <div
                  className="h-full rounded-full bg-blue-500 transition-all duration-500"
                  style={{
                    width: phase.id === "convert"
                      ? `${Math.min(99, (job.progressPct / 45) * 100)}%`
                      : `${Math.min(99, ((job.progressPct - 50) / 50) * 100)}%`,
                  }}
                />
              </div>
            )}
          </div>
        )
      })}

      {allDone && (
        <div className="flex items-center gap-2 rounded-lg border border-emerald-500/40 bg-emerald-500/10 px-3 py-2 text-sm text-emerald-300">
          <Check size={16} />
          Engine compilado y listo para cargar
        </div>
      )}
      {allFailed && (
        <div className="flex items-center gap-2 rounded-lg border border-red-500/40 bg-red-500/10 px-3 py-2 text-sm text-red-300">
          <XCircle size={16} />
          Compilación fallida — revisa el log
        </div>
      )}
      {job.status === "cancelled" && (
        <div className="flex items-center gap-2 rounded-lg border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-sm text-amber-300">
          <XCircle size={16} />
          Compilación cancelada
        </div>
      )}
    </div>
  )
}

// ── Elapsed timer ────────────────────────────────────────────────

function ElapsedTimer({ startedAt }: { startedAt: string | null }) {
  const [elapsed, setElapsed] = useState("")
  useEffect(() => {
    if (!startedAt) return
    const start = new Date(startedAt).getTime()
    const tick = () => {
      const s = Math.floor((Date.now() - start) / 1000)
      const m = Math.floor(s / 60)
      setElapsed(`${m}m ${s % 60}s`)
    }
    tick()
    const id = setInterval(tick, 1000)
    return () => clearInterval(id)
  }, [startedAt])
  if (!elapsed) return null
  return <span className="text-xs text-muted-foreground">Tiempo: {elapsed}</span>
}

// ── VRAM estimate panel ──────────────────────────────────────────

function mb(v: number) {
  return v >= 1024 ? `${(v / 1024).toFixed(1)} GB` : `${v} MB`
}

function quantizationLabel(quant: string | null | undefined) {
  const value = String(quant ?? "").toLowerCase()
  if (value === "awq") return "AWQ (4-bit)"
  if (value === "gptq") return "GPTQ (4-bit)"
  if (value === "fp16") return "FP16"
  if (value === "bf16") return "BF16"
  if (value === "int8") return "INT8"
  if (value === "fp8") return "FP8"
  return value ? value.toUpperCase() : "Desconocida"
}

function runtimeDtypeOptions(quant: string | null | undefined) {
  const sourceQuant = String(quant ?? "").toLowerCase()
  const checkpointIs4Bit = sourceQuant === "awq" || sourceQuant === "gptq"

  return [
    { value: "fp16", label: "fp16", disabled: false },
    { value: "bf16", label: "bf16", disabled: false },
    {
      value: "int8",
      label: checkpointIs4Bit ? "int8 (no aplica a checkpoint 4-bit)" : "int8",
      disabled: checkpointIs4Bit,
    },
    {
      value: "fp8",
      label: checkpointIs4Bit ? "fp8 (no aplica a checkpoint 4-bit)" : "fp8",
      disabled: checkpointIs4Bit,
    },
  ] as const
}

const MAX_BATCH_PRESETS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
const MAX_INPUT_PRESETS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
const MAX_SEQ_PRESETS = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]

function clampCompileValue(value: number, min: number, max?: number) {
  if (!Number.isFinite(value)) return min
  const rounded = Math.round(value)
  if (max != null) return Math.min(max, Math.max(min, rounded))
  return Math.max(min, rounded)
}

function CompileNumberField({
  label,
  value,
  onChange,
  presets,
  min,
  max,
  step = 1,
  helpText,
}: {
  label: string
  value: number
  onChange: (value: number) => void
  presets: number[]
  min: number
  max?: number
  step?: number
  helpText?: string
}) {
  const presetValue = presets.includes(value) ? String(value) : "custom"

  return (
    <div>
      <label className="mb-1 block text-sm font-medium">{label}</label>
      <div className="grid grid-cols-[minmax(0,1fr)_110px] gap-2">
        <select
          value={presetValue}
          onChange={(e) => {
            const next = e.target.value
            if (next === "custom") return
            onChange(clampCompileValue(Number(next), min, max))
          }}
          className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
        >
          {presets.map((preset) => (
            <option key={preset} value={preset}>
              {preset.toLocaleString()}
            </option>
          ))}
          <option value="custom">Personalizado</option>
        </select>
        <input
          type="number"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(clampCompileValue(Number(e.target.value), min, max))}
          className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
        />
      </div>
      {helpText && <p className="mt-1 text-xs text-muted-foreground">{helpText}</p>}
    </div>
  )
}

function VramEstimatePanel({ estimate, estimating }: { estimate: VramEstimate | null; estimating: boolean }) {
  if (estimating) {
    return (
      <div className="flex items-center gap-2 rounded-md border border-border bg-muted/30 px-3 py-2 text-xs text-muted-foreground">
        <Loader2 size={12} className="animate-spin" />
        Calculando estimación…
      </div>
    )
  }
  if (!estimate) return null

  const hasCompleteEstimate =
    Number.isFinite(estimate.serve?.vramPerGpuMb) &&
    Number.isFinite(estimate.build?.vramPerGpuMb) &&
    Number.isFinite(estimate.disk?.totalPeakMb)

  if (!hasCompleteEstimate) {
    return (
      <div className="flex items-start gap-2 rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-200">
        <AlertTriangle size={12} className="mt-0.5 shrink-0" />
        La estimación de memoria no está completa. Revisa `config.json` del modelo o vuelve a abrir el modal.
      </div>
    )
  }

  const hasWarnings = estimate.warnings.length > 0
  const hasTightWarning = estimate.warnings.some((warning) => warning.startsWith("Muy justo"))
  const serveMb = estimate.serve.vramPerGpuMb
  const buildMb = estimate.build.vramPerGpuMb
  const tpLabel = estimate.tpSize > 1 ? ` × ${estimate.tpSize} GPUs` : ""

  return (
    <div className="space-y-1.5">
      {/* Main estimate row */}
      <div className={`rounded-md border px-3 py-2 text-xs ${
        hasWarnings
          ? hasTightWarning
            ? "border-amber-500/40 bg-amber-500/10"
            : "border-red-500/40 bg-red-500/10"
          : "border-border bg-muted/30"
      }`}>
        <div className="mb-1.5 flex items-center gap-1.5 font-medium text-foreground">
          <Cpu size={12} />
          Estimación de recursos
          {estimate.estimatedParamsB && (
            <span className="ml-auto font-normal text-muted-foreground">
              ~{estimate.estimatedParamsB}B params · checkpoint {quantizationLabel(estimate.quant)}
            </span>
          )}
        </div>

        <div className="grid grid-cols-3 gap-2 text-center">
          <div className="rounded border border-border/50 bg-background/50 px-2 py-1.5">
            <div className="text-muted-foreground">Compilar</div>
            <div className={`font-mono font-semibold ${
              hasWarnings ? (hasTightWarning ? "text-amber-300" : "text-red-300") : "text-amber-300"
            }`}>
              {mb(buildMb)}{tpLabel}
            </div>
            <div className="text-muted-foreground opacity-70">por GPU</div>
          </div>
          <div className="rounded border border-border/50 bg-background/50 px-2 py-1.5">
            <div className="text-muted-foreground">Servir</div>
            <div className="font-mono font-semibold text-blue-300">
              {mb(serveMb)}{tpLabel}
            </div>
            <div className="text-muted-foreground opacity-70">por GPU</div>
          </div>
          <div className="rounded border border-border/50 bg-background/50 px-2 py-1.5">
            <div className="flex items-center justify-center gap-1 text-muted-foreground">
              <HardDrive size={10} /> Disco
            </div>
            <div className="font-mono font-semibold text-purple-300">
              {mb(estimate.disk.totalPeakMb)}
            </div>
            <div className="text-muted-foreground opacity-70">pico total</div>
          </div>
        </div>

        {/* KV cache breakdown */}
        <div className="mt-1.5 flex gap-3 text-muted-foreground opacity-80">
          <span>Pesos: {mb(estimate.serve.breakdown.weightsMb)}</span>
          <span>KV cache: {mb(estimate.serve.breakdown.kvCacheMb)}</span>
          <span>Overhead: {mb(estimate.serve.breakdown.overheadMb)}</span>
          {!estimate.configFound && <span className="text-amber-400">⚠ config.json no encontrado</span>}
        </div>
      </div>

      {/* Warnings */}
      {estimate.warnings.map((w, i) => (
        <div
          key={i}
          className={`flex items-start gap-1.5 rounded-md border px-3 py-1.5 text-xs ${
            w.startsWith("Muy justo")
              ? "border-amber-500/40 bg-amber-500/10 text-amber-200"
              : "border-red-500/40 bg-red-500/10 text-red-300"
          }`}
        >
          <AlertTriangle size={12} className="mt-0.5 shrink-0" />
          {w}
        </div>
      ))}
    </div>
  )
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
  const openedForRef = useRef<string | null>(null)
  const estimateRequestRef = useRef(0)

  // ── Form state ────────────────────────────────────────────────
  const [gpuSelection, setGpuSelection] = useState<"single0" | "single1" | "both">("single1")
  const [dtype, setDtype] = useState("fp16")
  const [maxBatchSize, setMaxBatchSize] = useState(1)
  const [maxInputLen, setMaxInputLen] = useState(2048)
  const [maxSeqLen, setMaxSeqLen] = useState(4096)
  const [engineName, setEngineName] = useState("")
  const [submitting, setSubmitting] = useState(false)
  const [estimate, setEstimate] = useState<VramEstimate | null>(null)
  const [estimating, setEstimating] = useState(false)

  // ── Progress state ────────────────────────────────────────────
  const [screen, setScreen] = useState<"form" | "progress">("form")
  const [job, setJob] = useState<CompileJob | null>(null)
  const [logLines, setLogLines] = useState<string[]>([])
  const logRef = useRef<HTMLDivElement>(null)

  // ── GPU resolution ────────────────────────────────────────────
  const gpuIndices = gpuSelection === "both" ? [0, 1] : gpuSelection === "single0" ? [0] : [1]

  // Reset once when the dialog opens for a given model.
  useEffect(() => {
    if (!open || !model) {
      openedForRef.current = null
      return
    }
    if (openedForRef.current === model.modelId) return

    openedForRef.current = model.modelId
    setScreen("form")
    setJob(null)
    setLogLines([])
    setDtype("fp16")
    setMaxBatchSize(1)
    setMaxInputLen(2048)
    setMaxSeqLen(4096)
    setEngineName(defaultEngineName(model.modelId, "fp16"))
    setEstimate(null)
    setEstimating(false)
    // Pick best single GPU default: prefer GPU 1 (3090) if available
    const hasGpu1 = gpus.some((g) => g.index === 1)
    setGpuSelection(hasGpu1 ? "single1" : "single0")
  }, [open, model, gpus])

  // Fetch VRAM estimate from backend (debounced), ignoring stale responses.
  useEffect(() => {
    if (!open || !model) return
    const tpSize = gpuSelection === "both" ? 2 : 1
    const requestId = ++estimateRequestRef.current
    const timer = setTimeout(async () => {
      setEstimating(true)
      try {
        const est = await api.trtllm.estimate({
          modelId: model.modelId,
          gpuIndices: gpuSelection === "both" ? [0, 1] : gpuSelection === "single0" ? [0] : [1],
          tpSize,
          dtype,
          maxBatchSize,
          maxSeqLen,
        })
        if (estimateRequestRef.current === requestId) {
          setEstimate(est)
        }
      } catch {
        if (estimateRequestRef.current === requestId) {
          setEstimate(null)
        }
      } finally {
        if (estimateRequestRef.current === requestId) {
          setEstimating(false)
        }
      }
    }, 400)
    return () => {
      clearTimeout(timer)
      if (estimateRequestRef.current === requestId) {
        setEstimating(false)
      }
    }
  }, [open, model, gpuSelection, dtype, maxBatchSize, maxSeqLen])

  useEffect(() => {
    const options = runtimeDtypeOptions(estimate?.quant)
    if (options.some((option) => option.value === dtype && option.disabled)) {
      setDtype("fp16")
    }
  }, [estimate?.quant, dtype])

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

  function gpuLabel(g: GPUState) {
    const vramGb = (g.totalVramMb / 1024).toFixed(0)
    return `GPU ${g.index} — ${g.name} (${vramGb} GB)`
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
    const normalizedBatch = clampCompileValue(maxBatchSize, 1, 256)
    const normalizedInput = clampCompileValue(maxInputLen, 128)
    const normalizedSeq = clampCompileValue(maxSeqLen, 256)

    if (normalizedInput > normalizedSeq) {
      toast.error("`Max input len` no puede ser mayor que `Max seq len`")
      return
    }

    setSubmitting(true)
    try {
      const created = await api.trtllm.compile({
        modelId: model.modelId,
        gpuIndices,
        dtype,
        maxBatchSize: normalizedBatch,
        maxInputLen: normalizedInput,
        maxSeqLen: normalizedSeq,
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
  const dtypeOptions = runtimeDtypeOptions(estimate?.quant)
  const selectedDtypeDisabled = dtypeOptions.some((option) => option.value === dtype && option.disabled)

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
                  <label className="mb-1 block text-sm font-medium">Runtime dtype</label>
                  <select
                    value={dtype}
                    onChange={(e) => setDtype(e.target.value)}
                    className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                  >
                    {dtypeOptions.map((option) => (
                      <option key={option.value} value={option.value} disabled={option.disabled}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                  <p className="mt-1 text-xs text-muted-foreground">
                    Ajusta la precision objetivo del engine. No cambia la cuantizacion base del checkpoint.
                  </p>
                  {selectedDtypeDisabled && (
                    <p className="mt-1 text-xs text-amber-300">
                      La opcion seleccionada no encaja con la cuantizacion detectada del checkpoint. Cambia a `fp16` o `bf16`.
                    </p>
                  )}
                </div>
                <div>
                  <CompileNumberField
                    label="Max batch size"
                    value={maxBatchSize}
                    onChange={setMaxBatchSize}
                    presets={MAX_BATCH_PRESETS}
                    min={1}
                    max={256}
                  />
                </div>
              </div>

              <div className="rounded-md border border-border bg-muted/20 px-3 py-2 text-xs text-muted-foreground">
                <span className="font-medium text-foreground">Checkpoint detectado:</span>{" "}
                {estimate ? quantizationLabel(estimate.quant) : "calculando..."}.
                {" "}Si el modelo fuente es AWQ o GPTQ, ya viene cuantizado en 4-bit; cambiar `runtime dtype` no lo convierte a otro formato base.
              </div>

              {/* Input / seq len */}
              <div className="grid grid-cols-2 gap-3">
                <CompileNumberField
                  label="Max input len"
                  value={maxInputLen}
                  onChange={setMaxInputLen}
                  presets={MAX_INPUT_PRESETS}
                  min={128}
                  helpText="Tokens máximos de entrada admitidos por el engine."
                />
                <CompileNumberField
                  label="Max seq len"
                  value={maxSeqLen}
                  onChange={setMaxSeqLen}
                  presets={MAX_SEQ_PRESETS}
                  min={256}
                  helpText="Ventana total de secuencia: entrada + salida."
                />
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

              {/* VRAM / disk estimate */}
              <VramEstimatePanel estimate={estimate} estimating={estimating} />

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
            <div className="space-y-3">
              {/* Stepper */}
              <CompileStepper job={job} />

              {/* Elapsed time */}
              <div className="flex justify-end">
                <ElapsedTimer startedAt={job.startedAt ?? null} />
              </div>

              {/* Log output */}
              <div>
                <div className="mb-1 text-xs font-medium text-muted-foreground">Log Docker</div>
                <div
                  ref={logRef}
                  className="h-40 overflow-y-auto rounded-md border border-border bg-background p-2 font-mono text-xs text-muted-foreground"
                >
                  {logLines.length === 0 ? (
                    <span className="opacity-50">Esperando salida de Docker…</span>
                  ) : (
                    logLines.map((line, i) => (
                      <div key={i} className="whitespace-pre-wrap break-all leading-5">{line}</div>
                    ))
                  )}
                </div>
              </div>

              {/* Error detail on failure */}
              {job.errorDetail && job.status === "failed" && (
                <details className="rounded-md border border-red-500/40 bg-red-500/10 px-3 py-2 text-xs text-red-300">
                  <summary className="cursor-pointer font-medium">Detalle del error</summary>
                  <pre className="mt-2 max-h-32 overflow-y-auto whitespace-pre-wrap break-all opacity-80">{job.errorDetail}</pre>
                </details>
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
