import type { ReactNode } from "react"
import { useEffect, useState } from "react"
import * as Dialog from "@radix-ui/react-dialog"
import { X } from "lucide-react"
import { toast } from "sonner"
import { api } from "@/api/client"
import {
  formatTokenCount,
  getBitNetConfig,
  getLlamaCppConfig,
  getModelContextSummary,
  getSGLangConfig,
  getTensorRTLLMConfig,
  getVllmConfig,
  getWhisperConfig,
  getWritableExtraConfig,
} from "@/lib/modelContext"
import { ScheduleEditor } from "@/components/models/ScheduleEditor"
import { getProbeOverrideHint, getProbeStatusLabel } from "@/lib/vllmProbe"
import { useIsAdmin } from "@/hooks/useAuth"
import type {
  BackendExtraConfig,
  EvictionSchedule,
  GPUState,
  Group,
  LoadPolicy,
  ModelMemoryEstimate,
  ModelState,
  VLLMConfig,
} from "@/types"

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
    extraConfig?: BackendExtraConfig
  }) => Promise<void>
}

const VLLM_GPU_MEMORY_UTILIZATION_PRESETS = [0.75, 0.8, 0.85, 0.9, 0.92, 0.95]
const VLLM_MAX_NUM_SEQS_PRESETS = [1, 2, 4, 8, 12, 16, 24, 32]

const VLLM_ROOT_KEYS = [
  "model_impl",
  "runner",
  "tensor_parallel_size",
  "max_model_len",
  "gpu_memory_utilization",
  "max_num_seqs",
  "enable_prefix_caching",
  "trust_remote_code",
  "hf_overrides",
  "chat_template",
  "tool_call_parser",
  "reasoning_parser",
]

const SGLANG_ROOT_KEYS = [
  "tensor_parallel_size",
  "context_length",
  "mem_fraction_static",
  "trust_remote_code",
  "disable_radix_cache",
]

const LLAMA_CPP_ROOT_KEYS = ["gpu_layers", "ctx_size", "flash_attn", "embedding"]
const BITNET_ROOT_KEYS = ["gpu_layers", "ctx_size", "flash_attn"]
const TENSORRT_LLM_ROOT_KEYS = ["max_batch_size", "context_length", "trust_remote_code"]
const WHISPER_ROOT_KEYS = ["diarization_enabled", "diarization_model_id"]

function FieldHint({ children }: { children: ReactNode }) {
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

function clampIntegerInput(value: string, min: number) {
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) return ""
  return String(Math.max(min, Math.round(parsed)))
}

function clampDecimalInput(value: string, min: number, max: number, digits = 2) {
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) return ""
  const clamped = Math.min(max, Math.max(min, parsed))
  return String(Number(clamped.toFixed(digits)))
}

function PresetNumberField({
  label,
  value,
  onValueChange,
  presets,
  placeholder,
  min,
  max,
  step,
  help,
}: {
  label: string
  value: string
  onValueChange: (value: string) => void
  presets: Array<number>
  placeholder: string
  min: number
  max?: number
  step: string
  help: string
}) {
  const presetValue = value !== "" && presets.some((preset) => String(preset) === value) ? value : "custom"

  return (
    <label className="block text-sm">
      <span className="mb-1 block text-muted-foreground">{label}</span>
      <div className="grid grid-cols-[minmax(0,1fr)_112px] gap-2">
        <select
          value={presetValue}
          onChange={(event) => {
            if (event.target.value === "custom") return
            onValueChange(event.target.value)
          }}
          className="w-full rounded-md border border-border bg-background px-3 py-2"
        >
          {presets.map((preset) => (
            <option key={preset} value={String(preset)}>
              {preset}
            </option>
          ))}
          <option value="custom">custom</option>
        </select>
        <input
          aria-label={label}
          type="number"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(event) => onValueChange(event.target.value)}
          placeholder={placeholder}
          className="w-full rounded-md border border-border bg-background px-3 py-2"
        />
      </div>
      <FieldHint>{help}</FieldHint>
    </label>
  )
}

function compactConfig<T extends Record<string, unknown>>(value: T): T {
  return Object.fromEntries(
    Object.entries(value).filter(([, entry]) => {
      if (entry == null) return false
      if (entry === "") return false
      if (Array.isArray(entry)) return entry.length > 0
      if (typeof entry === "object") return Object.keys(entry).length > 0
      return true
    }),
  ) as T
}

function omitKeys(config: BackendExtraConfig, keys: string[]): BackendExtraConfig {
  const next = { ...config }
  for (const key of keys) delete next[key]
  return next
}

function setNestedConfig(
  config: BackendExtraConfig,
  section: "vllm" | "sglang" | "llama_cpp" | "bitnet" | "tensorrt_llm" | "whisper",
  payload: Record<string, unknown>,
  legacyRootKeys: string[],
): BackendExtraConfig {
  const next = omitKeys(config, legacyRootKeys)
  const compacted = compactConfig(payload)
  if (Object.keys(compacted).length === 0) {
    delete next[section]
    return next
  }
  next[section] = compacted
  return next
}

export function ModelConfigModal({ model, gpus, open, onOpenChange, onSave }: ModelConfigModalProps) {
  const isAdmin = useIsAdmin()

  const [loadPolicy, setLoadPolicy] = useState<LoadPolicy>("on_demand")
  const [preferredGpu, setPreferredGpu] = useState<number | null>(null)
  const [autoReload, setAutoReload] = useState(false)
  const [schedules, setSchedules] = useState<EvictionSchedule[]>([])
  const [saving, setSaving] = useState(false)

  // Group assignment state (system_admin only)
  const [allGroups, setAllGroups] = useState<Group[]>([])
  const [selectedGroupIds, setSelectedGroupIds] = useState<Set<string>>(new Set())
  const [initialGroupIds, setInitialGroupIds] = useState<Set<string>>(new Set())
  const [loadingGroups, setLoadingGroups] = useState(false)

  const [tensorParallelSize, setTensorParallelSize] = useState("")
  const [maxModelLen, setMaxModelLen] = useState("")
  const [modelImpl, setModelImpl] = useState<"auto" | "vllm" | "transformers" | "">("")
  const [runner, setRunner] = useState<"generate" | "pooling" | "">("")
  const [hfOverrides, setHfOverrides] = useState("")
  const [chatTemplate, setChatTemplate] = useState("")
  const [toolCallParser, setToolCallParser] = useState("")
  const [reasoningParser, setReasoningParser] = useState("")
  const [maxNumSeqs, setMaxNumSeqs] = useState("")
  const [gpuMemoryUtilization, setGpuMemoryUtilization] = useState("")
  const [enablePrefixCaching, setEnablePrefixCaching] = useState(true)
  const [trustRemoteCode, setTrustRemoteCode] = useState(false)

  const [sglangTensorParallelSize, setSGLangTensorParallelSize] = useState("")
  const [sglangContextLength, setSGLangContextLength] = useState("")
  const [sglangMemFractionStatic, setSGLangMemFractionStatic] = useState("")
  const [sglangTrustRemoteCode, setSGLangTrustRemoteCode] = useState(false)
  const [disableRadixCache, setDisableRadixCache] = useState(false)

  const [llamaGpuLayers, setLlamaGpuLayers] = useState("")
  const [llamaCtxSize, setLlamaCtxSize] = useState("")
  const [llamaFlashAttn, setLlamaFlashAttn] = useState(false)
  const [llamaEmbedding, setLlamaEmbedding] = useState(false)

  const [bitnetGpuLayers, setBitnetGpuLayers] = useState("")
  const [bitnetCtxSize, setBitnetCtxSize] = useState("")
  const [bitnetFlashAttn, setBitnetFlashAttn] = useState(false)

  const [trtMaxBatchSize, setTRTMaxBatchSize] = useState("")
  const [trtContextLength, setTRTContextLength] = useState("")
  const [trtTrustRemoteCode, setTRTTrustRemoteCode] = useState(false)

  const [whisperDiarizationEnabled, setWhisperDiarizationEnabled] = useState(false)
  const [whisperDiarizationModelId, setWhisperDiarizationModelId] = useState("")
  const [memoryEstimate, setMemoryEstimate] = useState<ModelMemoryEstimate | null>(null)
  const [memoryEstimateError, setMemoryEstimateError] = useState<string | null>(null)
  const [estimatingMemory, setEstimatingMemory] = useState(false)
  const [probingMemory, setProbingMemory] = useState(false)

  const vllm = model ? getVllmConfig(model) : null
  const sglang = model ? getSGLangConfig(model) : null
  const llamaCpp = model ? getLlamaCppConfig(model) : null
  const bitnet = model ? getBitNetConfig(model) : null
  const tensorrt = model ? getTensorRTLLMConfig(model) : null
  const whisper = model ? getWhisperConfig(model) : null

  const recipeId = vllm?.recipeId ?? null
  const recipeNotes = (vllm?.recipeNotes ?? []) as string[]
  const recipeModelImpl = vllm?.recipeModelImpl ?? null
  const recipeRunner = vllm?.recipeRunner ?? null
  const recipeSuggestedConfig = (vllm?.suggestedConfig ?? {}) as Record<string, unknown>
  const recipeSuggestedTuning = (vllm?.suggestedTuning ?? {}) as Record<string, unknown>
  const probeStatus = vllm?.probeStatus ?? null
  const probeReason = vllm?.probeReason ?? null
  const probeObservedAt = vllm?.probeObservedAt ?? null
  const probeRecommendedModelImpl = vllm?.probeRecommendedModelImpl ?? null
  const probeRecommendedRunner = vllm?.probeRecommendedRunner ?? null

  useEffect(() => {
    if (!model) return
    setLoadPolicy(model.loadPolicy)
    setPreferredGpu(model.preferredGpu)
    setAutoReload(model.autoReload)
    setSchedules(model.schedules ?? [])

    setModelImpl(vllm?.modelImpl == null ? "" : vllm.modelImpl)
    setRunner(vllm?.runner == null ? "" : vllm.runner)
    setTensorParallelSize(vllm?.tensorParallelSize == null ? "" : String(vllm.tensorParallelSize))
    setMaxModelLen(vllm?.maxModelLen == null ? "" : String(vllm.maxModelLen))
    setMaxNumSeqs(vllm?.maxNumSeqs == null ? "" : String(vllm.maxNumSeqs))
    setGpuMemoryUtilization(vllm?.gpuMemoryUtilization == null ? "" : String(vllm.gpuMemoryUtilization))
    setEnablePrefixCaching(vllm?.enablePrefixCaching ?? true)
    setTrustRemoteCode(Boolean(vllm?.trustRemoteCode))
    setHfOverrides(
      vllm?.hfOverrides == null
        ? ""
        : typeof vllm.hfOverrides === "string"
          ? vllm.hfOverrides
          : JSON.stringify(vllm.hfOverrides),
    )
    setChatTemplate(vllm?.chatTemplate == null ? "" : String(vllm.chatTemplate))
    setToolCallParser(vllm?.toolCallParser == null ? "" : String(vllm.toolCallParser))
    setReasoningParser(vllm?.reasoningParser == null ? "" : String(vllm.reasoningParser))

    setSGLangTensorParallelSize(
      sglang?.tensorParallelSize == null ? "" : String(sglang.tensorParallelSize),
    )
    setSGLangContextLength(sglang?.contextLength == null ? "" : String(sglang.contextLength))
    setSGLangMemFractionStatic(
      sglang?.memFractionStatic == null ? "" : String(sglang.memFractionStatic),
    )
    setSGLangTrustRemoteCode(Boolean(sglang?.trustRemoteCode))
    setDisableRadixCache(Boolean(sglang?.disableRadixCache))

    setLlamaGpuLayers(llamaCpp?.gpuLayers == null ? "" : String(llamaCpp.gpuLayers))
    setLlamaCtxSize(llamaCpp?.ctxSize == null ? "" : String(llamaCpp.ctxSize))
    setLlamaFlashAttn(Boolean(llamaCpp?.flashAttn))
    setLlamaEmbedding(Boolean(llamaCpp?.embedding))

    setBitnetGpuLayers(bitnet?.gpuLayers == null ? "" : String(bitnet.gpuLayers))
    setBitnetCtxSize(bitnet?.ctxSize == null ? "" : String(bitnet.ctxSize))
    setBitnetFlashAttn(Boolean(bitnet?.flashAttn))

    setTRTMaxBatchSize(tensorrt?.maxBatchSize == null ? "" : String(tensorrt.maxBatchSize))
    setTRTContextLength(tensorrt?.contextLength == null ? "" : String(tensorrt.contextLength))
    setTRTTrustRemoteCode(Boolean(tensorrt?.trustRemoteCode))

    setWhisperDiarizationEnabled(Boolean(whisper?.diarizationEnabled))
    setWhisperDiarizationModelId(
      whisper?.diarizationModelId == null ? "" : String(whisper.diarizationModelId),
    )
  }, [bitnet, llamaCpp, model, sglang, tensorrt, vllm, whisper])

  // Load groups for system_admin when modal opens
  useEffect(() => {
    if (!open || !model || !isAdmin) return
    let cancelled = false
    setLoadingGroups(true)
    void (async () => {
      try {
        const [groups, modelGroupIds] = await Promise.all([
          api.groups.list(),
          api.groups.getModelGroups(model.modelId),
        ])
        if (cancelled) return
        setAllGroups(groups)
        const idSet = new Set(modelGroupIds)
        setSelectedGroupIds(new Set(idSet))
        setInitialGroupIds(new Set(idSet))
      } catch (err) {
        if (!cancelled) {
          toast.error(err instanceof Error ? err.message : "No se pudieron cargar los grupos")
        }
      } finally {
        if (!cancelled) setLoadingGroups(false)
      }
    })()
    return () => { cancelled = true }
  }, [open, model, isAdmin])

  const applyRecipeSuggestedConfig = () => {
    const suggested = recipeSuggestedConfig
    if (modelImpl === "" && vllm && typeof vllm.modelImpl === "string") {
      setModelImpl(vllm.modelImpl)
    }
    if (runner === "" && vllm && typeof vllm.runner === "string") {
      setRunner(vllm.runner)
    }
    if (typeof suggested.toolCallParser === "string") setToolCallParser(suggested.toolCallParser)
    if (typeof suggested.reasoningParser === "string") setReasoningParser(suggested.reasoningParser)
    if (typeof suggested.reasoning_parser === "string") setReasoningParser(suggested.reasoning_parser)
    if (typeof suggested.tool_call_parser === "string") setToolCallParser(suggested.tool_call_parser)
    if (typeof suggested.chatTemplate === "string") setChatTemplate(suggested.chatTemplate)
    if (typeof suggested.chat_template === "string") setChatTemplate(suggested.chat_template)
    if (typeof suggested.hfOverrides === "string") setHfOverrides(suggested.hfOverrides)
    if (typeof suggested.hf_overrides === "string") setHfOverrides(suggested.hf_overrides)
    if (typeof suggested.hfOverrides === "object" && suggested.hfOverrides !== null) {
      setHfOverrides(JSON.stringify(suggested.hfOverrides))
    }
    if (typeof suggested.hf_overrides === "object" && suggested.hf_overrides !== null) {
      setHfOverrides(JSON.stringify(suggested.hf_overrides))
    }
  }

  const applyRecipeSuggestedTuning = () => {
    const tuning = recipeSuggestedTuning
    if (typeof tuning.max_num_seqs === "number") setMaxNumSeqs(String(tuning.max_num_seqs))
    if (typeof tuning.maxNumSeqs === "number") setMaxNumSeqs(String(tuning.maxNumSeqs))
    if (typeof tuning.max_model_len === "number") setMaxModelLen(String(tuning.max_model_len))
    if (typeof tuning.maxModelLen === "number") setMaxModelLen(String(tuning.maxModelLen))
    if (typeof tuning.gpu_memory_utilization === "number") {
      setGpuMemoryUtilization(String(tuning.gpu_memory_utilization))
    }
    if (typeof tuning.gpuMemoryUtilization === "number") {
      setGpuMemoryUtilization(String(tuning.gpuMemoryUtilization))
    }
    if (typeof tuning.enable_prefix_caching === "boolean") setEnablePrefixCaching(tuning.enable_prefix_caching)
    if (typeof tuning.enablePrefixCaching === "boolean") setEnablePrefixCaching(tuning.enablePrefixCaching)
  }

  const applyProbeRecommendation = () => {
    if (probeRecommendedModelImpl != null) setModelImpl(probeRecommendedModelImpl)
    if (probeRecommendedRunner != null) setRunner(probeRecommendedRunner)
  }

  const differsFromProbe =
    (probeRecommendedModelImpl != null && modelImpl !== "" && probeRecommendedModelImpl !== modelImpl) ||
    (probeRecommendedRunner != null && runner !== "" && probeRecommendedRunner !== runner)
  const recipeDiffersFromProbe =
    (recipeModelImpl != null && probeRecommendedModelImpl != null && recipeModelImpl !== probeRecommendedModelImpl) ||
    (recipeRunner != null && probeRecommendedRunner != null && recipeRunner !== probeRecommendedRunner)
  const probeStatusLabel = getProbeStatusLabel(probeStatus)
  const probeOverrideHint = getProbeOverrideHint(probeStatus)
  const context = model ? getModelContextSummary(model) : null
  const showVLLMCompatibility =
    model?.backendType === "vllm" &&
    (Boolean(trustRemoteCode) ||
      hfOverrides.trim() !== "" ||
      chatTemplate.trim() !== "" ||
      toolCallParser.trim() !== "" ||
      reasoningParser.trim() !== "" ||
      Boolean(recipeId) ||
      Boolean(probeStatus))

  const buildExtraConfig = (): BackendExtraConfig | undefined => {
    if (!model) return undefined
    const current = getWritableExtraConfig(model)

    if (model.backendType === "vllm") {
      const next = setNestedConfig(
        current,
        "vllm",
        {
          ...(vllm ?? {}),
          recipeId,
          recipeNotes,
          recipeModelImpl,
          recipeRunner,
          suggestedConfig: recipeSuggestedConfig,
          suggestedTuning: recipeSuggestedTuning,
          probeStatus,
          probeReason,
          probeObservedAt,
          probeRecommendedModelImpl,
          probeRecommendedRunner,
          modelImpl: modelImpl === "" ? null : modelImpl,
          runner: runner === "" ? null : runner,
          tensorParallelSize: tensorParallelSize === "" ? null : Number(tensorParallelSize),
          maxModelLen: maxModelLen === "" ? null : Number(maxModelLen),
          maxNumSeqs: maxNumSeqs === "" ? null : Number(maxNumSeqs),
          gpuMemoryUtilization:
            gpuMemoryUtilization === "" ? null : Number(gpuMemoryUtilization),
          enablePrefixCaching,
          trustRemoteCode,
          hfOverrides: hfOverrides === "" ? null : hfOverrides,
          chatTemplate: chatTemplate === "" ? null : chatTemplate,
          toolCallParser: toolCallParser === "" ? null : toolCallParser,
          reasoningParser: reasoningParser === "" ? null : reasoningParser,
        } satisfies VLLMConfig,
        VLLM_ROOT_KEYS,
      )
      return Object.keys(next).length > 0 ? next : undefined
    }

    if (model.backendType === "sglang") {
      const next = setNestedConfig(
        current,
        "sglang",
        {
          ...(sglang ?? {}),
          tensorParallelSize:
            sglangTensorParallelSize === "" ? null : Number(sglangTensorParallelSize),
          contextLength: sglangContextLength === "" ? null : Number(sglangContextLength),
          memFractionStatic:
            sglangMemFractionStatic === "" ? null : Number(sglangMemFractionStatic),
          trustRemoteCode: sglangTrustRemoteCode,
          disableRadixCache,
        },
        SGLANG_ROOT_KEYS,
      )
      return Object.keys(next).length > 0 ? next : undefined
    }

    if (model.backendType === "llama_cpp") {
      const next = setNestedConfig(
        current,
        "llama_cpp",
        {
          ...(llamaCpp ?? {}),
          gpuLayers: llamaGpuLayers === "" ? null : Number(llamaGpuLayers),
          ctxSize: llamaCtxSize === "" ? null : Number(llamaCtxSize),
          flashAttn: llamaFlashAttn,
          embedding: llamaEmbedding,
        },
        LLAMA_CPP_ROOT_KEYS,
      )
      return Object.keys(next).length > 0 ? next : undefined
    }

    if (model.backendType === "bitnet") {
      const next = setNestedConfig(
        current,
        "bitnet",
        {
          ...(bitnet ?? {}),
          gpuLayers: bitnetGpuLayers === "" ? null : Number(bitnetGpuLayers),
          ctxSize: bitnetCtxSize === "" ? null : Number(bitnetCtxSize),
          flashAttn: bitnetFlashAttn,
        },
        BITNET_ROOT_KEYS,
      )
      return Object.keys(next).length > 0 ? next : undefined
    }

    if (model.backendType === "tensorrt_llm") {
      const next = setNestedConfig(
        current,
        "tensorrt_llm",
        {
          ...(tensorrt ?? {}),
          maxBatchSize: trtMaxBatchSize === "" ? null : Number(trtMaxBatchSize),
          contextLength: trtContextLength === "" ? null : Number(trtContextLength),
          trustRemoteCode: trtTrustRemoteCode,
        },
        TENSORRT_LLM_ROOT_KEYS,
      )
      return Object.keys(next).length > 0 ? next : undefined
    }

    if (model.backendType === "whisper") {
      const next = setNestedConfig(
        current,
        "whisper",
        {
          ...(whisper ?? {}),
          diarizationEnabled: whisperDiarizationEnabled,
          diarizationModelId:
            whisperDiarizationModelId.trim() === "" ? null : whisperDiarizationModelId.trim(),
        },
        WHISPER_ROOT_KEYS,
      )
      return Object.keys(next).length > 0 ? next : undefined
    }

    return Object.keys(current).length > 0 ? current : undefined
  }

  const estimateMemory = async (runProbe = false) => {
    if (!model) return
    if (runProbe) {
      setProbingMemory(true)
    } else {
      setEstimatingMemory(true)
    }
    setMemoryEstimateError(null)
    try {
      const next = await api.models.estimateMemory(model.modelId, {
        preferredGpu,
        extraConfig: buildExtraConfig(),
        runProbe,
      })
      setMemoryEstimate(next)
    } catch (error) {
      setMemoryEstimateError(error instanceof Error ? error.message : "No se pudo calcular la estimación")
    } finally {
      if (runProbe) {
        setProbingMemory(false)
      } else {
        setEstimatingMemory(false)
      }
    }
  }

  const estimateKey = JSON.stringify({
    modelId: model?.modelId ?? null,
    preferredGpu,
    extraConfig: model ? buildExtraConfig() ?? null : null,
  })

  useEffect(() => {
    if (!open || !model) return
    const timer = window.setTimeout(() => {
      void estimateMemory(false)
    }, 250)
    return () => window.clearTimeout(timer)
  }, [estimateKey, open, model])

  const handleSave = async () => {
    if (!model) return
    setSaving(true)
    try {
      const savePromise = onSave(model.modelId, {
        loadPolicy,
        preferredGpu,
        autoReload,
        schedules,
        extraConfig: buildExtraConfig(),
      })

      // Sync group assignments if admin and groups changed
      let groupSyncPromise: Promise<void> = Promise.resolve()
      if (isAdmin && allGroups.length > 0) {
        const addGroupIds = [...selectedGroupIds].filter((id) => !initialGroupIds.has(id))
        const removeGroupIds = [...initialGroupIds].filter((id) => !selectedGroupIds.has(id))
        if (addGroupIds.length > 0 || removeGroupIds.length > 0) {
          groupSyncPromise = api.groups.updateModelGroups(model.modelId, addGroupIds, removeGroupIds)
        }
      }

      await Promise.all([savePromise, groupSyncPromise])
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

            {model && (
              <FieldSection
                title="Prevision de memoria"
                description="Estimación rápida de uso de memoria para la configuración actual, con validación real opcional en runtimes compatibles."
              >
                <div className="space-y-3 text-sm">
                  {(estimatingMemory || probingMemory) && (
                    <div className="rounded-md border border-border bg-background px-3 py-2 text-xs text-muted-foreground">
                      {probingMemory ? "Ejecutando validación real del engine..." : "Recalculando estimación..."}
                    </div>
                  )}
                  {memoryEstimate && (
                    <>
                      <div className="grid gap-3 md:grid-cols-2">
                        <div className="rounded-md border border-border bg-background px-3 py-2 text-xs text-muted-foreground">
                          {`GPU objetivo: ${memoryEstimate.gpuIndex ?? "-"} · total ${memoryEstimate.totalVramMb ?? "-"} MB · libre ${memoryEstimate.freeVramMb ?? "-"} MB`}
                        </div>
                        <div className="rounded-md border border-border bg-background px-3 py-2 text-xs text-muted-foreground">
                          {`fuente=${memoryEstimate.source} · estado=${memoryEstimate.status}`}
                          {memoryEstimate.budgetVramMb != null ? ` · budget ${memoryEstimate.budgetVramMb} MB` : ""}
                        </div>
                      </div>

                      <div className="grid gap-3 md:grid-cols-2">
                        {memoryEstimate.estimatedWeightsMb != null && (
                          <div className="rounded-md border border-border bg-background px-3 py-2 text-xs text-muted-foreground">
                            {`pesos estimados: ${memoryEstimate.estimatedWeightsMb} MB`}
                          </div>
                        )}
                        {memoryEstimate.estimatedEngineMbPerGpu != null && (
                          <div className="rounded-md border border-border bg-background px-3 py-2 text-xs text-muted-foreground">
                            {`engine estimado por GPU: ${memoryEstimate.estimatedEngineMbPerGpu} MB`}
                          </div>
                        )}
                        {memoryEstimate.modelLoadingMemoryMb != null && (
                          <div className="rounded-md border border-border bg-background px-3 py-2 text-xs text-muted-foreground">
                            {`memoria real al cargar pesos: ${memoryEstimate.modelLoadingMemoryMb} MB`}
                          </div>
                        )}
                        {memoryEstimate.estimatedKvCacheMb != null && (
                          <div className="rounded-md border border-border bg-background px-3 py-2 text-xs text-muted-foreground">
                            {`KV cache estimada/real: ${memoryEstimate.estimatedKvCacheMb} MB`}
                          </div>
                        )}
                        {memoryEstimate.requestedContextLength != null && (
                          <div className="rounded-md border border-border bg-background px-3 py-2 text-xs text-muted-foreground">
                            {`contexto pedido: ${formatTokenCount(memoryEstimate.requestedContextLength)}`}
                          </div>
                        )}
                        {memoryEstimate.estimatedMaxContextLength != null && (
                          <div className="rounded-md border border-border bg-background px-3 py-2 text-xs text-muted-foreground">
                            {`contexto maximo estimado por engine: ${formatTokenCount(memoryEstimate.estimatedMaxContextLength)}`}
                          </div>
                        )}
                        {memoryEstimate.maximumConcurrency != null && (
                          <div className="rounded-md border border-border bg-background px-3 py-2 text-xs text-muted-foreground">
                            {`concurrencia maxima aproximada: ${memoryEstimate.maximumConcurrency.toFixed(2)}x`}
                          </div>
                        )}
                        {memoryEstimate.enginePresent != null && (
                          <div className="rounded-md border border-border bg-background px-3 py-2 text-xs text-muted-foreground">
                            {`engine presente: ${memoryEstimate.enginePresent ? "sí" : "no"}`}
                          </div>
                        )}
                      </div>

                      {memoryEstimate.fitsCurrentGpu != null && (
                        <p className={memoryEstimate.fitsCurrentGpu ? "text-xs text-emerald-300" : "text-xs text-amber-300"}>
                          {memoryEstimate.fitsCurrentGpu
                            ? "La configuración parece razonable para la GPU objetivo según la estimación disponible."
                            : "La configuración parece demasiado ajustada para la GPU objetivo; conviene reducir contexto o paralelismo."}
                        </p>
                      )}
                      {memoryEstimate.warning && <p className="text-xs text-amber-300">{memoryEstimate.warning}</p>}
                      {memoryEstimate.notes.length > 0 && (
                        <div className="space-y-1 text-xs text-muted-foreground">
                          {memoryEstimate.notes.map((note, index) => (
                            <p key={`memory-note-${index}`}>{note}</p>
                          ))}
                        </div>
                      )}
                    </>
                  )}
                  {memoryEstimateError && <p className="text-xs text-amber-300">{memoryEstimateError}</p>}
                  <div>
                    <button
                      type="button"
                      onClick={() => void estimateMemory(false)}
                      className="rounded-md border border-border px-3 py-2 text-xs text-foreground hover:bg-muted"
                    >
                      Recalcular estimación
                    </button>
                    {model.backendType === "vllm" && (
                      <button
                        type="button"
                        onClick={() => void estimateMemory(true)}
                        className="ml-2 rounded-md border border-sky-500/40 px-3 py-2 text-xs text-sky-100 hover:bg-sky-500/10"
                      >
                        Validar con engine
                      </button>
                    )}
                  </div>
                </div>
              </FieldSection>
            )}

            {model?.backendType === "vllm" && (
              <>
                {recipeId && (
                  <FieldSection
                    title="Recipe"
                    description="Sugerencias persistidas desde Explore para esta familia de modelo."
                  >
                    <div className="space-y-3 text-sm">
                      <div className="rounded-md border border-emerald-500/20 bg-emerald-500/10 px-3 py-2 text-emerald-100">
                        {`recipe=${recipeId}`}
                        {recipeModelImpl ? ` · model_impl=${recipeModelImpl}` : ""}
                        {recipeRunner ? ` · runner=${recipeRunner}` : ""}
                      </div>
                      {recipeNotes.length > 0 && (
                        <div className="space-y-1 text-xs text-muted-foreground">
                          {recipeNotes.map((note, index) => (
                            <p key={`${recipeId}-note-${index}`}>{note}</p>
                          ))}
                        </div>
                      )}
                      {Object.keys(recipeSuggestedConfig).length > 0 && (
                        <div className="rounded-md border border-border bg-background px-3 py-2 text-xs text-muted-foreground">
                          {Object.entries(recipeSuggestedConfig)
                            .map(([key, value]) => `${key}=${typeof value === "string" ? value : JSON.stringify(value)}`)
                            .join(", ")}
                        </div>
                      )}
                      {Object.keys(recipeSuggestedTuning).length > 0 && (
                        <div className="rounded-md border border-border bg-background px-3 py-2 text-xs text-muted-foreground">
                          tuning:{" "}
                          {Object.entries(recipeSuggestedTuning)
                            .map(([key, value]) => `${key}=${typeof value === "string" ? value : JSON.stringify(value)}`)
                            .join(", ")}
                        </div>
                      )}
                      <div>
                        <button
                          type="button"
                          onClick={applyRecipeSuggestedConfig}
                          className="rounded-md border border-emerald-500/40 px-3 py-2 text-xs text-emerald-100 hover:bg-emerald-500/10"
                        >
                          Reaplicar sugerencias de recipe
                        </button>
                        {Object.keys(recipeSuggestedTuning).length > 0 && (
                          <button
                            type="button"
                            onClick={applyRecipeSuggestedTuning}
                            className="ml-2 rounded-md border border-sky-500/40 px-3 py-2 text-xs text-sky-100 hover:bg-sky-500/10"
                          >
                            Aplicar tuning recomendado
                          </button>
                        )}
                      </div>
                    </div>
                  </FieldSection>
                )}

                {(probeStatus || probeRecommendedModelImpl || probeRecommendedRunner || probeReason || probeObservedAt) && (
                  <FieldSection
                    title="Probe"
                    description="Recomendacion verificada por probe para esta instalacion."
                  >
                    <div className="space-y-3 text-sm">
                      {probeStatusLabel && (
                        <div className="rounded-md border border-sky-500/20 bg-sky-500/10 px-3 py-2 text-sky-100">
                          {`probe=${probeStatusLabel}`}
                        </div>
                      )}
                      {(probeRecommendedModelImpl || probeRecommendedRunner) && (
                        <>
                          <div className="rounded-md border border-border bg-background px-3 py-2 text-xs text-muted-foreground">
                            {`recomendacion verificada`}
                            {probeRecommendedModelImpl ? ` · model_impl=${probeRecommendedModelImpl}` : ""}
                            {probeRecommendedRunner ? ` · runner=${probeRecommendedRunner}` : ""}
                          </div>
                          <div>
                            <button
                              type="button"
                              onClick={applyProbeRecommendation}
                              className="rounded-md border border-sky-500/40 px-3 py-2 text-xs text-sky-100 hover:bg-sky-500/10"
                            >
                              Aplicar recomendacion del probe
                            </button>
                          </div>
                        </>
                      )}
                      {differsFromProbe && (
                        <p className="text-xs text-amber-300">
                          La configuracion activa difiere de la recomendacion verificada por probe.
                        </p>
                      )}
                      {recipeDiffersFromProbe && (
                        <p className="text-xs text-amber-300">
                          La recipe base y la recomendacion final del probe no coinciden.
                        </p>
                      )}
                      {probeObservedAt && (
                        <p className="text-xs text-muted-foreground">{`observado=${probeObservedAt}`}</p>
                      )}
                      {probeOverrideHint && (
                        <p className="text-xs text-amber-300">{probeOverrideHint}</p>
                      )}
                      {probeReason && <p className="text-xs text-muted-foreground">{probeReason}</p>}
                    </div>
                  </FieldSection>
                )}

                <FieldSection
                  title="vLLM Runtime"
                  description="Solo knobs con impacto operativo directo: compatibilidad, contexto, memoria y concurrencia."
                >
                  {context && (
                    <div className="rounded-md border border-border bg-muted/20 px-3 py-2 text-xs text-muted-foreground">
                      <span className="font-medium text-foreground">Contexto del modelo:</span>{" "}
                      nativo {formatTokenCount(context.nativeContext)} · configurado {formatTokenCount(context.configuredContext)} ·
                      ventana operativa {formatTokenCount(context.maxInputTokens)} input / {formatTokenCount(context.maxOutputTokens)} output.
                      {" "}Input y output comparten la misma ventana total.
                    </div>
                  )}

                  <div className="grid gap-4 md:grid-cols-2">
                    <label className="block text-sm">
                      <span className="mb-1 block text-muted-foreground">Model impl</span>
                      <select
                        value={modelImpl}
                        onChange={(event) => setModelImpl(event.target.value as "auto" | "vllm" | "transformers" | "")}
                        className="w-full rounded-md border border-border bg-background px-3 py-2"
                      >
                        <option value="">heredar / auto</option>
                        <option value="auto">auto</option>
                        <option value="vllm">vllm</option>
                        <option value="transformers">transformers</option>
                      </select>
                      <FieldHint>`transformers` amplia cobertura; `vllm` fuerza la implementacion nativa si sabes que el modelo la soporta.</FieldHint>
                    </label>

                    <label className="block text-sm">
                      <span className="mb-1 block text-muted-foreground">Runner</span>
                      <select
                        value={runner}
                        onChange={(event) => setRunner(event.target.value as "generate" | "pooling" | "")}
                        className="w-full rounded-md border border-border bg-background px-3 py-2"
                      >
                        <option value="">autodetectar</option>
                        <option value="generate">generate</option>
                        <option value="pooling">pooling</option>
                      </select>
                      <FieldHint>`pooling` es para embeddings/rerank; no sirve chat generativo.</FieldHint>
                    </label>

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

                    <PresetNumberField
                      label="GPU memory utilization"
                      value={gpuMemoryUtilization}
                      onValueChange={(value) => setGpuMemoryUtilization(clampDecimalInput(value, 0.1, 0.99))}
                      presets={VLLM_GPU_MEMORY_UTILIZATION_PRESETS}
                      placeholder="heredar global"
                      min={0.1}
                      max={0.99}
                      step="0.01"
                      help="Normalmente `0.85-0.90`. Mas alto da mas KV cache, pero acerca el riesgo de OOM."
                    />

                    <PresetNumberField
                      label="Max concurrent sequences"
                      value={maxNumSeqs}
                      onValueChange={(value) => setMaxNumSeqs(clampIntegerInput(value, 1))}
                      presets={VLLM_MAX_NUM_SEQS_PRESETS}
                      placeholder="heredar global"
                      min={1}
                      step="1"
                      help="Limita cuantas peticiones procesa por iteracion. Subirlo mejora concurrencia pero usa mas memoria."
                    />
                  </div>

                  <label className="inline-flex items-center gap-2 text-sm text-muted-foreground">
                    <input
                      type="checkbox"
                      checked={enablePrefixCaching}
                      onChange={(event) => setEnablePrefixCaching(event.target.checked)}
                    />
                    prefix caching
                  </label>
                </FieldSection>

                {showVLLMCompatibility && (
                  <FieldSection
                    title="Compatibilidad avanzada"
                    description="Escapes puntuales para modelos problemáticos. No son knobs de tuning diario."
                  >
                    <div className="grid gap-4 md:grid-cols-2">
                      <label className="inline-flex items-center gap-2 text-sm text-muted-foreground">
                        <input
                          type="checkbox"
                          checked={trustRemoteCode}
                          onChange={(event) => setTrustRemoteCode(event.target.checked)}
                        />
                        trust remote code
                      </label>

                      <label className="block text-sm">
                        <span className="mb-1 block text-muted-foreground">Tool call parser</span>
                        <input
                          value={toolCallParser}
                          onChange={(event) => setToolCallParser(event.target.value)}
                          placeholder="hermes, qwen3_json, granite..."
                          className="w-full rounded-md border border-border bg-background px-3 py-2"
                        />
                      </label>

                      <label className="block text-sm">
                        <span className="mb-1 block text-muted-foreground">Reasoning parser</span>
                        <input
                          value={reasoningParser}
                          onChange={(event) => setReasoningParser(event.target.value)}
                          placeholder="deepseek_r1, qwen3..."
                          className="w-full rounded-md border border-border bg-background px-3 py-2"
                        />
                      </label>

                      <label className="block text-sm">
                        <span className="mb-1 block text-muted-foreground">HF overrides (JSON)</span>
                        <textarea
                          value={hfOverrides}
                          onChange={(event) => setHfOverrides(event.target.value)}
                          placeholder='{"architectures":["LlamaForCausalLM"]}'
                          rows={4}
                          className="w-full rounded-md border border-border bg-background px-3 py-2 font-mono text-xs"
                        />
                      </label>
                    </div>

                    <label className="block text-sm">
                      <span className="mb-1 block text-muted-foreground">Chat template</span>
                      <textarea
                        value={chatTemplate}
                        onChange={(event) => setChatTemplate(event.target.value)}
                        placeholder="ruta a template o contenido inline"
                        rows={4}
                        className="w-full rounded-md border border-border bg-background px-3 py-2 text-xs"
                      />
                      <FieldHint>Solo para repos que no traen `chat_template` correcto o cuando el probe lo pida.</FieldHint>
                    </label>
                  </FieldSection>
                )}
              </>
            )}

            {model?.backendType === "sglang" && (
              <FieldSection
                title="SGLang Runtime"
                description="Paralelismo, ventana de contexto y reserva de VRAM estática."
              >
                {context && (
                  <div className="rounded-md border border-border bg-muted/20 px-3 py-2 text-xs text-muted-foreground">
                    <span className="font-medium text-foreground">Contexto del modelo:</span>{" "}
                    nativo {formatTokenCount(context.nativeContext)} · configurado {formatTokenCount(context.configuredContext)}.
                  </div>
                )}
                <div className="grid gap-4 md:grid-cols-2">
                  <label className="block text-sm">
                    <span className="mb-1 block text-muted-foreground">Tensor parallel size</span>
                    <input
                      type="number"
                      min="1"
                      value={sglangTensorParallelSize}
                      onChange={(event) => setSGLangTensorParallelSize(event.target.value)}
                      placeholder="auto"
                      className="w-full rounded-md border border-border bg-background px-3 py-2"
                    />
                  </label>
                  <label className="block text-sm">
                    <span className="mb-1 block text-muted-foreground">Context length</span>
                    <input
                      type="number"
                      min="1"
                      value={sglangContextLength}
                      onChange={(event) => setSGLangContextLength(event.target.value)}
                      placeholder="usar el del modelo"
                      className="w-full rounded-md border border-border bg-background px-3 py-2"
                    />
                  </label>
                  <PresetNumberField
                    label="Mem fraction static"
                    value={sglangMemFractionStatic}
                    onValueChange={(value) => setSGLangMemFractionStatic(clampDecimalInput(value, 0.1, 0.99))}
                    presets={[0.7, 0.75, 0.8, 0.85, 0.9]}
                    placeholder="global"
                    min={0.1}
                    max={0.99}
                    step="0.01"
                    help="Porción de VRAM reservada por SGLang. Más alto mejora throughput, pero reduce margen para otros procesos."
                  />
                </div>
                <div className="grid gap-3 md:grid-cols-2">
                  <label className="inline-flex items-center gap-2 text-sm text-muted-foreground">
                    <input
                      type="checkbox"
                      checked={sglangTrustRemoteCode}
                      onChange={(event) => setSGLangTrustRemoteCode(event.target.checked)}
                    />
                    trust remote code
                  </label>
                  <label className="inline-flex items-center gap-2 text-sm text-muted-foreground">
                    <input
                      type="checkbox"
                      checked={disableRadixCache}
                      onChange={(event) => setDisableRadixCache(event.target.checked)}
                    />
                    disable radix cache
                  </label>
                </div>
              </FieldSection>
            )}

            {model?.backendType === "llama_cpp" && (
              <FieldSection
                title="llama.cpp Runtime"
                description="Lo relevante aquí es cuánto sube a GPU, el contexto y si actúa como modelo de embeddings."
              >
                {context && (
                  <div className="rounded-md border border-border bg-muted/20 px-3 py-2 text-xs text-muted-foreground">
                    <span className="font-medium text-foreground">Contexto del modelo:</span>{" "}
                    nativo {formatTokenCount(context.nativeContext)} · configurado {formatTokenCount(context.configuredContext)}.
                  </div>
                )}
                <div className="grid gap-4 md:grid-cols-2">
                  <label className="block text-sm">
                    <span className="mb-1 block text-muted-foreground">GPU layers</span>
                    <input
                      type="number"
                      min="0"
                      value={llamaGpuLayers}
                      onChange={(event) => setLlamaGpuLayers(event.target.value)}
                      placeholder="global"
                      className="w-full rounded-md border border-border bg-background px-3 py-2"
                    />
                    <FieldHint>`0` fuerza CPU; subirlo mueve más capas a GPU.</FieldHint>
                  </label>
                  <label className="block text-sm">
                    <span className="mb-1 block text-muted-foreground">Context size</span>
                    <input
                      type="number"
                      min="1"
                      value={llamaCtxSize}
                      onChange={(event) => setLlamaCtxSize(event.target.value)}
                      placeholder="global"
                      className="w-full rounded-md border border-border bg-background px-3 py-2"
                    />
                  </label>
                </div>
                <div className="grid gap-3 md:grid-cols-2">
                  <label className="inline-flex items-center gap-2 text-sm text-muted-foreground">
                    <input
                      type="checkbox"
                      checked={llamaFlashAttn}
                      onChange={(event) => setLlamaFlashAttn(event.target.checked)}
                    />
                    flash attention
                  </label>
                  <label className="inline-flex items-center gap-2 text-sm text-muted-foreground">
                    <input
                      type="checkbox"
                      checked={llamaEmbedding}
                      onChange={(event) => setLlamaEmbedding(event.target.checked)}
                    />
                    embedding mode
                  </label>
                </div>
              </FieldSection>
            )}

            {model?.backendType === "bitnet" && (
              <FieldSection
                title="BitNet Runtime"
                description="BitNet comparte casi todos los knobs relevantes con los runtimes GGUF: capas en GPU y contexto."
              >
                {context && (
                  <div className="rounded-md border border-border bg-muted/20 px-3 py-2 text-xs text-muted-foreground">
                    <span className="font-medium text-foreground">Contexto del modelo:</span>{" "}
                    nativo {formatTokenCount(context.nativeContext)} · configurado {formatTokenCount(context.configuredContext)}.
                  </div>
                )}
                <div className="grid gap-4 md:grid-cols-2">
                  <label className="block text-sm">
                    <span className="mb-1 block text-muted-foreground">GPU layers</span>
                    <input
                      type="number"
                      min="0"
                      value={bitnetGpuLayers}
                      onChange={(event) => setBitnetGpuLayers(event.target.value)}
                      placeholder="global"
                      className="w-full rounded-md border border-border bg-background px-3 py-2"
                    />
                  </label>
                  <label className="block text-sm">
                    <span className="mb-1 block text-muted-foreground">Context size</span>
                    <input
                      type="number"
                      min="1"
                      value={bitnetCtxSize}
                      onChange={(event) => setBitnetCtxSize(event.target.value)}
                      placeholder="global"
                      className="w-full rounded-md border border-border bg-background px-3 py-2"
                    />
                  </label>
                </div>
                <label className="inline-flex items-center gap-2 text-sm text-muted-foreground">
                  <input
                    type="checkbox"
                    checked={bitnetFlashAttn}
                    onChange={(event) => setBitnetFlashAttn(event.target.checked)}
                  />
                  flash attention
                </label>
              </FieldSection>
            )}

            {model?.backendType === "tensorrt_llm" && (
              <FieldSection
                title="TensorRT-LLM Runtime"
                description="Mantengo fuera los flags de wiring y despliegue; aquí solo salen los que cambian servicio y capacidad."
              >
                {context && (
                  <div className="rounded-md border border-border bg-muted/20 px-3 py-2 text-xs text-muted-foreground">
                    <span className="font-medium text-foreground">Contexto del modelo:</span>{" "}
                    nativo {formatTokenCount(context.nativeContext)} · configurado {formatTokenCount(context.configuredContext)}.
                  </div>
                )}
                <div className="grid gap-4 md:grid-cols-2">
                  <label className="block text-sm">
                    <span className="mb-1 block text-muted-foreground">Max batch size</span>
                    <input
                      type="number"
                      min="1"
                      value={trtMaxBatchSize}
                      onChange={(event) => setTRTMaxBatchSize(event.target.value)}
                      placeholder="global"
                      className="w-full rounded-md border border-border bg-background px-3 py-2"
                    />
                  </label>
                  <label className="block text-sm">
                    <span className="mb-1 block text-muted-foreground">Context length</span>
                    <input
                      type="number"
                      min="1"
                      value={trtContextLength}
                      onChange={(event) => setTRTContextLength(event.target.value)}
                      placeholder="deducido del engine"
                      className="w-full rounded-md border border-border bg-background px-3 py-2"
                    />
                  </label>
                </div>
                <label className="inline-flex items-center gap-2 text-sm text-muted-foreground">
                  <input
                    type="checkbox"
                    checked={trtTrustRemoteCode}
                    onChange={(event) => setTRTTrustRemoteCode(event.target.checked)}
                  />
                  trust remote code
                </label>
              </FieldSection>
            )}

            {model?.backendType === "whisper" && (
              <FieldSection
                title="Whisper Runtime"
                description="La única configuración per-model con impacto claro aquí es la diarización."
              >
                <label className="inline-flex items-center gap-2 text-sm text-muted-foreground">
                  <input
                    type="checkbox"
                    checked={whisperDiarizationEnabled}
                    onChange={(event) => setWhisperDiarizationEnabled(event.target.checked)}
                  />
                  habilitar diarización
                </label>
                <label className="block text-sm">
                  <span className="mb-1 block text-muted-foreground">Diarization model id</span>
                  <input
                    value={whisperDiarizationModelId}
                    onChange={(event) => setWhisperDiarizationModelId(event.target.value)}
                    placeholder="pyannote/speaker-diarization-3.1"
                    className="w-full rounded-md border border-border bg-background px-3 py-2"
                  />
                  <FieldHint>Déjalo vacío para usar el modelo por defecto del servidor.</FieldHint>
                </label>
              </FieldSection>
            )}

            {model && ["diffusers", "tts", "ollama", "acestep"].includes(model.backendType) && (
              <FieldSection
                title="Sin tuning per-model"
                description="Para este backend no estamos exponiendo knobs per-model porque los relevantes hoy son globales, request-level o internos al runtime."
              >
                <p className="text-sm text-muted-foreground">
                  Si aparece una necesidad operativa real, añadiremos el parámetro concreto en lugar de abrir todo el runtime.
                </p>
              </FieldSection>
            )}

            {isAdmin && (
              <FieldSection
                title="Grupos"
                description="Asigna o retira este modelo de grupos de acceso. Solo visible para system_admin."
              >
                {loadingGroups ? (
                  <div className="space-y-2">
                    {Array.from({ length: 3 }).map((_, i) => (
                      <div key={`group-skel-${i}`} className="h-6 animate-pulse rounded bg-muted" />
                    ))}
                  </div>
                ) : allGroups.length === 0 ? (
                  <p className="text-sm text-muted-foreground">No hay grupos configurados.</p>
                ) : (
                  <div className="space-y-2">
                    {allGroups.map((group) => (
                      <label key={group.id} className="flex items-center gap-2 text-sm">
                        <input
                          type="checkbox"
                          checked={selectedGroupIds.has(group.id)}
                          onChange={(e) => {
                            setSelectedGroupIds((prev) => {
                              const next = new Set(prev)
                              if (e.target.checked) {
                                next.add(group.id)
                              } else {
                                next.delete(group.id)
                              }
                              return next
                            })
                          }}
                        />
                        <span className="font-medium">{group.name}</span>
                        {group.description && (
                          <span className="text-muted-foreground">— {group.description}</span>
                        )}
                        {group.isDefault && (
                          <span className="rounded-full bg-blue-500/15 px-1.5 py-0.5 text-xs text-blue-600 dark:text-blue-400">
                            default
                          </span>
                        )}
                      </label>
                    ))}
                  </div>
                )}
              </FieldSection>
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
