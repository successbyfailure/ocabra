import type { ReactNode } from "react"
import { useEffect, useState } from "react"
import * as Dialog from "@radix-ui/react-dialog"
import { X } from "lucide-react"
import { formatTokenCount, getModelContextSummary } from "@/lib/modelContext"
import { ScheduleEditor } from "@/components/models/ScheduleEditor"
import { getProbeOverrideHint, getProbeStatusLabel } from "@/lib/vllmProbe"
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

const VLLM_GPU_MEMORY_UTILIZATION_PRESETS = [0.75, 0.8, 0.85, 0.9, 0.92, 0.95]
const VLLM_MAX_NUM_SEQS_PRESETS = [1, 2, 4, 8, 12, 16, 24, 32]

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

export function ModelConfigModal({ model, gpus, open, onOpenChange, onSave }: ModelConfigModalProps) {
  const [loadPolicy, setLoadPolicy] = useState<LoadPolicy>("on_demand")
  const [preferredGpu, setPreferredGpu] = useState<number | null>(null)
  const [autoReload, setAutoReload] = useState(false)
  const [schedules, setSchedules] = useState<EvictionSchedule[]>([])
  const [tensorParallelSize, setTensorParallelSize] = useState("")
  const [maxModelLen, setMaxModelLen] = useState("")
  const [modelImpl, setModelImpl] = useState<"auto" | "vllm" | "transformers" | "">("")
  const [runner, setRunner] = useState<"generate" | "pooling" | "">("")
  const [hfOverrides, setHfOverrides] = useState("")
  const [chatTemplate, setChatTemplate] = useState("")
  const [chatTemplateContentFormat, setChatTemplateContentFormat] = useState("")
  const [generationConfig, setGenerationConfig] = useState("")
  const [overrideGenerationConfig, setOverrideGenerationConfig] = useState("")
  const [toolCallParser, setToolCallParser] = useState("")
  const [toolParserPlugin, setToolParserPlugin] = useState("")
  const [reasoningParser, setReasoningParser] = useState("")
  const [languageModelOnly, setLanguageModelOnly] = useState<"inherit" | "on" | "off">("inherit")
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
  const recipeId = (model?.extraConfig?.vllm as VLLMConfig | undefined)?.recipeId ?? null
  const recipeNotes = ((model?.extraConfig?.vllm as VLLMConfig | undefined)?.recipeNotes ?? []) as string[]
  const recipeModelImpl = (model?.extraConfig?.vllm as VLLMConfig | undefined)?.recipeModelImpl ?? null
  const recipeRunner = (model?.extraConfig?.vllm as VLLMConfig | undefined)?.recipeRunner ?? null
  const recipeSuggestedConfig = ((model?.extraConfig?.vllm as VLLMConfig | undefined)?.suggestedConfig ?? {}) as Record<string, unknown>
  const recipeSuggestedTuning = ((model?.extraConfig?.vllm as VLLMConfig | undefined)?.suggestedTuning ?? {}) as Record<string, unknown>
  const probeStatus = (model?.extraConfig?.vllm as VLLMConfig | undefined)?.probeStatus ?? null
  const probeReason = (model?.extraConfig?.vllm as VLLMConfig | undefined)?.probeReason ?? null
  const probeObservedAt = (model?.extraConfig?.vllm as VLLMConfig | undefined)?.probeObservedAt ?? null
  const probeRecommendedModelImpl = (model?.extraConfig?.vllm as VLLMConfig | undefined)?.probeRecommendedModelImpl ?? null
  const probeRecommendedRunner = (model?.extraConfig?.vllm as VLLMConfig | undefined)?.probeRecommendedRunner ?? null

  useEffect(() => {
    if (!model) return
    setLoadPolicy(model.loadPolicy)
    setPreferredGpu(model.preferredGpu)
    setAutoReload(model.autoReload)
    setSchedules(model.schedules ?? [])
    const vllm = (model.extraConfig?.vllm as Record<string, unknown> | undefined) ?? {}
    setModelImpl(vllm.modelImpl == null ? "" : (String(vllm.modelImpl) as "auto" | "vllm" | "transformers"))
    setRunner(vllm.runner == null ? "" : (String(vllm.runner) as "generate" | "pooling"))
    setHfOverrides(vllm.hfOverrides == null ? "" : typeof vllm.hfOverrides === "string" ? vllm.hfOverrides : JSON.stringify(vllm.hfOverrides))
    setChatTemplate(vllm.chatTemplate == null ? "" : String(vllm.chatTemplate))
    setChatTemplateContentFormat(vllm.chatTemplateContentFormat == null ? "" : String(vllm.chatTemplateContentFormat))
    setGenerationConfig(vllm.generationConfig == null ? "" : String(vllm.generationConfig))
    setOverrideGenerationConfig(
      vllm.overrideGenerationConfig == null
        ? ""
        : typeof vllm.overrideGenerationConfig === "string"
          ? vllm.overrideGenerationConfig
          : JSON.stringify(vllm.overrideGenerationConfig),
    )
    setToolCallParser(vllm.toolCallParser == null ? "" : String(vllm.toolCallParser))
    setToolParserPlugin(vllm.toolParserPlugin == null ? "" : String(vllm.toolParserPlugin))
    setReasoningParser(vllm.reasoningParser == null ? "" : String(vllm.reasoningParser))
    setLanguageModelOnly(
      vllm.languageModelOnly == null ? "inherit" : vllm.languageModelOnly ? "on" : "off",
    )
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
                tensorParallelSize: tensorParallelSize === "" ? null : Number(tensorParallelSize),
                maxModelLen: maxModelLen === "" ? null : Number(maxModelLen),
                modelImpl: modelImpl === "" ? null : modelImpl,
                runner: runner === "" ? null : runner,
                hfOverrides: hfOverrides === "" ? null : hfOverrides,
                chatTemplate: chatTemplate === "" ? null : chatTemplate,
                chatTemplateContentFormat:
                  chatTemplateContentFormat === "" ? null : chatTemplateContentFormat,
                generationConfig: generationConfig === "" ? null : generationConfig,
                overrideGenerationConfig:
                  overrideGenerationConfig === "" ? null : overrideGenerationConfig,
                toolCallParser: toolCallParser === "" ? null : toolCallParser,
                toolParserPlugin: toolParserPlugin === "" ? null : toolParserPlugin,
                reasoningParser: reasoningParser === "" ? null : reasoningParser,
                languageModelOnly:
                  languageModelOnly === "inherit" ? null : languageModelOnly === "on",
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

  const applyRecipeSuggestedConfig = () => {
    const suggested = recipeSuggestedConfig
    if (modelImpl === "" && typeof (model?.extraConfig?.vllm as VLLMConfig | undefined)?.modelImpl === "string") {
      setModelImpl((model?.extraConfig?.vllm as VLLMConfig).modelImpl as "auto" | "vllm" | "transformers")
    }
    if (runner === "" && typeof (model?.extraConfig?.vllm as VLLMConfig | undefined)?.runner === "string") {
      setRunner((model?.extraConfig?.vllm as VLLMConfig).runner as "generate" | "pooling")
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
    if (typeof tuning.enable_chunked_prefill === "boolean") {
      setEnableChunkedPrefillMode(tuning.enable_chunked_prefill ? "on" : "off")
    }
    if (typeof tuning.enableChunkedPrefill === "boolean") {
      setEnableChunkedPrefillMode(tuning.enableChunkedPrefill ? "on" : "off")
    }
    if (typeof tuning.enforce_eager === "boolean") setEnforceEager(tuning.enforce_eager)
    if (typeof tuning.enforceEager === "boolean") setEnforceEager(tuning.enforceEager)
    if (typeof tuning.language_model_only === "boolean") {
      setLanguageModelOnly(tuning.language_model_only ? "on" : "off")
    }
    if (typeof tuning.languageModelOnly === "boolean") {
      setLanguageModelOnly(tuning.languageModelOnly ? "on" : "off")
    }
  }

  const applyProbeRecommendation = () => {
    if (probeRecommendedModelImpl != null) {
      setModelImpl(probeRecommendedModelImpl)
    }
    if (probeRecommendedRunner != null) {
      setRunner(probeRecommendedRunner)
    }
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
                  title="vLLM Basico"
                  description="Parametros con impacto directo en compatibilidad, reparto entre GPUs y limite de contexto."
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
                  </div>

                  <div className="grid gap-4 md:grid-cols-2">
                    <label className="block text-sm">
                      <span className="mb-1 block text-muted-foreground">HF overrides (JSON)</span>
                      <textarea
                        value={hfOverrides}
                        onChange={(event) => setHfOverrides(event.target.value)}
                        placeholder='{"architectures":["LlamaForCausalLM"]}'
                        rows={4}
                        className="w-full rounded-md border border-border bg-background px-3 py-2 font-mono text-xs"
                      />
                      <FieldHint>Escape hatch para compatibilidad puntual. Mantenerlo pequeno y especifico.</FieldHint>
                    </label>

                    <label className="block text-sm">
                      <span className="mb-1 block text-muted-foreground">Chat template</span>
                      <textarea
                        value={chatTemplate}
                        onChange={(event) => setChatTemplate(event.target.value)}
                        placeholder="ruta a template o contenido inline"
                        rows={4}
                        className="w-full rounded-md border border-border bg-background px-3 py-2 text-xs"
                      />
                      <FieldHint>Util si el repo no trae `chat_template` o el prompting nativo no encaja bien.</FieldHint>
                    </label>
                  </div>

                  <div className="grid gap-4 md:grid-cols-2">
                    <label className="block text-sm">
                      <span className="mb-1 block text-muted-foreground">Chat template content format</span>
                      <input
                        value={chatTemplateContentFormat}
                        onChange={(event) => setChatTemplateContentFormat(event.target.value)}
                        placeholder="auto, string, openai..."
                        className="w-full rounded-md border border-border bg-background px-3 py-2"
                      />
                    </label>

                    <label className="block text-sm">
                      <span className="mb-1 block text-muted-foreground">Generation config</span>
                      <input
                        value={generationConfig}
                        onChange={(event) => setGenerationConfig(event.target.value)}
                        placeholder="auto, vllm, ruta..."
                        className="w-full rounded-md border border-border bg-background px-3 py-2"
                      />
                    </label>
                  </div>

                  <label className="block text-sm">
                    <span className="mb-1 block text-muted-foreground">Override generation config (JSON)</span>
                    <textarea
                      value={overrideGenerationConfig}
                      onChange={(event) => setOverrideGenerationConfig(event.target.value)}
                      placeholder='{"temperature":0.2}'
                      rows={3}
                      className="w-full rounded-md border border-border bg-background px-3 py-2 font-mono text-xs"
                    />
                  </label>

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

                  <div className="grid gap-4 md:grid-cols-2">
                    <label className="block text-sm">
                      <span className="mb-1 block text-muted-foreground">Tool call parser</span>
                      <input
                        value={toolCallParser}
                        onChange={(event) => setToolCallParser(event.target.value)}
                        placeholder="hermes, qwen3_json, granite..."
                        className="w-full rounded-md border border-border bg-background px-3 py-2"
                      />
                      <FieldHint>Si lo rellenas, oCabra activa auto tool choice en el worker vLLM.</FieldHint>
                    </label>

                    <label className="block text-sm">
                      <span className="mb-1 block text-muted-foreground">Reasoning parser</span>
                      <input
                        value={reasoningParser}
                        onChange={(event) => setReasoningParser(event.target.value)}
                        placeholder="deepseek_r1, qwen3..."
                        className="w-full rounded-md border border-border bg-background px-3 py-2"
                      />
                      <FieldHint>Necesario solo para familias que exponen reasoning con parser dedicado.</FieldHint>
                    </label>
                  </div>

                  <div className="grid gap-4 md:grid-cols-2">
                    <label className="block text-sm">
                      <span className="mb-1 block text-muted-foreground">Tool parser plugin</span>
                      <input
                        value={toolParserPlugin}
                        onChange={(event) => setToolParserPlugin(event.target.value)}
                        placeholder="plugin Python opcional"
                        className="w-full rounded-md border border-border bg-background px-3 py-2"
                      />
                    </label>

                    <label className="block text-sm">
                      <span className="mb-1 block text-muted-foreground">Language model only</span>
                      <select
                        value={languageModelOnly}
                        onChange={(event) => setLanguageModelOnly(event.target.value as "inherit" | "on" | "off")}
                        className="w-full rounded-md border border-border bg-background px-3 py-2"
                      >
                        <option value="inherit">heredar</option>
                        <option value="on">forzar on</option>
                        <option value="off">forzar off</option>
                      </select>
                      <FieldHint>Para modelos multimodales: ahorra VRAM si solo quieres texto.</FieldHint>
                    </label>
                  </div>
                </FieldSection>

                <FieldSection
                  title="Rendimiento"
                  description="Toca concurrencia, throughput y cuanta VRAM reserva vLLM."
                >
                  <div className="grid gap-4 md:grid-cols-2">
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
