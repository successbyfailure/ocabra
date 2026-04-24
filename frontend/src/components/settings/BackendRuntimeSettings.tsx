import { useEffect, useState } from "react"
import { ChevronDown, ChevronRight } from "lucide-react"
import { toast } from "sonner"
import type { ServerConfig } from "@/types"

interface BackendRuntimeSettingsProps {
  config: ServerConfig
  onSave: (patch: Partial<ServerConfig>) => Promise<void>
}

function SectionHeader({
  title,
  open,
  onToggle,
}: {
  title: string
  open: boolean
  onToggle: () => void
}) {
  return (
    <button
      type="button"
      onClick={onToggle}
      className="flex w-full items-center justify-between gap-2 rounded-md bg-muted/30 px-3 py-2 text-sm font-medium cursor-pointer hover:bg-muted/50 transition-colors"
    >
      {title}
      {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
    </button>
  )
}

export function BackendRuntimeSettings({ config, onSave }: BackendRuntimeSettingsProps) {
  const [vllmGpuMemoryUtilization, setVLLMGpuMemoryUtilization] = useState(config.vllmGpuMemoryUtilization)
  const [vllmMaxNumSeqs, setVLLMMaxNumSeqs] = useState(config.vllmMaxNumSeqs ?? 0)
  const [vllmMaxNumBatchedTokens, setVLLMMaxNumBatchedTokens] = useState(config.vllmMaxNumBatchedTokens ?? 0)
  const [vllmEnablePrefixCaching, setVLLMEnablePrefixCaching] = useState(config.vllmEnablePrefixCaching)
  const [vllmEnforceEager, setVLLMEnforceEager] = useState(config.vllmEnforceEager)
  const [sglangMemFractionStatic, setSGLangMemFractionStatic] = useState(config.sglangMemFractionStatic)
  const [sglangContextLength, setSGLangContextLength] = useState(config.sglangContextLength ?? 0)
  const [sglangDisableRadixCache, setSGLangDisableRadixCache] = useState(config.sglangDisableRadixCache)
  const [llamaCppGpuLayers, setLlamaCppGpuLayers] = useState(config.llamaCppGpuLayers)
  const [llamaCppCtxSize, setLlamaCppCtxSize] = useState(config.llamaCppCtxSize)
  const [llamaCppFlashAttn, setLlamaCppFlashAttn] = useState(config.llamaCppFlashAttn)
  const [bitnetGpuLayers, setBitnetGpuLayers] = useState(config.bitnetGpuLayers)
  const [bitnetCtxSize, setBitnetCtxSize] = useState(config.bitnetCtxSize)
  const [bitnetFlashAttn, setBitnetFlashAttn] = useState(config.bitnetFlashAttn)
  const [diffusersTorchDtype, setDiffusersTorchDtype] = useState(config.diffusersTorchDtype)
  const [diffusersOffloadMode, setDiffusersOffloadMode] = useState(config.diffusersOffloadMode)
  const [diffusersEnableTorchCompile, setDiffusersEnableTorchCompile] = useState(config.diffusersEnableTorchCompile)
  const [diffusersEnableXformers, setDiffusersEnableXformers] = useState(config.diffusersEnableXformers)
  const [diffusersAllowTf32, setDiffusersAllowTf32] = useState(config.diffusersAllowTf32)
  const [tensorrtLlmEnabled, setTensorRTLLMEnabled] = useState(config.tensorrtLlmEnabled)
  const [tensorrtLlmMaxBatchSize, setTensorRTLLMMaxBatchSize] = useState(config.tensorrtLlmMaxBatchSize ?? 0)
  const [tensorrtLlmContextLength, setTensorRTLLMContextLength] = useState(config.tensorrtLlmContextLength ?? 0)
  const [openaiAudioMaxPartSizeMb, setOpenAIAudioMaxPartSizeMb] = useState(config.openaiAudioMaxPartSizeMb)
  const [whisperStartupTimeoutSeconds, setWhisperStartupTimeoutSeconds] = useState(config.whisperStartupTimeoutSeconds)

  const [openSections, setOpenSections] = useState<Record<string, boolean>>({
    vllm: true,
    sglang: false,
    llamacpp: false,
    diffusers: false,
    audio: false,
    tensorrt: false,
  })

  const toggleSection = (key: string) =>
    setOpenSections((prev) => ({ ...prev, [key]: !prev[key] }))

  useEffect(() => {
    setVLLMGpuMemoryUtilization(config.vllmGpuMemoryUtilization)
    setVLLMMaxNumSeqs(config.vllmMaxNumSeqs ?? 0)
    setVLLMMaxNumBatchedTokens(config.vllmMaxNumBatchedTokens ?? 0)
    setVLLMEnablePrefixCaching(config.vllmEnablePrefixCaching)
    setVLLMEnforceEager(config.vllmEnforceEager)
    setSGLangMemFractionStatic(config.sglangMemFractionStatic)
    setSGLangContextLength(config.sglangContextLength ?? 0)
    setSGLangDisableRadixCache(config.sglangDisableRadixCache)
    setLlamaCppGpuLayers(config.llamaCppGpuLayers)
    setLlamaCppCtxSize(config.llamaCppCtxSize)
    setLlamaCppFlashAttn(config.llamaCppFlashAttn)
    setBitnetGpuLayers(config.bitnetGpuLayers)
    setBitnetCtxSize(config.bitnetCtxSize)
    setBitnetFlashAttn(config.bitnetFlashAttn)
    setDiffusersTorchDtype(config.diffusersTorchDtype)
    setDiffusersOffloadMode(config.diffusersOffloadMode)
    setDiffusersEnableTorchCompile(config.diffusersEnableTorchCompile)
    setDiffusersEnableXformers(config.diffusersEnableXformers)
    setDiffusersAllowTf32(config.diffusersAllowTf32)
    setTensorRTLLMEnabled(config.tensorrtLlmEnabled)
    setTensorRTLLMMaxBatchSize(config.tensorrtLlmMaxBatchSize ?? 0)
    setTensorRTLLMContextLength(config.tensorrtLlmContextLength ?? 0)
    setOpenAIAudioMaxPartSizeMb(config.openaiAudioMaxPartSizeMb)
    setWhisperStartupTimeoutSeconds(config.whisperStartupTimeoutSeconds)
  }, [config])

  const save = async () => {
    try {
      await onSave({
        vllmGpuMemoryUtilization,
        vllmMaxNumSeqs: vllmMaxNumSeqs > 0 ? vllmMaxNumSeqs : null,
        vllmMaxNumBatchedTokens: vllmMaxNumBatchedTokens > 0 ? vllmMaxNumBatchedTokens : null,
        vllmEnablePrefixCaching,
        vllmEnforceEager,
        sglangMemFractionStatic,
        sglangContextLength: sglangContextLength > 0 ? sglangContextLength : null,
        sglangDisableRadixCache,
        llamaCppGpuLayers,
        llamaCppCtxSize,
        llamaCppFlashAttn,
        bitnetGpuLayers,
        bitnetCtxSize,
        bitnetFlashAttn,
        diffusersTorchDtype,
        diffusersOffloadMode,
        diffusersEnableTorchCompile,
        diffusersEnableXformers,
        diffusersAllowTf32,
        tensorrtLlmEnabled,
        tensorrtLlmMaxBatchSize: tensorrtLlmMaxBatchSize > 0 ? tensorrtLlmMaxBatchSize : null,
        tensorrtLlmContextLength: tensorrtLlmContextLength > 0 ? tensorrtLlmContextLength : null,
        openaiAudioMaxPartSizeMb,
        whisperStartupTimeoutSeconds,
      })
      toast.success("Backend runtime settings guardadas")
    } catch {
      // page-level toast is shown in Settings
    }
  }

  const inputClass = "mt-1 w-full rounded-md border border-border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
  const checkboxClass = "accent-primary h-4 w-4"

  return (
    <section className="space-y-3 rounded-lg border border-border bg-card p-4">
      <h2 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">Backend defaults</h2>

      {/* vLLM */}
      <div className="space-y-2">
        <SectionHeader title="vLLM" open={openSections.vllm} onToggle={() => toggleSection("vllm")} />
        {openSections.vllm && (
          <div className="grid gap-3 px-3 py-2 md:grid-cols-2">
            <label className="block text-sm text-muted-foreground">
              GPU memory utilization
              <input type="number" min={0.1} max={0.99} step="0.01" value={vllmGpuMemoryUtilization}
                onChange={(e) => setVLLMGpuMemoryUtilization(Number(e.target.value))} className={inputClass} />
              <p className="text-xs text-muted-foreground/70 mt-1">Fraccion de VRAM que vLLM puede usar. Reducir si hay OOM.</p>
            </label>
            <label className="block text-sm text-muted-foreground">
              Max concurrent sequences
              <input type="number" min={0} value={vllmMaxNumSeqs}
                onChange={(e) => setVLLMMaxNumSeqs(Number(e.target.value))} className={inputClass} />
              <p className="text-xs text-muted-foreground/70 mt-1">Maximo de secuencias procesadas en paralelo (0 = default).</p>
            </label>
            <label className="block text-sm text-muted-foreground">
              Max batched tokens
              <input type="number" min={0} value={vllmMaxNumBatchedTokens}
                onChange={(e) => setVLLMMaxNumBatchedTokens(Number(e.target.value))} className={inputClass} />
            </label>
            <div className="flex flex-col gap-2 justify-end">
              <label className="inline-flex items-center gap-2 text-sm text-muted-foreground">
                <input type="checkbox" checked={vllmEnablePrefixCaching} className={checkboxClass}
                  onChange={(e) => setVLLMEnablePrefixCaching(e.target.checked)} />
                Prefix caching
              </label>
              <label className="inline-flex items-center gap-2 text-sm text-muted-foreground">
                <input type="checkbox" checked={vllmEnforceEager} className={checkboxClass}
                  onChange={(e) => setVLLMEnforceEager(e.target.checked)} />
                Enforce eager
              </label>
            </div>
          </div>
        )}
      </div>

      {/* SGLang */}
      <div className="space-y-2">
        <SectionHeader title="SGLang" open={openSections.sglang} onToggle={() => toggleSection("sglang")} />
        {openSections.sglang && (
          <div className="grid gap-3 px-3 py-2 md:grid-cols-2">
            <label className="block text-sm text-muted-foreground">
              Mem fraction static
              <input type="number" min={0.1} max={0.99} step="0.01" value={sglangMemFractionStatic}
                onChange={(e) => setSGLangMemFractionStatic(Number(e.target.value))} className={inputClass} />
            </label>
            <label className="block text-sm text-muted-foreground">
              Context length
              <input type="number" min={0} value={sglangContextLength}
                onChange={(e) => setSGLangContextLength(Number(e.target.value))} className={inputClass} />
            </label>
            <label className="inline-flex items-center gap-2 text-sm text-muted-foreground">
              <input type="checkbox" checked={sglangDisableRadixCache} className={checkboxClass}
                onChange={(e) => setSGLangDisableRadixCache(e.target.checked)} />
              Disable radix cache
            </label>
          </div>
        )}
      </div>

      {/* llama.cpp / BitNet */}
      <div className="space-y-2">
        <SectionHeader title="llama.cpp / BitNet" open={openSections.llamacpp} onToggle={() => toggleSection("llamacpp")} />
        {openSections.llamacpp && (
          <div className="grid gap-3 px-3 py-2 md:grid-cols-2">
            <label className="block text-sm text-muted-foreground">
              llama.cpp GPU layers
              <input type="number" min={0} value={llamaCppGpuLayers}
                onChange={(e) => setLlamaCppGpuLayers(Number(e.target.value))} className={inputClass} />
            </label>
            <label className="block text-sm text-muted-foreground">
              llama.cpp ctx size
              <input type="number" min={1} value={llamaCppCtxSize}
                onChange={(e) => setLlamaCppCtxSize(Number(e.target.value))} className={inputClass} />
            </label>
            <label className="inline-flex items-center gap-2 text-sm text-muted-foreground">
              <input type="checkbox" checked={llamaCppFlashAttn} className={checkboxClass}
                onChange={(e) => setLlamaCppFlashAttn(e.target.checked)} />
              llama.cpp flash attention
            </label>
            <div />
            <label className="block text-sm text-muted-foreground">
              BitNet GPU layers
              <input type="number" min={0} value={bitnetGpuLayers}
                onChange={(e) => setBitnetGpuLayers(Number(e.target.value))} className={inputClass} />
            </label>
            <label className="block text-sm text-muted-foreground">
              BitNet ctx size
              <input type="number" min={1} value={bitnetCtxSize}
                onChange={(e) => setBitnetCtxSize(Number(e.target.value))} className={inputClass} />
            </label>
            <label className="inline-flex items-center gap-2 text-sm text-muted-foreground">
              <input type="checkbox" checked={bitnetFlashAttn} className={checkboxClass}
                onChange={(e) => setBitnetFlashAttn(e.target.checked)} />
              BitNet flash attention
            </label>
          </div>
        )}
      </div>

      {/* Diffusers */}
      <div className="space-y-2">
        <SectionHeader title="Diffusers" open={openSections.diffusers} onToggle={() => toggleSection("diffusers")} />
        {openSections.diffusers && (
          <div className="grid gap-3 px-3 py-2 md:grid-cols-2">
            <label className="block text-sm text-muted-foreground">
              Torch dtype
              <select value={diffusersTorchDtype} onChange={(e) => setDiffusersTorchDtype(e.target.value)} className={inputClass}>
                <option value="auto">auto</option>
                <option value="float16">float16</option>
                <option value="bfloat16">bfloat16</option>
                <option value="float32">float32</option>
              </select>
            </label>
            <label className="block text-sm text-muted-foreground">
              Offload mode
              <select value={diffusersOffloadMode} onChange={(e) => setDiffusersOffloadMode(e.target.value)} className={inputClass}>
                <option value="none">none</option>
                <option value="model">model</option>
                <option value="sequential">sequential</option>
              </select>
            </label>
            <div className="flex flex-col gap-2">
              <label className="inline-flex items-center gap-2 text-sm text-muted-foreground">
                <input type="checkbox" checked={diffusersEnableTorchCompile} className={checkboxClass}
                  onChange={(e) => setDiffusersEnableTorchCompile(e.target.checked)} />
                torch.compile
              </label>
              <label className="inline-flex items-center gap-2 text-sm text-muted-foreground">
                <input type="checkbox" checked={diffusersEnableXformers} className={checkboxClass}
                  onChange={(e) => setDiffusersEnableXformers(e.target.checked)} />
                xFormers
              </label>
              <label className="inline-flex items-center gap-2 text-sm text-muted-foreground">
                <input type="checkbox" checked={diffusersAllowTf32} className={checkboxClass}
                  onChange={(e) => setDiffusersAllowTf32(e.target.checked)} />
                Allow TF32
              </label>
            </div>
          </div>
        )}
      </div>

      {/* Audio / Whisper */}
      <div className="space-y-2">
        <SectionHeader title="Audio / Whisper" open={openSections.audio} onToggle={() => toggleSection("audio")} />
        {openSections.audio && (
          <div className="grid gap-3 px-3 py-2 md:grid-cols-2">
            <label className="block text-sm text-muted-foreground">
              Audio max part size (MB)
              <input type="number" min={1} value={openaiAudioMaxPartSizeMb}
                onChange={(e) => setOpenAIAudioMaxPartSizeMb(Number(e.target.value))} className={inputClass} />
            </label>
            <label className="block text-sm text-muted-foreground">
              Whisper startup timeout (s)
              <input type="number" min={1} value={whisperStartupTimeoutSeconds}
                onChange={(e) => setWhisperStartupTimeoutSeconds(Number(e.target.value))} className={inputClass} />
            </label>
          </div>
        )}
      </div>

      {/* TensorRT-LLM */}
      <div className="space-y-2">
        <SectionHeader title="TensorRT-LLM" open={openSections.tensorrt} onToggle={() => toggleSection("tensorrt")} />
        {openSections.tensorrt && (
          <div className="grid gap-3 px-3 py-2 md:grid-cols-2">
            <label className="inline-flex items-center gap-2 text-sm text-muted-foreground col-span-full">
              <input type="checkbox" checked={tensorrtLlmEnabled} className={checkboxClass}
                onChange={(e) => setTensorRTLLMEnabled(e.target.checked)} />
              TensorRT-LLM enabled
            </label>
            <label className="block text-sm text-muted-foreground">
              Max batch size
              <input type="number" min={0} value={tensorrtLlmMaxBatchSize}
                onChange={(e) => setTensorRTLLMMaxBatchSize(Number(e.target.value))} className={inputClass} />
            </label>
            <label className="block text-sm text-muted-foreground">
              Context length
              <input type="number" min={0} value={tensorrtLlmContextLength}
                onChange={(e) => setTensorRTLLMContextLength(Number(e.target.value))} className={inputClass} />
            </label>
          </div>
        )}
      </div>

      <button
        type="button"
        onClick={() => void save()}
        className="rounded-md bg-primary px-3 py-2 text-sm font-medium text-primary-foreground"
      >
        Guardar defaults de backend
      </button>
    </section>
  )
}
