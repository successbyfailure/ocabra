import { useEffect, useState } from "react"
import { toast } from "sonner"
import type { ServerConfig } from "@/types"

interface BackendRuntimeSettingsProps {
  config: ServerConfig
  onSave: (patch: Partial<ServerConfig>) => Promise<void>
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

  return (
    <section className="space-y-4 rounded-lg border border-border bg-card p-4">
      <h2 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">Backend defaults</h2>

      <div className="grid gap-4 md:grid-cols-2">
        <div className="space-y-3 rounded-md border border-border/70 p-3">
          <h3 className="text-sm font-medium">vLLM</h3>
          <label className="block text-sm text-muted-foreground">
            GPU memory utilization
            <input
              type="number"
              min={0.1}
              max={0.99}
              step="0.01"
              value={vllmGpuMemoryUtilization}
              onChange={(event) => setVLLMGpuMemoryUtilization(Number(event.target.value))}
              className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2"
            />
          </label>
          <label className="block text-sm text-muted-foreground">
            Max concurrent sequences
            <input
              type="number"
              min={0}
              value={vllmMaxNumSeqs}
              onChange={(event) => setVLLMMaxNumSeqs(Number(event.target.value))}
              className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2"
            />
          </label>
          <label className="block text-sm text-muted-foreground">
            Max batched tokens
            <input
              type="number"
              min={0}
              value={vllmMaxNumBatchedTokens}
              onChange={(event) => setVLLMMaxNumBatchedTokens(Number(event.target.value))}
              className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2"
            />
          </label>
          <label className="inline-flex items-center gap-2 text-sm text-muted-foreground">
            <input
              type="checkbox"
              checked={vllmEnablePrefixCaching}
              onChange={(event) => setVLLMEnablePrefixCaching(event.target.checked)}
            />
            Prefix caching
          </label>
          <label className="inline-flex items-center gap-2 text-sm text-muted-foreground">
            <input
              type="checkbox"
              checked={vllmEnforceEager}
              onChange={(event) => setVLLMEnforceEager(event.target.checked)}
            />
            Enforce eager
          </label>
        </div>

        <div className="space-y-3 rounded-md border border-border/70 p-3">
          <h3 className="text-sm font-medium">SGLang</h3>
          <label className="block text-sm text-muted-foreground">
            Mem fraction static
            <input
              type="number"
              min={0.1}
              max={0.99}
              step="0.01"
              value={sglangMemFractionStatic}
              onChange={(event) => setSGLangMemFractionStatic(Number(event.target.value))}
              className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2"
            />
          </label>
          <label className="block text-sm text-muted-foreground">
            Context length
            <input
              type="number"
              min={0}
              value={sglangContextLength}
              onChange={(event) => setSGLangContextLength(Number(event.target.value))}
              className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2"
            />
          </label>
          <label className="inline-flex items-center gap-2 text-sm text-muted-foreground">
            <input
              type="checkbox"
              checked={sglangDisableRadixCache}
              onChange={(event) => setSGLangDisableRadixCache(event.target.checked)}
            />
            Disable radix cache
          </label>
        </div>

        <div className="space-y-3 rounded-md border border-border/70 p-3">
          <h3 className="text-sm font-medium">llama.cpp / BitNet</h3>
          <label className="block text-sm text-muted-foreground">
            llama.cpp GPU layers
            <input
              type="number"
              min={0}
              value={llamaCppGpuLayers}
              onChange={(event) => setLlamaCppGpuLayers(Number(event.target.value))}
              className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2"
            />
          </label>
          <label className="block text-sm text-muted-foreground">
            llama.cpp ctx size
            <input
              type="number"
              min={1}
              value={llamaCppCtxSize}
              onChange={(event) => setLlamaCppCtxSize(Number(event.target.value))}
              className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2"
            />
          </label>
          <label className="inline-flex items-center gap-2 text-sm text-muted-foreground">
            <input
              type="checkbox"
              checked={llamaCppFlashAttn}
              onChange={(event) => setLlamaCppFlashAttn(event.target.checked)}
            />
            llama.cpp flash attention
          </label>
          <label className="block text-sm text-muted-foreground">
            BitNet GPU layers
            <input
              type="number"
              min={0}
              value={bitnetGpuLayers}
              onChange={(event) => setBitnetGpuLayers(Number(event.target.value))}
              className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2"
            />
          </label>
          <label className="block text-sm text-muted-foreground">
            BitNet ctx size
            <input
              type="number"
              min={1}
              value={bitnetCtxSize}
              onChange={(event) => setBitnetCtxSize(Number(event.target.value))}
              className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2"
            />
          </label>
          <label className="inline-flex items-center gap-2 text-sm text-muted-foreground">
            <input
              type="checkbox"
              checked={bitnetFlashAttn}
              onChange={(event) => setBitnetFlashAttn(event.target.checked)}
            />
            BitNet flash attention
          </label>
        </div>

        <div className="space-y-3 rounded-md border border-border/70 p-3">
          <h3 className="text-sm font-medium">Diffusers / Audio / TensorRT-LLM</h3>
          <label className="block text-sm text-muted-foreground">
            Diffusers torch dtype
            <select
              value={diffusersTorchDtype}
              onChange={(event) => setDiffusersTorchDtype(event.target.value)}
              className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2"
            >
              <option value="auto">auto</option>
              <option value="float16">float16</option>
              <option value="bfloat16">bfloat16</option>
              <option value="float32">float32</option>
            </select>
          </label>
          <label className="block text-sm text-muted-foreground">
            Diffusers offload mode
            <select
              value={diffusersOffloadMode}
              onChange={(event) => setDiffusersOffloadMode(event.target.value)}
              className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2"
            >
              <option value="none">none</option>
              <option value="model">model</option>
              <option value="sequential">sequential</option>
            </select>
          </label>
          <label className="inline-flex items-center gap-2 text-sm text-muted-foreground">
            <input
              type="checkbox"
              checked={diffusersEnableTorchCompile}
              onChange={(event) => setDiffusersEnableTorchCompile(event.target.checked)}
            />
            Diffusers torch.compile
          </label>
          <label className="inline-flex items-center gap-2 text-sm text-muted-foreground">
            <input
              type="checkbox"
              checked={diffusersEnableXformers}
              onChange={(event) => setDiffusersEnableXformers(event.target.checked)}
            />
            Diffusers xFormers
          </label>
          <label className="inline-flex items-center gap-2 text-sm text-muted-foreground">
            <input
              type="checkbox"
              checked={diffusersAllowTf32}
              onChange={(event) => setDiffusersAllowTf32(event.target.checked)}
            />
            Diffusers allow TF32
          </label>
          <label className="block text-sm text-muted-foreground">
            OpenAI audio max part size (MB)
            <input
              type="number"
              min={1}
              value={openaiAudioMaxPartSizeMb}
              onChange={(event) => setOpenAIAudioMaxPartSizeMb(Number(event.target.value))}
              className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2"
            />
          </label>
          <label className="block text-sm text-muted-foreground">
            Whisper startup timeout (s)
            <input
              type="number"
              min={1}
              value={whisperStartupTimeoutSeconds}
              onChange={(event) => setWhisperStartupTimeoutSeconds(Number(event.target.value))}
              className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2"
            />
          </label>
          <label className="inline-flex items-center gap-2 text-sm text-muted-foreground">
            <input
              type="checkbox"
              checked={tensorrtLlmEnabled}
              onChange={(event) => setTensorRTLLMEnabled(event.target.checked)}
            />
            TensorRT-LLM enabled
          </label>
          <label className="block text-sm text-muted-foreground">
            TensorRT-LLM max batch size
            <input
              type="number"
              min={0}
              value={tensorrtLlmMaxBatchSize}
              onChange={(event) => setTensorRTLLMMaxBatchSize(Number(event.target.value))}
              className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2"
            />
          </label>
          <label className="block text-sm text-muted-foreground">
            TensorRT-LLM context length
            <input
              type="number"
              min={0}
              value={tensorrtLlmContextLength}
              onChange={(event) => setTensorRTLLMContextLength(Number(event.target.value))}
              className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2"
            />
          </label>
        </div>
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
