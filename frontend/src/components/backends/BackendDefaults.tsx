import { useEffect, useState } from "react"
import { ChevronDown } from "lucide-react"
import { toast } from "sonner"
import type { ServerConfig } from "@/types"

type Key = keyof ServerConfig

interface Field {
  key: Key
  label: string
  help?: string
  kind: "number" | "bool" | "select"
  min?: number
  max?: number
  step?: number
  options?: string[]
  /** 0 en el control = sin valor (null): «lo que decida el motor». */
  zeroIsNull?: boolean
}

/** Valores por defecto de cada backend (se aplican a los modelos que no fijan
 *  el suyo). Antes vivían todos juntos en Ajustes → Backends. */
const DEFAULTS: Record<string, { fields: Field[]; note?: string }> = {
  vllm: {
    fields: [
      { key: "vllmGpuMemoryUtilization", label: "Fracción de VRAM", kind: "number", min: 0.1, max: 0.99, step: 0.01, help: "Cuánta memoria de la GPU reserva vLLM al arrancar. Bájala si hay OOM." },
      { key: "vllmMaxNumSeqs", label: "Secuencias simultáneas", kind: "number", min: 0, zeroIsNull: true, help: "0 = lo que decida vLLM." },
      { key: "vllmMaxNumBatchedTokens", label: "Tokens por lote", kind: "number", min: 0, zeroIsNull: true, help: "0 = lo que decida vLLM." },
      { key: "vllmEnablePrefixCaching", label: "Cache de prefijos", kind: "bool" },
      { key: "vllmEnforceEager", label: "Modo eager (sin CUDA graphs)", kind: "bool" },
    ],
  },
  sglang: {
    fields: [
      { key: "sglangMemFractionStatic", label: "Fracción de VRAM estática", kind: "number", min: 0.1, max: 0.99, step: 0.01 },
      { key: "sglangContextLength", label: "Contexto", kind: "number", min: 0, zeroIsNull: true, help: "0 = el del modelo." },
      { key: "sglangDisableRadixCache", label: "Desactivar radix cache", kind: "bool" },
    ],
  },
  llama_cpp: {
    note: "Compartidos con llama.cpp (Makespace).",
    fields: [
      { key: "llamaCppGpuLayers", label: "Capas en GPU", kind: "number", min: 0, help: "999 = todas." },
      { key: "llamaCppCtxSize", label: "Contexto", kind: "number", min: 1 },
      { key: "llamaCppFlashAttn", label: "Flash attention", kind: "bool" },
    ],
  },
  bitnet: {
    fields: [
      { key: "bitnetGpuLayers", label: "Capas en GPU", kind: "number", min: 0 },
      { key: "bitnetCtxSize", label: "Contexto", kind: "number", min: 1 },
      { key: "bitnetFlashAttn", label: "Flash attention", kind: "bool" },
    ],
  },
  diffusers: {
    fields: [
      { key: "diffusersTorchDtype", label: "Precisión (dtype)", kind: "select", options: ["auto", "float16", "bfloat16", "float32"] },
      { key: "diffusersOffloadMode", label: "Descarga a CPU", kind: "select", options: ["none", "model", "sequential"], help: "model/sequential ahorran VRAM a costa de velocidad." },
      { key: "diffusersEnableTorchCompile", label: "torch.compile", kind: "bool" },
      { key: "diffusersEnableXformers", label: "xFormers", kind: "bool" },
      { key: "diffusersAllowTf32", label: "Permitir TF32", kind: "bool" },
    ],
  },
  whisper: {
    fields: [
      { key: "whisperStartupTimeoutSeconds", label: "Tiempo máximo de arranque (s)", kind: "number", min: 1 },
      { key: "openaiAudioMaxPartSizeMb", label: "Tamaño máximo por trozo de audio (MB)", kind: "number", min: 1, help: "Aplica a toda transcripción por /v1/audio." },
    ],
  },
  tensorrt_llm: {
    fields: [
      { key: "tensorrtLlmEnabled", label: "Activado", kind: "bool" },
      { key: "tensorrtLlmMaxBatchSize", label: "Tamaño máximo de lote", kind: "number", min: 0, zeroIsNull: true },
      { key: "tensorrtLlmContextLength", label: "Contexto", kind: "number", min: 0, zeroIsNull: true },
    ],
  },
}
DEFAULTS.llamacpp_mks = { ...DEFAULTS.llama_cpp, note: "Compartidos con llama.cpp." }

export function hasBackendDefaults(backendType: string): boolean {
  return backendType in DEFAULTS
}

type Values = Partial<Record<Key, unknown>>

function readValues(fields: Field[], config: ServerConfig): Values {
  return Object.fromEntries(fields.map((f) => [f.key, f.zeroIsNull ? (config[f.key] ?? 0) : config[f.key]]))
}

/** Los valores por defecto de un backend, plegados dentro de su tarjeta. */
export function BackendDefaults({
  backendType,
  config,
  onSave,
}: {
  backendType: string
  config: ServerConfig
  onSave: (patch: Partial<ServerConfig>) => Promise<void>
}) {
  const spec = DEFAULTS[backendType]
  const [open, setOpen] = useState(false)
  const [values, setValues] = useState<Values>(() => (spec ? readValues(spec.fields, config) : {}))
  const [saving, setSaving] = useState(false)
  useEffect(() => {
    if (spec) setValues(readValues(spec.fields, config))
  }, [config, spec])
  if (!spec) return null
  const initial = readValues(spec.fields, config)
  const dirty = spec.fields.some((f) => values[f.key] !== initial[f.key])
  const save = async () => {
    setSaving(true)
    try {
      const patch = Object.fromEntries(
        spec.fields.map((f) => [f.key, f.zeroIsNull && !Number(values[f.key]) ? null : values[f.key]]),
      ) as Partial<ServerConfig>
      await onSave(patch)
      toast.success("Valores por defecto guardados")
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "No se pudo guardar")
    } finally {
      setSaving(false)
    }
  }
  const input = "mt-1 w-full rounded-md border border-border bg-background px-2.5 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
  return (
    <div data-testid={`backend-defaults-${backendType}`}>
      <button
        type="button"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
      >
        <ChevronDown size={12} className={open ? "" : "-rotate-90"} />
        Valores por defecto ({spec.fields.length}){dirty ? " · sin guardar" : ""}
      </button>
      {open && (
        <div className="mt-2 space-y-3">
          <p className="text-[11px] text-muted-foreground">
            Para los modelos de este backend que no fijan el suyo en su editor.{spec.note ? ` ${spec.note}` : ""}
          </p>
          <div className="grid grid-cols-[repeat(auto-fill,minmax(min(100%,180px),1fr))] gap-3">
            {spec.fields.map((f) =>
              f.kind === "bool" ? (
                <label key={f.key} className="inline-flex items-center gap-2 self-end text-sm text-muted-foreground">
                  <input
                    type="checkbox"
                    className="h-4 w-4 accent-primary"
                    checked={Boolean(values[f.key])}
                    onChange={(e) => setValues({ ...values, [f.key]: e.target.checked })}
                  />
                  {f.label}
                </label>
              ) : (
                <label key={f.key} className="block text-sm text-muted-foreground">
                  {f.label}
                  {f.kind === "select" ? (
                    <select className={input} value={String(values[f.key] ?? "")} onChange={(e) => setValues({ ...values, [f.key]: e.target.value })}>
                      {f.options!.map((o) => (
                        <option key={o} value={o}>
                          {o}
                        </option>
                      ))}
                    </select>
                  ) : (
                    <input
                      type="number"
                      className={input}
                      min={f.min}
                      max={f.max}
                      step={f.step}
                      value={Number(values[f.key] ?? 0)}
                      onChange={(e) => setValues({ ...values, [f.key]: Number(e.target.value) })}
                    />
                  )}
                  {f.help && <span className="mt-0.5 block text-[11px] text-muted-foreground/70">{f.help}</span>}
                </label>
              ),
            )}
          </div>
          <button
            type="button"
            disabled={!dirty || saving}
            onClick={() => void save()}
            className="rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground disabled:opacity-40"
          >
            {saving ? "Guardando…" : "Guardar"}
          </button>
        </div>
      )}
    </div>
  )
}
