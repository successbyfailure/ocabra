import { useEffect, useMemo, useState } from "react"
import { toast } from "sonner"
import type { GPUState, ServerConfig } from "@/types"

interface GenerationServicesSettingsProps {
  config: ServerConfig
  gpus: GPUState[]
  onSave: (patch: Partial<ServerConfig>) => Promise<void>
}

type ServiceKey = "hunyuan" | "comfyui" | "a1111" | "acestep" | "unsloth"

interface ServiceMeta {
  key: ServiceKey
  label: string
  hint: string
  // Reasonable upper bound for the idle slider (seconds). Unsloth supports 0 = disabled.
  maxIdle: number
  allowDisable: boolean
}

const SERVICES: ServiceMeta[] = [
  {
    key: "hunyuan",
    label: "Hunyuan3D",
    hint: "Text/Image → 3D mesh. Generación corta-media (segundos).",
    maxIdle: 1800,
    allowDisable: false,
  },
  {
    key: "comfyui",
    label: "ComfyUI",
    hint: "Pipelines de imagen/vídeo basados en nodos.",
    maxIdle: 3600,
    allowDisable: false,
  },
  {
    key: "a1111",
    label: "Automatic1111",
    hint: "WebUI de Stable Diffusion.",
    maxIdle: 3600,
    allowDisable: false,
  },
  {
    key: "acestep",
    label: "ACE-Step",
    hint: "Generación de música. Cada pista tarda decenas de segundos.",
    maxIdle: 3600,
    allowDisable: false,
  },
  {
    key: "unsloth",
    label: "Unsloth Studio",
    hint: "Fine-tuning + chat + export. Training puede durar horas.",
    maxIdle: 7200,
    allowDisable: true,
  },
]

const KEY_MAP = {
  hunyuan: {
    idle: "hunyuanIdleUnloadSeconds",
    grace: "hunyuanGenerationGracePeriodS",
    gpu: "hunyuanPreferredGpu",
  },
  comfyui: {
    idle: "comfyuiIdleUnloadSeconds",
    grace: "comfyuiGenerationGracePeriodS",
    gpu: "comfyuiPreferredGpu",
  },
  a1111: {
    idle: "a1111IdleUnloadSeconds",
    grace: "a1111GenerationGracePeriodS",
    gpu: "a1111PreferredGpu",
  },
  acestep: {
    idle: "acestepIdleUnloadSeconds",
    grace: "acestepGenerationGracePeriodS",
    gpu: "acestepPreferredGpu",
  },
  unsloth: {
    idle: "unslothIdleUnloadSeconds",
    grace: "unslothGenerationGracePeriodS",
    gpu: "unslothPreferredGpu",
  },
} as const satisfies Record<
  ServiceKey,
  { idle: keyof ServerConfig; grace: keyof ServerConfig; gpu: keyof ServerConfig }
>

const inputClass =
  "mt-1 w-full rounded-md border border-border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"

function formatIdle(seconds: number, allowDisable: boolean): string {
  if (allowDisable && seconds === 0) return "deshabilitado"
  if (seconds < 60) return `${seconds} s`
  if (seconds < 3600) return `${Math.round(seconds / 60)} min`
  return `${(seconds / 3600).toFixed(1)} h`
}

function formatGrace(seconds: number): string {
  if (seconds === -1) return "indefinido"
  if (seconds === 0) return "inmediato"
  if (seconds < 60) return `${seconds} s`
  return `${Math.round(seconds / 60)} min`
}

export function GenerationServicesSettings({
  config,
  gpus,
  onSave,
}: GenerationServicesSettingsProps) {
  type ServiceState = { idle: number; grace: number; gpu: number }
  type LocalState = Record<ServiceKey, ServiceState> & { threshold: number }

  const initial = useMemo<LocalState>(
    () =>
      ({
        hunyuan: {
          idle: config.hunyuanIdleUnloadSeconds,
          grace: config.hunyuanGenerationGracePeriodS,
          gpu: config.hunyuanPreferredGpu,
        },
        comfyui: {
          idle: config.comfyuiIdleUnloadSeconds,
          grace: config.comfyuiGenerationGracePeriodS,
          gpu: config.comfyuiPreferredGpu,
        },
        a1111: {
          idle: config.a1111IdleUnloadSeconds,
          grace: config.a1111GenerationGracePeriodS,
          gpu: config.a1111PreferredGpu,
        },
        acestep: {
          idle: config.acestepIdleUnloadSeconds,
          grace: config.acestepGenerationGracePeriodS,
          gpu: config.acestepPreferredGpu,
        },
        unsloth: {
          idle: config.unslothIdleUnloadSeconds,
          grace: config.unslothGenerationGracePeriodS,
          gpu: config.unslothPreferredGpu,
        },
        threshold: config.generationGpuUtilThresholdPct,
      }) as LocalState,
    [config],
  )
  const [state, setState] = useState<LocalState>(initial)

  useEffect(() => {
    setState(initial)
  }, [initial])

  const updateService = (key: ServiceKey, patch: Partial<ServiceState>) =>
    setState((prev) => ({ ...prev, [key]: { ...prev[key], ...patch } }))

  const save = async () => {
    const patch: Partial<ServerConfig> = {
      generationGpuUtilThresholdPct: state.threshold,
    }
    for (const meta of SERVICES) {
      const map = KEY_MAP[meta.key]
      const s = state[meta.key]
      ;(patch as Record<string, unknown>)[map.idle] = s.idle
      ;(patch as Record<string, unknown>)[map.grace] = s.grace
      ;(patch as Record<string, unknown>)[map.gpu] = s.gpu
    }
    try {
      await onSave(patch)
      toast.success("Configuración de servicios de generación guardada")
    } catch {
      // upstream toast
    }
  }

  return (
    <section className="space-y-4 rounded-lg border border-border bg-card p-4">
      <header className="space-y-1">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">
          Servicios de generación
        </h2>
        <p className="text-xs text-muted-foreground/70">
          Timeouts de inactividad, grace de evicción y GPU preferida para cada servicio.
          Los cambios se aplican al ServiceManager en caliente; no es necesario reiniciar
          la api.
        </p>
      </header>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {SERVICES.map((meta) => {
          const s = state[meta.key]
          return (
            <div
              key={meta.key}
              className="space-y-3 rounded-md border border-border/60 bg-muted/10 p-3"
            >
              <div>
                <h3 className="text-sm font-medium">{meta.label}</h3>
                <p className="text-xs text-muted-foreground/70">{meta.hint}</p>
              </div>

              <label className="block text-sm text-muted-foreground">
                <span className="flex items-center justify-between">
                  Idle unload
                  <span className="text-xs text-muted-foreground/70">
                    {formatIdle(s.idle, meta.allowDisable)}
                  </span>
                </span>
                <input
                  type="number"
                  min={meta.allowDisable ? 0 : 30}
                  max={meta.maxIdle}
                  step={30}
                  value={s.idle}
                  onChange={(e) =>
                    updateService(meta.key, { idle: Number(e.target.value) })
                  }
                  className={inputClass}
                />
                <p className="text-xs text-muted-foreground/70 mt-1">
                  Segundos de inactividad antes de liberar VRAM.
                  {meta.allowDisable ? " 0 = nunca." : ""}
                </p>
              </label>

              <label className="block text-sm text-muted-foreground">
                <span className="flex items-center justify-between">
                  Grace period
                  <span className="text-xs text-muted-foreground/70">
                    {formatGrace(s.grace)}
                  </span>
                </span>
                <input
                  type="number"
                  min={-1}
                  max={1800}
                  step={30}
                  value={s.grace}
                  onChange={(e) =>
                    updateService(meta.key, { grace: Number(e.target.value) })
                  }
                  className={inputClass}
                />
                <p className="text-xs text-muted-foreground/70 mt-1">
                  Espera a que termine la generación activa. -1 = indefinido, 0 = inmediato.
                </p>
              </label>

              <label className="block text-sm text-muted-foreground">
                GPU preferida
                <select
                  value={s.gpu}
                  onChange={(e) =>
                    updateService(meta.key, { gpu: Number(e.target.value) })
                  }
                  className={inputClass}
                >
                  {gpus.length === 0 ? (
                    <>
                      <option value={0}>GPU 0</option>
                      <option value={1}>GPU 1</option>
                    </>
                  ) : (
                    gpus.map((g) => (
                      <option key={g.index} value={g.index}>
                        GPU {g.index} — {g.name}
                      </option>
                    ))
                  )}
                </select>
              </label>
            </div>
          )
        })}
      </div>

      <div className="space-y-2 rounded-md border border-border/60 bg-muted/10 p-3">
        <h3 className="text-sm font-medium">Detección de uso (heurístico GPU)</h3>
        <p className="text-xs text-muted-foreground/70">
          Para servicios que no exponen un endpoint de estado de generación
          (Hunyuan, ACE-Step, Unsloth Studio), el sistema marca el servicio como
          ocupado cuando la utilización de su GPU preferida supera este umbral.
          ComfyUI y A1111 usan sus propios endpoints y no dependen de este valor.
        </p>
        <label className="block text-sm text-muted-foreground">
          <span className="flex items-center justify-between">
            Umbral utilización GPU
            <span className="text-xs text-muted-foreground/70">{state.threshold}%</span>
          </span>
          <input
            type="range"
            min={1}
            max={99}
            step={1}
            value={state.threshold}
            onChange={(e) =>
              setState((prev) => ({ ...prev, threshold: Number(e.target.value) }))
            }
            className="mt-2 w-full accent-primary"
          />
        </label>
      </div>

      <div className="flex justify-end">
        <button
          type="button"
          onClick={save}
          className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90"
        >
          Guardar
        </button>
      </div>
    </section>
  )
}
