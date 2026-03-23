import { useEffect, useState } from "react"
import { toast } from "sonner"
import type { GPUState, ServerConfig } from "@/types"

interface GPUSettingsProps {
  gpus: GPUState[]
  config: ServerConfig
  onSave: (patch: Partial<ServerConfig>) => Promise<void>
}

export function GPUSettings({ gpus, config, onSave }: GPUSettingsProps) {
  const [defaultGpuIndex, setDefaultGpuIndex] = useState(config.defaultGpuIndex)
  const [vramPressureThresholdPct, setVramPressureThresholdPct] = useState(config.vramPressureThresholdPct)
  const [maxTemperatureC, setMaxTemperatureC] = useState(config.maxTemperatureC)

  useEffect(() => {
    setDefaultGpuIndex(config.defaultGpuIndex)
    setVramPressureThresholdPct(config.vramPressureThresholdPct)
    setMaxTemperatureC(config.maxTemperatureC)
  }, [config.defaultGpuIndex, config.maxTemperatureC, config.vramPressureThresholdPct])

  const save = async () => {
    try {
      await onSave({
        defaultGpuIndex,
        vramPressureThresholdPct,
        maxTemperatureC,
      })
      toast.success("GPU settings guardadas")
    } catch {
      // page-level toast is shown in Settings
    }
  }

  return (
    <section className="space-y-3 rounded-lg border border-border bg-card p-4">
      <h2 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">GPU</h2>

      <label className="block text-sm text-muted-foreground">
        GPU preferida por defecto
        <select
          value={defaultGpuIndex}
          onChange={(event) => setDefaultGpuIndex(Number(event.target.value))}
          className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2"
        >
          {gpus.map((gpu) => (
            <option key={gpu.index} value={gpu.index}>
              GPU {gpu.index} - {gpu.name}
            </option>
          ))}
        </select>
      </label>

      <label className="block text-sm text-muted-foreground">
        Umbral presion VRAM (%): {vramPressureThresholdPct}
        <input
          type="range"
          min={50}
          max={98}
          value={vramPressureThresholdPct}
          onChange={(event) => setVramPressureThresholdPct(Number(event.target.value))}
          className="mt-2 w-full"
        />
      </label>

      <label className="block text-sm text-muted-foreground">
        Temperatura maxima alerta (C)
        <input
          type="number"
          min={50}
          max={100}
          value={maxTemperatureC}
          onChange={(event) => setMaxTemperatureC(Number(event.target.value))}
          className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2"
        />
      </label>

      <button
        type="button"
        onClick={() => void save()}
        className="rounded-md bg-primary px-3 py-2 text-sm font-medium text-primary-foreground"
      >
        Guardar GPU
      </button>
    </section>
  )
}
