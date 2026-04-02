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
  const [idleEvictionCheckIntervalSeconds, setIdleEvictionCheckIntervalSeconds] = useState(config.idleEvictionCheckIntervalSeconds)
  const [modelLoadWaitTimeoutSeconds, setModelLoadWaitTimeoutSeconds] = useState(config.modelLoadWaitTimeoutSeconds)
  const [pressureEvictionDrainTimeoutSeconds, setPressureEvictionDrainTimeoutSeconds] = useState(config.pressureEvictionDrainTimeoutSeconds)
  const [vramBufferMb, setVramBufferMb] = useState(config.vramBufferMb)
  const [vramPressureThresholdPct, setVramPressureThresholdPct] = useState(config.vramPressureThresholdPct)
  const [maxTemperatureC, setMaxTemperatureC] = useState(config.maxTemperatureC)

  useEffect(() => {
    setDefaultGpuIndex(config.defaultGpuIndex)
    setIdleEvictionCheckIntervalSeconds(config.idleEvictionCheckIntervalSeconds)
    setModelLoadWaitTimeoutSeconds(config.modelLoadWaitTimeoutSeconds)
    setPressureEvictionDrainTimeoutSeconds(config.pressureEvictionDrainTimeoutSeconds)
    setVramBufferMb(config.vramBufferMb)
    setVramPressureThresholdPct(config.vramPressureThresholdPct)
    setMaxTemperatureC(config.maxTemperatureC)
  }, [
    config.defaultGpuIndex,
    config.idleEvictionCheckIntervalSeconds,
    config.maxTemperatureC,
    config.modelLoadWaitTimeoutSeconds,
    config.pressureEvictionDrainTimeoutSeconds,
    config.vramBufferMb,
    config.vramPressureThresholdPct,
  ])

  const save = async () => {
    try {
      await onSave({
        defaultGpuIndex,
        idleEvictionCheckIntervalSeconds,
        modelLoadWaitTimeoutSeconds,
        pressureEvictionDrainTimeoutSeconds,
        vramBufferMb,
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
      <h2 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">Scheduler y GPU</h2>

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

      <div className="grid gap-2 md:grid-cols-2">
        <label className="text-sm text-muted-foreground">
          Idle eviction check (s)
          <input
            type="number"
            min={1}
            value={idleEvictionCheckIntervalSeconds}
            onChange={(event) => setIdleEvictionCheckIntervalSeconds(Number(event.target.value))}
            className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2"
          />
        </label>
        <label className="text-sm text-muted-foreground">
          VRAM buffer (MB)
          <input
            type="number"
            min={0}
            value={vramBufferMb}
            onChange={(event) => setVramBufferMb(Number(event.target.value))}
            className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2"
          />
        </label>
        <label className="text-sm text-muted-foreground">
          Model load wait timeout (s)
          <input
            type="number"
            min={1}
            value={modelLoadWaitTimeoutSeconds}
            onChange={(event) => setModelLoadWaitTimeoutSeconds(Number(event.target.value))}
            className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2"
          />
        </label>
        <label className="text-sm text-muted-foreground">
          Pressure drain timeout (s)
          <input
            type="number"
            min={1}
            value={pressureEvictionDrainTimeoutSeconds}
            onChange={(event) => setPressureEvictionDrainTimeoutSeconds(Number(event.target.value))}
            className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2"
          />
        </label>
      </div>

      <div className="grid gap-2 md:grid-cols-2">
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
      </div>

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
