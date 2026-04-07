import { useEffect, useState, useCallback } from "react"
import { Zap, RotateCcw } from "lucide-react"
import { toast } from "sonner"
import { useAuthStore } from "@/stores/authStore"
import type { GPUState, ServerConfig } from "@/types"

interface PowerLimits {
  gpu_index: number
  current_w: number
  default_w: number
  min_w: number
  max_w: number
  persistence_mode: boolean
}

function GPUPowerCard({ gpu, isAdmin }: { gpu: GPUState; isAdmin: boolean }) {
  const [limits, setLimits] = useState<PowerLimits | null>(null)
  const [targetW, setTargetW] = useState(0)
  const [saving, setSaving] = useState(false)

  const fetchLimits = useCallback(async () => {
    try {
      const resp = await fetch(`/ocabra/gpus/${gpu.index}/power-limits`, { credentials: "include" })
      if (resp.ok) {
        const data: PowerLimits = await resp.json()
        setLimits(data)
        setTargetW(data.current_w)
      }
    } catch { /* ignore */ }
  }, [gpu.index])

  useEffect(() => { fetchLimits() }, [fetchLimits])

  const applyPowerLimit = async () => {
    setSaving(true)
    try {
      const resp = await fetch(`/ocabra/gpus/${gpu.index}/power`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ power_limit_w: targetW }),
      })
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}))
        throw new Error(err.detail ?? `HTTP ${resp.status}`)
      }
      const data: PowerLimits = await resp.json()
      setLimits(data)
      setTargetW(data.current_w)
      toast.success(`GPU ${gpu.index}: power limit ${data.current_w}W`)
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Error al cambiar power limit")
    } finally {
      setSaving(false)
    }
  }

  const resetDefault = async () => {
    setSaving(true)
    try {
      const resp = await fetch(`/ocabra/gpus/${gpu.index}/power`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ power_limit_w: 0 }),
      })
      if (!resp.ok) throw new Error("Reset failed")
      const data: PowerLimits = await resp.json()
      setLimits(data)
      setTargetW(data.current_w)
      toast.success(`GPU ${gpu.index}: power limit reset to ${data.current_w}W`)
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Error al resetear")
    } finally {
      setSaving(false)
    }
  }

  if (!limits) return null

  const pctOfDefault = limits.default_w > 0 ? Math.round((limits.current_w / limits.default_w) * 100) : 0
  const isReduced = limits.current_w < limits.default_w

  return (
    <div className="rounded-md border border-border/60 bg-muted/10 p-3 space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Zap size={14} className={isReduced ? "text-amber-400" : "text-emerald-400"} />
          <span className="text-sm font-medium">GPU {gpu.index}</span>
          <span className="text-xs text-muted-foreground">{gpu.name}</span>
        </div>
        <div className="flex items-center gap-2 text-xs">
          <span className="font-mono">
            {limits.current_w}W
            {isReduced && <span className="text-amber-400 ml-1">({pctOfDefault}%)</span>}
          </span>
          <span className="text-muted-foreground">/ {limits.max_w}W max</span>
        </div>
      </div>

      {isAdmin && (
        <div className="space-y-1.5">
          <div className="flex items-center gap-2">
            <input
              type="range"
              min={limits.min_w}
              max={limits.max_w}
              step={5}
              value={targetW}
              onChange={(e) => setTargetW(Number(e.target.value))}
              className="flex-1"
            />
            <span className="w-14 text-right font-mono text-xs">{targetW}W</span>
          </div>
          <div className="flex items-center justify-between text-[10px] text-muted-foreground">
            <span>{limits.min_w}W</span>
            <span>Default: {limits.default_w}W</span>
            <span>{limits.max_w}W</span>
          </div>
          <div className="flex gap-2 pt-1">
            <button
              type="button"
              onClick={applyPowerLimit}
              disabled={saving || targetW === limits.current_w}
              className="rounded-md bg-primary px-2.5 py-1 text-xs font-medium text-primary-foreground disabled:opacity-50"
            >
              {saving ? "..." : "Aplicar"}
            </button>
            {isReduced && (
              <button
                type="button"
                onClick={resetDefault}
                disabled={saving}
                className="flex items-center gap-1 rounded-md border border-border px-2.5 py-1 text-xs text-muted-foreground hover:bg-muted disabled:opacity-50"
              >
                <RotateCcw size={10} /> Reset default
              </button>
            )}
          </div>
        </div>
      )}

      {!isAdmin && isReduced && (
        <p className="text-[10px] text-amber-400">
          Power limit reducido a {pctOfDefault}% del default ({limits.default_w}W)
        </p>
      )}
    </div>
  )
}

interface GPUSettingsProps {
  gpus: GPUState[]
  config: ServerConfig
  onSave: (patch: Partial<ServerConfig>) => Promise<void>
}

export function GPUSettings({ gpus, config, onSave }: GPUSettingsProps) {
  const isAdmin = useAuthStore((s) => s.user?.role === "system_admin")
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

      {/* Power limit controls per GPU */}
      <h2 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground pt-4">
        Limites de potencia
      </h2>
      <p className="text-xs text-muted-foreground -mt-1">
        Reduce el TDP para ahorrar energia y reducir ruido/temperatura.
      </p>
      <div className="space-y-2">
        {gpus.map((gpu) => (
          <GPUPowerCard key={gpu.index} gpu={gpu} isAdmin={isAdmin} />
        ))}
      </div>
    </section>
  )
}
