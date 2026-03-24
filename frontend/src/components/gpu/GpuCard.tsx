import type { GPUState } from "@/types"
import { PowerGauge } from "./PowerGauge"
import { VramBar } from "./VramBar"

interface GpuCardProps {
  gpu: GPUState
}

export function GpuCard({ gpu }: GpuCardProps) {
  const highTemp = gpu.temperatureC > 80
  const highUtilization = gpu.utilizationPct > 80

  return (
    <article
      className={`rounded-lg border bg-card p-4 shadow-sm ${
        highTemp || highUtilization ? "border-red-500/70" : "border-border"
      }`}
    >
      <div className="mb-4 flex items-start justify-between gap-4">
        <div>
          <p className="text-sm text-muted-foreground">GPU #{gpu.index}</p>
          <h3 className="text-base font-semibold text-foreground">{gpu.name}</h3>
        </div>
        <PowerGauge powerDrawW={gpu.powerDrawW} powerLimitW={gpu.powerLimitW} />
      </div>

      <VramBar used={gpu.usedVramMb} total={gpu.totalVramMb} locked={gpu.lockedVramMb} />

      <div className="mt-4 grid grid-cols-2 gap-3 text-sm">
        <div className="rounded-md bg-muted/50 px-3 py-2">
          <span className="text-muted-foreground">Utilization</span>
          <p className={highUtilization ? "animate-pulse font-semibold text-red-400" : "font-semibold"}>
            {gpu.utilizationPct.toFixed(1)}%
          </p>
        </div>

        <div className="rounded-md bg-muted/50 px-3 py-2">
          <span className="text-muted-foreground">Temperature</span>
          <p className={highTemp ? "font-semibold text-orange-400" : "font-semibold"}>
            {gpu.temperatureC.toFixed(1)}°C
          </p>
        </div>
      </div>

      <div className="mt-4 rounded-md border border-border/60 bg-background/40 p-3">
        <div className="mb-2 flex items-center justify-between">
          <span className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
            Processes (nvidia-smi)
          </span>
          <span className="text-xs text-muted-foreground">{gpu.processes.length}</span>
        </div>

        {gpu.processes.length === 0 ? (
          <p className="text-xs text-muted-foreground">No active GPU processes</p>
        ) : (
          <div className="space-y-1">
            {gpu.processes.map((process) => (
              <div
                key={`${process.processType}-${process.pid}`}
                className="flex items-center justify-between gap-2 rounded bg-muted/40 px-2 py-1 text-xs"
              >
                <span className="min-w-0 truncate font-mono text-foreground/90">
                  {process.pid} · {process.processName ?? "unknown"}
                </span>
                <span className="shrink-0 text-muted-foreground">
                  {process.processType} · {process.usedVramMb.toLocaleString()} MB
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </article>
  )
}
