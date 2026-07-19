import type { GPUState, ModelState } from "@/types"
import { useGpuStore, type GpuHistoryPoint } from "@/stores/gpuStore"
import { MetricGauge } from "./MetricGauge"
import { MemoryBars } from "./MemoryBars"
import { PowerBlock } from "./PowerBlock"
import { LoadedModelList } from "./LoadedModelList"
import { METRIC, pct, tempColor } from "./metrics"

interface GpuCardProps {
  gpu: GPUState
  models?: ModelState[]
  activity?: Record<string, number>
}

// Keep the most recent `max` points with a stable stride (tail sampling), so the
// area slides smoothly instead of realigning on every update.
function downsample(pts: GpuHistoryPoint[], max: number): GpuHistoryPoint[] {
  if (pts.length <= max) return pts
  const step = Math.ceil(pts.length / max)
  const out: GpuHistoryPoint[] = []
  for (let i = pts.length - 1; i >= 0; i -= step) out.push(pts[i])
  return out.reverse()
}

const gb = (mb: number) => (mb / 1024).toFixed(1)

export function GpuCard({ gpu, models = [], activity = {} }: GpuCardProps) {
  const raw = useGpuStore((s) => s.history[gpu.index] ?? [])
  const powerHistory = downsample(raw, 60).map((p) => p.powerPct)

  const gpuModels = models.filter(
    (m) => m.status === "loaded" && (m.currentGpu ?? []).includes(gpu.index),
  )

  const vramPct = pct(gpu.usedVramMb, gpu.totalVramMb)
  const lockedPct = pct(gpu.lockedVramMb, gpu.totalVramMb)
  const powPct = gpu.powerLimitW > 0 ? (gpu.powerDrawW / gpu.powerLimitW) * 100 : 0
  const hot = gpu.temperatureC >= 78 || vramPct >= 90

  return (
    <article className={`rounded-2xl border bg-card p-[18px] shadow-sm ${hot ? "border-red-500/60" : "border-border"}`}>
      <div className="mb-4 flex items-start justify-between gap-2.5">
        <div className="min-w-0">
          <p className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">GPU #{gpu.index}</p>
          <h3 className="truncate text-[15.5px] font-semibold tracking-tight text-foreground">{gpu.name}</h3>
        </div>
        <span className="inline-flex shrink-0 items-center gap-1.5 rounded-full border border-border bg-muted/50 px-2.5 py-1 text-[12.5px] font-semibold tabular-nums">
          <span className="h-[7px] w-[7px] rounded-full" style={{ background: tempColor(gpu.temperatureC) }} />
          {gpu.temperatureC.toFixed(0)}°C
        </span>
      </div>

      <div className="grid grid-cols-[auto_1fr] items-center gap-[18px]">
        <MetricGauge
          pct={gpu.utilizationPct}
          color={METRIC.util}
          label="Uso GPU"
          value={gpu.utilizationPct.toFixed(0)}
          unit="%"
          caption="Uso GPU"
        />
        <MemoryBars
          name="VRAM"
          usedLabel={gb(gpu.usedVramMb)}
          totalLabel={gb(gpu.totalVramMb)}
          usedPct={vramPct}
          secondaryName="bloqueada"
          secondaryPct={lockedPct}
        />
      </div>

      <PowerBlock
        label="Consumo eléctrico"
        powerW={gpu.powerDrawW}
        powerLimitW={gpu.powerLimitW}
        history={powerHistory}
        subtitle={`${Math.round(powPct)}% del límite · últimos 10 min`}
      />

      <div className="mt-[15px] border-t border-border pt-3">
        <p className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
          Modelos ({gpuModels.length})
        </p>
        <LoadedModelList models={gpuModels} activity={activity} emptyLabel="Sin modelos gestionados en esta GPU" />
      </div>

      <details className="group mt-[15px] border-t border-border">
        <summary className="flex cursor-pointer list-none items-center justify-between pt-3 text-[11.5px] font-medium text-muted-foreground [&::-webkit-details-marker]:hidden">
          <span>Procesos ({gpu.processes.length})</span>
          <span className="text-muted-foreground/70 transition-transform group-open:rotate-180">▾</span>
        </summary>
        <div className="flex flex-col gap-1.5 pb-0.5 pt-2.5">
          {gpu.processes.length === 0 ? (
            <p className="text-[11.5px] text-muted-foreground">Sin procesos activos en la GPU</p>
          ) : (
            gpu.processes.map((process) => (
              <div
                key={`${process.processType}-${process.pid}`}
                className="flex items-center justify-between gap-2 rounded-lg bg-muted/50 px-2.5 py-1.5 text-[11.5px]"
              >
                <span className="min-w-0 truncate font-mono text-foreground/90">
                  {process.pid} · {process.processName ?? "unknown"}
                </span>
                <span className="shrink-0 tabular-nums text-muted-foreground">
                  {process.processType} · {process.usedVramMb.toLocaleString()} MB
                </span>
              </div>
            ))
          )}
        </div>
      </details>
    </article>
  )
}
