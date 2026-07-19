import type { GPUState, ModelState, TokenGpuStats } from "@/types"
import { useGpuStore, type GpuHistoryPoint } from "@/stores/gpuStore"
import { ConcentricGauge } from "./ConcentricGauge"
import { MemoryBars } from "./MemoryBars"
import { MiniTrends } from "./MiniTrends"
import { LoadedModelList } from "./LoadedModelList"
import { METRIC, pct, tempColor } from "./metrics"

interface GpuCardProps {
  gpu: GPUState
  models?: ModelState[]
  activity?: Record<string, { inFlight: number; oldestSeconds: number }>
  stuckThreshold?: number
  tokenStats?: TokenGpuStats | null
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
const EMPTY_HISTORY: GpuHistoryPoint[] = []

function fmtTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(n >= 100_000 ? 0 : 1)}k`
  return String(n)
}

export function GpuCard({
  gpu,
  models = [],
  activity = {},
  stuckThreshold = 300,
  tokenStats = null,
}: GpuCardProps) {
  const raw = useGpuStore((s) => s.history[gpu.index] ?? EMPTY_HISTORY)
  const hist = downsample(raw, 40)

  const vramPct = pct(gpu.usedVramMb, gpu.totalVramMb)
  const lockedPct = pct(gpu.lockedVramMb, gpu.totalVramMb)
  const powPct = gpu.powerLimitW > 0 ? (gpu.powerDrawW / gpu.powerLimitW) * 100 : 0
  const hot = gpu.temperatureC >= 78 || vramPct >= 90

  const gpuModels = models.filter(
    (m) => m.status === "loaded" && (m.currentGpu ?? []).includes(gpu.index),
  )
  const inputTokens = tokenStats?.inputTokens ?? 0
  const outputTokens = tokenStats?.outputTokens ?? 0
  const totalTokens = inputTokens + outputTokens

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

      <div className="flex items-center gap-[18px]">
        <ConcentricGauge
          outer={{ value: gpu.utilizationPct, label: "Uso", color: METRIC.util }}
          inner={{ value: powPct, label: "Potencia", color: METRIC.power }}
          centerValue={gpu.utilizationPct.toFixed(0)}
          centerUnit="%"
          centerSub={`${Math.round(gpu.powerDrawW)} W`}
        />
        <div className="min-w-0 flex-1">
          <MemoryBars
            name="VRAM"
            usedLabel={gb(gpu.usedVramMb)}
            totalLabel={gb(gpu.totalVramMb)}
            usedPct={vramPct}
            secondaryName="bloqueada"
            secondaryPct={lockedPct}
          />
          <div className="mt-3">
            <MiniTrends
              metrics={[
                { label: "Uso GPU", data: hist.map((p) => p.util), color: METRIC.util, value: `${gpu.utilizationPct.toFixed(0)}%` },
                { label: "Potencia", data: hist.map((p) => p.powerPct), color: METRIC.power, value: `${Math.round(gpu.powerDrawW)}W` },
                { label: "VRAM", data: hist.map((p) => p.vramPct), color: METRIC.mem, value: `${Math.round(vramPct)}%` },
              ]}
            />
          </div>
        </div>
      </div>

      <div className="mt-[15px] grid grid-cols-3 gap-2 border-t border-border pt-3">
        <div className="min-w-0 rounded-lg bg-muted/45 px-2.5 py-2">
          <span className="block truncate text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
            Tokens
          </span>
          <span className="block truncate text-[13px] font-semibold tabular-nums">
            {fmtTokens(totalTokens)}
          </span>
        </div>
        <div className="min-w-0 rounded-lg bg-muted/45 px-2.5 py-2">
          <span className="block truncate text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
            Entrada
          </span>
          <span className="block truncate text-[13px] font-semibold tabular-nums">
            {fmtTokens(inputTokens)}
          </span>
        </div>
        <div className="min-w-0 rounded-lg bg-muted/45 px-2.5 py-2">
          <span className="block truncate text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
            Salida
          </span>
          <span className="block truncate text-[13px] font-semibold tabular-nums">
            {fmtTokens(outputTokens)}
          </span>
        </div>
      </div>

      <div className="mt-[15px] border-t border-border pt-3">
        <p className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
          Modelos ({gpuModels.length})
        </p>
        <LoadedModelList
          models={gpuModels}
          activity={activity}
          stuckThreshold={stuckThreshold}
          emptyLabel="Sin modelos gestionados en esta GPU"
        />
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
