import type { GPUState } from "@/types"
import { useGpuStore, type GpuHistoryPoint } from "@/stores/gpuStore"
import { PowerGauge } from "./PowerGauge"
import { VramBar } from "./VramBar"
import { Line, LineChart, ResponsiveContainer, Tooltip, YAxis } from "recharts"

interface GpuCardProps {
  gpu: GPUState
}

// Validated categorical palette (dataviz skill): fixed order blue / aqua / yellow.
// CVD-safe (worst adjacent ΔE > 40) and contrast-checked against both surfaces.
const CHART_COLORS = {
  util:  { stroke: "#2a78d6" },
  vram:  { stroke: "#1baf7a" },
  power: { stroke: "#eda100" },
}

// Keep the most recent `max` points with a stable stride. Sampling from the
// tail (instead of across the whole growing buffer) means existing points keep
// their x-position between renders, so the live lines slide smoothly instead of
// jittering/realigning on every 2s update.
function downsample(pts: GpuHistoryPoint[], max: number): GpuHistoryPoint[] {
  if (pts.length <= max) return pts
  const step = Math.ceil(pts.length / max)
  const out: GpuHistoryPoint[] = []
  for (let i = pts.length - 1; i >= 0; i -= step) out.push(pts[i])
  return out.reverse()
}

const SERIES = [
  { key: "util", label: "Util", color: CHART_COLORS.util.stroke },
  { key: "vramPct", label: "VRAM", color: CHART_COLORS.vram.stroke },
  { key: "powerPct", label: "Power", color: CHART_COLORS.power.stroke },
] as const

function MiniChart({ gpuIndex }: { gpuIndex: number }) {
  const raw = useGpuStore((s) => s.history[gpuIndex] ?? [])
  const history = downsample(raw, 120)
  if (history.length < 2) return null

  return (
    <div className="mt-3 h-16 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={history} margin={{ top: 4, right: 2, bottom: 0, left: 0 }}>
          <Tooltip
            contentStyle={{
              backgroundColor: "hsl(var(--card))",
              border: "1px solid hsl(var(--border))",
              borderRadius: "6px",
              fontSize: "11px",
              padding: "4px 8px",
            }}
            labelFormatter={() => ""}
            formatter={(value: number, name: string) => {
              const labels: Record<string, string> = { util: "Util", vramPct: "VRAM", powerPct: "Power" }
              return [`${value.toFixed(1)}%`, labels[name] ?? name]
            }}
          />

          <YAxis domain={[0, 100]} hide />
          {SERIES.map((s) => (
            <Line
              key={s.key}
              type="monotone"
              dataKey={s.key}
              stroke={s.color}
              strokeWidth={s.key === "powerPct" ? 2 : 1.5}
              dot={false}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>

      <div className="mt-1 flex gap-3 text-[10px] text-muted-foreground">
        <span className="flex items-center gap-1">
          <span className="inline-block h-1.5 w-3 rounded-full" style={{ background: CHART_COLORS.util.stroke }} />
          Util
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block h-1.5 w-3 rounded-full" style={{ background: CHART_COLORS.vram.stroke }} />
          VRAM
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block h-1.5 w-3 rounded-full" style={{ background: CHART_COLORS.power.stroke }} />
          Power
        </span>
      </div>
    </div>
  )
}

// Status palette (dataviz skill): good / warning / critical by temperature.
function tempColorClass(tempC: number): string {
  if (tempC >= 90) return "text-red-500"
  if (tempC >= 80) return "text-amber-500"
  return "text-emerald-500"
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
          <p className={`font-semibold ${tempColorClass(gpu.temperatureC)}`}>
            {gpu.temperatureC.toFixed(1)}°C
          </p>
        </div>
      </div>

      <MiniChart gpuIndex={gpu.index} />

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
