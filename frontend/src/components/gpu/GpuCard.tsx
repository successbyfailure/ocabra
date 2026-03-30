import type { GPUState } from "@/types"
import { useGpuStore } from "@/stores/gpuStore"
import { PowerGauge } from "./PowerGauge"
import { VramBar } from "./VramBar"
import { Area, AreaChart, ResponsiveContainer, Tooltip, YAxis } from "recharts"

interface GpuCardProps {
  gpu: GPUState
}

const CHART_COLORS = {
  util:  { stroke: "#6366f1", fill: "#6366f130" },
  vram:  { stroke: "#22d3ee", fill: "#22d3ee25" },
  power: { stroke: "#f59e0b", fill: "#f59e0b20" },
}

function MiniChart({ gpuIndex }: { gpuIndex: number }) {
  const history = useGpuStore((s) => s.history[gpuIndex] ?? [])
  if (history.length < 2) return null

  return (
    <div className="mt-3 h-16 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={history} margin={{ top: 2, right: 0, bottom: 0, left: 0 }}>
          <defs>
            <linearGradient id={`util-${gpuIndex}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%"  stopColor={CHART_COLORS.util.stroke} stopOpacity={0.3} />
              <stop offset="95%" stopColor={CHART_COLORS.util.stroke} stopOpacity={0} />
            </linearGradient>
            <linearGradient id={`vram-${gpuIndex}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%"  stopColor={CHART_COLORS.vram.stroke} stopOpacity={0.3} />
              <stop offset="95%" stopColor={CHART_COLORS.vram.stroke} stopOpacity={0} />
            </linearGradient>
            <linearGradient id={`power-${gpuIndex}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%"  stopColor={CHART_COLORS.power.stroke} stopOpacity={0.25} />
              <stop offset="95%" stopColor={CHART_COLORS.power.stroke} stopOpacity={0} />
            </linearGradient>
          </defs>

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
              const labels: Record<string, string> = { util: "Util", vram: "VRAM", power: "Power" }
              return [`${value.toFixed(1)}%`, labels[name] ?? name]
            }}
          />

          <YAxis domain={[0, 100]} hide />
          <Area
            type="monotone"
            dataKey="util"
            stroke={CHART_COLORS.util.stroke}
            fill={`url(#util-${gpuIndex})`}
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
          />
          <Area
            type="monotone"
            dataKey="vramPct"
            stroke={CHART_COLORS.vram.stroke}
            fill={`url(#vram-${gpuIndex})`}
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
          />
          <Area
            type="monotone"
            dataKey="powerPct"
            stroke={CHART_COLORS.power.stroke}
            fill={`url(#power-${gpuIndex})`}
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
          />
        </AreaChart>
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
