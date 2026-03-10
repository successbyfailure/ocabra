import type { EnergyStats } from "@/types"

interface EnergyPanelProps {
  data: EnergyStats
}

export function EnergyPanel({ data }: EnergyPanelProps) {
  return (
    <div className="rounded-lg border border-border bg-card p-3">
      <h3 className="mb-3 text-sm font-semibold text-muted-foreground">Energy</h3>
      <div className="space-y-3">
        {data.byGpu.map((gpu) => (
          <div key={gpu.gpuIndex} className="rounded-md border border-border bg-background/60 p-3">
            <p className="font-medium">GPU {gpu.gpuIndex}</p>
            <p className="text-sm text-muted-foreground">Consumo actual: {gpu.powerDrawW.toFixed(1)} W</p>
            <p className="text-sm text-muted-foreground">kWh esta sesion: {gpu.totalKwh.toFixed(3)}</p>
          </div>
        ))}
      </div>
      <div className="mt-3 border-t border-border pt-3 text-sm text-muted-foreground">
        <p>Total kWh: {data.totalKwh.toFixed(3)}</p>
        <p>Coste estimado: EUR {data.estimatedCostEur.toFixed(3)}</p>
      </div>
    </div>
  )
}
