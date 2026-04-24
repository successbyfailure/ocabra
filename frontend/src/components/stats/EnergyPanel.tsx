import type { EnergyStats, ServerPower } from "@/types"

interface EnergyPanelProps {
  data: EnergyStats
  serverPower?: ServerPower | null
}

export function EnergyPanel({ data, serverPower }: EnergyPanelProps) {
  const cpuPower = serverPower?.cpuPowerW
  const cpuTemp = serverPower?.cpuTempC
  const totalPower = serverPower?.totalPowerW
  const totalServerKwh = data.totalServerKwh ?? data.totalKwh

  return (
    <div className="space-y-3">
      {/* GPU cards */}
      <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
        {data.byGpu.map((gpu) => (
          <div key={gpu.gpuIndex} className="rounded-md border border-border bg-background/60 p-3">
            <p className="font-medium">GPU {gpu.gpuIndex}</p>
            <p className="text-sm text-muted-foreground">
              Consumo: <span className="font-semibold text-foreground">{gpu.powerDrawW.toFixed(0)} W</span>
            </p>
            <p className="text-sm text-muted-foreground">Periodo: {gpu.totalKwh.toFixed(3)} kWh</p>
          </div>
        ))}

        {/* CPU card (if available) */}
        {cpuPower != null && (
          <div className="rounded-md border border-border bg-background/60 p-3">
            <p className="font-medium">CPU</p>
            <p className="text-sm text-muted-foreground">
              Consumo: <span className="font-semibold text-foreground">{cpuPower.toFixed(0)} W</span>
            </p>
            {cpuTemp != null && (
              <p className="text-sm text-muted-foreground">Temp: {cpuTemp.toFixed(0)}°C</p>
            )}
            {data.cpuKwh != null && (
              <p className="text-sm text-muted-foreground">Periodo: {data.cpuKwh.toFixed(3)} kWh</p>
            )}
          </div>
        )}
      </div>

      {/* Totals */}
      <div className="flex flex-wrap items-center gap-6 border-t border-border pt-3 text-sm">
        {totalPower != null && (
          <div>
            <span className="text-muted-foreground">Potencia total: </span>
            <span className="font-semibold">{totalPower.toFixed(0)} W</span>
          </div>
        )}
        <div>
          <span className="text-muted-foreground">Energia periodo: </span>
          <span className="font-semibold">{totalServerKwh.toFixed(3)} kWh</span>
        </div>
        <div>
          <span className="text-muted-foreground">Coste: </span>
          <span className="font-semibold">{data.estimatedCostEur.toFixed(3)} EUR</span>
        </div>
      </div>
    </div>
  )
}
