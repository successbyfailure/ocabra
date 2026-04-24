import type { EnergyStats, OverviewStats } from "@/types"

interface CostSavingsCardProps {
  overview: OverviewStats
  energy: EnergyStats
}

export function CostSavingsCard({ overview, energy }: CostSavingsCardProps) {
  const openaiCost = overview.estimatedCostUsd ?? 0
  const electricityCost = energy.estimatedCostEur ?? 0
  const savings = Math.max(0, openaiCost - electricityCost)

  return (
    <div className="rounded-lg border border-border bg-card p-3">
      <h3 className="mb-3 text-sm font-semibold text-muted-foreground">Coste equivalente OpenAI</h3>
      <div className="grid gap-3 md:grid-cols-3">
        <div className="rounded-md border border-border bg-background/60 p-3">
          <p className="text-xs text-muted-foreground">En OpenAI costaria</p>
          <p className="text-2xl font-semibold text-orange-400">${openaiCost.toFixed(2)}</p>
          <p className="text-xs text-muted-foreground">
            {(overview.totalInputTokens + overview.totalOutputTokens).toLocaleString()} tokens totales
          </p>
        </div>
        <div className="rounded-md border border-border bg-background/60 p-3">
          <p className="text-xs text-muted-foreground">Coste real electricidad</p>
          <p className="text-2xl font-semibold text-blue-400">{electricityCost.toFixed(3)} EUR</p>
          <p className="text-xs text-muted-foreground">
            {(energy.totalServerKwh ?? energy.totalKwh).toFixed(3)} kWh consumidos
          </p>
        </div>
        <div className="rounded-md border border-border bg-background/60 p-3">
          <p className="text-xs text-muted-foreground">Ahorro estimado</p>
          <p className="text-2xl font-semibold text-emerald-400">${savings.toFixed(2)}</p>
          <p className="text-xs text-muted-foreground">
            {openaiCost > 0 ? `${((savings / openaiCost) * 100).toFixed(0)}% ahorro` : "—"}
          </p>
        </div>
      </div>
    </div>
  )
}
