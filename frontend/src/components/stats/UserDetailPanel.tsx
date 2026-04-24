import { useEffect, useState } from "react"
import { toast } from "sonner"
import { api } from "@/api/client"
import { TokensChart } from "@/components/stats/TokensChart"
import type { StatsParams, UserDetailStats } from "@/types"

interface UserDetailPanelProps {
  userId: string
  username: string
  params: StatsParams
  onClose: () => void
}

export function UserDetailPanel({ userId, username, params, onClose }: UserDetailPanelProps) {
  const [data, setData] = useState<UserDetailStats | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let active = true
    setLoading(true)
    api.stats
      .userDetail(userId, params)
      .then((d) => {
        if (active) setData(d)
      })
      .catch((err) => {
        if (active) toast.error(err instanceof Error ? err.message : "Error cargando detalle")
      })
      .finally(() => {
        if (active) setLoading(false)
      })
    return () => {
      active = false
    }
  }, [userId, params])

  if (loading) {
    return (
      <div className="rounded-lg border border-border bg-card p-4">
        <div className="h-64 animate-pulse rounded-lg bg-muted" />
      </div>
    )
  }

  if (!data) return null

  const totalTokens = data.totalInputTokens + data.totalOutputTokens
  const tokenSeries = {
    totalInputTokens: data.totalInputTokens,
    totalOutputTokens: data.totalOutputTokens,
    byBackend: [],
    series: data.tokenSeries,
  }

  return (
    <div className="rounded-lg border border-primary/30 bg-card p-4 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold">
          Detalle de <span className="font-mono text-primary">{username}</span>
        </h3>
        <button
          type="button"
          onClick={onClose}
          className="text-xs text-muted-foreground hover:text-foreground px-2 py-1 rounded border border-border hover:bg-muted"
        >
          Cerrar
        </button>
      </div>

      {/* KPIs */}
      <div className="grid gap-3 md:grid-cols-4">
        <div className="rounded-md border border-border bg-background/60 p-3">
          <p className="text-xs text-muted-foreground">Requests</p>
          <p className="text-xl font-semibold">{data.totalRequests}</p>
          <p className="text-xs text-muted-foreground">Errores: {data.totalErrors}</p>
        </div>
        <div className="rounded-md border border-border bg-background/60 p-3">
          <p className="text-xs text-muted-foreground">Tokens totales</p>
          <p className="text-xl font-semibold">{totalTokens.toLocaleString()}</p>
          <p className="text-xs text-muted-foreground">In: {data.totalInputTokens.toLocaleString()} / Out: {data.totalOutputTokens.toLocaleString()}</p>
        </div>
        <div className="rounded-md border border-border bg-background/60 p-3">
          <p className="text-xs text-muted-foreground">Energia</p>
          <p className="text-xl font-semibold">{data.totalEnergyWh.toFixed(1)} <span className="text-sm font-normal text-muted-foreground">Wh</span></p>
        </div>
        <div className="rounded-md border border-border bg-background/60 p-3">
          <p className="text-xs text-muted-foreground">Coste equiv. OpenAI</p>
          <p className="text-xl font-semibold text-orange-400">${data.estimatedCostUsd.toFixed(2)}</p>
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        {/* Top models */}
        <div className="rounded-md border border-border bg-background/40 p-3">
          <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">Top modelos</p>
          <div className="space-y-2">
            {(() => {
              const maxReq = Math.max(1, ...data.topModels.map((m) => m.requests))
              return data.topModels.map((m) => (
                <div key={m.modelId} className="space-y-0.5">
                  <div className="flex justify-between text-xs">
                    <span className="font-medium truncate max-w-[60%]" title={m.modelId}>{m.modelId}</span>
                    <span className="text-muted-foreground">{m.requests} req</span>
                  </div>
                  <div className="h-1.5 w-full rounded-full bg-muted overflow-hidden">
                    <div className="h-full rounded-full bg-primary/60 transition-all" style={{ width: `${(m.requests / maxReq) * 100}%` }} />
                  </div>
                </div>
              ))
            })()}
            {data.topModels.length === 0 && <p className="text-xs text-muted-foreground">Sin datos</p>}
          </div>
        </div>

        {/* By request kind */}
        <div className="rounded-md border border-border bg-background/40 p-3">
          <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">Tipos de peticion</p>
          <div className="space-y-2">
            {(() => {
              const maxCount = Math.max(1, ...data.byRequestKind.map((r) => r.count))
              return data.byRequestKind.map((r) => (
                <div key={r.requestKind} className="space-y-0.5">
                  <div className="flex justify-between text-xs">
                    <span className="font-medium">{r.requestKind}</span>
                    <span className="text-muted-foreground">{r.count}</span>
                  </div>
                  <div className="h-1.5 w-full rounded-full bg-muted overflow-hidden">
                    <div className="h-full rounded-full bg-cyan-500/60 transition-all" style={{ width: `${(r.count / maxCount) * 100}%` }} />
                  </div>
                </div>
              ))
            })()}
            {data.byRequestKind.length === 0 && <p className="text-xs text-muted-foreground">Sin datos</p>}
          </div>
        </div>
      </div>

      {/* Token time series */}
      {data.tokenSeries.length > 0 && (
        <TokensChart data={tokenSeries} />
      )}
    </div>
  )
}
