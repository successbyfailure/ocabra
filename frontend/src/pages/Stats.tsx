import { useEffect, useMemo, useState } from "react"
import { toast } from "sonner"
import { api } from "@/api/client"
import { DateRangePicker, defaultDateRange, type DateRangeValue } from "@/components/stats/DateRangePicker"
import { EnergyPanel } from "@/components/stats/EnergyPanel"
import { PerformanceTable } from "@/components/stats/PerformanceTable"
import { RequestsChart } from "@/components/stats/RequestsChart"
import { TokensChart } from "@/components/stats/TokensChart"
import type { EnergyStats, PerformanceStats, RequestStats, TokenStats } from "@/types"

const EMPTY_REQUESTS: RequestStats = {
  totalRequests: 0,
  errorRate: 0,
  avgDurationMs: 0,
  p50DurationMs: 0,
  p95DurationMs: 0,
  series: [],
}

const EMPTY_TOKENS: TokenStats = {
  totalInputTokens: 0,
  totalOutputTokens: 0,
  series: [],
}

const EMPTY_ENERGY: EnergyStats = {
  totalKwh: 0,
  estimatedCostEur: 0,
  byGpu: [],
}

const EMPTY_PERFORMANCE: PerformanceStats = {
  byModel: [],
}

export function Stats() {
  const [loading, setLoading] = useState(true)
  const [range, setRange] = useState<DateRangeValue>(defaultDateRange)
  const [models, setModels] = useState<{ id: string; label: string }[]>([])
  const [modelId, setModelId] = useState<string>("")
  const [requests, setRequests] = useState<RequestStats>(EMPTY_REQUESTS)
  const [tokens, setTokens] = useState<TokenStats>(EMPTY_TOKENS)
  const [energy, setEnergy] = useState<EnergyStats>(EMPTY_ENERGY)
  const [performance, setPerformance] = useState<PerformanceStats>(EMPTY_PERFORMANCE)

  const params = useMemo(
    () => ({
      from: new Date(range.from).toISOString(),
      to: new Date(range.to).toISOString(),
      modelId: modelId || undefined,
    }),
    [modelId, range.from, range.to],
  )

  useEffect(() => {
    let active = true

    const load = async () => {
      try {
        const [modelList, req, tok, ene, perf] = await Promise.all([
          api.models.list(),
          api.stats.requests(params),
          api.stats.tokens(params),
          api.stats.energy(params),
          api.stats.performance(modelId || undefined),
        ])

        if (!active) return

        setModels(modelList.map((model) => ({ id: model.modelId, label: model.displayName })))
        setRequests(req)
        setTokens(tok)
        setEnergy(ene)
        setPerformance(perf)
      } catch (err) {
        if (active) {
          toast.error(err instanceof Error ? err.message : "No se pudieron cargar stats")
        }
      } finally {
        if (active) setLoading(false)
      }
    }

    void load()
    const timer = window.setInterval(() => {
      void load()
    }, 30_000)

    return () => {
      active = false
      window.clearInterval(timer)
    }
  }, [modelId, params])

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-2xl font-semibold">Stats</h1>
        <p className="text-muted-foreground">Requests, tokens, latencia, energia y rendimiento.</p>
      </div>

      <DateRangePicker value={range} onChange={setRange} />

      <select
        value={modelId}
        onChange={(event) => setModelId(event.target.value)}
        className="w-full rounded-md border border-border bg-card px-3 py-2 text-sm md:w-80"
      >
        <option value="">Todos los modelos</option>
        {models.map((model) => (
          <option key={model.id} value={model.id}>
            {model.label}
          </option>
        ))}
      </select>

      {loading ? (
        <div className="space-y-3">
          <div className="h-72 animate-pulse rounded-lg bg-muted" />
          <div className="h-72 animate-pulse rounded-lg bg-muted" />
          <div className="h-56 animate-pulse rounded-lg bg-muted" />
        </div>
      ) : (
        <>
          <div className="grid gap-3 xl:grid-cols-2">
            <RequestsChart data={requests} />
            <TokensChart data={tokens} />
          </div>
          <div className="grid gap-3 xl:grid-cols-[340px_minmax(0,1fr)]">
            <EnergyPanel data={energy} />
            <PerformanceTable data={performance} />
          </div>
        </>
      )}
    </div>
  )
}
