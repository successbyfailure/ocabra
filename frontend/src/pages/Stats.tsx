import { useEffect, useMemo, useState } from "react"
import { toast } from "sonner"
import { api } from "@/api/client"
import { DateRangePicker, defaultDateRange, type DateRangeValue } from "@/components/stats/DateRangePicker"
import { EnergyPanel } from "@/components/stats/EnergyPanel"
import { PerformanceTable } from "@/components/stats/PerformanceTable"
import { RequestsChart } from "@/components/stats/RequestsChart"
import { TokensChart } from "@/components/stats/TokensChart"
import type { EnergyStats, OverviewStats, PerformanceStats, RequestStats, TokenStats } from "@/types"

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
  byBackend: [],
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

const EMPTY_OVERVIEW: OverviewStats = {
  totalRequests: 0,
  totalErrors: 0,
  avgDurationMs: 0,
  tokenizedRequests: 0,
  totalInputTokens: 0,
  totalOutputTokens: 0,
  byBackend: [],
  byRequestKind: [],
}

function OverviewPanel({ data }: { data: OverviewStats }) {
  const errorPct = data.totalRequests > 0 ? (data.totalErrors / data.totalRequests) * 100 : 0

  return (
    <div className="rounded-lg border border-border bg-card p-3">
      <h3 className="mb-3 text-sm font-semibold text-muted-foreground">Overview</h3>
      <div className="grid gap-3 md:grid-cols-3">
        <div className="rounded-md border border-border bg-background/60 p-3">
          <p className="text-xs text-muted-foreground">Requests</p>
          <p className="text-2xl font-semibold">{data.totalRequests}</p>
          <p className="text-xs text-muted-foreground">Errores: {data.totalErrors} ({errorPct.toFixed(1)}%)</p>
        </div>
        <div className="rounded-md border border-border bg-background/60 p-3">
          <p className="text-xs text-muted-foreground">Latencia media</p>
          <p className="text-2xl font-semibold">{data.avgDurationMs} ms</p>
          <p className="text-xs text-muted-foreground">Tokenizadas: {data.tokenizedRequests}</p>
        </div>
        <div className="rounded-md border border-border bg-background/60 p-3">
          <p className="text-xs text-muted-foreground">Tokens</p>
          <p className="text-2xl font-semibold">{data.totalOutputTokens}</p>
          <p className="text-xs text-muted-foreground">Entrada: {data.totalInputTokens}</p>
        </div>
      </div>

      <div className="mt-3 grid gap-3 lg:grid-cols-2">
        <div className="rounded-md border border-border bg-background/40 p-3">
          <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">Por backend</p>
          <div className="space-y-1 text-sm">
            {data.byBackend.map((row) => (
              <p key={row.backendType}>
                {row.backendType}: {row.totalRequests} req, {row.avgLatencyMs} ms avg, {(row.errorRate * 100).toFixed(1)}% err
              </p>
            ))}
            {data.byBackend.length === 0 ? <p className="text-muted-foreground">Sin datos</p> : null}
          </div>
        </div>
        <div className="rounded-md border border-border bg-background/40 p-3">
          <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">Por tipo request</p>
          <div className="space-y-1 text-sm">
            {data.byRequestKind.map((row) => (
              <p key={row.requestKind}>
                {row.requestKind}: {row.totalRequests} req, {row.avgLatencyMs} ms avg, {(row.errorRate * 100).toFixed(1)}% err
              </p>
            ))}
            {data.byRequestKind.length === 0 ? <p className="text-muted-foreground">Sin datos</p> : null}
          </div>
        </div>
      </div>
    </div>
  )
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
  const [overview, setOverview] = useState<OverviewStats>(EMPTY_OVERVIEW)

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
        const [modelList, req, tok, ene, perf, over] = await Promise.all([
          api.models.list(),
          api.stats.requests(params),
          api.stats.tokens(params),
          api.stats.energy(params),
          api.stats.performance(params),
          api.stats.overview(params),
        ])

        if (!active) return

        setModels(modelList.map((model) => ({ id: model.modelId, label: model.displayName })))
        setRequests(req)
        setTokens(tok)
        setEnergy(ene)
        setPerformance(perf)
        setOverview(over)
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
          <OverviewPanel data={overview} />
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
