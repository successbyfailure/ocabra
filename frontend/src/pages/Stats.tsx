import { useEffect, useMemo, useState } from "react"
import * as Tabs from "@radix-ui/react-tabs"
import { toast } from "sonner"
import { api } from "@/api/client"
import { StyledSelect } from "@/components/common/StyledSelect"
import { useIsModelManager } from "@/hooks/useAuth"
import { DateRangePicker, defaultDateRange, type DateRangeValue } from "@/components/stats/DateRangePicker"
import { EnergyPanel } from "@/components/stats/EnergyPanel"
import { PerformanceTable } from "@/components/stats/PerformanceTable"
import { RequestsChart } from "@/components/stats/RequestsChart"
import { TokensChart } from "@/components/stats/TokensChart"
import { CostSavingsCard } from "@/components/stats/CostSavingsCard"
import { UserDetailPanel } from "@/components/stats/UserDetailPanel"
import { ApiKeyPanel } from "@/components/stats/ApiKeyPanel"
import { FederationPanel } from "@/components/stats/FederationPanel"
import type {
  ByApiKeyStats,
  ByGroupStats,
  ByUserStats,
  EnergyStats,
  FederationStats,
  MyGroupStats,
  OverviewStats,
  PerformanceStats,
  RecentRequestsData,
  RequestStats,
  ServerPower,
  TokenStats,
} from "@/types"

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

const EMPTY_BY_USER: ByUserStats = { byUser: [] }
const EMPTY_BY_GROUP: ByGroupStats = { byGroup: [] }
const EMPTY_BY_API_KEY: ByApiKeyStats = { byApiKey: [] }
const EMPTY_MY_GROUP: MyGroupStats = { groupId: null, groupName: null, stats: EMPTY_OVERVIEW }
const EMPTY_RECENT: RecentRequestsData = { requests: [] }
const EMPTY_FEDERATION: FederationStats = { localCount: 0, remoteCount: 0, byNode: [] }

function OverviewPanel({ data, title = "Overview" }: { data: OverviewStats; title?: string }) {
  const errorPct = data.totalRequests > 0 ? (data.totalErrors / data.totalRequests) * 100 : 0

  return (
    <div className="rounded-lg border border-border bg-card p-3">
      <h3 className="mb-3 text-sm font-semibold text-muted-foreground">{title}</h3>
      <div className="grid gap-3 md:grid-cols-3">
        <div className="rounded-md border border-border bg-background/60 p-3">
          <p className="text-xs text-muted-foreground">Requests</p>
          <p className="text-2xl font-semibold">{data.totalRequests}</p>
          <p className="text-xs text-muted-foreground">Errores: {data.totalErrors} ({errorPct.toFixed(1)}%)</p>
        </div>
        <div className="rounded-md border border-border bg-background/60 p-3">
          <p className="text-xs text-muted-foreground">Latencia media (P50)</p>
          <p className="text-2xl font-semibold">{data.avgDurationMs} <span className="text-base font-normal text-muted-foreground">ms</span></p>
          <p className="text-xs text-muted-foreground">Requests tokenizadas: {data.tokenizedRequests.toLocaleString()}</p>
        </div>
        <div className="rounded-md border border-border bg-background/60 p-3">
          <p className="text-xs text-muted-foreground">Tokens generados</p>
          <p className="text-2xl font-semibold">{data.totalOutputTokens.toLocaleString()}</p>
          <p className="text-xs text-muted-foreground">Entrada: {data.totalInputTokens.toLocaleString()} tokens</p>
        </div>
      </div>

      <div className="mt-3 grid gap-3 lg:grid-cols-2">
        <div className="rounded-md border border-border bg-background/40 p-3">
          <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">Por backend</p>
          <div className="space-y-2">
            {(() => {
              const maxReq = Math.max(1, ...data.byBackend.map((r) => r.totalRequests))
              return data.byBackend.map((row) => (
                <div key={row.backendType} className="space-y-0.5">
                  <div className="flex justify-between text-xs">
                    <span className="font-medium">{row.backendType}</span>
                    <span className="text-muted-foreground">{row.totalRequests} req · {row.avgLatencyMs} ms · {(row.errorRate * 100).toFixed(1)}% err</span>
                  </div>
                  <div className="h-1.5 w-full rounded-full bg-muted overflow-hidden">
                    <div className="h-full rounded-full bg-primary/60 transition-all" style={{ width: `${(row.totalRequests / maxReq) * 100}%` }} />
                  </div>
                </div>
              ))
            })()}
            {data.byBackend.length === 0 && <p className="text-sm text-muted-foreground">Sin datos</p>}
          </div>
        </div>
        <div className="rounded-md border border-border bg-background/40 p-3">
          <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">Por tipo request</p>
          <div className="space-y-2">
            {(() => {
              const maxReq = Math.max(1, ...data.byRequestKind.map((r) => r.totalRequests))
              return data.byRequestKind.map((row) => (
                <div key={row.requestKind} className="space-y-0.5">
                  <div className="flex justify-between text-xs">
                    <span className="font-medium">{row.requestKind}</span>
                    <span className="text-muted-foreground">{row.totalRequests} req · {row.avgLatencyMs} ms · {(row.errorRate * 100).toFixed(1)}% err</span>
                  </div>
                  <div className="h-1.5 w-full rounded-full bg-muted overflow-hidden">
                    <div className="h-full rounded-full bg-cyan-500/60 transition-all" style={{ width: `${(row.totalRequests / maxReq) * 100}%` }} />
                  </div>
                </div>
              ))
            })()}
            {data.byRequestKind.length === 0 && <p className="text-sm text-muted-foreground">Sin datos</p>}
          </div>
        </div>
      </div>
    </div>
  )
}

function OwnStatsPanel({ data }: { data: OverviewStats }) {
  return (
    <div className="rounded-lg border border-border bg-card p-3">
      <h3 className="mb-3 text-sm font-semibold text-muted-foreground">Mis estadisticas</h3>
      <div className="grid gap-3 md:grid-cols-3">
        <div className="rounded-md border border-border bg-background/60 p-3">
          <p className="text-xs text-muted-foreground">Mis requests</p>
          <p className="text-2xl font-semibold">{data.totalRequests}</p>
          <p className="text-xs text-muted-foreground">Errores: {data.totalErrors}</p>
        </div>
        <div className="rounded-md border border-border bg-background/60 p-3">
          <p className="text-xs text-muted-foreground">Tokens enviados</p>
          <p className="text-2xl font-semibold">{data.totalInputTokens}</p>
          <p className="text-xs text-muted-foreground">Recibidos: {data.totalOutputTokens}</p>
        </div>
        <div className="rounded-md border border-border bg-background/60 p-3">
          <p className="text-xs text-muted-foreground">Latencia media</p>
          <p className="text-2xl font-semibold">{data.avgDurationMs} ms</p>
          <p className="text-xs text-muted-foreground">
            {data.estimatedCostUsd != null && data.estimatedCostUsd > 0
              ? `Equiv. OpenAI: $${data.estimatedCostUsd.toFixed(2)}`
              : `Tokenizadas: ${data.tokenizedRequests}`}
          </p>
        </div>
      </div>
    </div>
  )
}

function ByUserPanel({
  data,
  params,
}: {
  data: ByUserStats
  params: { from: string; to: string; modelId?: string }
}) {
  const [selectedUserId, setSelectedUserId] = useState<string | null>(null)
  const selectedUser = data.byUser.find((u) => u.userId === selectedUserId)

  return (
    <div className="space-y-4">
      <div className="rounded-lg border border-border bg-card p-3">
        <h3 className="mb-3 text-sm font-semibold text-muted-foreground">Stats por usuario</h3>
        {data.byUser.length === 0 ? (
          <p className="text-sm text-muted-foreground py-4 text-center">Sin datos</p>
        ) : (
          <div className="overflow-auto">
            <table className="min-w-full divide-y divide-border text-sm">
              <thead className="bg-muted/50">
                <tr>
                  <th className="px-3 py-2 text-left font-medium text-muted-foreground">Usuario</th>
                  <th className="px-3 py-2 text-right font-medium text-muted-foreground">Requests</th>
                  <th className="px-3 py-2 text-right font-medium text-muted-foreground">Errores</th>
                  <th className="px-3 py-2 text-right font-medium text-muted-foreground">Lat. media</th>
                  <th className="px-3 py-2 text-right font-medium text-muted-foreground">Tokens in</th>
                  <th className="px-3 py-2 text-right font-medium text-muted-foreground">Tokens out</th>
                  <th className="px-3 py-2 text-right font-medium text-muted-foreground">Energia (Wh)</th>
                  <th className="px-3 py-2 text-right font-medium text-muted-foreground">Coste ref.</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border bg-card">
                {data.byUser.map((row) => (
                  <tr
                    key={row.userId}
                    className={`cursor-pointer transition-colors ${selectedUserId === row.userId ? "bg-primary/10" : "hover:bg-muted/30"}`}
                    onClick={() => setSelectedUserId(selectedUserId === row.userId ? null : row.userId)}
                  >
                    <td className="px-3 py-2 font-mono">{row.username}</td>
                    <td className="px-3 py-2 text-right">{row.totalRequests}</td>
                    <td className="px-3 py-2 text-right">{row.totalErrors}</td>
                    <td className="px-3 py-2 text-right">{row.avgDurationMs} ms</td>
                    <td className="px-3 py-2 text-right">{row.totalInputTokens.toLocaleString()}</td>
                    <td className="px-3 py-2 text-right">{row.totalOutputTokens.toLocaleString()}</td>
                    <td className="px-3 py-2 text-right">{(row.totalEnergyWh ?? 0).toFixed(1)}</td>
                    <td className="px-3 py-2 text-right text-orange-400">${(row.estimatedCostUsd ?? 0).toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {selectedUserId && selectedUser && (
        <UserDetailPanel
          userId={selectedUserId}
          username={selectedUser.username}
          params={params}
          onClose={() => setSelectedUserId(null)}
        />
      )}
    </div>
  )
}

function ByGroupPanel({ data }: { data: ByGroupStats }) {
  return (
    <div className="rounded-lg border border-border bg-card p-3">
      <h3 className="mb-3 text-sm font-semibold text-muted-foreground">Stats por grupo</h3>
      {data.byGroup.length === 0 ? (
        <p className="text-sm text-muted-foreground py-4 text-center">Sin datos</p>
      ) : (
        <div className="overflow-auto">
          <table className="min-w-full divide-y divide-border text-sm">
            <thead className="bg-muted/50">
              <tr>
                <th className="px-3 py-2 text-left font-medium text-muted-foreground">Grupo</th>
                <th className="px-3 py-2 text-right font-medium text-muted-foreground">Requests</th>
                <th className="px-3 py-2 text-right font-medium text-muted-foreground">Errores</th>
                <th className="px-3 py-2 text-right font-medium text-muted-foreground">Lat. media</th>
                <th className="px-3 py-2 text-right font-medium text-muted-foreground">Tokens in</th>
                <th className="px-3 py-2 text-right font-medium text-muted-foreground">Tokens out</th>
                <th className="px-3 py-2 text-right font-medium text-muted-foreground">Energia (Wh)</th>
                <th className="px-3 py-2 text-right font-medium text-muted-foreground">Coste ref.</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border bg-card">
              {data.byGroup.map((row) => (
                <tr key={row.groupId} className="hover:bg-muted/30">
                  <td className="px-3 py-2 font-mono">{row.groupName}</td>
                  <td className="px-3 py-2 text-right">{row.totalRequests}</td>
                  <td className="px-3 py-2 text-right">{row.totalErrors}</td>
                  <td className="px-3 py-2 text-right">{row.avgDurationMs} ms</td>
                  <td className="px-3 py-2 text-right">{row.totalInputTokens.toLocaleString()}</td>
                  <td className="px-3 py-2 text-right">{row.totalOutputTokens.toLocaleString()}</td>
                  <td className="px-3 py-2 text-right">{(row.totalEnergyWh ?? 0).toFixed(1)}</td>
                  <td className="px-3 py-2 text-right text-orange-400">${(row.estimatedCostUsd ?? 0).toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

function MyGroupPanel({ data }: { data: MyGroupStats }) {
  const stats = data.stats
  const errorPct = stats.totalRequests > 0 ? (stats.totalErrors / stats.totalRequests) * 100 : 0

  return (
    <div className="rounded-lg border border-border bg-card p-3">
      <h3 className="mb-3 text-sm font-semibold text-muted-foreground">
        Mi grupo{data.groupName ? ` — ${data.groupName}` : ""}
      </h3>
      {data.groupId == null ? (
        <p className="text-sm text-muted-foreground py-4 text-center">No perteneces a ningun grupo.</p>
      ) : (
        <div className="grid gap-3 md:grid-cols-3">
          <div className="rounded-md border border-border bg-background/60 p-3">
            <p className="text-xs text-muted-foreground">Requests</p>
            <p className="text-2xl font-semibold">{stats.totalRequests}</p>
            <p className="text-xs text-muted-foreground">Errores: {stats.totalErrors} ({errorPct.toFixed(1)}%)</p>
          </div>
          <div className="rounded-md border border-border bg-background/60 p-3">
            <p className="text-xs text-muted-foreground">Tokens entrada</p>
            <p className="text-2xl font-semibold">{stats.totalInputTokens}</p>
            <p className="text-xs text-muted-foreground">Salida: {stats.totalOutputTokens}</p>
          </div>
          <div className="rounded-md border border-border bg-background/60 p-3">
            <p className="text-xs text-muted-foreground">Latencia media</p>
            <p className="text-2xl font-semibold">{stats.avgDurationMs} ms</p>
            <p className="text-xs text-muted-foreground">Tokenizadas: {stats.tokenizedRequests}</p>
          </div>
        </div>
      )}
    </div>
  )
}

function RequestLogPanel({ data }: { data: RecentRequestsData }) {
  return (
    <div className="rounded-lg border border-border bg-card p-3">
      <h3 className="mb-3 text-sm font-semibold text-muted-foreground">Log de peticiones</h3>
      {data.requests.length === 0 ? (
        <p className="text-sm text-muted-foreground py-4 text-center">Sin datos</p>
      ) : (
        <div className="overflow-auto">
          <table className="min-w-full divide-y divide-border text-sm">
            <thead className="bg-muted/50">
              <tr>
                <th className="px-3 py-2 text-left font-medium text-muted-foreground">Hora</th>
                <th className="px-3 py-2 text-left font-medium text-muted-foreground">Modelo</th>
                <th className="px-3 py-2 text-left font-medium text-muted-foreground">Tipo</th>
                <th className="px-3 py-2 text-right font-medium text-muted-foreground">Duracion</th>
                <th className="px-3 py-2 text-right font-medium text-muted-foreground">Tokens in/out</th>
                <th className="px-3 py-2 text-left font-medium text-muted-foreground">Usuario</th>
                <th className="px-3 py-2 text-left font-medium text-muted-foreground">Estado</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border bg-card">
              {data.requests.map((req) => {
                const statusColor =
                  req.error
                    ? "text-red-400"
                    : req.statusCode != null && req.statusCode < 400
                      ? "text-emerald-400"
                      : "text-muted-foreground"
                const statusLabel =
                  req.error
                    ? `Error: ${req.error.substring(0, 40)}`
                    : req.statusCode != null
                      ? String(req.statusCode)
                      : "—"
                const hora = new Date(req.startedAt).toLocaleTimeString([], {
                  hour: "2-digit",
                  minute: "2-digit",
                  second: "2-digit",
                })
                return (
                  <tr key={req.id} className="hover:bg-muted/30">
                    <td className="px-3 py-2 font-mono text-xs">{hora}</td>
                    <td className="px-3 py-2 max-w-[12rem] truncate" title={req.modelId}>{req.modelId}</td>
                    <td className="px-3 py-2 text-muted-foreground">{req.requestKind ?? req.backendType ?? "—"}</td>
                    <td className="px-3 py-2 text-right">{req.durationMs != null ? `${req.durationMs} ms` : "—"}</td>
                    <td className="px-3 py-2 text-right">
                      {req.inputTokens != null || req.outputTokens != null
                        ? `${req.inputTokens ?? 0} / ${req.outputTokens ?? 0}`
                        : "—"}
                    </td>
                    <td className="px-3 py-2 text-muted-foreground">{req.username ?? "—"}</td>
                    <td className={`px-3 py-2 text-xs font-medium ${statusColor}`} title={req.error ?? undefined}>
                      {statusLabel}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

export function Stats() {
  const isManagerOrAdmin = useIsModelManager()

  const [loading, setLoading] = useState(true)
  const [range, setRange] = useState<DateRangeValue>(defaultDateRange)
  const [models, setModels] = useState<{ id: string; label: string }[]>([])
  const [modelId, setModelId] = useState<string>("")
  const [activeTab, setActiveTab] = useState("overview")

  // Global stats (visible to manager/admin)
  const [requests, setRequests] = useState<RequestStats>(EMPTY_REQUESTS)
  const [tokens, setTokens] = useState<TokenStats>(EMPTY_TOKENS)
  const [energy, setEnergy] = useState<EnergyStats>(EMPTY_ENERGY)
  const [performance, setPerformance] = useState<PerformanceStats>(EMPTY_PERFORMANCE)
  const [overview, setOverview] = useState<OverviewStats>(EMPTY_OVERVIEW)
  const [byUser, setByUser] = useState<ByUserStats>(EMPTY_BY_USER)
  const [byGroup, setByGroup] = useState<ByGroupStats>(EMPTY_BY_GROUP)
  const [byApiKey, setByApiKey] = useState<ByApiKeyStats>(EMPTY_BY_API_KEY)
  const [recent, setRecent] = useState<RecentRequestsData>(EMPTY_RECENT)
  const [federationStats, setFederationStats] = useState<FederationStats>(EMPTY_FEDERATION)
  const [serverPower, setServerPower] = useState<ServerPower | null>(null)

  // Own stats
  const [myStats, setMyStats] = useState<OverviewStats>(EMPTY_OVERVIEW)
  const [myGroup, setMyGroup] = useState<MyGroupStats>(EMPTY_MY_GROUP)

  const params = useMemo(
    () => ({
      from: new Date(range.from).toISOString(),
      to: new Date(range.to).toISOString(),
      modelId: modelId || undefined,
    }),
    [modelId, range.from, range.to],
  )

  const hasFederation = federationStats.localCount + federationStats.remoteCount > 0

  useEffect(() => {
    let active = true

    const load = async () => {
      try {
        if (isManagerOrAdmin) {
          const [modelList, req, tok, ene, perf, over, bu, bg, bak, rec, my, mg, fed, sp] = await Promise.all([
            api.models.list(),
            api.stats.requests(params),
            api.stats.tokens(params),
            api.stats.energy(params),
            api.stats.performance(params),
            api.stats.overview(params),
            api.stats.byUser(params),
            api.stats.byGroup(params),
            api.stats.byApiKey(params).catch(() => EMPTY_BY_API_KEY),
            api.stats.recent(20),
            api.stats.my(params),
            api.stats.myGroup(params),
            api.stats.federation(params).catch(() => EMPTY_FEDERATION),
            api.stats.serverPower().catch(() => null),
          ])

          if (!active) return

          setModels(modelList.map((model) => ({ id: model.modelId, label: model.displayName })))
          setRequests(req)
          setTokens(tok)
          setEnergy(ene)
          setPerformance(perf)
          setOverview(over)
          setByUser(bu)
          setByGroup(bg)
          setByApiKey(bak)
          setRecent(rec)
          setMyStats(my)
          setMyGroup(mg)
          setFederationStats(fed)
          setServerPower(sp)
        } else {
          const [modelList, ene, my, mg, sp] = await Promise.all([
            api.models.list(),
            api.stats.energy(params),
            api.stats.my(params),
            api.stats.myGroup(params),
            api.stats.serverPower().catch(() => null),
          ])

          if (!active) return

          setModels(modelList.map((model) => ({ id: model.modelId, label: model.displayName })))
          setEnergy(ene)
          setMyStats(my)
          setMyGroup(mg)
          setServerPower(sp)
        }
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
  }, [isManagerOrAdmin, modelId, params])

  const tabTriggerClass = (value: string) =>
    `px-4 py-2 text-sm font-medium rounded-t-md border-b-2 transition-colors ${
      activeTab === value
        ? "border-primary text-foreground"
        : "border-transparent text-muted-foreground hover:text-foreground hover:border-border"
    }`

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-2xl font-semibold">Stats</h1>
        <p className="text-muted-foreground">Requests, tokens, latencia, energia, costes y rendimiento.</p>
      </div>

      {/* Filter bar */}
      <div className="flex flex-wrap items-end gap-3 rounded-lg border border-border bg-card p-3">
        <div className="flex-1 min-w-[200px]">
          <DateRangePicker value={range} onChange={setRange} />
        </div>
        <div className="min-w-[200px] md:w-80">
          <StyledSelect
            value={modelId}
            onValueChange={setModelId}
            className="w-full"
            placeholder="Todos los modelos"
            options={[
              { value: "", label: "Todos los modelos" },
              ...models.map((model) => ({ value: model.id, label: model.label })),
            ]}
          />
        </div>
      </div>

      {loading ? (
        <div className="space-y-3">
          <div className="h-72 animate-pulse rounded-lg bg-muted" />
          <div className="h-72 animate-pulse rounded-lg bg-muted" />
          <div className="h-56 animate-pulse rounded-lg bg-muted" />
        </div>
      ) : (
        <Tabs.Root value={activeTab} onValueChange={setActiveTab}>
          <Tabs.List className="flex border-b border-border overflow-x-auto" aria-label="Secciones de estadisticas">
            <Tabs.Trigger value="overview" className={tabTriggerClass("overview")}>
              Resumen
            </Tabs.Trigger>
            {isManagerOrAdmin && (
              <Tabs.Trigger value="models" className={tabTriggerClass("models")}>
                Modelos
              </Tabs.Trigger>
            )}
            {isManagerOrAdmin && (
              <Tabs.Trigger value="by-user" className={tabTriggerClass("by-user")}>
                Usuarios
              </Tabs.Trigger>
            )}
            {isManagerOrAdmin && (
              <Tabs.Trigger value="by-group" className={tabTriggerClass("by-group")}>
                Grupos
              </Tabs.Trigger>
            )}
            {isManagerOrAdmin && (
              <Tabs.Trigger value="api-keys" className={tabTriggerClass("api-keys")}>
                API Keys
              </Tabs.Trigger>
            )}
            {isManagerOrAdmin && hasFederation && (
              <Tabs.Trigger value="federation" className={tabTriggerClass("federation")}>
                Federacion
              </Tabs.Trigger>
            )}
            {isManagerOrAdmin && (
              <Tabs.Trigger value="log" className={tabTriggerClass("log")}>
                Log
              </Tabs.Trigger>
            )}
          </Tabs.List>

          <div className="pt-4">
            <Tabs.Content value="overview" className="space-y-4">
              {/* Cost savings card (admin only, when there's data) */}
              {isManagerOrAdmin && (overview.estimatedCostUsd ?? 0) > 0 && (
                <CostSavingsCard overview={overview} energy={energy} />
              )}

              {/* Energy panel */}
              <div className="rounded-lg border border-border bg-card p-3">
                <h3 className="mb-3 text-sm font-semibold text-muted-foreground">Energia del servidor</h3>
                <EnergyPanel data={energy} serverPower={serverPower} />
              </div>

              {isManagerOrAdmin ? (
                <>
                  <OverviewPanel data={overview} title="Overview global" />
                  <div className="grid gap-3 xl:grid-cols-2">
                    <RequestsChart data={requests} />
                    <TokensChart data={tokens} />
                  </div>
                </>
              ) : null}

              {/* Personal stats — always shown at bottom of overview */}
              <div className="grid gap-4 lg:grid-cols-2">
                <OwnStatsPanel data={myStats} />
                <MyGroupPanel data={myGroup} />
              </div>
            </Tabs.Content>

            {isManagerOrAdmin && (
              <Tabs.Content value="models">
                <PerformanceTable data={performance} />
              </Tabs.Content>
            )}

            {isManagerOrAdmin && (
              <Tabs.Content value="by-user">
                <ByUserPanel data={byUser} params={params} />
              </Tabs.Content>
            )}

            {isManagerOrAdmin && (
              <Tabs.Content value="by-group">
                <ByGroupPanel data={byGroup} />
              </Tabs.Content>
            )}

            {isManagerOrAdmin && (
              <Tabs.Content value="api-keys">
                <ApiKeyPanel data={byApiKey} />
              </Tabs.Content>
            )}

            {isManagerOrAdmin && hasFederation && (
              <Tabs.Content value="federation">
                <FederationPanel data={federationStats} />
              </Tabs.Content>
            )}

            {isManagerOrAdmin && (
              <Tabs.Content value="log">
                <RequestLogPanel data={recent} />
              </Tabs.Content>
            )}
          </div>
        </Tabs.Root>
      )}
    </div>
  )
}
