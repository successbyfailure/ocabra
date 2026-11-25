import type { BenchmarkAttempt, BenchmarkRun } from "@/types"
import { parseBenchmarkResults, type BenchmarkMetric } from "./resultAdapters"

export type RankingDimension = "corrala" | "escriba" | "bfcl" | "ifeval" | "fulcrum"
export type RankingFactor = RankingDimension | "reliability" | "performance" | "efficiency"
export type RankingProfileId = "balanced" | "quality" | "latency" | "efficiency" | "custom"

export type RankingWeights = Record<RankingFactor, number>

export interface RankingProfile {
  id: Exclude<RankingProfileId, "custom">
  label: string
  description: string
  weights: RankingWeights
}

export interface RankingFilters {
  ocabraVersion: string
  hardware: string
}

export interface RankingSource {
  dimension: RankingDimension
  runId: string
  app: string
  suite: string
  revision: string
  createdAt: string
  quality: number
  performance: number | null
}

export interface ModelRankingRow {
  id: string
  model: string
  profile: string
  backend: string
  score: number
  quality: number
  reliability: number
  performance: number
  efficiency: number
  coverage: number
  dimensions: RankingDimension[]
  eligible: boolean
  versions: string[]
  hardware: string[]
  sources: RankingSource[]
  contributions: Record<RankingFactor, number>
}

interface Observation {
  identity: string
  model: string
  profile: string
  backend: string
  dimension: RankingDimension
  run: BenchmarkRun
  attempt?: BenchmarkAttempt
  quality: number
  speedValue: number | null
  speedDirection: "higher" | "lower" | null
  reliability: number
  gpuCount: number
  version: string
  hardware: string
  performance: number
  efficiency: number
}

const QUALITY_DIMENSIONS: RankingDimension[] = ["corrala", "escriba", "bfcl", "ifeval", "fulcrum"]
const UNKNOWN_BACKEND = "backend no informado"

export const RANKING_PROFILES: RankingProfile[] = [
  {
    id: "balanced",
    label: "Producción equilibrada",
    description: "Calidad real y pública, fiabilidad, rendimiento y coste de GPU.",
    weights: { corrala: 18, escriba: 18, bfcl: 17, ifeval: 12, fulcrum: 10, reliability: 10, performance: 10, efficiency: 5 },
  },
  {
    id: "quality",
    label: "Calidad máxima",
    description: "Prioriza las tareas reales y las evaluaciones de calidad.",
    weights: { corrala: 22, escriba: 22, bfcl: 18, ifeval: 13, fulcrum: 15, reliability: 5, performance: 3, efficiency: 2 },
  },
  {
    id: "latency",
    label: "Baja latencia",
    description: "Da más peso al tiempo de respuesta sin ignorar la calidad.",
    weights: { corrala: 11, escriba: 10, bfcl: 9, ifeval: 8, fulcrum: 7, reliability: 15, performance: 35, efficiency: 5 },
  },
  {
    id: "efficiency",
    label: "Infra eficiente",
    description: "Premia el resultado obtenido por GPU utilizada.",
    weights: { corrala: 13, escriba: 13, bfcl: 9, ifeval: 8, fulcrum: 7, reliability: 15, performance: 10, efficiency: 25 },
  },
]

export const DEFAULT_CUSTOM_WEIGHTS: RankingWeights = {
  corrala: 15,
  escriba: 15,
  bfcl: 14,
  ifeval: 14,
  fulcrum: 12,
  reliability: 10,
  performance: 10,
  efficiency: 10,
}

function clamp(value: number): number {
  return Math.max(0, Math.min(1, value))
}

function dimensionFor(run: BenchmarkRun): RankingDimension | null {
  if (run.app === "corrala" && run.suite === "llm") return "corrala"
  if (run.app === "escriba" && run.suite === "llm") return "escriba"
  if (run.app === "bfcl") return "bfcl"
  if (run.app === "ifeval") return "ifeval"
  if (run.app === "fulcrum" && run.suite === "orquestacion") return "fulcrum"
  return null
}

function latestAttempt(run: BenchmarkRun, profile: string): BenchmarkAttempt | undefined {
  return [...run.attempts].reverse().find((attempt) => (
    attempt.status === "completed"
    && (attempt.model === profile || attempt.environment.model?.profile === profile)
  ))
}

function withNode(backend: string, node: string | null | undefined): string {
  return node ? `${backend} @ ${node}` : backend
}

function attemptBackend(attempt: BenchmarkAttempt | undefined): string {
  return attempt?.environment.model?.backends?.join("+") || UNKNOWN_BACKEND
}

function attemptHardware(attempt: BenchmarkAttempt | undefined): string {
  const gpus = attempt?.environment.hardware?.gpus ?? []
  if (gpus.length === 0) return "GPU no informada"
  return gpus.map((gpu) => gpu.name || `GPU ${gpu.index}`).join(" + ")
}

function attemptVersion(attempt: BenchmarkAttempt | undefined): string {
  return attempt?.environment.software?.ocabra || "versión no informada"
}

function numericMetric(metrics: BenchmarkMetric[], keys: string[]): BenchmarkMetric | undefined {
  for (const key of keys) {
    const found = metrics.find((metric) => metric.key === key && typeof metric.value === "number")
    if (found) return found
  }
  return undefined
}

function harmonicMean(left: number, right: number): number {
  return left > 0 && right > 0 ? (2 * left * right) / (left + right) : 0
}

function observationQuality(run: BenchmarkRun, profile: string, dimension: RankingDimension): number | null {
  const result = run.result[profile]
  if (!result) return null
  const rows = parseBenchmarkResults({ [profile]: result })
  if (rows.length === 0) return null

  if (dimension === "corrala") {
    const global = rows.find((row) => row.scope.startsWith("Global")) ?? rows[0]
    const value = numericMetric(global.metrics, ["pass_rate", "primary_rate"])?.value
    return typeof value === "number" ? clamp(value) : null
  }
  if (dimension === "escriba") {
    const values = rows.flatMap((row) => {
      const coverage = numericMetric(row.metrics, ["coverage"])?.value
      const precision = numericMetric(row.metrics, ["precision"])?.value
      return typeof coverage === "number" && typeof precision === "number"
        ? [harmonicMean(clamp(coverage), clamp(precision))]
        : []
    })
    return values.length > 0 ? values.reduce((sum, value) => sum + value, 0) / values.length : null
  }
  const keys = dimension === "bfcl"
    ? ["non_live_accuracy", "simple_accuracy", "multiple_accuracy"]
    : dimension === "fulcrum"
      ? ["capability_rate", "perfect_rate"]
      : ["strict_prompt_accuracy", "strict_instruction_accuracy", "loose_prompt_accuracy"]
  const value = numericMetric(rows.flatMap((row) => row.metrics), keys)?.value
  return typeof value === "number" ? clamp(value) : null
}

function speedMetric(run: BenchmarkRun, profile: string): BenchmarkMetric | undefined {
  const result = run.result[profile]
  if (!result) return undefined
  const metrics = parseBenchmarkResults({ [profile]: result }).flatMap((row) => row.metrics)
  return numericMetric(metrics, ["latency_s_p50", "speed_tok_s", "quality_tok_s", "decode_tok_s_p50", "elapsed"])
}

function attemptReliability(attempt: BenchmarkAttempt | undefined): number {
  if (!attempt) return 1
  const count = attempt.environment.requests?.count ?? 0
  const errors = attempt.environment.requests?.errors ?? 0
  if (attempt.status !== "completed") return 0
  return count > 0 ? clamp(1 - errors / count) : 1
}

function observations(runs: BenchmarkRun[], filters: RankingFilters): Observation[] {
  const result: Observation[] = []
  runs.filter((run) => run.status === "completed" && Object.keys(run.result).length > 0).forEach((run) => {
    const dimension = dimensionFor(run)
    if (!dimension) return
    Object.keys(run.result).forEach((profile) => {
      const attempt = latestAttempt(run, profile)
      const version = attemptVersion(attempt)
      const hardware = attemptHardware(attempt)
      if (filters.ocabraVersion && version !== filters.ocabraVersion) return
      if (filters.hardware && hardware !== filters.hardware) return
      const quality = observationQuality(run, profile, dimension)
      if (quality === null) return
      const speed = speedMetric(run, profile)
      // El mismo modelo en otro nodo de la federación es otra fila: otro hardware.
      const backend = withNode(attemptBackend(attempt), run.node)
      const parsed = parseBenchmarkResults({ [profile]: run.result[profile] })
      const model = parsed[0]?.model || profile
      result.push({
        identity: `${profile}::${backend}`,
        model,
        profile,
        backend,
        dimension,
        run,
        attempt,
        quality,
        speedValue: typeof speed?.value === "number" && speed.value > 0 ? speed.value : null,
        speedDirection: speed?.better ?? null,
        reliability: attemptReliability(attempt),
        gpuCount: Math.max(1, attempt?.environment.hardware?.gpus?.length ?? 1),
        version,
        hardware,
        performance: 0,
        efficiency: 0,
      })
    })
  })

  const knownBackends = new Map<string, Set<string>>()
  result.forEach((item) => {
    if (item.backend === UNKNOWN_BACKEND) return
    const values = knownBackends.get(item.profile) ?? new Set<string>()
    values.add(item.backend)
    knownBackends.set(item.profile, values)
  })
  result.forEach((item) => {
    const candidates = knownBackends.get(item.profile)
    if (item.backend !== UNKNOWN_BACKEND || candidates?.size !== 1) return
    item.backend = [...candidates][0]
    item.identity = `${item.profile}::${item.backend}`
  })

  const byRun = new Map<string, Observation[]>()
  result.forEach((item) => byRun.set(item.run.id, [...(byRun.get(item.run.id) ?? []), item]))
  byRun.forEach((items) => {
    const comparable = items.filter((item) => item.speedValue !== null)
    comparable.forEach((item) => {
      const ratios = comparable.map((candidate) => {
        const value = candidate.speedValue as number
        return item.speedDirection === "lower" ? 1 / value : value
      })
      const own = item.speedDirection === "lower" ? 1 / (item.speedValue as number) : item.speedValue as number
      item.performance = clamp(own / Math.max(...ratios))
    })
    const maxEfficiency = Math.max(0, ...comparable.map((item) => item.performance / item.gpuCount))
    comparable.forEach((item) => {
      item.efficiency = maxEfficiency > 0 ? (item.performance / item.gpuCount) / maxEfficiency : 0
    })
  })
  return result
}

function preferredSuite(dimension: RankingDimension, suite: string): number {
  if (dimension === "bfcl") return suite === "core" ? 2 : 1
  if (dimension === "ifeval") return suite === "full" ? 2 : 1
  return 1
}

function latestByDimension(items: Observation[]): Observation[] {
  const selected = new Map<RankingDimension, Observation>()
  items.forEach((item) => {
    const current = selected.get(item.dimension)
    if (!current) {
      selected.set(item.dimension, item)
      return
    }
    const suiteDelta = preferredSuite(item.dimension, item.run.suite) - preferredSuite(current.dimension, current.run.suite)
    if (suiteDelta > 0 || (suiteDelta === 0 && item.run.createdAt > current.run.createdAt)) {
      selected.set(item.dimension, item)
    }
  })
  return [...selected.values()]
}

export function normalizeWeights(weights: RankingWeights): RankingWeights {
  const entries = Object.entries(weights) as Array<[RankingFactor, number]>
  const total = entries.reduce((sum, [, value]) => sum + Math.max(0, value), 0) || 1
  return Object.fromEntries(entries.map(([key, value]) => [key, Math.max(0, value) / total])) as RankingWeights
}

export function buildModelRanking(
  runs: BenchmarkRun[],
  weights: RankingWeights,
  filters: RankingFilters = { ocabraVersion: "", hardware: "" },
): ModelRankingRow[] {
  const normalized = normalizeWeights(weights)
  const grouped = new Map<string, Observation[]>()
  observations(runs, filters).forEach((item) => grouped.set(item.identity, [...(grouped.get(item.identity) ?? []), item]))

  const ranking = [...grouped.entries()].map(([id, allItems]) => {
    const items = latestByDimension(allItems)
    const dimensions = items.map((item) => item.dimension)
    const qualityWeight = QUALITY_DIMENSIONS.reduce((sum, dimension) => sum + normalized[dimension], 0)
    const qualityPoints = items.reduce((sum, item) => sum + item.quality * normalized[item.dimension], 0)
    const reliability = items.reduce((sum, item) => sum + item.reliability, 0) / items.length
    const measuredPerformance = items.filter((item) => item.speedValue !== null)
    const performance = measuredPerformance.length > 0
      ? measuredPerformance.reduce((sum, item) => sum + item.performance, 0) / measuredPerformance.length
      : 0
    const efficiency = measuredPerformance.length > 0
      ? measuredPerformance.reduce((sum, item) => sum + item.efficiency, 0) / measuredPerformance.length
      : 0
    const contributions: Record<RankingFactor, number> = {
      corrala: (items.find((item) => item.dimension === "corrala")?.quality ?? 0) * normalized.corrala,
      escriba: (items.find((item) => item.dimension === "escriba")?.quality ?? 0) * normalized.escriba,
      bfcl: (items.find((item) => item.dimension === "bfcl")?.quality ?? 0) * normalized.bfcl,
      ifeval: (items.find((item) => item.dimension === "ifeval")?.quality ?? 0) * normalized.ifeval,
      fulcrum: (items.find((item) => item.dimension === "fulcrum")?.quality ?? 0) * normalized.fulcrum,
      reliability: reliability * normalized.reliability,
      performance: performance * normalized.performance,
      efficiency: efficiency * normalized.efficiency,
    }
    const score = Object.values(contributions).reduce((sum, value) => sum + value, 0) * 100
    const first = items[0]
    return {
      id,
      model: first.model,
      profile: first.profile,
      backend: first.backend,
      score,
      quality: qualityWeight > 0 ? (qualityPoints / qualityWeight) * 100 : 0,
      reliability: reliability * 100,
      performance: performance * 100,
      efficiency: efficiency * 100,
      coverage: dimensions.length / QUALITY_DIMENSIONS.length,
      dimensions,
      eligible: dimensions.length >= 2,
      versions: Array.from(new Set(items.map((item) => item.version))),
      hardware: Array.from(new Set(items.map((item) => item.hardware))),
      sources: items.map((item) => ({
        dimension: item.dimension,
        runId: item.run.id,
        app: item.run.app,
        suite: item.run.suite,
        revision: item.run.revision,
        createdAt: item.run.createdAt,
        quality: item.quality * 100,
        performance: item.speedValue === null ? null : item.performance * 100,
      })),
      contributions,
    } satisfies ModelRankingRow
  })
  return ranking.sort((left, right) => Number(right.eligible) - Number(left.eligible) || right.score - left.score || left.model.localeCompare(right.model))
}

export function rankingFilterOptions(runs: BenchmarkRun[]): { versions: string[]; hardware: string[] } {
  const attempts = runs.flatMap((run) => run.attempts).filter((attempt) => attempt.status === "completed")
  return {
    versions: Array.from(new Set(attempts.map(attemptVersion).filter((value) => value !== "versión no informada"))).sort().reverse(),
    hardware: Array.from(new Set(attempts.map(attemptHardware).filter((value) => value !== "GPU no informada"))).sort(),
  }
}
