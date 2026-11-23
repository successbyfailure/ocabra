import type { BenchmarkRun } from "@/types"

export type EstimateConfidence = "high" | "medium" | "low"
export type EstimateSource = "model_history" | "scaled_model_history" | "suite_history" | "scaled_suite_history" | "fallback"

export interface ModelTimeEstimate {
  model: string
  seconds: number
  lowerSeconds: number
  upperSeconds: number
  confidence: EstimateConfidence
  source: EstimateSource
  samples: number
}

export interface BenchmarkTimeEstimate {
  seconds: number
  lowerSeconds: number
  upperSeconds: number
  setupSeconds: number
  confidence: EstimateConfidence
  historicalModels: number
  models: ModelTimeEstimate[]
}

const FALLBACK_SECONDS: Record<string, number> = {
  "corrala/llm": 8 * 60,
  "bfcl/simple_python": 20 * 60,
  "bfcl/core": 90 * 60,
  "ifeval/quick": 10 * 60,
  "ifeval/full": 100 * 60,
  "escriba/llm": 3 * 60,
  "escriba/stt": 12 * 60,
  "escriba/stt_diarization": 40 * 60,
  "escriba/embeddings": 10 * 60,
}

const CORRALA_CASES: Record<string, number> = {
  quick: 34,
  standard: 54,
  full: 83,
  final: 249,
}

const IFEVAL_CASES: Record<string, number> = { quick: 50, full: 541 }

function median(values: number[]): number {
  if (values.length === 0) return 0
  const sorted = [...values].sort((left, right) => left - right)
  const middle = Math.floor(sorted.length / 2)
  return sorted.length % 2 === 0 ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle]
}

function durationSeconds(startedAt: string, finishedAt: string | null): number | null {
  if (!finishedAt) return null
  const seconds = (new Date(finishedAt).getTime() - new Date(startedAt).getTime()) / 1_000
  return Number.isFinite(seconds) && seconds > 0 ? seconds : null
}

function argsKey(args: string[]): string {
  return JSON.stringify(args)
}

function corralaProfile(args: string[]): string {
  const index = args.indexOf("--profile")
  return index >= 0 ? args[index + 1] ?? "standard" : "standard"
}

function workloadScale(
  sourceApp: string,
  sourceSuite: string,
  sourceArgs: string[],
  targetApp: string,
  targetSuite: string,
  targetArgs: string[],
): number | null {
  if (sourceApp !== targetApp) return null
  if (sourceApp === "corrala" && sourceSuite === "llm" && targetSuite === "llm") {
    const sourceCases = CORRALA_CASES[corralaProfile(sourceArgs)]
    const targetCases = CORRALA_CASES[corralaProfile(targetArgs)]
    return sourceCases && targetCases ? targetCases / sourceCases : null
  }
  if (sourceApp === "ifeval") {
    const sourceCases = IFEVAL_CASES[sourceSuite]
    const targetCases = IFEVAL_CASES[targetSuite]
    return sourceCases && targetCases ? targetCases / sourceCases : null
  }
  return null
}

function usableAttempt(run: BenchmarkRun, model: string): number[] {
  return run.attempts.flatMap((attempt) => {
    if (attempt.model !== model || attempt.status !== "completed") return []
    if (attempt.environment.requests?.count === 0) return []
    const seconds = durationSeconds(attempt.startedAt, attempt.finishedAt)
    return seconds === null ? [] : [seconds]
  })
}

function ranges(seconds: number, confidence: EstimateConfidence): [number, number] {
  if (confidence === "high") return [seconds * 0.85, seconds * 1.2]
  if (confidence === "medium") return [seconds * 0.7, seconds * 1.4]
  return [seconds * 0.5, seconds * 1.8]
}

function modelEstimate(
  runs: BenchmarkRun[],
  app: string,
  suite: string,
  suiteArgs: string[],
  model: string,
): ModelTimeEstimate {
  const completed = runs.filter((run) => run.status === "completed")
  const exactRuns = completed.filter((run) => (
    run.app === app && run.suite === suite && argsKey(run.suiteArgs) === argsKey(suiteArgs)
  ))
  const direct = exactRuns.flatMap((run) => usableAttempt(run, model))
  let seconds: number
  let confidence: EstimateConfidence
  let source: EstimateSource
  let samples: number

  if (direct.length > 0) {
    seconds = median(direct)
    confidence = direct.length >= 2 ? "high" : "medium"
    source = "model_history"
    samples = direct.length
  } else {
    const scaledModel = completed.flatMap((run) => {
      const scale = workloadScale(run.app, run.suite, run.suiteArgs, app, suite, suiteArgs)
      return scale === null ? [] : usableAttempt(run, model).map((value) => value * scale)
    })
    if (scaledModel.length > 0) {
      seconds = median(scaledModel)
      confidence = "medium"
      source = "scaled_model_history"
      samples = scaledModel.length
    } else {
      const suiteSamples = exactRuns.flatMap((run) => run.models.flatMap((candidate) => usableAttempt(run, candidate)))
      if (suiteSamples.length > 0) {
        seconds = median(suiteSamples)
        confidence = suiteSamples.length >= 3 ? "medium" : "low"
        source = "suite_history"
        samples = suiteSamples.length
      } else {
        const scaledSuite = completed.flatMap((run) => {
          const scale = workloadScale(run.app, run.suite, run.suiteArgs, app, suite, suiteArgs)
          if (scale === null) return []
          return run.models.flatMap((candidate) => usableAttempt(run, candidate).map((value) => value * scale))
        })
        if (scaledSuite.length > 0) {
          seconds = median(scaledSuite)
          confidence = "low"
          source = "scaled_suite_history"
          samples = scaledSuite.length
        } else {
          seconds = FALLBACK_SECONDS[`${app}/${suite}`] ?? 15 * 60
          if (app === "corrala" && suite === "llm") {
            seconds *= (CORRALA_CASES[corralaProfile(suiteArgs)] ?? 54) / 54
          }
          confidence = "low"
          source = "fallback"
          samples = 0
        }
      }
    }
  }
  const [lowerSeconds, upperSeconds] = ranges(seconds, confidence)
  return { model, seconds, lowerSeconds, upperSeconds, confidence, source, samples }
}

function setupEstimate(runs: BenchmarkRun[], app: string, suite: string, suiteArgs: string[]): number {
  const overheads = runs.flatMap((run) => {
    if (
      run.status !== "completed"
      || run.app !== app
      || run.suite !== suite
      || argsKey(run.suiteArgs) !== argsKey(suiteArgs)
      || !run.startedAt
      || !run.finishedAt
      || run.contaminatedAttempts > 0
    ) return []
    const total = durationSeconds(run.startedAt, run.finishedAt)
    if (total === null) return []
    const attemptSeconds = run.attempts.reduce((sum, attempt) => (
      sum + (durationSeconds(attempt.startedAt, attempt.finishedAt) ?? 0)
    ), 0)
    const overhead = total - attemptSeconds
    return overhead >= 0 && overhead <= 2 * 60 * 60 ? [overhead] : []
  })
  return overheads.length > 0 ? median(overheads) : 60
}

export function estimateBenchmarkTime(
  runs: BenchmarkRun[],
  app: string,
  suite: string,
  suiteArgs: string[],
  models: string[],
): BenchmarkTimeEstimate | null {
  if (!app || !suite || models.length === 0) return null
  const estimates = models.map((model) => modelEstimate(runs, app, suite, suiteArgs, model))
  const setupSeconds = setupEstimate(runs, app, suite, suiteArgs)
  const historicalModels = estimates.filter((item) => item.source !== "fallback").length
  const directModels = estimates.filter((item) => item.source === "model_history").length
  const confidence: EstimateConfidence = directModels === models.length && estimates.every((item) => item.confidence === "high")
    ? "high"
    : historicalModels >= Math.ceil(models.length / 2) ? "medium" : "low"
  return {
    seconds: setupSeconds + estimates.reduce((sum, item) => sum + item.seconds, 0),
    lowerSeconds: setupSeconds + estimates.reduce((sum, item) => sum + item.lowerSeconds, 0),
    upperSeconds: setupSeconds + estimates.reduce((sum, item) => sum + item.upperSeconds, 0),
    setupSeconds,
    confidence,
    historicalModels,
    models: estimates,
  }
}

export function formatEstimatedDuration(seconds: number): string {
  const minutes = Math.max(1, Math.round(seconds / 60))
  if (minutes < 60) return `${minutes} min`
  const hours = Math.floor(minutes / 60)
  const remainder = minutes % 60
  return remainder === 0 ? `${hours} h` : `${hours} h ${remainder} min`
}
