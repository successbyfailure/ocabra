import { describe, expect, it } from "vitest"
import {
  estimateBenchmarkRemaining,
  estimateBenchmarkTime,
  formatEstimatedDuration,
} from "@/pages/benchmarks/benchmarkEstimate"
import type { BenchmarkRun } from "@/types"

function completedRun(options: {
  app: string
  suite: string
  suiteArgs?: string[]
  model?: string
  attemptSeconds: number
  requestCount?: number
}): BenchmarkRun {
  const model = options.model ?? "model-a"
  const started = new Date("2026-11-23T00:01:00Z")
  const finished = new Date(started.getTime() + options.attemptSeconds * 1_000)
  return {
    id: `${options.app}-${options.suite}-${model}`,
    app: options.app,
    suite: options.suite,
    revision: "revision",
    models: [model],
    suiteArgs: options.suiteArgs ?? [],
    status: "completed",
    phase: "completed",
    scheduleId: null,
    isBaseline: false,
    autoPause: true,
    pauseReason: null,
    currentModelIndex: 1,
    currentModel: null,
    testProgress: null,
    result: { [model]: { status: "completed" } },
    contaminatedAttempts: 0,
    error: null,
    externalActivityAt: null,
    createdAt: "2026-11-23T00:00:00Z",
    startedAt: "2026-11-23T00:00:00Z",
    finishedAt: finished.toISOString(),
    attempts: [{
      id: `attempt-${model}`,
      model,
      attemptNumber: 1,
      status: "completed",
      environment: { requests: { count: options.requestCount ?? 10, errors: 0 } },
      returnCode: 0,
      externalRequests: [],
      error: null,
      startedAt: started.toISOString(),
      finishedAt: finished.toISOString(),
      result: {},
    }],
  }
}

describe("benchmark time estimate", () => {
  it("uses the accepted duration of the same model and configuration", () => {
    const estimate = estimateBenchmarkTime([
      completedRun({ app: "bfcl", suite: "core", attemptSeconds: 600 }),
    ], "bfcl", "core", [], ["model-a"])

    expect(estimate?.models[0]).toMatchObject({
      seconds: 600,
      source: "model_history",
      confidence: "medium",
      samples: 1,
    })
    expect(estimate?.setupSeconds).toBe(60)
    expect(estimate?.seconds).toBe(660)
  })

  it("scales IFEval quick samples to the 541-case full suite", () => {
    const estimate = estimateBenchmarkTime([
      completedRun({ app: "ifeval", suite: "quick", attemptSeconds: 100 }),
    ], "ifeval", "full", [], ["model-a"])

    expect(estimate?.models[0].source).toBe("scaled_model_history")
    expect(estimate?.models[0].seconds).toBeCloseTo(1_082)
  })

  it("scales Corrala profiles by their actual number of cases", () => {
    const estimate = estimateBenchmarkTime([
      completedRun({
        app: "corrala", suite: "llm", suiteArgs: ["--profile", "standard"], attemptSeconds: 540,
      }),
    ], "corrala", "llm", ["--profile", "final"], ["model-a"])

    expect(estimate?.models[0].source).toBe("scaled_model_history")
    expect(estimate?.models[0].seconds).toBeCloseTo(2_490)
  })

  it("ignores completed attempts that made no inference requests", () => {
    const estimate = estimateBenchmarkTime([
      completedRun({ app: "ifeval", suite: "quick", attemptSeconds: 3, requestCount: 0 }),
    ], "ifeval", "quick", [], ["model-a"])

    expect(estimate?.models[0]).toMatchObject({ source: "fallback", seconds: 900, samples: 0 })
    expect(estimate?.confidence).toBe("low")
  })

  it("learns from resumed models in an active run without counting discarded chains", () => {
    const run = completedRun({ app: "ifeval", suite: "quick", attemptSeconds: 400 })
    const finalAttempt = run.attempts[0]
    run.status = "running"
    run.currentModelIndex = 1
    run.currentModel = "model-b"
    run.finishedAt = null
    run.models = ["model-a", "model-b"]
    run.attempts = [
      { ...finalAttempt, id: "discarded-1", attemptNumber: 1, status: "interrupted", startedAt: "2026-11-23T00:00:00Z", finishedAt: "2026-11-23T00:03:20Z" },
      { ...finalAttempt, id: "discarded-2", attemptNumber: 2, status: "contaminated", startedAt: "2026-11-23T00:03:20Z", finishedAt: "2026-11-23T00:08:20Z" },
      { ...finalAttempt, id: "resumed-1", attemptNumber: 3, status: "interrupted", startedAt: "2026-11-23T00:08:20Z", finishedAt: "2026-11-23T00:11:40Z" },
      { ...finalAttempt, id: "accepted", attemptNumber: 4, status: "completed", startedAt: "2026-11-23T00:11:40Z", finishedAt: "2026-11-23T00:18:20Z" },
    ]

    const estimate = estimateBenchmarkTime([run], "ifeval", "quick", [], ["model-a"])

    expect(estimate?.models[0]).toMatchObject({ source: "model_history", seconds: 600 })
  })

  it("estimates the current test and complete queue from live case progress", () => {
    const history = completedRun({ app: "ifeval", suite: "quick", model: "model-a", attemptSeconds: 100 })
    const active = completedRun({ app: "ifeval", suite: "quick", model: "model-a", attemptSeconds: 100 })
    active.id = "active"
    active.status = "running"
    active.phase = "running"
    active.models = ["model-a", "model-b"]
    active.currentModelIndex = 0
    active.currentModel = "model-a"
    active.finishedAt = null
    active.testProgress = { completed: 5, total: 10, unit: "casos", stage: "generating", updatedAt: "2026-11-23T00:00:50Z" }
    active.attempts = [{
      ...active.attempts[0], status: "running", startedAt: "2026-11-23T00:00:00Z", finishedAt: null,
    }]

    const remaining = estimateBenchmarkRemaining(
      [history, active], active, new Date("2026-11-23T00:00:50Z"),
    )

    expect(remaining?.currentSeconds).toBeCloseTo(50)
    expect(remaining?.totalSeconds).toBeCloseTo(150)
  })

  it("does not count time spent paused as useful benchmark time", () => {
    const history = completedRun({ app: "ifeval", suite: "quick", attemptSeconds: 100 })
    const paused = completedRun({ app: "ifeval", suite: "quick", attemptSeconds: 100 })
    paused.status = "paused"
    paused.phase = "paused"
    paused.currentModelIndex = 0
    paused.currentModel = "model-a"
    paused.finishedAt = null
    paused.testProgress = { completed: 5, total: 10, unit: "casos", stage: "generating", updatedAt: "2026-11-23T00:00:50Z" }
    paused.attempts = [{
      ...paused.attempts[0], status: "interrupted", startedAt: "2026-11-23T00:00:00Z", finishedAt: "2026-11-23T00:00:50Z",
    }]

    const remaining = estimateBenchmarkRemaining(
      [history, paused], paused, new Date("2026-11-23T01:00:00Z"),
    )

    expect(remaining?.currentSeconds).toBeCloseTo(50)
  })

  it("formats useful human durations", () => {
    expect(formatEstimatedDuration(90)).toBe("2 min")
    expect(formatEstimatedDuration(3_600)).toBe("1 h")
    expect(formatEstimatedDuration(5_400)).toBe("1 h 30 min")
  })
})
