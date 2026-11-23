import { describe, expect, it } from "vitest"
import { paretoIds, scatterPointRows } from "@/pages/benchmarks/BenchmarkModelRanking"
import {
  buildModelRanking,
  RANKING_PROFILES,
  type ModelRankingRow,
} from "@/pages/benchmarks/modelRanking"
import type { BenchmarkRun } from "@/types"

function run(
  app: string,
  suite: string,
  profile: string,
  quality: number,
  options: { backend?: string; version?: string; gpu?: string; speed?: number; errors?: number } = {},
): BenchmarkRun {
  const metrics = app === "corrala"
    ? { metrics: [{ model: profile, suite: "routing", runs: 10, status: "completed", quality: { pass_rate: quality, primary_rate: quality, errors: options.errors ?? 0 }, speed: { latency_s_p50: options.speed ?? 2 } }] }
    : app === "escriba"
      ? { [profile]: { status: "OK", transcript_size: "M", speed: { tok_s: options.speed ?? 50 }, quality: { default: { coverage: quality, precision: quality } } } }
      : app === "bfcl"
        ? { benchmark: "bfcl_v4", model: profile, status: "completed", metrics: { non_live_accuracy: quality, latency_s_p50: options.speed ?? 2 } }
        : { benchmark: "ifeval", model: profile, status: "completed", metrics: { strict_prompt_accuracy: quality, latency_s_p50: options.speed ?? 2 } }
  return {
    id: `${app}-${suite}-${profile}-${options.backend ?? "llama_cpp"}`,
    app,
    suite,
    revision: "revision-1",
    models: [profile],
    suiteArgs: [],
    status: "completed",
    phase: "completed",
    scheduleId: null,
    isBaseline: false,
    autoPause: true,
    pauseReason: null,
    currentModelIndex: 1,
    currentModel: null,
    testProgress: null,
    result: { [profile]: metrics },
    contaminatedAttempts: 0,
    error: null,
    externalActivityAt: null,
    createdAt: "2026-11-23T12:00:00Z",
    startedAt: "2026-11-23T12:00:01Z",
    finishedAt: "2026-11-23T12:01:00Z",
    attempts: [{
      id: `attempt-${app}-${profile}`,
      model: profile,
      attemptNumber: 1,
      status: "completed",
      environment: {
        software: { ocabra: options.version ?? "0.128.0" },
        model: { profile, backends: [options.backend ?? "llama_cpp"] },
        hardware: { gpus: [{ index: 0, name: options.gpu ?? "RTX 3090" }] },
        requests: { count: 10, errors: options.errors ?? 0 },
      },
      returnCode: 0,
      externalRequests: [],
      error: null,
      startedAt: "2026-11-23T12:00:01Z",
      finishedAt: "2026-11-23T12:01:00Z",
      result: metrics,
    }],
  }
}

const balanced = RANKING_PROFILES[0].weights

describe("global benchmark ranking", () => {
  it("combines the four quality dimensions and keeps model plus backend as identity", () => {
    const runs = [
      run("corrala", "llm", "qwen", 0.8),
      run("escriba", "llm", "qwen", 0.8),
      run("bfcl", "core", "qwen", 0.8),
      run("ifeval", "full", "qwen", 0.8),
      run("bfcl", "core", "qwen-ollama", 0.9, { backend: "ollama" }),
      run("ifeval", "full", "qwen-ollama", 0.9, { backend: "ollama" }),
    ]

    const ranking = buildModelRanking(runs, balanced)

    expect(ranking).toHaveLength(2)
    expect(ranking[0]).toMatchObject({ model: "qwen", backend: "llama_cpp", coverage: 1, eligible: true })
    expect(ranking[0].score).toBeCloseTo(85)
    expect(ranking[1]).toMatchObject({ model: "qwen-ollama", backend: "ollama", coverage: 0.5, eligible: true })
    expect(ranking[1].score).toBeLessThan(ranking[0].score)
  })

  it("does not let a model with only one public benchmark enter the ranking", () => {
    const ranking = buildModelRanking([
      run("bfcl", "core", "partial", 1),
      run("corrala", "llm", "covered", 0.6),
      run("ifeval", "full", "covered", 0.6),
    ], balanced)

    expect(ranking.find((row) => row.model === "partial")).toMatchObject({ eligible: false, coverage: 0.25 })
    expect(ranking[0].model).toBe("covered")
  })

  it("filters cohorts by the recorded oCabra version and GPU", () => {
    const runs = [
      run("bfcl", "core", "old", 0.7, { version: "0.127.0", gpu: "RTX 3060" }),
      run("ifeval", "full", "old", 0.7, { version: "0.127.0", gpu: "RTX 3060" }),
      run("bfcl", "core", "new", 0.8, { version: "0.128.0", gpu: "RTX 3090" }),
      run("ifeval", "full", "new", 0.8, { version: "0.128.0", gpu: "RTX 3090" }),
    ]

    const ranking = buildModelRanking(runs, balanced, { ocabraVersion: "0.128.0", hardware: "RTX 3090" })

    expect(ranking.map((row) => row.model)).toEqual(["new"])
  })

  it("identifies the quality-performance Pareto frontier", () => {
    const row = (id: string, quality: number, performance: number): ModelRankingRow => ({
      id, model: id, profile: id, backend: "test", score: 0, quality, reliability: 100,
      performance, efficiency: 100, coverage: 1, dimensions: ["bfcl", "ifeval"], eligible: true,
      versions: [], hardware: [], sources: [],
      contributions: { corrala: 0, escriba: 0, bfcl: 0, ifeval: 0, reliability: 0, performance: 0, efficiency: 0 },
    })
    expect([...paretoIds([row("balanced", 80, 80), row("slow", 90, 60), row("dominated", 70, 70)])].sort()).toEqual(["balanced", "slow"])
  })

  it("labels every scatter point and disambiguates duplicate models by backend", () => {
    const rows = buildModelRanking([
      run("bfcl", "core", "same-model", 0.8, { backend: "llama_cpp" }),
      run("ifeval", "full", "same-model", 0.8, { backend: "llama_cpp" }),
      run("bfcl", "core", "same-model", 0.9, { backend: "ollama" }),
      run("ifeval", "full", "same-model", 0.9, { backend: "ollama" }),
    ], balanced)

    expect(scatterPointRows(rows).map((row) => row.pointLabel).sort()).toEqual([
      "same-model · llama_cpp",
      "same-model · ollama",
    ])
  })
})
