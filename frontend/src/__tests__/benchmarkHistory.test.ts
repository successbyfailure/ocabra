import { describe, expect, it } from "vitest"
import { historicalRows } from "@/pages/benchmarks/BenchmarkHistory"
import type { BenchmarkRun } from "@/types"

function run(id: string, model: string, score: number, baseline = false): BenchmarkRun {
  return {
    id, app: "ifeval", suite: "quick", revision: "abc", models: [model], suiteArgs: [],
    status: "completed", phase: "completed", scheduleId: null, isBaseline: baseline,
    autoPause: true, pauseReason: null, currentModelIndex: 1, currentModel: null,
    result: { [model]: { benchmark: "ifeval", model, status: "completed", cases: 50,
      metrics: { strict_prompt_accuracy: score, latency_s_p50: 2 } } },
    contaminatedAttempts: 0, error: null, externalActivityAt: null,
    createdAt: "2026-11-20T00:00:00Z", startedAt: null, finishedAt: null, attempts: [],
  }
}

describe("benchmark historical comparison", () => {
  it("keeps run identity while normalizing comparable metrics", () => {
    const rows = historicalRows([run("baseline", "qwen", 0.8, true), run("candidate", "qwen", 0.9)])

    expect(rows).toHaveLength(2)
    expect(rows.map((row) => row.run.id)).toEqual(["baseline", "candidate"])
    expect(rows[1].metrics.find((metric) => metric.key === "strict_prompt_accuracy")?.value).toBe(0.9)
    expect(rows[1].scope).toBe("ifeval · 50 casos")
  })
})
