import { describe, expect, it } from "vitest"
import {
  preferredMetric,
  rankedMetricRows,
} from "@/pages/benchmarks/BenchmarkResultsDashboard"
import type { BenchmarkResultRow } from "@/pages/benchmarks/resultAdapters"

function row(model: string, value: number, better: "higher" | "lower" = "higher"): BenchmarkResultRow {
  return {
    model,
    scope: "global",
    status: "completed",
    raw: {},
    metrics: [{ key: "score", label: "Score", value, format: "number", better }],
  }
}

describe("benchmark results dashboard", () => {
  it("ranks higher and lower metrics in the useful direction", () => {
    expect(rankedMetricRows([row("slow", 1), row("fast", 2)], "score")[0].row.model).toBe("fast")
    expect(rankedMetricRows([row("slow", 2, "lower"), row("fast", 1, "lower")], "score")[0].row.model).toBe("fast")
  })

  it("selects the benchmark's primary quality metric before secondary values", () => {
    expect(preferredMetric([
      { key: "simple_accuracy", label: "Simple", value: 0.8, format: "percent", better: "higher" },
      { key: "non_live_accuracy", label: "Non-live", value: 0.7, format: "percent", better: "higher" },
    ])).toBe("non_live_accuracy")
  })
})
