import { describe, expect, it } from "vitest"
import { bestMetricValues, formatMetric, parseBenchmarkResults } from "@/pages/benchmarks/resultAdapters"

describe("benchmark result adapters", () => {
  it("shows the quality and speed persisted by Escriba", () => {
    const rows = parseBenchmarkResults({
      "gemma4:12b": {
        "gemma4:12b": {
          status: "OK",
          transcript_size: "M",
          speed: { tok_s: 96.3, elapsed: 12.5, out_tokens: 1202 },
          quality: {
            default: {
              coverage: 0.51,
              precision: 0.635,
              claims: 27,
              valid_json: true,
              tok_s: 121,
            },
          },
        },
      },
    })

    expect(rows).toHaveLength(1)
    expect(rows[0]).toMatchObject({ model: "gemma4:12b", scope: "Transcripción M · default", status: "OK" })
    expect(Object.fromEntries(rows[0].metrics.map((metric) => [metric.key, metric.value]))).toMatchObject({
      coverage: 0.51,
      precision: 0.635,
      claims: 27,
      valid_json: true,
      speed_tok_s: 96.3,
      elapsed: 12.5,
    })
    expect(formatMetric(rows[0].metrics.find((metric) => metric.key === "coverage")!)).toBe("51.0%")
  })

  it("keeps supporting Corrala's normalized metrics", () => {
    const rows = parseBenchmarkResults({
      "bonsai-2:27b": {
        metrics: [{
          model: "bonsai-2:27b",
          suite: "mail-routing",
          status: "completed",
          quality: { pass_rate: 0.9, primary_rate: 0.8, errors: 1 },
          speed: { latency_s_p50: 1.25, decode_tok_s_p50: 48.2 },
        }],
      },
    })

    expect(rows).toHaveLength(1)
    expect(rows[0].scope).toBe("mail-routing")
    expect(Object.fromEntries(rows[0].metrics.map((metric) => [metric.key, metric.value]))).toMatchObject({
      pass_rate: 0.9,
      primary_rate: 0.8,
      latency_s_p50: 1.25,
      decode_tok_s_p50: 48.2,
      errors: 1,
    })
  })

  it("highlights the best comparable metric and falls back for unknown runners", () => {
    const rows = parseBenchmarkResults({
      fast: { nested: { score: 0.75 }, status: "ok" },
      slow: { nested: { score: 0.5 }, status: "ok" },
    })
    expect(rows[0].metrics.find((metric) => metric.key === "nested.score")?.value).toBe(0.75)

    const comparable = parseBenchmarkResults({
      a: { a: { speed: { tok_s: 10 }, quality: { default: { coverage: 0.8 } } } },
      b: { b: { speed: { tok_s: 20 }, quality: { default: { coverage: 0.7 } } } },
    })
    expect(bestMetricValues(comparable).get("speed_tok_s")).toBe(20)
    expect(bestMetricValues(comparable).get("coverage")).toBe(0.8)
  })
})
