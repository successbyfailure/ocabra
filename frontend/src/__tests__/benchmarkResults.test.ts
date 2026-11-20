import { describe, expect, it } from "vitest"
import { benchmarkExport } from "@/pages/benchmarks/BenchmarkResults"
import { bestMetricValues, formatMetric, parseBenchmarkResults } from "@/pages/benchmarks/resultAdapters"
import type { BenchmarkRun } from "@/types"

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

  it("renders official BFCL and IFEval scores as comparable percentages", () => {
    const bfcl = parseBenchmarkResults({
      modelA: {
        benchmark: "bfcl_v4", model: "modelA", status: "completed",
        metrics: { simple_accuracy: 0.82, parallel_accuracy: 0.61 },
      },
    })[0]
    const ifeval = parseBenchmarkResults({
      modelB: {
        benchmark: "ifeval", model: "modelB", status: "completed", cases: 50,
        metrics: { strict_prompt_accuracy: 0.7, loose_prompt_accuracy: 0.76 },
      },
    })[0]

    expect(bfcl.scope).toBe("bfcl_v4")
    expect(formatMetric(bfcl.metrics[0])).toBe("82.0%")
    expect(ifeval.scope).toBe("ifeval · 50 casos")
    expect(formatMetric(ifeval.metrics[0])).toBe("70.0%")
  })

  it("exports results together with versions, backend and every observed GPU", () => {
    const run = {
      id: "run-1", app: "ifeval", suite: "quick", revision: "abc", models: ["qwen"],
      suiteArgs: [], status: "completed", phase: "completed", autoPause: true,
      pauseReason: null, currentModelIndex: 1, currentModel: null,
      result: { qwen: { score: 1 } }, contaminatedAttempts: 0, error: null,
      externalActivityAt: null, createdAt: "2026-01-01T00:00:00Z",
      startedAt: "2026-01-01T00:00:01Z", finishedAt: "2026-01-01T00:01:00Z",
      attempts: [{
        id: "attempt-1", model: "qwen", attemptNumber: 1, status: "completed",
        environment: {
          model: { profile: "qwen", backends: ["llamacpp_mks"] },
          software: {
            ocabra: "0.124.0",
            scorer: "4.0.0",
            backends: [{ type: "llamacpp_mks", sourceCommit: "96fff6c0123456789" }],
          },
          hardware: { gpus: [{ index: 0, name: "RTX 3090", totalVramMb: 24576 }] },
        },
        returnCode: 0, externalRequests: [], error: null,
        startedAt: "2026-01-01T00:00:01Z", finishedAt: "2026-01-01T00:01:00Z",
        result: { score: 1 },
      }],
    } satisfies BenchmarkRun

    const exported = benchmarkExport(run)

    expect(exported.schemaVersion).toBe(1)
    expect(exported.results).toEqual(run.result)
    expect(exported.attempts[0].environment).toMatchObject({
      software: {
        ocabra: "0.124.0",
        scorer: "4.0.0",
        backends: [{ type: "llamacpp_mks", sourceCommit: "96fff6c0123456789" }],
      },
      model: { backends: ["llamacpp_mks"] },
      hardware: { gpus: [{ index: 0, name: "RTX 3090" }] },
    })
  })
})
