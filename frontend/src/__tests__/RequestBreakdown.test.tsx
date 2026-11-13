import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import type { RecentRequest } from "@/types"
import { CacheCell, cacheInfo, prefillSpeedText, requestPhases, TimingBreakdown } from "@/components/dashboard/RequestBreakdown"

const req = (over: Partial<RecentRequest> = {}): RecentRequest =>
  ({
    id: "r", modelId: "gemma4:26b", backendType: "llamacpp_mks", requestKind: "chat", endpointPath: "/v1/chat/completions",
    statusCode: 200, startedAt: "2026-11-13T10:00:00Z", durationMs: 50646, inputTokens: 51107, outputTokens: 487, error: null,
    userId: null, username: null, groupId: null, groupName: null, apiKeyName: null, clientAddr: null, userAgent: null,
    generationMs: 4646, cachedTokens: 0,
    runtimeTelemetry: { queue_ms: 20780, load_ms: 0, prefill_ms: 24462, ttft_ms: 46031 },
    ...over,
  }) as RecentRequest

describe("request breakdown", () => {
  it("splits the duration into queue, prefill, generation and the unexplained rest", () => {
    const split = requestPhases(req())!
    expect(split.total).toBe(50646)
    expect(split.phases.map((p) => [p.key, p.ms])).toEqual([
      ["queue", 20780],
      ["prefill", 24462],
      ["generation", 4646],
    ])
  })

  it("time the engine doesn't account for (slow client, streaming) shows up as rest", () => {
    const split = requestPhases(req({ durationMs: 51009, generationMs: 91, runtimeTelemetry: { prefill_ms: 176, queue_ms: 0 } }))!
    expect(split.phases[split.phases.length - 1]).toMatchObject({ key: "other", ms: 51009 - 176 - 91 })
  })

  it("no breakdown when the backend reports neither prefill nor generation", () => {
    expect(requestPhases(req({ generationMs: null, runtimeTelemetry: { queue_ms: 3 } }))).toBeNull()
  })

  it("renders prefill and generation under the total", () => {
    render(<TimingBreakdown r={req({ runtimeTelemetry: { prefill_ms: 204, queue_ms: 0 }, generationMs: 7601, durationMs: 8143 })} />)
    const el = screen.getByTestId("timing-breakdown")
    expect(el.textContent).toContain("8.1 s")
    expect(el.textContent).toContain("prefill 204 ms")
    expect(el.textContent).toContain("gen 7.6 s")
  })

  it("cache: share of the input served from the prompt cache", () => {
    expect(cacheInfo(req({ inputTokens: 64445, cachedTokens: 64242 }))).toEqual({ cached: 64242, input: 64445, pct: 100 })
    expect(cacheInfo(req({ cachedTokens: null }))).toBeNull()
    render(<CacheCell r={req({ inputTokens: 71256, cachedTokens: 35000 })} />)
    expect(screen.getByTestId("cache-cell").textContent).toContain("49%")
  })
  it("prefill speed with almost everything cached says how many tokens were new", () => {
    const fmt = (t: number | null | undefined) => `${t} t/s`
    expect(prefillSpeedText(req({ inputTokens: 3003, cachedTokens: 3002, promptTokensPerSecond: 2.1 }), fmt)).toBe("1 token nuevo")
    expect(prefillSpeedText(req({ inputTokens: 3003, cachedTokens: 3003, promptTokensPerSecond: 0 }), fmt)).toBe("todo en cache")
    expect(prefillSpeedText(req({ inputTokens: 51107, cachedTokens: 0, promptTokensPerSecond: 2089 }), fmt)).toBe("2089 t/s")
  })

  // 2026-11-13, ornith con 141k tokens de cache: 1,1 s antes del prefill
  // (tokenizar el prompt entero) salía como «resto» y el prefill de 332 tokens
  // nuevos a 793 t/s parecía lento.
  const ornith = req({
    modelId: "ornith-1.5:35b", durationMs: 8070, inputTokens: 141422, cachedTokens: 141090,
    promptTokensPerSecond: 793.4, generationMs: 6520,
    runtimeTelemetry: { queue_ms: 0, load_ms: 0, prefill_ms: 418, ttft_ms: 1545 },
  })

  it("the time before prefill is preparation, not rest", () => {
    const split = requestPhases(ornith)!
    expect(split.phases.map((p) => [p.key, Math.round(p.ms)])).toEqual([
      ["prep", 1127],
      ["prefill", 418],
      ["generation", 6520],
    ])
  })

  it("prefill speed on a cached request says over how many new tokens", () => {
    const fmt = (t: number | null | undefined) => `${Math.round(t ?? 0)} t/s`
    expect(prefillSpeedText(ornith, fmt)).toBe("793 t/s · 332 nuevos")
  })

  it("names the preparation under the bar when it weighs", () => {
    render(<TimingBreakdown r={ornith} />)
    expect(screen.getByTestId("timing-breakdown").textContent).toContain("preparación 1.1 s")
  })
})
