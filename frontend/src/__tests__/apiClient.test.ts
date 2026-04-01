import { afterEach, describe, expect, it, vi } from "vitest"

import { api } from "@/api/client"

describe("api client mappings", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
  })

  it("maps acetep capabilities and music generation fields", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify([
          {
            model_id: "acestep/turbo",
            display_name: "ACE-Step Turbo",
            backend_type: "acestep",
            status: "configured",
            load_policy: "on_demand",
            auto_reload: false,
            preferred_gpu: 1,
            current_gpu: [],
            vram_used_mb: 0,
            capabilities: {
              music_generation: true,
              streaming: true,
              context_length: 0,
            },
          },
        ]),
        {
          status: 200,
          headers: { "Content-Type": "application/json" },
        },
      ),
    )

    const models = await api.models.list()

    expect(models[0]?.backendType).toBe("acestep")
    expect(models[0]?.capabilities.musicGeneration).toBe(true)
    expect(models[0]?.capabilities.streaming).toBe(true)
  })

  it("maps idle eviction interval from server config payloads", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({
          defaultGpuIndex: 1,
          idleTimeoutSeconds: 300,
          idleEvictionCheckIntervalSeconds: 42,
          vramBufferMb: 512,
          vramPressureThresholdPct: 85,
          logLevel: "info",
          litellmBaseUrl: "",
          litellmAdminKey: "",
          litellmAutoSync: false,
          energyCostEurKwh: 0.31,
          modelsDir: "/srv/models",
          downloadDir: "/srv/models/downloads",
          maxTemperatureC: 88,
          globalSchedules: [],
        }),
        {
          status: 200,
          headers: { "Content-Type": "application/json" },
        },
      ),
    )

    const config = await api.config.get()

    expect(config.idleEvictionCheckIntervalSeconds).toBe(42)
    expect(config.modelsDir).toBe("/srv/models")
  })

  it("encodes download job ids in SSE stream URLs", () => {
    const constructedUrls: string[] = []
    class MockEventSource {
      url: string
      close = vi.fn()

      constructor(url: string) {
        this.url = url
        constructedUrls.push(url)
      }
    }

    vi.stubGlobal("EventSource", MockEventSource as unknown as typeof EventSource)

    api.downloads.streamProgress("job id/with spaces")

    expect(constructedUrls[0]).toBe("/ocabra/downloads/job%20id%2Fwith%20spaces/stream")
  })

  it("maps snake_case estimate payloads to the camelCase shape used by the UI", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({
          estimated_params_b: 32,
          quant: "awq",
          tp_size: 1,
          config_found: true,
          serve: {
            vram_per_gpu_mb: 10240,
            vram_total_mb: 10240,
            breakdown: {
              weights_mb: 8192,
              kv_cache_mb: 1024,
              overhead_mb: 1024,
            },
          },
          build: {
            vram_per_gpu_mb: 12288,
            vram_total_mb: 12288,
          },
          disk: {
            engine_mb: 9000,
            checkpoint_mb_temp: 8500,
            total_peak_mb: 17500,
          },
          warnings: ["Build needs ~12.0GB/GPU but selected GPU 1 has only 10.0GB free"],
        }),
        {
          status: 200,
          headers: { "Content-Type": "application/json" },
        },
      ),
    )

    const estimate = await api.trtllm.estimate({
      modelId: "vllm/Qwen/Qwen3-32B-AWQ",
      gpuIndices: [1],
      tpSize: 1,
      dtype: "fp16",
      maxBatchSize: 1,
      maxSeqLen: 4096,
    })

    expect(estimate.serve.vramPerGpuMb).toBe(10240)
    expect(estimate.serve.breakdown.weightsMb).toBe(8192)
    expect(estimate.build.vramTotalMb).toBe(12288)
    expect(estimate.disk.totalPeakMb).toBe(17500)
    expect(estimate.warnings[0]).toContain("selected GPU 1")
    expect(String(fetchMock.mock.calls[0]?.[0])).toContain("gpu_indices=1")
  })
})
