import { afterEach, describe, expect, it, vi } from "vitest"

import { api } from "@/api/client"
import { useAuthStore } from "@/stores/authStore"

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
            pin_permanent: false,
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

  it("serializes token stats all-time flags", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({
          totalInputTokens: 0,
          totalOutputTokens: 0,
          byBackend: [],
          byGpu: [],
          series: [],
        }),
        {
          status: 200,
          headers: { "Content-Type": "application/json" },
        },
      ),
    )

    await api.stats.tokens({ allTime: true, includeSeries: false })

    expect(String(fetchMock.mock.calls[0]?.[0])).toBe(
      "/ocabra/stats/tokens?allTime=true&includeSeries=false",
    )
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

  it("maps runtime presets and their MAX context mode", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify([
          {
            id: "preset-1",
            profile_id: "gemma4:12b",
            preset_key: "max",
            label: "Máximo efectivo",
            context_mode: "max",
            context_tokens: null,
            parallelism: 2,
            kv_cache_dtype: "q8",
            placement_policy: "forced",
            preferred_gpus: [1],
            execution_policy: "auto",
            load_policy: "warm",
            enabled: true,
            is_default: false,
            priority: 40,
          },
        ]),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    )

    const presets = await api.profiles.listRuntimePresets("gemma4:12b")

    expect(presets[0]).toMatchObject({
      presetKey: "max",
      contextMode: "max",
      contextTokens: null,
      parallelism: 2,
      preferredGpus: [1],
    })
  })

  it("serializes a fixed runtime preset using backend field names", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({
          id: "preset-2",
          profile_id: "gemma4:12b",
          preset_key: "64k",
          label: "Normal",
          context_mode: "fixed",
          context_tokens: 65536,
          parallelism: 2,
          kv_cache_dtype: "q8",
          placement_policy: "auto",
          preferred_gpus: null,
          execution_policy: "auto",
          enabled: true,
          is_default: true,
          priority: 20,
        }),
        { status: 201, headers: { "Content-Type": "application/json" } },
      ),
    )

    await api.profiles.createRuntimePreset("gemma4:12b", {
      presetKey: "64k",
      label: "Normal",
      contextMode: "fixed",
      contextTokens: 65536,
      parallelism: 2,
      kvCacheDtype: "q8",
      placementPolicy: "auto",
      executionPolicy: "auto",
      isDefault: true,
      priority: 20,
    })

    const body = JSON.parse(String(fetchMock.mock.calls[0]?.[1]?.body))
    expect(body).toMatchObject({
      preset_key: "64k",
      context_mode: "fixed",
      context_tokens: 65536,
      parallelism: 2,
      is_default: true,
    })
  })

  it("maps resolved runtime capacity snapshots", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify([
          {
            preset_key: "max",
            resolved: true,
            requested_context: null,
            effective_context: 262144,
            model_max: 262144,
            hardware_max: 262144,
            target_gpus: [1],
            fits: true,
            runtime_key: "runtime:abc",
            warnings: [],
          },
        ]),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    )

    const rows = await api.profiles.resolveRuntimePresets("gemma4:12b")

    expect(rows[0]).toMatchObject({
      presetKey: "max",
      effectiveContext: 262144,
      targetGpus: [1],
      runtimeKey: "runtime:abc",
    })
  })

  it("clears the auth user on 401 so ProtectedRoute kicks the user back to /login", async () => {
    useAuthStore.getState().setUser({
      id: "u1",
      username: "u",
      email: null,
      role: "system_admin",
      createdAt: "",
    } as never)

    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ detail: "Session expired" }), {
        status: 401,
        headers: { "Content-Type": "application/json" },
      }),
    )

    await expect(api.models.list()).rejects.toThrow("Session expired")
    expect(useAuthStore.getState().user).toBeNull()
  })

  it("does not clear the auth user on a failed /auth/login attempt", async () => {
    useAuthStore.getState().setUser({
      id: "u1",
      username: "u",
      email: null,
      role: "system_admin",
      createdAt: "",
    } as never)

    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ detail: "Bad credentials" }), {
        status: 401,
        headers: { "Content-Type": "application/json" },
      }),
    )

    await expect(api.auth.login("x", "y")).rejects.toThrow("Bad credentials")
    expect(useAuthStore.getState().user).not.toBeNull()
  })
  it("maps logical models and keeps ':' and '/' of the model key in the path", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({
          modelKey: "org/gemma4:26b",
          displayName: "Gemma 4 26B",
          category: "llm",
          defaultBackend: "llamacpp_mks",
          deployments: [
            {
              modelId: "llamacpp_mks/org/gemma4:26b",
              backend: "llamacpp_mks",
              isDefault: true,
              installed: true,
              status: "loaded",
              loadPolicy: "pin",
              pinPermanent: false,
              preferredGpu: 1,
              artifact: { kind: "gguf", source: "ollama", ref: "gemma4:26b", mmproj: true },
              profiles: ["org/gemma4:26b"],
            },
          ],
          compatibleBackends: [
            { backend: "llamacpp_mks", status: "compatible", installed: true, reason: "GGUF", deployed: true },
            { backend: "bitnet", status: "probable", installed: false, reason: "ternario" },
          ],
          publicProfiles: ["org/gemma4:26b"],
          hiddenProfiles: [],
          registryMetadata: null,
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    )

    const model = await api.logicalModels.patch("org/gemma4:26b", { defaultBackend: "llamacpp_mks" })

    const [url, init] = fetchSpy.mock.calls[0] as [string, RequestInit]
    expect(url).toBe("/ocabra/logical-models/org%2Fgemma4%3A26b")
    expect(init.method).toBe("PATCH")
    expect(JSON.parse(String(init.body))).toEqual({ defaultBackend: "llamacpp_mks" })
    expect(model.deployments[0]?.artifact?.mmproj).toBe(true)
    expect(model.deployments[0]?.preferredGpu).toBe(1)
    expect(model.compatibleBackends[1]).toEqual({
      backend: "bitnet",
      status: "probable",
      installed: false,
      reason: "ternario",
    })
    expect(model.compatibleBackends[0]?.deployed).toBe(true)
  })

  it("maps compatible_backends on Ollama search cards", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify([
          {
            name: "gemma4",
            description: "Gemma",
            tags: ["vision"],
            size_gb: 9.6,
            pulls: 10,
            compatible_backends: [
              { backend: "llamacpp_mks", status: "compatible", installed: true, reason: "GGUF del tag" },
            ],
          },
        ]),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    )
    const cards = await api.registry.searchOllama("gemma4")
    expect(cards[0]?.sizeGb).toBe(9.6)
    expect(cards[0]?.compatibleBackends?.[0]?.backend).toBe("llamacpp_mks")
  })

  it("sends model_key, default_backend and backends in register_config", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ job_id: "j1", source: "ollama", model_ref: "gemma4:e4b", status: "queued" }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    )
    await api.downloads.enqueue("ollama", "gemma4:e4b", null, {
      displayName: "gemma4",
      loadPolicy: "on_demand",
      modelKey: "gemma4:e4b",
      defaultBackend: "llamacpp_mks",
      backends: ["llamacpp_mks", "ollama"],
    })
    const body = JSON.parse(String((fetchSpy.mock.calls[0] as [string, RequestInit])[1].body))
    expect(body.register_config).toEqual(
      expect.objectContaining({
        model_key: "gemma4:e4b",
        default_backend: "llamacpp_mks",
        backends: ["llamacpp_mks", "ollama"],
      }),
    )
  })
})
