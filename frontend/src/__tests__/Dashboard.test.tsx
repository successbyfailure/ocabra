// @vitest-environment jsdom
import { render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { Dashboard } from "@/pages/Dashboard"
import { useAuthStore } from "@/stores/authStore"
import { useDownloadStore } from "@/stores/downloadStore"
import { useGpuStore } from "@/stores/gpuStore"
import { useModelStore } from "@/stores/modelStore"
import { useServiceStore } from "@/stores/serviceStore"

const {
  listGpus,
  listModels,
  listDownloads,
  listServices,
  getHostStats,
  getTokens,
  getRequestStats,
  getEnergyStats,
  getServerPower,
  getActivity,
  getOllamaRuntime,
  getRecent,
  getPeers,
} = vi.hoisted(() => ({
  listGpus: vi.fn(),
  listModels: vi.fn(),
  listDownloads: vi.fn(),
  listServices: vi.fn(),
  getHostStats: vi.fn(),
  getTokens: vi.fn(),
  getRequestStats: vi.fn(),
  getEnergyStats: vi.fn(),
  getServerPower: vi.fn(),
  getActivity: vi.fn(),
  getOllamaRuntime: vi.fn(),
  getRecent: vi.fn(),
  getPeers: vi.fn(),
}))

vi.mock("@/hooks/useWebSocket", () => ({
  useWebSocket: () => ({ connected: true, lastEvent: null }),
}))

vi.mock("@/api/client", () => ({
  api: {
    gpus: { list: listGpus },
    models: { list: listModels, activity: getActivity, ollamaRuntime: getOllamaRuntime },
    downloads: { list: listDownloads },
    services: { list: listServices },
    host: { stats: getHostStats },
    stats: {
      tokens: getTokens,
      requests: getRequestStats,
      energy: getEnergyStats,
      serverPower: getServerPower,
      recent: getRecent,
      cancelRequest: vi.fn(),
    },
    federation: { getPeers },
  },
}))

// jsdom intentaría conectar de verdad: un socket que nunca abre deja al hook
// en su respaldo REST, que es lo que se prueba aquí.
class FakeWebSocket {
  onopen: (() => void) | null = null
  onmessage: ((e: { data: string }) => void) | null = null
  onclose: (() => void) | null = null
  send() {}
  close() {}
}

describe("Dashboard", () => {
  beforeEach(() => {
    vi.stubGlobal("WebSocket", FakeWebSocket)
    vi.stubGlobal(
      "ResizeObserver",
      class ResizeObserver {
        observe() {}
        unobserve() {}
        disconnect() {}
      },
    )

    useGpuStore.setState({ gpus: [], lastUpdated: null })
    useModelStore.setState({ models: {} })
    useDownloadStore.setState({ jobs: [] })
    useServiceStore.setState({ services: {} })
    useAuthStore.setState({
      user: {
        id: "user-1",
        username: "manager",
        email: null,
        role: "model_manager",
        createdAt: "2026-03-01T00:00:00Z",
      },
      isLoading: false,
    })

    listGpus.mockResolvedValue([
      {
        index: 0,
        name: "RTX 3090",
        totalVramMb: 24576,
        freeVramMb: 16000,
        usedVramMb: 8576,
        utilizationPct: 34,
        temperatureC: 61,
        powerDrawW: 220,
        powerLimitW: 350,
        lockedVramMb: 1024,
        processes: [],
      },
    ])

    listModels.mockResolvedValue([
      {
        modelId: "mistral-7b",
        displayName: "Mistral 7B",
        backendType: "vllm",
        status: "loaded",
        loadPolicy: "warm",
        pinPermanent: false,
        preferredGpu: null,
        currentGpu: [0],
        vramUsedMb: 6000,
        capabilities: {
          chat: true,
          completion: true,
          tools: false,
          vision: false,
          embeddings: false,
          pooling: false,
          rerank: false,
          classification: false,
          score: false,
          reasoning: false,
          imageGeneration: false,
          audioTranscription: false,
          musicGeneration: false,
          tts: false,
        audioInput: false,
        audioOutput: false,
        videoInput: false,
          streaming: true,
          contextLength: 8192,
        },
        lastRequestAt: null,
        loadedAt: "2026-03-23T12:00:00Z",
      },
      {
        modelId: "qwen-14b",
        displayName: "Qwen 14B",
        backendType: "vllm",
        status: "loading",
        loadPolicy: "on_demand",
        pinPermanent: false,
        preferredGpu: 1,
        currentGpu: [1],
        vramUsedMb: 0,
        capabilities: {
          chat: true,
          completion: true,
          tools: true,
          vision: false,
          embeddings: false,
          pooling: false,
          rerank: false,
          classification: false,
          score: false,
          reasoning: true,
          imageGeneration: false,
          audioTranscription: false,
          musicGeneration: false,
          tts: false,
        audioInput: false,
        audioOutput: false,
        videoInput: false,
          streaming: true,
          contextLength: 32768,
        },
        lastRequestAt: null,
        loadedAt: null,
      },
    ])

    listDownloads.mockResolvedValue([
      {
        jobId: "job-1",
        source: "huggingface",
        modelRef: "meta-llama/Llama-3.1-8B",
        status: "downloading",
        progressPct: 42,
        speedMbS: 12.2,
        etaSeconds: 180,
        error: null,
        startedAt: "2026-03-10T00:00:00Z",
        completedAt: null,
      },
    ])

    listServices.mockResolvedValue([])

    getHostStats.mockResolvedValue({
      cpuPct: 15,
      cpuCount: 64,
      cpuCountPhysical: 32,
      memTotalMb: 131072,
      memUsedMb: 32768,
      memPct: 25,
      swapTotalMb: 0,
      swapUsedMb: 0,
      swapPct: 0,
      loadAvg1m: 1.1,
      loadAvg5m: 1.2,
      loadAvg15m: 1.3,
    })
    getServerPower.mockResolvedValue({
      cpuPowerW: 90,
      cpuTempC: 55,
      totalGpuPowerW: 220,
      totalPowerW: 310,
    })
    getTokens.mockImplementation((params: { allTime?: boolean; from?: string }) => {
      if (params.allTime) {
        return Promise.resolve({
          totalInputTokens: 1_250_000,
          totalOutputTokens: 750_000,
          byBackend: [{ backendType: "vllm", inputTokens: 1_250_000, outputTokens: 750_000 }],
          byGpu: [
            { gpuIndex: 0, inputTokens: 1_000_000, outputTokens: 500_000 },
            { gpuIndex: null, inputTokens: 250_000, outputTokens: 250_000 },
          ],
          series: [],
        })
      }
      return Promise.resolve({
        totalInputTokens: 1000,
        totalOutputTokens: 500,
        byBackend: [{ backendType: "vllm", inputTokens: 1000, outputTokens: 500 }],
        byGpu: [{ gpuIndex: 0, inputTokens: 1000, outputTokens: 500 }],
        series: [],
      })
    })
    getRequestStats.mockResolvedValue({
      totalRequests: 4,
      errorRate: 0,
      rejections: 0,
      avgDurationMs: 120,
      p50DurationMs: 100,
      p95DurationMs: 180,
      series: [],
    })
    getEnergyStats.mockResolvedValue({
      totalKwh: 0.12,
      estimatedCostEur: 0.03,
      byGpu: [{ gpuIndex: 0, totalKwh: 0.12, powerDrawW: 220 }],
      totalServerKwh: 0.25,
    })
    getActivity.mockResolvedValue({ activity: {}, stuckThresholdSeconds: 300 })
    getOllamaRuntime.mockResolvedValue({})
    getRecent.mockResolvedValue({ requests: [] })
    getPeers.mockResolvedValue([])
  })

  it("loads and renders dashboard sections with API data", async () => {
    render(<Dashboard />)

    await waitFor(() => {
      expect(screen.getAllByText("#0 RTX 3090").length).toBeGreaterThan(0)
      expect(screen.getAllByText("Mistral 7B").length).toBeGreaterThan(0)
      expect(screen.getByText("Qwen 14B")).toBeTruthy()
      expect(screen.getByText("Peticiones · 1 h / 24 h")).toBeTruthy()
      expect(screen.getByText("310 W")).toBeTruthy()
      expect(screen.getByText(/Nada esperando/)).toBeTruthy()
    })

    expect(listGpus).toHaveBeenCalledTimes(1)
    expect(listModels).toHaveBeenCalledTimes(1)
    expect(listServices).toHaveBeenCalledTimes(1)
    // Retiradas del dashboard: agentes activos y la sección de descargas
    // (ahora es un indicador de la cabecera).
    expect(screen.queryByText(/Active agents/i)).toBeNull()
    expect(screen.queryByText("Descargas activas")).toBeNull()
  })

  it("renders live requests, the queue and the context bar with its one-GPU mark", async () => {
    const now = Date.now() / 1000
    const base = {
      username: "ana",
      inputTokensEstimate: 9000,
      outputTokensEstimate: null,
      tokensPerSecond: null,
      lastActivitySecondsAgo: 0.2,
      phase: "generating",
      waitingOnModelId: null,
      streaming: true,
      requestKind: "chat",
      profileId: "mistral",
      apiKeyName: null,
      source: "api",
      kvReservedTokens: 20000,
      queuePosition: null,
      queuedMs: null,
      maxOutputTokens: null,
      expectedDurationMs: 20000,
    }
    getActivity.mockResolvedValue({
      stuckThresholdSeconds: 300,
      idleTimeoutSeconds: 300,
      activity: {
        "mistral-7b": {
          inFlight: 3,
          oldestSeconds: 12,
          slots: 2,
          queuedCount: 1,
          prefillTokensPerSecondTotal: 1800,
          generationTokensPerSecondTotal: 42.5,
          requests: [
            {
              ...base,
              requestId: "gen",
              stage: "generating",
              startedAt: now - 12,
              elapsedSeconds: 12,
              elapsedMs: 12000,
              admittedAfterMs: 0,
              firstTokenAfterMs: 4000,
              prompt: { totalTokens: 9000, cachedTokens: 3000, processedTokens: 9000, source: "live" },
              prefillPct: 100,
              prefillTokensPerSecond: 1500,
              prefillSource: "live",
              outputTokens: 340,
              generationTokensPerSecond: 42.5,
              estimatedRemainingMs: 8000,
            },
            {
              ...base,
              requestId: "pre",
              username: "luis",
              stage: "prefill",
              startedAt: now - 3,
              elapsedSeconds: 3,
              elapsedMs: 3000,
              admittedAfterMs: 0,
              firstTokenAfterMs: null,
              prompt: { totalTokens: 50000, cachedTokens: 0, processedTokens: 30000, source: "live" },
              prefillPct: 60,
              prefillTokensPerSecond: 1800,
              prefillSource: "live",
              outputTokens: null,
              generationTokensPerSecond: null,
              estimatedRemainingMs: 30000,
            },
            {
              ...base,
              requestId: "q",
              username: "eva",
              stage: "queued",
              startedAt: now - 1,
              elapsedSeconds: 1,
              elapsedMs: 1000,
              admittedAfterMs: null,
              firstTokenAfterMs: null,
              queuePosition: 0,
              queuedMs: 1000,
              prompt: { totalTokens: 400, cachedTokens: null, processedTokens: null, source: "estimate" },
              prefillPct: null,
              prefillTokensPerSecond: null,
              prefillSource: null,
              outputTokens: null,
              generationTokensPerSecond: null,
              estimatedRemainingMs: null,
              queueWaitEstimateMs: 8000,
            },
          ],
        },
      },
      contexts: {
        "mistral-7b": {
          mode: "pool",
          scaleMaxTokens: 262144,
          activeMaxTokens: 98304,
          activePresetKey: "normal",
          activeGpus: [0],
          singleGpuMaxTokens: 98304,
          tiers: [
            { presetKey: "normal", gpus: [0], maxTokens: 98304, slots: 4 },
            { presetKey: "extended", gpus: [0, 1], maxTokens: 262144, slots: 4 },
          ],
          reservedTokens: 40000,
          segments: [
            { requestId: "gen", reservedTokens: 20000, usedTokens: 9340 },
            { requestId: "pre", reservedTokens: 20000, usedTokens: 30000 },
          ],
          usagePct: null,
          source: "live",
        },
      },
    })

    render(<Dashboard />)

    await waitFor(() => {
      expect(screen.getByText("prefill 60% · 30.0k/50.0k")).toBeTruthy()
      expect(screen.getByText("340 tok")).toBeTruthy()
      // Totales de la card: 2 en curso de 2 slots, 1 en cola.
      expect(screen.getByText("2/2")).toBeTruthy()
      // Cola: puesto, usuario y espera estimada.
      expect(screen.getAllByText("#1").length).toBeGreaterThan(0)
      expect(screen.getByText("eva")).toBeTruthy()
      expect(screen.getByTestId("requests-waiting").textContent).toContain("entra en ~8 s")
      // Barra de contexto: marca del tope de una GPU y escala de dos.
      expect(screen.getByText("96k · #0 RTX 3090")).toBeTruthy()
      // La GPU 1 no está en el store del test: sale solo con su índice.
      expect(screen.getByText("256k · #0 RTX 3090 + #1")).toBeTruthy()
      expect(screen.getByText(/rayado: requiere escalar/)).toBeTruthy()
    })
    // La petición en cola no sale como barra en la card, solo en la cola.
    expect(document.querySelectorAll('[data-model-id] [data-stage="queued"]').length).toBe(0)
    expect(document.querySelectorAll('[data-model-id] [data-stage="prefill"]').length).toBe(1)
  })

  it("explains that external runtimes do not count as active models", async () => {
    listModels.mockResolvedValue([])
    listServices.mockResolvedValue([
      {
        serviceId: "comfyui",
        serviceType: "comfyui",
        displayName: "ComfyUI",
        baseUrl: "http://comfyui:8188",
        uiUrl: "http://localhost:8188",
        healthPath: "/",
        unloadPath: "/free",
        preferredGpu: 1,
        idleUnloadAfterSeconds: 600,
        idleAction: "stop",
        enabled: false,
        serviceAlive: true,
        runtimeLoaded: false,
        status: "disabled",
        activeModelRef: null,
        lastActivityAt: null,
        lastHealthCheckAt: null,
        lastUnloadAt: null,
        detail: "disabled_but_container_running:ocabra-comfyui-1",
        extra: {},
      },
    ])

    render(<Dashboard />)

    await waitFor(() => {
      expect(screen.getByText("ComfyUI")).toBeTruthy()
      expect(screen.getByText(/Hay runtimes externos ocupando GPU/i)).toBeTruthy()
    })
  })
})
