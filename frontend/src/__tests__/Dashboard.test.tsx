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
    },
  },
}))

describe("Dashboard", () => {
  beforeEach(() => {
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
        autoReload: true,
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
        autoReload: false,
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
  })

  it("loads and renders dashboard sections with API data", async () => {
    render(<Dashboard />)

    await waitFor(() => {
      expect(screen.getByText("RTX 3090")).toBeTruthy()
      expect(screen.getAllByText("Mistral 7B").length).toBeGreaterThan(0)
      expect(screen.getByText("Qwen 14B")).toBeTruthy()
      expect(screen.getByText("Tokens · historico")).toBeTruthy()
      expect(screen.getAllByText("2.0M").length).toBeGreaterThan(0)
      expect(screen.getAllByText("1.5M").length).toBeGreaterThan(0)
    })

    expect(listGpus).toHaveBeenCalledTimes(1)
    expect(listModels).toHaveBeenCalledTimes(1)
    expect(listDownloads).toHaveBeenCalledTimes(1)
    expect(listServices).toHaveBeenCalledTimes(1)
    expect(getTokens).toHaveBeenCalledWith({ allTime: true, includeSeries: false })
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
