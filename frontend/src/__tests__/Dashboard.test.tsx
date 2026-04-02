// @vitest-environment jsdom
import { render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { Dashboard } from "@/pages/Dashboard"
import { useDownloadStore } from "@/stores/downloadStore"
import { useGpuStore } from "@/stores/gpuStore"
import { useModelStore } from "@/stores/modelStore"
import { useServiceStore } from "@/stores/serviceStore"

const { listGpus, listModels, listDownloads, listServices } = vi.hoisted(() => ({
  listGpus: vi.fn(),
  listModels: vi.fn(),
  listDownloads: vi.fn(),
  listServices: vi.fn(),
}))

vi.mock("@/hooks/useWebSocket", () => ({
  useWebSocket: () => ({ connected: true, lastEvent: null }),
}))

vi.mock("@/api/client", () => ({
  api: {
    gpus: { list: listGpus },
    models: { list: listModels },
    downloads: { list: listDownloads },
    services: { list: listServices },
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
  })

  it("loads and renders dashboard sections with API data", async () => {
    render(<Dashboard />)

    await waitFor(() => {
      expect(screen.getByText("RTX 3090")).toBeTruthy()
      expect(screen.getByText("Mistral 7B")).toBeTruthy()
      expect(screen.getByText("Qwen 14B")).toBeTruthy()
      expect(screen.getByText("meta-llama/Llama-3.1-8B")).toBeTruthy()
    })

    expect(screen.getByText("Live updates connected")).toBeTruthy()
    expect(listGpus).toHaveBeenCalledTimes(1)
    expect(listModels).toHaveBeenCalledTimes(1)
    expect(listDownloads).toHaveBeenCalledTimes(1)
    expect(listServices).toHaveBeenCalledTimes(1)
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
