// @vitest-environment jsdom
import { render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { Dashboard } from "@/pages/Dashboard"
import { useDownloadStore } from "@/stores/downloadStore"
import { useGpuStore } from "@/stores/gpuStore"
import { useModelStore } from "@/stores/modelStore"

const { listGpus, listModels, listDownloads } = vi.hoisted(() => ({
  listGpus: vi.fn(),
  listModels: vi.fn(),
  listDownloads: vi.fn(),
}))

vi.mock("@/hooks/useWebSocket", () => ({
  useWebSocket: () => ({ connected: true, lastEvent: null }),
}))

vi.mock("@/api/client", () => ({
  api: {
    gpus: { list: listGpus },
    models: { list: listModels },
    downloads: { list: listDownloads },
  },
}))

describe("Dashboard", () => {
  beforeEach(() => {
    useGpuStore.setState({ gpus: [], lastUpdated: null })
    useModelStore.setState({ models: {} })
    useDownloadStore.setState({ jobs: [] })

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
          reasoning: false,
          imageGeneration: false,
          audioTranscription: false,
          tts: false,
          streaming: true,
          contextLength: 8192,
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
  })

  it("loads and renders dashboard sections with API data", async () => {
    render(<Dashboard />)

    await waitFor(() => {
      expect(screen.getByText("RTX 3090")).toBeTruthy()
      expect(screen.getByText("Mistral 7B")).toBeTruthy()
      expect(screen.getByText("meta-llama/Llama-3.1-8B")).toBeTruthy()
    })

    expect(screen.getByText("Live updates connected")).toBeTruthy()
    expect(listGpus).toHaveBeenCalledTimes(1)
    expect(listModels).toHaveBeenCalledTimes(1)
    expect(listDownloads).toHaveBeenCalledTimes(1)
  })
})
