// @vitest-environment jsdom
import { render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { Settings } from "@/pages/Settings"

const { configGet, configPatch, gpusList, localModelsList, syncLiteLLM } = vi.hoisted(() => ({
  configGet: vi.fn(),
  configPatch: vi.fn(),
  gpusList: vi.fn(),
  localModelsList: vi.fn(),
  syncLiteLLM: vi.fn(),
}))

vi.mock("sonner", () => ({
  toast: {
    success: vi.fn(),
    error: vi.fn(),
  },
}))

vi.mock("@/api/client", () => ({
  api: {
    config: {
      get: configGet,
      patch: configPatch,
      syncLiteLLM,
    },
    gpus: {
      list: gpusList,
    },
    registry: {
      listLocal: localModelsList,
    },
  },
}))

describe("Settings", () => {
  beforeEach(() => {
    configGet.mockResolvedValue({
      defaultGpuIndex: 1,
      idleTimeoutSeconds: 123,
      idleEvictionCheckIntervalSeconds: 15,
      vramBufferMb: 768,
      vramPressureThresholdPct: 91,
      logLevel: "warning",
      litellmBaseUrl: "http://litellm:4000",
      litellmAdminKey: "***",
      litellmAutoSync: true,
      energyCostEurKwh: 0.42,
      modelsDir: "/srv/models",
      downloadDir: "/srv/models/downloads",
      maxTemperatureC: 93,
      globalSchedules: [
        {
          id: "night",
          days: [1, 2, 3],
          start: "01:00",
          end: "05:00",
          enabled: true,
        },
      ],
    })
    gpusList.mockResolvedValue([
      {
        index: 0,
        name: "RTX 3060",
        totalVramMb: 12288,
        freeVramMb: 6400,
        usedVramMb: 5888,
        utilizationPct: 42,
        temperatureC: 58,
        powerDrawW: 142,
        powerLimitW: 170,
        lockedVramMb: 512,
        processes: [],
      },
      {
        index: 1,
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
    localModelsList.mockResolvedValue([
      {
        modelId: "vllm/mistral-7b",
        path: "/srv/models/mistral-7b",
        sizeGb: 12.5,
        backendType: "vllm",
        configured: true,
      },
    ])
    configPatch.mockResolvedValue({})
    syncLiteLLM.mockResolvedValue({ syncedModels: 0 })
    localStorage.clear()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("renders server config from the API and ignores localStorage fallbacks", async () => {
    localStorage.setItem("ocabra.modelsDir", "/legacy/models")
    localStorage.setItem("ocabra.downloadDir", "/legacy/downloads")
    localStorage.setItem("ocabra.maxTemperatureC", "77")

    const getItemSpy = vi.spyOn(Storage.prototype, "getItem")

    render(<Settings />)

    await waitFor(() => {
      expect((screen.getByLabelText("MODELS_DIR") as HTMLInputElement).value).toBe("/srv/models")
    })

    expect((screen.getByLabelText("MODELS_DIR") as HTMLInputElement).value).toBe("/srv/models")
    expect((screen.getByLabelText("Carpeta de descarga de modelos") as HTMLInputElement).value).toBe(
      "/srv/models/downloads",
    )
    expect((screen.getByLabelText("Temperatura maxima alerta (C)") as HTMLInputElement).value).toBe("93")
    expect((screen.getByLabelText("GPU preferida por defecto") as HTMLSelectElement).value).toBe("1")
    expect((screen.getByLabelText("URL proxy LiteLLM") as HTMLInputElement).value).toBe("http://litellm:4000")
    expect(getItemSpy).not.toHaveBeenCalled()
  })
})
