// @vitest-environment jsdom
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { Settings } from "@/pages/Settings"

const {
  configGet,
  configPatch,
  gpusList,
  localModelsList,
  modelsList,
  profilesListAll,
  syncLiteLLM,
} = vi.hoisted(() => ({
  configGet: vi.fn(),
  configPatch: vi.fn(),
  gpusList: vi.fn(),
  localModelsList: vi.fn(),
  modelsList: vi.fn(),
  profilesListAll: vi.fn(),
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
    // GeneralSettings rellena los selectores de modelo por defecto con
    // api.models.list() y api.profiles.listAll(). Sin estos, api.models era
    // undefined y el componente reventaba al leer `.list` — el .catch() del
    // componente no lo cubre porque el fallo es el acceso a la propiedad,
    // antes de que exista la promesa. Settings renderizaba vacio y el test
    // fallaba con "no encuentro /srv/models" en vez de con el TypeError real.
    models: {
      list: modelsList,
    },
    profiles: {
      listAll: profilesListAll,
    },
  },
}))

describe("Settings", () => {
  beforeEach(() => {
    configGet.mockResolvedValue({
      defaultGpuIndex: 1,
      idleTimeoutSeconds: 123,
      idleEvictionCheckIntervalSeconds: 15,
      modelLoadWaitTimeoutSeconds: 720,
      pressureEvictionDrainTimeoutSeconds: 60,
      pinReloadPollIntervalSeconds: 120,
      vramBufferMb: 768,
      vramPressureThresholdPct: 91,
      openaiAudioMaxPartSizeMb: 256,
      whisperStartupTimeoutSeconds: 300,
      logLevel: "warning",
      litellmBaseUrl: "http://litellm:4000",
      litellmAdminKey: "***",
      litellmAutoSync: true,
      energyCostEurKwh: 0.42,
      modelsDir: "/srv/models",
      downloadDir: "/srv/models/downloads",
      maxTemperatureC: 93,
      vllmGpuMemoryUtilization: 0.85,
      vllmMaxNumSeqs: 16,
      vllmMaxNumBatchedTokens: 8192,
      vllmEnablePrefixCaching: true,
      vllmEnforceEager: true,
      sglangMemFractionStatic: 0.9,
      sglangContextLength: null,
      sglangDisableRadixCache: false,
      llamaCppGpuLayers: 0,
      llamaCppCtxSize: 4096,
      llamaCppFlashAttn: false,
      bitnetGpuLayers: 0,
      bitnetCtxSize: 4096,
      bitnetFlashAttn: false,
      diffusersTorchDtype: "auto",
      diffusersOffloadMode: "none",
      diffusersEnableTorchCompile: false,
      diffusersEnableXformers: false,
      diffusersAllowTf32: true,
      tensorrtLlmEnabled: false,
      tensorrtLlmMaxBatchSize: null,
      tensorrtLlmContextLength: null,
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
    modelsList.mockResolvedValue([])
    profilesListAll.mockResolvedValue([])
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
    // Settings esta repartida en pestanas y solo monta la activa, asi que hay
    // que ir a cada una antes de comprobar sus campos: modelsDir/downloadDir
    // viven en "Almacenamiento", la temperatura y la GPU por defecto en
    // "GPUs" y la URL del proxy en "LiteLLM". El test se escribio cuando la
    // pagina era una sola columna y comprobaba los cinco campos de golpe, asi
    // que desde que hay pestanas no encontraba ninguno.
    localStorage.setItem("ocabra.modelsDir", "/legacy/models")
    localStorage.setItem("ocabra.downloadDir", "/legacy/downloads")
    localStorage.setItem("ocabra.maxTemperatureC", "77")

    const getItemSpy = vi.spyOn(Storage.prototype, "getItem")

    render(<Settings />)

    // Radix Tabs cambia de pestana en mousedown, no en click: con
    // fireEvent.click el data-state se quedaba en "inactive" y el contenido
    // nunca se montaba.
    const irA = async (nombre: RegExp) => {
      const tab = await screen.findByRole("tab", { name: nombre })
      fireEvent.mouseDown(tab)
    }

    await irA(/Almacenamiento/i)
    await waitFor(() => {
      expect(screen.getByDisplayValue("/srv/models")).toBeTruthy()
    })
    expect((screen.getByLabelText("Carpeta de descarga de modelos") as HTMLInputElement).value).toBe(
      "/srv/models/downloads",
    )

    await irA(/GPUs/i)
    await waitFor(() => {
      expect((screen.getByLabelText("Temperatura maxima alerta (C)") as HTMLInputElement).value).toBe("93")
    })
    expect((screen.getByLabelText("GPU preferida por defecto") as HTMLSelectElement).value).toBe("1")

    await irA(/LiteLLM/i)
    await waitFor(() => {
      expect((screen.getByLabelText("URL proxy LiteLLM") as HTMLInputElement).value).toBe(
        "http://litellm:4000",
      )
    })

    expect(getItemSpy).not.toHaveBeenCalled()
  })
})
