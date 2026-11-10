// @vitest-environment jsdom
import { fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { GpuWidget } from "@/components/dashboard/DeviceWidget"
import { QueueSection } from "@/components/dashboard/QueueSection"
import { RequestProgress } from "@/components/dashboard/RequestProgress"
import { DownloadQueue } from "@/components/downloads/DownloadQueue"
import { DownloadsIndicator } from "@/components/layout/DownloadsIndicator"
import { useDownloadStore } from "@/stores/downloadStore"
import type { ActiveRequestProgress, GPUState, ModelState } from "@/types"

vi.mock("@/api/client", () => ({
  api: {
    downloads: {
      list: vi.fn().mockResolvedValue([]),
      streamProgress: vi.fn(() => ({ onmessage: null, onerror: null, close: () => {} })),
      cancel: vi.fn(),
    },
    stats: { cancelRequest: vi.fn() },
  },
}))

const gpu = (index: number, totalVramMb: number): GPUState => ({
  index,
  name: index === 0 ? "RTX 3060" : "RTX 3090",
  totalVramMb,
  freeVramMb: 0,
  usedVramMb: totalVramMb - 1000,
  utilizationPct: 50,
  temperatureC: 60,
  powerDrawW: 100,
  powerLimitW: 300,
  lockedVramMb: 0,
  processes: [],
})

const nemotron = {
  modelId: "llamacpp_mks/nemotron-3-nano:30b",
  displayName: "Nemotron 3 Nano 30B",
  backendType: "llamacpp_mks",
  status: "loaded",
  currentGpu: [0, 1],
  vramUsedMb: 30972,
  ramUsedMb: 0,
} as unknown as ModelState

function req(extra: Partial<ActiveRequestProgress>): ActiveRequestProgress {
  return {
    requestId: "r",
    username: "ana",
    elapsedSeconds: 5,
    elapsedMs: 5000,
    inputTokensEstimate: 10000,
    outputTokensEstimate: null,
    tokensPerSecond: null,
    lastActivitySecondsAgo: 0.5,
    phase: "prompt_eval",
    waitingOnModelId: null,
    stage: "prefill",
    ...extra,
  }
}

describe("dashboard en vivo", () => {
  beforeEach(() => {
    useDownloadStore.setState({ jobs: [], queueOpen: false })
  })

  it("usa la VRAM medida de cada GPU, no la mitad del total", () => {
    const vram = { [nemotron.modelId]: { "0": 9566, "1": 21406 } }
    const { unmount } = render(<GpuWidget gpu={gpu(0, 12288)} models={[nemotron]} activity={{}} vramByGpu={vram} />)
    expect(screen.getByText("9.3/30.2 GB")).toBeTruthy()
    unmount()
    render(<GpuWidget gpu={gpu(1, 24576)} models={[nemotron]} activity={{}} vramByGpu={vram} />)
    expect(screen.getByText("20.9/30.2 GB")).toBeTruthy()
  })

  it("ordena la cola por arranque previsto y explica el adelantamiento por KV", () => {
    const queued = (id: string, user: string, pos: number, start: number) =>
      req({
        requestId: id,
        username: user,
        stage: "queued",
        queuePosition: pos,
        predictedStartOrder: start,
        kvReservedTokens: 1000,
        queueWaitEstimateMs: start * 1000,
        prompt: { totalTokens: 100, cachedTokens: null, processedTokens: null, source: "estimate" },
      })
    render(
      <QueueSection
        models={{}}
        activity={{
          m: {
            inFlight: 2,
            oldestSeconds: 1,
            slots: 4,
            requests: [queued("grande", "bea", 0, 1), queued("pequeña", "cris", 1, 0)],
          },
        }}
      />,
    )
    const rows = screen.getAllByRole("listitem")
    expect(rows[0].textContent).toContain("cris")
    expect(rows[0].textContent).toContain("se adelanta: cabe en el KV libre")
    expect(rows[1].textContent).toContain("bea")
    expect(rows[1].textContent).toContain("espera KV: la adelantan otras")
  })

  it("muestra el prefill estimado cuando el backend no lo informa", () => {
    render(
      <RequestProgress
        req={req({
          prefillPct: 40,
          prefillTokensPerSecond: 1000,
          prefillSource: "historical",
          prompt: { totalTokens: 10000, cachedTokens: null, processedTokens: null, source: "estimate" },
        })}
        hue="#000"
        stuckThreshold={300}
        canCancel={false}
      />,
    )
    expect(screen.getByText("prefill ~40% · ~10.0k tok")).toBeTruthy()
    expect(screen.getByTitle(/Ritmo de prefill medido/)).toBeTruthy()
  })

  it("la píldora de la cabecera abre y cierra el panel de descargas", () => {
    useDownloadStore.setState({
      jobs: [
        {
          jobId: "j",
          source: "huggingface",
          modelRef: "unsloth/Qwen3-GGUF",
          status: "downloading",
          progressPct: 63,
          speedMbS: 80,
          etaSeconds: 100,
          error: null,
          startedAt: new Date().toISOString(),
          completedAt: null,
          kind: "install",
        },
      ],
    })
    render(
      <>
        <DownloadsIndicator />
        <DownloadQueue />
      </>,
    )
    // Cerrado: solo la píldora, sin barra flotante.
    expect(screen.queryByText(/Cola de descargas/)).toBeNull()
    fireEvent.click(screen.getByTestId("downloads-indicator"))
    expect(screen.getByText(/Cola de descargas/)).toBeTruthy()
    fireEvent.click(screen.getByTestId("downloads-indicator"))
    expect(screen.queryByText(/Cola de descargas/)).toBeNull()
  })
})

describe("estado del modelo, memoria y potencia de CPU", () => {
  const base = {
    modelId: "llamacpp_mks/qwen",
    displayName: "Qwen",
    backendType: "llamacpp_mks",
    status: "loaded",
    loadPolicy: "warm",
    currentGpu: [1],
    vramUsedMb: 22800,
    vramEstimateMb: 21500,
    ramUsedMb: 0,
    capabilities: {},
  } as unknown as ModelState

  it("prioriza esperar espacio, luego cargar, y distingue sin streaming de parada", async () => {
    const { cardState } = await import("@/components/dashboard/ModelLiveCard")
    expect(
      cardState(base, undefined, { phase: "waiting_for_gpu", waitingOn: "gemma4:26b", loadingForMs: null, typicalLoadMs: null }, 300),
    ).toMatchObject({ visual: "waiting_space", detail: "desalojando gemma4:26b" })
    expect(
      cardState({ ...base, status: "loading" } as ModelState, undefined, { phase: "loading", waitingOn: null, loadingForMs: 30_000, typicalLoadMs: 40_000 }, 300),
    ).toMatchObject({ visual: "loading", detail: "30 s de ~40 s habituales" })
    const act = (stage: string, stalled = false) => ({
      inFlight: 1,
      oldestSeconds: 70,
      requests: [req({ stage: stage as ActiveRequestProgress["stage"], stalled })],
    })
    expect(cardState(base, act("processing"), undefined, 300)).toMatchObject({ visual: "processing" })
    expect(cardState(base, act("generating", true), undefined, 300)).toMatchObject({ visual: "stuck", label: "Sin actividad" })
    expect(cardState(base, act("generating"), undefined, 300)).toMatchObject({ visual: "generating" })
    expect(cardState(base, undefined, undefined, 300)).toMatchObject({ visual: "idle" })
  })

  it("la card muestra la memoria medida por GPU con la reserva, en ámbar si la supera", async () => {
    const { ModelLiveCard } = await import("@/components/dashboard/ModelLiveCard")
    render(
      <ModelLiveCard
        model={base}
        gpus={[gpu(1, 24576)]}
        vramByGpu={{ "1": 22800 }}
        nowMs={Date.now()}
        stuckThreshold={300}
        onUnload={() => undefined}
        canCancel={() => false}
      />,
    )
    const memory = screen.getByTestId("model-memory")
    expect(memory.textContent).toContain("22.3 GB de 21.0 reservados")
    expect(memory.textContent).toContain("22.3 / 24.0 GB")
    expect(memory.querySelector(".text-amber-400")).toBeTruthy()
  })

  it("la barra de potencia de la CPU escala contra el TDP y lo dice", async () => {
    const { CpuWidget } = await import("@/components/dashboard/DeviceWidget")
    const stats = {
      cpuModel: "AMD Ryzen Threadripper 3960X 24-Core Processor",
      cpuPowerMaxW: 280,
      cpuPowerMaxSource: "tdp" as const,
      cpuPct: 20, cpuCount: 48, cpuCountPhysical: 24, loadAvg1m: 1, loadAvg5m: 1, loadAvg15m: 1,
      memTotalMb: 1000, memUsedMb: 500, memFreeMb: 500, memPct: 50, swapTotalMb: 0, swapUsedMb: 0, swapPct: 0,
    }
    render(
      <CpuWidget
        stats={stats}
        serverPower={{ cpuPowerW: 140, cpuTempC: 60, totalGpuPowerW: 0, totalPowerW: 140 }}
        history={[]}
        models={[]}
        activity={{}}
        ollamaRuntime={{}}
      />,
    )
    expect(screen.getByText("140/280 W")).toBeTruthy()
    expect(screen.getByText(/potencia sobre TDP 280 W/)).toBeTruthy()
    expect(screen.getByText("Ryzen Threadripper 3960X · 24C/48T")).toBeTruthy()
  })
})


describe("memoria repartida entre GPUs", () => {
  it("no marca OOM por un reparto desigual si el total está dentro de la reserva", async () => {
    const { ModelLiveCard } = await import("@/components/dashboard/ModelLiveCard")
    const nemo = {
      ...nemotron,
      loadPolicy: "on_demand",
      vramEstimateMb: 32000,
      capabilities: {},
    } as unknown as ModelState
    render(
      <ModelLiveCard
        model={nemo}
        gpus={[gpu(0, 12288), gpu(1, 24576)]}
        vramByGpu={{ "0": 9566, "1": 21406 }}
        nowMs={Date.now()}
        stuckThreshold={300}
        onUnload={() => undefined}
        canCancel={() => false}
      />,
    )
    const memory = screen.getByTestId("model-memory")
    expect(memory.textContent).toContain("30.2 GB de 31.3 reservados")
    expect(memory.textContent).not.toContain("riesgo de OOM")
    expect(memory.querySelector(".text-amber-400")).toBeNull()
  })
})

describe("contexto cambiado en vivo", () => {
  it("la barra dice el tamaño de carga cuando se cambió sin recargar", async () => {
    const { ContextBar } = await import("@/components/dashboard/ContextBar")
    render(
      <ContextBar
        requestOrder={new Map()}
        ctx={{
          mode: "pool",
          scaleMaxTokens: 262144,
          activeMaxTokens: 131072,
          liveResized: true,
          nominalMaxTokens: 262144,
          activePresetKey: null,
          activeGpus: [1],
          singleGpuMaxTokens: 262144,
          tiers: [{ presetKey: null, gpus: [1], maxTokens: 262144, slots: 4 }],
          reservedTokens: 0,
          segments: [],
          usagePct: null,
          source: "estimate",
        }}
      />,
    )
    expect(screen.getByText(/cambiado en vivo \(cargado con 256k\)/)).toBeTruthy()
  })
})
