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
