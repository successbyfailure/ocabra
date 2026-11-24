// @vitest-environment jsdom
import { act, render, waitFor } from "@testing-library/react"
import { describe, expect, it, beforeEach, vi } from "vitest"
import { useWebSocket } from "@/hooks/useWebSocket"
import { useGpuStore } from "@/stores/gpuStore"
import { useModelStore } from "@/stores/modelStore"
import { api } from "@/api/client"
import type { ModelState } from "@/types"

class MockWebSocket {
  static instances: MockWebSocket[] = []

  onopen: (() => void) | null = null
  onclose: (() => void) | null = null
  onmessage: ((event: { data: string }) => void) | null = null
  close = vi.fn(() => {
    this.onclose?.()
  })

  constructor(_url: string) {
    MockWebSocket.instances.push(this)
  }

  emitOpen() {
    this.onopen?.()
  }

  emitMessage(data: unknown) {
    this.onmessage?.({ data: JSON.stringify(data) })
  }

  emitClose() {
    this.onclose?.()
  }
}

function Harness() {
  useWebSocket()
  return null
}

describe("useWebSocket", () => {
  beforeEach(() => {
    MockWebSocket.instances = []
    vi.useRealTimers()
    vi.stubGlobal("WebSocket", MockWebSocket)
    // The hook resyncs the model list on open and after model events; default
    // it to empty so unrelated tests don't hit the network.
    vi.spyOn(api.models, "list").mockResolvedValue([])
    useGpuStore.setState({ gpus: [], lastUpdated: null })
    useModelStore.setState({ models: {} })
  })

  it("updates gpu store from gpu_stats event", async () => {
    render(<Harness />)
    const socket = MockWebSocket.instances[0]

    act(() => {
      socket.emitOpen()
      socket.emitMessage({
        type: "gpu_stats",
        data: [
          {
            index: 0,
            name: "RTX 3090",
            totalVramMb: 24576,
            freeVramMb: 12000,
            usedVramMb: 12576,
            utilizationPct: 71,
            temperatureC: 70,
            powerDrawW: 280,
            powerLimitW: 350,
            lockedVramMb: 2048,
            processes: [],
          },
        ],
      })
    })

    await waitFor(() => {
      expect(useGpuStore.getState().gpus).toHaveLength(1)
      expect(useGpuStore.getState().gpus[0].name).toBe("RTX 3090")
    })
  })

  it("reconnects with exponential backoff after close", async () => {
    vi.useFakeTimers()

    render(<Harness />)
    const firstSocket = MockWebSocket.instances[0]

    act(() => {
      firstSocket.emitClose()
    })

    expect(MockWebSocket.instances).toHaveLength(1)

    await act(async () => {
      vi.advanceTimersByTime(1000)
    })

    expect(MockWebSocket.instances).toHaveLength(2)
  })

  it("coalesces a burst of model events into one authoritative list resync", async () => {
    const canonical = "llama_cpp/deadbydawn/chaos-gguf::Chaos-Q4_K_M"
    const listModels = vi.spyOn(api.models, "list").mockResolvedValue([
      {
        modelId: canonical,
        backendType: "llama_cpp",
        status: "loading",
        runtimePhase: "upsizing",
        runtimeSlots: 4,
      } as ModelState,
    ])
    render(<Harness />)
    const socket = MockWebSocket.instances[0]
    // The open itself triggers one resync; ignore it and count events after.
    act(() => socket.emitOpen())
    await waitFor(() => expect(listModels).toHaveBeenCalled())
    listModels.mockClear()

    // A burst of three derived events must collapse into a single list() call.
    act(() => {
      for (let i = 0; i < 3; i++) {
        socket.emitMessage({
          type: "model_event",
          data: {
            model_id: `${canonical}::0123456789ab`,
            canonical_model_id: canonical,
            new_status: "loading",
          },
        })
      }
    })

    await waitFor(() => expect(listModels).toHaveBeenCalledTimes(1))
    expect(useModelStore.getState().models[canonical]?.runtimePhase).toBe("upsizing")
    // No phantom derived card from the ``::<hash>`` worker id.
    expect(useModelStore.getState().models[`${canonical}::0123456789ab`]).toBeUndefined()
    listModels.mockRestore()
  })

  it("does not invent a phantom model from a partial update", () => {
    useModelStore.getState().updateModel("unknown/model", { status: "loading" })
    expect(useModelStore.getState().models["unknown/model"]).toBeUndefined()
  })
})
