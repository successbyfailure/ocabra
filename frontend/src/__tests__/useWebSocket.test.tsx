// @vitest-environment jsdom
import { act, render, waitFor } from "@testing-library/react"
import { describe, expect, it, beforeEach, vi } from "vitest"
import { useWebSocket } from "@/hooks/useWebSocket"
import { useGpuStore } from "@/stores/gpuStore"

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
    useGpuStore.setState({ gpus: [], lastUpdated: null })
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
})
