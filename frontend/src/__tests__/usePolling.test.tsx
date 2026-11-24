// @vitest-environment jsdom
import { render } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { usePolling } from "@/hooks/usePolling"

function Harness({ cb, interval = 1000 }: { cb: () => void; interval?: number }) {
  usePolling(cb, interval)
  return null
}

function setHidden(hidden: boolean) {
  Object.defineProperty(document, "hidden", { configurable: true, get: () => hidden })
  document.dispatchEvent(new Event("visibilitychange"))
}

describe("usePolling", () => {
  beforeEach(() => {
    vi.useFakeTimers()
    setHidden(false)
  })
  afterEach(() => {
    vi.useRealTimers()
  })

  it("runs immediately and then on the interval", () => {
    const cb = vi.fn()
    render(<Harness cb={cb} interval={1000} />)
    expect(cb).toHaveBeenCalledTimes(1)
    vi.advanceTimersByTime(3000)
    expect(cb).toHaveBeenCalledTimes(4)
  })

  it("pauses while the tab is hidden and refreshes on return", () => {
    const cb = vi.fn()
    render(<Harness cb={cb} interval={1000} />)
    expect(cb).toHaveBeenCalledTimes(1)

    setHidden(true)
    vi.advanceTimersByTime(5000)
    // No polling while hidden.
    expect(cb).toHaveBeenCalledTimes(1)

    setHidden(false)
    // Immediate catch-up refresh on return...
    expect(cb).toHaveBeenCalledTimes(2)
    // ...and the interval resumes.
    vi.advanceTimersByTime(1000)
    expect(cb).toHaveBeenCalledTimes(3)
  })
})
