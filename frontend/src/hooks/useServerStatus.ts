import { useEffect, useState } from "react"

export interface ServerStatus {
  loads: {
    queue_depth: number
    waiting: number
    active: number
    in_progress: string[]
  }
  workers: {
    loaded_count: number
    loaded_ids: string[]
    in_flight_requests: number
  }
}

export function useServerStatus(intervalMs = 5000): ServerStatus | null {
  const [status, setStatus] = useState<ServerStatus | null>(null)

  useEffect(() => {
    let cancelled = false
    const controller = new AbortController()

    const poll = async () => {
      try {
        const r = await fetch("/ocabra/status", {
          signal: controller.signal,
          credentials: "include",
        })
        if (!r.ok) return
        const body = (await r.json()) as ServerStatus
        if (!cancelled) setStatus(body)
      } catch {
        // Ignore — the badge just disappears until the next poll succeeds.
      }
    }

    void poll()
    const id = window.setInterval(poll, intervalMs)
    return () => {
      cancelled = true
      controller.abort()
      window.clearInterval(id)
    }
  }, [intervalMs])

  return status
}
