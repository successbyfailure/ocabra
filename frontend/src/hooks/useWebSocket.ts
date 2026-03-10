import { useEffect, useRef, useState } from "react"
import type { ModelStatus, WSEvent } from "@/types"
import { useGpuStore } from "@/stores/gpuStore"
import { useModelStore } from "@/stores/modelStore"
import { useDownloadStore } from "@/stores/downloadStore"

const getWebSocketUrl = () =>
  `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.host}/ocabra/ws`

function normalizeEvent(rawEvent: unknown): WSEvent | null {
  if (typeof rawEvent !== "object" || rawEvent === null) {
    return null
  }

  const maybeEvent = rawEvent as Record<string, unknown>
  if (typeof maybeEvent.type !== "string") {
    return null
  }

  if (maybeEvent.type === "model_event") {
    const data = maybeEvent.data as Record<string, unknown>
    if (data && typeof data.model_id === "string") {
      return {
        type: "model_event",
        data: {
          event: String(data.event ?? "status_changed"),
          modelId: data.model_id,
          status: String(data.status ?? "configured") as ModelStatus,
        },
      }
    }
  }

  if (maybeEvent.type === "download_progress") {
    const data = maybeEvent.data as Record<string, unknown>
    if (data && typeof data.job_id === "string") {
      return {
        type: "download_progress",
        data: {
          jobId: data.job_id,
          pct: Number(data.pct ?? 0),
          speedMbS: Number(data.speed_mb_s ?? data.speedMbS ?? 0),
        },
      }
    }
  }

  return maybeEvent as WSEvent
}

export function useWebSocket() {
  const [connected, setConnected] = useState(false)
  const [lastEvent, setLastEvent] = useState<WSEvent | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectDelayMs = useRef(1000)
  const reconnectTimer = useRef<number | null>(null)

  const setGpus = useGpuStore((state) => state.setGpus)
  const updateModel = useModelStore((state) => state.updateModel)
  const updateJob = useDownloadStore((state) => state.updateJob)

  useEffect(() => {
    let cancelled = false

    function connect() {
      if (cancelled) return
      const ws = new WebSocket(getWebSocketUrl())
      wsRef.current = ws

      ws.onopen = () => {
        setConnected(true)
        reconnectDelayMs.current = 1000
      }

      ws.onmessage = (e) => {
        try {
          const event = normalizeEvent(JSON.parse(e.data))
          if (!event) return

          if (event.type === "gpu_stats") {
            setGpus(event.data)
          } else if (event.type === "model_event") {
            updateModel(event.data.modelId, { status: event.data.status })
          } else if (event.type === "download_progress") {
            updateJob(event.data.jobId, {
              progressPct: event.data.pct,
              speedMbS: event.data.speedMbS,
              status: event.data.pct >= 100 ? "completed" : "downloading",
            })
          }

          setLastEvent(event)
        } catch {
          // ignore malformed messages
        }
      }

      ws.onclose = () => {
        setConnected(false)
        if (!cancelled) {
          reconnectTimer.current = window.setTimeout(connect, reconnectDelayMs.current)
          reconnectDelayMs.current = Math.min(reconnectDelayMs.current * 2, 30_000)
        }
      }
    }

    connect()
    return () => {
      cancelled = true
      if (reconnectTimer.current !== null) {
        window.clearTimeout(reconnectTimer.current)
      }
      wsRef.current?.close()
    }
  }, [setGpus, updateJob, updateModel])

  return { connected, lastEvent }
}
