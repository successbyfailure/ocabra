// Stub — Stream 1-D implements the full logic.
// Structure is defined here so other streams can import it.
import { useEffect, useRef, useState } from "react"
import type { WSEvent } from "@/types"

const WS_URL = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ocabra/ws`

export function useWebSocket() {
  const [connected, setConnected] = useState(false)
  const [lastEvent, setLastEvent] = useState<WSEvent | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const retryDelay = useRef(1000)

  useEffect(() => {
    let cancelled = false

    function connect() {
      if (cancelled) return
      const ws = new WebSocket(WS_URL)
      wsRef.current = ws

      ws.onopen = () => {
        setConnected(true)
        retryDelay.current = 1000
      }

      ws.onmessage = (e) => {
        try {
          const event = JSON.parse(e.data) as WSEvent
          setLastEvent(event)
        } catch {
          // ignore malformed messages
        }
      }

      ws.onclose = () => {
        setConnected(false)
        if (!cancelled) {
          setTimeout(connect, retryDelay.current)
          retryDelay.current = Math.min(retryDelay.current * 2, 30_000)
        }
      }
    }

    connect()
    return () => {
      cancelled = true
      wsRef.current?.close()
    }
  }, [])

  return { connected, lastEvent }
}
