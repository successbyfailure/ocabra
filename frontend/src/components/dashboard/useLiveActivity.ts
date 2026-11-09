import { useEffect, useRef, useState } from "react"
import { api } from "@/api/client"
import type { ModelActivityMap } from "@/types"

const EMPTY: ModelActivityMap = { activity: {}, contexts: {}, stuckThresholdSeconds: 300 }
const FALLBACK_POLL_MS = 3000

const wsUrl = () =>
  `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.host}/ocabra/ws`

/**
 * Actividad en vivo de los modelos para el dashboard.
 *
 * El backend la empuja cada segundo por `/ocabra/ws` a quien manda
 * `{"subscribe": "models_activity"}`. Mientras el socket no está abierto
 * (arranque, reconexión, proxy sin WebSocket) se consulta por REST cada 3 s,
 * así que el dashboard nunca se queda congelado sin avisar.
 *
 * Abre su propio socket: el global de `Layout` no se suscribe, para no
 * calcular la actividad para cada pestaña que no sea el dashboard.
 */
export function useLiveActivity(): { data: ModelActivityMap; live: boolean; receivedAt: number } {
  const [data, setData] = useState<ModelActivityMap>(EMPTY)
  const [live, setLive] = useState(false)
  const [receivedAt, setReceivedAt] = useState(0)
  const liveRef = useRef(false)

  useEffect(() => {
    let cancelled = false
    let ws: WebSocket | null = null
    let reconnectTimer: number | null = null
    let delay = 1000

    const accept = (payload: ModelActivityMap) => {
      if (cancelled) return
      setData({ ...EMPTY, ...payload })
      setReceivedAt(Date.now())
    }

    const connect = () => {
      if (cancelled) return
      try {
        ws = new WebSocket(wsUrl())
      } catch {
        return
      }
      ws.onopen = () => {
        delay = 1000
        ws?.send(JSON.stringify({ subscribe: "models_activity" }))
      }
      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data)
          if (msg?.type === "models_activity" && msg.data) {
            liveRef.current = true
            setLive(true)
            accept(msg.data as ModelActivityMap)
          }
        } catch {
          /* mensaje malformado: se ignora */
        }
      }
      ws.onclose = () => {
        liveRef.current = false
        setLive(false)
        if (!cancelled) {
          reconnectTimer = window.setTimeout(connect, delay)
          delay = Math.min(delay * 2, 30_000)
        }
      }
    }

    const poll = async () => {
      if (liveRef.current) return
      try {
        accept(await api.models.activity())
      } catch {
        /* se conserva lo último */
      }
    }

    connect()
    void poll()
    const pollTimer = window.setInterval(() => void poll(), FALLBACK_POLL_MS)
    return () => {
      cancelled = true
      window.clearInterval(pollTimer)
      if (reconnectTimer !== null) window.clearTimeout(reconnectTimer)
      ws?.close()
    }
  }, [])

  return { data, live, receivedAt }
}
