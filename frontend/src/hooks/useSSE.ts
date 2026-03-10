import { useEffect } from "react"

export function useSSE<T = unknown>(url: string, onMessage: (data: T) => void) {
  useEffect(() => {
    if (!url) return

    const source = new EventSource(url)
    source.onmessage = (event) => {
      try {
        onMessage(JSON.parse(event.data) as T)
      } catch {
        // Ignore malformed payloads.
      }
    }

    return () => {
      source.close()
    }
  }, [onMessage, url])
}
