import { describe, expect, it } from "vitest"
import { normalizeEvent } from "@/hooks/useWebSocket"

describe("download_progress del WebSocket", () => {
  it("lee progress_pct del trabajo completo que publica el backend", () => {
    // El backend publica DownloadJob.model_dump(): no hay campo "pct".
    // Leer "pct" daba 0 en cada evento y pisaba el valor bueno del SSE.
    const event = normalizeEvent({
      type: "download_progress",
      data: {
        job_id: "j1",
        status: "downloading",
        progress_pct: 42.5,
        speed_mb_s: 37.2,
        eta_seconds: 90,
        error: null,
      },
    })
    expect(event).toEqual({
      type: "download_progress",
      data: {
        jobId: "j1",
        pct: 42.5,
        speedMbS: 37.2,
        etaSeconds: 90,
        status: "downloading",
        error: null,
      },
    })
  })

  it("respeta el estado real (un fallo no se queda en «descargando»)", () => {
    const event = normalizeEvent({
      type: "download_progress",
      data: { job_id: "j2", status: "failed", progress_pct: 12, error: "disco lleno" },
    })
    expect(event?.type === "download_progress" && event.data.status).toBe("failed")
  })
})
