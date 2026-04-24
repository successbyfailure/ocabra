import { useEffect, useRef, useState } from "react"
import type { GPUState, ModelStatus, ServiceStatus, WSEvent } from "@/types"
import { useGpuStore } from "@/stores/gpuStore"
import { useModelStore } from "@/stores/modelStore"
import { useDownloadStore } from "@/stores/downloadStore"
import { useServiceStore } from "@/stores/serviceStore"
import { useBackendsStore } from "@/stores/backendsStore"

const getWebSocketUrl = () =>
  `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.host}/ocabra/ws`

function normalizeGpuStats(rawData: unknown): GPUState[] | null {
  if (!Array.isArray(rawData)) {
    return null
  }

  return rawData.map((gpu) => {
    const data = (gpu ?? {}) as Record<string, unknown>
    return {
      index: Number(data.index ?? 0),
      name: String(data.name ?? "GPU"),
      totalVramMb: Number(data.total_vram_mb ?? data.totalVramMb ?? 0),
      freeVramMb: Number(data.free_vram_mb ?? data.freeVramMb ?? 0),
      usedVramMb: Number(data.used_vram_mb ?? data.usedVramMb ?? 0),
      utilizationPct: Number(data.utilization_pct ?? data.utilizationPct ?? 0),
      temperatureC: Number(data.temperature_c ?? data.temperatureC ?? 0),
      powerDrawW: Number(data.power_draw_w ?? data.powerDrawW ?? 0),
      powerLimitW: Number(data.power_limit_w ?? data.powerLimitW ?? 0),
      lockedVramMb: Number(data.locked_vram_mb ?? data.lockedVramMb ?? 0),
      processes: Array.isArray(data.processes)
        ? data.processes.map((processRaw) => {
            const process = (processRaw ?? {}) as Record<string, unknown>
            return {
              pid: Number(process.pid ?? 0),
              processName: (process.process_name ?? process.processName ?? null) as string | null,
              processType: String(process.process_type ?? process.processType ?? "compute") as
                | "compute"
                | "graphics",
              usedVramMb: Number(process.used_vram_mb ?? process.usedVramMb ?? 0),
            }
          })
        : [],
    }
  })
}

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
          status: String(data.status ?? data.new_status ?? "configured") as ModelStatus,
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

  if (maybeEvent.type === "service_event") {
    const data = maybeEvent.data as Record<string, unknown>
    if (data && typeof data.service_id === "string") {
      const svc = (data.service ?? {}) as Record<string, unknown>
      return {
        type: "service_event",
        data: {
          event: String(data.event ?? ""),
          serviceId: data.service_id,
          status: String(data.status ?? "unknown") as ServiceStatus,
          service: {
            serviceId: String(svc.service_id ?? svc.serviceId ?? data.service_id),
            serviceType: String(svc.service_type ?? svc.serviceType ?? ""),
            displayName: String(svc.display_name ?? svc.displayName ?? ""),
            uiUrl: String(svc.ui_url ?? svc.uiUrl ?? ""),
            preferredGpu: svc.preferred_gpu == null && svc.preferredGpu == null ? null : Number(svc.preferred_gpu ?? svc.preferredGpu),
            idleUnloadAfterSeconds: Number(svc.idle_unload_after_seconds ?? svc.idleUnloadAfterSeconds ?? 600),
            enabled: Boolean(svc.enabled ?? true),
            serviceAlive: Boolean(svc.service_alive ?? svc.serviceAlive),
            runtimeLoaded: Boolean(svc.runtime_loaded ?? svc.runtimeLoaded),
            status: String(svc.status ?? "unknown") as ServiceStatus,
            activeModelRef: (svc.active_model_ref ?? svc.activeModelRef ?? null) as string | null,
            lastActivityAt: (svc.last_activity_at ?? svc.lastActivityAt ?? null) as string | null,
            lastHealthCheckAt: (svc.last_health_check_at ?? svc.lastHealthCheckAt ?? null) as string | null,
            detail: (svc.detail ?? null) as string | null,
            isGenerating: Boolean(svc.is_generating ?? svc.isGenerating ?? false),
            queueDepth: Number(svc.queue_depth ?? svc.queueDepth ?? 0),
            vramUsedMb: svc.vram_used_mb == null && svc.vramUsedMb == null ? null : Number(svc.vram_used_mb ?? svc.vramUsedMb),
            gpuUtilPct: svc.gpu_util_pct == null && svc.gpuUtilPct == null ? null : Number(svc.gpu_util_pct ?? svc.gpuUtilPct),
            cpuPct: svc.cpu_pct == null && svc.cpuPct == null ? null : Number(svc.cpu_pct ?? svc.cpuPct),
            memUsedMb: svc.mem_used_mb == null && svc.memUsedMb == null ? null : Number(svc.mem_used_mb ?? svc.memUsedMb),
            memLimitMb: svc.mem_limit_mb == null && svc.memLimitMb == null ? null : Number(svc.mem_limit_mb ?? svc.memLimitMb),
          },
        },
      }
    }
  }

  if (maybeEvent.type === "gpu_stats") {
    const data = normalizeGpuStats(maybeEvent.data)
    if (!data) {
      return null
    }
    return {
      type: "gpu_stats",
      data,
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
  const updateService = useServiceStore((state) => state.updateService)
  const handleBackendEvent = useBackendsStore((state) => state.handleWSEvent)

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
          } else if (event.type === "service_event") {
            updateService(event.data.serviceId, event.data.service)
          } else if (
            event.type === "backend_installed" ||
            event.type === "backend_uninstalled" ||
            event.type === "backend_progress"
          ) {
            handleBackendEvent(event.type, event.data)
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
  }, [setGpus, updateJob, updateModel, updateService, handleBackendEvent])

  return { connected, lastEvent }
}
