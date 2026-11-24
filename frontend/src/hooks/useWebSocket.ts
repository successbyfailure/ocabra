import { useEffect, useRef, useState } from "react"
import type { DownloadJob, GPUState, ModelStatus, ServiceStatus, WSEvent } from "@/types"
import { useGpuStore } from "@/stores/gpuStore"
import { useModelStore } from "@/stores/modelStore"
import { useDownloadStore } from "@/stores/downloadStore"
import { useServiceStore } from "@/stores/serviceStore"
import { useBackendsStore } from "@/stores/backendsStore"
import { useAgentsStore } from "@/stores/agentsStore"
import { useMCPStore } from "@/stores/mcpStore"
import { api } from "@/api/client"
import { getRemoteNode } from "@/lib/remoteNode"

const REMOTE_POLL_MS = 5_000

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

export function normalizeEvent(rawEvent: unknown): WSEvent | null {
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
          canonicalModelId:
            typeof data.canonical_model_id === "string"
              ? data.canonical_model_id
              : undefined,
          status: String(data.status ?? data.new_status ?? "configured") as ModelStatus,
        },
      }
    }
  }

  if (maybeEvent.type === "download_progress") {
    const data = maybeEvent.data as Record<string, unknown>
    if (data && typeof data.job_id === "string") {
      // The backend publishes the whole job (``DownloadJob.model_dump``), the
      // same payload the SSE stream and GET /downloads carry: the percentage
      // is ``progress_pct``. Reading ``pct`` (absent) turned every WebSocket
      // update into 0 %, which overwrote the SSE value a moment later — the
      // bar flashed the real figure and fell back to 0 (2026-11-16).
      const num = (value: unknown): number | null =>
        value === null || value === undefined || value === "" ? null : Number(value)
      return {
        type: "download_progress",
        data: {
          jobId: data.job_id,
          pct: Number(data.progress_pct ?? data.progressPct ?? data.pct ?? 0),
          speedMbS: num(data.speed_mb_s ?? data.speedMbS),
          etaSeconds: num(data.eta_seconds ?? data.etaSeconds),
          status: (data.status as DownloadJob["status"] | undefined) ?? null,
          error: (data.error as string | null | undefined) ?? null,
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

/** Ventana de agrupación de ``model_event``: en una ráfaga (una carga toca
 *  varios modelos a la vez) llegaban N eventos y se disparaba un GET por cada
 *  uno, sin orden garantizado. Se juntan en un único resync desde la lista. */
const MODEL_EVENT_DEBOUNCE_MS = 200

export function useWebSocket() {
  const [connected, setConnected] = useState(false)
  const [lastEvent, setLastEvent] = useState<WSEvent | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectDelayMs = useRef(1000)
  const reconnectTimer = useRef<number | null>(null)
  const modelResyncTimer = useRef<number | null>(null)

  const setGpus = useGpuStore((state) => state.setGpus)
  const updateJob = useDownloadStore((state) => state.updateJob)
  const updateService = useServiceStore((state) => state.updateService)
  const handleBackendEvent = useBackendsStore((state) => state.handleWSEvent)
  const handleAgentEvent = useAgentsStore((state) => state.handleWSEvent)
  const handleMCPEvent = useMCPStore((state) => state.handleWSEvent)

  const setModels = useModelStore((state) => state.setModels)

  useEffect(() => {
    // Remote panel: our /ocabra/ws only carries *this* server's events, so
    // poll the remote node's GPUs and models through the panel proxy instead.
    if (getRemoteNode()) {
      let stopped = false
      const poll = async () => {
        try {
          const [gpus, models] = await Promise.all([api.gpus.list(), api.models.list()])
          if (stopped) return
          setGpus(gpus)
          setModels(models)
          setConnected(true)
        } catch {
          if (!stopped) setConnected(false)
        }
      }
      void poll()
      const timer = window.setInterval(() => void poll(), REMOTE_POLL_MS)
      return () => {
        stopped = true
        window.clearInterval(timer)
      }
    }

    let cancelled = false

    // Coalesce model events: a burst of them collapses into one authoritative
    // ``models.list()`` (correct order + federation info), debounced. Also used
    // to resync after a reconnect, when we may have missed events while down.
    const resyncModels = () => {
      void api.models
        .list()
        .then((models) => {
          if (!cancelled) setModels(models)
        })
        .catch(() => {
          /* transient: the next event or reconnect retries */
        })
    }
    const scheduleModelResync = () => {
      if (modelResyncTimer.current !== null) return
      modelResyncTimer.current = window.setTimeout(() => {
        modelResyncTimer.current = null
        resyncModels()
      }, MODEL_EVENT_DEBOUNCE_MS)
    }

    function connect() {
      if (cancelled) return
      const ws = new WebSocket(getWebSocketUrl())
      wsRef.current = ws

      ws.onopen = () => {
        setConnected(true)
        reconnectDelayMs.current = 1000
        // Resync whatever changed while the socket was down.
        resyncModels()
      }

      ws.onmessage = (e) => {
        try {
          const event = normalizeEvent(JSON.parse(e.data))
          if (!event) return

          if (event.type === "gpu_stats") {
            setGpus(event.data)
          } else if (event.type === "model_event") {
            // Runtime presets publish events with their derived ``::<hash>``
            // worker id. Instead of one GET per event (unordered, and the GET
            // misses federation info), coalesce the burst into a single
            // debounced ``models.list()`` resync that is authoritative.
            scheduleModelResync()
          } else if (event.type === "download_progress") {
            updateJob(event.data.jobId, {
              progressPct: event.data.pct,
              speedMbS: event.data.speedMbS,
              etaSeconds: event.data.etaSeconds,
              status: event.data.status ?? (event.data.pct >= 100 ? "completed" : "downloading"),
              error: event.data.error,
            })
          } else if (event.type === "service_event") {
            updateService(event.data.serviceId, event.data.service)
          } else if (
            event.type === "backend_installed" ||
            event.type === "backend_uninstalled" ||
            event.type === "backend_progress"
          ) {
            handleBackendEvent(event.type, event.data)
          } else if (event.type === "agent_updated") {
            handleAgentEvent(event.type, event.data)
          } else if (event.type === "mcp_server_health_changed") {
            handleMCPEvent(event.type, event.data)
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
      if (modelResyncTimer.current !== null) {
        window.clearTimeout(modelResyncTimer.current)
        modelResyncTimer.current = null
      }
      wsRef.current?.close()
    }
  }, [setGpus, setModels, updateJob, updateService, handleBackendEvent, handleAgentEvent, handleMCPEvent])

  return { connected, lastEvent }
}
