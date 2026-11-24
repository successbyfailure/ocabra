// Tipos del dominio «system» (REF-15).
// Extraídos de types/index.ts; el barrel index.ts los reexporta con export *.

import type { BackendModuleState } from "./backends"
import type { GPUState } from "./gpus"
import type { ModelStatus } from "./models"
import type { DownloadJob } from "./registry"
import type { ServiceState, ServiceStatus } from "./services"

export interface HostStats {
  cpuModel?: string | null;
  /** Potencia máxima de la CPU (W): ajuste, límite RAPL o TDP conocido. */
  cpuPowerMaxW?: number | null;
  cpuPowerMaxSource?: "config" | "rapl" | "tdp" | null;
  cpuPct: number;
  cpuCount: number;
  cpuCountPhysical: number;
  loadAvg1m: number;
  loadAvg5m: number;
  loadAvg15m: number;
  memTotalMb: number;
  memUsedMb: number;
  memFreeMb: number;
  memPct: number;
  swapTotalMb: number;
  swapUsedMb: number;
  swapPct: number;
}

// WebSocket events
export type WSEvent =
  | { type: "gpu_stats"; data: GPUState[] }
  | {
      type: "model_event";
      data: {
        event: string;
        modelId: string;
        canonicalModelId?: string;
        status: ModelStatus;
      };
    }
  | {
      type: "download_progress";
      data: {
        jobId: string;
        pct: number;
        speedMbS: number | null;
        etaSeconds: number | null;
        status: DownloadJob["status"] | null;
        error: string | null;
      };
    }
  | {
      type: "service_event";
      data: {
        event: string;
        serviceId: string;
        status: ServiceStatus;
        service: ServiceState;
      };
    }
  | { type: "system_alert"; data: { level: "info" | "warning" | "error"; message: string } }
  | { type: "peer_online"; data: { peer_id: string; name: string } }
  | { type: "peer_offline"; data: { peer_id: string; name: string } }
  | { type: "remote_model_loaded"; data: { peer_id: string; model_id: string } }
  | {
      type: "remote_model_unloaded";
      data: { peer_id: string; model_id: string };
    }
  | { type: "backend_installed"; data: BackendModuleState }
  | { type: "backend_uninstalled"; data: { backendType: string } }
  | { type: "backend_progress"; data: BackendModuleState }
  | { type: "agent_updated"; data: { slug: string } }
  | {
      type: "mcp_server_health_changed";
      data: {
        id: string;
        health_status: "unknown" | "healthy" | "unhealthy";
        last_error?: string | null;
      };
    };
