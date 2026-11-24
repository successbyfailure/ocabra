// Cliente tipado de /ocabra/* — dominio «services» (REF-15).
// Extraído de api/client.ts; usa la capa HTTP única de api/http.ts.

import { request } from "@/api/http"
import { isRecord } from "@/api/_helpers"
import type {
  ServiceState,
} from "@/types"

export function toServiceState(raw: unknown): ServiceState {
  const data = isRecord(raw) ? raw : {};
  return {
    serviceId: String(data.service_id ?? data.serviceId ?? ""),
    serviceType: String(data.service_type ?? data.serviceType ?? ""),
    displayName: String(data.display_name ?? data.displayName ?? ""),
    uiUrl: String(data.ui_url ?? data.uiUrl ?? ""),
    preferredGpu:
      data.preferred_gpu == null && data.preferredGpu == null
        ? null
        : Number(data.preferred_gpu ?? data.preferredGpu),
    idleUnloadAfterSeconds: Number(
      data.idle_unload_after_seconds ?? data.idleUnloadAfterSeconds ?? 600,
    ),
    enabled: Boolean(data.enabled ?? true),
    serviceAlive: Boolean(data.service_alive ?? data.serviceAlive),
    runtimeLoaded: Boolean(data.runtime_loaded ?? data.runtimeLoaded),
    status: String(data.status ?? "unknown") as ServiceState["status"],
    activeModelRef: (data.active_model_ref ?? data.activeModelRef ?? null) as
      string | null,
    lastActivityAt: (data.last_activity_at ?? data.lastActivityAt ?? null) as
      string | null,
    lastHealthCheckAt: (data.last_health_check_at ??
      data.lastHealthCheckAt ??
      null) as string | null,
    detail: (data.detail ?? null) as string | null,
    isGenerating: Boolean(data.is_generating ?? data.isGenerating ?? false),
    queueDepth: Number(data.queue_depth ?? data.queueDepth ?? 0),
    vramUsedMb:
      data.vram_used_mb == null && data.vramUsedMb == null
        ? null
        : Number(data.vram_used_mb ?? data.vramUsedMb),
    gpuUtilPct:
      data.gpu_util_pct == null && data.gpuUtilPct == null
        ? null
        : Number(data.gpu_util_pct ?? data.gpuUtilPct),
    cpuPct:
      data.cpu_pct == null && data.cpuPct == null
        ? null
        : Number(data.cpu_pct ?? data.cpuPct),
    memUsedMb:
      data.mem_used_mb == null && data.memUsedMb == null
        ? null
        : Number(data.mem_used_mb ?? data.memUsedMb),
    memLimitMb:
      data.mem_limit_mb == null && data.memLimitMb == null
        ? null
        : Number(data.mem_limit_mb ?? data.memLimitMb),
  };
}


export const servicesApiModule = {
  services: {
    list: async () =>
      (await request<unknown[]>("GET", "/ocabra/services")).map(toServiceState),
    refresh: async (serviceId: string) =>
      toServiceState(
        await request<unknown>(
          "POST",
          `/ocabra/services/${encodeURIComponent(serviceId)}/refresh`,
        ),
      ),
    unload: async (serviceId: string) =>
      toServiceState(
        await request<unknown>(
          "POST",
          `/ocabra/services/${encodeURIComponent(serviceId)}/unload`,
        ),
      ),
    start: async (serviceId: string) =>
      toServiceState(
        await request<unknown>(
          "POST",
          `/ocabra/services/${encodeURIComponent(serviceId)}/start`,
        ),
      ),
    patch: async (serviceId: string, patch: { enabled: boolean }) =>
      toServiceState(
        await request<unknown>(
          "PATCH",
          `/ocabra/services/${encodeURIComponent(serviceId)}`,
          patch,
        ),
      ),
  },

};
