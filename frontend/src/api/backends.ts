// Cliente tipado de /ocabra/* — dominio «backends» (REF-15).
// Extraído de api/client.ts; usa la capa HTTP única de api/http.ts.

import { extractErrorMessage, httpFetch, request, sseUrl } from "@/api/http"
import { isRecord } from "@/api/_helpers"
import type {
  CompileJob,
} from "@/types"

export function toBackendModuleState(
  raw: unknown,
): import("@/types").BackendModuleState {
  const d = isRecord(raw) ? raw : {};
  const status = String(
    d.install_status ?? d.installStatus ?? "not_installed",
  ) as import("@/types").BackendInstallStatus;
  const alwaysAvailable = Boolean(
    d.always_available ?? d.alwaysAvailable ?? false,
  );
  const installSource = (d.install_source ??
    d.installSource ??
    null) as import("@/types").BackendInstallSource;
  return {
    backendType: String(d.backend_type ?? d.backendType ?? ""),
    displayName: String(
      d.display_name ?? d.displayName ?? d.backend_type ?? d.backendType ?? "",
    ),
    description: String(d.description ?? ""),
    tags: Array.isArray(d.tags) ? (d.tags as unknown[]).map(String) : [],
    installStatus:
      alwaysAvailable && status === "installed" ? "built-in" : status,
    installedVersion: (d.installed_version ?? d.installedVersion ?? null) as
      string | null,
    installedAt: (d.installed_at ?? d.installedAt ?? null) as string | null,
    installSource,
    estimatedSizeMb: Number(d.estimated_size_mb ?? d.estimatedSizeMb ?? 0),
    actualSizeMb:
      d.actual_size_mb == null && d.actualSizeMb == null
        ? null
        : Number(d.actual_size_mb ?? d.actualSizeMb),
    modelsLoaded: Number(d.models_loaded ?? d.modelsLoaded ?? 0),
    hasUpdate: Boolean(d.has_update ?? d.hasUpdate ?? false),
    installProgress:
      d.install_progress == null && d.installProgress == null
        ? null
        : Number(d.install_progress ?? d.installProgress),
    installDetail: (d.install_detail ?? d.installDetail ?? null) as
      string | null,
    error: (d.error ?? null) as string | null,
    alwaysAvailable,
    ociRef: (d.oci_ref ?? d.ociRef ?? null) as string | null,
    ociDigest: (d.oci_digest ?? d.ociDigest ?? null) as string | null,
    packages: (isRecord(d.packages) ? d.packages : {}) as Record<string, string>,
    declaredPackages: Array.isArray(d.declared_packages ?? d.declaredPackages)
      ? ((d.declared_packages ?? d.declaredPackages) as unknown[]).map(String)
      : [],
    gitRef: (d.git_ref ?? d.gitRef ?? null) as string | null,
  };
}


export function toBackendPublishedInfo(raw: unknown): import("@/types").BackendPublishedInfo {
  const d = isRecord(raw) ? raw : {};
  const p = isRecord(d.published) ? d.published : null;
  return {
    updateAvailable: Boolean(d.update_available ?? d.updateAvailable ?? false),
    published: p
      ? {
          ref: String(p.ref ?? ""),
          digest: String(p.digest ?? ""),
          sizeBytes: Number(p.size_bytes ?? p.sizeBytes ?? 0),
          version: (p.version ?? null) as string | null,
          packages: (isRecord(p.packages) ? p.packages : {}) as Record<string, string>,
          gpuArchs: (p.gpu_archs ?? p.gpuArchs ?? null) as string | null,
          created: (p.created ?? null) as string | null,
        }
      : null,
  };
}


export function toCompileJob(raw: unknown): CompileJob {
  const data = isRecord(raw) ? raw : {};
  const cfg = isRecord(data.config) ? data.config : {};
  return {
    jobId: String(data.job_id ?? data.jobId ?? ""),
    sourceModel: String(data.source_model ?? data.sourceModel ?? ""),
    engineName: String(data.engine_name ?? data.engineName ?? ""),
    gpuIndices: Array.isArray(data.gpu_indices ?? data.gpuIndices)
      ? ((data.gpu_indices ?? data.gpuIndices) as unknown[]).map(Number)
      : [],
    dtype: String(data.dtype ?? "fp16"),
    config: {
      maxBatchSize: Number(cfg.max_batch_size ?? cfg.maxBatchSize ?? 1),
      maxInputLen: Number(cfg.max_input_len ?? cfg.maxInputLen ?? 2048),
      maxSeqLen: Number(cfg.max_seq_len ?? cfg.maxSeqLen ?? 4096),
    },
    status: String(data.status ?? "pending") as CompileJob["status"],
    phase: (data.phase ?? null) as CompileJob["phase"],
    progressPct: Number(data.progress_pct ?? data.progressPct ?? 0),
    errorDetail: (data.error_detail ?? data.errorDetail ?? null) as
      string | null,
    engineDir: (data.engine_dir ?? data.engineDir ?? null) as string | null,
    startedAt: (data.started_at ?? data.startedAt ?? null) as string | null,
    finishedAt: (data.finished_at ?? data.finishedAt ?? null) as string | null,
  };
}


export function toVramEstimate(raw: Record<string, unknown>): VramEstimate {
  const serveRaw = (raw.serve ?? {}) as Record<string, unknown>;
  const buildRaw = (raw.build ?? {}) as Record<string, unknown>;
  const diskRaw = (raw.disk ?? {}) as Record<string, unknown>;
  const breakdownRaw = (serveRaw.breakdown ?? {}) as Record<string, unknown>;

  return {
    estimatedParamsB:
      raw.estimated_params_b == null ? null : Number(raw.estimated_params_b),
    quant: String(raw.quant ?? "unknown"),
    tpSize: Number(raw.tp_size ?? 1),
    configFound: Boolean(raw.config_found),
    serve: {
      vramPerGpuMb: Number(serveRaw.vram_per_gpu_mb ?? 0),
      vramTotalMb: Number(serveRaw.vram_total_mb ?? 0),
      breakdown: {
        weightsMb: Number(breakdownRaw.weights_mb ?? 0),
        kvCacheMb: Number(breakdownRaw.kv_cache_mb ?? 0),
        overheadMb: Number(breakdownRaw.overhead_mb ?? 0),
      },
    },
    build: {
      vramPerGpuMb: Number(buildRaw.vram_per_gpu_mb ?? 0),
      vramTotalMb: Number(buildRaw.vram_total_mb ?? 0),
    },
    disk: {
      engineMb: Number(diskRaw.engine_mb ?? 0),
      checkpointMbTemp: Number(diskRaw.checkpoint_mb_temp ?? 0),
      totalPeakMb: Number(diskRaw.total_peak_mb ?? 0),
    },
    warnings: Array.isArray(raw.warnings) ? raw.warnings.map(String) : [],
  };
}


export interface VramEstimate {
  estimatedParamsB: number | null;
  quant: string;
  tpSize: number;
  configFound: boolean;
  serve: {
    vramPerGpuMb: number;
    vramTotalMb: number;
    breakdown: { weightsMb: number; kvCacheMb: number; overheadMb: number };
  };
  build: { vramPerGpuMb: number; vramTotalMb: number };
  disk: { engineMb: number; checkpointMbTemp: number; totalPeakMb: number };
  warnings: string[];
}


export const backendsApiModule = {
  backends: {
    list: async (): Promise<import("@/types").BackendModuleState[]> =>
      (await request<unknown[]>("GET", "/ocabra/backends")).map(
        toBackendModuleState,
      ),
    // NOTE: POST /install is modelled as GET here because EventSource only supports GET.
    // The backend exposes the install endpoint as GET so SSE works with a query param.
    published: async (
      backendType: string,
      refresh = false,
    ): Promise<import("@/types").BackendPublishedInfo> =>
      toBackendPublishedInfo(
        await request<unknown>(
          "GET",
          `/ocabra/backends/${encodeURIComponent(backendType)}/published${refresh ? "?refresh=true" : ""}`,
        ),
      ),
    install: (
      backendType: string,
      method: import("@/types").BackendInstallMethod = "auto",
    ): EventSource => {
      const qs = new URLSearchParams({ method });
      return new EventSource(
        sseUrl(`/ocabra/backends/${encodeURIComponent(backendType)}/install?${qs.toString()}`),
      );
    },
    uninstall: async (
      backendType: string,
    ): Promise<import("@/types").BackendModuleState | null> => {
      const raw = await request<unknown>(
        "POST",
        `/ocabra/backends/${encodeURIComponent(backendType)}/uninstall`,
      );
      return raw ? toBackendModuleState(raw) : null;
    },
    logs: async (backendType: string): Promise<string> => {
      const res = await httpFetch(`/ocabra/backends/${encodeURIComponent(backendType)}/logs`);
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(extractErrorMessage(err) ?? res.statusText);
      }
      return res.text();
    },
  },

  trtllm: {
    compile: async (body: {
      modelId: string;
      gpuIndices: number[];
      dtype: string;
      maxBatchSize: number;
      maxInputLen: number;
      maxSeqLen: number;
      engineName: string;
    }): Promise<CompileJob> =>
      toCompileJob(
        await request<unknown>("POST", "/ocabra/trtllm/compile", {
          model_id: body.modelId,
          gpu_indices: body.gpuIndices,
          dtype: body.dtype,
          max_batch_size: body.maxBatchSize,
          max_input_len: body.maxInputLen,
          max_seq_len: body.maxSeqLen,
          engine_name: body.engineName,
        }),
      ),
    list: async (): Promise<CompileJob[]> =>
      (await request<unknown[]>("GET", "/ocabra/trtllm/compile")).map(
        toCompileJob,
      ),
    cancel: async (jobId: string): Promise<CompileJob> =>
      toCompileJob(
        await request<unknown>(
          "DELETE",
          `/ocabra/trtllm/compile/${encodeURIComponent(jobId)}`,
        ),
      ),
    streamUrl: (jobId: string): string =>
      sseUrl(`/ocabra/trtllm/compile/${encodeURIComponent(jobId)}/stream`),
    deleteEngine: async (engineName: string): Promise<void> => {
      await request<unknown>(
        "DELETE",
        `/ocabra/trtllm/engines/${encodeURIComponent(engineName)}`,
      );
    },
    estimate: async (params: {
      modelId: string;
      gpuIndices: number[];
      tpSize: number;
      dtype: string;
      maxBatchSize: number;
      maxSeqLen: number;
    }): Promise<VramEstimate> => {
      const qs = new URLSearchParams();
      qs.set("model_id", params.modelId);
      qs.set("tp_size", String(params.tpSize));
      qs.set("dtype", params.dtype);
      qs.set("max_batch_size", String(params.maxBatchSize));
      qs.set("max_seq_len", String(params.maxSeqLen));
      params.gpuIndices.forEach((gpuIndex) =>
        qs.append("gpu_indices", String(gpuIndex)),
      );
      const raw = await request<Record<string, unknown>>(
        "GET",
        `/ocabra/trtllm/estimate?${qs}`,
      );
      return toVramEstimate(raw);
    },
  },

};
