// Typed API client for oCabra internal API (/ocabra/*)
// Stream 1-D fills in the method implementations.
// Streams 4-A/B/C/D consume this client.

import type {
  GPUState,
  GPUStatHistory,
  ModelState,
  ModelPatchRequest,
  DownloadJob,
  DownloadSource,
  HFModelCard,
  HFModelDetail,
  OllamaModelCard,
  LocalModel,
  ServerConfig,
  RequestStats,
  EnergyStats,
  PerformanceStats,
  StatsParams,
} from "@/types"

const BASE = ""

type AnyRecord = Record<string, unknown>

function isRecord(value: unknown): value is AnyRecord {
  return typeof value === "object" && value !== null
}

function toModelCapabilities(raw: unknown): ModelState["capabilities"] {
  const data = isRecord(raw) ? raw : {}
  return {
    chat: Boolean(data.chat),
    completion: Boolean(data.completion),
    tools: Boolean(data.tools),
    vision: Boolean(data.vision),
    embeddings: Boolean(data.embeddings),
    reasoning: Boolean(data.reasoning),
    imageGeneration: Boolean(data.image_generation ?? data.imageGeneration),
    audioTranscription: Boolean(data.audio_transcription ?? data.audioTranscription),
    tts: Boolean(data.tts),
    streaming: Boolean(data.streaming),
    contextLength: Number(data.context_length ?? data.contextLength ?? 0),
  }
}

function toGpuState(raw: unknown): GPUState {
  const data = isRecord(raw) ? raw : {}
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
  }
}

function toModelState(raw: unknown): ModelState {
  const data = isRecord(raw) ? raw : {}
  return {
    modelId: String(data.model_id ?? data.modelId ?? ""),
    displayName: String(data.display_name ?? data.displayName ?? data.model_id ?? data.modelId ?? "Unknown model"),
    backendType: String(data.backend_type ?? data.backendType ?? "vllm") as ModelState["backendType"],
    status: String(data.status ?? "configured") as ModelState["status"],
    loadPolicy: String(data.load_policy ?? data.loadPolicy ?? "on_demand") as ModelState["loadPolicy"],
    autoReload: Boolean(data.auto_reload ?? data.autoReload),
    preferredGpu: (data.preferred_gpu ?? data.preferredGpu ?? null) as number | null,
    currentGpu: Array.isArray(data.current_gpu ?? data.currentGpu)
      ? ((data.current_gpu ?? data.currentGpu) as unknown[]).map((gpu) => Number(gpu))
      : [],
    vramUsedMb: Number(data.vram_used_mb ?? data.vramUsedMb ?? 0),
    capabilities: toModelCapabilities(data.capabilities),
    lastRequestAt: (data.last_request_at ?? data.lastRequestAt ?? null) as string | null,
    loadedAt: (data.loaded_at ?? data.loadedAt ?? null) as string | null,
  }
}

function toDownloadJob(raw: unknown): DownloadJob {
  const data = isRecord(raw) ? raw : {}
  return {
    jobId: String(data.job_id ?? data.jobId ?? ""),
    source: String(data.source ?? "huggingface") as DownloadSource,
    modelRef: String(data.model_ref ?? data.modelRef ?? ""),
    status: String(data.status ?? "queued") as DownloadJob["status"],
    progressPct: Number(data.progress_pct ?? data.progressPct ?? 0),
    speedMbS: data.speed_mb_s === null || data.speedMbS === null ? null : Number(data.speed_mb_s ?? data.speedMbS ?? 0),
    etaSeconds: data.eta_seconds === null || data.etaSeconds === null ? null : Number(data.eta_seconds ?? data.etaSeconds ?? 0),
    error: (data.error ?? null) as string | null,
    startedAt: String(data.started_at ?? data.startedAt ?? new Date(0).toISOString()),
    completedAt: (data.completed_at ?? data.completedAt ?? null) as string | null,
  }
}

function toServerConfig(raw: unknown): ServerConfig {
  const data = isRecord(raw) ? raw : {}
  return {
    defaultGpuIndex: Number(data.default_gpu_index ?? data.defaultGpuIndex ?? 0),
    idleTimeoutSeconds: Number(data.idle_timeout_seconds ?? data.idleTimeoutSeconds ?? 0),
    vramBufferMb: Number(data.vram_buffer_mb ?? data.vramBufferMb ?? 0),
    vramPressureThresholdPct: Number(data.vram_pressure_threshold_pct ?? data.vramPressureThresholdPct ?? 0),
    logLevel: String(data.log_level ?? data.logLevel ?? "info"),
    litellmBaseUrl: String(data.litellm_base_url ?? data.litellmBaseUrl ?? ""),
    litellmAdminKey: String(data.litellm_admin_key ?? data.litellmAdminKey ?? ""),
    litellmAutoSync: Boolean(data.litellm_auto_sync ?? data.litellmAutoSync),
    energyCostEurKwh: Number(data.energy_cost_eur_kwh ?? data.energyCostEurKwh ?? 0),
  }
}

async function request<T>(
  method: string,
  path: string,
  body?: unknown,
): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method,
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail ?? res.statusText)
  }
  if (res.status === 204) {
    return undefined as T
  }
  return res.json() as Promise<T>
}

function buildQuery(params: Record<string, string | number | undefined>): string {
  const search = new URLSearchParams()
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== "") {
      search.set(key, String(value))
    }
  })
  const query = search.toString()
  return query ? `?${query}` : ""
}

export const api = {
  gpus: {
    list: async () => (await request<unknown[]>("GET", "/ocabra/gpus")).map(toGpuState),
    get: async (index: number) => toGpuState(await request<unknown>("GET", `/ocabra/gpus/${index}`)),
    stats: (index: number, window = "5m") =>
      request<GPUStatHistory>("GET", `/ocabra/gpus/${index}/stats${buildQuery({ window })}`),
  },

  models: {
    list: async () => (await request<unknown[]>("GET", "/ocabra/models")).map(toModelState),
    get: async (modelId: string) =>
      toModelState(await request<unknown>("GET", `/ocabra/models/${encodeURIComponent(modelId)}`)),
    load: async (modelId: string) =>
      toModelState(await request<unknown>("POST", `/ocabra/models/${encodeURIComponent(modelId)}/load`)),
    unload: (modelId: string) => request<void>("POST", `/ocabra/models/${encodeURIComponent(modelId)}/unload`),
    patch: (modelId: string, patch: ModelPatchRequest) =>
      request<unknown>("PATCH", `/ocabra/models/${encodeURIComponent(modelId)}`, patch).then(toModelState),
    delete: (modelId: string) => request<void>("DELETE", `/ocabra/models/${encodeURIComponent(modelId)}`),
  },

  downloads: {
    list: async () => (await request<unknown[]>("GET", "/ocabra/downloads")).map(toDownloadJob),
    enqueue: (source: DownloadSource, modelRef: string) =>
      request<unknown>("POST", "/ocabra/downloads", { source, model_ref: modelRef }).then(toDownloadJob),
    cancel: (jobId: string) => request<void>("DELETE", `/ocabra/downloads/${encodeURIComponent(jobId)}`),
    streamProgress: (jobId: string): EventSource =>
      new EventSource(`/ocabra/downloads/${jobId}/stream`),
  },

  registry: {
    searchHF: (q: string, task?: string, limit = 20) => {
      const params = new URLSearchParams({ q, limit: String(limit) })
      if (task) params.set("task", task)
      return request<HFModelCard[]>("GET", `/ocabra/registry/hf/search?${params}`)
    },
    getHFDetail: (repoId: string) =>
      request<HFModelDetail>("GET", `/ocabra/registry/hf/${encodeURIComponent(repoId)}`),
    searchOllama: (q: string) =>
      request<OllamaModelCard[]>("GET", `/ocabra/registry/ollama/search?q=${encodeURIComponent(q)}`),
    listLocal: () => request<LocalModel[]>("GET", "/ocabra/registry/local"),
  },

  stats: {
    requests: (params: StatsParams) => {
      const query = buildQuery({ from: params.from, to: params.to, model_id: params.modelId })
      return request<RequestStats>("GET", `/ocabra/stats/requests${query}`)
    },
    energy: (params: StatsParams) => {
      const query = buildQuery({ from: params.from, to: params.to })
      return request<EnergyStats>("GET", `/ocabra/stats/energy${query}`)
    },
    performance: (modelId?: string) => {
      const query = buildQuery({ model_id: modelId })
      return request<PerformanceStats>("GET", `/ocabra/stats/performance${query}`)
    },
  },

  config: {
    get: async () => toServerConfig(await request<unknown>("GET", "/ocabra/config")),
    patch: async (patch: Partial<ServerConfig>) =>
      toServerConfig(await request<unknown>("PATCH", "/ocabra/config", patch)),
    syncLiteLLM: async () => {
      const result = await request<{ synced_models: number }>("POST", "/ocabra/config/litellm/sync")
      return { syncedModels: result.synced_models }
    },
  },
}
