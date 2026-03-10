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
    list: () => request<GPUState[]>("GET", "/ocabra/gpus"),
    get: (index: number) => request<GPUState>("GET", `/ocabra/gpus/${index}`),
    stats: (index: number, window = "5m") =>
      request<GPUStatHistory>("GET", `/ocabra/gpus/${index}/stats${buildQuery({ window })}`),
  },

  models: {
    list: () => request<ModelState[]>("GET", "/ocabra/models"),
    get: (modelId: string) => request<ModelState>("GET", `/ocabra/models/${encodeURIComponent(modelId)}`),
    load: (modelId: string) => request<ModelState>("POST", `/ocabra/models/${encodeURIComponent(modelId)}/load`),
    unload: (modelId: string) => request<void>("POST", `/ocabra/models/${encodeURIComponent(modelId)}/unload`),
    patch: (modelId: string, patch: ModelPatchRequest) =>
      request<ModelState>("PATCH", `/ocabra/models/${encodeURIComponent(modelId)}`, patch),
    delete: (modelId: string) => request<void>("DELETE", `/ocabra/models/${encodeURIComponent(modelId)}`),
  },

  downloads: {
    list: () => request<DownloadJob[]>("GET", "/ocabra/downloads"),
    enqueue: (source: DownloadSource, modelRef: string) =>
      request<DownloadJob>("POST", "/ocabra/downloads", { source, model_ref: modelRef }),
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
    get: () => request<ServerConfig>("GET", "/ocabra/config"),
    patch: (patch: Partial<ServerConfig>) => request<ServerConfig>("PATCH", "/ocabra/config", patch),
    syncLiteLLM: async () => {
      const result = await request<{ synced_models: number }>("POST", "/ocabra/config/litellm/sync")
      return { syncedModels: result.synced_models }
    },
  },
}
