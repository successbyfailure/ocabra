// Typed API client for oCabra internal API (/ocabra/*)
// Stream 1-D fills in the method implementations.
// Streams 4-A/B/C/D consume this client.

import type {
  GPUState,
  ModelState,
  DownloadJob,
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
  return res.json() as Promise<T>
}

export const api = {
  gpus: {
    list: () => request<GPUState[]>("GET", "/ocabra/gpus"),
    get: (index: number) => request<GPUState>("GET", `/ocabra/gpus/${index}`),
  },

  models: {
    list: () => request<ModelState[]>("GET", "/ocabra/models"),
    get: (modelId: string) => request<ModelState>("GET", `/ocabra/models/${encodeURIComponent(modelId)}`),
    load: (modelId: string) => request<ModelState>("POST", `/ocabra/models/${encodeURIComponent(modelId)}/load`),
    unload: (modelId: string) => request<void>("POST", `/ocabra/models/${encodeURIComponent(modelId)}/unload`),
    patch: (modelId: string, patch: Partial<ModelState>) =>
      request<ModelState>("PATCH", `/ocabra/models/${encodeURIComponent(modelId)}`, patch),
    delete: (modelId: string) => request<void>("DELETE", `/ocabra/models/${encodeURIComponent(modelId)}`),
  },

  downloads: {
    list: () => request<DownloadJob[]>("GET", "/ocabra/downloads"),
    enqueue: (source: "huggingface" | "ollama", modelRef: string) =>
      request<DownloadJob>("POST", "/ocabra/downloads", { source, model_ref: modelRef }),
    cancel: (jobId: string) => request<void>("DELETE", `/ocabra/downloads/${jobId}`),
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
    requests: (params: StatsParams) =>
      request<RequestStats>("GET", `/ocabra/stats/requests?${new URLSearchParams(params as Record<string, string>)}`),
    energy: (params: StatsParams) =>
      request<EnergyStats>("GET", `/ocabra/stats/energy?${new URLSearchParams(params as Record<string, string>)}`),
    performance: (modelId?: string) => {
      const params = modelId ? `?model_id=${encodeURIComponent(modelId)}` : ""
      return request<PerformanceStats>("GET", `/ocabra/stats/performance${params}`)
    },
  },

  config: {
    get: () => request<ServerConfig>("GET", "/ocabra/config"),
    patch: (patch: Partial<ServerConfig>) => request<ServerConfig>("PATCH", "/ocabra/config", patch),
    syncLiteLLM: () => request<{ synced_models: number }>("POST", "/ocabra/config/litellm/sync"),
  },
}
