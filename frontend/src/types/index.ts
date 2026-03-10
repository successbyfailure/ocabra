// Base types mirroring docs/CONTRACTS.md
// Stream 1-D expands these. All streams consume them.

export interface GPUState {
  index: number
  name: string
  totalVramMb: number
  freeVramMb: number
  usedVramMb: number
  utilizationPct: number
  temperatureC: number
  powerDrawW: number
  powerLimitW: number
  lockedVramMb: number
}

export type ModelStatus =
  | "discovered"
  | "configured"
  | "loading"
  | "loaded"
  | "unloading"
  | "unloaded"
  | "error"

export type LoadPolicy = "pin" | "warm" | "on_demand"
export type BackendType = "vllm" | "diffusers" | "whisper" | "tts"

export interface ModelCapabilities {
  chat: boolean
  completion: boolean
  tools: boolean
  vision: boolean
  embeddings: boolean
  reasoning: boolean
  imageGeneration: boolean
  audioTranscription: boolean
  tts: boolean
  streaming: boolean
  contextLength: number
}

export interface ModelState {
  modelId: string
  displayName: string
  backendType: BackendType
  status: ModelStatus
  loadPolicy: LoadPolicy
  autoReload: boolean
  preferredGpu: number | null
  currentGpu: number[]
  vramUsedMb: number
  capabilities: ModelCapabilities
  lastRequestAt: string | null
  loadedAt: string | null
}

export interface DownloadJob {
  jobId: string
  source: "huggingface" | "ollama"
  modelRef: string
  status: "queued" | "downloading" | "completed" | "failed" | "cancelled"
  progressPct: number
  speedMbS: number | null
  etaSeconds: number | null
  error: string | null
  startedAt: string
  completedAt: string | null
}

export interface HFModelCard {
  repoId: string
  modelName: string
  task: string | null
  downloads: number
  likes: number
  sizeGb: number | null
  tags: string[]
  gated: boolean
}

export interface HFModelDetail extends HFModelCard {
  siblings: Record<string, unknown>[]
  readmeExcerpt: string | null
  suggestedBackend: BackendType
  estimatedVramGb: number | null
}

export interface OllamaModelCard {
  name: string
  description: string
  tags: string[]
  sizeGb: number | null
  pulls: number
}

export interface LocalModel {
  modelId: string
  path: string
  sizeGb: number
  backendType: BackendType
  configured: boolean
}

export interface ServerConfig {
  defaultGpuIndex: number
  idleTimeoutSeconds: number
  vramBufferMb: number
  vramPressureThresholdPct: number
  logLevel: string
  litellmBaseUrl: string
  litellmAdminKey: string
  litellmAutoSync: boolean
  energyCostEurKwh: number
}

export interface StatsParams {
  from?: string
  to?: string
  modelId?: string
}

export interface RequestStats {
  totalRequests: number
  errorRate: number
  avgDurationMs: number
  p50DurationMs: number
  p95DurationMs: number
  series: { timestamp: string; count: number }[]
}

export interface EnergyStats {
  totalKwh: number
  estimatedCostEur: number
  byGpu: { gpuIndex: number; totalKwh: number; powerDrawW: number }[]
}

export interface PerformanceStats {
  byModel: {
    modelId: string
    totalRequests: number
    avgLatencyMs: number
    tokensPerSecond: number
    errorCount: number
    uptimePct: number
  }[]
}

// WebSocket events
export type WSEvent =
  | { type: "gpu_stats"; data: GPUState[] }
  | { type: "model_event"; data: { event: string; modelId: string; status: ModelStatus } }
  | { type: "download_progress"; data: { jobId: string; pct: number; speedMbS: number } }
  | { type: "system_alert"; data: { level: "warn" | "error"; message: string } }
