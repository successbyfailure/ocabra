// Base types mirroring docs/CONTRACTS.md
// Stream 1-D expands these. All streams consume them.

export interface GPUProcessInfo {
  pid: number
  processName: string | null
  processType: "compute" | "graphics"
  usedVramMb: number
}

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
  processes: GPUProcessInfo[]
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
export type BackendType =
  | "vllm"
  | "bitnet"
  | "acestep"
  | "diffusers"
  | "whisper"
  | "tts"
  | "ollama"
  | "tensorrt_llm"
  | "llama_cpp"
  | "sglang"

export interface ModelCapabilities {
  chat: boolean
  completion: boolean
  tools: boolean
  vision: boolean
  embeddings: boolean
  pooling: boolean
  rerank: boolean
  classification: boolean
  score: boolean
  reasoning: boolean
  imageGeneration: boolean
  audioTranscription: boolean
  musicGeneration: boolean
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
  diskSizeBytes: number | null
  capabilities: ModelCapabilities
  lastRequestAt: string | null
  loadedAt: string | null
  errorMessage?: string | null
  schedules?: EvictionSchedule[]
  extraConfig?: BackendExtraConfig
}

export interface ModelsStorageStats {
  path: string
  totalBytes: number
  usedBytes: number
  freeBytes: number
}

export interface VLLMConfig {
  recipeId?: string | null
  recipeNotes?: string[]
  recipeModelImpl?: "auto" | "vllm" | "transformers" | null
  recipeRunner?: "generate" | "pooling" | null
  suggestedConfig?: Record<string, unknown>
  suggestedTuning?: Record<string, unknown>
  probeStatus?: HFVLLMRuntimeProbe["status"] | null
  probeReason?: string | null
  probeObservedAt?: string | null
  probeRecommendedModelImpl?: HFVLLMRuntimeProbe["recommendedModelImpl"] | null
  probeRecommendedRunner?: HFVLLMRuntimeProbe["recommendedRunner"] | null
  modelImpl?: "auto" | "vllm" | "transformers" | null
  runner?: "generate" | "pooling" | null
  hfOverrides?: string | Record<string, unknown> | null
  chatTemplate?: string | null
  chatTemplateContentFormat?: string | null
  generationConfig?: string | null
  overrideGenerationConfig?: string | Record<string, unknown> | null
  toolCallParser?: string | null
  toolParserPlugin?: string | null
  reasoningParser?: string | null
  languageModelOnly?: boolean | null
  maxNumSeqs?: number | null
  maxNumBatchedTokens?: number | null
  tensorParallelSize?: number | null
  maxModelLen?: number | null
  gpuMemoryUtilization?: number | null
  enablePrefixCaching?: boolean
  enableChunkedPrefill?: boolean | null
  enforceEager?: boolean
  trustRemoteCode?: boolean
  swapSpace?: number | null
  kvCacheDtype?: string | null
}

export interface SGLangConfig {
  tensorParallelSize?: number | null
  memFractionStatic?: number | null
  contextLength?: number | null
  trustRemoteCode?: boolean
  disableRadixCache?: boolean
}

export interface LlamaCppConfig {
  gpuLayers?: number | null
  ctxSize?: number | null
  flashAttn?: boolean
  embedding?: boolean
}

export interface BitNetConfig {
  gpuLayers?: number | null
  ctxSize?: number | null
  flashAttn?: boolean
}

export interface TensorRTLLMConfig {
  maxBatchSize?: number | null
  contextLength?: number | null
  trustRemoteCode?: boolean
}

export interface WhisperConfig {
  diarizationEnabled?: boolean
  diarizationModelId?: string | null
}

export interface ModelMemoryEstimate {
  backendType: BackendType
  gpuIndex: number | null
  totalVramMb: number | null
  freeVramMb: number | null
  budgetVramMb: number | null
  requestedContextLength: number | null
  estimatedWeightsMb: number | null
  estimatedEngineMbPerGpu: number | null
  estimatedKvCacheMb: number | null
  estimatedMaxContextLength: number | null
  modelLoadingMemoryMb: number | null
  maximumConcurrency: number | null
  tensorParallelSize: number | null
  fitsCurrentGpu: boolean | null
  enginePresent: boolean | null
  source: "heuristic" | "runtime_probe"
  status: "ok" | "warning" | "error"
  warning: string | null
  notes: string[]
}

export type BackendExtraConfig = Record<string, unknown> & {
  vllm?: VLLMConfig
  sglang?: SGLangConfig
  llama_cpp?: LlamaCppConfig
  bitnet?: BitNetConfig
  tensorrt_llm?: TensorRTLLMConfig
  whisper?: WhisperConfig
}

export interface HFVLLMRuntimeProbe {
  status:
    | "supported_native"
    | "supported_transformers_backend"
    | "supported_pooling"
    | "needs_remote_code"
    | "missing_chat_template"
    | "missing_tool_parser"
    | "missing_reasoning_parser"
    | "needs_hf_overrides"
    | "unsupported_tokenizer"
    | "unsupported_architecture"
    | "unavailable"
    | "unknown"
  reason: string | null
  recommendedModelImpl: "auto" | "vllm" | "transformers" | null
  recommendedRunner: "generate" | "pooling" | null
  tokenizerLoad: boolean | null
  configLoad: boolean | null
  observedAt: string | null
}

export interface HFVLLMSupport {
  classification: "native_vllm" | "transformers_backend" | "pooling" | "unsupported" | "unknown"
  label: string
  modelImpl: "auto" | "vllm" | "transformers" | null
  runner: "generate" | "pooling" | null
  taskMode: "generate" | "multimodal_generate" | "pooling" | "multimodal_pooling" | null
  requiredOverrides: string[]
  recipeId: string | null
  recipeNotes: string[]
  recipeModelImpl: "auto" | "vllm" | "transformers" | null
  recipeRunner: "generate" | "pooling" | null
  suggestedConfig: Record<string, unknown>
  suggestedTuning: Record<string, unknown>
  runtimeProbe: HFVLLMRuntimeProbe | null
}

export interface ModelPatchRequest {
  loadPolicy?: LoadPolicy
  preferredGpu?: number | null
  autoReload?: boolean
  displayName?: string
  schedules?: EvictionSchedule[]
  extraConfig?: BackendExtraConfig
}

export interface EvictionSchedule {
  id: string
  days: number[]
  start: string
  end: string
  enabled: boolean
}

export interface DownloadJob {
  jobId: string
  source: "huggingface" | "ollama" | "bitnet"
  modelRef: string
  artifact?: string | null
  registerConfig?: {
    displayName?: string
    loadPolicy?: LoadPolicy
    autoReload?: boolean
    preferredGpu?: number | null
    extraConfig?: BackendExtraConfig
  } | null
  status: "queued" | "downloading" | "completed" | "failed" | "cancelled"
  progressPct: number
  speedMbS: number | null
  etaSeconds: number | null
  error: string | null
  startedAt: string
  completedAt: string | null
}

export type DownloadSource = "huggingface" | "ollama" | "bitnet"
export type DownloadStatus = "queued" | "downloading" | "completed" | "failed" | "cancelled"

export interface HFModelCard {
  repoId: string
  modelName: string
  task: string | null
  downloads: number
  likes: number
  sizeGb: number | null
  tags: string[]
  gated: boolean
  suggestedBackend: BackendType
  compatibility?: string
  compatibilityReason?: string | null
  vllmSupport?: HFVLLMSupport | null
}

export interface HFModelDetail extends HFModelCard {
  siblings: Record<string, unknown>[]
  readmeExcerpt: string | null
  suggestedBackend: BackendType
  estimatedVramGb: number | null
  vllmSupport?: HFVLLMSupport | null
}

export interface HFModelVariant {
  variantId: string
  label: string
  artifact: string | null
  sizeGb: number | null
  format: string
  quantization: string | null
  backendType: BackendType
  isDefault: boolean
  installable?: boolean
  compatibility?: string
  compatibilityReason?: string | null
  vllmSupport?: HFVLLMSupport | null
}

export interface OllamaModelCard {
  name: string
  description: string
  tags: string[]
  sizeGb: number | null
  pulls: number
}

export interface OllamaModelVariant {
  name: string
  tag: string
  sizeGb: number | null
  parameterSize: string | null
  quantization: string | null
  contextWindow: string | null
  modality: string | null
  updatedHint: string | null
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
  idleEvictionCheckIntervalSeconds: number
  modelLoadWaitTimeoutSeconds: number
  pressureEvictionDrainTimeoutSeconds: number
  vramBufferMb: number
  vramPressureThresholdPct: number
  openaiAudioMaxPartSizeMb: number
  whisperStartupTimeoutSeconds: number
  logLevel: string
  litellmBaseUrl: string
  litellmAdminKey: string
  litellmAutoSync: boolean
  energyCostEurKwh: number
  modelsDir: string
  downloadDir: string
  maxTemperatureC: number
  vllmGpuMemoryUtilization: number
  vllmMaxNumSeqs: number | null
  vllmMaxNumBatchedTokens: number | null
  vllmEnablePrefixCaching: boolean
  vllmEnforceEager: boolean
  sglangMemFractionStatic: number
  sglangContextLength: number | null
  sglangDisableRadixCache: boolean
  llamaCppGpuLayers: number
  llamaCppCtxSize: number
  llamaCppFlashAttn: boolean
  bitnetGpuLayers: number
  bitnetCtxSize: number
  bitnetFlashAttn: boolean
  diffusersTorchDtype: string
  diffusersOffloadMode: string
  diffusersEnableTorchCompile: boolean
  diffusersEnableXformers: boolean
  diffusersAllowTf32: boolean
  tensorrtLlmEnabled: boolean
  tensorrtLlmMaxBatchSize: number | null
  tensorrtLlmContextLength: number | null
  globalSchedules: EvictionSchedule[]
  requireApiKeyOpenai: boolean
  requireApiKeyOllama: boolean
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
    backendType: string
    requestKinds: string[]
    totalRequests: number
    avgLatencyMs: number
    p95LatencyMs: number
    requestsPerMinute: number
    tokensPerSecond: number
    totalInputTokens: number
    totalOutputTokens: number
    tokenizedRequests: number
    errorCount: number
    uptimePct: number
    loadCount: number
    avgLoadMs: number
    p95LoadMs: number
    lastLoadMs: number
  }[]
}

export interface TokenStats {
  totalInputTokens: number
  totalOutputTokens: number
  byBackend: { backendType: string; inputTokens: number; outputTokens: number }[]
  series: { timestamp: string; inputTokens: number; outputTokens: number }[]
}

export interface OverviewStats {
  totalRequests: number
  totalErrors: number
  avgDurationMs: number
  tokenizedRequests: number
  totalInputTokens: number
  totalOutputTokens: number
  byBackend: {
    backendType: string
    totalRequests: number
    errorRate: number
    avgLatencyMs: number
    p95LatencyMs: number
  }[]
  byRequestKind: {
    requestKind: string
    totalRequests: number
    errorRate: number
    avgLatencyMs: number
    p95LatencyMs: number
  }[]
}

export interface GPUStatHistoryPoint {
  timestamp: string
  usedVramMb: number
  utilizationPct: number
  temperatureC: number
  powerDrawW: number
}

export interface GPUStatHistory {
  gpuIndex: number
  window: string
  points: GPUStatHistoryPoint[]
}

export type ServiceStatus = "unknown" | "idle" | "active" | "unreachable" | "disabled"

export interface ServiceState {
  serviceId: string
  serviceType: string
  displayName: string
  uiUrl: string
  preferredGpu: number | null
  idleUnloadAfterSeconds: number
  enabled: boolean
  serviceAlive: boolean
  runtimeLoaded: boolean
  status: ServiceStatus
  activeModelRef: string | null
  lastActivityAt: string | null
  lastHealthCheckAt: string | null
  detail: string | null
  // Generation metrics
  isGenerating: boolean
  queueDepth: number
  vramUsedMb: number | null
  gpuUtilPct: number | null
  // Container resource metrics
  cpuPct: number | null
  memUsedMb: number | null
  memLimitMb: number | null
}

export interface HostStats {
  cpuPct: number
  cpuCount: number
  cpuCountPhysical: number
  loadAvg1m: number
  loadAvg5m: number
  loadAvg15m: number
  memTotalMb: number
  memUsedMb: number
  memFreeMb: number
  memPct: number
  swapTotalMb: number
  swapUsedMb: number
  swapPct: number
}

// TRT-LLM compile jobs
export type CompileStatus = "pending" | "running" | "done" | "failed" | "cancelled"
export type CompilePhase = "convert" | "build" | null

export interface CompileJob {
  jobId: string
  sourceModel: string
  engineName: string
  gpuIndices: number[]
  dtype: string
  config: {
    maxBatchSize: number
    maxInputLen: number
    maxSeqLen: number
  }
  status: CompileStatus
  phase: CompilePhase
  progressPct: number
  errorDetail: string | null
  engineDir: string | null
  startedAt: string | null
  finishedAt: string | null
}

// Auth types
export type UserRole = "user" | "model_manager" | "system_admin"

export interface AuthUser {
  id: string
  username: string
  email: string | null
  role: UserRole
  createdAt: string
}

export interface ApiKey {
  id: string
  name: string
  keyPrefix: string
  expiresAt: string | null
  lastUsedAt: string | null
  isRevoked: boolean
  createdAt: string
}

export interface Group {
  id: string
  name: string
  description: string
  isDefault: boolean
  memberCount: number
  modelCount: number
  createdAt: string
}

// Admin user management
export interface AdminUser {
  id: string
  username: string
  email: string | null
  role: UserRole
  isActive: boolean
  createdAt: string
}

export interface GroupMember {
  userId: string
  username: string
  role: UserRole
}

export interface CreateUserPayload {
  username: string
  password: string
  role: UserRole
  email?: string | null
}

export interface UpdateUserPayload {
  role?: UserRole
  isActive?: boolean
  email?: string | null
}

export interface CreateGroupPayload {
  name: string
  description?: string | null
}

export interface UpdateGroupPayload {
  name?: string
  description?: string | null
}

// WebSocket events
export type WSEvent =
  | { type: "gpu_stats"; data: GPUState[] }
  | { type: "model_event"; data: { event: string; modelId: string; status: ModelStatus } }
  | { type: "download_progress"; data: { jobId: string; pct: number; speedMbS: number } }
  | { type: "service_event"; data: { event: string; serviceId: string; status: ServiceStatus; service: ServiceState } }
  | { type: "system_alert"; data: { level: "warn" | "error"; message: string } }
