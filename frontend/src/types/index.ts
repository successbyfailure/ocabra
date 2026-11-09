// Base types mirroring docs/CONTRACTS.md
// Stream 1-D expands these. All streams consume them.

export type ChatAudioFormat = "wav" | "mp3" | "m4a" | "webm" | "ogg" | "flac";

export interface ChatTextPart {
  type: "text";
  text: string;
}

export interface ChatImageUrlPart {
  type: "image_url";
  image_url: { url: string };
}

export interface ChatInputAudioPart {
  type: "input_audio";
  input_audio: { data: string; format: ChatAudioFormat };
}

export type ChatContentPart =
  ChatTextPart | ChatImageUrlPart | ChatInputAudioPart;

export interface ChatRequestMessage {
  role: "system" | "user" | "assistant";
  content: string | ChatContentPart[];
}

export interface GPUProcessInfo {
  pid: number;
  processName: string | null;
  processType: "compute" | "graphics";
  usedVramMb: number;
}

export interface GPUState {
  index: number;
  name: string;
  totalVramMb: number;
  freeVramMb: number;
  usedVramMb: number;
  utilizationPct: number;
  temperatureC: number;
  powerDrawW: number;
  powerLimitW: number;
  lockedVramMb: number;
  processes: GPUProcessInfo[];
}

export type ModelStatus =
  | "discovered"
  | "configured"
  | "loading"
  | "loaded"
  | "unloading"
  | "unloaded"
  | "error";

export type RuntimePhase = ModelStatus | "upsizing" | "downsizing";

export type LoadPolicy = "pin" | "warm" | "on_demand";
export type BackendType =
  | "vllm"
  | "bitnet"
  | "acestep"
  | "diffusers"
  | "mage"
  | "whisper"
  | "whisper_cpp"
  | "vibeasr"
  | "tts"
  | "chatterbox"
  | "piper"
  | "upscaler"
  | "flashvsr"
  | "voxtral"
  | "ollama"
  | "tensorrt_llm"
  | "llama_cpp"
  | "llamacpp_mks"
  | "sglang";

/**
 * Backends that run a llama.cpp server: upstream and the Makespace fork.
 * Separate backend types so both can be installed and compared A/B, but UI
 * that keys off "this is llama.cpp" must treat them alike.
 *
 * The fork is named `llamacpp_mks` — no underscore after "llama" — on
 * purpose: upstream's build variants are `llama_cpp_<cuda|rocm|vulkan|cpu>`,
 * so a `llama_cpp_mks` would have been parsed as a runtime called "mks" by
 * code that splits on the suffix. The spelling keeps the two namespaces
 * disjoint, but it also means `startsWith("llama_cpp")` does not match the
 * fork: use this helper instead.
 */
export const LLAMA_CPP_FAMILY: readonly BackendType[] = ["llama_cpp", "llamacpp_mks"];

export function isLlamaCppFamily(backendType: string | undefined | null): boolean {
  return backendType === "llama_cpp" || backendType === "llamacpp_mks";
}

export interface ModelCapabilities {
  chat: boolean;
  completion: boolean;
  tools: boolean;
  vision: boolean;
  embeddings: boolean;
  pooling: boolean;
  rerank: boolean;
  classification: boolean;
  score: boolean;
  reasoning: boolean;
  imageGeneration: boolean;
  imageEditing: boolean;
  audioTranscription: boolean;
  musicGeneration: boolean;
  tts: boolean;
  audioInput: boolean;
  audioOutput: boolean;
  videoInput: boolean;
  streaming: boolean;
  contextLength: number;
  nativeContextLength: number;
}

export interface ModelState {
  modelId: string;
  displayName: string;
  backendType: BackendType;
  status: ModelStatus;
  loadPolicy: LoadPolicy;
  /** Only meaningful when loadPolicy === "pin". False (default, "soft pin"):
   * preloaded at startup, never evicted by the routine LRU pass, but still
   * evictable as an absolute last resort under VRAM pressure (and reloads
   * automatically once room frees up). True ("hard pin"): excluded from
   * that last-resort path too — only an explicit admin unload moves it. */
  pinPermanent: boolean;
  preferredGpu: number | null;
  currentGpu: number[];
  vramUsedMb: number;
  /** Live state of the multimodal projector, for backends that can load and
   *  drop it at runtime (the Makespace llama.cpp fork). "active" | "inactive",
   *  or null when the question does not apply — supporting vision and having
   *  it currently loaded are different things there. */
  visionState?: "active" | "inactive" | null;
  /** VRAM the scheduler reserved for this load. vramUsedMb is
   *  max(measured, reserved), so a measured value above this means the model
   *  is running on borrowed margin. 0 when not loaded. */
  vramEstimateMb?: number;
  ramUsedMb: number;
  diskSizeBytes: number | null;
  capabilities: ModelCapabilities;
  lastRequestAt: string | null;
  loadedAt: string | null;
  errorMessage?: string | null;
  runtimePhase?: RuntimePhase;
  runtimeWorkerId?: string | null;
  runtimeSlots?: number;
  canCancelLoad?: boolean;
  canDownsize?: boolean;
  schedules?: EvictionSchedule[];
  extraConfig?: BackendExtraConfig;
  profiles?: ModelProfile[];
  createdAt?: string | null;
  updatedAt?: string | null;
  releaseDate?: string | null;
  lastUpdated?: string | null;
  metadataFetchedAt?: string | null;
  federation?: {
    remote: boolean;
    nodeName: string;
    nodeId: string;
  } | null;
}

export interface ModelsStorageStats {
  path: string;
  totalBytes: number;
  usedBytes: number;
  freeBytes: number;
}

export interface VLLMConfig {
  recipeId?: string | null;
  recipeNotes?: string[];
  recipeModelImpl?: "auto" | "vllm" | "transformers" | null;
  recipeRunner?: "generate" | "pooling" | null;
  suggestedConfig?: Record<string, unknown>;
  suggestedTuning?: Record<string, unknown>;
  probeStatus?: HFVLLMRuntimeProbe["status"] | null;
  probeReason?: string | null;
  probeObservedAt?: string | null;
  probeRecommendedModelImpl?: HFVLLMRuntimeProbe["recommendedModelImpl"] | null;
  probeRecommendedRunner?: HFVLLMRuntimeProbe["recommendedRunner"] | null;
  modelImpl?: "auto" | "vllm" | "transformers" | null;
  runner?: "generate" | "pooling" | null;
  hfOverrides?: string | Record<string, unknown> | null;
  chatTemplate?: string | null;
  chatTemplateContentFormat?: string | null;
  generationConfig?: string | null;
  overrideGenerationConfig?: string | Record<string, unknown> | null;
  toolCallParser?: string | null;
  toolParserPlugin?: string | null;
  reasoningParser?: string | null;
  languageModelOnly?: boolean | null;
  maxNumSeqs?: number | null;
  maxNumBatchedTokens?: number | null;
  tensorParallelSize?: number | null;
  maxModelLen?: number | null;
  gpuMemoryUtilization?: number | null;
  enablePrefixCaching?: boolean;
  enableChunkedPrefill?: boolean | null;
  enforceEager?: boolean;
  trustRemoteCode?: boolean;
  swapSpace?: number | null;
  kvCacheDtype?: string | null;
  limitMmPerPrompt?: { image?: number; audio?: number; video?: number } | null;
}

export interface SGLangConfig {
  tensorParallelSize?: number | null;
  memFractionStatic?: number | null;
  contextLength?: number | null;
  trustRemoteCode?: boolean;
  disableRadixCache?: boolean;
}

export type LlamaKvCacheType =
  "f16" | "q8_0" | "q5_1" | "q5_0" | "q4_1" | "q4_0";
export type LlamaCppSplitMode = "layer" | "row" | "none";
export type LlamaCppSplitStrategy = "evenly" | "favor_main" | "single";
export type LlamaCppRuntime = "cuda" | "rocm" | "vulkan" | "cpu";
export type LlamaCppVisionMode = "enabled" | "disabled" | "on_demand";

export interface SpeculativeConfig {
  // Optional: self-speculative MTP (qwen3.8, gemma4) has a spec type and no
  // external draft model.
  draftModelId?: string | null;
  specType?: string | null;
  draftN?: number | null;
  draftMin?: number | null;
  draftPMin?: number | null;
  [key: string]: unknown;
}

export interface LlamaCppConfig {
  gpuLayers?: number | null;
  ctxSize?: number | null;
  flashAttn?: boolean;
  embedding?: boolean;
  // Sprint 17.1 — Tier 1 advanced controls.
  threads?: number | null;
  batchSize?: number | null;
  ubatchSize?: number | null;
  mlock?: boolean;
  // Tri-state: undefined/null => use mmap (default); false => --no-mmap.
  mmap?: boolean | null;
  seed?: number | null;
  noKvOffload?: boolean;
  ropeFreqBase?: number | null;
  ropeFreqScale?: number | null;
  // Sprint 17.2 — KV cache quantization
  cacheTypeK?: LlamaKvCacheType | null;
  cacheTypeV?: LlamaKvCacheType | null;
  // Sprint 17.3: Multi-GPU + MoE CPU offload.
  mainGpu?: number | null;
  tensorSplit?: number[] | null;
  splitMode?: LlamaCppSplitMode | null;
  disabledGpus?: number[] | null;
  splitStrategy?: LlamaCppSplitStrategy | null;
  nCpuMoe?: number | null;
  overrideTensor?: string | null;
  // --- Sprint 17.4 ---
  speculative?: SpeculativeConfig | null;
  runtime?: LlamaCppRuntime | null;
  parallelSlots?: number | null;
  kvUnified?: boolean | null;
  kvUnifiedPerSlot?: number | null;
  contBatching?: boolean | null;
  keepAliveSeconds?: number | null;
  mmprojModelId?: string | null;
  visionMode?: LlamaCppVisionMode | null;
  jinja?: boolean | null;
  reasoning?: "on" | "off" | "auto" | null;
  reasoningFormat?: "none" | "deepseek" | "deepseek-legacy" | "auto" | null;
  reasoningBudget?: number | null;
}

export interface BitNetConfig {
  gpuLayers?: number | null;
  ctxSize?: number | null;
  flashAttn?: boolean;
}

export interface TensorRTLLMConfig {
  maxBatchSize?: number | null;
  contextLength?: number | null;
  trustRemoteCode?: boolean;
}

export interface WhisperConfig {
  diarizationEnabled?: boolean;
  diarizationModelId?: string | null;
  numWorkers?: number | null;
}

export interface ModelMemoryEstimate {
  backendType: BackendType;
  gpuIndex: number | null;
  totalVramMb: number | null;
  freeVramMb: number | null;
  budgetVramMb: number | null;
  requestedContextLength: number | null;
  estimatedWeightsMb: number | null;
  estimatedEngineMbPerGpu: number | null;
  estimatedKvCacheMb: number | null;
  estimatedMaxContextLength: number | null;
  modelLoadingMemoryMb: number | null;
  maximumConcurrency: number | null;
  tensorParallelSize: number | null;
  fitsCurrentGpu: boolean | null;
  enginePresent: boolean | null;
  source: "heuristic" | "runtime_probe" | "gguf_header";
  status: "ok" | "warning" | "error";
  warning: string | null;
  notes: string[];
}

// Raw shape of GET /ocabra/models/{id}/capacity (snake_case, passed through).
export interface ModelCapacity {
  model_id: string;
  backend_type: string;
  applicable: boolean;
  weights_mb: number;
  overhead_mb: number;
  kv_dtype: string;
  target_gpu: number;
  gpus: { index: number; total_mb: number; free_mb: number }[];
  note?: string;
  fits_free_now?: boolean;
  concurrency_label?: "slots" | "concurrency";
  arch?: {
    layers: number;
    n_kv_heads: number;
    key_length: number;
    value_length: number;
    native_context: number;
    kv_bytes_per_token: number;
    kv_mb_per_1k_tokens: number;
  };
  capacity?: {
    slots: number;
    max_context: number;
    max_context_capped: number;
  }[];
  vram_curve?: { context: number; vram_mb: number }[];
  validation?: {
    requested_context: number | string;
    slots: number;
    effective_context: number;
    max_context: number;
    fits: boolean;
    warnings: string[];
  };
}

export type BackendExtraConfig = Record<string, unknown> & {
  vllm?: VLLMConfig;
  sglang?: SGLangConfig;
  llama_cpp?: LlamaCppConfig;
  bitnet?: BitNetConfig;
  tensorrt_llm?: TensorRTLLMConfig;
  whisper?: WhisperConfig;
};

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
    | "unknown";
  reason: string | null;
  recommendedModelImpl: "auto" | "vllm" | "transformers" | null;
  recommendedRunner: "generate" | "pooling" | null;
  tokenizerLoad: boolean | null;
  configLoad: boolean | null;
  observedAt: string | null;
}

export interface HFVLLMSupport {
  classification:
    | "native_vllm"
    | "transformers_backend"
    | "pooling"
    | "unsupported"
    | "unknown";
  label: string;
  modelImpl: "auto" | "vllm" | "transformers" | null;
  runner: "generate" | "pooling" | null;
  taskMode:
    | "generate"
    | "multimodal_generate"
    | "pooling"
    | "multimodal_pooling"
    | null;
  requiredOverrides: string[];
  recipeId: string | null;
  recipeNotes: string[];
  recipeModelImpl: "auto" | "vllm" | "transformers" | null;
  recipeRunner: "generate" | "pooling" | null;
  suggestedConfig: Record<string, unknown>;
  suggestedTuning: Record<string, unknown>;
  runtimeProbe: HFVLLMRuntimeProbe | null;
}

export interface ModelPatchRequest {
  loadPolicy?: LoadPolicy;
  preferredGpu?: number | null;
  pinPermanent?: boolean;
  displayName?: string;
  schedules?: EvictionSchedule[];
  extraConfig?: BackendExtraConfig;
}

export interface EvictionSchedule {
  id: string;
  days: number[];
  start: string;
  end: string;
  enabled: boolean;
}

export interface DownloadJob {
  jobId: string;
  source: "huggingface" | "ollama" | "bitnet";
  modelRef: string;
  artifact?: string | null;
  registerConfig?: {
    displayName?: string;
    loadPolicy?: LoadPolicy;
    /** Explicit backend choice; omit to let the server infer one. */
    backendType?: BackendType;
    pinPermanent?: boolean;
    preferredGpu?: number | null;
    extraConfig?: BackendExtraConfig;
    /** Public name of the model (`gemma4:e4b`, `qwen3-8b-q4_k_m`). */
    modelKey?: string;
    /** Deployment that serves the public name. */
    defaultBackend?: BackendType;
    /** Every deployment to create from the same file (default included). */
    backends?: BackendType[];
  } | null;
  status: "queued" | "downloading" | "completed" | "failed" | "cancelled";
  progressPct: number;
  speedMbS: number | null;
  etaSeconds: number | null;
  error: string | null;
  startedAt: string;
  completedAt: string | null;
  kind: "install" | "update";
}

export type DownloadSource = "huggingface" | "ollama" | "bitnet";
export type DownloadStatus =
  "queued" | "downloading" | "completed" | "failed" | "cancelled";

/**
 * One backend able to serve a file, from the server's compatibility engine
 * (`core/compatibility.py`). `probable` = should work but not verified
 * (`reason` says why); `installed` = the backend module is on this node.
 * `deployed` only appears on logical models: a deployment already exists.
 */
export type CompatibilityStatus = "compatible" | "probable";

export interface CompatibleBackend {
  backend: BackendType;
  status: CompatibilityStatus;
  installed: boolean;
  reason: string;
  deployed?: boolean;
}

export interface DeploymentArtifact {
  kind?: string;
  source?: string;
  ref?: string;
  repo?: string;
  file?: string;
  mmproj?: boolean;
  draft?: boolean;
}

/** One backend of a logical model (`<backend>/<modelKey>`). */
export interface ModelDeployment {
  modelId: string;
  backend: BackendType;
  isDefault: boolean;
  installed: boolean;
  status: ModelStatus | null;
  loadPolicy: LoadPolicy;
  pinPermanent: boolean;
  /** GPU the scheduler should prefer for this deployment; null = auto. */
  preferredGpu: number | null;
  artifact: DeploymentArtifact | null;
  profiles: string[];
}

/** GET /ocabra/logical-models — one entry per public model name. */
export interface LogicalModel {
  modelKey: string;
  displayName: string;
  category: string | null;
  defaultBackend: BackendType;
  deployments: ModelDeployment[];
  compatibleBackends: CompatibleBackend[];
  publicProfiles: string[];
  hiddenProfiles: string[];
  registryMetadata: Record<string, unknown> | null;
}

export interface RemoveDeploymentResult {
  modelKey: string;
  removed: string;
  newDefaultBackend: BackendType | null;
}

export interface HFModelCard {
  repoId: string;
  modelName: string;
  task: string | null;
  downloads: number;
  likes: number;
  sizeGb: number | null;
  tags: string[];
  gated: boolean;
  suggestedBackend: BackendType;
  // Backends that this repo can run on (suggestedBackend always first).
  // Empty/missing array means just [suggestedBackend] is supported.
  backendOptions?: BackendType[];
  compatibility?: string;
  compatibilityReason?: string | null;
  vllmSupport?: HFVLLMSupport | null;
  compatibleBackends?: CompatibleBackend[];
  createdAt?: string | null;
  lastModified?: string | null;
}

export interface HFModelDetail extends HFModelCard {
  siblings: Record<string, unknown>[];
  readmeExcerpt: string | null;
  suggestedBackend: BackendType;
  estimatedVramGb: number | null;
  vllmSupport?: HFVLLMSupport | null;
}

export interface HFModelVariant {
  variantId: string;
  label: string;
  artifact: string | null;
  sizeGb: number | null;
  format: string;
  quantization: string | null;
  backendType: BackendType;
  isDefault: boolean;
  installable?: boolean;
  compatibility?: string;
  compatibilityReason?: string | null;
  vllmSupport?: HFVLLMSupport | null;
  compatibleBackends?: CompatibleBackend[];
}

export interface OllamaModelCard {
  name: string;
  description: string;
  tags: string[];
  sizeGb: number | null;
  pulls: number;
  compatibleBackends?: CompatibleBackend[];
}

export interface OllamaModelVariant {
  name: string;
  tag: string;
  sizeGb: number | null;
  parameterSize: string | null;
  quantization: string | null;
  contextWindow: string | null;
  modality: string | null;
  updatedHint: string | null;
  compatibleBackends?: CompatibleBackend[];
}

export interface LocalModel {
  modelId: string;
  path: string;
  sizeGb: number;
  backendType: BackendType;
  configured: boolean;
  // Where the file was discovered. "ollama-shared" entries point at a GGUF
  // blob inside Ollama's content-addressed store and can be served by our
  // llama.cpp backend without re-downloading.
  source?: "huggingface" | "gguf" | "ollama" | "ollama-shared";
}

export interface ServerConfig {
  defaultGpuIndex: number;
  idleTimeoutSeconds: number;
  idleEvictionCheckIntervalSeconds: number;
  modelLoadWaitTimeoutSeconds: number;
  pressureEvictionDrainTimeoutSeconds: number;
  pinReloadPollIntervalSeconds: number;
  vramBufferMb: number;
  vramPressureThresholdPct: number;
  maxInflightPerModel: number;
  openaiAudioMaxPartSizeMb: number;
  whisperStartupTimeoutSeconds: number;
  logLevel: string;
  litellmBaseUrl: string;
  litellmAdminKey: string;
  litellmAutoSync: boolean;
  energyCostEurKwh: number;
  modelsDir: string;
  downloadDir: string;
  maxTemperatureC: number;
  gpuThrottleEnabled: boolean;
  gpuThrottlePowerReductionPct: number;
  gpuThrottleCooldownS: number;
  gpuThrottleRecoveryMarginC: number;
  vllmGpuMemoryUtilization: number;
  vllmMaxNumSeqs: number | null;
  vllmMaxNumBatchedTokens: number | null;
  vllmEnablePrefixCaching: boolean;
  vllmEnforceEager: boolean;
  sglangMemFractionStatic: number;
  sglangContextLength: number | null;
  sglangDisableRadixCache: boolean;
  llamaCppGpuLayers: number;
  llamaCppCtxSize: number;
  llamaCppFlashAttn: boolean;
  bitnetGpuLayers: number;
  bitnetCtxSize: number;
  bitnetFlashAttn: boolean;
  diffusersTorchDtype: string;
  diffusersOffloadMode: string;
  diffusersEnableTorchCompile: boolean;
  diffusersEnableXformers: boolean;
  diffusersAllowTf32: boolean;
  tensorrtLlmEnabled: boolean;
  tensorrtLlmMaxBatchSize: number | null;
  tensorrtLlmContextLength: number | null;
  globalSchedules: EvictionSchedule[];
  requireApiKeyOpenai: boolean;
  requireApiKeyOllama: boolean;
  newUserGroupNames: string[];
  defaultReasoningEffort: "low" | "medium" | "high" | "off";
  realtimeDefaultSttModel: string;
  realtimeDefaultTtsModel: string;
  federationEnabled: boolean;
  federationNodeName: string;
  batchPriorityEnabled: boolean;
  batchPreemptMode: "never" | "instant" | "conditional";
  batchPreemptMinElapsedS: number;
  batchPreemptGraceS: number;
  batchFreeSlotsPct: number;
  batchResumeCooldownSlotsS: number;
  batchResumeCooldownEvictionS: number;
  batchInterJobCooldownS: number;
  hunyuanIdleUnloadSeconds: number;
  hunyuanGenerationGracePeriodS: number;
  hunyuanPreferredGpu: number;
  trellis2IdleUnloadSeconds: number;
  trellis2GenerationGracePeriodS: number;
  trellis2PreferredGpu: number;
  comfyuiIdleUnloadSeconds: number;
  comfyuiGenerationGracePeriodS: number;
  comfyuiPreferredGpu: number;
  a1111IdleUnloadSeconds: number;
  a1111GenerationGracePeriodS: number;
  a1111PreferredGpu: number;
  acestepIdleUnloadSeconds: number;
  acestepGenerationGracePeriodS: number;
  acestepPreferredGpu: number;
  unslothIdleUnloadSeconds: number;
  unslothGenerationGracePeriodS: number;
  unslothPreferredGpu: number;
  generationGpuUtilThresholdPct: number;
  // Bloque 20 — routing/estimator/sessions
  routingEnabled?: boolean;
  routerConfidenceFloor?: number;
  routerFallbackDelayMs?: number;
  maxDrainTimeoutS?: number;
  sessionPauseThresholdS?: number;
  sessionMaxIdleS?: number;
}

export interface StatsParams {
  from?: string;
  to?: string;
  modelId?: string;
  allTime?: boolean;
  includeSeries?: boolean;
}

export interface RequestStats {
  totalRequests: number;
  errorRate: number;
  rejections: number;
  avgDurationMs: number;
  p50DurationMs: number;
  p95DurationMs: number;
  series: { timestamp: string; count: number }[];
}

export type RequestLoadPhase =
  "loading" | "waiting_for_gpu" | "prompt_eval" | "generating";

export interface ActiveRequestProgress {
  requestId: string;
  username?: string | null;
  elapsedSeconds: number;
  inputTokensEstimate: number | null;
  outputTokensEstimate: number | null;
  tokensPerSecond: number | null;
  lastActivitySecondsAgo: number;
  phase: RequestLoadPhase;
  waitingOnModelId: string | null;
  // ── Dashboard en vivo (v0.61.0) ──
  /** Fase real: en cola del gate, cargando, prefill, generando, o
   *  "processing" para peticiones sin streaming (no hay eventos que ver). */
  stage?: RequestStage;
  streaming?: boolean | null;
  requestKind?: string | null;
  profileId?: string | null;
  apiKeyName?: string | null;
  source?: string | null;
  /** Epoch en segundos. */
  startedAt?: number;
  elapsedMs?: number;
  /** ms desde la llegada hasta obtener slot / hasta el primer token. */
  admittedAfterMs?: number | null;
  firstTokenAfterMs?: number | null;
  kvReservedTokens?: number | null;
  /** Puesto en el turno real del gate (0 = la siguiente en entrar). */
  queuePosition?: number | null;
  queuedMs?: number | null;
  prompt?: {
    totalTokens: number | null;
    cachedTokens: number | null;
    processedTokens: number | null;
    /** "live": lo informa llama.cpp; "estimate": recuento aproximado. */
    source: "live" | "estimate";
  };
  prefillPct?: number | null;
  prefillTokensPerSecond?: number | null;
  /** live: lo informa llama.cpp · historical: ritmo medido del modelo ·
   *  regression: pendiente del estimador · estimate: inferido del primer token. */
  prefillSource?: "live" | "historical" | "regression" | "estimate" | null;
  outputTokens?: number | null;
  maxOutputTokens?: number | null;
  generationTokensPerSecond?: number | null;
  expectedDurationMs?: number | null;
  estimatedRemainingMs?: number | null;
  queueWaitEstimateMs?: number | null;
  /** El backend ve una señal de vida (tokens o progreso de prefill) y lleva
   *  15 s parada. Las peticiones sin señal que vigilar nunca se marcan. */
  stalled?: boolean;
  /** Orden en que se prevé que entre, teniendo en cuenta el KV libre: puede
   *  diferir del turno (una pequeña adelanta a una grande que no cabe). */
  predictedStartOrder?: number | null;
}

export type RequestStage =
  | "queued"
  /** Admitida por oCabra, pero el backend aún no ha empezado con ella
   *  (llama.cpp procesa los prompts de uno en uno). */
  | "backend_wait"
  | "loading"
  | "waiting_for_gpu"
  | "prefill"
  | "generating"
  | "processing";

/** Fase de carga de un modelo (dashboard). */
export interface ModelPhaseInfo {
  /** waiting_for_gpu: desalojando a waitingOn para hacer sitio ·
   *  waiting_for_service_gpu: GPU retenida por el servicio waitingOn. */
  phase: "ready" | "loading" | "waiting_for_gpu" | "waiting_for_service_gpu";
  waitingOn: string | null;
  loadingForMs: number | null;
  /** Mediana de lo que tarda en cargar este modelo (30 días). */
  typicalLoadMs: number | null;
}

/** Un nivel de contexto: lo que cabe con una colocación de GPUs dada. */
export interface ContextTier {
  presetKey: string | null;
  gpus: number[];
  maxTokens: number;
  slots: number | null;
}

/** Datos de la barra de contexto de un modelo cargado. */
export interface ModelContextInfo {
  /** pool = KV unificada (llama.cpp mks); slots = parcelas fijas (Ollama,
   *  llama.cpp sin pool); paged = vLLM. */
  mode: "pool" | "slots" | "paged";
  scaleMaxTokens: number | null;
  activeMaxTokens: number | null;
  activePresetKey: string | null;
  activeGpus: number[];
  /** Tope del mayor nivel de una sola GPU: la marca de la barra. */
  singleGpuMaxTokens: number | null;
  tiers: ContextTier[];
  reservedTokens: number | null;
  segments: { requestId: string; reservedTokens: number | null; usedTokens: number | null }[];
  usagePct: number | null;
  source: "live" | "estimate";
}

/** Fair-share gate snapshot: queue depth and KV pool occupancy. Only present
 *  for backends that really serve concurrent slots and have had traffic —
 *  others never create a gate, so consumers must degrade gracefully. */
export interface ModelAdmission {
  slots: number;
  queuedRequests: number;
  slotSaturated: boolean;
  computeSaturated: boolean;
  admissionLimit: number;
  /** null when the worker does not publish a unified KV pool. */
  kvCapacityTokens: number | null;
  kvReservedTokens: number | null;
  kvFreeTokens: number | null;
}

export interface ModelActivity {
  inFlight: number;
  oldestSeconds: number;
  slots?: number | null;
  requests?: ActiveRequestProgress[];
  /** How many in-flight requests sit in each phase. */
  phases?: Record<string, number>;
  /** Sum over requests actually emitting tokens; null when none are. Pair it
   *  with generatingCount — tokensPerSecond is null outside `generating`, so a
   *  model busy in prompt-eval would otherwise read as 0 t/s at full load. */
  tokensPerSecondTotal?: number | null;
  generatingCount?: number;
  admission?: ModelAdmission;
  /** Active runtime preset and its resolved context, read from the derived
   *  worker state. Without these the KV bar is a number with no reference. */
  runtimePreset?: string | null;
  effectiveContext?: number | null;
  stages?: Partial<Record<RequestStage, number>>;
  queuedCount?: number;
  /** t/s medidos en su propia fase (no divididos por el tiempo total). */
  prefillTokensPerSecondTotal?: number | null;
  generationTokensPerSecondTotal?: number | null;
}

export interface ModelActivityMap {
  activity: Record<string, ModelActivity>;
  /** Barra de contexto de cada modelo cargado, aunque no tenga peticiones. */
  contexts?: Record<string, ModelContextInfo>;
  /** MB medidos por NVML de cada modelo en cada GPU: {modelId: {"0": MB}}. */
  vramByGpu?: Record<string, Record<string, number>>;
  /** Fase de carga de cada modelo activo o esperando. */
  modelStates?: Record<string, ModelPhaseInfo>;
  stuckThresholdSeconds: number;
  /** Seconds of inactivity after which an on_demand model is unloaded. */
  idleTimeoutSeconds?: number;
}

export interface OllamaRuntimeInfo {
  sizeMb: number;
  sizeVramMb: number;
  gpuPct: number;
  cpuPct: number;
  contextLength: number | null;
}

export interface GpuUsageTotal {
  gpuIndex: number;
  samples: number;
  hoursTotal: number;
  hoursActive: number;
  activeRatio: number;
  avgUtilizationPct: number;
  avgVramMb: number;
  peakVramMb: number;
  avgTempC: number;
  kwh: number;
}

export interface GpuUsageSeriesPoint {
  t: string;
  gpuIndex: number;
  utilizationPct: number;
  vramMb: number;
  powerW: number;
  tempC: number;
}

export interface GpuUsageStats {
  from: string;
  to: string;
  bucket: "hour" | "day";
  activeThresholdPct: number;
  totals: GpuUsageTotal[];
  series: GpuUsageSeriesPoint[];
}

export interface EnergyStats {
  totalKwh: number;
  estimatedCostEur: number;
  byGpu: { gpuIndex: number; totalKwh: number; powerDrawW: number }[];
  cpuKwh?: number;
  totalServerKwh?: number;
}

export interface PerformanceStats {
  byModel: {
    modelId: string;
    backendType: string;
    requestKinds: string[];
    totalRequests: number;
    avgLatencyMs: number;
    p95LatencyMs: number;
    requestsPerMinute: number;
    tokensPerSecond: number;
    totalInputTokens: number;
    totalOutputTokens: number;
    tokenizedRequests: number;
    errorCount: number;
    uptimePct: number;
    loadCount: number;
    avgLoadMs: number;
    p95LoadMs: number;
    lastLoadMs: number;
  }[];
}

export interface TokenGpuStats {
  gpuIndex: number | null;
  inputTokens: number;
  outputTokens: number;
}

export interface TokenStats {
  totalInputTokens: number;
  totalOutputTokens: number;
  byBackend: {
    backendType: string;
    inputTokens: number;
    outputTokens: number;
  }[];
  byGpu: TokenGpuStats[];
  series: { timestamp: string; inputTokens: number; outputTokens: number }[];
}

export interface OverviewStats {
  totalRequests: number;
  totalErrors: number;
  avgDurationMs: number;
  tokenizedRequests: number;
  totalInputTokens: number;
  totalOutputTokens: number;
  estimatedCostUsd?: number;
  totalEnergyWh?: number;
  byBackend: {
    backendType: string;
    totalRequests: number;
    errorRate: number;
    avgLatencyMs: number;
    p95LatencyMs: number;
  }[];
  byRequestKind: {
    requestKind: string;
    totalRequests: number;
    errorRate: number;
    avgLatencyMs: number;
    p95LatencyMs: number;
  }[];
}

export interface RecentRequest {
  id: string;
  modelId: string;
  backendType: string | null;
  requestKind: string | null;
  endpointPath: string | null;
  statusCode: number | null;
  startedAt: string;
  durationMs: number | null;
  inputTokens: number | null;
  outputTokens: number | null;
  error: string | null;
  userId: string | null;
  username: string | null;
  groupId: string | null;
  groupName: string | null;
  apiKeyName: string | null;
  clientAddr: string | null;
  userAgent: string | null;
  inFlight?: boolean;
  elapsedMs?: number | null;
  estimatedRemainingMs?: number | null;
  workerKey?: string | null;
  queued?: boolean;
  queuePosition?: number;
  slots?: number;
  queueWaitEstimateMs?: number | null;
  // v0.60.0: rendimiento de peticiones terminadas.
  queueMs?: number | null;
  promptTokensPerSecond?: number | null;
  predictedTokensPerSecond?: number | null;
  generationMs?: number | null;
  cachedTokens?: number | null;
  effectiveContext?: number | null;
  runtimePreset?: string | null;
  gpuIndex?: number | null;
  runtimeTelemetry?: {
    prompt_tokens_estimate?: number;
    kv_tokens_reserved?: number;
    queue_ms?: number;
    load_ms?: number;
    ttft_ms?: number;
    prefill_ms?: number;
    slot_id?: number;
    physical_slots?: number;
    admission_limit?: number;
    admission_concurrency?: number;
  } | null;
}

export interface RecentRequestsData {
  requests: RecentRequest[];
}

export interface UserStatRow {
  userId: string;
  username: string;
  totalRequests: number;
  totalErrors: number;
  avgDurationMs: number;
  totalInputTokens: number;
  totalOutputTokens: number;
  totalEnergyWh?: number;
  estimatedCostUsd?: number;
}

export interface GroupStatRow {
  groupId: string;
  groupName: string;
  totalRequests: number;
  totalErrors: number;
  avgDurationMs: number;
  totalInputTokens: number;
  totalOutputTokens: number;
  totalEnergyWh?: number;
  estimatedCostUsd?: number;
}

export interface ByUserStats {
  byUser: UserStatRow[];
}

export interface ByGroupStats {
  byGroup: GroupStatRow[];
}

export interface MyGroupStats {
  groupId: string | null;
  groupName: string | null;
  stats: OverviewStats;
}

export interface ApiKeyStatRow {
  apiKeyName: string;
  userId: string | null;
  username: string | null;
  totalRequests: number;
  totalErrors: number;
  avgDurationMs: number;
  totalInputTokens: number;
  totalOutputTokens: number;
  totalEnergyWh: number;
  estimatedCostUsd: number;
}

export interface ByApiKeyStats {
  byApiKey: ApiKeyStatRow[];
}

export interface UserDetailStats {
  userId: string;
  username: string;
  totalRequests: number;
  totalErrors: number;
  totalInputTokens: number;
  totalOutputTokens: number;
  totalEnergyWh: number;
  estimatedCostUsd: number;
  topModels: {
    modelId: string;
    requests: number;
    inputTokens: number;
    outputTokens: number;
  }[];
  byRequestKind: { requestKind: string; count: number }[];
  tokenSeries: {
    timestamp: string;
    inputTokens: number;
    outputTokens: number;
  }[];
}

export interface FederationStats {
  localCount: number;
  remoteCount: number;
  byNode: { nodeId: string; count: number; avgDurationMs: number }[];
}

export interface BatchVsInteractiveSummary {
  totalRequests: number;
  totalErrors: number;
  avgDurationMs: number;
  totalInputTokens: number;
  totalOutputTokens: number;
  totalTokens: number;
  totalEnergyWh: number;
  estimatedCostUsd: number;
}

export interface BatchVsInteractiveStats {
  batch: BatchVsInteractiveSummary;
  interactive: BatchVsInteractiveSummary;
  dailySeries: { date: string; batch: number; interactive: number }[];
}

export interface BatchPriorityCooldown {
  workerKey: string;
  kind: string;
  remainingSeconds: number;
}

export interface BatchPriorityEvent {
  type: string;
  workerKey: string;
  batchId: string;
  lineIdx: number;
  elapsedSeconds: number;
  mode: string;
  secondsAgo: number;
}

export interface BatchPriorityStatus {
  totalPreemptions: number;
  activeCooldowns: BatchPriorityCooldown[];
  recentEvents: BatchPriorityEvent[];
}

export interface ServerPower {
  cpuPowerW: number | null;
  cpuTempC: number | null;
  totalGpuPowerW: number | null;
  totalPowerW: number | null;
}

export interface GPUStatHistoryPoint {
  timestamp: string;
  usedVramMb: number;
  utilizationPct: number;
  temperatureC: number;
  powerDrawW: number;
}

export interface GPUStatHistory {
  gpuIndex: number;
  window: string;
  points: GPUStatHistoryPoint[];
}

export type ServiceStatus =
  | "unknown"
  | "idle"
  | "active"
  | "unreachable"
  | "disabled"
  | "building"
  | "starting";

export interface ServiceState {
  serviceId: string;
  serviceType: string;
  displayName: string;
  uiUrl: string;
  preferredGpu: number | null;
  idleUnloadAfterSeconds: number;
  enabled: boolean;
  serviceAlive: boolean;
  runtimeLoaded: boolean;
  status: ServiceStatus;
  activeModelRef: string | null;
  lastActivityAt: string | null;
  lastHealthCheckAt: string | null;
  detail: string | null;
  // Generation metrics
  isGenerating: boolean;
  queueDepth: number;
  vramUsedMb: number | null;
  gpuUtilPct: number | null;
  // Container resource metrics
  cpuPct: number | null;
  memUsedMb: number | null;
  memLimitMb: number | null;
}

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

// TRT-LLM compile jobs
export type CompileStatus =
  "pending" | "running" | "done" | "failed" | "cancelled";
export type CompilePhase = "convert" | "build" | null;

export interface CompileJob {
  jobId: string;
  sourceModel: string;
  engineName: string;
  gpuIndices: number[];
  dtype: string;
  config: {
    maxBatchSize: number;
    maxInputLen: number;
    maxSeqLen: number;
  };
  status: CompileStatus;
  phase: CompilePhase;
  progressPct: number;
  errorDetail: string | null;
  engineDir: string | null;
  startedAt: string | null;
  finishedAt: string | null;
}

// Auth types
export type UserRole = "user" | "model_manager" | "system_admin";

export interface AuthUser {
  id: string;
  username: string;
  email: string | null;
  role: UserRole;
  createdAt: string;
}

export interface ApiKey {
  id: string;
  name: string;
  keyPrefix: string;
  expiresAt: string | null;
  lastUsedAt: string | null;
  isRevoked: boolean;
  createdAt: string;
  groupId?: string | null;
}

// An API key as seen by an admin in the "all keys" view, including its owner.
export interface AdminApiKey extends ApiKey {
  userId: string;
  username: string;
}

export interface Group {
  id: string;
  name: string;
  description: string;
  isDefault: boolean;
  grantsAllModels: boolean;
  memberCount: number;
  modelCount: number;
  createdAt: string;
}

// Admin user management
export interface AdminUser {
  id: string;
  username: string;
  email: string | null;
  role: UserRole;
  isActive: boolean;
  createdAt: string;
}

export interface GroupMember {
  userId: string;
  username: string;
  role: UserRole;
}

export interface CreateUserPayload {
  username: string;
  password: string;
  role: UserRole;
  email?: string | null;
}

export interface UpdateUserPayload {
  role?: UserRole;
  isActive?: boolean;
  email?: string | null;
}

export interface CreateGroupPayload {
  name: string;
  description?: string | null;
  grantsAllModels?: boolean;
}

export interface UpdateGroupPayload {
  name?: string;
  description?: string | null;
  grantsAllModels?: boolean;
}

// Model profiles
export type ProfileCategory =
  | "llm"
  | "embedding"
  | "rerank"
  | "tts"
  | "stt"
  | "image"
  | "music"
  | "video_upscaling";

export interface RuntimePolicy {
  selection: "prefer_loaded" | "prefer_default";
  restoreDefaultWhenIdle: boolean;
  downgradeGraceSeconds: number;
  minimumResidencySeconds: number;
  smallRequestsExtendLargeTtl: boolean;
  remoteWarmThresholdSeconds: number;
  maxContextTokens: number | null;
  maxOutputTokens: number | null;
  validatedContextTokens: Record<string, number>;
  validatedVramMb: Record<string, number>;
  defaultOutputReservationTokens: number | null;
}

export interface RuntimePreset {
  id: string;
  profileId: string;
  presetKey: string;
  label: string;
  contextMode: "fixed" | "max";
  contextTokens: number | null;
  parallelism: number;
  kvCacheDtype: "fp16" | "bf16" | "fp8" | "q8" | "q4";
  placementPolicy: "auto" | "preferred" | "forced";
  preferredGpus: number[] | null;
  executionPolicy: "auto" | "local" | "remote";
  enabled: boolean;
  isDefault: boolean;
  priority: number;
  createdAt: string | null;
  updatedAt: string | null;
}

export interface RuntimePresetResolution {
  presetKey: string;
  resolved: boolean;
  requestedContext: number | null;
  effectiveContext: number | null;
  modelMax: number | null;
  hardwareMax: number | null;
  policyMax: number | null;
  requestedParallelism: number;
  effectiveParallelism: number;
  targetGpus: number[];
  fits: boolean;
  runtimeKey: string | null;
  warnings: string[];
}

export interface RuntimePresetBenchmark {
  presetKey: string;
  effectiveContext: number;
  workerKey: string;
  wasCold: boolean;
  loadMs: number;
  generationMs: number;
  outputTokens: number | null;
  tokensPerSecond: number | null;
}

export interface RuntimePresetCreate {
  presetKey: string;
  label: string;
  contextMode: "fixed" | "max";
  contextTokens?: number | null;
  parallelism: number;
  kvCacheDtype: RuntimePreset["kvCacheDtype"];
  placementPolicy: RuntimePreset["placementPolicy"];
  preferredGpus?: number[] | null;
  executionPolicy: RuntimePreset["executionPolicy"];
  enabled?: boolean;
  isDefault?: boolean;
  priority?: number;
}

export type RuntimePresetUpdate = Partial<
  Omit<RuntimePresetCreate, "presetKey">
>;

export interface ModelProfile {
  profileId: string;
  profileKind: "execution" | "router";
  baseModelId: string | null;
  displayName: string | null;
  description: string | null;
  category: ProfileCategory;
  loadOverrides: Record<string, unknown> | null;
  requestDefaults: Record<string, unknown> | null;
  assets: Record<string, unknown> | null;
  enabled: boolean;
  isDefault: boolean;
  // Bloque 20 — when non-empty, this profile is a Router: resolve()
  // walks these profile IDs in order and returns the first available
  // one. An empty/null array means "regular profile".
  routingTargets: string[] | null;
  runtimePolicy: RuntimePolicy | null;
  runtimePresets: RuntimePreset[];
  aliases: ProfileAlias[];
  createdAt: string | null;
  updatedAt: string | null;
}

export interface ProfileAlias {
  aliasId: string;
  profileId: string;
  presetKey: string | null;
  hidden: boolean;
}

export interface ProfileCreate {
  profileId: string;
  displayName?: string;
  description?: string;
  category: string;
  loadOverrides?: Record<string, unknown>;
  requestDefaults?: Record<string, unknown>;
  enabled?: boolean;
  isDefault?: boolean;
  routingTargets?: string[] | null;
  runtimePolicy?: RuntimePolicy | null;
}

export interface ProfileUpdate {
  displayName?: string;
  description?: string;
  category?: string;
  loadOverrides?: Record<string, unknown>;
  requestDefaults?: Record<string, unknown>;
  enabled?: boolean;
  isDefault?: boolean;
  routingTargets?: string[] | null;
  runtimePolicy?: RuntimePolicy | null;
}

export interface RouterCreate {
  profileId: string;
  displayName?: string;
  description?: string;
  category: ProfileCategory;
  requestDefaults?: Record<string, unknown>;
  enabled?: boolean;
  routingTargets: string[];
}

// Federation types
export interface FederationPeerGpu {
  index: number;
  name: string;
  total_vram_mb: number;
  free_vram_mb: number;
}

export interface FederationPeerModel {
  model_id: string;
  status: string;
  profiles: string[];
}

export interface FederationPeerLoad {
  active_requests: number;
  gpu_utilization_avg_pct: number;
}

export interface FederationPeer {
  peer_id: string;
  name: string;
  url: string;
  access_level: "inference" | "full";
  enabled: boolean;
  online: boolean;
  last_heartbeat: string | null;
  latency_ms?: number;
  gpus: FederationPeerGpu[];
  models: FederationPeerModel[];
  load: FederationPeerLoad;
}

export interface FederationPeerCreate {
  name: string;
  url: string;
  api_key: string;
  access_level: "inference" | "full";
}

export interface FederationPeerUpdate {
  name?: string;
  url?: string;
  api_key?: string;
  access_level?: "inference" | "full";
  enabled?: boolean;
}

export interface FederationTestResult {
  success: boolean;
  latency_ms: number | null;
  error?: string;
  node_info?: Record<string, unknown>;
}

export type BatchEndpoint =
  | "/v1/chat/completions"
  | "/v1/embeddings"
  | "/v1/completions"
  | "/v1/images/generations"
  | "/v1/audio/speech"
  | "/v1/audio/transcriptions"
  | "/v1/audio/generate"
  | "/v1/pooling"
  | "/v1/score"
  | "/v1/rerank"
  | "/v1/classify"
  | "/v1/images/edits";

export type BatchStatus =
  | "validating"
  | "in_progress"
  | "finalizing"
  | "completed"
  | "failed"
  | "expired"
  | "cancelling"
  | "cancelled";

export interface OpenAIBatchJob {
  id: string;
  object: "batch";
  endpoint: BatchEndpoint;
  errors: unknown;
  input_file_id: string;
  completion_window: string;
  status: BatchStatus;
  output_file_id: string | null;
  error_file_id: string | null;
  created_at: number | null;
  in_progress_at: number | null;
  expires_at: number | null;
  finalizing_at: number | null;
  completed_at: number | null;
  failed_at: number | null;
  expired_at: number | null;
  cancelling_at: number | null;
  cancelled_at: number | null;
  request_counts: { total: number; completed: number; failed: number };
  total_tokens: number;
  metadata: Record<string, unknown> | null;
  user: { id: string; username: string | null };
}

export interface OpenAIFileObject {
  id: string;
  object: "file";
  bytes: number;
  created_at: number;
  filename: string;
  purpose: string;
  status: string;
  status_details: string | null;
}

// Backend modules (Bloque 15) — re-exported for convenience.
export type {
  BackendInstallStatus,
  BackendInstallMethod,
  BackendInstallSource,
  BackendModuleState,
} from "./backends";

import type { BackendModuleState } from "./backends";

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
      data: { jobId: string; pct: number; speedMbS: number };
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
  | { type: "system_alert"; data: { level: "warn" | "error"; message: string } }
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
