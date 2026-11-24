// Tipos del dominio «models» (REF-15).
// Extraídos de types/index.ts; el barrel index.ts los reexporta con export *.

import type { HFVLLMRuntimeProbe } from "./registry"
import type { ActiveRequestProgress, FinishedRequest, RequestStage } from "./stats"

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
  | "comfyui_video"
  | "voxtral"
  | "ollama"
  | "tensorrt_llm"
  | "llama_cpp"
  | "llamacpp_mks"
  | "sglang"
  | "strata";

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
  /** Generación de vídeo (/v1/videos). Opcional: los fixtures antiguos no lo traen. */
  videoGeneration?: boolean;
  /** Escalado de vídeo / imagen (upscalers). El backend los expone en
   *  BackendCapabilities; opcionales porque los fixtures antiguos no los traen. */
  videoUpscaling?: boolean;
  imageUpscaling?: boolean;
  streaming: boolean;
  contextLength: number;
  nativeContextLength: number;
  /** Tope de salida y paralelismo que publica el backend (0 = sin dato).
   *  Opcionales: no todos los fixtures/caminos los rellenan. */
  maxOutputTokens?: number;
  parallelSlots?: number;
  kvCacheShared?: boolean;
  contextPerRequest?: number;
}

export interface ModelState {
  modelId: string;
  /** Id nativo del runtime (backend_model_id); el canónico es modelId. */
  backendModelId?: string | null;
  /** Modo mantenimiento: las peticiones nuevas reciben 503. */
  maintenance?: boolean;
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
  /** Estado de contexto en vivo del worker (llama.cpp): tamaño actual, máximo,
   *  si puede crecer… Forma libre; la consume el panel de contexto en vivo. */
  liveContext?: Record<string, unknown> | null;
  schedules?: EvictionSchedule[];
  extraConfig?: BackendExtraConfig;
  profiles?: ModelProfile[];
  createdAt?: string | null;
  updatedAt?: string | null;
  releaseDate?: string | null;
  lastUpdated?: string | null;
  metadataFetchedAt?: string | null;
  federation?: {
    /** Only on other nodes (not configured here). */
    remote: boolean;
    nodeName: string;
    nodeId: string;
    /** Where it is loaded right now ("aquí" = this server). */
    loadedOn?: string[];
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

export interface EvictionSchedule {
  id: string;
  days: number[];
  start: string;
  end: string;
  enabled: boolean;
}

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

/** Datos de la barra de contexto de un modelo cargado. */
export interface ModelContextInfo {
  /** pool = KV unificada (llama.cpp mks); slots = parcelas fijas (Ollama,
   *  llama.cpp sin pool); paged = vLLM. */
  mode: "pool" | "slots" | "paged";
  scaleMaxTokens: number | null;
  activeMaxTokens: number | null;
  /** El contexto se cambió en vivo sin recargar (v0.65.0); nominal = el de carga. */
  liveResized?: boolean;
  nominalMaxTokens?: number | null;
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

/** Un nivel de contexto: lo que cabe con una colocación de GPUs dada. */
export interface ContextTier {
  presetKey: string | null;
  gpus: number[];
  maxTokens: number;
  slots: number | null;
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
  /** MB de RAM del host (RSS en vivo) de los modelos que la usan a propósito
   *  aunque corran en GPU (Strata). */
  ramByModel?: Record<string, number>;
  /** Terminadas en los últimos 20 s, la más reciente primero (incluye 429). */
  finished?: FinishedRequest[];
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

export interface ServerConfig {
  defaultGpuIndex: number;
  idleTimeoutSeconds: number;
  idleEvictionCheckIntervalSeconds: number;
  modelLoadWaitTimeoutSeconds: number;
  pressureEvictionDrainTimeoutSeconds: number;
  evictionFollowupGraceSeconds: number;
  evictionFollowupMaxChainSeconds: number;
  pinReloadPollIntervalSeconds: number;
  vramBufferMb: number;
  vramPressureThresholdPct: number;
  maxInflightPerModel: number;
  openaiAudioMaxPartSizeMb: number;
  whisperStartupTimeoutSeconds: number;
  logLevel: string;
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
  /** Servidor Ollama único: peticiones simultáneas por modelo y contexto máximo. */
  ollamaNumParallel?: number;
  ollamaMaxContext?: number;
  newUserGroupNames: string[];
  defaultReasoningEffort: "low" | "medium" | "high" | "off";
  realtimeDefaultSttModel: string;
  realtimeDefaultTtsModel: string;
  federationEnabled: boolean;
  federationNodeName: string;
  federationAcceptJoinRequests: boolean;
  /** functional: as private as possible without getting in the way; zero_data: nothing kept. */
  privacyMode: "functional" | "zero_data";
  /** Where temporary files go (RAM in zero data) and how much room is left. */
  privacyTempDir?: { dir: string; in_memory: boolean; free_mb: number | null } | null;
  /** Batch/upload files are deleted this long after their batch ends (0 = keep). */
  openaiFilesRetentionHours: number;
  batchPriorityEnabled: boolean;
  batchPreemptMode: "never" | "instant" | "conditional";
  batchPreemptMinElapsedS: number;
  batchPreemptGraceS: number;
  batchFreeSlotsPct: number;
  batchResumeCooldownSlotsS: number;
  batchResumeCooldownEvictionS: number;
  batchInterJobCooldownS: number;
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
  maxDrainTimeoutS?: number;
  sessionPauseThresholdS?: number;
  sessionMaxIdleS?: number;
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
  /** Detail of the last applied context probe, by preset (snake_case, as the API stores it). */
  contextProbe: Record<string, Record<string, unknown>>;
  defaultOutputReservationTokens: number | null;
  /** Claves que la UI no edita (p. ej. reserve_mb_by_preset), devueltas tal cual al guardar. */
  passthrough?: Record<string, unknown>;
}

export interface ContextProbeStep {
  context: number;
  usedMb: Record<string, number>;
  fits: boolean;
  note: string;
  /** Modo «contexto objetivo»: capas (expertos MoE o capas enteras) en RAM. */
  inRam?: number | null;
  /** Lo que proyectaba llama.cpp para ese reparto (ya calibrado). */
  projectedMb?: Record<string, number>;
  ramNeedMb?: number | null;
}

/** Resultado del modo «contexto objetivo, lo más rápido». */
export interface ContextProbeOffload {
  targetContext: number;
  poolTokens: number;
  inRam: number;
  layers: number;
  moe: boolean;
  placement: Record<string, number>;
  ramNeedMb: number | null;
  ramAvailableMb: number | null;
  mlock: boolean;
  firstGuess: number;
}

export interface ContextProbeJob {
  jobId: string;
  profileId: string;
  presetKey: string;
  status:
    | "pending"
    | "preparing"
    | "planning"
    | "loading"
    | "probing"
    | "validating"
    | "vision"
    | "restoring"
    | "done"
    | "error";
  fullValidation: boolean;
  measureVision: boolean;
  targetContext: number | null;
  gpuIndices: number[];
  contextIsPool: boolean;
  parallelism: number;
  steps: ContextProbeStep[];
  evicted: string[];
  error: string | null;
  startedAt: number;
  finishedAt: number | null;
  result: {
    maxContext: number;
    /** Presente cuando el sondeo buscaba el reparto para un contexto objetivo. */
    offload?: ContextProbeOffload | null;
    usedMb: Record<string, number>;
    limitMb: Record<string, number>;
    kvMbPer1k: Record<string, number>;
    cappedBy: string;
    trainedContext: number;
    marginMb: number;
    fullValidated: boolean;
    peakMb?: Record<string, number>;
    validationPromptTokens?: number | null;
    reducedByValidation?: number;
    vision?: {
      projector: string;
      costMb: Record<string, number>;
      maxContextWithVision: number;
    } | null;
  } | null;
}

/** Colocación propia de un peldaño (GPU/CPU). null = la de siempre: todo en
 *  GPU, una sola GPU salvo "forced" con varias, reparto calculado. */
export interface PresetPlacement {
  /** single: una GPU · when_needed: repartir si no cabe en ninguna · always: siempre repartido */
  splitPolicy: "single" | "when_needed" | "always" | null;
  splitMode: "layer" | "row" | "tensor" | null;
  /** Proporción por GPU, en el orden de preferredGpus (solo con varias). */
  tensorSplit: number[] | null;
  /** Capas en GPU; null = todas. */
  gpuLayers: number | null;
  /** Capas MoE cuyos expertos van a la CPU. */
  nCpuMoe: number | null;
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
  enabled: boolean;
  isDefault: boolean;
  priority: number;
  placement?: PresetPlacement | null;
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
  /** VRAM estimada de una carga con este preset, total y por GPU (MB). */
  estimatedVramMb?: number | null;
  /** RAM del host que ocupa lo que va a la CPU (MB). */
  cpuMb?: number | null;
  vramByGpu?: Record<string, number>;
  /** Desglose por GPU: pesos, KV, rs_cache, fijo, extra (MB). */
  vramBreakdownByGpu?: Record<string, Record<string, number>>;
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
  enabled?: boolean;
  isDefault?: boolean;
  priority?: number;
  placement?: PresetPlacement | null;
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
  /** Solo routers: qué contexto anuncia (null = "max"). */
  routerContext?: RouterContextPolicy | null;
  runtimePolicy: RuntimePolicy | null;
  runtimePresets: RuntimePreset[];
  aliases: ProfileAlias[];
  createdAt: string | null;
  updatedAt: string | null;
}

/**
 * Contexto que anuncia un router en /v1/models y /api/tags. El enrutado no
 * cambia (una petición que solo cabe en un destino sigue yendo a él); cambia
 * el número que leen los clientes para decidir cuándo compactar.
 * - max: el mayor de sus destinos.
 * - common: el que alcanza más de la mitad de sus destinos.
 * - fixed: `tokens`, nunca por encima del mayor de sus destinos.
 */
export interface RouterContextPolicy {
  mode: "max" | "common" | "fixed";
  tokens?: number | null;
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
  /** null borra la clave (sin sobrescrituras / sin valores por defecto). */
  loadOverrides?: Record<string, unknown> | null;
  requestDefaults?: Record<string, unknown> | null;
  enabled?: boolean;
  isDefault?: boolean;
  routingTargets?: string[] | null;
  runtimePolicy?: RuntimePolicy | null;
  /** Solo routers; null vuelve al valor por defecto ("max"). */
  routerContext?: RouterContextPolicy | null;
}

export interface RouterCreate {
  profileId: string;
  displayName?: string;
  description?: string;
  category: ProfileCategory;
  requestDefaults?: Record<string, unknown>;
  enabled?: boolean;
  routingTargets: string[];
  /** Qué contexto anuncia el router (null = "max"). */
  routerContext?: RouterContextPolicy | null;
}

/**
 * Compatibilidad de voces OpenAI (independiente de los routers): fija a
 * qué perfil propio responde cada par (modelo oficial de OpenAI, nombre
 * de voz) para un cliente que solo sabe mandar esos nombres.
 */
export interface OpenAITTSVoiceMapEntry {
  openaiModel: string;
  voiceName: string;
  targetProfileId: string;
  createdAt: string | null;
  updatedAt: string | null;
}

/**
 * Compatibilidad de imagen OpenAI: a qué perfil propio responde cada nombre
 * oficial de modelo de imagen (gpt-image-1, dall-e-3...).
 */
export interface OpenAIImageModelMapEntry {
  openaiModel: string;
  targetProfileId: string;
  createdAt: string | null;
  updatedAt: string | null;
}

/** A model's federation strategy: where its requests are served. */
export interface ModelFederationStrategy {
  model_id: string;
  /** auto: warm beats cold, local bias · local_only · prefer: nodes in order. */
  strategy: "auto" | "local_only" | "prefer";
  /** prefer: peer names (or "local"), tried in order. */
  nodes: string[];
  /** prefer: when none can serve it, go on as auto (false = 503). */
  fallback: boolean;
  /** The model exists on this node. */
  local: boolean;
  /** Every peer, with how it has this model: loaded, cold, or null (doesn't). */
  peers: { name: string; peer_id: string; online: boolean; model: "loaded" | "cold" | null }[];
}
