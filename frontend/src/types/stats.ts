// Tipos del dominio «stats» (REF-15).
// Extraídos de types/index.ts; el barrel index.ts los reexporta con export *.

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
  /** Federation: node that served it (null = here), node that sent it to us,
   *  and why it went where it did (see lib/federationRoute). */
  node?: string | null;
  inboundFrom?: string | null;
  route?: string | null;

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
    /** false: lo procesado es real (de /slots) pero el total es estimado. */
    totalKnown?: boolean;
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
  /** El restante es "como mucho": la petición ya superó la salida típica del
   *  modelo y se cuenta hasta su max_tokens. */
  remainingIsUpperBound?: boolean;
  queueWaitEstimateMs?: number | null;
  /** Transcripción (faster-whisper): avance real del audio y velocidad
   *  respecto al tiempo real (×N = N veces más rápido que el audio). */
  audio?: {
    phase: "preparing" | "transcribing" | "diarizing" | null;
    totalSeconds: number | null;
    processedSeconds: number;
    pct: number | null;
    realtimeFactor: number | null;
    segments: number;
  };
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

/** Petición recién terminada (feed en vivo del dashboard, 20 s). */
export interface FinishedRequest {
  /** Federation: node that served it (null = here), node that sent it to us,
   *  and why it went where it did (see lib/federationRoute). */
  node?: string | null;
  inboundFrom?: string | null;
  route?: string | null;

  requestId: string | null;
  model: string;
  backendType: string | null;
  gpuIndex: number | null;
  requestKind: string | null;
  statusCode: number | null;
  error: string | null;
  username: string | null;
  apiKeyName: string | null;
  durationMs: number;
  inputTokens: number | null;
  cachedTokens: number | null;
  outputTokens: number | null;
  promptTokensPerSecond: number | null;
  predictedTokensPerSecond: number | null;
  ttftMs: number | null;
  queueMs: number | null;
  /** Transcripciones: duración del audio (con durationMs, el ×N final). */
  audioSeconds?: number | null;
  /** Vídeo generado: fotogramas por segundo de generación y fotogramas. */
  videoFps?: number | null;
  videoFrames?: number | null;
  /** Epoch en segundos. */
  endedAt: number;
  endedAgoMs: number;
}

export interface RecentRequest {
  /** Federation: node that served it (null = here), node that sent it to us,
   *  and why it went where it did (see lib/federationRoute). */
  node?: string | null;
  inboundFrom?: string | null;
  route?: string | null;

  id: string;
  modelId: string;
  /** Nombre que usó el cliente (perfil, alias): distingue un -diarized. */
  requestedModel?: string | null;
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
  /** Transcripciones: duración del audio (con durationMs, el ×N final). */
  audioSeconds?: number | null;
  /** Vídeo generado: fotogramas por segundo de generación y fotogramas. */
  videoFps?: number | null;
  videoFrames?: number | null;
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
  /** Only on a «0 data» node: the last requests are kept in memory. */
  zeroData?: ZeroDataSummary;
}

/** What the last requests of a «0 data» node add up to (kept in memory). */
export interface ZeroDataSummary {
  kept: number;
  requests: number;
  errors: number;
  avgDurationMs: number | null;
  inputTokens: number;
  outputTokens: number;
  avgPredictedTokensPerSecond: number | null;
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

export interface FederationNodeTraffic {
  node: string;
  count: number;
  errors: number;
  inputTokens: number;
  outputTokens: number;
  avgDurationMs: number;
  p50DurationMs: number;
  p95DurationMs: number;
  topModels: { model: string; count: number }[];
  /** Outbound to cloud nodes: what they cost in the window. */
  nodeId?: string | null;
  costUsd?: number | null;
  usdPerRequest?: number | null;
  usdPer1kTokens?: number | null;
}

export interface FederationStats {
  localCount: number;
  remoteCount: number;
  inboundCount?: number;
  fallbackCount?: number;
  outbound?: FederationNodeTraffic[];
  inbound?: FederationNodeTraffic[];
  routes?: { route: string; count: number }[];
  fallbacksByModel?: { model: string; count: number }[];
  byNode: { nodeId: string; nodeName?: string; count: number; avgDurationMs: number }[];
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

export interface WorkSpeed {
  kinds: string[];
  unit: "x_realtime" | "chars_per_s" | "inputs_per_s" | "tokens_per_s" | "documents_per_s" | "s_per_image" | "frames_per_s" | "mp_per_s" | string;
  value: number;
  samples: number;
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
    /** Velocidad de lo que no se mide en tokens generados, en su unidad
     *  (×tiempo real, caracteres/s, documentos/s, s por imagen…). */
    workSpeeds?: WorkSpeed[];
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

export interface ServerPower {
  cpuPowerW: number | null;
  cpuTempC: number | null;
  totalGpuPowerW: number | null;
  totalPowerW: number | null;
}
