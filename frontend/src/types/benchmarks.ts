// Tipos del dominio «benchmarks» (REF-15).
// Extraídos de types/index.ts; el barrel index.ts los reexporta con export *.

export type BenchmarkRunStatus =
  | "queued"
  | "running"
  | "pausing"
  | "paused"
  | "cancelling"
  | "completed"
  | "failed"
  | "cancelled";

export interface BenchmarkSuiteCatalog {
  name: string;
  description: string;
  resumable: boolean;
  needsPreparation: boolean;
}

export interface BenchmarkAppCatalog {
  app: string;
  description: string;
  defaultRevision: string;
  suites: BenchmarkSuiteCatalog[];
}

export interface BenchmarkAttempt {
  id: string;
  model: string;
  attemptNumber: number;
  status: "running" | "interrupted" | "contaminated" | "completed" | "failed";
  environment: {
    software?: {
      ocabra?: string;
      python?: string;
      scorer?: string;
      backends?: Array<{
        type: string;
        installedVersion?: string | null;
        packages?: Record<string, string>;
        ociDigest?: string | null;
        gitRef?: string | null;
        sourceCommit?: string | null;
        node?: string | null;
  sampling?: "benchmark" | "recommended";
  /** Pod propio: se pide, se prepara, se mide y se apaga si se quiere. */
  cloud?: BenchmarkCloudSpec | null;
  /** Más pruebas del mismo pedido (mismos modelos y nodo/pod). */
  battery?: BenchmarkBatteryItem[];
      }>;
    };
    benchmark?: {
      app?: string;
      suite?: string;
      requestedRevision?: string;
      commit?: string | null;
      repository?: string;
      runner?: string;
    };
    model?: {
      profile?: string;
      backends?: string[];
      runtimePresets?: string[];
      effectiveContexts?: number[];
    };
    hardware?: {
      gpus?: Array<{ index: number; name?: string | null; totalVramMb?: number | null; inferred?: boolean }>;
      nodes?: string[];
      /** Nodo remoto: si se pudo leer su tráfico propio durante la medida. */
      nodeTrafficVerified?: boolean;
      /** Dónde se sirvieron los modelos auxiliares de la suite (reranker, juez…). */
      auxiliaryNodes?: string[];
      /** De dónde salen backend y GPU de un intento remoto: del propio nodo o (peor) del hub. */
      source?: "node" | "hub";
    };
    /** Nodo remoto donde se midió: nombre, versión de oCabra y GPU. */
    node?: {
      name?: string;
      ocabra?: string;
      online?: boolean;
      gpus?: Array<{ index?: number; name?: string | null; totalVramMb?: number | null }>;
      ocabraAtEnd?: string;
      cuda?: { ok?: boolean; driver_cuda?: string | null; devices?: number | null; error?: string | null };
      cloud?: {
        provider?: string; podId?: string | null; gpuTypeId?: string | null; gpuCount?: number | null;
        cloudType?: string | null; image?: string | null; costPerHr?: number | null; dataCenterId?: string | null;
        readiness?: {
          wait_s?: number | null; boot_s?: number | null; connect_s?: number | null; prepare_s?: number | null;
          total_s?: number | null; ready_with_errors?: boolean; items?: Record<string, number | null>;
        } | null;
      } | null;
    };
    requests?: {
      count?: number;
      errors?: number;
      inputTokens?: number;
      outputTokens?: number;
    };
  };
  returnCode: number | null;
  externalRequests: Array<Record<string, unknown>>;
  error: string | null;
  startedAt: string;
  finishedAt: string | null;
  result: Record<string, unknown> | null;
}

export interface BenchmarkRun {
  id: string;
  app: string;
  suite: string;
  revision: string;
  models: string[];
  suiteArgs: string[];
  status: BenchmarkRunStatus;
  phase: string;
  scheduleId: string | null;
  isBaseline: boolean;
  /** Fallo conocido del arnés que invalida la nota (se mide con la versión de oCabra). */
  scoringIssue?: { reason: string; fixedIn: string } | null;
  cloudRequestId?: string | null;
  batchId?: string | null;
  cloudNodeId?: string | null;
  autoPause: boolean;
  /** Nodo de la federación donde se midió (null = este nodo). */
  node?: string | null;
  sampling?: "benchmark" | "recommended";
  /** Pod propio: se pide, se prepara, se mide y se apaga si se quiere. */
  cloud?: BenchmarkCloudSpec | null;
  /** Más pruebas del mismo pedido (mismos modelos y nodo/pod). */
  battery?: BenchmarkBatteryItem[];
  pauseReason: "manual" | "external_traffic" | null;
  currentModelIndex: number;
  currentModel: string | null;
  testProgress: {
    completed: number;
    total: number;
    unit: string;
    stage: string;
    updatedAt: string;
  } | null;
  result: Record<string, Record<string, unknown>>;
  contaminatedAttempts: number;
  error: string | null;
  externalActivityAt: string | null;
  createdAt: string;
  startedAt: string | null;
  finishedAt: string | null;
  attempts: BenchmarkAttempt[];
}

export interface BenchmarkSchedule {
  id: string;
  name: string;
  app: string;
  suite: string;
  revision: string;
  models: string[];
  suiteArgs: string[];
  autoPause: boolean;
  node?: string | null;
  sampling?: "benchmark" | "recommended";
  /** Pod propio: se pide, se prepara, se mide y se apaga si se quiere. */
  cloud?: BenchmarkCloudSpec | null;
  /** Más pruebas del mismo pedido (mismos modelos y nodo/pod). */
  battery?: BenchmarkBatteryItem[];
  enabled: boolean;
  weekdays: number[];
  timeOfDay: string;
  timezone: string;
  nextRunAt: string | null;
  lastRunAt: string | null;
  lastRunId: string | null;
  createdAt: string;
  updatedAt: string;
}

export interface BenchmarkScheduleInput {
  name: string;
  app: string;
  suite: string;
  revision?: string;
  models: string[];
  suiteArgs?: string[];
  autoPause?: boolean;
  node?: string | null;
  sampling?: "benchmark" | "recommended";
  /** Pod propio: se pide, se prepara, se mide y se apaga si se quiere. */
  cloud?: BenchmarkCloudSpec | null;
  /** Más pruebas del mismo pedido (mismos modelos y nodo/pod). */
  battery?: BenchmarkBatteryItem[];
  enabled?: boolean;
  weekdays: number[];
  timeOfDay: string;
  timezone: string;
}

// Backend modules (Bloque 15) — re-exported for convenience.

export type BenchmarkDiagnosisCategory =
  | "plantilla"
  | "sampling"
  | "limites"
  | "servido"
  | "modelo"
  | "no_reproduce"
  | "sin_verificar"

export interface BenchmarkDiagnosisCase {
  case_id: string;
  category: BenchmarkDiagnosisCategory;
  case_category: string;
  failure: string;
  evidence: string[];
}

export interface BenchmarkDiagnosisReport {
  model: string;
  app: string;
  suite: string;
  repeats: number;
  replayed_cases: number;
  findings: string[];
  summary: {
    total: number;
    counts: Record<BenchmarkDiagnosisCategory, number>;
    percent_of_failures: Record<BenchmarkDiagnosisCategory, number>;
  };
  params?: { differences?: Record<string, { efectivo: unknown; recomendado: unknown }> };
  reference?: { repo: string | null };
  cases: BenchmarkDiagnosisCase[];
  candidates: { train: number; eval: number };
}

export interface BenchmarkDiagnosis {
  id: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
  phase: string | null;
  progress: { completed: number; total: number; cases?: number } | null;
  error: string | null;
  createdAt: string;
  finishedAt: string | null;
  attemptId: string;
  runId: string;
  app: string;
  suite: string;
  model: string;
  report?: BenchmarkDiagnosisReport | null;
}


export interface BenchmarkCloudSpec {
  /** GPUs aceptables, por orden de preferencia. */
  gpu_type_ids: string[];
  gpu_count: number;
  cloud_type: "COMMUNITY" | "SECURE" | "BOTH" | null;
  backends: string[];
  recipes: string[];
  volume_id?: string | null;
  privacy_mode: "functional" | "zero_data";
  wait_minutes: number;
  ready_timeout_minutes: number;
  max_runtime_minutes: number;
  /** Disco del pod en GB; sin él, el de los ajustes. */
  container_disk_gb?: number | null;
  terminate_after: boolean;
}


export interface BenchmarkBatteryItem {
  app: string;
  suite: string;
  suite_args?: string[];
}
