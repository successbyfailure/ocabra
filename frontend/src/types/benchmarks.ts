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
      gpus?: Array<{ index: number; name?: string | null; totalVramMb?: number | null }>;
      nodes?: string[];
      /** Nodo remoto: si se pudo leer su tráfico propio durante la medida. */
      nodeTrafficVerified?: boolean;
      /** Dónde se sirvieron los modelos auxiliares de la suite (reranker, juez…). */
      auxiliaryNodes?: string[];
    };
    /** Nodo remoto donde se midió: nombre, versión de oCabra y GPU. */
    node?: {
      name?: string;
      ocabra?: string;
      online?: boolean;
      gpus?: Array<{ index?: number; name?: string | null; totalVramMb?: number | null }>;
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
  autoPause: boolean;
  /** Nodo de la federación donde se midió (null = este nodo). */
  node?: string | null;
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
  status: "queued" | "running" | "completed" | "failed";
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
