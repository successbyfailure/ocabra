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
  enabled?: boolean;
  weekdays: number[];
  timeOfDay: string;
  timezone: string;
}

// Backend modules (Bloque 15) — re-exported for convenience.
