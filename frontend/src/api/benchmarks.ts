// Cliente tipado de /ocabra/* — dominio «benchmarks» (REF-15).
// Extraído de api/client.ts; usa la capa HTTP única de api/http.ts.

import { request } from "@/api/http"
import { buildQuery } from "@/api/_helpers"
import type {
  BenchmarkAppCatalog,
  BenchmarkDiagnosis,
  BenchmarkRun,
  BenchmarkSchedule,
  BenchmarkScheduleInput,
} from "@/types"

export const benchmarksApiModule = {
  benchmarks: {
    catalog: () =>
      request<BenchmarkAppCatalog[]>("GET", "/ocabra/benchmarks/catalog"),
    list: (limit = 100) =>
      request<BenchmarkRun[]>(
        "GET",
        `/ocabra/benchmarks/runs${buildQuery({ limit })}`,
      ),
    get: (runId: string) =>
      request<BenchmarkRun>(
        "GET",
        `/ocabra/benchmarks/runs/${encodeURIComponent(runId)}`,
      ),
    create: (payload: {
      app: string;
      suite: string;
      models: string[];
      revision?: string;
      suiteArgs?: string[];
      autoPause?: boolean;
      node?: string | null;
      sampling?: "benchmark" | "recommended";
    }) =>
      request<BenchmarkRun>("POST", "/ocabra/benchmarks/runs", {
        app: payload.app,
        suite: payload.suite,
        models: payload.models,
        revision: payload.revision || null,
        suite_args: payload.suiteArgs ?? [],
        auto_pause: payload.autoPause ?? true,
        node: payload.node || null,
        sampling: payload.sampling ?? "benchmark",
      }),
    pause: (runId: string) =>
      request<BenchmarkRun>(
        "POST",
        `/ocabra/benchmarks/runs/${encodeURIComponent(runId)}/pause`,
      ),
    resume: (runId: string) =>
      request<BenchmarkRun>(
        "POST",
        `/ocabra/benchmarks/runs/${encodeURIComponent(runId)}/resume`,
      ),
    retry: (runId: string) =>
      request<BenchmarkRun>(
        "POST",
        `/ocabra/benchmarks/runs/${encodeURIComponent(runId)}/retry`,
      ),
    baseline: (runId: string) =>
      request<BenchmarkRun>(
        "POST",
        `/ocabra/benchmarks/runs/${encodeURIComponent(runId)}/baseline`,
      ),
    cancel: (runId: string) =>
      request<BenchmarkRun>(
        "POST",
        `/ocabra/benchmarks/runs/${encodeURIComponent(runId)}/cancel`,
      ),
    diagnoses: {
      list: (attemptId?: string) =>
        request<BenchmarkDiagnosis[]>(
          "GET",
          `/ocabra/benchmarks/diagnoses${buildQuery({ attempt_id: attemptId })}`,
        ),
      get: (diagnosisId: string) =>
        request<BenchmarkDiagnosis>(
          "GET",
          `/ocabra/benchmarks/diagnoses/${encodeURIComponent(diagnosisId)}`,
        ),
      create: (
        attemptId: string,
        payload: { repeats?: number; maxCases?: number; compareModel?: string | null; allowLoad?: boolean } = {},
      ) =>
        request<BenchmarkDiagnosis>(
          "POST",
          `/ocabra/benchmarks/attempts/${encodeURIComponent(attemptId)}/diagnoses`,
          {
            repeats: payload.repeats ?? 3,
            max_cases: payload.maxCases ?? 30,
            compare_model: payload.compareModel || null,
            allow_load: payload.allowLoad ?? false,
          },
        ),
      cancel: (diagnosisId: string) =>
        request<BenchmarkDiagnosis>(
          "POST",
          `/ocabra/benchmarks/diagnoses/${encodeURIComponent(diagnosisId)}/cancel`,
        ),
      remove: (diagnosisId: string) =>
        request<void>("DELETE", `/ocabra/benchmarks/diagnoses/${encodeURIComponent(diagnosisId)}`),
      fileUrl: (diagnosisId: string, name: string) =>
        `/ocabra/benchmarks/diagnoses/${encodeURIComponent(diagnosisId)}/files/${encodeURIComponent(name)}`,
    },
    log: (runId: string, lines = 500) =>
      request<{ lines: string[]; attemptId: string | null }>(
        "GET",
        `/ocabra/benchmarks/runs/${encodeURIComponent(runId)}/log${buildQuery({ lines })}`,
      ),
    schedules: {
      list: () => request<BenchmarkSchedule[]>("GET", "/ocabra/benchmarks/schedules"),
      create: (payload: BenchmarkScheduleInput) =>
        request<BenchmarkSchedule>("POST", "/ocabra/benchmarks/schedules", {
          name: payload.name, app: payload.app, suite: payload.suite,
          revision: payload.revision || null, models: payload.models,
          suite_args: payload.suiteArgs ?? [], auto_pause: payload.autoPause ?? true,
          node: payload.node || null,
          sampling: payload.sampling ?? "benchmark",
          enabled: payload.enabled ?? false, weekdays: payload.weekdays,
          time_of_day: payload.timeOfDay, timezone: payload.timezone,
        }),
      update: (scheduleId: string, payload: BenchmarkScheduleInput) =>
        request<BenchmarkSchedule>(
          "PUT", `/ocabra/benchmarks/schedules/${encodeURIComponent(scheduleId)}`, {
            name: payload.name, app: payload.app, suite: payload.suite,
            revision: payload.revision || null, models: payload.models,
            suite_args: payload.suiteArgs ?? [], auto_pause: payload.autoPause ?? true,
            node: payload.node || null,
            sampling: payload.sampling ?? "benchmark",
            enabled: payload.enabled ?? false, weekdays: payload.weekdays,
            time_of_day: payload.timeOfDay, timezone: payload.timezone,
          },
        ),
      remove: (scheduleId: string) => request<void>(
        "DELETE", `/ocabra/benchmarks/schedules/${encodeURIComponent(scheduleId)}`,
      ),
      run: (scheduleId: string) => request<BenchmarkRun>(
        "POST", `/ocabra/benchmarks/schedules/${encodeURIComponent(scheduleId)}/run`,
      ),
    },
  },

};
