// Cliente tipado de /ocabra/* — dominio «benchmarks» (REF-15).
// Extraído de api/client.ts; usa la capa HTTP única de api/http.ts.

import { request } from "@/api/http"
import { buildQuery } from "@/api/_helpers"
import type {
  BenchmarkAppCatalog,
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
    }) =>
      request<BenchmarkRun>("POST", "/ocabra/benchmarks/runs", {
        app: payload.app,
        suite: payload.suite,
        models: payload.models,
        revision: payload.revision || null,
        suite_args: payload.suiteArgs ?? [],
        auto_pause: payload.autoPause ?? true,
        node: payload.node || null,
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
