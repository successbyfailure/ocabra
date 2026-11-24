// Cliente tipado de /ocabra/* — dominio «stats» (REF-15).
// Extraído de api/client.ts; usa la capa HTTP única de api/http.ts.

import { request } from "@/api/http"
import { buildQuery } from "@/api/_helpers"
import type {
  BatchPriorityStatus,
  BatchVsInteractiveStats,
  ByApiKeyStats,
  ByGroupStats,
  ByUserStats,
  EnergyStats,
  FederationStats,
  GpuUsageStats,
  MyGroupStats,
  OverviewStats,
  PerformanceStats,
  RecentRequestsData,
  RequestStats,
  ServerPower,
  StatsParams,
  TokenStats,
  UserDetailStats,
} from "@/types"

export function toRoutingDecisions(raw: unknown): RoutingDecisionsResponse {
  const d = (raw ?? {}) as Record<string, unknown>;
  const routers = Array.isArray(d.routers) ? d.routers : [];
  return {
    routers: routers.map((r) => {
      const rd = (r ?? {}) as Record<string, unknown>;
      const targets = Array.isArray(rd.targets) ? rd.targets : [];
      return {
        routerProfileId: String(rd.router_profile_id ?? ""),
        total: Number(rd.total ?? 0),
        targets: targets.map((t) => {
          const td = (t ?? {}) as Record<string, unknown>;
          return {
            modelId: String(td.model_id ?? ""),
            count: Number(td.count ?? 0),
            okCount: Number(td.ok_count ?? 0),
            avgDurationMs: Number(td.avg_duration_ms ?? 0),
            share: Number(td.share ?? 0),
          };
        }),
      };
    }),
  };
}


export type RoutingDecisionsResponse = {
  routers: {
    routerProfileId: string;
    total: number;
    targets: {
      modelId: string;
      count: number;
      okCount: number;
      avgDurationMs: number;
      share: number;
    }[];
  }[];
};


export const statsApiModule = {
  routingStats: (params: StatsParams) => {
    const query = buildQuery({ from: params.from, to: params.to });
    return request<unknown>("GET", `/ocabra/stats/routing${query}`).then(
      toRoutingDecisions,
    );
  },

  stats: {
    requests: (params: StatsParams) => {
      const query = buildQuery({
        from: params.from,
        to: params.to,
        model_id: params.modelId,
      });
      return request<RequestStats>("GET", `/ocabra/stats/requests${query}`);
    },
    energy: (params: StatsParams) => {
      const query = buildQuery({ from: params.from, to: params.to });
      return request<EnergyStats>("GET", `/ocabra/stats/energy${query}`);
    },
    gpu: (params: StatsParams & { activeThresholdPct?: number }) => {
      const query = buildQuery({
        from: params.from,
        to: params.to,
        activeThresholdPct: params.activeThresholdPct,
      });
      return request<GpuUsageStats>("GET", `/ocabra/stats/gpu${query}`);
    },
    tokens: (params: StatsParams) => {
      const query = buildQuery({
        from: params.from,
        to: params.to,
        model_id: params.modelId,
        allTime: params.allTime,
        includeSeries: params.includeSeries,
      });
      return request<TokenStats>("GET", `/ocabra/stats/tokens${query}`);
    },
    performance: (params: StatsParams) => {
      const query = buildQuery({
        from: params.from,
        to: params.to,
        model_id: params.modelId,
      });
      return request<PerformanceStats>(
        "GET",
        `/ocabra/stats/performance${query}`,
      );
    },
    overview: (params: StatsParams) => {
      const query = buildQuery({
        from: params.from,
        to: params.to,
        model_id: params.modelId,
      });
      return request<OverviewStats>("GET", `/ocabra/stats/overview${query}`);
    },
    recent: (limit = 20) =>
      request<RecentRequestsData>("GET", `/ocabra/stats/recent?limit=${limit}`),
    cancelRequest: (requestId: string) =>
      request<{ cancelled: boolean; requestId: string; hadTask: boolean }>(
        "POST",
        `/ocabra/stats/requests/${encodeURIComponent(requestId)}/cancel`,
      ),
    myRecent: (limit = 20) =>
      request<RecentRequestsData>(
        "GET",
        `/ocabra/stats/my-recent?limit=${limit}`,
      ),
    byUser: (params: StatsParams) => {
      const query = buildQuery({ from: params.from, to: params.to });
      return request<ByUserStats>("GET", `/ocabra/stats/by-user${query}`);
    },
    byGroup: (params: StatsParams) => {
      const query = buildQuery({ from: params.from, to: params.to });
      return request<ByGroupStats>("GET", `/ocabra/stats/by-group${query}`);
    },
    my: (params: StatsParams) => {
      const query = buildQuery({
        from: params.from,
        to: params.to,
        model_id: params.modelId,
      });
      return request<OverviewStats>("GET", `/ocabra/stats/my${query}`);
    },
    myGroup: (params: StatsParams) => {
      const query = buildQuery({ from: params.from, to: params.to });
      return request<MyGroupStats>("GET", `/ocabra/stats/my-group${query}`);
    },
    byApiKey: (params: StatsParams) => {
      const query = buildQuery({ from: params.from, to: params.to });
      return request<ByApiKeyStats>("GET", `/ocabra/stats/by-api-key${query}`);
    },
    userDetail: (userId: string, params: StatsParams) => {
      const query = buildQuery({ from: params.from, to: params.to });
      return request<UserDetailStats>(
        "GET",
        `/ocabra/stats/user/${encodeURIComponent(userId)}/detail${query}`,
      );
    },
    federation: (params: StatsParams) => {
      const query = buildQuery({ from: params.from, to: params.to });
      return request<FederationStats>(
        "GET",
        `/ocabra/stats/federation${query}`,
      );
    },
    batchVsInteractive: (params: StatsParams) => {
      const query = buildQuery({ from: params.from, to: params.to });
      return request<BatchVsInteractiveStats>(
        "GET",
        `/ocabra/stats/batch-vs-interactive${query}`,
      );
    },
    batchPriorityStatus: () =>
      request<BatchPriorityStatus>(
        "GET",
        `/ocabra/stats/batch-priority-status`,
      ),
    serverPower: async (): Promise<ServerPower> => {
      const raw = await request<Record<string, unknown>>(
        "GET",
        `/ocabra/stats/server-power`,
      );
      return {
        cpuPowerW: raw.cpuPowerW ?? raw.cpu_power_w ?? null,
        cpuTempC: raw.cpuTempC ?? raw.cpu_temp_c ?? null,
        totalGpuPowerW: raw.totalGpuPowerW ?? raw.total_gpu_power_w ?? null,
        totalPowerW: raw.totalPowerW ?? raw.total_power_w ?? null,
      } as ServerPower;
    },
  },

};
