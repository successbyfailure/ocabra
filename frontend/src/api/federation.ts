// Cliente tipado de /ocabra/* — dominio «federation» (REF-15).
// Extraído de api/client.ts; usa la capa HTTP única de api/http.ts.

import { request } from "@/api/http"
import { isRecord } from "@/api/_helpers"
import type {
  CloudGpuType,
  CloudNode,
  CloudNodeLaunch,
  CloudProviderInfo,
  CloudReplicableModel,
  CloudSettings,
  FederationConnectionMode,
  FederationInvite,
  FederationJoinOutgoing,
  FederationJoinRequest,
  FederationMode,
  FederationNodeInfo,
  FederationPeer,
  FederationPeerCreate,
  FederationPeerUpdate,
  FederationTestResult,
} from "@/types"

export function toFederationPeer(raw: unknown): FederationPeer {
  const d = isRecord(raw) ? raw : {};
  return {
    // The API returns ``id``; reading only peer_id/peerId left every peer with
    // an empty id, so edit/delete/test from the UI hit /peers/ and failed.
    peer_id: String(d.id ?? d.peer_id ?? d.peerId ?? ""),
    name: String(d.name ?? ""),
    url: String(d.url ?? ""),
    access_level: String(d.access_level ?? d.accessLevel ?? "inference") as
      "inference" | "full",
    enabled: Boolean(d.enabled ?? true),
    online: Boolean(d.online ?? false),
    last_heartbeat: (d.last_heartbeat ?? d.lastHeartbeat ?? null) as
      string | null,
    latency_ms:
      d.latency_ms != null
        ? Number(d.latency_ms)
        : d.latencyMs != null
          ? Number(d.latencyMs)
          : undefined,
    gpus: Array.isArray(d.gpus)
      ? d.gpus.map((g) => {
          const gpu = isRecord(g) ? g : {};
          return {
            index: Number(gpu.index ?? 0),
            name: String(gpu.name ?? "GPU"),
            total_vram_mb: Number(gpu.total_vram_mb ?? gpu.totalVramMb ?? 0),
            free_vram_mb: Number(gpu.free_vram_mb ?? gpu.freeVramMb ?? 0),
          };
        })
      : [],
    models: Array.isArray(d.models)
      ? d.models.map((m) => {
          const model = isRecord(m) ? m : {};
          return {
            model_id: String(model.model_id ?? model.modelId ?? ""),
            status: String(model.status ?? "unknown"),
            profiles: Array.isArray(model.profiles)
              ? model.profiles.map(String)
              : [],
          };
        })
      : [],
    load: (() => {
      const load = isRecord(d.load) ? d.load : {};
      return {
        active_requests: Number(
          load.active_requests ?? load.activeRequests ?? 0,
        ),
        gpu_utilization_avg_pct: Number(
          load.gpu_utilization_avg_pct ?? load.gpuUtilizationAvgPct ?? 0,
        ),
      };
    })(),
    connection_mode: String(d.connection_mode ?? "http") as FederationConnectionMode,
    mode: String(d.mode ?? "manual") as FederationMode,
    notes: (d.notes ?? null) as string | null,
    remote_node_id: (d.remote_node_id ?? null) as string | null,
    local_access_level: (d.local_access_level ?? null) as "inference" | "full" | null,
    join_state: (d.join_state ?? null) as FederationPeer["join_state"],
    join_verification_code: (d.join_verification_code ?? null) as string | null,
    privacy_mode: (d.privacy_mode ?? d.privacyMode ?? null) as "functional" | "zero_data" | null,
    rules: (isRecord(d.rules) ? d.rules : null) as import("@/types").FederationPeer["rules"],
    effective_rules: (isRecord(d.effective_rules) ? d.effective_rules : null) as
      import("@/types").FederationRules | null,
    their_rules: (isRecord(d.their_rules) ? d.their_rules : null) as
      import("@/types").FederationRules | null,
    dial_state: (d.dial_state ?? null) as string | null,
    last_error: (d.last_error ?? null) as string | null,
    tunnel_connected: Boolean(d.tunnel_connected ?? false),
    requests_proxied: Number(d.requests_proxied ?? 0),
    last_proxied_at: d.last_proxied_at != null ? Number(d.last_proxied_at) : null,
    created_at: (d.created_at ?? null) as string | null,
    updated_at: (d.updated_at ?? null) as string | null,
    tunnel: isRecord(d.tunnel)
      ? {
          remote_node_id: (d.tunnel.remote_node_id ?? null) as string | null,
          remote_node_name: (d.tunnel.remote_node_name ?? null) as string | null,
          remote_version: (d.tunnel.remote_version ?? null) as string | null,
          connected_at: d.tunnel.connected_at != null ? Number(d.tunnel.connected_at) : null,
          active_streams: Number(d.tunnel.active_streams ?? 0),
          requests_sent: Number(d.tunnel.requests_sent ?? 0),
          requests_served: Number(d.tunnel.requests_served ?? 0),
        }
      : null,
  };
}


export function toFederationTestResult(raw: unknown): FederationTestResult {
  const d = isRecord(raw) ? raw : {};
  return {
    success: Boolean(d.success ?? false),
    latency_ms: d.latency_ms != null ? Number(d.latency_ms) : null,
    error: (d.error ?? undefined) as string | undefined,
    node_info: isRecord(d.node_info)
      ? (d.node_info as Record<string, unknown>)
      : undefined,
  };
}


export const federationApiModule = {
  federation: {
    getPeers: async (): Promise<FederationPeer[]> =>
      (await request<unknown[]>("GET", "/ocabra/federation/peers")).map(
        toFederationPeer,
      ),

    addPeer: async (data: FederationPeerCreate): Promise<FederationPeer> =>
      toFederationPeer(
        await request<unknown>("POST", "/ocabra/federation/peers", {
          name: data.name,
          url: data.url,
          api_key: data.api_key,
          access_level: data.access_level,
          ...(data.connection_mode !== undefined && {
            connection_mode: data.connection_mode,
          }),
          ...(data.notes !== undefined && { notes: data.notes }),
        }),
      ),

    getRules: async (): Promise<import("@/types").FederationNodeRules> =>
      request<import("@/types").FederationNodeRules>("GET", "/ocabra/federation/rules"),
    setRules: async (
      rules: Partial<import("@/types").FederationRules>,
    ): Promise<import("@/types").FederationNodeRules> =>
      request<import("@/types").FederationNodeRules>("PUT", "/ocabra/federation/rules", rules),
    /** Where a model's requests are served (core/federation_policy.py). */
    getStrategy: async (modelId: string): Promise<import("@/types").ModelFederationStrategy> =>
      request<import("@/types").ModelFederationStrategy>(
        "GET",
        `/ocabra/federation/strategies?model_id=${encodeURIComponent(modelId)}`,
      ),
    setStrategy: async (
      modelId: string,
      strategy: Pick<import("@/types").ModelFederationStrategy, "strategy" | "nodes" | "fallback">,
    ): Promise<import("@/types").ModelFederationStrategy> =>
      request<import("@/types").ModelFederationStrategy>("PUT", "/ocabra/federation/strategies", {
        model_id: modelId,
        ...strategy,
      }),
    rulesUsage: async (id: string): Promise<import("@/types").FederationRulesUsage> =>
      request<import("@/types").FederationRulesUsage>(
        "GET",
        `/ocabra/federation/peers/${encodeURIComponent(id)}/rules-usage`,
      ),
    updatePeer: async (
      id: string,
      data: FederationPeerUpdate,
    ): Promise<FederationPeer> =>
      toFederationPeer(
        await request<unknown>(
          "PATCH",
          `/ocabra/federation/peers/${encodeURIComponent(id)}`,
          {
            ...(data.name !== undefined && { name: data.name }),
            ...(data.url !== undefined && { url: data.url }),
            ...(data.api_key !== undefined && { api_key: data.api_key }),
            ...(data.access_level !== undefined && {
              access_level: data.access_level,
            }),
            ...(data.enabled !== undefined && { enabled: data.enabled }),
            ...(data.notes !== undefined && { notes: data.notes }),
            // «Le sirvo como» and its rules: dropped here until 0.98.1, so
            // the server got an empty PATCH and nothing changed.
            ...(data.mode !== undefined && { mode: data.mode }),
            ...(data.rules !== undefined && { rules: data.rules }),
          },
        ),
      ),

    deletePeer: async (id: string): Promise<void> => {
      await request<unknown>(
        "DELETE",
        `/ocabra/federation/peers/${encodeURIComponent(id)}`,
      );
    },

    testPeer: async (id: string): Promise<FederationTestResult> =>
      toFederationTestResult(
        await request<unknown>(
          "POST",
          `/ocabra/federation/peers/${encodeURIComponent(id)}/test`,
        ),
      ),

    getNode: async (): Promise<FederationNodeInfo> =>
      request<FederationNodeInfo>("GET", "/ocabra/federation/node"),

    getJoinRequests: async (includeDecided = false): Promise<FederationJoinRequest[]> =>
      request<FederationJoinRequest[]>(
        "GET",
        `/ocabra/federation/join-requests${includeDecided ? "?include_decided=true" : ""}`,
      ),

    acceptJoinRequest: async (
      id: string,
      data: { name?: string; notes?: string } = {},
    ): Promise<FederationPeer> =>
      toFederationPeer(
        await request<unknown>(
          "POST",
          `/ocabra/federation/join-requests/${encodeURIComponent(id)}/accept`,
          data,
        ),
      ),

    denyJoinRequest: async (id: string): Promise<void> => {
      await request<unknown>(
        "POST",
        `/ocabra/federation/join-requests/${encodeURIComponent(id)}/deny`,
      );
    },

    getInvites: async (): Promise<FederationInvite[]> =>
      request<FederationInvite[]>("GET", "/ocabra/federation/invites"),

    createInvite: async (data: {
      label: string;
      max_uses?: number;
      expires_in_hours?: number | null;
    }): Promise<FederationInvite> =>
      request<FederationInvite>("POST", "/ocabra/federation/invites", {
        mode: "swarm",
        ...data,
      }),

    revokeInvite: async (id: string): Promise<void> => {
      await request<unknown>(
        "DELETE",
        `/ocabra/federation/invites/${encodeURIComponent(id)}`,
      );
    },

    join: async (data: FederationJoinOutgoing): Promise<FederationPeer> =>
      toFederationPeer(
        await request<unknown>("POST", "/ocabra/federation/join", {
          mode: "swarm",
          ...data,
        }),
      ),

    cloud: {
      getRunpod: async (): Promise<CloudProviderInfo> =>
        request<CloudProviderInfo>("GET", "/ocabra/cloud/runpod"),
      updateRunpod: async (data: {
        api_key?: string;
        settings?: Partial<CloudSettings>;
      }): Promise<CloudProviderInfo> =>
        request<CloudProviderInfo>("PUT", "/ocabra/cloud/runpod", data),
      gpuTypes: async (): Promise<CloudGpuType[]> =>
        request<CloudGpuType[]>("GET", "/ocabra/cloud/runpod/gpu-types"),
      nodes: async (): Promise<CloudNode[]> =>
        request<CloudNode[]>("GET", "/ocabra/cloud/nodes"),
      launch: async (data: CloudNodeLaunch): Promise<CloudNode> =>
        request<CloudNode>("POST", "/ocabra/cloud/nodes", data),
      terminate: async (id: string): Promise<CloudNode> =>
        request<CloudNode>("DELETE", `/ocabra/cloud/nodes/${encodeURIComponent(id)}`),
      replicableModels: async (): Promise<CloudReplicableModel[]> =>
        request<CloudReplicableModel[]>("GET", "/ocabra/cloud/replicable-models"),
      addModels: async (
        id: string,
        models: string[],
        opts: { recipes?: string[]; download_only?: string[]; backends?: string[] } = {},
      ): Promise<CloudNode> =>
        request<CloudNode>("POST", `/ocabra/cloud/nodes/${encodeURIComponent(id)}/models`, {
          models,
          ...opts,
        }),
      /** Models captured from earlier pods (core/cloud/recipes.py). */
      recipes: async (): Promise<import("@/types").CloudRecipe[]> =>
        request<import("@/types").CloudRecipe[]>("GET", "/ocabra/cloud/recipes"),
      /** A llama.cpp model from HF (.gguf) or Ollama, picked in the launch dialog. */
      createRecipe: async (body: {
        source: "huggingface" | "ollama";
        model_ref: string;
        artifact?: string | null;
        display_name?: string | null;
        size_bytes?: number | null;
        context?: number | null;
        parallel?: number | null;
        load_policy?: "on_demand" | "warm" | "pin";
        backend_type?: string;
      }): Promise<import("@/types").CloudRecipe> =>
        request<import("@/types").CloudRecipe>("POST", "/ocabra/cloud/recipes", body),
      updateRecipe: async (
        id: string,
        changes: {
          context?: number;
          parallel?: number;
          load_policy?: "on_demand" | "warm" | "pin";
          display_name?: string;
        },
      ): Promise<import("@/types").CloudRecipe> =>
        request<import("@/types").CloudRecipe>(
          "PATCH",
          `/ocabra/cloud/recipes/${encodeURIComponent(id)}`,
          changes,
        ),
      deleteRecipe: async (id: string): Promise<void> => {
        await request<unknown>("DELETE", `/ocabra/cloud/recipes/${encodeURIComponent(id)}`)
      },
      volumes: async (): Promise<import("@/types").CloudVolume[]> =>
        request<import("@/types").CloudVolume[]>("GET", "/ocabra/cloud/volumes"),
      dataCenters: async (): Promise<import("@/types").CloudDataCenter[]> =>
        request<import("@/types").CloudDataCenter[]>("GET", "/ocabra/cloud/datacenters"),
      createVolume: async (data: {
        name: string;
        size_gb: number;
        data_center: string;
      }): Promise<import("@/types").CloudVolume> =>
        request<import("@/types").CloudVolume>("POST", "/ocabra/cloud/volumes", data),
      resizeVolume: async (id: string, sizeGb: number): Promise<import("@/types").CloudVolume> =>
        request<import("@/types").CloudVolume>(
          "PATCH",
          `/ocabra/cloud/volumes/${encodeURIComponent(id)}`,
          { size_gb: sizeGb },
        ),
      deleteVolume: async (id: string): Promise<void> => {
        await request<unknown>("DELETE", `/ocabra/cloud/volumes/${encodeURIComponent(id)}`);
      },
      terminateOrphan: async (podId: string): Promise<void> => {
        await request<unknown>(
          "POST",
          `/ocabra/cloud/orphans/${encodeURIComponent(podId)}/terminate`,
        );
      },
    },

    joinCheck: async (id: string): Promise<FederationPeer> =>
      toFederationPeer(
        await request<unknown>(
          "POST",
          `/ocabra/federation/peers/${encodeURIComponent(id)}/join-check`,
        ),
      ),
  },
};
