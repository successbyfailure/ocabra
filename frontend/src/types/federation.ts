// Tipos del dominio «federation» (REF-15).
// Extraídos de types/index.ts; el barrel index.ts los reexporta con export *.

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
  /** Lo que el peer anuncia del modelo (chat, embeddings, rerank…). */
  capabilities?: Record<string, boolean>;
}

export interface FederationPeerLoad {
  active_requests: number;
  gpu_utilization_avg_pct: number;
}

/** http: we call its URL · tunnel_in: it dials us · tunnel_out: we dial it */
export type FederationConnectionMode = "http" | "tunnel_in" | "tunnel_out";

/** manual: legacy one-way peer · swarm: "Soy parte del enjambre" · rules: "Mi server mis reglas" */
export type FederationMode = "manual" | "swarm" | "rules";

export interface FederationTunnelInfo {
  remote_node_id: string | null;
  remote_node_name: string | null;
  remote_version: string | null;
  connected_at: number | null;
  active_streams: number;
  requests_sent: number;
  requests_served: number;
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
  connection_mode: FederationConnectionMode;
  mode: FederationMode;
  notes: string | null;
  remote_node_id: string | null;
  local_access_level: "inference" | "full" | null;
  join_state: "pending" | "denied" | "expired" | null;
  join_verification_code: string | null;
  /** The peer's privacy mode (heartbeat); null from nodes older than 0.92.1. */
  privacy_mode?: "functional" | "zero_data" | null;
  /** «rules» mode: our adjustments for this peer over the node defaults. */
  rules?: Partial<FederationRules> | null;
  /** What we apply to it (null = everything allowed: swarm). */
  effective_rules?: FederationRules | null;
  /** What it applies to us (its heartbeat). */
  their_rules?: FederationRules | null;
  dial_state: string | null;
  last_error: string | null;
  tunnel_connected: boolean;
  tunnel: FederationTunnelInfo | null;
  requests_proxied: number;
  /** Última vez que proxyeamos una petición a este peer (epoch segundos). */
  last_proxied_at?: number | null;
  created_at?: string | null;
  updated_at?: string | null;
}

export interface FederationNodeInfo {
  enabled: boolean;
  node_id: string | null;
  node_name: string | null;
  public_url: string | null;
  accept_join_requests: boolean;
  version: string;
  pending_join_requests: number;
}

export interface FederationJoinRequest {
  id: string;
  status: "pending" | "accepted" | "denied" | "expired";
  mode: FederationMode;
  remote_node_id: string;
  remote_node_name: string;
  remote_version: string | null;
  message: string | null;
  source_ip: string | null;
  gpus: { name?: string; total_vram_mb?: number }[];
  verification_code: string;
  via_invite: boolean;
  peer_id: string | null;
  created_at: string | null;
  expires_at: string | null;
  decided_at: string | null;
}

export interface FederationInvite {
  id: string;
  label: string;
  mode: FederationMode;
  max_uses: number;
  uses: number;
  expires_at: string | null;
  revoked: boolean;
  usable: boolean;
  created_at: string | null;
  /** Only in the create response. */
  code?: string;
  join_url?: string | null;
}

export interface CloudSettings {
  image: string;
  cloud_type: "COMMUNITY" | "SECURE";
  default_ttl_minutes: number;
  default_idle_minutes: number;
  monthly_budget_usd: number;
  container_disk_gb: number;
  pass_hf_token: boolean;
  hub_url: string;
}

export interface CloudOrphan {
  pod_id: string;
  name: string;
  status: string;
  cost_per_hr: number | null;
}

export interface CloudProviderInfo {
  provider: string;
  has_api_key: boolean;
  settings: CloudSettings;
  month_spend_usd?: number;
  orphans?: CloudOrphan[];
  last_reconcile_at?: number | null;
  last_error?: string | null;
}

export interface CloudGpuType {
  id: string;
  name: string;
  memory_gb: number | null;
  community: boolean;
  secure: boolean;
  price_community: number | null;
  price_secure: number | null;
  max_count: { community?: number; secure?: number };
  availability: "NONE" | "LOW" | "MEDIUM" | "HIGH";
}

export interface CloudNode {
  id: string;
  provider: string;
  provider_pod_id: string | null;
  name: string;
  status: "creating" | "starting" | "running" | "stopping" | "terminated" | "error";
  gpu_type_id: string;
  gpu_count: number;
  cloud_type: string;
  image: string;
  cost_per_hr: number | null;
  cost_so_far: number;
  max_runtime_minutes: number | null;
  idle_shutdown_minutes: number | null;
  terminates_at: string | null;
  peer_id: string | null;
  peer_name: string | null;
  peer_online: boolean;
  requests_proxied: number;
  preload_models: CloudPreloadItem[];
  network_volume_id?: string | null;
  data_center_id?: string | null;
  error: string | null;
  created_at: string | null;
  started_at: string | null;
  last_active_at: string | null;
  terminated_at: string | null;
  terminate_reason: string | null;
  /** What it was launched with (null for nodes launched before 0.101). */
  privacy_mode?: "functional" | "zero_data" | null;
}

export interface CloudReplicableModel {
  model_id: string;
  display_name: string;
  backend_type?: string;
  size_bytes: number | null;
  loaded: boolean;
  replicable: boolean;
  reason: string | null;
}

export interface CloudPreloadItem {
  model_id: string;
  /** ready: downloaded and registered, not loaded («solo descargar»). */
  status: "pending" | "installing" | "downloading" | "registering" | "loading" | "loaded" | "ready" | "error";
  progress?: number;
  /** A model captured from an earlier pod (not one of ours). */
  recipe_id?: string;
  /** "backend": only installs a backend (model_id = "backend:<type>"). */
  kind?: "backend";
  /** false: download only, load when asked for. */
  load?: boolean;
  error?: string;
  node_model_id?: string;
  /** Replication steps (backend install, downloads, register, load). */
  steps?: { kind: string; label?: string }[];
  step?: number;
  /** What the current step is working on (e.g. the HF repo being fetched). */
  phase?: string | null;
}

export interface CloudNodeLaunch {
  gpu_type_id: string;
  gpu_count?: number;
  cloud_type?: "COMMUNITY" | "SECURE";
  max_runtime_minutes?: number;
  idle_shutdown_minutes?: number;
  name?: string;
  models?: string[];
  /** Models captured from earlier pods (GET /cloud/recipes ids). */
  recipes?: string[];
  /** Ids from models/recipes to download but not load. */
  download_only?: string[];
  /** Backends to install first (the ones the models need are added anyway). */
  backends?: string[];
  /** Network volume to mount as the node's model/backend cache. */
  volume_id?: string;
  privacy_mode?: "functional" | "zero_data";
}

/** Petición de máquina que se reintenta hasta que hay capacidad. */
export interface CloudLaunchRequestCreate extends CloudNodeLaunch {
  /** Otras GPUs aceptables, por orden de preferencia. */
  alternatives?: string[];
  wait_minutes?: number;
  retry_interval_minutes?: number;
}

export interface CloudLaunchRequest {
  id: string;
  status: "waiting" | "launched" | "expired" | "cancelled" | "failed";
  gpuTypeIds: string[];
  gpuCount: number;
  cloudType: string | null;
  name: string | null;
  models: string[];
  recipes: string[];
  backends: string[];
  retryIntervalS: number;
  expiresAt: string | null;
  attempts: number;
  lastAttemptAt: string | null;
  lastError: string | null;
  cloudNodeId: string | null;
  launchedGpuTypeId: string | null;
  purpose: "manual" | "benchmark";
  benchmarkRunId: string | null;
  createdAt: string | null;
  finishedAt: string | null;
}

/** A model a pod had (set up there, e.g. by hand in its panel), offered again. */
export interface CloudRecipe {
  id: string;
  node_model_id: string;
  display_name: string;
  backend_type: string;
  /** "huggingface: repo · file", "ollama: tag". */
  source: string | null;
  size_bytes: number | null;
  context: number | null;
  parallel: number | null;
  vram_estimate_mb: number | null;
  load_policy?: "on_demand" | "warm" | "pin" | null;
  last_node_name: string | null;
  last_gpu_type_id: string | null;
  last_used_at: string | null;
  captured_at: string;
}

/** What the launch dialog takes along: our models and pod recipes. */
export interface CloudPreloadPick {
  models: string[];
  recipes: string[];
  /** Ids (of either) to download but not load. */
  downloadOnly: string[];
  /** Backends chosen to install on the node (besides what the models need). */
  backends: string[];
}

export interface CloudVolume {
  id: string;
  name: string;
  size_gb: number;
  data_center: string;
  type?: string | null;
  usd_per_month: number;
  /** Name of the live node using it, if any. */
  in_use_by?: string | null;
}

export interface CloudDataCenter {
  id: string;
  name: string;
  region?: string | null;
  gpus: { id: string; name: string; availability: string }[];
}

export interface FederationJoinOutgoing {
  hub_url: string;
  message?: string;
  invite_code?: string;
  name?: string;
  mode?: FederationMode;
}

export interface FederationPeerCreate {
  name: string;
  url: string;
  api_key: string;
  access_level: "inference" | "full";
  /** "http" (marcamos nosotros) o "tunnel_in" (el peer disca hacia aquí). */
  connection_mode?: FederationConnectionMode;
  notes?: string | null;
}

export interface FederationPeerUpdate {
  name?: string;
  url?: string;
  api_key?: string;
  access_level?: "inference" | "full";
  enabled?: boolean;
  notes?: string | null;
  mode?: "swarm" | "rules";
  rules?: Partial<FederationRules> | null;
}

/** «Mi servidor, mis reglas»: what a peer in rules mode may do on a node. */
export interface FederationRules {
  local_first: boolean;
  allow_loads: boolean;
  allow_evict: boolean;
  max_requests_per_hour: number | null;
  max_wh_per_month: number | null;
  allow_downloads: boolean;
  allow_admin: boolean;
  offered_models: string[] | null;
  hours: [number, number] | null;
}

export interface FederationNodeRules {
  base: Partial<FederationRules>;
  defaults: Partial<FederationRules>;
  effective: FederationRules;
}

export interface FederationRulesUsage {
  requests_last_hour: number;
  max_requests_per_hour: number | null;
  wh_this_month: number;
  max_wh_per_month: number | null;
}

export interface FederationTestResult {
  success: boolean;
  latency_ms: number | null;
  error?: string;
  node_info?: Record<string, unknown>;
}
