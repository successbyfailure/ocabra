import type { BackendType, EvictionSchedule, LoadPolicy, ModelCapabilities, ModelStatus } from "@/types"

export type Json = null | boolean | number | string | Json[] | { [key: string]: Json }
export type JsonObject = { [key: string]: Json }

export type FieldType =
  | "int"
  | "float"
  | "bool"
  | "enum"
  | "string"
  | "text"
  | "int_list"
  | "float_list"
  | "json"
  | "string_list"

/** Una opción editable, tal y como la describe GET /models/{id}/config-schema. */
export interface SchemaField {
  key: string
  path: string[]
  type: FieldType
  label: string
  group: string
  help: string
  default: Json
  nullable: boolean
  reload: boolean
  memory: boolean
  advanced: boolean
  min?: number
  max?: number
  step?: number
  unit?: string
  widget?: string
  options?: { value: Json; label: string }[]
  requires?: Record<string, Json>
}

export interface ConfigSchema {
  backend: BackendType
  section: string
  groups: { id: string; label: string }[]
  fields: SchemaField[]
  vramOverride: boolean
  isLlm: boolean
  unknownKeys: string[]
}

/** GET /ocabra/models/{id} sin transformar: el editor trabaja con las claves
 *  tal y como se guardan (snake_case), para que el diff sea exacto. */
export interface RawModel {
  model_id: string
  backend_model_id: string
  backend_type: BackendType
  display_name: string
  status: ModelStatus
  runtime_phase?: string | null
  load_policy: LoadPolicy
  pin_permanent: boolean
  preferred_gpu: number | null
  current_gpu: number[]
  vram_used_mb: number
  vram_estimate_mb?: number
  ram_used_mb?: number
  disk_size_bytes: number | null
  capabilities: ModelCapabilities & Record<string, unknown>
  extra_config: JsonObject | null
  schedules?: EvictionSchedule[]
  error_message?: string | null
  loaded_at?: string | null
  last_request_at?: string | null
  release_date?: string | null
  last_updated?: string | null
  metadata_fetched_at?: string | null
  can_cancel_load?: boolean
  can_downsize?: boolean
  vision_state?: string | null
  live_context?: Record<string, unknown> | null
  runtime_slots?: number
}

/** Lo que se edita de un despliegue. */
export interface DeploymentDraft {
  loadPolicy: LoadPolicy
  pinPermanent: boolean
  preferredGpu: number | null
  extraConfig: JsonObject
  schedules: EvictionSchedule[]
}

export interface PreviewGpu {
  index: number
  mb: number
  breakdown: Record<string, number>
}

export interface MemoryEstimate {
  backend_type?: string
  gpu_index?: number | null
  total_vram_mb?: number | null
  free_vram_mb?: number | null
  requested_context_length?: number | null
  estimated_weights_mb?: number | null
  estimated_kv_cache_mb?: number | null
  estimated_max_context_length?: number | null
  estimated_max_total_context?: number | null
  model_loading_memory_mb?: number | null
  maximum_concurrency?: number | null
  fits_current_gpu?: boolean | null
  source?: string | null
  status?: string | null
  warning?: string | null
  notes?: string[]
  breakdown_mb?: Record<string, number>
}

export interface PreviewResult {
  errors: { key: string; label: string; message: string }[]
  unknownKeys: string[]
  estimate: MemoryEstimate
  gpuIndices: number[]
  gpus: PreviewGpu[]
  launch: { argv?: string[]; env?: Record<string, string>; visibleGpus?: number[]; error?: string } | null
}

export interface CapacityRow {
  slots: number
  max_context: number
  max_context_capped: number
}

export interface CapacityInfo {
  applicable: boolean
  weights_mb?: number
  overhead_mb?: number
  kv_dtype?: string
  target_gpu?: number
  note?: string
  concurrency_label?: string
  fits_free_now?: boolean
  arch?: {
    layers: number
    n_kv_heads: number
    native_context: number
    kv_mb_per_1k_tokens: number
  }
  capacity?: CapacityRow[]
  vram_curve?: { context: number; vram_mb: number }[]
  gpus?: { index: number; total_mb: number; free_mb: number }[]
}

/** Una entrada del diff del borrador, para el panel de cambios. */
export interface DraftChange {
  key: string
  label: string
  from: Json | undefined
  to: Json | undefined
  reload: boolean
  memory: boolean
}

/** GET /models/{id}/insights: lo medido del modelo y lo que lee su plantilla. */
export interface ModelInsights {
  rates: {
    prefillTps: number | null
    prefillSource: "historical" | "regression" | null
    generationTps: number | null
    typicalOutputTokens: number | null
  }
  typicalLoadMs: number | null
  template: { effortLevels: boolean; thinkingSwitch: boolean } | null
  profileIds: string[]
  maintenance: boolean
}
