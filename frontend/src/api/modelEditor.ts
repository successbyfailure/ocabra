import { request } from "@/api/client"
import type {
  CapacityInfo,
  ConfigSchema,
  ModelInsights,
  JsonObject,
  PreviewResult,
  RawModel,
} from "@/components/model-editor/types"
import type { BackendType, EvictionSchedule, LoadPolicy } from "@/types"

const enc = encodeURIComponent

export interface AutoSetupRung {
  key: string
  label: string
  gpus: number[]
  context: number
  /** Opciones de llama.cpp del peldaño: `n_cpu_moe` (capas con los expertos
   *  en RAM), `gpu_layers` (capas en GPU; 0 = todo en CPU) y el reparto. */
  placement: Record<string, unknown>
  vram_by_gpu: Record<string, number>
  host_mb: number
}

export interface AutoSetupResult {
  rungs: AutoSetupRung[]
  needsCpu: boolean
  warnings: string[]
  budgets: { index: number; name: string; total_mb: number; budget_mb: number }[]
  applied: string[]
}

/** Llamadas que usa el editor de modelos. Trabajan con la configuración tal y
 *  como se guarda (snake_case), sin la vista camelCase del resto del cliente. */
export const modelEditorApi = {
  get: (modelId: string) => request<RawModel>("GET", `/ocabra/models/${enc(modelId)}`),

  insights: (modelId: string) => request<ModelInsights>("GET", `/ocabra/models/${enc(modelId)}/insights`),

  schema: (modelId: string) =>
    request<ConfigSchema>("GET", `/ocabra/models/${enc(modelId)}/config-schema`),

  preview: (modelId: string, body: { preferredGpu: number | null; extraConfig: JsonObject }) =>
    request<PreviewResult>("POST", `/ocabra/models/${enc(modelId)}/preview`, body),

  capacity: (
    modelId: string,
    params: { gpu?: number | null; slots?: number[]; kvDtype?: string; context?: number | null },
  ) => {
    const q = new URLSearchParams()
    if (params.gpu != null) q.set("gpu", String(params.gpu))
    if (params.slots?.length) q.set("slots", params.slots.join(","))
    if (params.kvDtype) q.set("kv_dtype", params.kvDtype)
    if (params.context) q.set("context", String(params.context))
    const qs = q.toString()
    return request<CapacityInfo>("GET", `/ocabra/models/${enc(modelId)}/capacity${qs ? `?${qs}` : ""}`)
  },

  /** PATCH en formato JSON Merge Patch: un null borra la clave guardada.
   *  El backend (ModelPatch) acepta también backendType y vramEstimateMb. */
  patch: (
    modelId: string,
    body: {
      displayName?: string
      backendType?: BackendType
      loadPolicy?: LoadPolicy
      pinPermanent?: boolean
      preferredGpu?: number | null
      vramEstimateMb?: number | null
      extraConfig?: JsonObject
      schedules?: EvictionSchedule[]
    },
  ) => request<RawModel>("PATCH", `/ocabra/models/${enc(modelId)}`, body),

  ollamaSettings: () =>
    request<{
      numParallel: number
      maxContext: number
      running: { numParallel: number | null; maxContext: number | null } | null
      pending: boolean
      applyStatus: string
      applyDetail: string | null
    }>("GET", "/ocabra/ollama/server/settings"),

  applyOllamaSettings: () => request<{ status: string; detail: string | null }>("POST", "/ocabra/ollama/server/apply-settings"),

  /** Carga el modelo tal y como está configurado y le hace una petición corta. */
  smokeTest: (modelId: string, cold: boolean) =>
    request<{
      loaded: boolean
      load_ms: number | null
      request_kind: string | null
      request_ok: boolean | null
      request_ms: number | null
      tokens_per_second: number | null
      error: string | null
    }>("POST", `/ocabra/models/${enc(modelId)}/smoke-test`, { cold }),

  /** Planifica los peldaños desde la proyección de memoria de llama.cpp; con
   *  `apply` reemplaza los presets del perfil por defecto. */
  autoSetup: (modelId: string, body: { gpus: number[]; cpu: boolean; apply: boolean }) =>
    request<AutoSetupResult>("POST", `/ocabra/models/${enc(modelId)}/auto-setup`, body),

  setMaintenance: (modelId: string, enabled: boolean) =>
    request<unknown>("POST", `/ocabra/models/${enc(modelId)}/maintenance`, { enabled }),

  createAlias: (profileId: string, body: { alias_id: string; preset_key: string | null; hidden: boolean }) =>
    request<unknown>("POST", `/ocabra/profiles/${enc(profileId)}/aliases`, body),

  deleteAlias: (profileId: string, aliasId: string) =>
    request<void>("DELETE", `/ocabra/profiles/${enc(profileId)}/aliases/${enc(aliasId)}`),

}
