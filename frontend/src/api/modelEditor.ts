import { request } from "@/api/client"
import type {
  CapacityInfo,
  ConfigSchema,
  JsonObject,
  PreviewResult,
  RawModel,
} from "@/components/model-editor/types"
import type { EvictionSchedule, LoadPolicy } from "@/types"

const enc = encodeURIComponent

/** Llamadas que usa el editor de modelos. Trabajan con la configuración tal y
 *  como se guarda (snake_case), sin la vista camelCase del resto del cliente. */
export const modelEditorApi = {
  get: (modelId: string) => request<RawModel>("GET", `/ocabra/models/${enc(modelId)}`),

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

  /** PATCH en formato JSON Merge Patch: un null borra la clave guardada. */
  patch: (
    modelId: string,
    body: {
      displayName?: string
      loadPolicy?: LoadPolicy
      pinPermanent?: boolean
      preferredGpu?: number | null
      extraConfig?: JsonObject
      schedules?: EvictionSchedule[]
    },
  ) => request<RawModel>("PATCH", `/ocabra/models/${enc(modelId)}`, body),

  setMaintenance: (modelId: string, enabled: boolean) =>
    request<unknown>("POST", `/ocabra/models/${enc(modelId)}/maintenance`, { enabled }),

  createAlias: (profileId: string, body: { alias_id: string; preset_key: string | null; hidden: boolean }) =>
    request<unknown>("POST", `/ocabra/profiles/${enc(profileId)}/aliases`, body),

  deleteAlias: (profileId: string, aliasId: string) =>
    request<void>("DELETE", `/ocabra/profiles/${enc(profileId)}/aliases/${enc(aliasId)}`),

}
