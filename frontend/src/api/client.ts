// Barrel de compatibilidad del cliente de API (REF-15).
//
// api/client.ts se troceó por dominios (api/models.ts, api/federation.ts, …),
// todos sobre la capa HTTP única de api/http.ts. Este fichero recompone el
// objeto ``api`` y reexporta los mismos símbolos que los sitios de llamada ya
// importaban desde "@/api/client", para no tocar ni una sola importación.

export { request } from "@/api/http"

export * from "@/api/gpus"
export * from "@/api/models"
export * from "@/api/registry"
export * from "@/api/stats"
export * from "@/api/system"
export * from "@/api/batches"
export * from "@/api/benchmarks"
export * from "@/api/backends"
export * from "@/api/services"
export * from "@/api/auth"
export * from "@/api/federation"

import { gpusApiModule } from "@/api/gpus"
import { modelsApiModule } from "@/api/models"
import { registryApiModule } from "@/api/registry"
import { statsApiModule } from "@/api/stats"
import { systemApiModule } from "@/api/system"
import { batchesApiModule } from "@/api/batches"
import { benchmarksApiModule } from "@/api/benchmarks"
import { backendsApiModule } from "@/api/backends"
import { servicesApiModule } from "@/api/services"
import { authApiModule } from "@/api/auth"
import { federationApiModule } from "@/api/federation"

export const api = {
  ...gpusApiModule,
  ...modelsApiModule,
  ...registryApiModule,
  ...statsApiModule,
  ...systemApiModule,
  ...batchesApiModule,
  ...benchmarksApiModule,
  ...backendsApiModule,
  ...servicesApiModule,
  ...authApiModule,
  ...federationApiModule,
};
