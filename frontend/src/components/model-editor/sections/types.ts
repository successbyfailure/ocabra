import type { GPUState, LogicalModel } from "@/types"
import type { ModelEditorState } from "../useModelEditor"
import type { LadderRung } from "../usePresetLadder"
import type { ModelInsights } from "../types"

/** Lo que recibe cada sección del editor. */
export interface SectionProps {
  editor: ModelEditorState
  /** Sin rol model_manager la página es de solo lectura. */
  readOnly: boolean
  isAdmin: boolean
  /** Modelo lógico (nombre público y sus despliegues). */
  logical: LogicalModel | null
  refreshLogical: () => Promise<void>
  gpus: GPUState[]
  /** VRAM que este despliegue ocupa ahora en cada GPU (MB), si está cargado. */
  liveVramByGpu: Record<string, number>
  /** Presets activos del perfil por defecto: si hay, mandan sobre el contexto,
   *  los slots, la KV y las GPUs de la configuración base. */
  presets?: LadderRung[]
  /** Lo medido del modelo (ritmos, carga en frío) y lo que lee su plantilla. */
  insights?: ModelInsights | null
  reloadInsights?: () => Promise<void>
}
