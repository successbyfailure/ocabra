import { isLlamaCppFamily } from "@/types";
import { toCamelView } from "@/lib/configKeys"
import type {
  BackendExtraConfig,
  BitNetConfig,
  LlamaCppConfig,
  ModelState,
  ProfileCategory,
  SGLangConfig,
  TensorRTLLMConfig,
  VLLMConfig,
  WhisperConfig,
} from "@/types"

export interface ModelContextSummary {
  nativeContext: number | null
  configuredContext: number | null
  maxInputTokens: number | null
  maxOutputTokens: number | null
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null
}

function getConfigSection<T>(model: ModelState, key: keyof BackendExtraConfig): T | null {
  const extraConfig = model.extraConfig
  if (!isRecord(extraConfig)) return null
  const nested = extraConfig[key]
  // camelCase view with snake_case winning: the server stores snake_case
  // since migration 0034, and older rows (or the client mapping) may carry
  // both spellings — the snake one is what the backend actually applies.
  if (isRecord(nested)) return toCamelView(nested) as T
  return toCamelView(extraConfig) as T
}

function toPositiveNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? value : null
}

export function getVllmConfig(model: ModelState): VLLMConfig | null {
  if (model.backendType !== "vllm") return null
  return getConfigSection<VLLMConfig>(model, "vllm")
}

export function getSGLangConfig(model: ModelState): SGLangConfig | null {
  if (model.backendType !== "sglang") return null
  return getConfigSection<SGLangConfig>(model, "sglang")
}

export function getLlamaCppConfig(model: ModelState): LlamaCppConfig | null {
  // Upstream llama.cpp and the Makespace fork share the same config
  // section, so both read the same options here. getConfigSection already
  // folds snake_case (persisted) and camelCase (legacy UI writes) into one
  // camelCase view, snake first.
  if (!isLlamaCppFamily(model.backendType)) return null
  const config = getConfigSection<LlamaCppConfig>(model, "llama_cpp")
  if (config == null) return null
  if (Array.isArray(config.tensorSplit)) {
    config.tensorSplit = (config.tensorSplit as unknown[]).filter(
      (v): v is number => typeof v === "number" && Number.isFinite(v),
    )
  }
  if (Array.isArray(config.disabledGpus)) {
    config.disabledGpus = (config.disabledGpus as unknown[]).filter(
      (v): v is number => typeof v === "number" && Number.isInteger(v),
    )
  }
  return config
}

export function getBitNetConfig(model: ModelState): BitNetConfig | null {
  if (model.backendType !== "bitnet") return null
  return getConfigSection<BitNetConfig>(model, "bitnet")
}

export function getTensorRTLLMConfig(model: ModelState): TensorRTLLMConfig | null {
  if (model.backendType !== "tensorrt_llm") return null
  return getConfigSection<TensorRTLLMConfig>(model, "tensorrt_llm")
}

export function getWhisperConfig(model: ModelState): WhisperConfig | null {
  if (model.backendType !== "whisper") return null
  return getConfigSection<WhisperConfig>(model, "whisper")
}

export function getModelContextSummary(model: ModelState): ModelContextSummary {
  // Prefer the actual native ceiling (e.g. 262144 for gemma4:26b) when the
  // backend reports it; fall back to the served context otherwise. Before,
  // "nativo" showed OLLAMA_CONTEXT_LENGTH (65536) which is the effective cap,
  // not what the model was trained for.
  const nativeFromBackend =
    typeof model.capabilities.nativeContextLength === "number" && model.capabilities.nativeContextLength > 0
      ? model.capabilities.nativeContextLength
      : null
  const servedContext =
    typeof model.capabilities.contextLength === "number" && model.capabilities.contextLength > 0
      ? model.capabilities.contextLength
      : null
  const nativeContext = nativeFromBackend ?? servedContext

  let configuredContext: number | null = null
  switch (model.backendType) {
    case "vllm": {
      const vllm = getVllmConfig(model)
      configuredContext = toPositiveNumber(vllm?.maxModelLen)
      break
    }
    case "sglang": {
      const sglang = getSGLangConfig(model)
      configuredContext = toPositiveNumber(sglang?.contextLength ?? (sglang as Record<string, unknown> | null)?.context_length)
      break
    }
    case "llama_cpp":
    case "llamacpp_mks": {
      const llamaCpp = getLlamaCppConfig(model)
      configuredContext = toPositiveNumber(llamaCpp?.ctxSize ?? (llamaCpp as Record<string, unknown> | null)?.ctx_size)
      break
    }
    case "bitnet": {
      const bitnet = getBitNetConfig(model)
      configuredContext = toPositiveNumber(bitnet?.ctxSize ?? (bitnet as Record<string, unknown> | null)?.ctx_size)
      break
    }
    case "tensorrt_llm": {
      const tensorrt = getTensorRTLLMConfig(model)
      configuredContext = toPositiveNumber(
        tensorrt?.contextLength ?? (tensorrt as Record<string, unknown> | null)?.context_length,
      )
      break
    }
  }

  configuredContext ??= servedContext ?? nativeContext

  return {
    nativeContext,
    configuredContext,
    maxInputTokens: configuredContext,
    maxOutputTokens: configuredContext,
  }
}

export type ModelCategory = "chat" | "embeddings" | "image" | "tts" | "stt" | "music" | "video" | "other"

export const MODEL_CATEGORY_ORDER: ModelCategory[] = [
  "chat",
  "embeddings",
  "image",
  "tts",
  "stt",
  "music",
  "video",
  "other",
]

export const MODEL_CATEGORY_LABELS: Record<ModelCategory, string> = {
  chat: "Chat",
  embeddings: "Embeddings",
  image: "Imagen",
  tts: "TTS",
  stt: "STT",
  music: "Música",
  video: "Video",
  other: "Otros",
}

/** La categoría del backend (``ProfileCategory``) llevada al espacio, más
 *  grueso, de la lista de modelos. Única tabla de correspondencia; la usan
 *  también las páginas de Modelos y Perfiles. */
export const PROFILE_CATEGORY_TO_MODEL_CATEGORY: Record<ProfileCategory, ModelCategory> = {
  llm: "chat",
  embedding: "embeddings",
  rerank: "embeddings",
  tts: "tts",
  stt: "stt",
  image: "image",
  music: "music",
  video_upscaling: "video",
}

export function profileToModelCategory(category: string | null | undefined): ModelCategory | null {
  if (category && category in PROFILE_CATEGORY_TO_MODEL_CATEGORY) {
    return PROFILE_CATEGORY_TO_MODEL_CATEGORY[category as ProfileCategory]
  }
  return null
}

/**
 * ÚNICO clasificador por capacidades (el respaldo cuando el backend no da
 * `category`). Devuelve una ``ProfileCategory`` o ``null`` si no reconoce
 * ninguna capacidad. Antes había dos clasificadores con prioridades distintas
 * (este y el del editor de perfiles); ahora ambos salen de aquí.
 */
export function inferProfileCategory(
  caps: Record<string, unknown> | undefined,
): ProfileCategory | null {
  const c = caps ?? {}
  const has = (...keys: string[]) => keys.some((k) => Boolean(c[k]))
  if (has("audio_transcription", "audioTranscription")) return "stt"
  if (has("tts")) return "tts"
  if (has("music_generation", "musicGeneration")) return "music"
  if (has("image_generation", "imageGeneration", "image_editing", "imageEditing")) return "image"
  if (has("video_upscaling", "videoUpscaling")) return "video_upscaling"
  if (has("rerank")) return "rerank"
  if (has("embeddings", "pooling", "classification", "score")) return "embedding"
  if (has("chat", "completion")) return "llm"
  return null
}

/** Categoría de perfil de un modelo según el backend: el perfil de ejecución
 *  por defecto (o el primero). ``null`` si el modelo no trae perfiles. */
function backendProfileCategory(model: ModelState): ProfileCategory | null {
  const execution = (model.profiles ?? []).filter((p) => p.profileKind === "execution")
  const chosen = execution.find((p) => p.isDefault) ?? execution[0]
  return chosen?.category ?? null
}

/**
 * Categoría de un modelo para la lista. Fuente única: la ``category`` del
 * backend (perfil de ejecución). Si el modelo no trae perfiles, se cae al
 * único respaldo por capacidades. Los backends de vídeo que aún no exponen la
 * capacidad `video_upscaling` se reconocen por su tipo.
 */
export function categorizeModel(model: ModelState): ModelCategory {
  const backend = profileToModelCategory(backendProfileCategory(model))
  if (backend) return backend
  if (
    model.backendType === "upscaler" ||
    model.backendType === "flashvsr" ||
    model.backendType === "comfyui_video"
  )
    return "video"
  return (
    profileToModelCategory(
      inferProfileCategory(model.capabilities as unknown as Record<string, unknown>),
    ) ?? "other"
  )
}
