// Tipos del dominio «registry» (REF-15).
// Extraídos de types/index.ts; el barrel index.ts los reexporta con export *.

import type { BackendExtraConfig, BackendType, CompatibleBackend, LoadPolicy } from "./models"

export interface DownloadJob {
  jobId: string;
  source: "huggingface" | "ollama" | "bitnet";
  modelRef: string;
  artifact?: string | null;
  registerConfig?: {
    displayName?: string;
    loadPolicy?: LoadPolicy;
    /** Explicit backend choice; omit to let the server infer one. */
    backendType?: BackendType;
    pinPermanent?: boolean;
    preferredGpu?: number | null;
    extraConfig?: BackendExtraConfig;
    /** Public name of the model (`gemma4:e4b`, `qwen3-8b-q4_k_m`). */
    modelKey?: string;
    /** llama.cpp only: devices the model may use. After the download the
     *  server plans its presets from llama.cpp's memory projection. */
    placement?: { gpus: number[]; cpu: boolean };
    /** Deployment that serves the public name. */
    defaultBackend?: BackendType;
    /** Every deployment to create from the same file (default included). */
    backends?: BackendType[];
  } | null;
  status: "queued" | "downloading" | "completed" | "failed" | "cancelled";
  progressPct: number;
  speedMbS: number | null;
  etaSeconds: number | null;
  error: string | null;
  startedAt: string;
  completedAt: string | null;
  kind: "install" | "update";
}

export type DownloadSource = "huggingface" | "ollama" | "bitnet";

export interface HFVLLMRuntimeProbe {
  status:
    | "supported_native"
    | "supported_transformers_backend"
    | "supported_pooling"
    | "needs_remote_code"
    | "missing_chat_template"
    | "missing_tool_parser"
    | "missing_reasoning_parser"
    | "needs_hf_overrides"
    | "unsupported_tokenizer"
    | "unsupported_architecture"
    | "unavailable"
    | "unknown";
  reason: string | null;
  recommendedModelImpl: "auto" | "vllm" | "transformers" | null;
  recommendedRunner: "generate" | "pooling" | null;
  tokenizerLoad: boolean | null;
  configLoad: boolean | null;
  observedAt: string | null;
}

export interface HFVLLMSupport {
  classification:
    | "native_vllm"
    | "transformers_backend"
    | "pooling"
    | "unsupported"
    | "unknown";
  label: string;
  modelImpl: "auto" | "vllm" | "transformers" | null;
  runner: "generate" | "pooling" | null;
  taskMode:
    | "generate"
    | "multimodal_generate"
    | "pooling"
    | "multimodal_pooling"
    | null;
  requiredOverrides: string[];
  recipeId: string | null;
  recipeNotes: string[];
  recipeModelImpl: "auto" | "vllm" | "transformers" | null;
  recipeRunner: "generate" | "pooling" | null;
  suggestedConfig: Record<string, unknown>;
  suggestedTuning: Record<string, unknown>;
  runtimeProbe: HFVLLMRuntimeProbe | null;
}

export interface HFModelCard {
  repoId: string;
  modelName: string;
  task: string | null;
  downloads: number;
  likes: number;
  sizeGb: number | null;
  tags: string[];
  gated: boolean;
  suggestedBackend: BackendType;
  // Backends that this repo can run on (suggestedBackend always first).
  // Empty/missing array means just [suggestedBackend] is supported.
  backendOptions?: BackendType[];
  compatibility?: string;
  compatibilityReason?: string | null;
  vllmSupport?: HFVLLMSupport | null;
  compatibleBackends?: CompatibleBackend[];
  createdAt?: string | null;
  lastModified?: string | null;
}

export interface HFModelVariant {
  variantId: string;
  label: string;
  artifact: string | null;
  sizeGb: number | null;
  format: string;
  quantization: string | null;
  backendType: BackendType;
  isDefault: boolean;
  installable?: boolean;
  compatibility?: string;
  compatibilityReason?: string | null;
  vllmSupport?: HFVLLMSupport | null;
  compatibleBackends?: CompatibleBackend[];
}

export interface OllamaModelCard {
  name: string;
  description: string;
  tags: string[];
  sizeGb: number | null;
  pulls: number;
  compatibleBackends?: CompatibleBackend[];
}

export interface OllamaModelVariant {
  name: string;
  tag: string;
  sizeGb: number | null;
  parameterSize: string | null;
  quantization: string | null;
  contextWindow: string | null;
  modality: string | null;
  updatedHint: string | null;
  compatibleBackends?: CompatibleBackend[];
}

export interface LocalModel {
  modelId: string;
  path: string;
  sizeGb: number;
  backendType: BackendType;
  configured: boolean;
  // Where the file was discovered. "ollama-shared" entries point at a GGUF
  // blob inside Ollama's content-addressed store and can be served by our
  // llama.cpp backend without re-downloading.
  source?: "huggingface" | "gguf" | "ollama" | "ollama-shared";
  /** Metadatos del tokenizer que trae el escaneo local (GGUF). Opcionales. */
  vocabSize?: number | null;
  bosId?: number | null;
  eosId?: number | null;
}
