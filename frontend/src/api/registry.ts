// Cliente tipado de /ocabra/* — dominio «registry» (REF-15).
// Extraído de api/client.ts; usa la capa HTTP única de api/http.ts.

import { request, sseUrl } from "@/api/http"
import { isRecord } from "@/api/_helpers"
import { toCompatibleBackends } from "@/api/models"
import type {
  BackendType,
  DownloadJob,
  DownloadSource,
  HFModelCard,
  HFModelVariant,
  LoadPolicy,
  LocalModel,
  OllamaModelCard,
  OllamaModelVariant,
} from "@/types"

export function toHFVLLMSupport(raw: unknown): HFModelCard["vllmSupport"] {
  const data = isRecord(raw) ? raw : null;
  if (!data) return null;
  const probeRaw = data.runtime_probe ?? data.runtimeProbe;
  const probe = isRecord(probeRaw)
    ? {
        status: String(probeRaw.status ?? "unknown") as
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
          | "unknown",
        reason: (probeRaw.reason ?? null) as string | null,
        recommendedModelImpl: (probeRaw.recommended_model_impl ??
          probeRaw.recommendedModelImpl ??
          null) as "auto" | "vllm" | "transformers" | null,
        recommendedRunner: (probeRaw.recommended_runner ??
          probeRaw.recommendedRunner ??
          null) as "generate" | "pooling" | null,
        tokenizerLoad:
          probeRaw.tokenizer_load == null && probeRaw.tokenizerLoad == null
            ? null
            : Boolean(probeRaw.tokenizer_load ?? probeRaw.tokenizerLoad),
        configLoad:
          probeRaw.config_load == null && probeRaw.configLoad == null
            ? null
            : Boolean(probeRaw.config_load ?? probeRaw.configLoad),
        observedAt: (probeRaw.observed_at ?? probeRaw.observedAt ?? null) as
          string | null,
      }
    : null;

  return {
    classification: String(data.classification ?? "unknown") as
      | "native_vllm"
      | "transformers_backend"
      | "pooling"
      | "unsupported"
      | "unknown",
    label: String(data.label ?? "unknown"),
    modelImpl: (data.model_impl ?? data.modelImpl ?? null) as
      "auto" | "vllm" | "transformers" | null,
    runner: (data.runner ?? null) as "generate" | "pooling" | null,
    taskMode: (data.task_mode ?? data.taskMode ?? null) as
      | "generate"
      | "multimodal_generate"
      | "pooling"
      | "multimodal_pooling"
      | null,
    requiredOverrides: Array.isArray(
      data.required_overrides ?? data.requiredOverrides,
    )
      ? ((data.required_overrides ?? data.requiredOverrides) as unknown[]).map(
          String,
        )
      : [],
    recipeId: (data.recipe_id ?? data.recipeId ?? null) as string | null,
    recipeNotes: Array.isArray(data.recipe_notes ?? data.recipeNotes)
      ? ((data.recipe_notes ?? data.recipeNotes) as unknown[]).map(String)
      : [],
    recipeModelImpl: (data.recipe_model_impl ??
      data.recipeModelImpl ??
      null) as "auto" | "vllm" | "transformers" | null,
    recipeRunner: (data.recipe_runner ?? data.recipeRunner ?? null) as
      "generate" | "pooling" | null,
    suggestedConfig: isRecord(data.suggested_config ?? data.suggestedConfig)
      ? ((data.suggested_config ?? data.suggestedConfig) as Record<
          string,
          unknown
        >)
      : {},
    suggestedTuning: isRecord(data.suggested_tuning ?? data.suggestedTuning)
      ? ((data.suggested_tuning ?? data.suggestedTuning) as Record<
          string,
          unknown
        >)
      : {},
    runtimeProbe: probe,
  };
}


export function toDownloadJob(raw: unknown): DownloadJob {
  const data = isRecord(raw) ? raw : {};
  const registerConfigRaw = isRecord(
    data.register_config ?? data.registerConfig,
  )
    ? ((data.register_config ?? data.registerConfig) as Record<string, unknown>)
    : null;
  return {
    jobId: String(data.job_id ?? data.jobId ?? ""),
    source: String(data.source ?? "huggingface") as DownloadSource,
    modelRef: String(data.model_ref ?? data.modelRef ?? ""),
    artifact: (data.artifact ?? null) as string | null,
    registerConfig: registerConfigRaw
      ? {
          displayName: registerConfigRaw.display_name as string | undefined,
          loadPolicy: (registerConfigRaw.load_policy ?? "on_demand") as
            "pin" | "warm" | "on_demand",
          pinPermanent: Boolean(registerConfigRaw.pin_permanent),
          preferredGpu: (registerConfigRaw.preferred_gpu ?? null) as
            number | null,
          extraConfig: isRecord(registerConfigRaw.extra_config)
            ? (registerConfigRaw.extra_config as {
                vllm?: import("@/types").VLLMConfig;
              })
            : undefined,
        }
      : null,
    status: String(data.status ?? "queued") as DownloadJob["status"],
    progressPct: Number(data.progress_pct ?? data.progressPct ?? 0),
    speedMbS:
      data.speed_mb_s === null || data.speedMbS === null
        ? null
        : Number(data.speed_mb_s ?? data.speedMbS ?? 0),
    etaSeconds:
      data.eta_seconds === null || data.etaSeconds === null
        ? null
        : Number(data.eta_seconds ?? data.etaSeconds ?? 0),
    error: (data.error ?? null) as string | null,
    startedAt: String(
      data.started_at ?? data.startedAt ?? new Date(0).toISOString(),
    ),
    completedAt: (data.completed_at ?? data.completedAt ?? null) as
      string | null,
    kind: (data.kind === "update"
      ? "update"
      : "install") as DownloadJob["kind"],
  };
}


export function toOllamaModelCard(raw: unknown): OllamaModelCard {
  const data = isRecord(raw) ? raw : {};
  const size = data.size_gb ?? data.sizeGb;
  return {
    name: String(data.name ?? ""),
    description: String(data.description ?? ""),
    tags: Array.isArray(data.tags) ? data.tags.map(String) : [],
    sizeGb: size == null ? null : Number(size),
    pulls: Number(data.pulls ?? 0),
    compatibleBackends: toCompatibleBackends(
      data.compatible_backends ?? data.compatibleBackends,
    ),
  };
}


export function toHFModelCard(raw: unknown): HFModelCard {
  const data = isRecord(raw) ? raw : {};
  return {
    repoId: String(data.repo_id ?? data.repoId ?? ""),
    modelName: String(data.model_name ?? data.modelName ?? ""),
    task: (data.task ?? null) as string | null,
    downloads: Number(data.downloads ?? 0),
    likes: Number(data.likes ?? 0),
    sizeGb:
      data.size_gb != null
        ? Number(data.size_gb)
        : data.sizeGb != null
          ? Number(data.sizeGb)
          : null,
    tags: Array.isArray(data.tags) ? data.tags.map(String) : [],
    gated: Boolean(data.gated),
    suggestedBackend: String(
      data.suggested_backend ?? data.suggestedBackend ?? "vllm",
    ) as HFModelCard["suggestedBackend"],
    backendOptions: Array.isArray(data.backend_options ?? data.backendOptions)
      ? ((data.backend_options ?? data.backendOptions) as unknown[]).map(
          (item) => String(item) as BackendType,
        )
      : undefined,
    compatibility: String(data.compatibility ?? "unknown"),
    compatibilityReason: (data.compatibility_reason ??
      data.compatibilityReason ??
      null) as string | null,
    vllmSupport: toHFVLLMSupport(data.vllm_support ?? data.vllmSupport),
    compatibleBackends: toCompatibleBackends(
      data.compatible_backends ?? data.compatibleBackends,
    ),
  };
}


export function toHFModelVariant(raw: unknown): HFModelVariant {
  const data = isRecord(raw) ? raw : {};
  return {
    variantId: String(data.variant_id ?? data.variantId ?? ""),
    label: String(data.label ?? ""),
    artifact: (data.artifact ?? null) as string | null,
    sizeGb:
      data.size_gb != null
        ? Number(data.size_gb)
        : data.sizeGb != null
          ? Number(data.sizeGb)
          : null,
    format: String(data.format ?? "unknown"),
    quantization: (data.quantization ?? null) as string | null,
    backendType: String(
      data.backend_type ?? data.backendType ?? "vllm",
    ) as HFModelVariant["backendType"],
    isDefault: Boolean(data.is_default ?? data.isDefault),
    installable: data.installable == null ? true : Boolean(data.installable),
    compatibility: String(data.compatibility ?? "unknown"),
    compatibilityReason: (data.compatibility_reason ??
      data.compatibilityReason ??
      null) as string | null,
    vllmSupport: toHFVLLMSupport(data.vllm_support ?? data.vllmSupport),
    compatibleBackends: toCompatibleBackends(
      data.compatible_backends ?? data.compatibleBackends,
    ),
  };
}


export function toLocalModel(raw: unknown): LocalModel {
  const data = isRecord(raw) ? raw : {};
  const modelRef = String(
    data.model_ref ?? data.modelRef ?? data.model_id ?? data.modelId ?? "",
  );
  const source = (data.source ?? null) as LocalModel["source"] | null;
  return {
    modelId: modelRef,
    path: String(data.path ?? ""),
    sizeGb: Number(data.size_gb ?? data.sizeGb ?? 0),
    backendType: String(
      data.backend_type ?? data.backendType ?? "vllm",
    ) as LocalModel["backendType"],
    configured: Boolean(data.configured ?? false),
    source: source ?? undefined,
    vocabSize: (data.vocab_size ?? data.vocabSize ?? null) as number | null,
    bosId: (data.bos_id ?? data.bosId ?? null) as number | null,
    eosId: (data.eos_id ?? data.eosId ?? null) as number | null,
  };
}


export function toOllamaVariant(raw: unknown): OllamaModelVariant {
  const data = isRecord(raw) ? raw : {};
  return {
    name: String(data.name ?? ""),
    tag: String(data.tag ?? ""),
    sizeGb:
      data.size_gb != null
        ? Number(data.size_gb)
        : data.sizeGb != null
          ? Number(data.sizeGb)
          : null,
    parameterSize: (data.parameter_size ?? data.parameterSize ?? null) as
      string | null,
    quantization: (data.quantization ?? null) as string | null,
    contextWindow: (data.context_window ?? data.contextWindow ?? null) as
      string | null,
    modality: (data.modality ?? null) as string | null,
    updatedHint: (data.updated_hint ?? data.updatedHint ?? null) as
      string | null,
    compatibleBackends: toCompatibleBackends(
      data.compatible_backends ?? data.compatibleBackends,
    ),
  };
}


export const registryApiModule = {
  downloads: {
    list: async () =>
      (await request<unknown[]>("GET", "/ocabra/downloads")).map(toDownloadJob),
    enqueue: (
      source: DownloadSource,
      modelRef: string,
      artifact?: string | null,
      registerConfig?: DownloadJob["registerConfig"],
    ) =>
      request<unknown>("POST", "/ocabra/downloads", {
        source,
        model_ref: modelRef,
        artifact: artifact ?? null,
        // The wire format is snake_case and `request` does a plain
        // JSON.stringify, so the mapping has to happen here. It did not, and
        // the server — which reads snake_case — silently discarded every
        // field: the load policy and display name picked in the install modal,
        // and the vLLM recipe preconfiguration in extraConfig.
        register_config: registerConfig
          ? {
              display_name: registerConfig.displayName,
              load_policy: registerConfig.loadPolicy,
              backend_type: registerConfig.backendType,
              pin_permanent: registerConfig.pinPermanent,
              preferred_gpu: registerConfig.preferredGpu,
              extra_config: registerConfig.extraConfig,
              model_key: registerConfig.modelKey,
              default_backend: registerConfig.defaultBackend,
              backends: registerConfig.backends,
              placement: registerConfig.placement,
            }
          : null,
      }).then(toDownloadJob),
    cancel: (jobId: string) =>
      request<void>("DELETE", `/ocabra/downloads/${encodeURIComponent(jobId)}`),
    clearHistory: (status = "failed,cancelled,completed") =>
      request<{ deleted: number }>(
        "DELETE",
        `/ocabra/downloads?status=${encodeURIComponent(status)}`,
      ),
    streamProgress: (jobId: string): EventSource =>
      new EventSource(sseUrl(`/ocabra/downloads/${encodeURIComponent(jobId)}/stream`)),
  },

  registry: {
    /**
     * Register weights already on disk under an explicitly chosen backend.
     * Covers what the download flow cannot: attaching an Ollama-downloaded
     * GGUF to a llama.cpp backend without re-downloading several GB, or
     * serving the same file with a different backend than the scanner
     * inferred.
     */
    registerLocal: (body: {
      modelRef: string;
      path: string;
      backendType: BackendType;
      displayName?: string;
      loadPolicy?: LoadPolicy;
      preferredGpu?: number | null;
    }): Promise<unknown> =>
      request<unknown>("POST", "/ocabra/registry/local/register", {
        model_ref: body.modelRef,
        path: body.path,
        backend_type: body.backendType,
        display_name: body.displayName ?? null,
        load_policy: body.loadPolicy ?? "on_demand",
        preferred_gpu: body.preferredGpu ?? null,
      }),
    searchHF: async (
      q: string,
      task?: string,
      limit = 20,
    ): Promise<HFModelCard[]> => {
      const params = new URLSearchParams({ q, limit: String(limit) });
      if (task) params.set("task", task);
      const raw = await request<unknown[]>(
        "GET",
        `/ocabra/registry/hf/search?${params}`,
      );
      return raw.map(toHFModelCard);
    },
    getHFVariants: async (repoId: string): Promise<HFModelVariant[]> =>
      (
        await request<unknown[]>(
          "GET",
          `/ocabra/registry/hf/${encodeURIComponent(repoId)}/variants`,
        )
      ).map(toHFModelVariant),
    searchBitnet: async (q: string, limit = 20): Promise<HFModelCard[]> => {
      const params = new URLSearchParams({ q, limit: String(limit) });
      const raw = await request<unknown[]>(
        "GET",
        `/ocabra/registry/bitnet/search?${params}`,
      );
      return raw.map(toHFModelCard);
    },
    getBitnetVariants: async (repoId: string): Promise<HFModelVariant[]> =>
      (
        await request<unknown[]>(
          "GET",
          `/ocabra/registry/bitnet/${encodeURIComponent(repoId)}/variants`,
        )
      ).map(toHFModelVariant),
    searchOllama: async (q: string): Promise<OllamaModelCard[]> =>
      (
        await request<unknown[]>(
          "GET",
          `/ocabra/registry/ollama/search?q=${encodeURIComponent(q)}`,
        )
      ).map(toOllamaModelCard),
    getOllamaVariants: async (
      modelName: string,
    ): Promise<OllamaModelVariant[]> =>
      (
        await request<unknown[]>(
          "GET",
          `/ocabra/registry/ollama/${encodeURIComponent(modelName)}/variants`,
        )
      ).map(toOllamaVariant),
    listLocal: async (opts: { refresh?: boolean } = {}) =>
      (
        await request<unknown[]>(
          "GET",
          opts.refresh
            ? "/ocabra/registry/local?refresh=true"
            : "/ocabra/registry/local",
        )
      ).map(toLocalModel),
  },

};
