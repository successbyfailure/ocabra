import { useEffect, useMemo, useState } from "react"
import * as Dialog from "@radix-ui/react-dialog"
import * as Tabs from "@radix-ui/react-tabs"
import { ChevronDown, ChevronRight, Search, X } from "lucide-react"
import { toast } from "sonner"
import { api } from "@/api/client"
import { HFModelCard } from "@/components/explore/HFModelCard"
import { OllamaModelCard } from "@/components/explore/OllamaModelCard"
import { SearchFilters } from "@/components/explore/SearchFilters"
import { BackendIcons, backendLabel, compatibleToItems } from "@/components/models/BackendIcons"
import { getProbeOverrideHint, getProbeStatusLabel } from "@/lib/vllmProbe"
import { useDownloadStore } from "@/stores/downloadStore"
import type {
  BackendType,
  CompatibleBackend,
  DownloadSource,
  HFModelCard as HFCardType,
  HFModelVariant,
  LoadPolicy,
  OllamaModelCard as OllamaCardType,
  OllamaModelVariant,
  VLLMConfig,
} from "@/types"

interface InstallTarget {
  source: DownloadSource
  modelRef: string
  title: string
  ollamaBase?: string
  // Backends the repo can be served with, when more than one applies (e.g.
  // safetensors + GGUF in one upload, or upstream llama.cpp vs the Makespace
  // fork). The registration used to always re-infer a single backend and
  // ignore the choice; it now honours `backend_type` in register_config.
  backendOptions?: BackendType[]
  // Card-level compatibility (fallback when the variant carries none).
  compatibleBackends?: CompatibleBackend[]
}

const LLAMACPP_PREFERRED: BackendType = "llamacpp_mks"

function isGgufVariant(variant: HFModelVariant | null | undefined): boolean {
  if (!variant) return false
  return variant.format?.toLowerCase() === "gguf" || Boolean(variant.artifact?.toLowerCase().endsWith(".gguf"))
}

/**
 * Public name proposed for a download. Ollama: the tag itself
 * (`gemma4:e4b`). HuggingFace/BitNet: repo name without the org, lowercased,
 * plus `-<quant>` for a GGUF variant (`qwen3-8b-q4_k_m`); a trailing `-gguf`
 * in the repo name is dropped in that case so it does not read twice.
 */
export function suggestModelKey(
  source: DownloadSource,
  ref: string,
  variant?: HFModelVariant | null,
): string {
  if (source === "ollama") return ref.trim()
  const repoName = (ref.split("/").pop() ?? ref).trim().toLowerCase()
  const quant = variant?.quantization?.trim().toLowerCase()
  if (isGgufVariant(variant) && quant) {
    const base = repoName.replace(/[-_.]gguf$/, "")
    return base.endsWith(`-${quant}`) ? base : `${base}-${quant}`
  }
  return repoName
}

/**
 * Default backend preselected in the install dialog: the production
 * llama.cpp for GGUF/Ollama when installed, else the variant's own backend,
 * else the first installed compatible one.
 */
export function pickDefaultBackend(
  compatible: CompatibleBackend[],
  opts: { gguf: boolean; variantBackend?: BackendType | null },
): BackendType | "" {
  const installed = compatible.filter((entry) => entry.installed)
  if (opts.gguf && installed.some((entry) => entry.backend === LLAMACPP_PREFERRED)) {
    return LLAMACPP_PREFERRED
  }
  if (opts.variantBackend && installed.some((entry) => entry.backend === opts.variantBackend)) {
    return opts.variantBackend
  }
  return installed[0]?.backend ?? ""
}

function AdvancedDetails({ children }: { children: React.ReactNode }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="space-y-2">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
      >
        {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        Detalles avanzados
      </button>
      {open && <div className="space-y-2">{children}</div>}
    </div>
  )
}

export function getProbePreconfigurationNotice(variant: HFModelVariant | null): string | null {
  const support = variant?.vllmSupport
  const probe = support?.runtimeProbe
  if (!support || !probe) return null
  if (!probe.recommendedModelImpl && !probe.recommendedRunner) return null

  return [
    "La preconfiguracion automatica usara la recomendacion verificada por probe",
    probe.recommendedModelImpl ? `model_impl=${probe.recommendedModelImpl}` : null,
    probe.recommendedRunner ? `runner=${probe.recommendedRunner}` : null,
  ]
    .filter(Boolean)
    .join(" · ")
}

export function getRecipeProbeDifferenceNotice(variant: HFModelVariant | null): string | null {
  const support = variant?.vllmSupport
  const probe = support?.runtimeProbe
  if (!support || !probe) return null

  const recipeModelImpl = support.recipeModelImpl
  const recipeRunner = support.recipeRunner
  const finalModelImpl = probe.recommendedModelImpl ?? support.modelImpl
  const finalRunner = probe.recommendedRunner ?? support.runner

  if (
    (!recipeModelImpl || recipeModelImpl === finalModelImpl) &&
    (!recipeRunner || recipeRunner === finalRunner)
  ) {
    return null
  }

  return [
    "La recomendacion final del probe difiere de la recipe base",
    recipeModelImpl ? `recipe model_impl=${recipeModelImpl}` : null,
    recipeRunner ? `recipe runner=${recipeRunner}` : null,
    finalModelImpl ? `final model_impl=${finalModelImpl}` : null,
    finalRunner ? `final runner=${finalRunner}` : null,
  ]
    .filter(Boolean)
    .join(" · ")
}

function toRecipeRegisterConfig(
  title: string,
  loadPolicy: LoadPolicy,
  variant: HFModelVariant | null,
  backendType?: BackendType | null,
) {
  const support = variant?.vllmSupport
  // camelCase here, like the rest of the frontend; api/client.ts maps the
  // payload to the snake_case keys the backend reads. It used to ship this
  // object verbatim, so every field was silently dropped — including the
  // whole vLLM recipe preconfiguration.
  const base = {
    displayName: title,
    loadPolicy,
    ...(backendType ? { backendType } : {}),
  }
  if (!support) {
    return base
  }

  const vllmConfig: VLLMConfig = {
    recipeId: support.recipeId,
    recipeNotes: support.recipeNotes,
    recipeModelImpl: support.recipeModelImpl,
    recipeRunner: support.recipeRunner,
    suggestedConfig: support.suggestedConfig,
    suggestedTuning: support.suggestedTuning,
    probeStatus: support.runtimeProbe?.status ?? null,
    probeReason: support.runtimeProbe?.reason ?? null,
    probeObservedAt: support.runtimeProbe?.observedAt ?? null,
    probeRecommendedModelImpl: support.runtimeProbe?.recommendedModelImpl ?? null,
    probeRecommendedRunner: support.runtimeProbe?.recommendedRunner ?? null,
    modelImpl: support.modelImpl,
    runner: support.runner,
  }

  for (const [key, value] of Object.entries(support.suggestedConfig)) {
    if (key === "tool_call_parser") vllmConfig.toolCallParser = String(value)
    if (key === "reasoning_parser") vllmConfig.reasoningParser = String(value)
    if (key === "chat_template") vllmConfig.chatTemplate = String(value)
    if (key === "hf_overrides") {
      vllmConfig.hfOverrides =
        typeof value === "string" || (typeof value === "object" && value !== null)
          ? (value as string | Record<string, unknown>)
          : null
    }
  }

  return {
    ...base,
    extraConfig: {
      vllm: vllmConfig,
    },
  }
}

export function Explore() {
  const [activeTab, setActiveTab] = useState<"hf" | "bitnet" | "ollama">("hf")
  const [query, setQuery] = useState("mistral")
  const [debouncedQuery, setDebouncedQuery] = useState(query)
  const [taskFilter, setTaskFilter] = useState("")
  const [sizeFilter, setSizeFilter] = useState("")
  const [gatedFilter, setGatedFilter] = useState("")
  const [hfResults, setHfResults] = useState<HFCardType[]>([])
  const [ollamaResults, setOllamaResults] = useState<OllamaCardType[]>([])
  const [bitnetResults, setBitnetResults] = useState<HFCardType[]>([])
  const [loading, setLoading] = useState(false)
  const [installTarget, setInstallTarget] = useState<InstallTarget | null>(null)
  const [selectedBackendType, setSelectedBackendType] = useState<BackendType | "">("")
  const [ollamaVariants, setOllamaVariants] = useState<OllamaModelVariant[]>([])
  const [hfVariants, setHfVariants] = useState<HFModelVariant[]>([])
  const [selectedHFVariantId, setSelectedHFVariantId] = useState("")
  const [bitnetVariants, setBitnetVariants] = useState<HFModelVariant[]>([])
  const [selectedBitnetVariantId, setSelectedBitnetVariantId] = useState("")
  const [selectedVariant, setSelectedVariant] = useState("")
  const [variantLoading, setVariantLoading] = useState(false)
  const [targetDir, setTargetDir] = useState("/models")
  const [loadPolicy, setLoadPolicy] = useState<LoadPolicy>("on_demand")
  const [modelKey, setModelKey] = useState("")
  const [deployBackends, setDeployBackends] = useState<BackendType[]>([])
  const [defaultBackend, setDefaultBackend] = useState<BackendType | "">("")
  const selectedHFVariant = installTarget?.source === "huggingface"
    ? hfVariants.find((v) => v.variantId === selectedHFVariantId) ?? null
    : null
  const selectedBitnetVariant = installTarget?.source === "bitnet"
    ? bitnetVariants.find((v) => v.variantId === selectedBitnetVariantId) ?? null
    : null
  const probePreconfigurationNotice = getProbePreconfigurationNotice(selectedHFVariant)
  const recipeProbeDifferenceNotice = getRecipeProbeDifferenceNotice(selectedHFVariant)
  const probeStatusLabel = getProbeStatusLabel(selectedHFVariant?.vllmSupport?.runtimeProbe?.status)
  const probeOverrideHint = getProbeOverrideHint(selectedHFVariant?.vllmSupport?.runtimeProbe?.status)
  const canInstallBitnet = installTarget?.source !== "bitnet" || Boolean(selectedBitnetVariant?.artifact)
  const selectedOllamaVariant = installTarget?.source === "ollama"
    ? ollamaVariants.find((v) => v.name === selectedVariant) ?? null
    : null
  const selectedFileVariant = selectedHFVariant ?? selectedBitnetVariant
  // Backends that can serve the file about to be downloaded: the selected
  // variant's list when the server sent one, else the card's.
  const currentCompat: CompatibleBackend[] = useMemo(() => {
    const fromVariant =
      installTarget?.source === "ollama"
        ? selectedOllamaVariant?.compatibleBackends
        : selectedFileVariant?.compatibleBackends
    if (fromVariant && fromVariant.length > 0) return fromVariant
    return installTarget?.compatibleBackends ?? []
  }, [installTarget, selectedOllamaVariant, selectedFileVariant])
  const currentIsGguf = installTarget?.source === "ollama" || isGgufVariant(selectedFileVariant)
  const variantBackend = selectedFileVariant?.backendType ?? null

  // Re-propose name and deployments whenever the file to download changes.
  useEffect(() => {
    if (!installTarget) {
      setModelKey("")
      setDeployBackends([])
      setDefaultBackend("")
      return
    }
    const ref = installTarget.source === "ollama" ? selectedVariant || installTarget.modelRef : installTarget.modelRef
    setModelKey(suggestModelKey(installTarget.source, ref, selectedFileVariant))
    const preferred = pickDefaultBackend(currentCompat, { gguf: currentIsGguf, variantBackend })
    setDefaultBackend(preferred)
    const initial: BackendType[] = preferred ? [preferred] : []
    // An Ollama pull keeps its Ollama deployment as fallback (server default).
    if (
      installTarget.source === "ollama" &&
      preferred !== "ollama" &&
      currentCompat.some((entry) => entry.backend === "ollama" && entry.installed)
    ) {
      initial.push("ollama")
    }
    setDeployBackends(initial)
  }, [installTarget, selectedVariant, selectedFileVariant, currentCompat, currentIsGguf, variantBackend])

  const toggleDeployBackend = (backend: BackendType, checked: boolean) => {
    const next = checked
      ? [...new Set([...deployBackends, backend])]
      : deployBackends.filter((b) => b !== backend)
    setDeployBackends(next)
    // The default must be one of the deployments being created.
    if (!defaultBackend || !next.includes(defaultBackend)) setDefaultBackend(next[0] ?? "")
  }

  const jobs = useDownloadStore((state) => state.jobs)
  const setJobs = useDownloadStore((state) => state.setJobs)
  const addJob = useDownloadStore((state) => state.addJob)

  useEffect(() => {
    const timer = window.setTimeout(() => {
      setDebouncedQuery(query)
    }, 300)
    return () => window.clearTimeout(timer)
  }, [query])

  const refreshJobs = async () => {
    const list = await api.downloads.list()
    setJobs(list)
  }

  useEffect(() => {
    void refreshJobs().catch(() => {
      // silent bootstrap failure
    })
    const timer = window.setInterval(() => {
      void refreshJobs().catch(() => {
        // silent background refresh failure
      })
    }, 30_000)
    return () => window.clearInterval(timer)
  }, [setJobs])

  useEffect(() => {
    const q = debouncedQuery.trim()
    if (!q) {
      setHfResults([])
      setBitnetResults([])
      setOllamaResults([])
      return
    }

    let active = true
    setLoading(true)

    const run = async () => {
      try {
        if (activeTab === "hf") {
          const data = await api.registry.searchHF(q, taskFilter || undefined, 30)
          if (!active) return
          let next = data
          if (sizeFilter === "small") next = next.filter((item) => (item.sizeGb ?? 0) < 4)
          if (sizeFilter === "medium") next = next.filter((item) => (item.sizeGb ?? 0) >= 4 && (item.sizeGb ?? 0) <= 12)
          if (sizeFilter === "large") next = next.filter((item) => (item.sizeGb ?? 0) > 12)
          if (gatedFilter !== "") next = next.filter((item) => String(item.gated) === gatedFilter)
          setHfResults(next)
        } else if (activeTab === "bitnet") {
          const data = await api.registry.searchBitnet(q, 30)
          if (!active) return
          let next = data
          if (sizeFilter === "small") next = next.filter((item) => (item.sizeGb ?? 0) < 4)
          if (sizeFilter === "medium") next = next.filter((item) => (item.sizeGb ?? 0) >= 4 && (item.sizeGb ?? 0) <= 12)
          if (sizeFilter === "large") next = next.filter((item) => (item.sizeGb ?? 0) > 12)
          if (gatedFilter !== "") next = next.filter((item) => String(item.gated) === gatedFilter)
          setBitnetResults(next)
        } else {
          const data = await api.registry.searchOllama(q)
          if (!active) return
          setOllamaResults(data)
        }
      } catch (err) {
        if (active) {
          toast.error(err instanceof Error ? err.message : "No se pudo buscar")
        }
      } finally {
        if (active) setLoading(false)
      }
    }

    void run()

    return () => {
      active = false
    }
  }, [activeTab, debouncedQuery, gatedFilter, sizeFilter, taskFilter])

  useEffect(() => {
    const target = installTarget
    if (!target) {
      setOllamaVariants([])
      setSelectedVariant("")
      setHfVariants([])
      setSelectedHFVariantId("")
      setBitnetVariants([])
      setSelectedBitnetVariantId("")
      setVariantLoading(false)
      return
    }

    if (target.source === "huggingface") {
      let active = true
      setVariantLoading(true)
      setHfVariants([])
      setSelectedHFVariantId("")

      void api.registry
        .getHFVariants(target.modelRef)
        .then((variants) => {
          if (!active) return
          setHfVariants(variants)
          const preferred = variants.find((v) => v.isDefault) ?? variants[0]
          if (preferred) setSelectedHFVariantId(preferred.variantId)
        })
        .catch((err) => {
          if (!active) return
          toast.error(err instanceof Error ? err.message : "No se pudieron cargar variantes HF")
        })
        .finally(() => {
          if (active) setVariantLoading(false)
        })

      return () => {
        active = false
      }
    }

    if (target.source === "bitnet") {
      let active = true
      setVariantLoading(true)
      setBitnetVariants([])
      setSelectedBitnetVariantId("")

      void api.registry
        .getBitnetVariants(target.modelRef)
        .then((variants) => {
          if (!active) return
          setBitnetVariants(variants)
          const preferred = variants.find((v) => v.isDefault) ?? variants[0]
          if (preferred) setSelectedBitnetVariantId(preferred.variantId)
        })
        .catch((err) => {
          if (!active) return
          toast.error(err instanceof Error ? err.message : "No se pudieron cargar variantes BitNet")
        })
        .finally(() => {
          if (active) setVariantLoading(false)
        })

      return () => {
        active = false
      }
    }

    let active = true
    setVariantLoading(true)
    setOllamaVariants([])
    setSelectedVariant(target.modelRef)

    void api.registry
      .getOllamaVariants(target.ollamaBase ?? target.modelRef.split(":", 1)[0])
      .then((variants) => {
        if (!active) return
        setOllamaVariants(variants)
        const preferred = variants.find((v) => v.tag === "latest") ?? variants[0]
        if (preferred) setSelectedVariant(preferred.name)
      })
      .catch((err) => {
        if (!active) return
        toast.error(err instanceof Error ? err.message : "No se pudieron cargar variantes")
      })
      .finally(() => {
        if (active) setVariantLoading(false)
      })

    return () => {
      active = false
    }
  }, [installTarget])

  const install = async () => {
    if (!installTarget) return
    if (selectedHFVariant && selectedHFVariant.installable === false) {
      toast.error(selectedHFVariant.compatibilityReason ?? "Esta variante no es compatible con el stack actual")
      return
    }
    const modelRef = installTarget.source === "ollama" ? (selectedVariant || installTarget.modelRef) : installTarget.modelRef
    const artifact = installTarget.source === "huggingface"
      ? (selectedHFVariant?.artifact ?? null)
      : installTarget.source === "bitnet"
        ? (selectedBitnetVariant?.artifact ?? null)
        : null
    if (installTarget.source === "bitnet" && !selectedBitnetVariant?.artifact) {
      toast.error("Selecciona una variante GGUF BitNet")
      return
    }

    const key = modelKey.trim()
    if (/\s/.test(key)) {
      toast.error("El nombre del modelo no puede contener espacios")
      return
    }
    const useDeployments = currentCompat.length > 0
    if (useDeployments && (deployBackends.length === 0 || !defaultBackend)) {
      toast.error("Elige al menos un backend y cuál sirve el nombre del modelo")
      return
    }

    // With compatibility data the dialog picks the deployments; without it
    // (older server) fall back to the explicit choice, then the variant's own
    // backend, then nothing (the server infers).
    const chosenBackend: BackendType | null = useDeployments
      ? (defaultBackend as BackendType)
      : installTarget.source === "ollama"
        ? null
        : ((selectedBackendType || selectedHFVariant?.backendType || null) as BackendType | null)
    const deploymentConfig = {
      ...(key ? { modelKey: key } : {}),
      ...(useDeployments
        ? {
            defaultBackend: defaultBackend as BackendType,
            // Default first; the server skips it when creating the extras.
            backends: [
              defaultBackend as BackendType,
              ...deployBackends.filter((b) => b !== defaultBackend),
            ],
          }
        : {}),
    }
    const registerConfig = {
      ...(installTarget.source === "huggingface"
        ? toRecipeRegisterConfig(
            installTarget.title,
            loadPolicy,
            selectedHFVariant,
            chosenBackend,
          )
        : {
            displayName: installTarget.title,
            loadPolicy,
            ...(chosenBackend ? { backendType: chosenBackend } : {}),
          }),
      ...deploymentConfig,
    }
    try {
      const job = await api.downloads.enqueue(
        installTarget.source,
        modelRef,
        artifact,
        registerConfig,
      )
      addJob(job)
      toast.success(`Descarga iniciada en ${targetDir} (${loadPolicy}) con preconfiguracion`)
      setInstallTarget(null)
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "No se pudo iniciar descarga")
    }
  }

  const clearHistory = async () => {
    try {
      const result = await api.downloads.clearHistory()
      setJobs(jobs.filter((j) => j.status === "queued" || j.status === "downloading"))
      toast.success(`${result.deleted} jobs eliminados`)
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "No se pudo limpiar historial")
    }
  }

  return (
    <div className="space-y-5 pb-32">
      <div>
        <h1 className="text-2xl font-semibold">Explore</h1>
        <p className="text-muted-foreground">Buscar modelos en HuggingFace, BitNet y Ollama.</p>
      </div>

      <div className="relative">
        {loading ? (
          <div className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 animate-spin rounded-full border-2 border-primary border-t-transparent" />
        ) : (
          <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
        )}
        <input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Buscar modelo"
          className="w-full rounded-lg border border-border bg-card pl-10 pr-9 py-2 focus:outline-none focus:ring-2 focus:ring-primary/50"
        />
        {query && (
          <button
            type="button"
            onClick={() => setQuery("")}
            className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
          >
            <X size={14} />
          </button>
        )}
      </div>

      <Tabs.Root value={activeTab} onValueChange={(value) => setActiveTab(value as "hf" | "bitnet" | "ollama")}>
        <Tabs.List className="flex gap-1 border-b border-border">
          <Tabs.Trigger
            value="hf"
            className="px-4 py-2 text-sm font-medium transition-colors text-muted-foreground hover:text-foreground data-[state=active]:text-foreground data-[state=active]:border-b-2 data-[state=active]:border-primary"
          >
            HuggingFace
          </Tabs.Trigger>
          <Tabs.Trigger
            value="bitnet"
            className="px-4 py-2 text-sm font-medium transition-colors text-muted-foreground hover:text-foreground data-[state=active]:text-foreground data-[state=active]:border-b-2 data-[state=active]:border-primary"
          >
            BitNet
          </Tabs.Trigger>
          <Tabs.Trigger
            value="ollama"
            className="px-4 py-2 text-sm font-medium transition-colors text-muted-foreground hover:text-foreground data-[state=active]:text-foreground data-[state=active]:border-b-2 data-[state=active]:border-primary"
          >
            Ollama
          </Tabs.Trigger>
        </Tabs.List>

        <Tabs.Content value="hf" className="mt-3 space-y-3">
          <SearchFilters
            task={taskFilter}
            size={sizeFilter}
            gated={gatedFilter}
            onTaskChange={setTaskFilter}
            onSizeChange={setSizeFilter}
            onGatedChange={setGatedFilter}
          />
          {loading ? (
            <div className="grid gap-3 md:grid-cols-2">
              {Array.from({ length: 6 }).map((_, idx) => (
                <div key={`hf-skeleton-${idx}`} className="h-32 animate-pulse rounded-lg bg-muted" />
              ))}
            </div>
          ) : (
            <div className="grid gap-3 md:grid-cols-2">
              {hfResults.map((model) => (
                <HFModelCard
                  key={model.repoId}
                  model={model}
                  onInstall={(item) => {
                    setInstallTarget({
                      source: "huggingface",
                      modelRef: item.repoId,
                      title: item.modelName,
                      backendOptions: item.backendOptions,
                      compatibleBackends: item.compatibleBackends,
                    })
                    // Reset so a previous repo's choice never leaks into the
                    // next install; the selector defaults to options[0].
                    setSelectedBackendType("")
                  }}
                />
              ))}
            </div>
          )}
        </Tabs.Content>

        
        <Tabs.Content value="bitnet" className="mt-3 space-y-3">
          <SearchFilters
            task={taskFilter}
            size={sizeFilter}
            gated={gatedFilter}
            onTaskChange={setTaskFilter}
            onSizeChange={setSizeFilter}
            onGatedChange={setGatedFilter}
          />
          {loading ? (
            <div className="grid gap-3 md:grid-cols-2">
              {Array.from({ length: 6 }).map((_, idx) => (
                <div key={`bitnet-skeleton-${idx}`} className="h-32 animate-pulse rounded-lg bg-muted" />
              ))}
            </div>
          ) : (
            <div className="grid gap-3 md:grid-cols-2">
              {bitnetResults.map((model) => (
                <HFModelCard
                  key={model.repoId}
                  model={model}
                  onInstall={(item) =>
                    setInstallTarget({
                      source: "bitnet",
                      modelRef: item.repoId,
                      title: item.modelName,
                      compatibleBackends: item.compatibleBackends,
                    })
                  }
                />
              ))}
            </div>
          )}
        </Tabs.Content>

<Tabs.Content value="ollama" className="mt-3">
          {loading ? (
            <div className="grid gap-3 md:grid-cols-2">
              {Array.from({ length: 6 }).map((_, idx) => (
                <div key={`ollama-skeleton-${idx}`} className="h-32 animate-pulse rounded-lg bg-muted" />
              ))}
            </div>
          ) : (
            <div className="grid gap-3 md:grid-cols-2">
              {ollamaResults.map((model) => (
                <OllamaModelCard
                  key={model.name}
                  model={model}
                  onInstall={(item) =>
                    setInstallTarget({
                      source: "ollama",
                      modelRef: item.name,
                      title: item.name,
                      ollamaBase: item.name,
                      compatibleBackends: item.compatibleBackends,
                    })
                  }
                />
              ))}
            </div>
          )}
        </Tabs.Content>
      </Tabs.Root>

      {jobs.some((j) => j.status === "failed" || j.status === "cancelled" || j.status === "completed") && (
        <div className="flex justify-end">
          <button
            type="button"
            onClick={() => void clearHistory()}
            className="rounded-md border border-border px-3 py-1.5 text-xs text-muted-foreground hover:bg-muted"
          >
            Limpiar historial
          </button>
        </div>
      )}

      <Dialog.Root open={Boolean(installTarget)} onOpenChange={(next) => !next && setInstallTarget(null)}>
        <Dialog.Portal>
          <Dialog.Overlay className="fixed inset-0 z-40 bg-black/60" />
          <Dialog.Content className="fixed left-1/2 top-1/2 z-50 w-[95vw] max-w-md -translate-x-1/2 -translate-y-1/2 rounded-lg border border-border bg-card p-5">
            <Dialog.Title className="text-lg font-semibold">Instalar modelo</Dialog.Title>
            <Dialog.Description className="mt-1 text-sm text-muted-foreground">
              {installTarget?.title}
            </Dialog.Description>

            <div className="mt-4 space-y-3">
              {installTarget?.source === "ollama" && (
                <label className="block text-sm text-muted-foreground">
                  Variante (billones / quant)
                  <select
                    value={selectedVariant}
                    disabled={variantLoading}
                    onChange={(event) => setSelectedVariant(event.target.value)}
                    className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2"
                  >
                    {variantLoading ? (
                      <option value="">Cargando variantes...</option>
                    ) : ollamaVariants.length === 0 ? (
                      <option value={installTarget.modelRef}>{installTarget.modelRef}</option>
                    ) : (
                      ollamaVariants.map((variant) => (
                        <option key={variant.name} value={variant.name}>
                          {variant.name}
                          {variant.sizeGb ? ` · ${variant.sizeGb.toFixed(1)} GB` : ""}
                          {variant.quantization ? ` · ${variant.quantization}` : ""}
                        </option>
                      ))
                    )}
                  </select>
                </label>
              )}
              {installTarget?.source === "bitnet" && (
                <label className="block text-sm text-muted-foreground">
                  Variante GGUF BitNet
                  <select
                    value={selectedBitnetVariantId}
                    disabled={variantLoading}
                    onChange={(event) => setSelectedBitnetVariantId(event.target.value)}
                    className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2"
                  >
                    {variantLoading ? (
                      <option value="">Cargando variantes...</option>
                    ) : bitnetVariants.length === 0 ? (
                      <option value="">Sin variantes compatibles</option>
                    ) : (
                      bitnetVariants.map((variant) => (
                        <option key={variant.variantId} value={variant.variantId}>
                          {variant.label}
                          {variant.sizeGb ? ` · ${variant.sizeGb.toFixed(1)} GB` : ""}
                          {variant.quantization ? ` · ${variant.quantization}` : ""}
                        </option>
                      ))
                    )}
                  </select>
                </label>
              )}
              {installTarget?.source === "huggingface" && (
                <>
                  <label className="block text-sm text-muted-foreground">
                    Variante HF
                    <select
                      value={selectedHFVariantId}
                      disabled={variantLoading}
                      onChange={(event) => setSelectedHFVariantId(event.target.value)}
                      className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2"
                    >
                      {variantLoading ? (
                        <option value="">Cargando variantes...</option>
                      ) : hfVariants.length === 0 ? (
                        <option value="">Default</option>
                      ) : (
                        hfVariants.map((variant) => (
                          <option key={variant.variantId} value={variant.variantId}>
                            {variant.label}
                            {variant.sizeGb ? ` · ${variant.sizeGb.toFixed(1)} GB` : ""}
                            {variant.quantization ? ` · ${variant.quantization}` : ""}
                            {variant.backendType ? ` · ${variant.backendType}` : ""}
                            {variant.installable === false ? " · incompatible" : ""}
                          </option>
                        ))
                      )}
                    </select>
                  </label>
                  {selectedHFVariant?.compatibilityReason && (
                    <div
                      className={`rounded-md border px-3 py-2 text-sm ${
                        selectedHFVariant.installable === false
                          ? "border-red-500/30 bg-red-500/10 text-red-200"
                          : selectedHFVariant.compatibility === "warning"
                            ? "border-amber-500/30 bg-amber-500/10 text-amber-200"
                            : "border-border bg-background text-muted-foreground"
                      }`}
                    >
                      {selectedHFVariant.compatibilityReason}
                    </div>
                  )}
                  {(selectedHFVariant?.vllmSupport || probePreconfigurationNotice || recipeProbeDifferenceNotice) && (
                    <AdvancedDetails>
                      {selectedHFVariant?.vllmSupport && (
                        <div className="rounded-md border border-border bg-background px-3 py-2 text-xs text-muted-foreground">
                          {`vLLM: ${selectedHFVariant.vllmSupport.label}`}
                          {selectedHFVariant.vllmSupport.modelImpl
                            ? ` · model_impl=${selectedHFVariant.vllmSupport.modelImpl}`
                            : ""}
                          {selectedHFVariant.vllmSupport.runner
                            ? ` · runner=${selectedHFVariant.vllmSupport.runner}`
                            : ""}
                          {selectedHFVariant.vllmSupport.requiredOverrides.length > 0
                            ? ` · requiere ${selectedHFVariant.vllmSupport.requiredOverrides.join(", ")}`
                            : ""}
                          {probeStatusLabel
                            ? ` · probe=${probeStatusLabel}`
                            : ""}
                        </div>
                      )}
                      {selectedHFVariant?.vllmSupport?.runtimeProbe && (
                        <div className="rounded-md border border-sky-500/20 bg-sky-500/10 px-3 py-2 text-xs text-sky-100">
                          {`Recomendacion verificada por probe`}
                          {selectedHFVariant.vllmSupport.runtimeProbe.recommendedModelImpl
                            ? ` · model_impl=${selectedHFVariant.vllmSupport.runtimeProbe.recommendedModelImpl}`
                            : ""}
                          {selectedHFVariant.vllmSupport.runtimeProbe.recommendedRunner
                            ? ` · runner=${selectedHFVariant.vllmSupport.runtimeProbe.recommendedRunner}`
                            : ""}
                          {selectedHFVariant.vllmSupport.runtimeProbe.reason
                            ? ` · ${selectedHFVariant.vllmSupport.runtimeProbe.reason}`
                            : ""}
                          {selectedHFVariant.vllmSupport.runtimeProbe.observedAt
                            ? ` · observado=${selectedHFVariant.vllmSupport.runtimeProbe.observedAt}`
                            : ""}
                          {probeOverrideHint ? ` · ${probeOverrideHint}` : ""}
                        </div>
                      )}
                      {probePreconfigurationNotice && (
                        <div className="rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-100">
                          {probePreconfigurationNotice}
                        </div>
                      )}
                      {recipeProbeDifferenceNotice && (
                        <div className="rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-100">
                          {recipeProbeDifferenceNotice}
                        </div>
                      )}
                    </AdvancedDetails>
                  )}
                  {selectedHFVariant?.vllmSupport?.recipeId && (
                    <div className="rounded-md border border-emerald-500/20 bg-emerald-500/10 px-3 py-2 text-xs text-emerald-100">
                      {`Al completar la descarga se preconfigurara recipe=${selectedHFVariant.vllmSupport.recipeId}`}
                      {selectedHFVariant.vllmSupport.recipeModelImpl
                        ? ` · recipe_model_impl=${selectedHFVariant.vllmSupport.recipeModelImpl}`
                        : ""}
                      {selectedHFVariant.vllmSupport.recipeRunner
                        ? ` · recipe_runner=${selectedHFVariant.vllmSupport.recipeRunner}`
                        : ""}
                      {Object.keys(selectedHFVariant.vllmSupport.suggestedConfig).length > 0
                        ? ` · ${Object.entries(selectedHFVariant.vllmSupport.suggestedConfig)
                            .map(([key, value]) => `${key}=${typeof value === "string" ? value : JSON.stringify(value)}`)
                            .join(", ")}`
                        : ""}
                    </div>
                  )}
                </>
              )}
              {currentCompat.length > 0 && (
                <div className="space-y-1 text-sm text-muted-foreground">
                  <span className="block">Backends compatibles con este fichero</span>
                  <BackendIcons items={compatibleToItems(currentCompat)} />
                </div>
              )}
              <label className="block text-sm text-muted-foreground">
                Nombre del modelo
                <input
                  aria-label="Nombre del modelo"
                  value={modelKey}
                  onChange={(event) => setModelKey(event.target.value)}
                  placeholder="nombre público, p. ej. gemma4:e4b"
                  className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2 font-mono text-sm text-foreground"
                />
                <span className="mt-1 block text-xs">
                  Nombre con el que se pedirá el modelo en la API. Los demás backends quedan
                  accesibles como «backend/nombre» sin listarse.
                </span>
              </label>
              {currentCompat.length > 0 && (
                <fieldset className="space-y-1.5 rounded-md border border-border px-3 py-2 text-sm">
                  <legend className="px-1 text-muted-foreground">Despliegues a crear</legend>
                  {currentCompat.map((entry) => {
                    const checked = deployBackends.includes(entry.backend)
                    return (
                      <div key={entry.backend} className="flex items-center justify-between gap-2">
                        <label
                          className={`inline-flex items-center gap-2 ${entry.installed ? "" : "text-muted-foreground/60"}`}
                          title={entry.installed ? entry.reason : "no instalado"}
                        >
                          <input
                            type="checkbox"
                            aria-label={`Desplegar con ${entry.backend}`}
                            checked={checked}
                            disabled={!entry.installed}
                            onChange={(event) => toggleDeployBackend(entry.backend, event.target.checked)}
                          />
                          {backendLabel(entry.backend)}
                          {entry.status === "probable" && <span className="text-xs">(probable)</span>}
                          {!entry.installed && <span className="text-xs">(no instalado)</span>}
                        </label>
                        <label className="inline-flex items-center gap-1 text-xs text-muted-foreground">
                          <input
                            type="radio"
                            name="default-backend"
                            aria-label={`Por defecto: ${entry.backend}`}
                            checked={defaultBackend === entry.backend}
                            disabled={!checked}
                            onChange={() => setDefaultBackend(entry.backend)}
                          />
                          por defecto
                        </label>
                      </div>
                    )
                  })}
                  <span className="block text-xs text-muted-foreground">
                    El backend por defecto sirve el nombre del modelo. Se puede cambiar después en Modelos.
                  </span>
                </fieldset>
              )}
              <label className="block text-sm text-muted-foreground">
                Carpeta destino
                <input
                  value={targetDir}
                  onChange={(event) => setTargetDir(event.target.value)}
                  className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2"
                />
              </label>
              {installTarget?.source !== "ollama" && currentCompat.length === 0 &&
                (() => {
                  // Options the repo actually supports, falling back to the
                  // selected variant's own backend. Only worth a selector when
                  // there is a real choice to make.
                  const options =
                    installTarget?.backendOptions && installTarget.backendOptions.length > 0
                      ? installTarget.backendOptions
                      : selectedHFVariant?.backendType
                        ? [selectedHFVariant.backendType]
                        : []
                  if (options.length < 2) return null
                  return (
                    <label className="block text-sm text-muted-foreground">
                      backend
                      <select
                        value={selectedBackendType || options[0]}
                        onChange={(event) =>
                          setSelectedBackendType(event.target.value as BackendType)
                        }
                        className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2"
                      >
                        {options.map((option) => (
                          <option key={option} value={option}>
                            {option}
                          </option>
                        ))}
                      </select>
                      <span className="mt-1 block text-xs">
                        Este repositorio se puede servir con varios backends. El elegido
                        queda fijado al registrar el modelo.
                      </span>
                    </label>
                  )
                })()}
              <label className="block text-sm text-muted-foreground">
                load_policy
                <select
                  value={loadPolicy}
                  onChange={(event) => setLoadPolicy(event.target.value as LoadPolicy)}
                  className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2"
                >
                  <option value="on_demand">on_demand</option>
                  <option value="warm">warm</option>
                  <option value="pin">pin</option>
                </select>
              </label>
            </div>

            <div className="mt-4 flex justify-end gap-2">
              <Dialog.Close className="rounded-md border border-border px-3 py-2 text-sm hover:bg-muted">
                Cancelar
              </Dialog.Close>
              <button
                type="button"
                onClick={() => void install()}
                disabled={selectedHFVariant?.installable === false || !canInstallBitnet}
                className="rounded-md bg-primary px-3 py-2 text-sm font-medium text-primary-foreground"
              >
                Iniciar descarga
              </button>
            </div>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>

    </div>
  )
}
