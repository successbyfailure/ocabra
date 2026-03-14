import { useEffect, useState } from "react"
import * as Dialog from "@radix-ui/react-dialog"
import * as Tabs from "@radix-ui/react-tabs"
import { toast } from "sonner"
import { api } from "@/api/client"
import { DownloadQueue } from "@/components/downloads/DownloadQueue"
import { HFModelCard } from "@/components/explore/HFModelCard"
import { OllamaModelCard } from "@/components/explore/OllamaModelCard"
import { SearchFilters } from "@/components/explore/SearchFilters"
import { getProbeOverrideHint, getProbeStatusLabel } from "@/lib/vllmProbe"
import { useDownloadStore } from "@/stores/downloadStore"
import type {
  DownloadJob,
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
) {
  const support = variant?.vllmSupport
  if (!support) {
    return {
      displayName: title,
      loadPolicy,
    }
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
    displayName: title,
    loadPolicy,
    extraConfig: {
      vllm: vllmConfig,
    },
  }
}

export function Explore() {
  const [activeTab, setActiveTab] = useState<"hf" | "ollama">("hf")
  const [query, setQuery] = useState("mistral")
  const [debouncedQuery, setDebouncedQuery] = useState(query)
  const [taskFilter, setTaskFilter] = useState("")
  const [sizeFilter, setSizeFilter] = useState("")
  const [gatedFilter, setGatedFilter] = useState("")
  const [hfResults, setHfResults] = useState<HFCardType[]>([])
  const [ollamaResults, setOllamaResults] = useState<OllamaCardType[]>([])
  const [loading, setLoading] = useState(false)
  const [installTarget, setInstallTarget] = useState<InstallTarget | null>(null)
  const [ollamaVariants, setOllamaVariants] = useState<OllamaModelVariant[]>([])
  const [hfVariants, setHfVariants] = useState<HFModelVariant[]>([])
  const [selectedHFVariantId, setSelectedHFVariantId] = useState("")
  const [selectedVariant, setSelectedVariant] = useState("")
  const [variantLoading, setVariantLoading] = useState(false)
  const [targetDir, setTargetDir] = useState("/models")
  const [loadPolicy, setLoadPolicy] = useState<LoadPolicy>("on_demand")
  const selectedHFVariant = installTarget?.source === "huggingface"
    ? hfVariants.find((v) => v.variantId === selectedHFVariantId) ?? null
    : null
  const probePreconfigurationNotice = getProbePreconfigurationNotice(selectedHFVariant)
  const recipeProbeDifferenceNotice = getRecipeProbeDifferenceNotice(selectedHFVariant)
  const probeStatusLabel = getProbeStatusLabel(selectedHFVariant?.vllmSupport?.runtimeProbe?.status)
  const probeOverrideHint = getProbeOverrideHint(selectedHFVariant?.vllmSupport?.runtimeProbe?.status)

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
    const artifact = selectedHFVariant?.artifact ?? null
    const registerConfig =
      installTarget.source === "huggingface"
        ? toRecipeRegisterConfig(installTarget.title, loadPolicy, selectedHFVariant)
        : {
            displayName: installTarget.title,
            loadPolicy,
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

  const cancelJob = async (job: DownloadJob) => {
    try {
      await api.downloads.cancel(job.jobId)
      setJobs(jobs.map((item) => (item.jobId === job.jobId ? { ...item, status: "cancelled" } : item)))
      toast.success("Descarga cancelada")
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "No se pudo cancelar")
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
        <p className="text-muted-foreground">Buscar modelos en HuggingFace y Ollama.</p>
      </div>

      <input
        value={query}
        onChange={(event) => setQuery(event.target.value)}
        placeholder="Buscar modelo"
        className="w-full rounded-lg border border-border bg-card px-3 py-2"
      />

      <Tabs.Root value={activeTab} onValueChange={(value) => setActiveTab(value as "hf" | "ollama")}>
        <Tabs.List className="inline-flex rounded-lg border border-border bg-card p-1">
          <Tabs.Trigger
            value="hf"
            className="rounded-md px-3 py-1.5 text-sm text-muted-foreground data-[state=active]:bg-muted data-[state=active]:text-foreground"
          >
            HuggingFace
          </Tabs.Trigger>
          <Tabs.Trigger
            value="ollama"
            className="rounded-md px-3 py-1.5 text-sm text-muted-foreground data-[state=active]:bg-muted data-[state=active]:text-foreground"
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
                  onInstall={(item) =>
                    setInstallTarget({
                      source: "huggingface",
                      modelRef: item.repoId,
                      title: item.modelName,
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
              <label className="block text-sm text-muted-foreground">
                Carpeta destino
                <input
                  value={targetDir}
                  onChange={(event) => setTargetDir(event.target.value)}
                  className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2"
                />
              </label>
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
                disabled={selectedHFVariant?.installable === false}
                className="rounded-md bg-primary px-3 py-2 text-sm font-medium text-primary-foreground"
              >
                Iniciar descarga
              </button>
            </div>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>

      <DownloadQueue jobs={jobs} onCancel={cancelJob} />
    </div>
  )
}
