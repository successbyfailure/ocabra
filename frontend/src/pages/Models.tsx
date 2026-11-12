import { useEffect, useMemo, useState } from "react"
import { useNavigate } from "react-router-dom"
import * as Dialog from "@radix-ui/react-dialog"
import * as Tooltip from "@radix-ui/react-tooltip"
import { toast } from "sonner"
import { api } from "@/api/client"
import { EmptyState } from "@/components/common/EmptyState"
import { CompileModal } from "@/components/models/CompileModal"
import { LogicalModelCard } from "@/components/models/LogicalModelCard"
import { ModelGridCard } from "@/components/models/ModelGridCard"
import { ProfileModal } from "@/components/models/ProfileModal"
import { useWebSocket } from "@/hooks/useWebSocket"
import {
  categorizeModel,
  getModelContextSummary,
  MODEL_CATEGORY_LABELS,
  MODEL_CATEGORY_ORDER,
  type ModelCategory,
} from "@/lib/modelContext"
import { useBackendsStore } from "@/stores/backendsStore"
import { useGpuStore } from "@/stores/gpuStore"
import { useModelStore } from "@/stores/modelStore"
import { ArrowDown, ArrowUp, Loader2, RefreshCw, Search } from "lucide-react"
import { ConfirmDialog } from "@/components/common/ConfirmDialog"
import { SkeletonList } from "@/components/common/Skeleton"
import { StyledSelect } from "@/components/common/StyledSelect"
import type {
  BackendType,
  LoadPolicy,
  LogicalModel,
  ModelDeployment,
  ModelProfile,
  ModelState,
  ModelStatus,
  ModelsStorageStats,
} from "@/types"

type ViewMode = "model" | "deployment"

/** Server profile category → page tab, for models whose default deployment
 *  is not in the model store yet. */
const PROFILE_CATEGORY_TO_TAB: Record<string, ModelCategory> = {
  llm: "chat",
  embedding: "embeddings",
  rerank: "embeddings",
  tts: "tts",
  stt: "stt",
  image: "image",
  music: "music",
  video_upscaling: "video",
}

export function categorizeLogicalModel(
  model: LogicalModel,
  states: Record<string, ModelState | undefined>,
): ModelCategory {
  const preferred =
    model.deployments.find((dep) => dep.isDefault && states[dep.modelId]) ??
    model.deployments.find((dep) => states[dep.modelId])
  const state = preferred ? states[preferred.modelId] : undefined
  if (state) return categorizeModel(state)
  return (model.category && PROFILE_CATEGORY_TO_TAB[model.category]) || "other"
}

export function Models() {
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [query, setQuery] = useState("")
  const [statusFilter, setStatusFilter] = useState<ModelStatus | "all">("all")
  const [categoryTab, setCategoryTab] = useState<ModelCategory>("chat")
  const [backendFilter, setBackendFilter] = useState<BackendType | "all">("all")
  const [gpuFilter, setGpuFilter] = useState<string>("all")
  const [sortKey, setSortKey] = useState<
    "name" | "backend" | "policy" | "gpu" | "ctxNative" | "ctxConfig" | "io" | "vram" | "size" | "status"
  >("name")
  const [sortDir, setSortDir] = useState<"asc" | "desc">("asc")
  const [busyModelId, setBusyModelId] = useState<string | null>(null)
  const [updatingAll, setUpdatingAll] = useState(false)
  const [refreshingMetadata, setRefreshingMetadata] = useState(false)
  const navigate = useNavigate()
  // El editor de modelos es una página: /models/edit/<backend>/<modelo>.
  const openEditor = (modelId: string) => navigate(`/models/edit/${encodeURIComponent(modelId)}`)
  const [deleteModel, setDeleteModel] = useState<ModelState | null>(null)
  const [compileModel, setCompileModel] = useState<ModelState | null>(null)
  const [storage, setStorage] = useState<ModelsStorageStats | null>(null)
  const [viewMode, setViewMode] = useState<ViewMode>("model")
  const [logicalModels, setLogicalModels] = useState<LogicalModel[]>([])
  const [logicalError, setLogicalError] = useState<string | null>(null)
  const [removeTarget, setRemoveTarget] = useState<{ model: LogicalModel; deployment: ModelDeployment } | null>(null)

  // Confirm dialog state
  const [confirmAction, setConfirmAction] = useState<{ modelId: string; action: "load" | "unload" } | null>(null)

  // Profile modal state
  const [profileModalModel, setProfileModalModel] = useState<ModelState | null>(null)
  const [profileModalProfile, setProfileModalProfile] = useState<ModelProfile | null>(null)
  const [profileModalOpen, setProfileModalOpen] = useState(false)

  useWebSocket()

  const gpus = useGpuStore((state) => state.gpus)
  const setGpus = useGpuStore((state) => state.setGpus)

  const models = useModelStore((state) => state.models)
  const setModels = useModelStore((state) => state.setModels)
  const loadModel = useModelStore((state) => state.loadModel)
  const unloadModel = useModelStore((state) => state.unloadModel)
  // Pre-fetch backends so the BackendBadge in ModelCard has data on first paint.
  const fetchBackends = useBackendsStore((state) => state.fetchAll)

  const modelList = useMemo(() => Object.values(models), [models])

  const refresh = async () => {
    const [gpuList, modelListResp, storageStats] = await Promise.all([
      api.gpus.list(),
      api.models.list(),
      api.models.storage(),
    ])
    setGpus(gpuList)
    setModels(modelListResp)
    setStorage(storageStats)
    await refreshLogical()
  }

  // Separate from the deployment list so a failure here (older server, role
  // without access) never takes the per-deployment view down with it.
  const refreshLogical = async () => {
    try {
      setLogicalModels(await api.logicalModels.list())
      setLogicalError(null)
    } catch (err) {
      setLogicalError(err instanceof Error ? err.message : "No se pudieron cargar los modelos lógicos")
    }
  }

  const formatBytes = (bytes: number) => {
    if (bytes <= 0) return "0 B"
    const units = ["B", "KB", "MB", "GB", "TB"]
    let value = bytes
    let unitIndex = 0
    while (value >= 1024 && unitIndex < units.length - 1) {
      value /= 1024
      unitIndex += 1
    }
    return `${value.toFixed(value >= 100 || unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`
  }

  useEffect(() => {
    let active = true

    const bootstrap = async () => {
      try {
        await Promise.all([refresh(), fetchBackends()])
        if (active) setError(null)
      } catch (err) {
        if (active) {
          setError(err instanceof Error ? err.message : "No se pudieron cargar los modelos")
        }
      } finally {
        if (active) setLoading(false)
      }
    }

    void bootstrap()
    const timer = window.setInterval(() => {
      void refresh().catch(() => {
        // silent background refresh error
      })
    }, 30_000)

    return () => {
      active = false
      window.clearInterval(timer)
    }
  }, [setGpus, setModels, fetchBackends])

  // Every model annotated with its category + derived sort fields, filtered
  // by everything EXCEPT the active category tab — this is what the tab
  // counts are computed from, so switching tabs doesn't make the other
  // counts jump around.
  const matchingOtherFilters = useMemo(() => {
    return modelList
      .map((model) => {
        const context = getModelContextSummary(model)
        return {
          model,
          category: categorizeModel(model),
          context,
          modelGpu: model.currentGpu[0] ?? model.preferredGpu,
          ioTokens: Math.max(context.maxInputTokens ?? 0, context.maxOutputTokens ?? 0),
        }
      })
      .filter(({ model, modelGpu }) => {
        const matchesQuery =
          model.displayName.toLowerCase().includes(query.toLowerCase()) ||
          model.modelId.toLowerCase().includes(query.toLowerCase())
        const matchesStatus = statusFilter === "all" || model.status === statusFilter
        const matchesBackend = backendFilter === "all" || model.backendType === backendFilter
        const matchesGpu = gpuFilter === "all" || String(modelGpu) === gpuFilter
        return matchesQuery && matchesStatus && matchesBackend && matchesGpu
      })
  }, [backendFilter, gpuFilter, modelList, query, statusFilter])

  const logicalMatchingOtherFilters = useMemo(() => {
    const q = query.toLowerCase()
    return logicalModels
      .map((model) => ({ model, category: categorizeLogicalModel(model, models) }))
      .filter(({ model }) => {
        const matchesQuery =
          !q ||
          model.modelKey.toLowerCase().includes(q) ||
          model.displayName.toLowerCase().includes(q) ||
          model.deployments.some((dep) => dep.modelId.toLowerCase().includes(q))
        const matchesStatus =
          statusFilter === "all" ||
          model.deployments.some((dep) => (models[dep.modelId]?.status ?? dep.status) === statusFilter)
        const matchesBackend =
          backendFilter === "all" || model.deployments.some((dep) => dep.backend === backendFilter)
        const matchesGpu =
          gpuFilter === "all" ||
          model.deployments.some((dep) => {
            const state = models[dep.modelId]
            return state != null && String(state.currentGpu[0] ?? state.preferredGpu) === gpuFilter
          })
        return matchesQuery && matchesStatus && matchesBackend && matchesGpu
      })
  }, [backendFilter, gpuFilter, logicalModels, models, query, statusFilter])

  const categoryCounts = useMemo(() => {
    const counts = new Map<ModelCategory, number>()
    const source = viewMode === "model" ? logicalMatchingOtherFilters : matchingOtherFilters
    for (const { category } of source) {
      counts.set(category, (counts.get(category) ?? 0) + 1)
    }
    return counts
  }, [logicalMatchingOtherFilters, matchingOtherFilters, viewMode])

  const logicalFiltered = useMemo(() => {
    const visible = logicalMatchingOtherFilters.filter(({ category }) => category === categoryTab)
    visible.sort((left, right) => {
      const cmp = (left.model.displayName || left.model.modelKey)
        .toLowerCase()
        .localeCompare((right.model.displayName || right.model.modelKey).toLowerCase())
      return sortDir === "asc" ? cmp : -cmp
    })
    return visible
  }, [categoryTab, logicalMatchingOtherFilters, sortDir])

  const filtered = useMemo(() => {
    const visible = matchingOtherFilters.filter(({ category }) => category === categoryTab)

    visible.sort((left, right) => {
      const byKey: Record<typeof sortKey, string | number> = {
        name: left.model.displayName.toLowerCase(),
        backend: left.model.backendType,
        policy: left.model.loadPolicy,
        gpu: left.modelGpu ?? -1,
        ctxNative: left.context.nativeContext ?? -1,
        ctxConfig: left.context.configuredContext ?? -1,
        io: left.ioTokens,
        vram: left.model.vramUsedMb,
        size: left.model.diskSizeBytes ?? -1,
        status: left.model.status,
      }
      const otherByKey: Record<typeof sortKey, string | number> = {
        name: right.model.displayName.toLowerCase(),
        backend: right.model.backendType,
        policy: right.model.loadPolicy,
        gpu: right.modelGpu ?? -1,
        ctxNative: right.context.nativeContext ?? -1,
        ctxConfig: right.context.configuredContext ?? -1,
        io: right.ioTokens,
        vram: right.model.vramUsedMb,
        size: right.model.diskSizeBytes ?? -1,
        status: right.model.status,
      }

      const a = byKey[sortKey]
      const b = otherByKey[sortKey]
      const cmp = typeof a === "string" && typeof b === "string" ? a.localeCompare(b) : Number(a) - Number(b)
      return sortDir === "asc" ? cmp : -cmp
    })

    return visible
  }, [categoryTab, matchingOtherFilters, sortDir, sortKey])

  const runAction = async (modelId: string, action: () => Promise<void>) => {
    setBusyModelId(modelId)
    try {
      await action()
      toast.success("Accion completada")
      await refresh()
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Error")
    } finally {
      setBusyModelId(null)
    }
  }

  const handleUpdateModel = async (model: ModelState) => {
    setBusyModelId(model.modelId)
    try {
      await api.models.update(model.modelId)
      toast.success(`Re-pull encolado · ${model.displayName || model.modelId}`)
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "No se pudo actualizar el modelo")
    } finally {
      setBusyModelId(null)
    }
  }

  const handleRefreshMetadata = async () => {
    if (refreshingMetadata) return
    setRefreshingMetadata(true)
    try {
      const result = await api.models.refreshMetadata()
      const refreshed = result.refreshed.length
      const skipped = result.skipped.length
      toast.success(
        `Metadata actualizada: ${refreshed} modelo${refreshed === 1 ? "" : "s"}` +
          (skipped > 0 ? ` · ${skipped} sin origen remoto` : ""),
      )
      await refresh()
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Error al refrescar metadata")
    } finally {
      setRefreshingMetadata(false)
    }
  }

  const handleUpdateAll = async () => {
    if (updatingAll) return
    if (!confirm("Se encolará un re-pull de todos los modelos actualizables. ¿Continuar?")) return
    setUpdatingAll(true)
    try {
      const result = await api.models.updateAll()
      const enq = result.enqueued.length
      const skipped = result.skipped.length
      toast.success(
        `Re-pull encolado: ${enq} modelo${enq === 1 ? "" : "s"}` +
          (skipped > 0 ? ` · ${skipped} omitido${skipped === 1 ? "" : "s"}` : ""),
      )
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Error al actualizar modelos")
    } finally {
      setUpdatingAll(false)
    }
  }

  const togglePin = async (model: ModelState) => {
    const nextPolicy: LoadPolicy = model.loadPolicy === "pin" ? "on_demand" : "pin"
    await runAction(model.modelId, async () => {
      await api.models.patch(model.modelId, { loadPolicy: nextPolicy })
    })
  }


  const openProfileModal = (model: ModelState, profile: ModelProfile | null) => {
    setProfileModalModel(model)
    setProfileModalProfile(profile)
    setProfileModalOpen(true)
  }

  const handleToggleProfileEnabled = async (profile: ModelProfile) => {
    try {
      await api.profiles.update(profile.profileId, { enabled: !profile.enabled })
      toast.success(profile.enabled ? "Perfil desactivado" : "Perfil activado")
      await refresh()
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Error al cambiar estado del perfil")
    }
  }

  const runLogicalAction = async (modelKey: string, action: () => Promise<string>) => {
    setBusyModelId(modelKey)
    try {
      const message = await action()
      toast.success(message)
      await refresh()
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Error")
    } finally {
      setBusyModelId(null)
    }
  }

  const handleSetDefaultBackend = (model: LogicalModel, backend: BackendType) =>
    void runLogicalAction(model.modelKey, async () => {
      await api.logicalModels.patch(model.modelKey, { defaultBackend: backend })
      return `${model.modelKey} se sirve ahora con ${backend}`
    })

  const handleAddDeployment = (model: LogicalModel, backend: BackendType) =>
    void runLogicalAction(model.modelKey, async () => {
      await api.logicalModels.addDeployment(model.modelKey, { backend })
      return `Despliegue ${backend}/${model.modelKey} creado`
    })

  const handleRemoveDeployment = (model: LogicalModel, deployment: ModelDeployment) =>
    void runLogicalAction(model.modelKey, async () => {
      const result = await api.logicalModels.removeDeployment(model.modelKey, deployment.backend)
      if (model.deployments.length <= 1) {
        return `${model.modelKey} eliminado (los ficheros se conservan)`
      }
      return result.newDefaultBackend && deployment.isDefault
        ? `Despliegue quitado · ${model.modelKey} se sirve ahora con ${result.newDefaultBackend}`
        : "Despliegue quitado"
    })

  const openDeploymentConfig = (modelId: string) => openEditor(modelId)


  const removeDescription = (() => {
    if (!removeTarget) return ""
    const { model, deployment } = removeTarget
    const parts = [
      `Se quitará el despliegue ${deployment.modelId}. Solo se borra la configuración: los ficheros del modelo se conservan.`,
    ]
    if (model.deployments.length <= 1) {
      parts.push(`Es el único despliegue, así que el modelo ${model.modelKey} desaparecerá.`)
    } else if (deployment.isDefault) {
      const next = model.deployments.find((dep) => dep.modelId !== deployment.modelId)
      parts.push(
        `Es el backend por defecto: el nombre público ${model.modelKey} pasará antes a otro despliegue${
          next ? ` (probablemente ${next.backend})` : ""
        }.`,
      )
    }
    return parts.join(" ")
  })()

  const visibleCount = viewMode === "model" ? logicalFiltered.length : filtered.length
  const deleteModelProfiles = deleteModel?.profiles ?? []

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold">Models</h1>
          <p className="text-muted-foreground">Gestion de modelos instalados y ciclo de vida.</p>
        </div>
        <div className="flex flex-wrap gap-2">
          <div className="inline-flex rounded-md border border-border p-0.5" role="group" aria-label="Vista">
            {(
              [
                ["model", "Por modelo"],
                ["deployment", "Por despliegue"],
              ] as const
            ).map(([mode, label]) => (
              <button
                key={mode}
                type="button"
                aria-pressed={viewMode === mode}
                onClick={() => setViewMode(mode)}
                className={`rounded px-3 py-1 text-sm transition-colors ${
                  viewMode === mode ? "bg-primary/15 font-medium text-foreground" : "text-muted-foreground hover:text-foreground"
                }`}
              >
                {label}
              </button>
            ))}
          </div>
          <button
            type="button"
            onClick={() => void handleRefreshMetadata()}
            disabled={refreshingMetadata || modelList.length === 0}
            className="inline-flex items-center gap-1.5 rounded-md border border-border px-3 py-1.5 text-sm hover:bg-muted disabled:cursor-not-allowed disabled:opacity-50"
            title="Refresca fechas de release y last-modified desde el registry upstream (HuggingFace / Ollama)"
          >
            {refreshingMetadata ? <Loader2 size={14} className="animate-spin" /> : <RefreshCw size={14} />}
            Refrescar metadata
          </button>
          <button
            type="button"
            onClick={() => void handleUpdateAll()}
            disabled={updatingAll || modelList.length === 0}
            className="inline-flex items-center gap-1.5 rounded-md border border-border px-3 py-1.5 text-sm hover:bg-muted disabled:cursor-not-allowed disabled:opacity-50"
            title="Re-pull de todos los modelos con origen remoto conocido"
          >
            {updatingAll ? <Loader2 size={14} className="animate-spin" /> : <RefreshCw size={14} />}
            Actualizar todos
          </button>
        </div>
      </div>

      {storage && (() => {
        const usedPct = storage.totalBytes > 0 ? Math.max(0, Math.min(100, (storage.usedBytes / storage.totalBytes) * 100)) : 0
        const barColor = usedPct > 90 ? "bg-red-500" : usedPct > 75 ? "bg-amber-500" : "bg-blue-500"
        return (
          <div className="rounded-lg border border-border bg-card p-3">
            <div className="mb-2 flex items-center justify-between gap-3 text-sm">
              <div>
                <p className="font-medium">Almacenamiento de modelos</p>
                <p className="text-xs text-muted-foreground">{storage.path}</p>
              </div>
              <div className="text-right text-xs text-muted-foreground">
                <p className="font-medium text-foreground">{formatBytes(storage.usedBytes)} usados</p>
                <p>{formatBytes(storage.freeBytes)} libres de {formatBytes(storage.totalBytes)}</p>
              </div>
            </div>
            <div className="h-3 overflow-hidden rounded-full bg-muted" role="progressbar" aria-valuenow={Math.round(usedPct)} aria-valuemin={0} aria-valuemax={100} aria-label="Espacio usado en disco">
              <div
                className={`h-full rounded-full transition-[width] ${barColor}`}
                style={{ width: `${usedPct}%` }}
              />
            </div>
          </div>
        )
      })()}

      <div className="grid gap-2 rounded-lg border border-border bg-card p-3 md:grid-cols-5">
        <div className="relative">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Buscar modelo"
            className="w-full rounded-md border border-border bg-background pl-9 pr-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
          />
        </div>

        <StyledSelect
          value={statusFilter}
          onValueChange={(v) => setStatusFilter(v as ModelStatus | "all")}
          className="w-full"
          options={[
            { value: "all", label: "Status: todos" },
            { value: "configured", label: "configured" },
            { value: "loading", label: "loading" },
            { value: "loaded", label: "loaded" },
            { value: "unloading", label: "unloading" },
            { value: "unloaded", label: "unloaded" },
            { value: "error", label: "error" },
          ]}
        />

        <div className="flex items-center gap-1.5">
          <StyledSelect
            value={sortKey}
            onValueChange={(v) => setSortKey(v as typeof sortKey)}
            className="w-full"
            options={[
              { value: "name", label: "Ordenar: nombre" },
              { value: "status", label: "Ordenar: status" },
              { value: "vram", label: "Ordenar: VRAM" },
              { value: "gpu", label: "Ordenar: GPU" },
              { value: "policy", label: "Ordenar: policy" },
              { value: "backend", label: "Ordenar: backend" },
              { value: "ctxNative", label: "Ordenar: ctx nativo" },
              { value: "ctxConfig", label: "Ordenar: ctx config" },
              { value: "size", label: "Ordenar: tamaño" },
            ]}
          />
          <button
            type="button"
            onClick={() => setSortDir((d) => (d === "asc" ? "desc" : "asc"))}
            className="shrink-0 rounded-md border border-border p-2 text-muted-foreground hover:bg-muted hover:text-foreground"
            title={sortDir === "asc" ? "Ascendente" : "Descendente"}
          >
            {sortDir === "asc" ? <ArrowUp size={14} /> : <ArrowDown size={14} />}
          </button>
        </div>

        <StyledSelect
          value={backendFilter}
          onValueChange={(v) => setBackendFilter(v as BackendType | "all")}
          className="w-full"
          options={[
            { value: "all", label: "Backend: todos" },
            ...Array.from(new Set(modelList.map((model) => model.backendType))).sort().map((backend) => ({
              value: backend,
              label: backend,
            })),
          ]}
        />

        <StyledSelect
          value={gpuFilter}
          onValueChange={setGpuFilter}
          className="w-full"
          options={[
            { value: "all", label: "GPU: todas" },
            ...gpus.map((gpu) => ({ value: String(gpu.index), label: `GPU ${gpu.index}` })),
          ]}
        />
      </div>

      {/* Category tabs */}
      <div className="flex flex-wrap gap-1.5 border-b border-border pb-px">
        {MODEL_CATEGORY_ORDER.map((cat) => {
          const count = categoryCounts.get(cat) ?? 0
          const active = categoryTab === cat
          return (
            <button
              key={cat}
              type="button"
              onClick={() => setCategoryTab(cat)}
              disabled={count === 0 && !active}
              className={`flex items-center gap-1.5 rounded-t-md border-b-2 px-3 py-2 text-sm transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${
                active
                  ? "border-primary text-foreground font-medium"
                  : "border-transparent text-muted-foreground hover:text-foreground"
              }`}
            >
              {MODEL_CATEGORY_LABELS[cat]}
              <span
                className={`rounded-full px-1.5 py-0.5 text-xs ${
                  active ? "bg-primary/15 text-primary" : "bg-muted text-muted-foreground"
                }`}
              >
                {count}
              </span>
            </button>
          )
        })}
      </div>

      {!loading && (
        <p className="text-xs text-muted-foreground" aria-live="polite">
          {visibleCount} {viewMode === "model" ? "modelo" : "despliegue"}
          {visibleCount !== 1 ? "s" : ""} en {MODEL_CATEGORY_LABELS[categoryTab].toLowerCase()}
        </p>
      )}

      {viewMode === "model" && logicalError && !loading && (
        <div className="rounded-md border border-amber-500/40 bg-amber-500/10 px-4 py-3 text-sm text-amber-200">
          No se pudo cargar la vista por modelo ({logicalError}). La vista «Por despliegue» sigue disponible.
        </div>
      )}

      {loading ? (
        <SkeletonList count={5} />
      ) : viewMode === "model" ? (
        logicalFiltered.length === 0 ? (
          <EmptyState
            title="Sin modelos con esos filtros"
            description="Prueba a ampliar los criterios de búsqueda o elige otra categoría."
          />
        ) : (
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-3">
            {logicalFiltered.map(({ model, category }) => (
              <LogicalModelCard
                key={model.modelKey}
                model={model}
                categoryLabel={MODEL_CATEGORY_LABELS[category]}
                deploymentStates={models}
                busy={busyModelId === model.modelKey}
                onConfigure={(modelId) => openDeploymentConfig(modelId)}
                onEdit={(item) => openEditor((item.deployments.find((d) => d.isDefault) ?? item.deployments[0])?.modelId ?? `${item.defaultBackend}/${item.modelKey}`)}
                onLoad={(modelId) => setConfirmAction({ modelId, action: "load" })}
                onUnload={(modelId) => setConfirmAction({ modelId, action: "unload" })}
                onSetDefault={handleSetDefaultBackend}
                onAddDeployment={handleAddDeployment}
                onRemoveDeployment={(model, deployment) => setRemoveTarget({ model, deployment })}
              />
            ))}
          </div>
        )
      ) : filtered.length === 0 ? (
        <EmptyState
          title="Sin modelos con esos filtros"
          description="Prueba a ampliar los criterios de búsqueda o elige otra categoría."
        />
      ) : (
        <Tooltip.Provider delayDuration={300}>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-3">
            {filtered.map(({ model, category }) => (
              <ModelGridCard
                key={model.modelId}
                model={model}
                category={category}
                busy={busyModelId === model.modelId}
                onLoad={(modelId) => setConfirmAction({ modelId, action: "load" })}
                onUnload={(modelId) => setConfirmAction({ modelId, action: "unload" })}
                onCancelLoad={(modelId) =>
                  void runAction(modelId, async () => {
                    await api.models.cancelLoad(modelId)
                  })
                }
                onDownsize={(modelId) =>
                  void runAction(modelId, async () => {
                    await api.models.downsize(modelId)
                  })
                }
                onTogglePin={(item) => void togglePin(item)}
                onConfigure={(item) => openEditor(item.modelId)}
                onDelete={(item) => setDeleteModel(item)}
                onCompile={(item) => setCompileModel(item)}
                onUpdate={(item) => void handleUpdateModel(item)}
                onEditProfile={openProfileModal}
                onToggleProfileEnabled={(p) => void handleToggleProfileEnabled(p)}
              />
            ))}
          </div>
        </Tooltip.Provider>
      )}

      {error && (
        <div className="rounded-md border border-red-500/40 bg-red-500/10 px-4 py-3 text-sm text-red-200">
          {error}
        </div>
      )}


      <CompileModal
        model={compileModel}
        open={Boolean(compileModel)}
        onOpenChange={(open) => !open && setCompileModel(null)}
        onLoadNow={(modelId) => void runAction(modelId, () => loadModel(modelId))}
      />

      {/* Profile create/edit modal */}
      {profileModalModel && (
        <ProfileModal
          open={profileModalOpen}
          onOpenChange={(next) => {
            setProfileModalOpen(next)
            if (!next) {
              setProfileModalModel(null)
              setProfileModalProfile(null)
            }
          }}
          model={profileModalModel}
          profile={profileModalProfile}
          onSaved={() => void refresh()}
        />
      )}

      {/* Delete model dialog with cascade warning */}
      <Dialog.Root open={Boolean(deleteModel)} onOpenChange={(next) => !next && setDeleteModel(null)}>
        <Dialog.Portal>
          <Dialog.Overlay className="fixed inset-0 z-40 bg-black/60" />
          <Dialog.Content className="fixed left-1/2 top-1/2 z-50 w-[95vw] max-w-md -translate-x-1/2 -translate-y-1/2 rounded-lg border border-border bg-card p-5">
            <Dialog.Title className="text-lg font-semibold">Eliminar modelo</Dialog.Title>
            <Dialog.Description className="mt-1 text-sm text-muted-foreground">
              Esta accion eliminara la configuracion y los archivos del modelo.
            </Dialog.Description>
            {deleteModelProfiles.length > 0 && (
              <div className="mt-3 rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-sm">
                <p className="font-medium text-amber-200">
                  Se eliminaran {deleteModelProfiles.length} perfil{deleteModelProfiles.length !== 1 ? "es" : ""}:
                </p>
                <p className="mt-1 text-xs text-amber-300/80">
                  {deleteModelProfiles.map((p) => p.profileId).join(", ")}
                </p>
              </div>
            )}
            <div className="mt-4 flex justify-end gap-2">
              <Dialog.Close className="rounded-md border border-border px-3 py-2 text-sm hover:bg-muted">
                Cancelar
              </Dialog.Close>
              <button
                type="button"
                className="rounded-md border border-red-500/40 px-3 py-2 text-sm text-red-200 hover:bg-red-500/20"
                onClick={async () => {
                  if (!deleteModel) return
                  await runAction(deleteModel.modelId, async () => {
                    await api.models.delete(deleteModel.modelId)
                  })
                  setDeleteModel(null)
                }}
              >
                Eliminar
              </button>
            </div>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>


      {/* Remove one deployment (config only) */}
      <ConfirmDialog
        open={Boolean(removeTarget)}
        onOpenChange={(open) => !open && setRemoveTarget(null)}
        title="Quitar despliegue"
        description={removeDescription}
        confirmLabel="Quitar"
        variant="danger"
        onConfirm={() => {
          if (!removeTarget) return
          handleRemoveDeployment(removeTarget.model, removeTarget.deployment)
        }}
      />

      {/* Confirm Load/Unload */}
      <ConfirmDialog
        open={Boolean(confirmAction)}
        onOpenChange={(open) => !open && setConfirmAction(null)}
        title={confirmAction?.action === "load" ? "Cargar modelo" : "Descargar modelo"}
        description={
          confirmAction?.action === "load"
            ? `Se cargara el modelo "${confirmAction.modelId}" en GPU. Esto consumira VRAM.`
            : `Se descargara el modelo "${confirmAction?.modelId}" de GPU y liberara VRAM.`
        }
        confirmLabel={confirmAction?.action === "load" ? "Cargar" : "Descargar"}
        variant={confirmAction?.action === "unload" ? "danger" : "default"}
        onConfirm={() => {
          if (!confirmAction) return
          const { modelId, action } = confirmAction
          if (action === "load") {
            void runAction(modelId, () => loadModel(modelId))
          } else {
            void runAction(modelId, () => unloadModel(modelId))
          }
        }}
      />
    </div>
  )
}
