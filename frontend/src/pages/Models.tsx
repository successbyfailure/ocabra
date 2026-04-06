import { useEffect, useMemo, useState } from "react"
import * as Dialog from "@radix-ui/react-dialog"
import * as Tooltip from "@radix-ui/react-tooltip"
import { toast } from "sonner"
import { api } from "@/api/client"
import { EmptyState } from "@/components/common/EmptyState"
import { CompileModal } from "@/components/models/CompileModal"
import { ModelCard } from "@/components/models/ModelCard"
import { ModelConfigModal } from "@/components/models/ModelConfigModal"
import { ProfileModal } from "@/components/models/ProfileModal"
import { useWebSocket } from "@/hooks/useWebSocket"
import { getModelContextSummary } from "@/lib/modelContext"
import { useGpuStore } from "@/stores/gpuStore"
import { useModelStore } from "@/stores/modelStore"
import type { BackendExtraConfig, BackendType, LoadPolicy, ModelProfile, ModelState, ModelStatus, ModelsStorageStats } from "@/types"

function inferType(model: ModelState): "llm" | "image" | "audio" | "pooling" {
  if (model.capabilities.imageGeneration) return "image"
  if (model.capabilities.audioTranscription || model.capabilities.tts) return "audio"
  if (model.capabilities.pooling || model.capabilities.embeddings) return "pooling"
  return "llm"
}

export function Models() {
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [query, setQuery] = useState("")
  const [statusFilter, setStatusFilter] = useState<ModelStatus | "all">("all")
  const [typeFilter, setTypeFilter] = useState<"all" | "llm" | "image" | "audio" | "pooling">("all")
  const [backendFilter, setBackendFilter] = useState<BackendType | "all">("all")
  const [gpuFilter, setGpuFilter] = useState<string>("all")
  const [sortKey, setSortKey] = useState<
    "name" | "type" | "backend" | "policy" | "gpu" | "ctxNative" | "ctxConfig" | "io" | "vram" | "size" | "status"
  >("name")
  const [sortDir, setSortDir] = useState<"asc" | "desc">("asc")
  const [busyModelId, setBusyModelId] = useState<string | null>(null)
  const [configModel, setConfigModel] = useState<ModelState | null>(null)
  const [deleteModel, setDeleteModel] = useState<ModelState | null>(null)
  const [compileModel, setCompileModel] = useState<ModelState | null>(null)
  const [storage, setStorage] = useState<ModelsStorageStats | null>(null)

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
        await refresh()
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
  }, [setGpus, setModels])

  const filtered = useMemo(() => {
    const visible = modelList
      .map((model) => {
        const context = getModelContextSummary(model)
        return {
          model,
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
        const matchesType = typeFilter === "all" || inferType(model) === typeFilter
        const matchesBackend = backendFilter === "all" || model.backendType === backendFilter
        const matchesGpu = gpuFilter === "all" || String(modelGpu) === gpuFilter
        return matchesQuery && matchesStatus && matchesType && matchesBackend && matchesGpu
      })

    visible.sort((left, right) => {
      const byKey: Record<typeof sortKey, string | number> = {
        name: left.model.displayName.toLowerCase(),
        type: inferType(left.model),
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
        type: inferType(right.model),
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
  }, [backendFilter, gpuFilter, modelList, query, sortDir, sortKey, statusFilter, typeFilter])

  const toggleSort = (
    key: "name" | "type" | "backend" | "policy" | "gpu" | "ctxNative" | "ctxConfig" | "io" | "vram" | "size" | "status",
  ) => {
    if (sortKey === key) {
      setSortDir((current) => (current === "asc" ? "desc" : "asc"))
      return
    }
    setSortKey(key)
    setSortDir("asc")
  }

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

  const togglePin = async (model: ModelState) => {
    const nextPolicy: LoadPolicy = model.loadPolicy === "pin" ? "on_demand" : "pin"
    await runAction(model.modelId, async () => {
      await api.models.patch(model.modelId, { loadPolicy: nextPolicy })
    })
  }

  const saveConfig = async (
    modelId: string,
    patch: {
      loadPolicy: LoadPolicy
      preferredGpu: number | null
      autoReload: boolean
      schedules: ModelState["schedules"]
      extraConfig?: BackendExtraConfig
    },
  ) => {
    await runAction(modelId, async () => {
      await api.models.patch(modelId, {
        loadPolicy: patch.loadPolicy,
        preferredGpu: patch.preferredGpu,
        autoReload: patch.autoReload,
        schedules: patch.schedules,
        extraConfig: patch.extraConfig,
      })
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

  const deleteModelProfiles = deleteModel?.profiles ?? []

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-2xl font-semibold">Models</h1>
        <p className="text-muted-foreground">Gestion de modelos instalados y ciclo de vida.</p>
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

      <div className="grid gap-2 rounded-lg border border-border bg-card p-3 md:grid-cols-4">
        <input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Buscar modelo"
          className="rounded-md border border-border bg-background px-3 py-2 text-sm"
        />

        <select
          value={statusFilter}
          onChange={(event) => setStatusFilter(event.target.value as ModelStatus | "all")}
          className="rounded-md border border-border bg-background px-3 py-2 text-sm"
        >
          <option value="all">Status: todos</option>
          <option value="configured">configured</option>
          <option value="loading">loading</option>
          <option value="loaded">loaded</option>
          <option value="unloading">unloading</option>
          <option value="unloaded">unloaded</option>
          <option value="error">error</option>
        </select>

        <select
          value={typeFilter}
          onChange={(event) => setTypeFilter(event.target.value as "all" | "llm" | "image" | "audio" | "pooling")}
          className="rounded-md border border-border bg-background px-3 py-2 text-sm"
        >
          <option value="all">Tipo: todos</option>
          <option value="llm">llm</option>
          <option value="pooling">pooling</option>
          <option value="image">image</option>
          <option value="audio">audio</option>
        </select>

        <select
          value={backendFilter}
          onChange={(event) => setBackendFilter(event.target.value as BackendType | "all")}
          className="rounded-md border border-border bg-background px-3 py-2 text-sm"
        >
          <option value="all">Backend: todos</option>
          {Array.from(new Set(modelList.map((model) => model.backendType))).sort().map((backend) => (
            <option key={backend} value={backend}>
              {backend}
            </option>
          ))}
        </select>

        <select
          value={gpuFilter}
          onChange={(event) => setGpuFilter(event.target.value)}
          className="rounded-md border border-border bg-background px-3 py-2 text-sm"
        >
          <option value="all">GPU: todas</option>
          {gpus.map((gpu) => (
            <option key={gpu.index} value={gpu.index}>
              GPU {gpu.index}
            </option>
          ))}
        </select>
      </div>

      {!loading && (
        <p className="text-xs text-muted-foreground" aria-live="polite">
          {filtered.length === modelList.length
            ? `${modelList.length} modelo${modelList.length !== 1 ? "s" : ""}`
            : `${filtered.length} de ${modelList.length} modelo${modelList.length !== 1 ? "s" : ""} con los filtros aplicados`}
        </p>
      )}

      {loading ? (
        <div className="space-y-2" role="status" aria-label="Cargando lista de modelos">
          {Array.from({ length: 5 }).map((_, idx) => (
            <div key={`skeleton-${idx}`} className="h-12 animate-pulse rounded-md bg-muted" />
          ))}
        </div>
      ) : (
        <Tooltip.Provider delayDuration={300}>
          <div className="overflow-x-auto rounded-lg border border-border bg-card">
            <table className="min-w-full text-left">
              <thead className="bg-muted/40 text-xs uppercase text-muted-foreground">
                <tr>
                  <th className="px-3 py-2" aria-sort={sortKey === "name" ? (sortDir === "asc" ? "ascending" : "descending") : "none"}>
                    <button type="button" onClick={() => toggleSort("name")} className="flex items-center gap-1 hover:text-foreground">
                      Nombre {sortKey === "name" && (sortDir === "asc" ? "↑" : "↓")}
                    </button>
                  </th>
                  <th className="px-3 py-2" aria-sort={sortKey === "type" ? (sortDir === "asc" ? "ascending" : "descending") : "none"}>
                    <button type="button" onClick={() => toggleSort("type")} className="hover:text-foreground">
                      Tipo {sortKey === "type" && (sortDir === "asc" ? "↑" : "↓")}
                    </button>
                  </th>
                  <th className="px-3 py-2">
                    <button type="button" onClick={() => toggleSort("backend")} className="hover:text-foreground">
                      Backend {sortKey === "backend" && (sortDir === "asc" ? "↑" : "↓")}
                    </button>
                  </th>
                  <th className="px-3 py-2">
                    <Tooltip.Root>
                      <Tooltip.Trigger asChild>
                        <button type="button" onClick={() => toggleSort("policy")} className="hover:text-foreground underline decoration-dashed underline-offset-2">
                          Policy {sortKey === "policy" && (sortDir === "asc" ? "↑" : "↓")}
                        </button>
                      </Tooltip.Trigger>
                      <Tooltip.Portal>
                        <Tooltip.Content className="z-50 rounded-md border border-border bg-popover px-3 py-1.5 text-xs shadow-md" sideOffset={4}>
                          <strong>pin</strong>: siempre cargado · <strong>warm</strong>: cargado preventivamente · <strong>on_demand</strong>: carga al primer uso
                          <Tooltip.Arrow className="fill-border" />
                        </Tooltip.Content>
                      </Tooltip.Portal>
                    </Tooltip.Root>
                  </th>
                  <th className="px-3 py-2">
                    <button type="button" onClick={() => toggleSort("gpu")} className="hover:text-foreground">
                      GPU {sortKey === "gpu" && (sortDir === "asc" ? "↑" : "↓")}
                    </button>
                  </th>
                  <th className="px-3 py-2">
                    <Tooltip.Root>
                      <Tooltip.Trigger asChild>
                        <button type="button" onClick={() => toggleSort("ctxNative")} className="hover:text-foreground underline decoration-dashed underline-offset-2">
                          Ctx Nativo {sortKey === "ctxNative" && (sortDir === "asc" ? "↑" : "↓")}
                        </button>
                      </Tooltip.Trigger>
                      <Tooltip.Portal>
                        <Tooltip.Content className="z-50 rounded-md border border-border bg-popover px-3 py-1.5 text-xs shadow-md max-w-xs" sideOffset={4}>
                          Longitud de contexto máxima soportada por el modelo base (desde los pesos/arquitectura).
                          <Tooltip.Arrow className="fill-border" />
                        </Tooltip.Content>
                      </Tooltip.Portal>
                    </Tooltip.Root>
                  </th>
                  <th className="px-3 py-2">
                    <Tooltip.Root>
                      <Tooltip.Trigger asChild>
                        <button type="button" onClick={() => toggleSort("ctxConfig")} className="hover:text-foreground underline decoration-dashed underline-offset-2">
                          Ctx Config {sortKey === "ctxConfig" && (sortDir === "asc" ? "↑" : "↓")}
                        </button>
                      </Tooltip.Trigger>
                      <Tooltip.Portal>
                        <Tooltip.Content className="z-50 rounded-md border border-border bg-popover px-3 py-1.5 text-xs shadow-md max-w-xs" sideOffset={4}>
                          Longitud de contexto configurada para este deployment (puede ser menor que el nativo para ahorrar VRAM).
                          <Tooltip.Arrow className="fill-border" />
                        </Tooltip.Content>
                      </Tooltip.Portal>
                    </Tooltip.Root>
                  </th>
                  <th className="px-3 py-2">
                    <button type="button" onClick={() => toggleSort("io")} className="hover:text-foreground">
                      Input / Output {sortKey === "io" && (sortDir === "asc" ? "↑" : "↓")}
                    </button>
                  </th>
                  <th className="px-3 py-2">
                    <button type="button" onClick={() => toggleSort("vram")} className="hover:text-foreground">
                      VRAM {sortKey === "vram" && (sortDir === "asc" ? "↑" : "↓")}
                    </button>
                  </th>
                  <th className="px-3 py-2">
                    <button type="button" onClick={() => toggleSort("size")} className="hover:text-foreground">
                      Tamaño {sortKey === "size" && (sortDir === "asc" ? "↑" : "↓")}
                    </button>
                  </th>
                  <th className="px-3 py-2">
                    <button type="button" onClick={() => toggleSort("status")} className="hover:text-foreground">
                      Status {sortKey === "status" && (sortDir === "asc" ? "↑" : "↓")}
                    </button>
                  </th>
                  <th className="px-3 py-2 text-center">Perfiles</th>
                  <th className="px-3 py-2">Acciones</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map(({ model }) => (
                  <ModelCard
                    key={model.modelId}
                    model={model}
                    busy={busyModelId === model.modelId}
                    onLoad={(modelId) => void runAction(modelId, () => loadModel(modelId))}
                    onUnload={(modelId) => void runAction(modelId, () => unloadModel(modelId))}
                    onTogglePin={(item) => void togglePin(item)}
                    onConfigure={(item) => setConfigModel(item)}
                    onDelete={(item) => setDeleteModel(item)}
                    onCompile={(item) => setCompileModel(item)}
                    onEditProfile={openProfileModal}
                    onToggleProfileEnabled={(p) => void handleToggleProfileEnabled(p)}
                  />
                ))}
              </tbody>
            </table>
            {filtered.length === 0 && (
              <EmptyState
                title="Sin modelos con esos filtros"
                description="Prueba a ampliar los criterios de búsqueda."
                className="rounded-none border-0 border-t"
              />
            )}
          </div>
        </Tooltip.Provider>
      )}

      {error && (
        <div className="rounded-md border border-red-500/40 bg-red-500/10 px-4 py-3 text-sm text-red-200">
          {error}
        </div>
      )}

      <ModelConfigModal
        model={configModel}
        gpus={gpus}
        open={Boolean(configModel)}
        onOpenChange={(open) => !open && setConfigModel(null)}
        onSave={saveConfig}
      />

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
    </div>
  )
}
