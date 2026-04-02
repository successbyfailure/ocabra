import { useEffect, useMemo, useState } from "react"
import * as Dialog from "@radix-ui/react-dialog"
import { toast } from "sonner"
import { api } from "@/api/client"
import { CompileModal } from "@/components/models/CompileModal"
import { ModelCard } from "@/components/models/ModelCard"
import { ModelConfigModal } from "@/components/models/ModelConfigModal"
import { useWebSocket } from "@/hooks/useWebSocket"
import { getModelContextSummary } from "@/lib/modelContext"
import { useGpuStore } from "@/stores/gpuStore"
import { useModelStore } from "@/stores/modelStore"
import type { BackendExtraConfig, BackendType, LoadPolicy, ModelState, ModelStatus, ModelsStorageStats } from "@/types"

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

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-2xl font-semibold">Models</h1>
        <p className="text-muted-foreground">Gestion de modelos instalados y ciclo de vida.</p>
      </div>

      {storage && (
        <div className="rounded-lg border border-border bg-card p-3">
          <div className="mb-2 flex items-center justify-between gap-3 text-sm">
            <div>
              <p className="font-medium">Almacenamiento de modelos</p>
              <p className="text-xs text-muted-foreground">{storage.path}</p>
            </div>
            <div className="text-right text-xs text-muted-foreground">
              <p>{formatBytes(storage.freeBytes)} libres</p>
              <p>{formatBytes(storage.usedBytes)} usados de {formatBytes(storage.totalBytes)}</p>
            </div>
          </div>
          <div className="h-3 overflow-hidden rounded-full bg-muted">
            <div
              className="h-full rounded-full bg-emerald-500 transition-[width]"
              style={{
                width: `${storage.totalBytes > 0 ? Math.max(0, Math.min(100, (storage.freeBytes / storage.totalBytes) * 100)) : 0}%`,
              }}
            />
          </div>
        </div>
      )}

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

      {loading ? (
        <div className="space-y-2">
          {Array.from({ length: 5 }).map((_, idx) => (
            <div key={`skeleton-${idx}`} className="h-12 animate-pulse rounded-md bg-muted" />
          ))}
        </div>
      ) : (
        <div className="overflow-x-auto rounded-lg border border-border bg-card">
          <table className="min-w-full text-left">
            <thead className="bg-muted/40 text-xs uppercase text-muted-foreground">
              <tr>
                <th className="px-3 py-2"><button type="button" onClick={() => toggleSort("name")}>Nombre</button></th>
                <th className="px-3 py-2"><button type="button" onClick={() => toggleSort("type")}>Tipo</button></th>
                <th className="px-3 py-2"><button type="button" onClick={() => toggleSort("backend")}>Backend</button></th>
                <th className="px-3 py-2"><button type="button" onClick={() => toggleSort("policy")}>Policy</button></th>
                <th className="px-3 py-2"><button type="button" onClick={() => toggleSort("gpu")}>GPU</button></th>
                <th className="px-3 py-2"><button type="button" onClick={() => toggleSort("ctxNative")}>Ctx Nativo</button></th>
                <th className="px-3 py-2"><button type="button" onClick={() => toggleSort("ctxConfig")}>Ctx Config</button></th>
                <th className="px-3 py-2"><button type="button" onClick={() => toggleSort("io")}>Input / Output</button></th>
                <th className="px-3 py-2"><button type="button" onClick={() => toggleSort("vram")}>VRAM</button></th>
                <th className="px-3 py-2"><button type="button" onClick={() => toggleSort("size")}>Tamaño</button></th>
                <th className="px-3 py-2"><button type="button" onClick={() => toggleSort("status")}>Status</button></th>
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
                />
              ))}
            </tbody>
          </table>
          {filtered.length === 0 && (
            <p className="px-4 py-8 text-center text-sm text-muted-foreground">No hay modelos con esos filtros.</p>
          )}
        </div>
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

      <Dialog.Root open={Boolean(deleteModel)} onOpenChange={(next) => !next && setDeleteModel(null)}>
        <Dialog.Portal>
          <Dialog.Overlay className="fixed inset-0 z-40 bg-black/60" />
          <Dialog.Content className="fixed left-1/2 top-1/2 z-50 w-[95vw] max-w-md -translate-x-1/2 -translate-y-1/2 rounded-lg border border-border bg-card p-5">
            <Dialog.Title className="text-lg font-semibold">Eliminar modelo</Dialog.Title>
            <Dialog.Description className="mt-1 text-sm text-muted-foreground">
              Esta accion eliminara la configuracion y los archivos del modelo.
            </Dialog.Description>
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
