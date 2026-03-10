import { useEffect, useMemo, useState } from "react"
import { api } from "@/api/client"
import { LoadPolicyBadge } from "@/components/models/LoadPolicyBadge"
import { ModelStatusBadge } from "@/components/models/ModelStatusBadge"
import { useModelStore } from "@/stores/modelStore"

export function Models() {
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const models = useModelStore((state) => state.models)
  const setModels = useModelStore((state) => state.setModels)
  const loadModel = useModelStore((state) => state.loadModel)
  const unloadModel = useModelStore((state) => state.unloadModel)

  const modelList = useMemo(() => Object.values(models), [models])

  useEffect(() => {
    let active = true
    async function bootstrap() {
      try {
        const initial = await api.models.list()
        if (active) {
          setModels(initial)
          setError(null)
        }
      } catch (err) {
        if (active) {
          setError(err instanceof Error ? err.message : "Failed to load models")
        }
      } finally {
        if (active) {
          setLoading(false)
        }
      }
    }

    void bootstrap()
    return () => {
      active = false
    }
  }, [setModels])

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold mb-6">Models</h1>
      <p className="text-muted-foreground">Installed models and basic lifecycle actions.</p>

      <div className="space-y-3">
        {loading && (
          <div className="rounded-lg border border-border bg-card px-4 py-3 text-muted-foreground">
            Loading models...
          </div>
        )}

        {!loading &&
          modelList.map((model) => (
            <div
              key={model.modelId}
              className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-border bg-card px-4 py-3"
            >
              <div className="space-y-1">
                <p className="font-medium">{model.displayName}</p>
                <div className="flex flex-wrap gap-2">
                  <ModelStatusBadge status={model.status} />
                  <LoadPolicyBadge policy={model.loadPolicy} />
                </div>
              </div>
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={() => void loadModel(model.modelId)}
                  className="rounded-md border border-emerald-500/40 px-3 py-1 text-sm text-emerald-200 hover:bg-emerald-500/20"
                  disabled={model.status === "loaded" || model.status === "loading"}
                >
                  Load
                </button>
                <button
                  type="button"
                  onClick={() => void unloadModel(model.modelId)}
                  className="rounded-md border border-red-500/40 px-3 py-1 text-sm text-red-200 hover:bg-red-500/20"
                  disabled={model.status === "unloaded" || model.status === "unloading"}
                >
                  Unload
                </button>
              </div>
            </div>
          ))}

        {!loading && modelList.length === 0 && (
          <div className="rounded-lg border border-dashed border-border px-4 py-8 text-center text-muted-foreground">
            No models configured.
          </div>
        )}

        {error && (
          <div className="rounded-md border border-red-500/40 bg-red-500/10 px-4 py-3 text-sm text-red-200">
            {error}
          </div>
        )}
      </div>
    </div>
  )
}
