import { useEffect, useState } from "react"
import { toast } from "sonner"
import { api } from "@/api/client"
import { GeneralSettings } from "@/components/settings/GeneralSettings"
import { GlobalSchedules } from "@/components/settings/GlobalSchedules"
import { GPUSettings } from "@/components/settings/GPUSettings"
import { LiteLLMSettings } from "@/components/settings/LiteLLMSettings"
import { StorageSettings } from "@/components/settings/StorageSettings"
import type { GPUState, LocalModel, ServerConfig } from "@/types"

const EMPTY_CONFIG: ServerConfig = {
  defaultGpuIndex: 0,
  idleTimeoutSeconds: 300,
  vramBufferMb: 512,
  vramPressureThresholdPct: 85,
  logLevel: "info",
  litellmBaseUrl: "",
  litellmAdminKey: "",
  litellmAutoSync: false,
  energyCostEurKwh: 0,
  modelsDir: "/models",
  downloadDir: "/models/downloads",
  maxTemperatureC: 88,
  globalSchedules: [],
}

export function Settings() {
  const [loading, setLoading] = useState(true)
  const [config, setConfig] = useState<ServerConfig>(EMPTY_CONFIG)
  const [gpus, setGpus] = useState<GPUState[]>([])
  const [localModels, setLocalModels] = useState<LocalModel[]>([])

  useEffect(() => {
    let active = true

    const load = async () => {
      try {
        const [configData, gpusData, localData] = await Promise.all([
          api.config.get(),
          api.gpus.list(),
          api.registry.listLocal(),
        ])
        if (!active) return
        setConfig({
          ...configData,
          modelsDir: configData.modelsDir ?? localStorage.getItem("ocabra.modelsDir") ?? "/models",
          downloadDir:
            configData.downloadDir ?? localStorage.getItem("ocabra.downloadDir") ?? "/models/downloads",
          maxTemperatureC: configData.maxTemperatureC ?? Number(localStorage.getItem("ocabra.maxTemperatureC") ?? "88"),
        })
        setGpus(gpusData)
        setLocalModels(localData)
      } catch (err) {
        if (active) toast.error(err instanceof Error ? err.message : "No se pudo cargar settings")
      } finally {
        if (active) setLoading(false)
      }
    }

    void load()
    const timer = window.setInterval(() => {
      void load()
    }, 30_000)

    return () => {
      active = false
      window.clearInterval(timer)
    }
  }, [])

  const savePatch = async (patch: Partial<ServerConfig>) => {
    try {
      const next = await api.config.patch(patch)
      setConfig((prev) => ({ ...prev, ...next, ...patch }))
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "No se pudo guardar")
    }
  }

  return (
    <div className="space-y-4 pb-6">
      <div>
        <h1 className="text-2xl font-semibold">Settings</h1>
        <p className="text-muted-foreground">Configuracion general, GPU, LiteLLM, storage y schedules.</p>
      </div>

      {loading ? (
        <div className="space-y-3">
          {Array.from({ length: 5 }).map((_, idx) => (
            <div key={`settings-skeleton-${idx}`} className="h-40 animate-pulse rounded-lg bg-muted" />
          ))}
        </div>
      ) : (
        <div className="grid gap-4">
          <GeneralSettings config={config} onSave={savePatch} />
          <GPUSettings gpus={gpus} config={config} onSave={savePatch} />
          <LiteLLMSettings config={config} onSave={savePatch} />
          <StorageSettings localModels={localModels} config={config} />
          <GlobalSchedules config={config} onSave={savePatch} />
        </div>
      )}
    </div>
  )
}
