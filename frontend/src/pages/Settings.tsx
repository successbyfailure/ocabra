import { useEffect, useState } from "react"
import * as Tabs from "@radix-ui/react-tabs"
import {
  Settings as SettingsIcon,
  Cpu,
  Server,
  HardDrive,
  Link,
  Network,
  Sparkles,
} from "lucide-react"
import { toast } from "sonner"
import { SkeletonList } from "@/components/common/Skeleton"
import { api } from "@/api/client"
import { ApiAccessSettings } from "@/components/settings/ApiAccessSettings"
import { BackendRuntimeSettings } from "@/components/settings/BackendRuntimeSettings"
import { FederationSettings } from "@/components/settings/FederationSettings"
import { GenerationServicesSettings } from "@/components/settings/GenerationServicesSettings"
import { GeneralSettings } from "@/components/settings/GeneralSettings"
import { GlobalSchedules } from "@/components/settings/GlobalSchedules"
import { GPUSettings } from "@/components/settings/GPUSettings"
import { LiteLLMSettings } from "@/components/settings/LiteLLMSettings"
import { StorageSettings } from "@/components/settings/StorageSettings"
import type { GPUState, LocalModel, ServerConfig } from "@/types"

const EMPTY_CONFIG: ServerConfig = {
  defaultGpuIndex: 0,
  idleTimeoutSeconds: 300,
  idleEvictionCheckIntervalSeconds: 15,
  modelLoadWaitTimeoutSeconds: 720,
  pressureEvictionDrainTimeoutSeconds: 60,
  vramBufferMb: 512,
  vramPressureThresholdPct: 85,
  openaiAudioMaxPartSizeMb: 256,
  whisperStartupTimeoutSeconds: 300,
  logLevel: "info",
  litellmBaseUrl: "",
  litellmAdminKey: "",
  litellmAutoSync: false,
  energyCostEurKwh: 0,
  modelsDir: "/models",
  downloadDir: "/models/downloads",
  maxTemperatureC: 88,
  vllmGpuMemoryUtilization: 0.85,
  vllmMaxNumSeqs: 16,
  vllmMaxNumBatchedTokens: 8192,
  vllmEnablePrefixCaching: true,
  vllmEnforceEager: true,
  sglangMemFractionStatic: 0.9,
  sglangContextLength: null,
  sglangDisableRadixCache: false,
  llamaCppGpuLayers: 0,
  llamaCppCtxSize: 4096,
  llamaCppFlashAttn: false,
  bitnetGpuLayers: 0,
  bitnetCtxSize: 4096,
  bitnetFlashAttn: false,
  diffusersTorchDtype: "auto",
  diffusersOffloadMode: "none",
  diffusersEnableTorchCompile: false,
  diffusersEnableXformers: false,
  diffusersAllowTf32: true,
  tensorrtLlmEnabled: false,
  tensorrtLlmMaxBatchSize: null,
  tensorrtLlmContextLength: null,
  globalSchedules: [],
  requireApiKeyOpenai: true,
  requireApiKeyOllama: true,
  realtimeDefaultSttModel: "",
  realtimeDefaultTtsModel: "",
  federationEnabled: false,
  federationNodeName: "",
  hunyuanIdleUnloadSeconds: 300,
  hunyuanGenerationGracePeriodS: 120,
  hunyuanPreferredGpu: 1,
  comfyuiIdleUnloadSeconds: 600,
  comfyuiGenerationGracePeriodS: 120,
  comfyuiPreferredGpu: 1,
  a1111IdleUnloadSeconds: 600,
  a1111GenerationGracePeriodS: 120,
  a1111PreferredGpu: 1,
  acestepIdleUnloadSeconds: 1800,
  acestepGenerationGracePeriodS: 180,
  acestepPreferredGpu: 1,
  unslothIdleUnloadSeconds: 0,
  unslothGenerationGracePeriodS: -1,
  unslothPreferredGpu: 1,
  generationGpuUtilThresholdPct: 50,
}

const TAB_TRIGGER =
  "inline-flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-t-md transition-colors text-muted-foreground hover:text-foreground data-[state=active]:text-foreground data-[state=active]:border-b-2 data-[state=active]:border-primary data-[state=active]:bg-transparent"

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
        setConfig(configData)
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
      throw err
    }
  }

  return (
    <div className="space-y-4 pb-6">
      <div>
        <h1 className="text-2xl font-semibold">Settings</h1>
        <p className="text-muted-foreground">Configuracion general, GPU, LiteLLM, storage y schedules.</p>
      </div>

      {loading ? (
        <SkeletonList count={5} className="h-40" />
      ) : (
        <Tabs.Root defaultValue="general">
          <Tabs.List className="flex gap-1 border-b border-border mb-4 overflow-x-auto flex-nowrap whitespace-nowrap">
            <Tabs.Trigger value="general" className={TAB_TRIGGER}>
              <SettingsIcon size={14} />
              General
            </Tabs.Trigger>
            <Tabs.Trigger value="gpus" className={TAB_TRIGGER}>
              <Cpu size={14} />
              GPUs
            </Tabs.Trigger>
            <Tabs.Trigger value="backends" className={TAB_TRIGGER}>
              <Server size={14} />
              Backends
            </Tabs.Trigger>
            <Tabs.Trigger value="storage" className={TAB_TRIGGER}>
              <HardDrive size={14} />
              Almacenamiento
            </Tabs.Trigger>
            <Tabs.Trigger value="litellm" className={TAB_TRIGGER}>
              <Link size={14} />
              LiteLLM
            </Tabs.Trigger>
            <Tabs.Trigger value="generation" className={TAB_TRIGGER}>
              <Sparkles size={14} />
              Generación
            </Tabs.Trigger>
            <Tabs.Trigger value="federation" className={TAB_TRIGGER}>
              <Network size={14} />
              Federation
            </Tabs.Trigger>
          </Tabs.List>

          <Tabs.Content value="general">
            <div className="grid gap-4">
              <GeneralSettings config={config} onSave={savePatch} />
              <ApiAccessSettings config={config} onSave={savePatch} />
            </div>
          </Tabs.Content>

          <Tabs.Content value="gpus">
            <div className="grid gap-4">
              <GPUSettings gpus={gpus} config={config} onSave={savePatch} />
            </div>
          </Tabs.Content>

          <Tabs.Content value="backends">
            <div className="grid gap-4">
              <BackendRuntimeSettings config={config} onSave={savePatch} />
            </div>
          </Tabs.Content>

          <Tabs.Content value="storage">
            <div className="grid gap-4">
              <StorageSettings localModels={localModels} config={config} onSave={savePatch} />
              <GlobalSchedules config={config} onSave={savePatch} />
            </div>
          </Tabs.Content>

          <Tabs.Content value="litellm">
            <div className="grid gap-4">
              <LiteLLMSettings config={config} onSave={savePatch} />
            </div>
          </Tabs.Content>

          <Tabs.Content value="generation">
            <div className="grid gap-4">
              <GenerationServicesSettings config={config} gpus={gpus} onSave={savePatch} />
            </div>
          </Tabs.Content>

          <Tabs.Content value="federation">
            <div className="grid gap-4">
              <FederationSettings config={config} onSave={savePatch} />
            </div>
          </Tabs.Content>
        </Tabs.Root>
      )}
    </div>
  )
}
