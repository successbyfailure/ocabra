import { useEffect, useState } from "react"
import { toast } from "sonner"
import { api } from "@/api/client"
import type { ServerConfig } from "@/types"

interface GeneralSettingsProps {
  config: ServerConfig
  onSave: (patch: Partial<ServerConfig>) => Promise<void>
}

interface ModelOption {
  modelId: string
  displayName: string
}

const TTS_BACKENDS = new Set(["tts", "chatterbox", "voxtral"])
const STT_BACKENDS = new Set(["whisper"])

export function GeneralSettings({ config, onSave }: GeneralSettingsProps) {
  const [logLevel, setLogLevel] = useState(config.logLevel)
  const [energyCostEurKwh, setEnergyCostEurKwh] = useState(config.energyCostEurKwh)
  const [idleTimeoutSeconds, setIdleTimeoutSeconds] = useState(config.idleTimeoutSeconds)
  const [realtimeDefaultSttModel, setRealtimeDefaultSttModel] = useState(config.realtimeDefaultSttModel ?? "")
  const [realtimeDefaultTtsModel, setRealtimeDefaultTtsModel] = useState(config.realtimeDefaultTtsModel ?? "")

  const [sttModels, setSttModels] = useState<ModelOption[]>([])
  const [ttsModels, setTtsModels] = useState<ModelOption[]>([])

  useEffect(() => {
    setLogLevel(config.logLevel)
    setEnergyCostEurKwh(config.energyCostEurKwh)
    setIdleTimeoutSeconds(config.idleTimeoutSeconds)
    setRealtimeDefaultSttModel(config.realtimeDefaultSttModel ?? "")
    setRealtimeDefaultTtsModel(config.realtimeDefaultTtsModel ?? "")
  }, [config.energyCostEurKwh, config.idleTimeoutSeconds, config.logLevel, config.realtimeDefaultSttModel, config.realtimeDefaultTtsModel])

  useEffect(() => {
    api.models.list().then((models) => {
      setSttModels(
        models
          .filter((m) => STT_BACKENDS.has(m.backendType))
          .map((m) => ({ modelId: m.modelId, displayName: m.displayName || m.modelId }))
      )
      setTtsModels(
        models
          .filter((m) => TTS_BACKENDS.has(m.backendType))
          .map((m) => ({ modelId: m.modelId, displayName: m.displayName || m.modelId }))
      )
    }).catch(() => {})
  }, [])

  const save = async () => {
    try {
      await onSave({
        logLevel,
        idleTimeoutSeconds,
        energyCostEurKwh,
        realtimeDefaultSttModel,
        realtimeDefaultTtsModel,
      })
      toast.success("General settings guardadas")
    } catch {
      // page-level toast is shown in Settings
    }
  }

  return (
    <section className="space-y-3 rounded-lg border border-border bg-card p-4">
      <h2 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">General</h2>
      <label className="block text-sm text-muted-foreground">
        Log level
        <select
          value={logLevel}
          onChange={(event) => setLogLevel(event.target.value)}
          className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2"
        >
          <option value="debug">debug</option>
          <option value="info">info</option>
          <option value="warning">warning</option>
          <option value="error">error</option>
        </select>
      </label>

      <div className="grid gap-2 md:grid-cols-2">
        <label className="text-sm text-muted-foreground">
          Idle timeout (s)
          <input
            type="number"
            min={0}
            value={idleTimeoutSeconds}
            onChange={(event) => setIdleTimeoutSeconds(Number(event.target.value))}
            className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2"
          />
        </label>

        <label className="text-sm text-muted-foreground">
          Coste energia (EUR/kWh)
          <input
            type="number"
            min={0}
            step="0.01"
            value={energyCostEurKwh}
            onChange={(event) => setEnergyCostEurKwh(Number(event.target.value))}
            className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2"
          />
        </label>
      </div>

      <h2 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground pt-2">Realtime API</h2>
      <p className="text-xs text-muted-foreground -mt-1">
        Modelos STT y TTS por defecto para sesiones /v1/realtime (WebSocket de audio bidireccional).
      </p>
      <div className="grid gap-2 md:grid-cols-2">
        <label className="text-sm text-muted-foreground">
          Modelo STT (Whisper)
          <select
            value={realtimeDefaultSttModel}
            onChange={(e) => setRealtimeDefaultSttModel(e.target.value)}
            className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
          >
            <option value="">— Sin configurar —</option>
            {sttModels.map((m) => (
              <option key={m.modelId} value={m.modelId}>
                {m.displayName}
              </option>
            ))}
          </select>
        </label>
        <label className="text-sm text-muted-foreground">
          Modelo TTS
          <select
            value={realtimeDefaultTtsModel}
            onChange={(e) => setRealtimeDefaultTtsModel(e.target.value)}
            className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
          >
            <option value="">— Sin configurar —</option>
            {ttsModels.map((m) => (
              <option key={m.modelId} value={m.modelId}>
                {m.displayName}
              </option>
            ))}
          </select>
        </label>
      </div>

      <button
        type="button"
        onClick={() => void save()}
        className="rounded-md bg-primary px-3 py-2 text-sm font-medium text-primary-foreground"
      >
        Guardar General
      </button>
    </section>
  )
}
