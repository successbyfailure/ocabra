import { useEffect, useState } from "react"
import { toast } from "sonner"
import type { ServerConfig } from "@/types"

interface GeneralSettingsProps {
  config: ServerConfig
  onSave: (patch: Partial<ServerConfig>) => Promise<void>
}

export function GeneralSettings({ config, onSave }: GeneralSettingsProps) {
  const [modelsDir, setModelsDir] = useState(config.modelsDir)
  const [logLevel, setLogLevel] = useState(config.logLevel)
  const [idleTimeoutSeconds, setIdleTimeoutSeconds] = useState(config.idleTimeoutSeconds)
  const [vramBufferMb, setVramBufferMb] = useState(config.vramBufferMb)

  useEffect(() => {
    setModelsDir(config.modelsDir)
    setLogLevel(config.logLevel)
    setIdleTimeoutSeconds(config.idleTimeoutSeconds)
    setVramBufferMb(config.vramBufferMb)
  }, [config.idleTimeoutSeconds, config.logLevel, config.modelsDir, config.vramBufferMb])

  const save = async () => {
    try {
      await onSave({
        logLevel,
        idleTimeoutSeconds,
        vramBufferMb,
        modelsDir,
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
        MODELS_DIR
        <input
          value={modelsDir}
          onChange={(event) => setModelsDir(event.target.value)}
          className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2"
        />
      </label>

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
          VRAM buffer (MB)
          <input
            type="number"
            min={0}
            value={vramBufferMb}
            onChange={(event) => setVramBufferMb(Number(event.target.value))}
            className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2"
          />
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
