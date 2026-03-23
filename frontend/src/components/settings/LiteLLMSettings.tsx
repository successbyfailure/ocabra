import { useEffect, useState } from "react"
import { toast } from "sonner"
import { api } from "@/api/client"
import type { ServerConfig } from "@/types"

interface LiteLLMSettingsProps {
  config: ServerConfig
  onSave: (patch: Partial<ServerConfig>) => Promise<void>
}

export function LiteLLMSettings({ config, onSave }: LiteLLMSettingsProps) {
  const [litellmBaseUrl, setLiteLLMBaseUrl] = useState(config.litellmBaseUrl)
  const [litellmAdminKey, setLiteLLMAdminKey] = useState(config.litellmAdminKey)
  const [litellmAutoSync, setLiteLLMAutoSync] = useState(config.litellmAutoSync)
  const [lastSync, setLastSync] = useState<{ at: string; ok: boolean; count: number } | null>(null)
  const [syncing, setSyncing] = useState(false)

  useEffect(() => {
    setLiteLLMBaseUrl(config.litellmBaseUrl)
    setLiteLLMAdminKey(config.litellmAdminKey)
    setLiteLLMAutoSync(config.litellmAutoSync)
  }, [config.litellmAdminKey, config.litellmAutoSync, config.litellmBaseUrl])

  const save = async () => {
    try {
      await onSave({ litellmBaseUrl, litellmAdminKey, litellmAutoSync })
      toast.success("LiteLLM settings guardadas")
    } catch {
      // page-level toast is shown in Settings
    }
  }

  const syncNow = async () => {
    setSyncing(true)
    try {
      const result = await api.config.syncLiteLLM()
      setLastSync({ at: new Date().toISOString(), ok: true, count: result.syncedModels })
      toast.success(`${result.syncedModels} modelos sincronizados`)
    } catch (err) {
      setLastSync({ at: new Date().toISOString(), ok: false, count: 0 })
      toast.error(err instanceof Error ? err.message : "Sync fallida")
    } finally {
      setSyncing(false)
    }
  }

  return (
    <section className="space-y-3 rounded-lg border border-border bg-card p-4">
      <h2 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">LiteLLM Sync</h2>

      <label className="block text-sm text-muted-foreground">
        URL proxy LiteLLM
        <input
          value={litellmBaseUrl}
          onChange={(event) => setLiteLLMBaseUrl(event.target.value)}
          className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2"
        />
      </label>

      <label className="block text-sm text-muted-foreground">
        API key admin
        <input
          type="password"
          value={litellmAdminKey}
          onChange={(event) => setLiteLLMAdminKey(event.target.value)}
          className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2"
        />
      </label>

      <label className="inline-flex items-center gap-2 text-sm text-muted-foreground">
        <input
          type="checkbox"
          checked={litellmAutoSync}
          onChange={(event) => setLiteLLMAutoSync(event.target.checked)}
        />
        Sync automatico al anadir/eliminar modelos
      </label>

      <div className="flex flex-wrap items-center gap-2">
        <button
          type="button"
          onClick={() => void save()}
          className="rounded-md bg-primary px-3 py-2 text-sm font-medium text-primary-foreground"
        >
          Guardar LiteLLM
        </button>
        <button
          type="button"
          onClick={() => void syncNow()}
          disabled={syncing}
          className="rounded-md border border-border px-3 py-2 text-sm hover:bg-muted disabled:opacity-60"
        >
          {syncing ? "Sincronizando..." : "Sync manual"}
        </button>
      </div>

      {lastSync && (
        <p className={`text-xs ${lastSync.ok ? "text-emerald-300" : "text-red-300"}`}>
          Ultima sync: {new Date(lastSync.at).toLocaleString()} - {lastSync.ok ? `ok (${lastSync.count})` : "error"}
        </p>
      )}
    </section>
  )
}
