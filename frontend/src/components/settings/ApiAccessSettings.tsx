import { useEffect, useState } from "react"
import { toast } from "sonner"
import type { ServerConfig } from "@/types"

interface ApiAccessSettingsProps {
  config: ServerConfig
  onSave: (patch: Partial<ServerConfig>) => Promise<void>
}

export function ApiAccessSettings({ config, onSave }: ApiAccessSettingsProps) {
  const [requireOpenai, setRequireOpenai] = useState(config.requireApiKeyOpenai)
  const [requireOllama, setRequireOllama] = useState(config.requireApiKeyOllama)

  useEffect(() => {
    setRequireOpenai(config.requireApiKeyOpenai)
    setRequireOllama(config.requireApiKeyOllama)
  }, [config.requireApiKeyOpenai, config.requireApiKeyOllama])

  const save = async () => {
    try {
      await onSave({ requireApiKeyOpenai: requireOpenai, requireApiKeyOllama: requireOllama })
      toast.success("Configuración de acceso guardada")
    } catch {
      // error shown by parent
    }
  }

  return (
    <section className="space-y-4 rounded-lg border border-border bg-card p-4">
      <div>
        <h2 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">
          Acceso a APIs externas
        </h2>
        <p className="mt-1 text-xs text-muted-foreground">
          Controla si los endpoints OpenAI (<code>/v1/…</code>) y Ollama (<code>/api/…</code>) requieren
          autenticación. La misma API key (<code>sk-ocabra-…</code>) funciona para ambos.
        </p>
      </div>

      <div className="space-y-3">
        <label className="flex items-center gap-3 cursor-pointer">
          <input
            type="checkbox"
            className="h-4 w-4 rounded border-border accent-primary"
            checked={requireOpenai}
            onChange={(e) => setRequireOpenai(e.target.checked)}
          />
          <span className="text-sm text-foreground">
            Requerir API key en endpoint OpenAI (<code className="text-xs">/v1/</code>)
          </span>
        </label>

        <label className="flex items-center gap-3 cursor-pointer">
          <input
            type="checkbox"
            className="h-4 w-4 rounded border-border accent-primary"
            checked={requireOllama}
            onChange={(e) => setRequireOllama(e.target.checked)}
          />
          <span className="text-sm text-foreground">
            Requerir API key en endpoint Ollama (<code className="text-xs">/api/</code>)
          </span>
        </label>
      </div>

      {(!requireOpenai || !requireOllama) && (
        <p className="text-xs text-amber-500">
          Los endpoints sin API key permiten acceso anónimo (solo grupo <em>default</em>).
        </p>
      )}

      <button
        type="button"
        onClick={save}
        className="rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground hover:bg-primary/90"
      >
        Guardar
      </button>
    </section>
  )
}
