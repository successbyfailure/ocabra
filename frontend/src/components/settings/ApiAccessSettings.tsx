import { useEffect, useState } from "react"
import { toast } from "sonner"
import { api } from "@/api/client"
import type { Group, ServerConfig } from "@/types"

interface ApiAccessSettingsProps {
  config: ServerConfig
  onSave: (patch: Partial<ServerConfig>) => Promise<void>
}

export function ApiAccessSettings({ config, onSave }: ApiAccessSettingsProps) {
  const [requireOpenai, setRequireOpenai] = useState(config.requireApiKeyOpenai)
  const [requireOllama, setRequireOllama] = useState(config.requireApiKeyOllama)
  const [newUserGroups, setNewUserGroups] = useState<string[]>(config.newUserGroupNames)
  const [groups, setGroups] = useState<Group[]>([])

  useEffect(() => {
    setRequireOpenai(config.requireApiKeyOpenai)
    setRequireOllama(config.requireApiKeyOllama)
    setNewUserGroups(config.newUserGroupNames)
  }, [config.requireApiKeyOpenai, config.requireApiKeyOllama, config.newUserGroupNames])

  useEffect(() => {
    const listGroups = api.groups?.list
    if (!listGroups) return
    void listGroups().then(setGroups).catch(() => setGroups([]))
  }, [])

  const save = async () => {
    try {
      await onSave({
        requireApiKeyOpenai: requireOpenai,
        requireApiKeyOllama: requireOllama,
        newUserGroupNames: newUserGroups,
      })
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

      <div>
        <p className="mb-2 text-sm font-medium text-foreground">Grupos para usuarios nuevos</p>
        <div className="grid gap-2 sm:grid-cols-2">
          {groups.map((group) => (
            <label key={group.id} className="flex cursor-pointer items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={newUserGroups.includes(group.name)}
                onChange={(event) => {
                  setNewUserGroups((current) =>
                    event.target.checked
                      ? [...current, group.name]
                      : current.filter((name) => name !== group.name),
                  )
                }}
                className="h-4 w-4 accent-primary"
              />
              <span>{group.name}</span>
              {group.isDefault && <span className="text-xs text-muted-foreground">(default)</span>}
            </label>
          ))}
        </div>
        <p className="mt-2 text-xs text-muted-foreground">
          Todas las API keys heredan los grupos de su propietario.
        </p>
      </div>

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
