import { useEffect, useState } from "react"
import { toast } from "sonner"
import { modelEditorApi } from "@/api/modelEditor"
import { ConfirmDialog } from "@/components/common/ConfirmDialog"
import { fmtContext } from "@/components/dashboard/format"
import type { ServerConfig } from "@/types"

type Status = Awaited<ReturnType<typeof modelEditorApi.ollamaSettings>>

const CONTEXTS = [16384, 32768, 65536, 131072, 262144]

/**
 * Ollama es un único servidor para todos sus modelos: su paralelismo y su
 * contexto máximo son globales. Guardar los deja configurados; aplicarlos
 * recrea el contenedor de Ollama, que descarga todos sus modelos.
 */
export function OllamaServerSettings({
  config,
  onSave,
}: {
  config: ServerConfig
  onSave: (patch: Partial<ServerConfig>) => Promise<void>
}) {
  const [parallel, setParallel] = useState(config.ollamaNumParallel ?? 2)
  const [maxContext, setMaxContext] = useState(config.ollamaMaxContext ?? 65536)
  const [status, setStatus] = useState<Status | null>(null)
  const [confirm, setConfirm] = useState(false)
  const [applying, setApplying] = useState(false)

  const refresh = async () => {
    try {
      setStatus(await modelEditorApi.ollamaSettings())
    } catch {
      setStatus(null)
    }
  }
  useEffect(() => {
    void refresh()
  }, [])
  useEffect(() => {
    setParallel(config.ollamaNumParallel ?? 2)
    setMaxContext(config.ollamaMaxContext ?? 65536)
  }, [config.ollamaNumParallel, config.ollamaMaxContext])

  const dirty = parallel !== (config.ollamaNumParallel ?? 2) || maxContext !== (config.ollamaMaxContext ?? 65536)

  const apply = async () => {
    setApplying(true)
    try {
      await modelEditorApi.applyOllamaSettings()
      for (let i = 0; i < 45; i++) {
        await new Promise((r) => setTimeout(r, 2000))
        const s = await modelEditorApi.ollamaSettings()
        setStatus(s)
        if (s.applyStatus === "done" || s.applyStatus === "error") {
          if (s.applyStatus === "done") toast.success("Ollama reiniciado con la nueva configuración")
          else toast.error(s.applyDetail ?? "No se pudo aplicar")
          break
        }
      }
    } catch (err) {
      toast.error(err instanceof Error ? err.message : String(err))
    } finally {
      setApplying(false)
    }
  }

  return (
    <section className="space-y-3 rounded-lg border border-border bg-card p-4" data-testid="ollama-server-settings">
      <div>
        <h2 className="text-base font-semibold">Servidor Ollama</h2>
        <p className="text-sm text-muted-foreground">
          Un único servidor para todos los modelos de Ollama. Cada modelo cargado reserva KV para el contexto × las peticiones
          simultáneas, así que subir cualquiera de los dos sube la memoria de todos.
        </p>
      </div>
      <div className="grid gap-4 sm:grid-cols-2">
        <label className="space-y-1 text-sm">
          <span className="font-medium">Peticiones simultáneas por modelo</span>
          <input
            type="number"
            min={1}
            max={32}
            value={parallel}
            onChange={(e) => setParallel(Math.max(1, Number(e.target.value) || 1))}
            className="w-full rounded-md border border-border bg-background px-3 py-2"
          />
        </label>
        <label className="space-y-1 text-sm">
          <span className="font-medium">Contexto máximo</span>
          <select
            value={maxContext}
            onChange={(e) => setMaxContext(Number(e.target.value))}
            className="w-full rounded-md border border-border bg-background px-3 py-2"
          >
            {[...new Set([...CONTEXTS, maxContext])].sort((a, b) => a - b).map((c) => (
              <option key={c} value={c}>
                {fmtContext(c)}
              </option>
            ))}
          </select>
          <span className="text-xs text-muted-foreground">Es el contexto por defecto y el tope: un cliente no puede pedir más.</span>
        </label>
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <button
          type="button"
          disabled={!dirty}
          onClick={async () => {
            await onSave({ ollamaNumParallel: parallel, ollamaMaxContext: maxContext })
            await refresh()
          }}
          className="rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground disabled:opacity-40"
        >
          Guardar
        </button>
        <button
          type="button"
          disabled={dirty || applying || !status?.pending}
          onClick={() => setConfirm(true)}
          className="rounded-md border border-amber-500/50 px-3 py-1.5 text-sm text-amber-200 hover:bg-amber-500/10 disabled:opacity-40"
        >
          {applying ? "Aplicando…" : "Aplicar (reinicia Ollama)"}
        </button>
        {status?.running && (
          <span className="text-xs text-muted-foreground">
            En marcha: {status.running.numParallel ?? "?"} simultáneas · {fmtContext(status.running.maxContext)}
            {status.pending ? " — hay cambios sin aplicar" : " — al día"}
          </span>
        )}
      </div>
      <ConfirmDialog
        open={confirm}
        onOpenChange={setConfirm}
        title="Reiniciar Ollama"
        description="Se recrea el contenedor de Ollama: todos sus modelos se descargan y las peticiones en curso a ellos fallan. Volverán a cargarse con la nueva configuración en su siguiente petición."
        confirmLabel="Reiniciar Ollama"
        variant="danger"
        onConfirm={() => {
          setConfirm(false)
          void apply()
        }}
      />
    </section>
  )
}
