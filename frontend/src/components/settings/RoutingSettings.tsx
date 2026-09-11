import { useEffect, useState } from "react"
import { GitBranch, Radio } from "lucide-react"
import { toast } from "sonner"
import type { ServerConfig } from "@/types"

interface Props {
  config: ServerConfig
  onSave: (patch: Partial<ServerConfig>) => Promise<void>
}

/**
 * Bloque 20 — routing/estimator/session knobs. All hot-reloadable, so
 * changes take effect on the next request without a backend restart.
 *
 * We split the form into two visual groups (Router · Sessions) to
 * mirror the two subsystems, but they share the same save action.
 */
export function RoutingSettings({ config, onSave }: Props) {
  const [routingEnabled, setRoutingEnabled] = useState<boolean>(config.routingEnabled ?? true)
  const [confidence, setConfidence] = useState<number>(config.routerConfidenceFloor ?? 0.3)
  const [fallbackMs, setFallbackMs] = useState<number>(config.routerFallbackDelayMs ?? 200)
  const [drainCap, setDrainCap] = useState<number>(config.maxDrainTimeoutS ?? 900)
  const [pauseS, setPauseS] = useState<number>(config.sessionPauseThresholdS ?? 120)
  const [idleS, setIdleS] = useState<number>(config.sessionMaxIdleS ?? 900)
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    setRoutingEnabled(config.routingEnabled ?? true)
    setConfidence(config.routerConfidenceFloor ?? 0.3)
    setFallbackMs(config.routerFallbackDelayMs ?? 200)
    setDrainCap(config.maxDrainTimeoutS ?? 900)
    setPauseS(config.sessionPauseThresholdS ?? 120)
    setIdleS(config.sessionMaxIdleS ?? 900)
  }, [config])

  const handleSave = async () => {
    setSaving(true)
    try {
      await onSave({
        routingEnabled,
        routerConfidenceFloor: confidence,
        routerFallbackDelayMs: fallbackMs,
        maxDrainTimeoutS: drainCap,
        sessionPauseThresholdS: pauseS,
        sessionMaxIdleS: idleS,
      })
      toast.success("Routing y Sessions actualizados")
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Error al guardar")
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="space-y-6">
      <section className="rounded-lg border border-border bg-card p-4">
        <div className="mb-3 flex items-center gap-2">
          <GitBranch size={16} className="text-primary" />
          <h3 className="text-sm font-semibold">Router de perfiles</h3>
        </div>
        <p className="mb-4 text-xs text-muted-foreground">
          Controla cómo los perfiles con <code>routing_targets</code> reenvían las peticiones.
          Ideal para tener una etiqueta única (ej. <code>gemma4:26b</code>) delante de
          varios backends con fallback en cascada.
        </p>

        <div className="grid gap-4 md:grid-cols-2">
          <label className="flex items-center gap-2 text-sm">
            <button
              type="button"
              role="switch"
              aria-checked={routingEnabled}
              onClick={() => setRoutingEnabled((v) => !v)}
              className={`relative h-5 w-9 rounded-full transition-colors ${
                routingEnabled ? "bg-emerald-500" : "bg-muted"
              }`}
            >
              <span
                className={`absolute top-0.5 block h-4 w-4 rounded-full bg-white transition-transform ${
                  routingEnabled ? "translate-x-4" : "translate-x-0.5"
                }`}
              />
            </button>
            <div>
              <div className="font-medium">Router activo</div>
              <div className="text-[11px] text-muted-foreground">
                Kill switch. Si está apagado, los <code>routing_targets</code> se ignoran.
              </div>
            </div>
          </label>

          <div>
            <label className="mb-1 block text-sm font-medium">
              Confianza mínima <span className="text-muted-foreground">({confidence.toFixed(2)})</span>
            </label>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={confidence}
              onChange={(e) => setConfidence(Number(e.target.value))}
              className="w-full"
            />
            <p className="mt-1 text-[10px] text-muted-foreground">
              Confianza del estimador necesaria para creer que un target "estará libre
              pronto" y esperar en vez de saltar al siguiente.
            </p>
          </div>

          <div>
            <label className="mb-1 block text-sm font-medium">Delay antes de fallback (ms)</label>
            <input
              type="number"
              min={0}
              value={fallbackMs}
              onChange={(e) => setFallbackMs(Number(e.target.value))}
              className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
            />
            <p className="mt-1 text-[10px] text-muted-foreground">
              Cuánto espera el router en un target ocupado antes de degradar al siguiente.
            </p>
          </div>

          <div>
            <label className="mb-1 block text-sm font-medium">Drain máximo (s)</label>
            <input
              type="number"
              min={1}
              value={drainCap}
              onChange={(e) => setDrainCap(Number(e.target.value))}
              className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
            />
            <p className="mt-1 text-[10px] text-muted-foreground">
              Tope duro del drain dinámico usado por pressure_eviction. Protege ante
              estimaciones desbocadas.
            </p>
          </div>
        </div>
      </section>

      <section className="rounded-lg border border-border bg-card p-4">
        <div className="mb-3 flex items-center gap-2">
          <Radio size={16} className="text-primary" />
          <h3 className="text-sm font-semibold">Sesiones Realtime</h3>
        </div>
        <p className="mb-4 text-xs text-muted-foreground">
          Umbrales para marcar sesiones como pausadas (workers reusables) y para
          liberar workers de sesiones zombie que dejaron de latir.
        </p>

        <div className="grid gap-4 md:grid-cols-2">
          <div>
            <label className="mb-1 block text-sm font-medium">Umbral de pausa (s)</label>
            <input
              type="number"
              min={1}
              value={pauseS}
              onChange={(e) => setPauseS(Number(e.target.value))}
              className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
            />
            <p className="mt-1 text-[10px] text-muted-foreground">
              Segundos sin actividad para marcar la sesión como pausada.
            </p>
          </div>

          <div>
            <label className="mb-1 block text-sm font-medium">Idle máximo (s)</label>
            <input
              type="number"
              min={1}
              value={idleS}
              onChange={(e) => setIdleS(Number(e.target.value))}
              className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
            />
            <p className="mt-1 text-[10px] text-muted-foreground">
              Segundos sin heartbeat después de los cuales el sweeper libera los workers.
            </p>
          </div>
        </div>
      </section>

      <div className="flex justify-end">
        <button
          type="button"
          onClick={() => void handleSave()}
          disabled={saving}
          className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
        >
          {saving ? "Guardando..." : "Guardar cambios"}
        </button>
      </div>
    </div>
  )
}
