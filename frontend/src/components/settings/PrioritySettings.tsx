import { useEffect, useState } from "react"
import { Gauge } from "lucide-react"
import { toast } from "sonner"
import type { ServerConfig } from "@/types"

interface PrioritySettingsProps {
  config: ServerConfig
  onSave: (patch: Partial<ServerConfig>) => Promise<void>
}

export function PrioritySettings({ config, onSave }: PrioritySettingsProps) {
  const [batchPriorityEnabled, setBatchPriorityEnabled] = useState(config.batchPriorityEnabled)
  const [batchPreemptMode, setBatchPreemptMode] = useState(config.batchPreemptMode)
  const [batchPreemptMinElapsedS, setBatchPreemptMinElapsedS] = useState(config.batchPreemptMinElapsedS)
  const [batchPreemptGraceS, setBatchPreemptGraceS] = useState(config.batchPreemptGraceS)
  const [batchFreeSlotsPct, setBatchFreeSlotsPct] = useState(config.batchFreeSlotsPct)
  const [batchResumeCooldownSlotsS, setBatchResumeCooldownSlotsS] = useState(config.batchResumeCooldownSlotsS)
  const [batchResumeCooldownEvictionS, setBatchResumeCooldownEvictionS] = useState(config.batchResumeCooldownEvictionS)
  const [batchInterJobCooldownS, setBatchInterJobCooldownS] = useState(config.batchInterJobCooldownS)

  useEffect(() => {
    setBatchPriorityEnabled(config.batchPriorityEnabled)
    setBatchPreemptMode(config.batchPreemptMode)
    setBatchPreemptMinElapsedS(config.batchPreemptMinElapsedS)
    setBatchPreemptGraceS(config.batchPreemptGraceS)
    setBatchFreeSlotsPct(config.batchFreeSlotsPct)
    setBatchResumeCooldownSlotsS(config.batchResumeCooldownSlotsS)
    setBatchResumeCooldownEvictionS(config.batchResumeCooldownEvictionS)
    setBatchInterJobCooldownS(config.batchInterJobCooldownS)
  }, [
    config.batchPriorityEnabled,
    config.batchPreemptMode,
    config.batchPreemptMinElapsedS,
    config.batchPreemptGraceS,
    config.batchFreeSlotsPct,
    config.batchResumeCooldownSlotsS,
    config.batchResumeCooldownEvictionS,
    config.batchInterJobCooldownS,
  ])

  const save = async () => {
    try {
      await onSave({
        batchPriorityEnabled,
        batchPreemptMode,
        batchPreemptMinElapsedS,
        batchPreemptGraceS,
        batchFreeSlotsPct,
        batchResumeCooldownSlotsS,
        batchResumeCooldownEvictionS,
        batchInterJobCooldownS,
      })
      toast.success("Prioridad guardada")
    } catch {
      // page-level toast is shown in Settings
    }
  }

  return (
    <section className="space-y-3 rounded-lg border border-border bg-card p-4">
      <div className="flex items-center gap-2">
        <Gauge size={16} className="text-muted-foreground" />
        <h2 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">
          Prioridad: lotes vs. interactivo
        </h2>
      </div>
      <p className="text-xs text-muted-foreground">
        Cuando una petición interactiva necesita un slot o un modelo que está ocupado por trabajo
        de la <code className="font-mono">Batches API</code>, estas reglas deciden si se espera o
        se cancela el trabajo del lote para dar paso de inmediato. Nunca afecta a dos peticiones
        interactivas entre sí — solo entra en juego cuando el que ocupa el recurso es un lote.
      </p>

      <label className="flex items-center gap-2 text-sm text-muted-foreground">
        <input
          type="checkbox"
          checked={batchPriorityEnabled}
          onChange={(event) => setBatchPriorityEnabled(event.target.checked)}
        />
        Activar prioridad de interactivo sobre lotes
      </label>

      <div className="space-y-2 rounded-md border border-border/60 p-3">
        <label className="block text-sm text-muted-foreground">
          Modo de cancelación
          <select
            disabled={!batchPriorityEnabled}
            value={batchPreemptMode}
            onChange={(event) =>
              setBatchPreemptMode(event.target.value as ServerConfig["batchPreemptMode"])
            }
            className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2 disabled:opacity-50"
          >
            <option value="never">Nunca — el lote nunca se cancela, el interactivo espera</option>
            <option value="instant">Instantáneo — cancela el lote de inmediato siempre</option>
            <option value="conditional">
              Condicional — deja terminar si ya lleva tiempo invertido y falta poco
            </option>
          </select>
        </label>

        {batchPreemptMode === "conditional" && (
          <div className="grid gap-2 md:grid-cols-2">
            <label className="text-sm text-muted-foreground">
              Tiempo ya invertido para considerar "no cancelar" (s)
              <input
                type="number"
                min={0}
                disabled={!batchPriorityEnabled}
                value={batchPreemptMinElapsedS}
                onChange={(event) => setBatchPreemptMinElapsedS(Number(event.target.value))}
                className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2 disabled:opacity-50"
              />
              <span className="mt-1 block text-xs text-muted-foreground/80">
                Solo se aplica la excepción de "dejar terminar" a peticiones de lote que ya lleven
                más de este tiempo corriendo — evita proteger tareas que apenas han empezado.
              </span>
            </label>
            <label className="text-sm text-muted-foreground">
              Margen restante para dejar terminar (s)
              <input
                type="number"
                min={0}
                disabled={!batchPriorityEnabled}
                value={batchPreemptGraceS}
                onChange={(event) => setBatchPreemptGraceS(Number(event.target.value))}
                className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2 disabled:opacity-50"
              />
              <span className="mt-1 block text-xs text-muted-foreground/80">
                Si a la petición del lote le queda menos de esto (estimado), se deja terminar en
                vez de cancelarla.
              </span>
            </label>
          </div>
        )}

        <label className="block text-sm text-muted-foreground">
          Fracción de slots a liberar cuando el lote satura el worker: {Math.round(batchFreeSlotsPct * 100)}%
          <input
            type="range"
            min={0}
            max={1}
            step={0.1}
            disabled={!batchPriorityEnabled}
            value={batchFreeSlotsPct}
            onChange={(event) => setBatchFreeSlotsPct(Number(event.target.value))}
            className="mt-2 w-full disabled:opacity-50"
          />
          <span className="mt-1 block text-xs text-muted-foreground/80">
            100% cancela todo lo que el lote tenga corriendo en ese worker; un valor menor cancela
            solo una parte, dejando que el resto termine de forma natural.
          </span>
        </label>
      </div>

      <div className="grid gap-2 md:grid-cols-3">
        <label className="text-sm text-muted-foreground">
          Cooldown tras liberar slot (s)
          <input
            type="number"
            min={0}
            disabled={!batchPriorityEnabled}
            value={batchResumeCooldownSlotsS}
            onChange={(event) => setBatchResumeCooldownSlotsS(Number(event.target.value))}
            className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2 disabled:opacity-50"
          />
          <span className="mt-1 block text-xs text-muted-foreground/80">
            Tras cancelar una petición del lote para liberar un slot, cuánto tiempo se le impide
            volver a ese worker antes de reintentar.
          </span>
        </label>
        <label className="text-sm text-muted-foreground">
          Cooldown tras desalojo de modelo (s)
          <input
            type="number"
            min={0}
            disabled={!batchPriorityEnabled}
            value={batchResumeCooldownEvictionS}
            onChange={(event) => setBatchResumeCooldownEvictionS(Number(event.target.value))}
            className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2 disabled:opacity-50"
          />
          <span className="mt-1 block text-xs text-muted-foreground/80">
            Más largo que el anterior — recargar un modelo es mucho más caro que perder una sola
            generación, así que el lote espera más antes de volver.
          </span>
        </label>
        <label className="text-sm text-muted-foreground">
          Pausa entre lotes consecutivos (s)
          <input
            type="number"
            min={0}
            value={batchInterJobCooldownS}
            onChange={(event) => setBatchInterJobCooldownS(Number(event.target.value))}
            className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2"
          />
          <span className="mt-1 block text-xs text-muted-foreground/80">
            No depende del interruptor de arriba — simplemente deja respirar al servidor entre un
            lote y el siguiente cuando hay varios en cola. 0 = sin pausa.
          </span>
        </label>
      </div>

      <button
        type="button"
        onClick={() => void save()}
        className="rounded-md bg-primary px-3 py-2 text-sm font-medium text-primary-foreground"
      >
        Guardar prioridad
      </button>
    </section>
  )
}
