import { useEffect, useState } from "react"
import { ShieldCheck } from "lucide-react"
import { toast } from "sonner"
import type { ServerConfig } from "@/types"

interface PrivacySettingsProps {
  config: ServerConfig
  onSave: (patch: Partial<ServerConfig>) => Promise<void>
}

const MODES: { value: ServerConfig["privacyMode"]; title: string; text: string }[] = [
  {
    value: "functional",
    title: "Funcional",
    text:
      "Lo más privado posible sin entorpecer el funcionamiento. Guarda los metadatos de cada petición (quién, qué modelo, cuánto tardó; nunca el contenido) para las " +
      "estadísticas, y escribe de forma temporal lo que haga falta (ficheros de batches, imágenes por URL), " +
      "que se borra en cuanto deja de hacer falta.",
  },
  {
    value: "zero_data",
    title: "0 data",
    text:
      "No guarda nada de las peticiones ni de quién las hace: sin estadísticas por petición, sin nombres en el " +
      "feed en vivo, sin batches ni ficheros (necesitan guardarse), imágenes siempre en línea, nada de contenido " +
      "en los logs y sin trazas externas. Pensado para nodos en máquinas ajenas. No protege frente al dueño de " +
      "la máquina, que puede leer su memoria y su disco.",
  },
]

export function PrivacySettings({ config, onSave }: PrivacySettingsProps) {
  const [mode, setMode] = useState(config.privacyMode)
  const [retention, setRetention] = useState(config.openaiFilesRetentionHours)

  useEffect(() => {
    setMode(config.privacyMode)
    setRetention(config.openaiFilesRetentionHours)
  }, [config.privacyMode, config.openaiFilesRetentionHours])

  const dirty = mode !== config.privacyMode || retention !== config.openaiFilesRetentionHours

  return (
    <section className="space-y-4 rounded-lg border border-border bg-card p-4">
      <div>
        <h2 className="flex items-center gap-1.5 text-sm font-semibold uppercase tracking-wide text-muted-foreground">
          <ShieldCheck size={14} /> Privacidad
        </h2>
        <p className="mt-1 text-xs text-muted-foreground">
          Qué guarda este servidor de las peticiones que sirve. Cada respuesta lo dice en la cabecera{" "}
          <code>X-Ocabra-Privacy</code>, y los demás nodos de la federación lo ven.
        </p>
      </div>
      <div className="grid gap-2 md:grid-cols-2">
        {MODES.map((m) => (
          <label
            key={m.value}
            className={`cursor-pointer rounded-md border p-3 text-sm ${
              mode === m.value ? "border-primary bg-primary/5" : "border-border hover:bg-muted"
            }`}
          >
            <span className="flex items-center gap-2 font-medium">
              <input
                type="radio"
                name="privacy-mode"
                className="accent-primary"
                checked={mode === m.value}
                onChange={() => setMode(m.value)}
              />
              {m.title}
            </span>
            <span className="mt-1 block text-xs text-muted-foreground">{m.text}</span>
          </label>
        ))}
      </div>
      {config.privacyMode === "zero_data" && config.privacyTempDir && (
        <p
          className={`text-xs ${
            !config.privacyTempDir.in_memory || (config.privacyTempDir.free_mb ?? 0) < 1024
              ? "text-amber-700 dark:text-amber-300"
              : "text-muted-foreground"
          }`}
        >
          {config.privacyTempDir.in_memory
            ? `Los ficheros temporales (subidas, audio y vídeo en proceso) van a memoria: ${config.privacyTempDir.dir}, ${config.privacyTempDir.free_mb ?? "?"} MB libres.`
            : `Los ficheros temporales van a disco (${config.privacyTempDir.dir}): este nodo no tiene /dev/shm.`}
          {config.privacyTempDir.in_memory && (config.privacyTempDir.free_mb ?? 0) < 1024
            ? " Es poco: una subida más grande fallará en vez de escribirse a disco."
            : ""}
        </p>
      )}
      {mode === "functional" && (
        <label className="flex flex-wrap items-center gap-2 text-sm">
          <span>Borrar ficheros de batches y subidas</span>
          <input
            type="number"
            min={0}
            className="w-24 rounded-md border border-border bg-background px-2 py-1 text-sm"
            value={retention}
            onChange={(e) => setRetention(Math.max(0, Number(e.target.value) || 0))}
          />
          <span className="text-xs text-muted-foreground">
            horas después de terminar su batch (0 = no borrar nunca; por defecto 168, una semana)
          </span>
        </label>
      )}
      <div className="flex justify-end">
        <button
          type="button"
          disabled={!dirty}
          className="rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground disabled:opacity-50"
          onClick={async () => {
            try {
              await onSave({ privacyMode: mode, openaiFilesRetentionHours: retention })
              toast.success(mode === "zero_data" ? "Modo 0 data activado" : "Privacidad guardada")
            } catch {
              // the parent already showed the error
            }
          }}
        >
          Guardar
        </button>
      </div>
    </section>
  )
}
