import { useCallback, useEffect, useState } from "react"
import { ArrowUpCircle, Loader2, RefreshCw } from "lucide-react"
import { toast } from "sonner"
import { request } from "@/api/client"

/** Estado de /ocabra/system/update (backend/ocabra/core/self_update.py). */
export interface SelfUpdateStatus {
  phase: "idle" | "checking" | "waiting_idle" | "updating" | "failed" | "done"
  current: string
  latest: string | null
  checked_at: number | null
  target: string | null
  message: string | null
  busy: string[]
  update_available: boolean
  policy: "off" | "notify" | "auto"
  idle_minutes: number
  deployment: { kind: "compose" | "local_build" | "node" | "manual"; reason: string; can_apply: boolean; api_image: string | null }
  updater: { running: boolean; exit_code: number; finished_at: string; log: string[] } | null
}

export const selfUpdateApi = {
  get: () => request<SelfUpdateStatus>("GET", "/ocabra/system/update"),
  check: () => request<SelfUpdateStatus>("POST", "/ocabra/system/update/check"),
  apply: (body: { version?: string; force?: boolean }) => request<SelfUpdateStatus>("POST", "/ocabra/system/update/apply", body),
  policy: (body: { policy?: SelfUpdateStatus["policy"]; idleMinutes?: number }) =>
    request<SelfUpdateStatus>("PATCH", "/ocabra/system/update/policy", body),
}

const POLICIES: { value: SelfUpdateStatus["policy"]; title: string; text: string }[] = [
  { value: "off", title: "Apagado", text: "No busca versiones nuevas." },
  { value: "notify", title: "Avisar", text: "Avisa de la versión nueva; se instala con el botón." },
  { value: "auto", title: "Automático", text: "Se instala sola cuando el nodo lleva un rato sin actividad (sin peticiones, batches ni descargas)." },
]

const KIND_TEXT: Record<SelfUpdateStatus["deployment"]["kind"], string> = {
  compose: "Imágenes publicadas (GHCR): se actualiza sola.",
  local_build: "Compila sus propias imágenes: avisa, pero se actualiza a mano (git pull + build).",
  node: "Imagen ocabra-node: se reemplaza desde el hub.",
  manual: "Sin acceso a Docker: se actualiza a mano.",
}

export function SelfUpdateSettings() {
  const [status, setStatus] = useState<SelfUpdateStatus | null>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const [restarting, setRestarting] = useState(false)

  const load = useCallback(async () => {
    try {
      setStatus(await selfUpdateApi.get())
      setRestarting(false)
    } catch {
      // Durante la actualización la API se reinicia: fallar aquí es lo normal.
      setRestarting((r) => r)
    }
  }, [])

  useEffect(() => {
    void load()
  }, [load])

  // Mientras actualiza, sondea hasta que vuelva la API (con otra versión).
  const updating = restarting || status?.phase === "updating" || !!status?.updater?.running
  useEffect(() => {
    if (!updating) return
    const id = window.setInterval(() => void load(), 4000)
    return () => window.clearInterval(id)
  }, [updating, load])

  const run = async (label: string, fn: () => Promise<SelfUpdateStatus>) => {
    setBusy(label)
    try {
      setStatus(await fn())
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(null)
    }
  }

  if (!status) {
    return (
      <section className="rounded-lg border border-border bg-card p-4 text-sm text-muted-foreground" data-testid="self-update">
        Actualizaciones: cargando…
      </section>
    )
  }

  const canApply = status.deployment.can_apply && status.update_available && !updating
  const lastRun = status.updater && !status.updater.running ? status.updater : null

  return (
    <section className="space-y-4 rounded-lg border border-border bg-card p-4" data-testid="self-update">
      <div>
        <h2 className="flex items-center gap-1.5 text-sm font-semibold uppercase tracking-wide text-muted-foreground">
          <ArrowUpCircle size={14} /> Actualizaciones de oCabra
        </h2>
        <p className="mt-1 text-xs text-muted-foreground">{KIND_TEXT[status.deployment.kind]}</p>
      </div>

      <div className="flex flex-wrap items-center gap-3 text-sm">
        <span>
          Versión <b>{status.current}</b>
        </span>
        <span className="text-muted-foreground">·</span>
        <span data-testid="self-update-latest">
          {status.latest == null
            ? "sin datos de la última publicada"
            : status.update_available
              ? <>publicada <b className="text-emerald-500">{status.latest}</b></>
              : `al día (última publicada ${status.latest})`}
        </span>
        <button
          type="button"
          disabled={busy !== null || updating}
          onClick={() => void run("check", selfUpdateApi.check)}
          className="inline-flex items-center gap-1 rounded-md border border-border px-2 py-1 text-xs hover:bg-muted disabled:opacity-50"
        >
          <RefreshCw size={12} className={busy === "check" ? "animate-spin" : ""} /> Buscar ahora
        </button>
      </div>

      <div className="grid gap-2 md:grid-cols-3">
        {POLICIES.map((p) => (
          <label
            key={p.value}
            className={`cursor-pointer rounded-md border p-3 text-sm ${status.policy === p.value ? "border-primary bg-primary/5" : "border-border hover:bg-muted"}`}
          >
            <span className="flex items-center gap-2 font-medium">
              <input
                type="radio"
                name="self-update-policy"
                className="accent-primary"
                checked={status.policy === p.value}
                disabled={busy !== null}
                onChange={() => void run("policy", () => selfUpdateApi.policy({ policy: p.value }))}
              />
              {p.title}
            </span>
            <span className="mt-1 block text-xs text-muted-foreground">{p.text}</span>
          </label>
        ))}
      </div>
      {status.policy === "auto" && (
        <label className="flex flex-wrap items-center gap-2 text-sm">
          <span>Espera</span>
          <input
            type="number"
            min={1}
            max={1440}
            defaultValue={status.idle_minutes}
            onBlur={(e) => {
              const v = Math.max(1, Math.min(1440, Number(e.target.value) || 10))
              if (v !== status.idle_minutes) void run("policy", () => selfUpdateApi.policy({ idleMinutes: v }))
            }}
            className="w-20 rounded-md border border-border bg-background px-2 py-1 text-sm"
          />
          <span className="text-xs text-muted-foreground">minutos sin actividad antes de instalar</span>
        </label>
      )}

      {!status.deployment.can_apply && status.deployment.reason && (
        <p className="text-xs text-amber-700 dark:text-amber-300">{status.deployment.reason}</p>
      )}
      {status.phase === "waiting_idle" && status.busy.length > 0 && (
        <p className="text-xs text-muted-foreground">Esperando a que quede libre: {status.busy.join(", ")}.</p>
      )}
      {status.message && <p className="text-xs text-muted-foreground">{status.message}</p>}

      {updating && (
        <p className="flex items-center gap-2 text-sm text-sky-500" data-testid="self-update-running">
          <Loader2 size={14} className="animate-spin" />
          Actualizando{status.target ? ` a ${status.target}` : ""}: la API se reinicia, esta página vuelve sola.
        </p>
      )}
      {lastRun && lastRun.log.length > 0 && (
        <details className="text-xs">
          <summary className={lastRun.exit_code === 0 ? "text-emerald-500" : "text-red-400"}>
            Última actualización: {lastRun.exit_code === 0 ? "correcta" : lastRun.exit_code === 4 ? "fallida, se volvió a la versión anterior" : `fallida (código ${lastRun.exit_code})`}
          </summary>
          <pre className="mt-1 max-h-48 overflow-auto rounded bg-muted p-2">{lastRun.log.join("\n")}</pre>
        </details>
      )}

      <div className="flex justify-end">
        <button
          type="button"
          disabled={!canApply || busy !== null}
          onClick={() => {
            if (!window.confirm(`Actualizar a ${status.latest}? La API se reiniciará (≈1 min).`)) return
            setRestarting(true)
            void run("apply", () => selfUpdateApi.apply({ version: status.latest ?? undefined }))
          }}
          className="rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground disabled:opacity-50"
          data-testid="self-update-apply"
        >
          {status.update_available ? `Actualizar a ${status.latest}` : "Actualizar"}
        </button>
      </div>
    </section>
  )
}
