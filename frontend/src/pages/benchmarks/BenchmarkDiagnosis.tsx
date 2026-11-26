import { useCallback, useEffect, useState } from "react"
import { Download, Loader2, Stethoscope } from "lucide-react"
import { toast } from "sonner"
import { api } from "@/api/client"
import type {
  BenchmarkAttempt,
  BenchmarkDiagnosis,
  BenchmarkDiagnosisCategory,
  BenchmarkRun,
} from "@/types"

/** Apps cuyos casos fallidos sabe leer el diagnóstico (backend: diagnosis/cases.py). */
export const DIAGNOSABLE_APPS = new Set(["ifeval", "bfcl", "corrala", "fulcrum"])

export const CATEGORY_LABEL: Record<BenchmarkDiagnosisCategory, string> = {
  plantilla: "Plantilla",
  sampling: "Sampling",
  limites: "Límites",
  servido: "Cuantización / servido",
  modelo: "Modelo",
  no_reproduce: "No se reproduce",
  sin_verificar: "Sin verificar",
}

const CATEGORY_CLASS: Record<BenchmarkDiagnosisCategory, string> = {
  plantilla: "text-amber-300",
  sampling: "text-sky-300",
  limites: "text-orange-300",
  servido: "text-violet-300",
  modelo: "text-emerald-300",
  no_reproduce: "text-muted-foreground",
  sin_verificar: "text-muted-foreground",
}

const ACTIVE = new Set(["queued", "running"])

function latestCompletedAttempts(attempts: BenchmarkAttempt[]): BenchmarkAttempt[] {
  const byModel = new Map<string, BenchmarkAttempt>()
  attempts.forEach((attempt) => {
    if (attempt.status === "completed") byModel.set(attempt.model, attempt)
  })
  return [...byModel.values()]
}

/** Diagnóstico «¿es el modelo o es cómo lo servimos?» de los intentos de una ejecución. */
export function BenchmarkDiagnoses({ run }: { run: BenchmarkRun }) {
  const attempts = latestCompletedAttempts(run.attempts)
  const [diagnoses, setDiagnoses] = useState<BenchmarkDiagnosis[]>([])
  const [busy, setBusy] = useState<string | null>(null)
  const [open, setOpen] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    const lists = await Promise.all(attempts.map((attempt) => api.benchmarks.diagnoses.list(attempt.id)))
    const all = lists.flat()
    const detailed = await Promise.all(
      all.map((item) => (item.status === "completed" ? api.benchmarks.diagnoses.get(item.id) : Promise.resolve(item))),
    )
    setDiagnoses(detailed)
  }, [attempts.map((attempt) => attempt.id).join(",")]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!DIAGNOSABLE_APPS.has(run.app) || attempts.length === 0) return
    void refresh().catch(() => undefined)
  }, [refresh, run.app, attempts.length])

  const active = diagnoses.some((item) => ACTIVE.has(item.status))
  useEffect(() => {
    if (!active) return
    const timer = window.setInterval(() => void refresh().catch(() => undefined), 10_000)
    return () => window.clearInterval(timer)
  }, [active, refresh])

  if (!DIAGNOSABLE_APPS.has(run.app) || attempts.length === 0) return null

  const launch = async (attempt: BenchmarkAttempt) => {
    setBusy(attempt.id)
    try {
      await api.benchmarks.diagnoses.create(attempt.id)
      toast.success(`Diagnóstico de ${attempt.model} encolado`)
      await refresh()
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "No se pudo encolar el diagnóstico")
    } finally {
      setBusy(null)
    }
  }

  return (
    <details className="mt-3 rounded-lg border border-border px-3 py-2 text-xs">
      <summary className="flex cursor-pointer list-none items-center gap-1.5 text-muted-foreground">
        <Stethoscope size={13} /> ¿Es el modelo o es cómo lo servimos? ({diagnoses.length})
      </summary>
      <p className="mt-2 text-muted-foreground">
        Clasifica cada fallo en plantilla, sampling, límites, cuantización/servido o modelo. Repite
        una muestra de casos con el muestreo del fabricante por la cola normal; solo corre con la
        cola de benchmarks vacía. Lo que queda como «modelo» se exporta como candidato a entrenamiento.
      </p>
      <div className="mt-2 flex flex-wrap gap-2">
        {attempts.map((attempt) => (
          <button
            key={attempt.id}
            type="button"
            disabled={busy !== null}
            onClick={() => void launch(attempt)}
            className="inline-flex items-center gap-1.5 rounded-md border border-border px-2 py-1 hover:bg-muted"
          >
            {busy === attempt.id ? <Loader2 size={12} className="animate-spin" /> : <Stethoscope size={12} />}
            Diagnosticar {attempt.model}
          </button>
        ))}
      </div>
      {diagnoses.map((item) => (
        <div key={item.id} className="mt-3 rounded-md border border-border/70 p-2">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <span className="font-mono">{item.model}</span>
            <span className="text-muted-foreground">
              {item.status === "completed"
                ? "completado"
                : item.status === "failed"
                  ? "falló"
                  : `${item.phase ?? item.status}${item.progress?.total ? ` · ${item.progress.completed}/${item.progress.total}` : ""}`}
            </span>
          </div>
          {item.error && <div className="mt-1 text-destructive">{item.error}</div>}
          {item.report && (
            <>
              <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1">
                {(Object.keys(CATEGORY_LABEL) as BenchmarkDiagnosisCategory[]).map((name) =>
                  item.report!.summary.counts[name] ? (
                    <span key={name} className={CATEGORY_CLASS[name]}>
                      {CATEGORY_LABEL[name]}: {item.report!.summary.counts[name]} (
                      {item.report!.summary.percent_of_failures[name]} %)
                    </span>
                  ) : null,
                )}
              </div>
              {item.report.findings.length > 0 && (
                <ul className="mt-2 list-disc pl-4 text-amber-200/90">
                  {item.report.findings.map((finding) => <li key={finding}>{finding}</li>)}
                </ul>
              )}
              <div className="mt-2 flex flex-wrap gap-2">
                {["report.md", "candidates.train.jsonl", "candidates.eval.jsonl"].map((name) => (
                  <a
                    key={name}
                    href={api.benchmarks.diagnoses.fileUrl(item.id, name)}
                    className="inline-flex items-center gap-1 rounded border border-border px-2 py-0.5 hover:bg-muted"
                  >
                    <Download size={11} /> {name}
                  </a>
                ))}
                <button type="button" onClick={() => setOpen(open === item.id ? null : item.id)} className="rounded border border-border px-2 py-0.5 hover:bg-muted">
                  {open === item.id ? "Ocultar casos" : `Ver ${item.report.cases.length} casos`}
                </button>
              </div>
              {open === item.id && (
                <div className="mt-2 max-h-96 overflow-auto">
                  <table className="w-full min-w-[640px] text-left">
                    <thead className="text-muted-foreground"><tr><th className="py-1">Caso</th><th>Categoría</th><th>Fallo</th><th>Evidencia</th></tr></thead>
                    <tbody>
                      {item.report.cases.map((row) => (
                        <tr key={row.case_id} className="border-t border-border/50 align-top">
                          <td className="py-1 font-mono">{row.case_id}</td>
                          <td className={CATEGORY_CLASS[row.category]}>{CATEGORY_LABEL[row.category]}</td>
                          <td className="max-w-64 truncate" title={row.failure}>{row.failure}</td>
                          <td className="text-muted-foreground">{row.evidence.join("; ")}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </>
          )}
        </div>
      ))}
    </details>
  )
}
