import { useState } from "react"
import { Cpu, Wand2 } from "lucide-react"
import { toast } from "sonner"
import { modelEditorApi, type AutoSetupResult, type AutoSetupRung } from "@/api/modelEditor"
import { ConfirmDialog } from "@/components/common/ConfirmDialog"
import { gpuColor } from "@/components/dashboard/GpuBadge"
import { shortGpuName } from "@/components/dashboard/gpuNames"
import { fmtContext } from "@/components/dashboard/format"
import type { GPUState } from "@/types"
import { fmtMb } from "../../PreviewPanel"
import { Callout, Hint } from "../../ui"

/** Backends para los que el servidor sabe calcular los peldaños solo. */
export const AUTO_SETUP_BACKENDS = ["llamacpp_mks", "llama_cpp"]

/** Qué parte del peldaño va a la RAM, en palabras. null = todo en GPU. */
export function describeOffload(placement: Record<string, unknown>, gpus: number[]): string | null {
  const moe = Number(placement.n_cpu_moe ?? 0)
  const layers = placement.gpu_layers
  if (moe > 0) return `expertos de ${moe} capas en RAM`
  if (gpus.length === 0 || layers === 0) return "todo en CPU (RAM)"
  if (typeof layers === "number" && layers < 999) return `${layers} capas en GPU, el resto en RAM`
  return null
}

/** Calcula los peldaños del modelo con la proyección de memoria de
 *  llama.cpp para las GPUs (y la CPU) que se marquen, y deja aplicarlos. */
export function AutoSetupCard({
  modelId,
  gpus,
  readOnly,
  onApplied,
}: {
  modelId: string
  gpus: GPUState[]
  readOnly: boolean
  onApplied: (applied: string[]) => Promise<void> | void
}) {
  const [open, setOpen] = useState(false)
  const [marked, setMarked] = useState<number[]>(() => gpus.map((g) => g.index))
  const [cpu, setCpu] = useState(false)
  const [busy, setBusy] = useState<"plan" | "apply" | null>(null)
  const [plan, setPlan] = useState<AutoSetupResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [confirm, setConfirm] = useState(false)

  const toggle = (index: number) => {
    setMarked((m) => (m.includes(index) ? m.filter((x) => x !== index) : [...m, index].sort((a, b) => a - b)))
    setPlan(null)
  }
  const body = { gpus: marked, cpu }
  const calculate = async () => {
    setBusy("plan")
    setError(null)
    setPlan(null)
    try {
      setPlan(await modelEditorApi.autoSetup(modelId, { ...body, apply: false }))
    } catch (e) {
      setError(e instanceof Error ? e.message : "No se pudo calcular")
    } finally {
      setBusy(null)
    }
  }
  const apply = async () => {
    setBusy("apply")
    setError(null)
    try {
      const r = await modelEditorApi.autoSetup(modelId, { ...body, apply: true })
      toast.success(r.applied.length ? `Peldaños aplicados: ${r.applied.join(", ")}` : "No había peldaños que aplicar")
      setPlan(null)
      setOpen(false)
      await onApplied(r.applied)
    } catch (e) {
      setError(e instanceof Error ? e.message : "No se pudieron aplicar")
    } finally {
      setBusy(null)
    }
  }

  if (!open) {
    return (
      <button
        type="button"
        disabled={readOnly}
        onClick={() => {
          setMarked(gpus.map((g) => g.index))
          setOpen(true)
        }}
        className="inline-flex items-center gap-1.5 rounded-md border border-violet-500/50 px-3 py-1.5 text-xs hover:bg-violet-500/10 disabled:opacity-40"
        data-testid="auto-setup-open"
      >
        <Wand2 size={13} /> Calcular peldaños automáticamente
      </button>
    )
  }

  return (
    <div className="space-y-3 rounded-xl border border-violet-500/40 bg-violet-500/5 p-3" data-testid="auto-setup">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <p className="text-sm font-semibold">Calcular peldaños automáticamente</p>
        <button type="button" className="text-xs text-muted-foreground hover:text-foreground" onClick={() => setOpen(false)}>
          Cerrar
        </button>
      </div>
      <Hint>
        Marca dónde puede correr el modelo. Se calcula con lo que el propio llama.cpp dice que ocupa: un peldaño «normal» en la GPU más grande con
        el mayor contexto que cabe y otro «extendido» con todas las marcadas. Si no cabe, se reparte entre GPUs o, con la CPU marcada, deja parte
        en la RAM (más lento).
      </Hint>
      <div className="flex flex-wrap gap-2">
        {gpus.map((g) => (
          <label key={g.index} className="flex items-center gap-1.5 rounded-lg border border-border px-2.5 py-1.5 text-xs" style={{ color: gpuColor(g.index) }}>
            <input type="checkbox" checked={marked.includes(g.index)} onChange={() => toggle(g.index)} aria-label={`Calcular con GPU ${g.index}`} />
            #{g.index} {shortGpuName(g.name)} · {fmtMb(g.totalVramMb)}
          </label>
        ))}
        <label className="flex items-center gap-1.5 rounded-lg border border-border px-2.5 py-1.5 text-xs text-slate-300">
          <input
            type="checkbox"
            checked={cpu}
            onChange={(e) => {
              setCpu(e.target.checked)
              setPlan(null)
            }}
            aria-label="Calcular con CPU"
          />
          <Cpu size={13} /> CPU (RAM)
        </label>
      </div>
      <button
        type="button"
        disabled={busy != null || (marked.length === 0 && !cpu)}
        onClick={() => void calculate()}
        className="rounded-md bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground disabled:opacity-40"
        data-testid="auto-setup-calculate"
      >
        {busy === "plan" ? "Calculando…" : "Calcular"}
      </button>
      {error && <Callout tone="error">{error}</Callout>}

      {plan && (
        <div className="space-y-2" data-testid="auto-setup-result">
          {plan.rungs.length === 0 ? (
            <Callout tone="error">Con lo marcado no sale ningún peldaño que quepa.</Callout>
          ) : (
            <div className="space-y-1.5">
              {plan.rungs.map((r) => (
                <RungRow key={r.key} rung={r} gpus={gpus} />
              ))}
            </div>
          )}
          {plan.needsCpu && (
            <Callout tone="warn">
              {cpu
                ? "Solo cabe dejando parte del modelo en la RAM: irá más lento."
                : "No cabe solo en las GPUs marcadas: marca la CPU (RAM) para dejar parte del modelo en memoria del sistema."}
            </Callout>
          )}
          {plan.warnings.map((w, i) => (
            <Callout key={i} tone="warn">
              {w}
            </Callout>
          ))}
          {plan.rungs.length > 0 && (
            <button
              type="button"
              disabled={readOnly || busy != null}
              onClick={() => setConfirm(true)}
              className="rounded-md border border-red-500/50 px-3 py-1.5 text-xs text-red-200 hover:bg-red-500/10 disabled:opacity-40"
              data-testid="auto-setup-apply"
            >
              {busy === "apply" ? "Aplicando…" : "Aplicar (reemplaza los peldaños actuales)"}
            </button>
          )}
        </div>
      )}
      <ConfirmDialog
        open={confirm}
        onOpenChange={(o) => !o && setConfirm(false)}
        title="Reemplazar los peldaños"
        description={`Se borran los peldaños actuales del perfil por defecto y se crean ${plan?.rungs.length ?? 0} nuevos con lo calculado. Se aplican en la siguiente carga.`}
        confirmLabel="Reemplazar"
        variant="danger"
        onConfirm={() => {
          setConfirm(false)
          void apply()
        }}
      />
    </div>
  )
}

function RungRow({ rung, gpus }: { rung: AutoSetupRung; gpus: GPUState[] }) {
  const offload = describeOffload(rung.placement ?? {}, rung.gpus)
  const name = (i: number) => `#${i} ${shortGpuName(gpus.find((g) => g.index === i)?.name)}`
  return (
    <div className="rounded-lg border border-border bg-card/60 px-3 py-2 text-xs" data-auto-rung={rung.key}>
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <span className="text-sm font-semibold">{rung.label}</span>
        <span className="tabular-nums">contexto {fmtContext(rung.context)}</span>
        <span className="text-muted-foreground">{rung.gpus.length ? rung.gpus.map(name).join(" + ") : "solo CPU"}</span>
      </div>
      <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 tabular-nums text-muted-foreground">
        {Object.entries(rung.vram_by_gpu ?? {}).map(([g, mb]) => (
          <span key={g} style={{ color: gpuColor(Number(g)) }}>
            #{g}: {fmtMb(mb)}
          </span>
        ))}
        {rung.host_mb > 0 && <span className="text-slate-300">RAM: {fmtMb(rung.host_mb)}</span>}
        <span className={offload ? "text-amber-300" : ""}>{offload ?? "todo en GPU"}</span>
      </div>
    </div>
  )
}
