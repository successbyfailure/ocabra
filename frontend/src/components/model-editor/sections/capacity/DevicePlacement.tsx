import { useRef } from "react"
import { Cpu, Star } from "lucide-react"
import { gpuColor } from "@/components/dashboard/GpuBadge"
import { shortGpuName } from "@/components/dashboard/gpuNames"
import { cn } from "@/lib/cn"
import type { GPUState } from "@/types"
import { fmtMb } from "../../PreviewPanel"
import type { MemoryShare } from "./Workbench"
import { Hint, Segmented } from "../../ui"

/** Dónde corre un peldaño: qué GPUs puede usar, cuál prueba primero, si se
 *  reparte entre varias y qué parte va a la CPU. */
export interface Placement {
  /** GPUs que puede usar. Todas marcadas y sin ★ = automática. */
  marked: number[]
  /** La que prueba primero (y la principal si se reparte). */
  star: number | null
  /** Con ≥2 marcadas: una sola GPU o todas a la vez. */
  together: boolean
  /** Con "una sola": si no cabe en ninguna, repartir entre las marcadas. */
  splitWhenNeeded: boolean
  /** Reparto entre GPUs (en el orden de `marked`); null = proporcional a la VRAM. */
  ratios: number[] | null
  splitMode: "layer" | "row" | "tensor" | null
  /** Parte en CPU. */
  cpu: boolean
  cpuMode: "layers" | "moe"
  /** Capas en GPU (modo capas). */
  gpuLayers: number | null
  /** Capas MoE con los expertos en CPU (modo MoE). */
  nCpuMoe: number | null
}

const CPU_COLOR = "#94a3b8"

export function placementSummary(p: Placement, gpus: GPUState[], layers: number | null): string {
  const name = (i: number) => `#${i} ${shortGpuName(gpus.find((g) => g.index === i)?.name)}`
  const all = gpus.length > 0 && p.marked.length === gpus.length
  let gpuText: string
  if (p.marked.length === 0) gpuText = "Solo en CPU"
  else if (p.marked.length === 1) gpuText = `Solo en la ${name(p.marked[0])}`
  else if (p.together) gpuText = `Repartido entre ${p.marked.map(name).join(" y ")}`
  else {
    const first = p.star != null ? `Prueba primero la ${name(p.star)}; si no cabe, otra` : all ? "En la GPU que tenga sitio" : `En una de ${p.marked.map(name).join(" o ")}`
    gpuText = p.splitWhenNeeded ? `${first}; si no cabe en ninguna, se reparte` : first
  }
  if (!p.cpu || p.marked.length === 0) return `${gpuText}.`
  if (p.cpuMode === "moe") return `${gpuText}, con los expertos de ${p.nCpuMoe ?? 0} capas en la CPU.`
  const onCpu = layers != null && p.gpuLayers != null ? Math.max(0, layers - p.gpuLayers) : null
  return `${gpuText}, con ${onCpu ?? "algunas"} capas en la CPU.`
}

export function DevicePlacement({
  value,
  onChange,
  gpus,
  readOnly,
  liveVramByGpu,
  memory,
  layers,
  cpuEditable = true,
  preferOnly = false,
}: {
  value: Placement
  onChange: (patch: Partial<Placement>) => void
  gpus: GPUState[]
  readOnly: boolean
  liveVramByGpu: Record<string, number>
  memory: MemoryShare[] | null
  /** Capas del modelo (para la parte en CPU). */
  layers: number | null
  cpuEditable?: boolean
  /** Backends que no reparten entre GPUs ni excluyen ninguna (bitnet): solo
   *  la ★ (GPU preferida) y la parte en CPU por capas. */
  preferOnly?: boolean
}) {
  const multi = value.marked.length > 1 && !preferOnly
  const mine = new Map((memory ?? []).map((m) => [m.gpu, Object.values(m.parts).reduce((a, b) => a + b, 0)]))
  const toggle = (index: number) => {
    const marked = value.marked.includes(index) ? value.marked.filter((g) => g !== index) : [...value.marked, index].sort((a, b) => a - b)
    onChange({ marked, star: marked.includes(value.star ?? -1) ? value.star : null, ratios: null })
  }
  // Barra: las GPUs que se usan a la vez (o la ★ / la primera) y la CPU.
  const barGpus = value.marked.length === 0 ? [] : value.together && multi ? value.marked : [value.star ?? value.marked[0]]
  const cpuShare = (() => {
    if (!value.cpu || !layers) return 0
    if (value.cpuMode === "moe") return Math.min(0.9, ((value.nCpuMoe ?? 0) / layers) * 0.85)
    const onGpu = value.gpuLayers ?? layers
    return Math.max(0, Math.min(1, (layers - onGpu) / layers))
  })()
  const gpuRatios =
    value.ratios && value.ratios.length === barGpus.length
      ? value.ratios
      : barGpus.map((i) => gpus.find((g) => g.index === i)?.totalVramMb ?? 1)
  const gpuTotal = gpuRatios.reduce((a, b) => a + b, 0) || 1
  const segments = [
    ...barGpus.map((i, k) => ({ key: `gpu${i}`, label: `#${i}`, color: gpuColor(i), share: (gpuRatios[k] / gpuTotal) * (1 - cpuShare) })),
    ...(value.cpu ? [{ key: "cpu", label: "CPU", color: CPU_COLOR, share: cpuShare }] : []),
  ]
  const showBar = segments.length > 1 || (value.cpu && barGpus.length === 0)
  const onBar = (shares: number[]) => {
    const patch: Partial<Placement> = {}
    if (value.cpu && layers) {
      const cpu = shares[shares.length - 1]
      if (value.cpuMode === "moe") patch.nCpuMoe = Math.max(0, Math.min(layers, Math.round((cpu / 0.85) * layers)))
      else patch.gpuLayers = Math.max(0, Math.min(layers, Math.round((1 - cpu) * layers)))
    }
    const g = value.cpu ? shares.slice(0, -1) : shares
    if (g.length > 1) {
      const sum = g.reduce((a, b) => a + b, 0) || 1
      patch.ratios = g.map((x) => Math.round((x / sum) * 1000) / 10)
    }
    onChange(patch)
  }
  return (
    <div className="space-y-3" data-testid="device-placement">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <p className="text-xs font-medium">Dónde corre</p>
        <span className="text-[11px] text-muted-foreground">{preferOnly ? "★: la GPU preferida (si no cabe, otra)" : "casilla: puede usarla · ★: la que prueba primero"}</span>
      </div>
      <div className="grid grid-cols-[repeat(auto-fill,minmax(150px,1fr))] gap-2">
        {gpus.map((g) => {
          const color = gpuColor(g.index)
          const on = value.marked.includes(g.index)
          const isStar = value.star === g.index
          const own = liveVramByGpu[String(g.index)] ?? 0
          const others = Math.max(0, g.usedVramMb - own)
          return (
            <div
              key={g.index}
              data-gpu-card={g.index}
              className={cn("rounded-xl border px-2.5 py-2 text-xs", on ? "" : "border-border opacity-70")}
              style={on ? { borderColor: `${color}aa`, background: `${color}10` } : undefined}
            >
              <div className="flex items-center justify-between gap-1">
                <button
                  type="button"
                  disabled={readOnly}
                  onClick={() => (preferOnly ? onChange({ star: isStar ? null : g.index }) : toggle(g.index))}
                  className="flex min-w-0 items-center gap-1.5 font-semibold"
                  style={{ color }}
                >
                  {!preferOnly && <input type="checkbox" readOnly checked={on} className="pointer-events-none" aria-label={`Usar GPU ${g.index}`} />}
                  <span className="truncate">
                    #{g.index} {shortGpuName(g.name)}
                  </span>
                </button>
                {on && (
                  <button
                    type="button"
                    disabled={readOnly}
                    title={isStar ? "La que prueba primero (quitar)" : "Probar esta primero"}
                    aria-label={`Preferida GPU ${g.index}`}
                    onClick={() => onChange({ star: isStar ? null : g.index })}
                    className={isStar ? "text-amber-300" : "text-muted-foreground/50 hover:text-amber-200"}
                  >
                    <Star size={13} fill={isStar ? "currentColor" : "none"} />
                  </button>
                )}
              </div>
              <p className="mt-1 text-[10.5px] text-muted-foreground">
                {fmtMb(g.totalVramMb)} · otros {fmtMb(others)}
                {mine.get(g.index) ? ` · este ${fmtMb(mine.get(g.index))}` : ""}
              </p>
            </div>
          )
        })}
        <div className={cn("rounded-xl border px-2.5 py-2 text-xs", value.cpu ? "border-slate-400/60 bg-slate-500/10" : "border-border opacity-70")} data-gpu-card="cpu">
          <button
            type="button"
            disabled={readOnly || !cpuEditable || !layers}
            onClick={() =>
              onChange(
                value.cpu
                  ? { cpu: false, gpuLayers: null, nCpuMoe: null }
                  : { cpu: true, gpuLayers: layers ? Math.max(0, layers - Math.max(1, Math.round(layers * 0.1))) : null },
              )
            }
            className="flex items-center gap-1.5 font-semibold text-slate-300 disabled:opacity-60"
          >
            <input type="checkbox" readOnly checked={value.cpu} className="pointer-events-none" aria-label="Usar CPU" />
            <Cpu size={13} /> CPU
          </button>
          <p className="mt-1 text-[10.5px] text-muted-foreground">{layers ? "parte del modelo en RAM (más lento)" : "sin datos de capas"}</p>
        </div>
      </div>

      {multi && (
        <div className="space-y-1.5">
          <Segmented
            size="sm"
            disabled={readOnly}
            value={value.together ? "together" : "single"}
            options={[
              { value: "single", label: "Una sola GPU", title: "La ★ (o la que tenga sitio); la otra solo si no cabe" },
              { value: "together", label: "Todas a la vez", title: "El modelo se reparte entre las marcadas" },
            ]}
            onChange={(v) => onChange({ together: v === "together", ratios: null })}
            testId="gpu-together"
          />
          {!value.together && (
            <label className="flex items-center gap-2 text-[11.5px]">
              <input type="checkbox" disabled={readOnly} checked={value.splitWhenNeeded} onChange={(e) => onChange({ splitWhenNeeded: e.target.checked })} />
              Si no cabe en ninguna, repartir entre las marcadas
            </label>
          )}
        </div>
      )}

      {showBar && (
        <div className="space-y-1.5">
          <SegmentBar segments={segments} editable={!readOnly} onChange={onBar} />
          <div className="flex flex-wrap items-center gap-2 text-[11px] text-muted-foreground">
            {barGpus.length > 1 && (
              <>
                <button type="button" disabled={readOnly || !value.ratios} className="rounded border border-border px-2 py-0.5 hover:bg-muted disabled:opacity-40" onClick={() => onChange({ ratios: null })}>
                  Proporcional a la VRAM
                </button>
                entre GPUs:
                <Segmented
                  size="sm"
                  disabled={readOnly}
                  value={value.splitMode ?? "layer"}
                  options={[
                    { value: "layer", label: "capas", title: "Cada GPU guarda capas completas; lo habitual" },
                    { value: "row", label: "filas", title: "Divide cada capa por filas; más tráfico entre GPUs" },
                    { value: "tensor", label: "tensores", title: "Divide tensores; solo con soporte en el binario" },
                  ]}
                  onChange={(v) => onChange({ splitMode: v === "layer" ? null : v })}
                />
              </>
            )}
            {value.cpu && !preferOnly && (
              <>
                a la CPU:
                <Segmented
                  size="sm"
                  disabled={readOnly || !cpuEditable}
                  value={value.cpuMode}
                  options={[
                    { value: "layers", label: "capas completas" },
                    { value: "moe", label: "expertos MoE", title: "Solo los expertos de las capas MoE: mucho más rápido en modelos MoE" },
                  ]}
                  onChange={(v) =>
                    onChange(
                      v === "moe"
                        ? { cpuMode: "moe", gpuLayers: null, nCpuMoe: layers ? Math.max(1, Math.round(layers * 0.25)) : 4 }
                        : { cpuMode: "layers", nCpuMoe: null, gpuLayers: layers ? Math.max(0, layers - Math.max(1, Math.round(layers * 0.1))) : null },
                    )
                  }
                />
              </>
            )}
          </div>
        </div>
      )}
      <Hint tone="info">{placementSummary(value, gpus, layers)}</Hint>
    </div>
  )
}

/** Barra de reparto con separadores arrastrables entre segmentos vecinos. */
export function SegmentBar({
  segments,
  editable,
  onChange,
}: {
  segments: { key: string; label: string; color: string; share: number }[]
  editable: boolean
  onChange: (shares: number[]) => void
}) {
  const ref = useRef<HTMLDivElement>(null)
  const total = segments.reduce((a, s) => a + s.share, 0) || 1
  const shares = segments.map((s) => s.share / total)
  const drag = (sep: number) => (e: React.PointerEvent) => {
    if (!editable || !ref.current) return
    e.preventDefault()
    const rect = ref.current.getBoundingClientRect()
    const move = (ev: PointerEvent) => {
      const at = Math.min(Math.max((ev.clientX - rect.left) / rect.width, 0), 1)
      const before = shares.slice(0, sep).reduce((a, b) => a + b, 0)
      const pair = shares[sep] + shares[sep + 1]
      const left = Math.min(Math.max(at - before, 0), pair)
      const next = [...shares]
      next[sep] = left
      next[sep + 1] = pair - left
      onChange(next)
    }
    const up = () => {
      window.removeEventListener("pointermove", move)
      window.removeEventListener("pointerup", up)
    }
    window.addEventListener("pointermove", move)
    window.addEventListener("pointerup", up)
  }
  return (
    <div ref={ref} className="relative flex h-8 overflow-hidden rounded-lg border border-border" data-testid="segment-bar">
      {segments.map((s, i) => (
        <div
          key={s.key}
          className="relative flex min-w-0 items-center justify-center truncate text-[11px] font-semibold"
          style={{ width: `${shares[i] * 100}%`, background: `${s.color}33`, color: s.color }}
        >
          {shares[i] > 0.08 ? `${s.label} · ${Math.round(shares[i] * 100)}%` : ""}
          {editable && i < segments.length - 1 && (
            <span
              role="separator"
              aria-label={`Separador ${s.label}`}
              onPointerDown={drag(i)}
              className="absolute inset-y-0 -right-1 z-10 w-2 cursor-col-resize bg-white/30 hover:bg-white/60"
            />
          )}
        </div>
      ))}
    </div>
  )
}
