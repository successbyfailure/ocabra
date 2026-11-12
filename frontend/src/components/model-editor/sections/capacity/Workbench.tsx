import { useMemo, useRef, type ReactNode } from "react"
import { gpuColor } from "@/components/dashboard/GpuBadge"
import { shortGpuName } from "@/components/dashboard/gpuNames"
import { fmtContext } from "@/components/dashboard/format"
import { cn } from "@/lib/cn"
import type { GPUState } from "@/types"
import { PART_META, fmtMb } from "../../PreviewPanel"
import type { CapacityInfo } from "../../types"
import { Callout, Hint, Panel, Segmented } from "../../ui"
import { ContextSlider } from "../ContextSection"

export type KvLabel = "fp16" | "bf16" | "fp8" | "q8" | "q4"

/** Lo que decide la memoria de un peldaño, igual para la base y para un preset. */
export interface WorkbenchState {
  /** Contexto por petición (por slot) o, con pool, el total compartido. */
  context: number
  /** Preset en modo "lo que quepa": el contexto lo decide el planificador. */
  maxMode?: boolean
  slots: number
  pool: boolean
  kv: KvLabel
  /** GPUs que usa a la vez (para el techo de la gráfica); null = la mayor. */
  gpus: number[] | null
}

export interface MemoryShare {
  gpu: number
  parts: Record<string, number>
}

const KV_OPTIONS: { value: KvLabel; label: string; hint: string }[] = [
  { value: "fp16", label: "f16", hint: "sin cuantizar · 2 bytes" },
  { value: "q8", label: "q8", hint: "mitad de memoria · 1 byte" },
  { value: "q4", label: "q4", hint: "un cuarto, algo de pérdida" },
]
const KV_BYTES: Record<KvLabel, number> = { fp16: 2, bf16: 2, fp8: 1, q8: 1, q4: 0.5 }

/** Banco de trabajo de Capacidad: a la izquierda los controles que cambian la
 *  memoria, a la derecha la memoria por GPU con su leyenda, y debajo la
 *  gráfica de contexto × peticiones simultáneas. */
export function Workbench({
  title,
  subtitle,
  value,
  onChange,
  readOnly,
  gpus,
  liveVramByGpu,
  memory,
  memoryNote,
  cap,
  native,
  allowMax,
  poolEditable,
  measured,
  devices,
  header,
  footer,
}: {
  title: ReactNode
  subtitle?: ReactNode
  value: WorkbenchState
  onChange: (patch: Partial<WorkbenchState>) => void
  readOnly: boolean
  gpus: GPUState[]
  liveVramByGpu: Record<string, number>
  memory: MemoryShare[] | null
  memoryNote?: ReactNode
  cap: CapacityInfo | null
  native: number | null
  allowMax: boolean
  poolEditable: boolean
  /** Contexto medido de verdad (sondeo), si lo hay. */
  measured?: number | null
  /** Dónde corre (GPUs y CPU); ocupa todo el ancho encima de los controles. */
  devices?: ReactNode
  header?: ReactNode
  footer?: ReactNode
}) {
  return (
    <Panel title={title} subtitle={subtitle}>
      {header}
      {devices && <div className="mb-5 border-b border-border pb-5">{devices}</div>}
      {/* Se apilan según el ancho disponible, no el de la ventana: el panel
          puede ser estrecho con la ventana ancha (lateral abierto). */}
      <div className="flex flex-wrap gap-5" data-testid="workbench">
        <div className="min-w-0 flex-[2_1_360px] space-y-5">
          <ContextKnobs value={value} onChange={onChange} readOnly={readOnly} native={native} allowMax={allowMax} poolEditable={poolEditable} />
        </div>
        <div className="min-w-0 flex-[1_1_280px] space-y-3">
          <MemoryBars gpus={gpus} memory={memory} liveVramByGpu={liveVramByGpu} selected={value.gpus} note={memoryNote} />
        </div>
      </div>
      {cap?.applicable && cap.arch && (
        <div className="mt-5 border-t border-border pt-4">
          <SlotsChart
            cap={cap}
            gpus={gpus}
            value={value}
            onChange={onChange}
            readOnly={readOnly}
            measured={measured ?? null}
            staticMb={memory?.length ? memory.reduce((a, m) => a + Object.entries(m.parts).filter(([k]) => k !== "kv").reduce((x, [, v]) => x + v, 0), 0) : null}
          />
        </div>
      )}
      {footer && <div className="mt-4 border-t border-border pt-4">{footer}</div>}
    </Panel>
  )
}

// ── Contexto ────────────────────────────────────────────────────────────────

function ContextKnobs({
  value,
  onChange,
  readOnly,
  native,
  allowMax,
  poolEditable,
}: {
  value: WorkbenchState
  onChange: (patch: Partial<WorkbenchState>) => void
  readOnly: boolean
  native: number | null
  allowMax: boolean
  poolEditable: boolean
}) {
  const total = value.pool ? value.context : value.context * value.slots
  return (
    <div className="space-y-4">
      <div>
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <p className="text-xs font-medium">{value.pool ? "Contexto del pool (compartido)" : "Contexto por petición"}</p>
          <div className="flex items-center gap-2">
            {allowMax && (
              <Segmented
                size="sm"
                disabled={readOnly}
                value={value.maxMode ? "max" : "fixed"}
                options={[
                  { value: "fixed", label: "Fijo" },
                  { value: "max", label: "Lo que quepa", title: "El planificador elige el mayor que cabe" },
                ]}
                onChange={(v) => onChange({ maxMode: v === "max" })}
              />
            )}
            <span className="text-2xl font-semibold tabular-nums" data-testid="ctx-value">
              {value.maxMode ? "máx." : fmtContext(value.context)}
            </span>
          </div>
        </div>
        {!value.maxMode && (
          <ContextSlider
            value={value.context}
            max={Math.max(native ?? 0, value.context, 131072)}
            native={native}
            onChange={(n) => onChange({ context: n })}
            disabled={readOnly}
            testId="ctx-slider"
          />
        )}
        {!value.maxMode && native && value.context > native && (
          <Hint tone="warn">Por encima del contexto nativo ({fmtContext(native)}): el modelo degradará.</Hint>
        )}
      </div>
      <div className="flex flex-wrap items-start gap-6">
        <div>
          <p className="mb-1 text-xs font-medium">Peticiones simultáneas</p>
          <Segmented
            size="sm"
            disabled={readOnly}
            value={[1, 2, 4, 8, 16].includes(value.slots) ? value.slots : -1}
            options={[1, 2, 4, 8, 16].map((n) => ({ value: n, label: String(n) }))}
            onChange={(n) => onChange({ slots: n })}
          />
        </div>
        <div>
          <p className="mb-1 text-xs font-medium">Precisión de la KV</p>
          <Segmented
            size="sm"
            disabled={readOnly}
            value={value.kv === "bf16" ? "fp16" : value.kv === "fp8" ? "q8" : value.kv}
            options={KV_OPTIONS.map((k) => ({ value: k.value, label: k.label, title: k.hint }))}
            onChange={(v) => onChange({ kv: v })}
          />
        </div>
        <div>
          <p className="mb-1 text-xs font-medium">Memoria de contexto</p>
          <Segmented
            size="sm"
            disabled={readOnly || !poolEditable}
            value={value.pool ? "pool" : "slots"}
            options={[
              { value: "slots", label: "Por petición", title: "Cada petición tiene su trozo fijo" },
              { value: "pool", label: "Pool compartido", title: "Todas comparten un depósito común mientras quede sitio" },
            ]}
            onChange={(v) => onChange({ pool: v === "pool" })}
          />
          {!poolEditable && <Hint>Común a todos los peldaños.</Hint>}
        </div>
      </div>
      {!value.maxMode && (
        <Hint>
          {value.pool
            ? `${value.slots} peticiones comparten ${fmtContext(total)}: una sola puede usarlo entero si las demás no lo ocupan.`
            : `${value.slots} × ${fmtContext(value.context)} = ${fmtContext(total)} de KV reservados.`}
        </Hint>
      )}
    </div>
  )
}

// ── Memoria ─────────────────────────────────────────────────────────────────

const LEGEND_ORDER = ["weights", "kv", "rs_cache", "draft", "fixed", "overhead", "engine", "model"]

export function MemoryBars({
  gpus,
  memory,
  liveVramByGpu,
  selected,
  note,
}: {
  gpus: GPUState[]
  memory: MemoryShare[] | null
  liveVramByGpu: Record<string, number>
  selected?: number[] | null
  note?: ReactNode
}) {
  const shares = new Map((memory ?? []).map((m) => [m.gpu, m]))
  const parts = new Set((memory ?? []).flatMap((m) => Object.keys(m.parts)))
  const verdict = (() => {
    if (!memory?.length) return null
    let fitsNow = true
    let fitsEmpty = true
    for (const m of memory) {
      const g = gpus.find((x) => x.index === m.gpu)
      if (!g) continue
      const mb = Object.values(m.parts).reduce((a, b) => a + b, 0)
      const others = Math.max(0, g.usedVramMb - (liveVramByGpu[String(g.index)] ?? 0))
      if (mb > g.totalVramMb - others) fitsNow = false
      if (mb > g.totalVramMb) fitsEmpty = false
    }
    return fitsNow ? ("ok" as const) : fitsEmpty ? ("warn" as const) : ("error" as const)
  })()
  return (
    <div className="space-y-2.5 rounded-xl border border-border bg-muted/10 p-3" data-testid="memory-bars">
      <div className="flex items-center justify-between text-xs">
        <span className="font-medium">Memoria por GPU</span>
        {memory?.length ? (
          <span className="tabular-nums text-muted-foreground">
            total {fmtMb(memory.reduce((a, m) => a + Object.values(m.parts).reduce((x, y) => x + y, 0), 0))}
          </span>
        ) : null}
      </div>
      {gpus.map((g) => {
        const share = shares.get(g.index)
        const total = Math.max(1, g.totalVramMb)
        const own = liveVramByGpu[String(g.index)] ?? 0
        const others = Math.max(0, g.usedVramMb - own)
        const mb = share ? Object.values(share.parts).reduce((a, b) => a + b, 0) : 0
        const free = total - others
        const pct = (v: number) => `${Math.min(100, (v / total) * 100)}%`
        const dim = selected != null && !selected.includes(g.index) && !share
        return (
          <div key={g.index} className={cn("space-y-1", dim && "opacity-40")} data-gpu-bar={g.index}>
            <div className="flex items-center justify-between text-[11px]">
              <span className="font-medium" style={{ color: gpuColor(g.index) }}>
                #{g.index} {shortGpuName(g.name)}
              </span>
              <span className="tabular-nums text-muted-foreground">
                {share ? (
                  <>
                    <span className={mb <= free ? "text-emerald-300" : mb <= total ? "text-amber-300" : "text-red-300"}>{fmtMb(mb)}</span> de {fmtMb(free)} libres
                  </>
                ) : (
                  <>{fmtMb(free)} libres</>
                )}
              </span>
            </div>
            <div className="relative flex h-4 overflow-hidden rounded-md bg-muted/40">
              <div className="h-full bg-slate-600/70" style={{ width: pct(others) }} title={`Otros modelos: ${fmtMb(others)}`} />
              {share &&
                LEGEND_ORDER.filter((p) => share.parts[p]).map((p) => (
                  <div
                    key={p}
                    className="h-full border-l border-black/20"
                    style={{ width: pct(share.parts[p]), background: PART_META[p]?.color ?? "#a78bfa" }}
                    title={`${PART_META[p]?.label ?? p}: ${fmtMb(share.parts[p])}`}
                  />
                ))}
              {share && mb > free && <div className="absolute inset-y-0 right-0 w-1 bg-red-500" />}
            </div>
          </div>
        )
      })}
      <div className="flex flex-wrap gap-x-3 gap-y-1 border-t border-border/60 pt-2 text-[10.5px] text-muted-foreground" data-testid="memory-legend">
        <span className="flex items-center gap-1">
          <span className="h-2 w-2 rounded-sm bg-slate-600/70" /> otros modelos
        </span>
        {LEGEND_ORDER.filter((p) => parts.has(p)).map((p) => (
          <span key={p} className="flex items-center gap-1">
            <span className="h-2 w-2 rounded-sm" style={{ background: PART_META[p]?.color ?? "#a78bfa" }} />
            {PART_LABEL[p] ?? PART_META[p]?.label ?? p}
          </span>
        ))}
        <span className="flex items-center gap-1">
          <span className="h-2 w-2 rounded-sm bg-muted/60" /> libre
        </span>
      </div>
      {verdict && (
        <Callout tone={verdict}>
          {verdict === "ok" ? "Cabe con lo que hay cargado ahora." : verdict === "warn" ? "Cabe, pero desalojando otros modelos." : "No cabe en estas GPUs."}
        </Callout>
      )}
      {!memory?.length && <Hint>Sin estimación todavía.</Hint>}
      {note}
    </div>
  )
}

const PART_LABEL: Record<string, string> = {
  weights: "pesos del modelo",
  kv: "KV (contexto)",
  rs_cache: "estado recurrente",
  draft: "decodificación especulativa",
  fixed: "fijo de la arquitectura",
  overhead: "buffers de CUDA",
  engine: "reserva del motor",
  model: "modelo",
}

// ── Gráfica: contexto × peticiones simultáneas ──────────────────────────────

const SLOT_SERIES = [1, 2, 4, 8, 16]
const SERIES_COLORS = ["#38bdf8", "#22c55e", "#a78bfa", "#f59e0b", "#f472b6"]
const MARKS = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]

/** VRAM según el contexto por petición, una curva por número de peticiones
 *  simultáneas, con el techo de las GPUs elegidas: donde se cruzan está el
 *  contexto máximo. Pulsar una curva elige ese número de peticiones; pulsar
 *  la gráfica, el contexto. */
export function SlotsChart({
  cap,
  gpus,
  value,
  onChange,
  readOnly,
  measured,
  staticMb,
}: {
  cap: CapacityInfo
  gpus: GPUState[]
  value: WorkbenchState
  onChange: (patch: Partial<WorkbenchState>) => void
  readOnly: boolean
  measured: number | null
  /** Lo que no es KV según la estimación (pesos, draft, fijos…): así la
   *  gráfica y las barras cuentan lo mismo. */
  staticMb?: number | null
}) {
  const ref = useRef<SVGSVGElement>(null)
  const W = 760
  const H = 230
  const P = { l: 52, r: 120, t: 12, b: 26 }
  const arch = cap.arch!
  const kvPer1k = (arch.kv_mb_per_1k_tokens / KV_BYTES[(cap.kv_dtype as KvLabel) ?? "fp16"] || arch.kv_mb_per_1k_tokens) * KV_BYTES[value.kv]
  const base = staticMb && staticMb > 0 ? staticMb : (cap.weights_mb ?? 0) + (cap.overhead_mb ?? 0)
  const vram = (ctx: number, slots: number) => base + (kvPer1k * ctx * (value.pool ? 1 : slots)) / 1000
  const chosen = gpus.filter((g) => (value.gpus ?? []).includes(g.index))
  const singleBig = gpus.reduce((a, g) => Math.max(a, g.totalVramMb), 0)
  const limit = chosen.length ? chosen.reduce((a, g) => a + g.totalVramMb, 0) : singleBig
  const limitLabel = chosen.length
    ? chosen.map((g) => `#${g.index}`).join(" + ")
    : `${gpus.length > 1 ? "la mayor GPU" : "la GPU"}`
  const both = gpus.reduce((a, g) => a + g.totalVramMb, 0)
  const xMax = Math.max(arch.native_context || 0, value.context, 32768)
  const series = value.pool ? [value.slots] : SLOT_SERIES
  const yMax = Math.max(both, vram(Math.min(xMax, 262144), value.pool ? 1 : 4)) * 1.02
  const lo = Math.log2(2048)
  const hi = Math.log2(xMax)
  const lx = (c: number) => P.l + ((Math.log2(Math.max(c, 2048)) - lo) / (hi - lo)) * (W - P.l - P.r)
  const ly = (mb: number) => H - P.b - (Math.min(mb, yMax) / yMax) * (H - P.t - P.b)
  const maxFor = (slots: number) => {
    const room = limit - base
    if (room <= 0) return 0
    return Math.floor((room * 1000) / (kvPer1k * (value.pool ? 1 : slots)))
  }
  const paths = useMemo(
    () =>
      series.map((s) => {
        const pts: string[] = []
        for (let e = lo; e <= hi + 0.001; e += 0.1) {
          const c = 2 ** e
          pts.push(`${pts.length ? "L" : "M"}${lx(c).toFixed(1)},${ly(vram(c, s)).toFixed(1)}`)
        }
        return { s, d: pts.join(" ") }
      }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [cap, value.pool, value.kv, value.slots, xMax, yMax],
  )
  const clickChart = (e: React.MouseEvent<SVGSVGElement>) => {
    if (readOnly || value.maxMode || !ref.current) return
    const rect = ref.current.getBoundingClientRect()
    const x = ((e.clientX - rect.left) / rect.width) * W
    if (x < P.l || x > W - P.r) return
    const ctx = 2 ** (lo + ((x - P.l) / (W - P.l - P.r)) * (hi - lo))
    const snap = MARKS.find((m) => Math.abs(Math.log2(m) - Math.log2(ctx)) < 0.12)
    onChange({ context: snap ?? Math.round(ctx / 1024) * 1024 })
  }
  const cur = vram(value.context, value.slots)
  return (
    <div className="space-y-1" data-testid="slots-chart">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <p className="text-xs font-medium">Contexto y peticiones simultáneas</p>
        <p className="text-[11px] text-muted-foreground">
          {fmtMb(base)} fijos (pesos y demás) · {kvPer1k.toFixed(1)} MB de KV por 1k tokens ({value.kv}) · pulsa una curva o un punto para elegir
        </p>
      </div>
      <svg ref={ref} viewBox={`0 0 ${W} ${H}`} className={cn("w-full", !readOnly && "cursor-crosshair")} onClick={clickChart} role="img" aria-label="VRAM según el contexto y las peticiones simultáneas">
        {[0.25, 0.5, 0.75, 1].map((f) => (
          <g key={f}>
            <line x1={P.l} x2={W - P.r} y1={ly(yMax * f)} y2={ly(yMax * f)} className="stroke-border" strokeDasharray="2 3" />
            <text x={P.l - 5} y={ly(yMax * f) + 3} textAnchor="end" className="fill-muted-foreground text-[9px]">
              {fmtMb(yMax * f)}
            </text>
          </g>
        ))}
        {MARKS.filter((m) => m <= xMax && m >= 4096).map((m) => (
          <text key={m} x={lx(m)} y={H - 8} textAnchor="middle" className="fill-muted-foreground text-[9px]">
            {fmtContext(m)}
          </text>
        ))}
        <rect x={P.l} y={P.t} width={W - P.l - P.r} height={Math.max(0, ly(limit) - P.t)} className="fill-red-500/5" />
        <line x1={P.l} x2={W - P.r} y1={ly(limit)} y2={ly(limit)} stroke="#f87171" strokeWidth={1.5} />
        <text x={W - P.r + 4} y={ly(limit) + 3} className="fill-red-300 text-[9.5px]">
          techo {limitLabel}
        </text>
        {chosen.length === 0 && gpus.length > 1 && (
          <g>
            <line x1={P.l} x2={W - P.r} y1={ly(both)} y2={ly(both)} stroke="#a3a3a3" strokeDasharray="5 3" />
            <text x={W - P.r + 4} y={ly(both) + 3} className="fill-muted-foreground text-[9.5px]">
              las {gpus.length} GPUs
            </text>
          </g>
        )}
        {measured ? (
          <g>
            <line x1={lx(measured)} x2={lx(measured)} y1={P.t} y2={H - P.b} stroke="#c084fc" strokeDasharray="3 3" />
            <text x={lx(measured) + 3} y={P.t + 9} className="fill-violet-300 text-[9.5px]">
              medido {fmtContext(measured)}
            </text>
          </g>
        ) : null}
        {paths.map(({ s, d }, i) => {
          const active = s === value.slots
          const color = value.pool ? SERIES_COLORS[0] : SERIES_COLORS[i % SERIES_COLORS.length]
          const m = maxFor(s)
          return (
            <g key={s} className={cn(!readOnly && !value.pool && "cursor-pointer")}>
              <path
                d={d}
                fill="none"
                stroke={color}
                strokeWidth={active ? 3 : 1.5}
                opacity={active ? 1 : 0.55}
                onClick={(e) => {
                  e.stopPropagation()
                  if (!readOnly && !value.pool) onChange({ slots: s })
                }}
              />
              <path d={d} fill="none" stroke="transparent" strokeWidth={10} onClick={(e) => {
                e.stopPropagation()
                if (!readOnly && !value.pool) onChange({ slots: s })
              }} />
              {m > 0 && m <= xMax * 1.01 && (
                <circle cx={lx(m)} cy={ly(limit)} r={3} fill={color} />
              )}
              <text x={W - P.r + 4} y={H - P.b - (series.length - 1 - i) * 13} className="text-[10px]" fill={color} fontWeight={active ? 700 : 400}>
                {value.pool ? `pool · ${s} a la vez` : `${s} a la vez`} → {m > 0 ? fmtContext(Math.min(m, arch.native_context || m)) : "no cabe"}
              </text>
            </g>
          )
        })}
        {!value.maxMode && (
          <g>
            <circle cx={lx(value.context)} cy={ly(cur)} r={6} className={cur > limit ? "fill-red-400" : "fill-emerald-400"} stroke="white" strokeWidth={1.5} />
            <text
              x={lx(value.context) > (W - P.r) * 0.75 ? lx(value.context) - 9 : lx(value.context) + 9}
              textAnchor={lx(value.context) > (W - P.r) * 0.75 ? "end" : "start"}
              y={ly(cur) + 16}
              className="fill-foreground text-[10px] font-medium"
            >
              {fmtContext(value.context)} × {value.pool ? "pool" : value.slots} · {fmtMb(cur)}
            </text>
          </g>
        )}
      </svg>
      <Hint>
        Cada curva es la memoria que pide ese número de peticiones simultáneas según el contexto de cada una; el punto donde cruza el techo
        rojo es el contexto máximo que cabe. Estimación del planificador.
      </Hint>
    </div>
  )
}
