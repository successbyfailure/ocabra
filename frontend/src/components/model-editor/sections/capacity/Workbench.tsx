import type { ReactNode } from "react"
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
 *  memoria y debajo cuánto contexto cabe según las peticiones a la vez; a la
 *  derecha la memoria por GPU con su leyenda y, debajo, las pruebas. */
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
  actions,
  planned,
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
  /** Pruebas del peldaño: van debajo de la memoria, junto a lo que miden. */
  actions?: ReactNode
  /** Máximo que da el planificador con lo elegido (con margen y medidas):
   *  manda sobre la estimación simple en la fila elegida. */
  planned?: number | null
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
          {cap?.applicable && cap.arch && (
            <SlotsChart
              cap={cap}
              gpus={gpus}
              value={value}
              onChange={onChange}
              readOnly={readOnly}
              measured={measured ?? null}
              planned={planned ?? null}
              estimateMb={memory?.length ? memory.reduce((a, m) => a + Object.values(m.parts).reduce((x, v) => x + v, 0), 0) : null}
              staticMb={memory?.length ? memory.reduce((a, m) => a + Object.entries(m.parts).filter(([k]) => k !== "kv").reduce((x, [, v]) => x + v, 0), 0) : null}
            />
          )}
        </div>
        <div className="min-w-0 flex-[1_1_280px] space-y-3">
          <MemoryBars gpus={gpus} memory={memory} liveVramByGpu={liveVramByGpu} selected={value.gpus} note={memoryNote} />
          {actions}
        </div>
      </div>
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

// ── ¿Cuánto contexto cabe? ──────────────────────────────────────────────────

const SLOT_SERIES = [1, 2, 4, 8, 16]
const MARKS = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]

/** Responde a una pregunta: con N peticiones a la vez, ¿hasta cuánto contexto
 *  puede tener cada una en estas GPUs? Una fila por N, con una barra de
 *  contexto (escala logarítmica) que se llena hasta lo que cabe y una marca
 *  donde está lo elegido. Pulsar una fila elige N; pulsar en su barra, también
 *  el contexto. Con pool compartido el contexto es el total de todas. */
export function SlotsChart({
  cap,
  gpus,
  value,
  onChange,
  readOnly,
  measured,
  planned,
  estimateMb,
  staticMb,
}: {
  cap: CapacityInfo
  gpus: GPUState[]
  value: WorkbenchState
  onChange: (patch: Partial<WorkbenchState>) => void
  readOnly: boolean
  measured: number | null
  planned?: number | null
  /** Memoria total de lo elegido según la vista previa (lo mismo que las barras). */
  estimateMb?: number | null
  /** Lo que no es KV según la estimación (pesos, draft, fijos…): así la
   *  tabla y las barras de memoria cuentan lo mismo. */
  staticMb?: number | null
}) {
  const arch = cap.arch!
  const kvPer1k = (arch.kv_mb_per_1k_tokens / KV_BYTES[(cap.kv_dtype as KvLabel) ?? "fp16"] || arch.kv_mb_per_1k_tokens) * KV_BYTES[value.kv]
  const base = staticMb && staticMb > 0 ? staticMb : (cap.weights_mb ?? 0) + (cap.overhead_mb ?? 0)
  const chosen = gpus.filter((g) => (value.gpus ?? []).includes(g.index))
  const limit = chosen.length ? chosen.reduce((a, g) => a + g.totalVramMb, 0) : gpus.reduce((a, g) => Math.max(a, g.totalVramMb), 0)
  const limitLabel = chosen.length ? chosen.map((g) => `#${g.index} ${shortGpuName(g.name)}`).join(" + ") : "la mayor GPU"
  const native = arch.native_context || null
  const xMax = Math.max(native ?? 0, value.context, 32768)
  const lo = Math.log2(2048)
  const hi = Math.log2(xMax)
  const pos = (c: number) => Math.min(1, Math.max(0, (Math.log2(Math.max(c, 2048)) - lo) / (hi - lo)))
  /** Contexto por petición (o total, con pool) que cabe con N a la vez. */
  const maxFor = (slots: number) => {
    const room = limit - base
    if (room <= 0 || kvPer1k <= 0) return 0
    const fit = Math.floor((room * 1000) / (kvPer1k * (value.pool ? 1 : slots)))
    return native ? Math.min(fit, native * (value.pool ? slots : 1)) : fit
  }
  // La memoria de lo elegido: la de la vista previa (igual que las barras)
  // cuando la hay; si no, la estimación simple.
  const usedMb = estimateMb && estimateMb > 0 ? estimateMb : base + (kvPer1k * value.context * (value.pool ? 1 : value.slots)) / 1000
  const fits = planned ? value.context <= planned : usedMb <= limit
  const rows = value.pool ? [value.slots] : Array.from(new Set([...SLOT_SERIES, value.slots])).sort((a, b) => a - b)
  const pick = (slots: number, e?: React.MouseEvent<HTMLDivElement>) => {
    if (readOnly) return
    const patch: Partial<WorkbenchState> = value.pool ? {} : { slots }
    if (e && !value.maxMode) {
      const rect = e.currentTarget.getBoundingClientRect()
      const ctx = 2 ** (lo + ((e.clientX - rect.left) / rect.width) * (hi - lo))
      const snap = MARKS.find((m) => Math.abs(Math.log2(m) - Math.log2(ctx)) < 0.15)
      patch.context = snap ?? Math.max(2048, Math.round(ctx / 1024) * 1024)
    }
    onChange(patch)
  }
  return (
    <div className="space-y-2" data-testid="slots-chart">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <p className="text-xs font-medium">
          {value.pool ? "¿Qué tamaño de pool cabe?" : "¿Cuánto contexto cabe para cada petición?"}
        </p>
        <p className="text-[11px] text-muted-foreground">en {limitLabel} · KV {value.kv} · estimación</p>
      </div>
      {!value.maxMode && (
        <p className={cn("text-[11.5px]", fits ? "text-emerald-300" : "text-red-300")} data-testid="slots-verdict">
          {value.pool
            ? `Lo elegido: pool de ${fmtContext(value.context)} para ${value.slots} a la vez → ${fmtMb(usedMb)} de ${fmtMb(limit)}`
            : `Lo elegido: ${value.slots} a la vez × ${fmtContext(value.context)} cada una → ${fmtMb(usedMb)} de ${fmtMb(limit)}`}
          {fits ? " · cabe" : " · no cabe"}
        </p>
      )}
      {value.maxMode && planned ? (
        <p className="text-[11.5px] text-emerald-300" data-testid="slots-verdict">
          Lo que quepa: el planificador le da {fmtContext(planned)}
          {value.pool ? " de pool" : " por petición"} con {value.slots} a la vez.
        </p>
      ) : null}
      <div className="space-y-1.5">
        {rows.map((s) => {
          const active = s === value.slots
          // En la fila elegida manda el planificador (margen de seguridad y
          // contexto medido); las demás son la estimación simple.
          const max = active && planned ? planned : maxFor(s)
          return (
            <div
              key={s}
              data-slots={s}
              style={{ gridTemplateColumns: value.pool ? "40px minmax(0,1fr) 120px" : "76px minmax(0,1fr) 104px" }}
              className={cn("grid items-center gap-2 rounded-md px-1.5 py-1 text-[11.5px]", active && "bg-primary/10 ring-1 ring-primary/40", !readOnly && !value.pool && "cursor-pointer hover:bg-muted/30")}
              onClick={() => !value.pool && pick(s)}
            >
              <span className={cn("tabular-nums", active ? "font-semibold" : "text-muted-foreground")}>
                {value.pool ? "pool" : `${s} a la vez`}
              </span>
              <div
                className="relative h-4 overflow-hidden rounded bg-red-500/10"
                onClick={(e) => {
                  e.stopPropagation()
                  pick(s, e)
                }}
                title="Pulsa para elegir ese contexto"
              >
                <div className="absolute inset-y-0 left-0 bg-emerald-500/35" style={{ width: `${pos(max) * 100}%` }} />
                {native && !value.pool && <div className="absolute inset-y-0 w-px bg-white/40" style={{ left: `${pos(native) * 100}%` }} title="Contexto nativo del modelo" />}
                {measured && active && <div className="absolute inset-y-0 w-0.5 bg-violet-400" style={{ left: `${pos(measured) * 100}%` }} title={`Medido: ${fmtContext(measured)}`} />}
                {active && !value.maxMode && (
                  <div
                    className={cn("absolute top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full border border-white", fits ? "bg-emerald-400" : "bg-red-400")}
                    style={{ left: `${pos(value.context) * 100}%` }}
                    data-testid="slots-choice"
                  />
                )}
              </div>
              <span className={cn("text-right tabular-nums", max > 0 ? "" : "text-red-300")}>
                {max > 0 ? `hasta ${fmtContext(max)}${value.pool ? " en total" : ""}` : "no cabe"}
                {active && planned ? <span className="block text-[9.5px] text-muted-foreground">planificador</span> : null}
              </span>
            </div>
          )
        })}
        <div className="grid gap-2 px-1.5" style={{ gridTemplateColumns: value.pool ? "40px minmax(0,1fr) 120px" : "76px minmax(0,1fr) 104px" }}>
          <span />
          <div className="relative h-3 text-[9.5px] text-muted-foreground">
            {MARKS.filter((m) => m <= xMax && m >= 4096).map((m) => (
              <span key={m} className="absolute -translate-x-1/2" style={{ left: `${pos(m) * 100}%` }}>
                {fmtContext(m)}
              </span>
            ))}
          </div>
          <span />
        </div>
      </div>
      {value.pool ? (
        <Hint>
          Con KV compartida no hay contexto por petición: todas tiran de un mismo depósito y una sola puede usarlo entero si las demás no lo
          ocupan. Por eso aquí solo importa el tamaño del pool; cuántas van a la vez apenas cambia la memoria (solo el estado por slot de los
          modelos híbridos y el borrador). Verde: el pool que cabe en {limitLabel}
          {planned ? ", según el planificador (con su margen de seguridad y lo medido)" : ""}.
          {measured ? " La raya violeta es lo medido con el sondeo." : ""} Pulsa la barra para elegir el tamaño del pool.
        </Hint>
      ) : (
        <Hint>
          Verde: el contexto que cabe con ese número de peticiones a la vez; rojo: no cabe.
          {native ? " La raya blanca es el contexto nativo del modelo (más allá degrada)." : ""}
          {measured ? " La violeta, lo medido de verdad con el sondeo." : ""} Pulsa una fila para elegir cuántas a la vez, o su barra para elegir
          también el contexto. {fmtMb(base)} son fijos (pesos y demás) y cada 1k tokens de contexto ocupa {kvPer1k.toFixed(1)} MB.
        </Hint>
      )}
    </div>
  )
}
