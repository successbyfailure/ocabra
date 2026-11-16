import type { ReactNode } from "react"
import { gpuColor } from "@/components/dashboard/GpuBadge"
import { shortGpuName } from "@/components/dashboard/gpuNames"
import { fmtContext } from "@/components/dashboard/format"
import { cn } from "@/lib/cn"
import type { GPUState } from "@/types"
import { PART_META, fmtMb } from "../../PreviewPanel"
import type { CapacityInfo } from "../../types"
import { Callout, Hint, Panel, Segmented } from "../../ui"
import { HATCH_IMPOSSIBLE, HATCH_SYSTEM, gpuOccupancy } from "./occupancy"

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

const sumParts = (m: MemoryShare) => Object.values(m.parts).reduce((a, b) => a + b, 0)

/** Banco de trabajo de Capacidad: a la izquierda el contexto (cuánto se elige
 *  y cuánto cabe, en una sola barra) y los controles que cambian la memoria;
 *  a la derecha la memoria por GPU con su leyenda y, debajo, las pruebas. */
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
   *  manda sobre la estimación simple con las peticiones elegidas. */
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
          <ContextKnobs
            value={value}
            onChange={onChange}
            readOnly={readOnly}
            native={native}
            allowMax={allowMax}
            poolEditable={poolEditable}
            cap={cap?.applicable && cap.arch ? cap : null}
            gpus={gpus}
            measured={measured ?? null}
            planned={planned ?? null}
            estimateMb={memory?.length ? memory.reduce((a, m) => a + sumParts(m), 0) : null}
            // Solo con desglose real: un peldaño medido trae un único "measured"
            // (KV incluida) y no se puede separar lo fijo de la KV.
            usableMb={Object.fromEntries(gpus.map((g) => [g.index, g.totalVramMb - gpuOccupancy(g, liveVramByGpu[String(g.index)] ?? 0).system]))}
            // Las GPUs de la cuenta: las elegidas o, si no hay, las que usa la
            // estimación (puede repartirse entre varias aunque no sea "a la vez").
            targetGpus={value.gpus?.length ? value.gpus : memory?.length ? memory.map((m) => m.gpu) : null}
            staticMb={memory?.some((m) => "kv" in m.parts) ? memory.reduce((a, m) => a + Object.entries(m.parts).filter(([k]) => k !== "kv").reduce((x, [, v]) => x + v, 0), 0) : null}
          />
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

const SLOT_SERIES = [1, 2, 4, 8, 16]
const MARKS = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]

/** Cuánto contexto cabe según la arquitectura y la memoria de las GPUs. */
function fitModel(
  cap: CapacityInfo,
  gpus: GPUState[],
  value: WorkbenchState,
  staticMb: number | null,
  targetGpus: number[] | null,
  usableMb: Record<number, number>,
) {
  const arch = cap.arch!
  const kvPer1k = (arch.kv_mb_per_1k_tokens / KV_BYTES[(cap.kv_dtype as KvLabel) ?? "fp16"] || arch.kv_mb_per_1k_tokens) * KV_BYTES[value.kv]
  const base = staticMb && staticMb > 0 ? staticMb : (cap.weights_mb ?? 0) + (cap.overhead_mb ?? 0)
  // Lo que puede usar un modelo en cada GPU: la tarjeta menos lo del sistema
  // (igual que el rayado de las barras de memoria).
  const usable = (g: GPUState) => usableMb[g.index] ?? g.totalVramMb
  const chosen = gpus.filter((g) => (targetGpus ?? []).includes(g.index))
  const limit = chosen.length ? chosen.reduce((a, g) => a + usable(g), 0) : gpus.reduce((a, g) => Math.max(a, usable(g)), 0)
  const limitLabel = chosen.length ? chosen.map((g) => `#${g.index} ${shortGpuName(g.name)}`).join(" + ") : "la mayor GPU"
  const native = arch.native_context || null
  /** Contexto por petición (o total, con pool) que cabe en memoria con N a
   *  la vez; con `capNative`, sin pasar del nativo (más allá degrada). */
  const maxFor = (slots: number, capNative = true) => {
    const room = limit - base
    if (room <= 0 || kvPer1k <= 0) return 0
    const fit = Math.floor((room * 1000) / (kvPer1k * (value.pool ? 1 : slots)))
    return native && capNative ? Math.min(fit, native * (value.pool ? slots : 1)) : fit
  }
  /** Memoria con un contexto dado, según la misma cuenta. */
  const mbFor = (context: number, slots: number) => base + (kvPer1k * context * (value.pool ? 1 : slots)) / 1000
  return { kvPer1k, base, limit, limitLabel, maxFor, mbFor }
}

function ContextKnobs({
  value,
  onChange,
  readOnly,
  native,
  allowMax,
  poolEditable,
  cap,
  gpus,
  measured,
  planned,
  estimateMb,
  staticMb,
  targetGpus,
  usableMb,
}: {
  value: WorkbenchState
  onChange: (patch: Partial<WorkbenchState>) => void
  readOnly: boolean
  native: number | null
  allowMax: boolean
  poolEditable: boolean
  /** Solo si sabe calcular cuánto cabe (arquitectura conocida). */
  cap: CapacityInfo | null
  gpus: GPUState[]
  measured: number | null
  planned: number | null
  /** Memoria total de lo elegido según la vista previa (lo mismo que las barras). */
  estimateMb: number | null
  /** Lo que no es KV según la estimación (pesos, draft, fijos…): así el
   *  contexto y las barras de memoria cuentan lo mismo. */
  staticMb: number | null
  /** GPUs sobre las que se cuenta lo que cabe; null = la mayor. */
  targetGpus: number[] | null
  /** MB que un modelo puede usar en cada GPU (sin lo del sistema). */
  usableMb: Record<number, number>
}) {
  return (
    <div className="space-y-4">
      <ContextControl
        value={value}
        onChange={onChange}
        readOnly={readOnly}
        native={native}
        allowMax={allowMax}
        cap={cap}
        gpus={gpus}
        measured={measured}
        planned={planned}
        estimateMb={estimateMb}
        staticMb={staticMb}
        targetGpus={targetGpus}
        usableMb={usableMb}
      />
      <div className="flex flex-wrap items-start gap-6">
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
    </div>
  )
}

/** Un solo control para el contexto: lo elegido (el tirador), lo que cabe
 *  (verde) y lo que no puede funcionar (rayado rojo), en la misma barra con
 *  escala logarítmica. Debajo, cuántas peticiones a la vez, cada una con el
 *  contexto que cabría así. Con pool compartido el contexto es el total. */
export function ContextControl({
  value,
  onChange,
  readOnly,
  native,
  allowMax,
  cap,
  gpus,
  measured,
  planned,
  estimateMb,
  staticMb,
  targetGpus,
  usableMb,
}: {
  value: WorkbenchState
  onChange: (patch: Partial<WorkbenchState>) => void
  readOnly: boolean
  native: number | null
  allowMax: boolean
  cap: CapacityInfo | null
  gpus: GPUState[]
  measured: number | null
  planned: number | null
  estimateMb: number | null
  staticMb: number | null
  /** GPUs sobre las que se cuenta lo que cabe; null = la mayor. */
  targetGpus: number[] | null
  /** MB que un modelo puede usar en cada GPU (sin lo del sistema). */
  usableMb: Record<number, number>
}) {
  const model = cap ? fitModel(cap, gpus, value, staticMb, targetGpus, usableMb) : null
  const nativeCtx = native ?? cap?.arch?.native_context ?? null
  const xMax = Math.max(nativeCtx ?? 0, value.context, 131072)
  const lo = Math.log2(2048)
  const hi = Math.log2(xMax)
  const pos = (c: number) => Math.min(1, Math.max(0, (Math.log2(Math.max(c, 2048)) - lo) / (hi - lo)))
  const marks = MARKS.filter((m) => m <= xMax && m >= 4096)
  // Con las peticiones elegidas manda el planificador (margen de seguridad y
  // contexto medido); para las demás, la estimación simple.
  const maxFor = (slots: number): number | null => (slots === value.slots && planned ? planned : model ? model.maxFor(slots) : null)
  // Verde hasta lo que cabe sin pasar del nativo; rayado desde donde ya no
  // cabe en memoria (entre los dos, cabe pero degrada).
  const fit = maxFor(value.slots)
  const memFit = planned ? planned : model ? model.maxFor(value.slots, false) : null
  const total = value.pool ? value.context : value.context * value.slots
  // Lo que se enseña es la vista previa (igual que las barras); el veredicto,
  // la misma cuenta que la barra, para que el tirador y el rayado no se
  // contradigan mientras llega la vista previa nueva.
  const usedMb = model ? (estimateMb && estimateMb > 0 ? estimateMb : model.mbFor(value.context, value.slots)) : null
  const fits = planned ? value.context <= planned : model ? model.mbFor(value.context, value.slots) <= model.limit : null
  const slotRows = value.pool ? SLOT_SERIES : Array.from(new Set([...SLOT_SERIES, value.slots])).sort((a, b) => a - b)
  const pickContext = (raw: number) => {
    const snap = marks.find((m) => Math.abs(Math.log2(m) - Math.log2(raw)) < 0.06)
    onChange({ context: snap ?? Math.max(2048, Math.round(raw / 1024) * 1024) })
  }
  return (
    <div className="space-y-2" data-testid="context-control">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div>
          <p className="text-xs font-medium">{value.pool ? "Contexto del pool (compartido)" : "Contexto por petición"}</p>
          {model && (
            <p className="text-[11px] text-muted-foreground">
              cabe en {model.limitLabel} · KV {value.kv} · estimación
            </p>
          )}
        </div>
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

      <div className="relative h-5 overflow-hidden rounded-md bg-muted/40" data-testid="context-bar">
        {fit != null && (
          <>
            <div className="absolute inset-y-0 left-0 bg-emerald-500/35" style={{ width: `${pos(fit) * 100}%` }} data-testid="context-fit" title={`Cabe hasta ${fmtContext(fit)}`} />
            {memFit != null && pos(memFit) < 1 && (
              <div
                className="absolute inset-y-0 right-0 border-l border-red-400/70"
                style={{ left: `${pos(memFit) * 100}%`, ...HATCH_IMPOSSIBLE }}
                data-testid="context-impossible"
                title={memFit > 0 ? `Más de ${fmtContext(memFit)} no cabe` : "No cabe"}
              />
            )}
          </>
        )}
        {nativeCtx && !value.pool && (
          <div className="absolute inset-y-0 w-0.5 bg-amber-400" style={{ left: `${pos(nativeCtx) * 100}%` }} title={`Contexto nativo: ${fmtContext(nativeCtx)}`} />
        )}
        {measured && <div className="absolute inset-y-0 w-0.5 bg-violet-400" style={{ left: `${pos(measured) * 100}%` }} title={`Medido: ${fmtContext(measured)}`} />}
        {!value.maxMode && (
          <div
            className={cn(
              "pointer-events-none absolute top-1/2 h-3.5 w-3.5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-white shadow",
              fits == null ? "bg-primary" : fits ? "bg-emerald-400" : "bg-red-400",
            )}
            style={{ left: `${pos(value.context) * 100}%` }}
            data-testid="context-choice"
          />
        )}
        {!value.maxMode && (
          <input
            type="range"
            min={lo}
            max={hi}
            step={0.01}
            value={Math.log2(Math.max(value.context, 2048))}
            disabled={readOnly}
            aria-label="Contexto"
            data-testid="ctx-slider"
            onChange={(e) => pickContext(2 ** Number(e.target.value))}
            className="absolute inset-0 h-full w-full cursor-pointer opacity-0 disabled:cursor-default"
          />
        )}
      </div>
      <div className="relative h-3 text-[9.5px] text-muted-foreground">
        {marks.map((m) => (
          <button
            key={m}
            type="button"
            disabled={readOnly || value.maxMode}
            onClick={() => onChange({ context: m })}
            className="absolute -translate-x-1/2 hover:text-foreground disabled:hover:text-muted-foreground"
            style={{ left: `${pos(m) * 100}%` }}
          >
            {fmtContext(m)}
          </button>
        ))}
      </div>

      {!value.maxMode && (fits != null || fit != null) && (
        <p className={cn("text-[11.5px]", fits === false ? "text-red-300" : "text-emerald-300")} data-testid="context-verdict">
          {model && usedMb != null
            ? value.pool
              ? `Pool de ${fmtContext(value.context)} para ${value.slots} a la vez → ${fmtMb(usedMb)} de ${fmtMb(model.limit)}`
              : `${value.slots} a la vez × ${fmtContext(value.context)} cada una → ${fmtMb(usedMb)} de ${fmtMb(model.limit)}`
            : `${fmtContext(value.context)}`}
          {fits ? " · cabe" : " · no cabe"}
          {fit != null && (fit > 0 ? ` · hasta ${fmtContext(fit)}${value.pool ? " en total" : ""}` : "")}
          {planned ? " (planificador)" : ""}
        </p>
      )}
      {value.maxMode && planned ? (
        <p className="text-[11.5px] text-emerald-300" data-testid="context-verdict">
          Lo que quepa: el planificador le da {fmtContext(planned)}
          {value.pool ? " de pool" : " por petición"} con {value.slots} a la vez.
        </p>
      ) : null}
      {!value.maxMode && nativeCtx && !value.pool && value.context > nativeCtx && (
        <Hint tone="warn">Por encima del contexto nativo ({fmtContext(nativeCtx)}): el modelo degradará.</Hint>
      )}

      <div className="pt-1">
        <p className="mb-1 text-xs font-medium">Peticiones simultáneas</p>
        <div className="inline-flex flex-wrap gap-1 rounded-lg border border-border bg-muted/30 p-0.5" data-testid="slots-picker">
          {slotRows.map((s) => {
            const active = s === value.slots
            const max = value.pool ? null : maxFor(s)
            return (
              <button
                key={s}
                type="button"
                data-slots={s}
                disabled={readOnly}
                aria-pressed={active}
                onClick={() => onChange({ slots: s })}
                title={max != null ? (max > 0 ? `Con ${s} a la vez caben hasta ${fmtContext(max)} por petición` : `Con ${s} a la vez no cabe`) : undefined}
                className={cn(
                  "min-w-[3.25rem] rounded-md px-2 py-0.5 text-center font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40",
                  active ? "bg-primary text-primary-foreground shadow-sm" : "text-muted-foreground hover:bg-muted hover:text-foreground",
                )}
              >
                <span className="block text-[11px] tabular-nums">{s}</span>
                {max != null && (
                  <span className={cn("block text-[9.5px] tabular-nums", active ? "opacity-90" : max > 0 ? "" : "text-red-300")}>
                    {max > 0 ? `≤ ${fmtContext(max)}` : "no cabe"}
                  </span>
                )}
              </button>
            )
          })}
        </div>
      </div>

      {!value.maxMode && (
        <Hint>
          {value.pool
            ? `${value.slots} peticiones comparten ${fmtContext(total)}: una sola puede usarlo entero si las demás no lo ocupan; cuántas van a la vez apenas cambia la memoria (solo el estado por slot de los modelos híbridos y el borrador).`
            : `${value.slots} × ${fmtContext(value.context)} = ${fmtContext(total)} de KV reservados. Debajo de cada número de peticiones, el contexto que cabría para cada una.`}
        </Hint>
      )}
      {fit != null && (
        <Hint>
          Verde: el contexto que cabe{model ? ` en ${model.limitLabel}` : ""}
          {planned ? ", según el planificador (con su margen de seguridad y lo medido)" : ""}; rayado rojo: no puede funcionar con esta configuración.
          {nativeCtx && !value.pool ? " La raya ámbar es el contexto nativo del modelo (más allá degrada)." : ""}
          {measured ? " La violeta, lo medido de verdad con el sondeo." : ""}
          {model ? ` ${fmtMb(model.base)} son fijos (pesos y demás) y cada 1k tokens de contexto ocupa ${model.kvPer1k.toFixed(1)} MB.` : ""}
        </Hint>
      )}
    </div>
  )
}

// ── Memoria ─────────────────────────────────────────────────────────────────

const LEGEND_ORDER = ["weights", "kv", "rs_cache", "draft", "fixed", "overhead", "engine", "model"]

/** Partes de una estimación en orden de leyenda; las que no conoce (p. ej.
 *  `measured` de un peldaño medido) van al final, para que siempre se pinten. */
const orderedParts = (keys: Iterable<string>) => {
  const set = new Set(keys)
  return [...LEGEND_ORDER.filter((p) => set.has(p)), ...[...set].filter((p) => !LEGEND_ORDER.includes(p)).sort()]
}

/** Memoria por GPU. Cada barra es una tarjeta: primero lo que reserva el
 *  sistema (rayado gris, ningún modelo puede usarlo), luego otros modelos
 *  cargados ahora y después este modelo por partes. Lo que se sale de la
 *  tarjeta, rayado rojo: así no cabe (desalojando otros, quizá sí). */
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
  const parts = orderedParts((memory ?? []).flatMap((m) => Object.keys(m.parts).filter((k) => m.parts[k] > 0)))
  const rows = gpus.map((g) => {
    const share = shares.get(g.index)
    const occ = gpuOccupancy(g, liveVramByGpu[String(g.index)] ?? 0)
    const total = Math.max(1, g.totalVramMb)
    const mb = share ? sumParts(share) : 0
    return {
      g,
      share,
      occ,
      total,
      mb,
      // Lo que no cabe ni desalojando a todos los demás.
      impossible: Math.max(0, occ.system + mb - total),
      // Lo que se sale de la tarjeta con lo que hay cargado ahora.
      overflow: share ? Math.max(0, occ.system + occ.others + mb - total) : 0,
    }
  })
  const anyOverflow = rows.some((r) => r.overflow > 0)
  const anyOthers = rows.some((r) => r.occ.others > 0)
  const verdict = (() => {
    if (!memory?.length) return null
    const used = rows.filter((r) => r.share)
    if (!used.length) return null
    if (used.some((r) => r.impossible > 0)) return "error" as const
    if (used.some((r) => r.overflow > 0)) return "warn" as const
    return "ok" as const
  })()
  return (
    <div className="space-y-2.5 rounded-xl border border-border bg-muted/10 p-3" data-testid="memory-bars">
      <div className="flex items-center justify-between text-xs">
        <span className="font-medium">Memoria por GPU</span>
        {memory?.length ? <span className="tabular-nums text-muted-foreground">total {fmtMb(memory.reduce((a, m) => a + sumParts(m), 0))}</span> : null}
      </div>
      {rows.map(({ g, share, occ, total, mb, impossible, overflow }) => {
        // Si no cabe, la escala crece para enseñar cuánto se sale de la tarjeta.
        const scale = total + overflow
        const pct = (v: number) => `${Math.max(0, Math.min(100, (v / scale) * 100))}%`
        const free = total - occ.system - occ.others
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
                    <span className={mb <= free ? "text-emerald-300" : impossible <= 0 ? "text-amber-300" : "text-red-300"}>{fmtMb(mb)}</span> de {fmtMb(free)} libres
                  </>
                ) : (
                  <>{fmtMb(free)} libres</>
                )}
              </span>
            </div>
            <div className="relative flex h-4 overflow-hidden rounded-md bg-muted/40">
              {occ.system > 0 && (
                <div className="h-full shrink-0" style={{ width: pct(occ.system), ...HATCH_SYSTEM }} title={`Sistema (driver, CUDA, escritorio y margen del planificador): ${fmtMb(occ.system)}`} data-part="system" />
              )}
              {occ.others > 0 && (
                <div
                  className="h-full shrink-0 border-l border-black/20 bg-slate-500/80"
                  style={{ width: pct(occ.others) }}
                  title={`Otros modelos cargados: ${fmtMb(occ.others)}`}
                  data-part="others"
                />
              )}
              {share &&
                orderedParts(Object.keys(share.parts).filter((k) => share.parts[k] > 0)).map((p) => (
                  <div
                    key={p}
                    className="h-full shrink-0 border-l border-black/20"
                    style={{ width: pct(share.parts[p]), background: PART_META[p]?.color ?? "#a78bfa" }}
                    title={`${PART_LABEL[p] ?? PART_META[p]?.label ?? p}: ${fmtMb(share.parts[p])}`}
                    data-part={p}
                  />
                ))}
              {overflow > 0 && (
                <div
                  className="absolute inset-y-0 right-0 border-l-2 border-red-400"
                  style={{ width: pct(overflow), ...HATCH_IMPOSSIBLE }}
                  title={
                    impossible > 0
                      ? `Se sale de la GPU: faltan ${fmtMb(impossible)} aunque se desalojen los demás`
                      : `Se sale de la GPU con lo cargado ahora (${fmtMb(overflow)}): habría que desalojar otros modelos`
                  }
                  data-part="impossible"
                />
              )}
            </div>
          </div>
        )
      })}
      <div className="flex flex-wrap gap-x-3 gap-y-1 border-t border-border/60 pt-2 text-[10.5px] text-muted-foreground" data-testid="memory-legend">
        <span className="flex items-center gap-1" title="Reserva del driver, contextos CUDA, escritorio y el margen del planificador: ningún modelo puede usarla">
          <span className="h-2 w-2 rounded-sm" style={HATCH_SYSTEM} /> sistema
        </span>
        {anyOthers && (
          <span className="flex items-center gap-1" title="Otros modelos cargados ahora: se pueden desalojar">
            <span className="h-2 w-2 rounded-sm bg-slate-500/80" /> otros modelos
          </span>
        )}
        {parts.map((p) => (
          <span key={p} className="flex items-center gap-1">
            <span className="h-2 w-2 rounded-sm" style={{ background: PART_META[p]?.color ?? "#a78bfa" }} />
            {PART_LABEL[p] ?? PART_META[p]?.label ?? p}
          </span>
        ))}
        <span className="flex items-center gap-1">
          <span className="h-2 w-2 rounded-sm bg-muted/60" /> libre
        </span>
        {anyOverflow && (
          <span className="flex items-center gap-1">
            <span className="h-2 w-2 rounded-sm" style={HATCH_IMPOSSIBLE} /> no cabe
          </span>
        )}
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
  measured: "medido (todo el modelo)",
}
