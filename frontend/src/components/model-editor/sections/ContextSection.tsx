import { useMemo } from "react"
import { gpuColor } from "@/components/dashboard/GpuBadge"
import { fmtContext } from "@/components/dashboard/format"
import { cn } from "@/lib/cn"
import type { GPUState } from "@/types"
import { getPath } from "../draft"
import { fmtMb } from "../PreviewPanel"
import { FieldRow, SchemaFields } from "../SchemaField"
import type { CapacityInfo, Json } from "../types"
import { Callout, Hint, Panel, Segmented } from "../ui"
import type { SectionProps } from "./types"
import { useCapacity } from "./useCapacity"

const MARKS = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]

/** Deslizador logarítmico de contexto con marcas en potencias de dos. */
export function ContextSlider({
  value,
  onChange,
  max,
  native,
  disabled,
  testId,
}: {
  value: number
  onChange: (n: number) => void
  max: number
  native?: number | null
  disabled?: boolean
  testId?: string
}) {
  const lo = Math.log2(2048)
  const hi = Math.log2(Math.max(max, 8192))
  const pos = ((Math.log2(Math.max(value, 2048)) - lo) / (hi - lo)) * 100
  const marks = MARKS.filter((m) => m <= Math.max(max, 8192))
  return (
    <div className="space-y-1" data-testid={testId}>
      <input
        type="range"
        min={lo}
        max={hi}
        step={0.01}
        value={Math.log2(Math.max(value, 2048))}
        disabled={disabled}
        aria-label="Contexto"
        onChange={(e) => {
          const raw = 2 ** Number(e.target.value)
          // Se ajusta a la potencia de dos más cercana si está muy cerca.
          const snap = marks.find((m) => Math.abs(Math.log2(m) - Number(e.target.value)) < 0.06)
          onChange(snap ?? Math.round(raw / 1024) * 1024)
        }}
        className="w-full"
      />
      <div className="relative h-4 text-[9.5px] text-muted-foreground">
        {marks.map((m) => (
          <button
            key={m}
            type="button"
            disabled={disabled}
            onClick={() => onChange(m)}
            className="absolute -translate-x-1/2 hover:text-foreground"
            style={{ left: `${((Math.log2(m) - lo) / (hi - lo)) * 100}%` }}
          >
            {fmtContext(m)}
          </button>
        ))}
        {native ? (
          <span className="absolute -top-6 h-3 w-px bg-amber-400" style={{ left: `${((Math.log2(native) - lo) / (hi - lo)) * 100}%` }} title={`Nativo: ${fmtContext(native)}`} />
        ) : null}
        <span className="sr-only">{pos}</span>
      </div>
    </div>
  )
}

/** Precisión de la KV de llama.cpp como tres botones (K y V a la vez), con la
 *  opción de separarlas en avanzado. */
const KV_SIMPLE = [
  { value: "f16", label: "f16", hint: "sin cuantizar" },
  { value: "q8_0", label: "q8", hint: "mitad de memoria" },
  { value: "q4_0", label: "q4", hint: "un cuarto, algo de pérdida" },
]

export function kvLabelForCapacity(cacheType: string | undefined): string {
  if (!cacheType || cacheType === "f16") return "fp16"
  if (cacheType.startsWith("q8")) return "q8"
  if (cacheType.startsWith("q4") || cacheType.startsWith("q5")) return "q4"
  return "fp16"
}

export function ContextSection(props: SectionProps) {
  const { editor, readOnly } = props
  if (!editor.draft || !editor.schema) return null
  const section = editor.schema.section
  if (section === "llama_cpp" || editor.schema.backend === "bitnet") return <LlamaContext {...props} />
  return (
    <div className="space-y-4">
      {(props.presets ?? []).length > 0 && <PresetsOverrideNotice count={props.presets!.length} what="el contexto y las peticiones simultáneas" />}
      <GenericContext {...props} />
      <SchemaFields editor={editor} group="context" readOnly={readOnly} showAdvanced exclude={contextKeys(editor.schema.backend)} />
    </div>
  )
}

function contextKeys(backend: string): string[] {
  return ["vllm.max_model_len", "sglang.context_length", "tensorrt_llm.context_length", "ollama.num_ctx", "strata.native_context_length"].filter((k) =>
    k.startsWith(backend === "llamacpp_mks" ? "llama_cpp" : backend),
  )
}

function GenericContext({ editor, readOnly }: SectionProps) {
  const field = editor.fields.find((f) => f.widget === "context")
  const { data: cap } = useCapacity(editor.model?.model_id, { slots: [1, 2, 4, 8, 16] })
  if (!field) {
    return <Callout tone="info">Este backend no tiene un contexto configurable por modelo.</Callout>
  }
  const raw = getPath(editor.draft!.extraConfig, field.path)
  const native = cap?.arch?.native_context ?? (Number(editor.model?.capabilities?.native_context_length) || null)
  const value = typeof raw === "number" ? raw : Number(field.default ?? native ?? 8192)
  // El tope del esquema (en Ollama, el máximo global) manda sobre el nativo.
  const ceiling = field.max ?? null
  const sliderMax = ceiling ? Math.min(ceiling, Math.max(native ?? 0, value, 8192)) : Math.max(native ?? 0, value, 131072)
  return (
    <Panel title="Contexto" subtitle={native ? `Contexto nativo del modelo: ${fmtContext(native)}.` : undefined}>
      <FieldRow field={field} editor={editor} readOnly={readOnly} compact>
        <div className="space-y-2">
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-semibold tabular-nums">{raw === undefined && field.default == null ? "auto" : fmtContext(value)}</span>
            {raw === undefined && (
              <span className="text-xs text-muted-foreground">{field.default == null ? "lo decide el motor" : "el valor por defecto"}</span>
            )}
          </div>
          <ContextSlider value={value} max={sliderMax} native={native} onChange={(n) => editor.setField(field.path, ceiling ? Math.min(n, ceiling) : n)} disabled={readOnly} />
          {ceiling ? <Hint>Tope: {fmtContext(ceiling)}.</Hint> : null}
        </div>
      </FieldRow>
      {cap?.applicable && <CapacityTable cap={cap} />}
    </Panel>
  )
}

function LlamaContext({ editor, readOnly, gpus, presets = [] }: SectionProps) {
  const sec = editor.schema!.backend === "bitnet" ? "bitnet" : "llama_cpp"
  const slotsKey = sec === "bitnet" ? "parallel" : "parallel_slots"
  const d = editor.draft!
  const s = (getPath(d.extraConfig, [sec]) ?? {}) as Record<string, Json>
  const ctx = Number(s.ctx_size ?? editor.fieldByKey.get(`${sec}.ctx_size`)?.default ?? 4096)
  const slots = Number(s[slotsKey] ?? editor.fieldByKey.get(`${sec}.${slotsKey}`)?.default ?? 1)
  const unified = s.kv_unified === true
  const ck = (s.cache_type_k as string | undefined) ?? "f16"
  const cv = (s.cache_type_v as string | undefined) ?? "f16"
  const flash = s.flash_attn === true || (s.flash_attn === undefined && editor.fieldByKey.get(`${sec}.flash_attn`)?.default === true)
  const kvLabel = kvLabelForCapacity(ck)
  const target = d.preferredGpu ?? editor.preview?.gpuIndices?.[0] ?? null
  const { data: cap } = useCapacity(editor.model?.model_id, { gpu: target, slots: [1, 2, 4, 8], kvDtype: kvLabel })
  const native = cap?.arch?.native_context ?? (Number(editor.model?.capabilities?.native_context_length) || null)
  const perRequest = unified ? ctx : Math.floor(ctx / Math.max(1, slots))
  const set = (key: string, value: Json | undefined) => editor.setField([sec, key], value)

  return (
    <div className="space-y-4">
      {presets.length > 0 && <PresetsOverrideNotice count={presets.length} />}
      <Panel title="¿Cuánto contexto y cuántas a la vez?" subtitle={presets.length ? "Base sin presets: cada preset sustituye estos valores al cargar." : undefined}>
        <div className="grid gap-5 lg:grid-cols-[1fr_auto]">
          <div className="space-y-4">
            <div>
              <div className="flex items-baseline justify-between">
                <span className="text-xs font-medium">{unified ? "Contexto total del pool" : "Contexto total (se reparte entre los slots)"}</span>
                <span className="text-2xl font-semibold tabular-nums" data-testid="ctx-value">
                  {fmtContext(ctx)}
                </span>
              </div>
              <ContextSlider value={ctx} max={Math.max(native ?? 0, ctx, 131072)} native={native} onChange={(n) => set("ctx_size", n)} disabled={readOnly} testId="ctx-slider" />
              {native && ctx > native && <Hint tone="warn">Por encima del contexto nativo ({fmtContext(native)}): necesita RoPE escalado o el modelo degradará.</Hint>}
            </div>

            <div className="flex flex-wrap items-end gap-6">
              <div>
                <p className="mb-1 text-xs font-medium">Peticiones simultáneas</p>
                <Segmented
                  size="sm"
                  disabled={readOnly}
                  value={[1, 2, 4, 8, 16].includes(slots) ? slots : -1}
                  options={[1, 2, 4, 8, 16].map((n) => ({ value: n, label: String(n) }))}
                  onChange={(n) => set(slotsKey, n)}
                />
              </div>
              <div>
                <p className="mb-1 text-xs font-medium">Precisión de la KV</p>
                <Segmented
                  size="sm"
                  disabled={readOnly}
                  value={ck === cv ? ck : "mixed"}
                  options={KV_SIMPLE.map((k) => ({
                    value: k.value,
                    label: k.label,
                    title: k.hint,
                    disabled: k.value !== "f16" && !flash,
                  }))}
                  onChange={(v) => {
                    set("cache_type_k", v === "f16" ? undefined : v)
                    set("cache_type_v", v === "f16" ? undefined : v)
                  }}
                />
                {!flash && <Hint tone="warn">Cuantizar la KV requiere flash attention (Aceleración).</Hint>}
              </div>
            </div>
          </div>

          <KvModeDiagram unified={unified} slots={slots} ctx={ctx} perRequest={perRequest} onChange={readOnly ? undefined : (v) => set("kv_unified", v)} />
        </div>
      </Panel>

      {cap?.applicable && cap.arch && (
        <Panel
          title="Capacidad"
          subtitle={`${fmtMb(cap.weights_mb)} de pesos · ${cap.arch.kv_mb_per_1k_tokens.toFixed(1)} MB de KV por cada 1k tokens (${kvLabel}).`}
        >
          <CapacityChart cap={cap} gpus={gpus} ctx={ctx} targetGpu={target} />
          <CapacityTable cap={cap} />
        </Panel>
      )}

      <SchemaFields
        title="Más opciones de contexto"
        editor={editor}
        group="context"
        exclude={[`${sec}.ctx_size`, `${sec}.${slotsKey}`, `${sec}.kv_unified`]}
        readOnly={readOnly}
        showAdvanced
      />
    </div>
  )
}

/** Dibujo de cómo se reparte la KV: N cajas iguales o un depósito común. */
function KvModeDiagram({
  unified,
  slots,
  ctx,
  perRequest,
  onChange,
}: {
  unified: boolean
  slots: number
  ctx: number
  perRequest: number
  onChange?: (v: boolean | undefined) => void
}) {
  const shown = Math.min(slots, 8)
  return (
    <div className="w-full min-w-[220px] max-w-xs space-y-2 rounded-xl border border-border bg-muted/20 p-3" data-testid="kv-mode">
      <Segmented
        size="sm"
        disabled={!onChange}
        value={unified ? "pool" : "slots"}
        options={[
          { value: "slots", label: "Por slot" },
          { value: "pool", label: "Pool compartido" },
        ]}
        onChange={(v) => onChange?.(v === "pool" ? true : undefined)}
      />
      {unified ? (
        <div className="space-y-1">
          <div className="flex gap-1">
            {Array.from({ length: shown }, (_, i) => (
              <div key={i} className="h-3 flex-1 rounded-sm bg-sky-400/60" />
            ))}
          </div>
          <div className="flex h-10 items-center justify-center rounded-md bg-sky-500/25 text-[11px] text-sky-100">{fmtContext(ctx)} compartidos</div>
          <Hint>Cada petición puede usar hasta {fmtContext(perRequest)} mientras quede sitio en el pool.</Hint>
        </div>
      ) : (
        <div className="space-y-1">
          <div className="flex gap-1">
            {Array.from({ length: shown }, (_, i) => (
              <div key={i} className="flex h-10 flex-1 items-center justify-center rounded-md bg-indigo-500/30 text-[10px] text-indigo-100">
                {fmtContext(perRequest)}
              </div>
            ))}
          </div>
          <Hint>Cada slot tiene su trozo fijo: {fmtContext(perRequest)} por petición.</Hint>
        </div>
      )}
    </div>
  )
}

/** VRAM frente a contexto, una curva por número de slots, con el total de
 *  cada GPU como techo. */
function CapacityChart({
  cap,
  gpus,
  ctx,
  targetGpu,
}: {
  cap: CapacityInfo
  gpus: GPUState[]
  ctx: number
  targetGpu: number | null
}) {
  const W = 560
  const H = 200
  const P = { l: 44, r: 12, t: 10, b: 24 }
  const arch = cap.arch!
  const base = (cap.weights_mb ?? 0) + (cap.overhead_mb ?? 0)
  // En llama.cpp ctx_size es la KV total (con o sin pool): la curva no depende de los slots.
  const kv = (c: number) => (arch.kv_mb_per_1k_tokens * c) / 1000
  const xMax = Math.max(arch.native_context || 0, ctx, 32768)
  const gpuTotals = gpus.map((g) => ({ index: g.index, mb: g.totalVramMb }))
  const both = gpuTotals.reduce((a, g) => a + g.mb, 0)
  const yMax = Math.max(both, base + kv(ctx)) * 1.05
  const lx = (c: number) => P.l + ((Math.log2(Math.max(c, 1024)) - 10) / (Math.log2(xMax) - 10)) * (W - P.l - P.r)
  const ly = (mb: number) => H - P.b - (mb / yMax) * (H - P.t - P.b)
  const curve = useMemo(() => {
    const pts: string[] = []
    for (let e = 10; e <= Math.log2(xMax) + 0.001; e += 0.25) {
      const c = 2 ** e
      pts.push(`${pts.length ? "L" : "M"}${lx(c).toFixed(1)},${ly(base + kv(c)).toFixed(1)}`)
    }
    return pts.join(" ")
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cap, xMax, yMax])
  const curMb = base + kv(ctx)
  const target = gpus.find((g) => g.index === targetGpu) ?? gpus[0]
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" data-testid="capacity-chart" role="img" aria-label="VRAM según el contexto">
      {[0.25, 0.5, 0.75, 1].map((f) => (
        <g key={f}>
          <line x1={P.l} x2={W - P.r} y1={ly(yMax * f)} y2={ly(yMax * f)} stroke="currentColor" className="text-border" strokeDasharray="2 3" />
          <text x={P.l - 4} y={ly(yMax * f) + 3} textAnchor="end" className="fill-muted-foreground text-[9px]">
            {fmtMb(yMax * f)}
          </text>
        </g>
      ))}
      {MARKS.filter((m) => m <= xMax && m >= 4096).map((m) => (
        <text key={m} x={lx(m)} y={H - 8} textAnchor="middle" className="fill-muted-foreground text-[9px]">
          {fmtContext(m)}
        </text>
      ))}
      {target && (
        <g>
          <rect x={P.l} y={P.t} width={W - P.l - P.r} height={Math.max(0, ly(target.totalVramMb) - P.t)} className="fill-red-500/5" />
          <line x1={P.l} x2={W - P.r} y1={ly(target.totalVramMb)} y2={ly(target.totalVramMb)} stroke={gpuColor(target.index)} strokeWidth={1.5} />
          <text x={W - P.r} y={ly(target.totalVramMb) - 3} textAnchor="end" className="text-[9.5px]" fill={gpuColor(target.index)}>
            1 GPU (#{target.index}) {fmtMb(target.totalVramMb)}
          </text>
        </g>
      )}
      {gpus.length > 1 && (
        <g>
          <line x1={P.l} x2={W - P.r} y1={ly(both)} y2={ly(both)} stroke="#a3a3a3" strokeDasharray="5 3" />
          <text x={W - P.r} y={ly(both) - 3} textAnchor="end" className="fill-muted-foreground text-[9.5px]">
            {gpus.length} GPUs {fmtMb(both)}
          </text>
        </g>
      )}
      <path d={curve} fill="none" stroke="#0ea5e9" strokeWidth={2} />
      <circle cx={lx(ctx)} cy={ly(curMb)} r={5} className={cn(target && curMb > target.totalVramMb ? "fill-red-400" : "fill-emerald-400")} stroke="white" strokeWidth={1.5} />
      <text x={lx(ctx) > W * 0.7 ? lx(ctx) - 8 : lx(ctx) + 8} y={ly(curMb) + 16} textAnchor={lx(ctx) > W * 0.7 ? "end" : "start"} className="fill-foreground text-[10px] font-medium">
        {fmtContext(ctx)} · {fmtMb(curMb)}
      </text>
    </svg>
  )
}

function CapacityTable({ cap }: { cap: CapacityInfo }) {
  if (!cap.capacity?.length) return null
  const max = Math.max(...cap.capacity.map((r) => r.max_context_capped || 0), 1)
  const label = cap.concurrency_label === "sequences" ? "secuencias" : "slots"
  return (
    <div className="mt-3 space-y-1">
      <p className="text-[11px] font-medium text-muted-foreground">Contexto máximo por petición en 1 GPU</p>
      {cap.capacity.map((row) => (
        <div key={row.slots} className="flex items-center gap-2 text-[11px]">
          <span className="w-16 tabular-nums text-muted-foreground">
            {row.slots} {label}
          </span>
          <div className="h-2 flex-1 overflow-hidden rounded bg-muted">
            <div className="h-full bg-sky-500/60" style={{ width: `${(row.max_context_capped / max) * 100}%` }} />
          </div>
          <span className="w-12 text-right tabular-nums">{fmtContext(row.max_context_capped)}</span>
        </div>
      ))}
      {cap.note && <Hint>{cap.note}</Hint>}
    </div>
  )
}

/** Con presets activos, contexto/slots/KV/GPUs los pone cada preset. */
export function PresetsOverrideNotice({ count, what = "el contexto, las peticiones simultáneas y la precisión de la KV" }: { count: number; what?: string }) {
  return (
    <Callout tone="info">
      Este modelo tiene {count} preset{count === 1 ? "" : "s"} activo{count === 1 ? "" : "s"}: {what} los fija el preset con el que se carga
      (sección Presets). Lo de aquí solo se usa si se carga sin preset.
    </Callout>
  )
}
