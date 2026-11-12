import { useRef } from "react"
import { Star } from "lucide-react"
import { gpuColor } from "@/components/dashboard/GpuBadge"
import { shortGpuName } from "@/components/dashboard/gpuNames"
import { cn } from "@/lib/cn"
import type { GPUState } from "@/types"
import { getPath } from "../draft"
import { fmtMb } from "../PreviewPanel"
import { FieldRow, SchemaFields } from "../SchemaField"
import type { Json } from "../types"
import { Hint, Panel, Segmented } from "../ui"
import type { SectionProps } from "./types"
import { useCapacity } from "./useCapacity"
import { PresetsOverrideNotice } from "./ContextSection"

const LLAMA = "llama_cpp"

export function GpuSection(props: SectionProps) {
  const { editor, readOnly, gpus } = props
  const d = editor.draft
  if (!d || !editor.schema) return null
  const isLlama = editor.schema.section === LLAMA || editor.schema.backend === "bitnet"
  const handled = [
    "llama_cpp.main_gpu",
    "llama_cpp.disabled_gpus",
    "llama_cpp.split_mode",
    "llama_cpp.split_strategy",
    "llama_cpp.tensor_split",
    "llama_cpp.gpu_layers",
    "bitnet.gpu_layers",
    "llama_cpp.mmproj_gpu",
    "vllm.gpu_memory_utilization",
    "sglang.mem_fraction_static",
  ]
  return (
    <div className="space-y-4">
      {(props.presets ?? []).some((r) => r.preset.placementPolicy !== "auto") && (
        <PresetsOverrideNotice count={(props.presets ?? []).length} what="las GPUs en las que carga" />
      )}
      <PreferredGpuPanel {...props} />
      {editor.schema.section === LLAMA && <LlamaPlacementPanel {...props} />}
      {isLlama && <GpuLayersPanel {...props} />}
      <FractionPanel {...props} />
      <SchemaFields title="Más opciones de colocación" editor={editor} group="gpus" exclude={handled} readOnly={readOnly} showAdvanced />
      {editor.fieldByKey.get("llama_cpp.mmproj_gpu") && (
        <Panel title="Proyector de visión">
          <FieldRow field={editor.fieldByKey.get("llama_cpp.mmproj_gpu")!} editor={editor} readOnly={readOnly}>
            <GpuPicker
              gpus={gpus}
              value={getPath(d.extraConfig, [LLAMA, "mmproj_gpu"]) as number | undefined}
              onChange={(v) => editor.setField([LLAMA, "mmproj_gpu"], v)}
              readOnly={readOnly}
              autoLabel="Automática"
            />
          </FieldRow>
        </Panel>
      )}
    </div>
  )
}

function GpuPicker({
  gpus,
  value,
  onChange,
  readOnly,
  autoLabel,
}: {
  gpus: GPUState[]
  value: number | null | undefined
  onChange: (v: number | undefined) => void
  readOnly?: boolean
  autoLabel: string
}) {
  return (
    <div className="flex flex-wrap gap-2">
      <button
        type="button"
        disabled={readOnly}
        onClick={() => onChange(undefined)}
        className={cn(
          "rounded-lg border px-3 py-1.5 text-xs",
          value == null ? "border-primary bg-primary/10" : "border-border hover:bg-muted",
        )}
      >
        {autoLabel}
      </button>
      {gpus.map((g) => {
        const color = gpuColor(g.index)
        const active = value === g.index
        return (
          <button
            key={g.index}
            type="button"
            disabled={readOnly}
            onClick={() => onChange(g.index)}
            className={cn("rounded-lg border px-3 py-1.5 text-xs font-medium", active ? "bg-muted" : "border-border hover:bg-muted/50")}
            style={active ? { borderColor: color, color } : { color }}
          >
            #{g.index} {shortGpuName(g.name)}
          </button>
        )
      })}
    </div>
  )
}

function PreferredGpuPanel({ editor, readOnly, gpus, liveVramByGpu }: SectionProps) {
  const d = editor.draft!
  return (
    <Panel title="¿Dónde corre?" subtitle="GPU en la que el planificador intenta cargar este despliegue primero.">
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3" data-testid="gpu-cards">
        <button
          type="button"
          disabled={readOnly}
          onClick={() => editor.update({ preferredGpu: null })}
          className={cn(
            "rounded-xl border p-3 text-left",
            d.preferredGpu == null ? "border-primary bg-primary/10 ring-1 ring-primary/40" : "border-border hover:bg-muted/30",
          )}
        >
          <div className="text-sm font-semibold">Automática</div>
          <p className="text-[11px] text-muted-foreground">La GPU por defecto del servidor, o la que tenga sitio.</p>
        </button>
        {gpus.map((g) => {
          const color = gpuColor(g.index)
          const active = d.preferredGpu === g.index
          const mine = liveVramByGpu[String(g.index)] ?? 0
          const others = Math.max(0, g.usedVramMb - mine)
          return (
            <button
              key={g.index}
              type="button"
              disabled={readOnly}
              onClick={() => editor.update({ preferredGpu: g.index })}
              className={cn("rounded-xl border p-3 text-left", active ? "ring-1" : "border-border hover:bg-muted/30")}
              style={active ? { borderColor: color, boxShadow: `0 0 0 1px ${color}66`, background: `${color}12` } : undefined}
              data-gpu-card={g.index}
            >
              <div className="flex items-center justify-between">
                <span className="text-sm font-semibold" style={{ color }}>
                  #{g.index} {shortGpuName(g.name)}
                </span>
                <span className="text-[11px] tabular-nums text-muted-foreground">{fmtMb(g.totalVramMb)}</span>
              </div>
              <div className="mt-2 flex h-2 overflow-hidden rounded bg-muted">
                <div className="bg-slate-500/70" style={{ width: `${(others / g.totalVramMb) * 100}%` }} />
                <div style={{ width: `${(mine / g.totalVramMb) * 100}%`, background: color }} />
              </div>
              <p className="mt-1 text-[10.5px] text-muted-foreground">
                {fmtMb(g.freeVramMb)} libres{mine ? ` · este modelo ${fmtMb(mine)}` : ""}
              </p>
            </button>
          )
        })}
      </div>
    </Panel>
  )
}

/** Reparto de llama.cpp: qué GPUs usa, cuál es la principal y en qué
 *  proporción se reparten las capas. */
function LlamaPlacementPanel({ editor, readOnly, gpus }: SectionProps) {
  const d = editor.draft!
  const section = (getPath(d.extraConfig, [LLAMA]) ?? {}) as Record<string, Json>
  const disabled = new Set((Array.isArray(section.disabled_gpus) ? section.disabled_gpus : []) as number[])
  const visible = gpus.filter((g) => !disabled.has(g.index))
  const splitMode = (section.split_mode as string | undefined) ?? undefined
  const single = splitMode === "none" || visible.length <= 1
  const mainPos = typeof section.main_gpu === "number" ? section.main_gpu : 0
  const split = Array.isArray(section.tensor_split) ? (section.tensor_split as number[]) : null
  const strategy = (section.split_strategy as string | undefined) ?? undefined

  const set = (key: string, value: Json | undefined) => editor.setField([LLAMA, key], value)

  const toggleGpu = (index: number) => {
    const next = new Set(disabled)
    if (next.has(index)) next.delete(index)
    else {
      if (gpus.length - next.size <= 1) return // al menos una
      next.add(index)
    }
    set("disabled_gpus", next.size ? [...next].sort() : undefined)
    // El reparto y la principal son posiciones entre las visibles: se rehacen.
    set("tensor_split", undefined)
    if (typeof section.main_gpu === "number") set("main_gpu", undefined)
  }

  const mode = single ? "none" : splitMode ?? "layer"
  return (
    <Panel title="Reparto entre GPUs" subtitle="llama.cpp puede usar una GPU o repartir el modelo entre varias.">
      <div className="space-y-3">
        <div className="flex flex-wrap gap-2" data-testid="gpu-include">
          {gpus.map((g) => {
            const color = gpuColor(g.index)
            const on = !disabled.has(g.index)
            const pos = visible.findIndex((v) => v.index === g.index)
            const isMain = on && pos === mainPos
            return (
              <div
                key={g.index}
                className={cn("flex items-center gap-2 rounded-lg border px-2.5 py-1.5", on ? "" : "opacity-50")}
                style={{ borderColor: on ? `${color}88` : undefined }}
              >
                <input type="checkbox" checked={on} disabled={readOnly} onChange={() => toggleGpu(g.index)} aria-label={`Usar GPU ${g.index}`} />
                <span className="text-xs font-medium" style={{ color }}>
                  #{g.index} {shortGpuName(g.name)}
                </span>
                {on && !single && (
                  <button
                    type="button"
                    disabled={readOnly}
                    title={isMain ? "GPU principal" : "Hacer principal"}
                    onClick={() => set("main_gpu", pos === 0 ? undefined : pos)}
                    className={isMain ? "text-amber-300" : "text-muted-foreground/50 hover:text-amber-200"}
                  >
                    <Star size={13} fill={isMain ? "currentColor" : "none"} />
                  </button>
                )}
              </div>
            )
          })}
        </div>

        <FieldRow field={editor.fieldByKey.get("llama_cpp.split_mode")!} editor={editor} readOnly={readOnly} compact>
          <Segmented
            size="sm"
            disabled={readOnly}
            value={splitMode ?? "__default"}
            options={[
              { value: "__default", label: "Por defecto (capas)" },
              { value: "none", label: "Una sola GPU", title: "Todo en la GPU principal" },
              { value: "layer", label: "Por capas", title: "Cada GPU guarda capas completas; lo habitual" },
              { value: "row", label: "Por filas", title: "Divide cada capa por filas; más tráfico entre GPUs" },
              { value: "tensor", label: "Por tensores", title: "Divide tensores; solo con soporte en el binario" },
            ]}
            onChange={(v) => set("split_mode", v === "__default" ? undefined : v)}
          />
        </FieldRow>

        {!single && mode !== "none" && (
          <div className="space-y-2">
            <Segmented
              size="sm"
              disabled={readOnly}
              value={split ? "custom" : strategy === "favor_main" ? "favor_main" : "evenly"}
              options={[
                { value: "evenly", label: "Proporcional a la VRAM" },
                { value: "favor_main", label: "Favorecer la principal" },
                { value: "custom", label: "A mano" },
              ]}
              onChange={(v) => {
                if (v === "custom") {
                  set("split_strategy", undefined)
                  set("tensor_split", visible.map((g) => Math.round(g.totalVramMb / 1024)))
                } else {
                  set("tensor_split", undefined)
                  set("split_strategy", v)
                }
              }}
            />
            <SplitBar
              gpus={visible}
              ratios={split ?? (strategy === "favor_main" ? visible.map((_, i) => (i === mainPos ? 1 : 0)) : visible.map((g) => g.totalVramMb))}
              editable={Boolean(split) && !readOnly}
              onChange={(r) => set("tensor_split", r)}
            />
            <Hint>
              {split
                ? "Arrastra el separador para cambiar qué parte de las capas va a cada GPU."
                : strategy === "favor_main"
                  ? "Llena primero la principal y usa la otra solo para lo que no quepa."
                  : "Cada GPU recibe capas en proporción a su VRAM total."}
            </Hint>
          </div>
        )}
      </div>
    </Panel>
  )
}

/** Barra dividida con separadores arrastrables. */
export function SplitBar({
  gpus,
  ratios,
  editable,
  onChange,
}: {
  gpus: GPUState[]
  ratios: number[]
  editable: boolean
  onChange: (ratios: number[]) => void
}) {
  const ref = useRef<HTMLDivElement>(null)
  const total = ratios.reduce((a, b) => a + b, 0) || 1
  const pcts = ratios.map((r) => (r / total) * 100)
  const startDrag = (sep: number) => (e: React.PointerEvent) => {
    if (!editable || !ref.current) return
    e.preventDefault()
    const rect = ref.current.getBoundingClientRect()
    const move = (ev: PointerEvent) => {
      const x = Math.min(Math.max(ev.clientX - rect.left, 0), rect.width)
      const at = (x / rect.width) * 100
      const before = pcts.slice(0, sep).reduce((a, b) => a + b, 0)
      const pairTotal = pcts[sep] + pcts[sep + 1]
      const left = Math.min(Math.max(at - before, 0), pairTotal)
      const next = [...pcts]
      next[sep] = left
      next[sep + 1] = pairTotal - left
      onChange(next.map((p) => Math.round(p * 10) / 10))
    }
    const up = () => {
      window.removeEventListener("pointermove", move)
      window.removeEventListener("pointerup", up)
    }
    window.addEventListener("pointermove", move)
    window.addEventListener("pointerup", up)
  }
  return (
    <div ref={ref} className="relative flex h-8 overflow-hidden rounded-lg border border-border" data-testid="split-bar">
      {gpus.map((g, i) => {
        const color = gpuColor(g.index)
        return (
          <div
            key={g.index}
            className="relative flex items-center justify-center text-[11px] font-semibold"
            style={{ width: `${pcts[i]}%`, background: `${color}33`, color }}
          >
            #{g.index} · {Math.round(pcts[i])}%
            {editable && i < gpus.length - 1 && (
              <span
                role="separator"
                aria-label={`Separador ${i}`}
                onPointerDown={startDrag(i)}
                className="absolute inset-y-0 -right-1 z-10 w-2 cursor-col-resize bg-white/30 hover:bg-white/60"
              />
            )}
          </div>
        )
      })}
    </div>
  )
}

/** Capas en GPU frente a CPU, dibujadas una a una. */
function GpuLayersPanel({ editor, readOnly }: SectionProps) {
  const section = editor.schema!.section === LLAMA ? LLAMA : "bitnet"
  const field = editor.fieldByKey.get(`${section}.gpu_layers`)
  const { data: cap } = useCapacity(editor.model?.model_id)
  if (!field) return null
  const total = cap?.arch?.layers ?? null
  const raw = getPath(editor.draft!.extraConfig, [section, "gpu_layers"])
  const value = typeof raw === "number" ? raw : Number(field.default ?? 0)
  const onGpu = total ? Math.min(value, total) : value
  const all = total != null && value >= total
  return (
    <Panel title="Capas en GPU" subtitle={total ? `El modelo tiene ${total} capas.` : "Cuántas capas se cargan en la GPU; el resto va a la CPU."}>
      <FieldRow field={field} editor={editor} readOnly={readOnly} compact>
        <div className="space-y-2">
          <div className="flex flex-wrap items-center gap-3">
            <input
              type="range"
              min={0}
              max={total ?? 99}
              disabled={readOnly}
              value={Math.min(value, total ?? 99)}
              onChange={(e) => {
                const n = Number(e.target.value)
                editor.setField([section, "gpu_layers"], total && n >= total ? 999 : n)
              }}
              className="w-64"
              aria-label="Capas en GPU"
            />
            <span className="text-sm tabular-nums">{all ? "todas" : value === 0 ? "ninguna (CPU)" : `${onGpu}${total ? ` de ${total}` : ""}`}</span>
            <Segmented
              size="sm"
              disabled={readOnly}
              value={all ? "all" : value === 0 ? "cpu" : "custom"}
              options={[
                { value: "cpu", label: "Solo CPU" },
                { value: "all", label: "Todas en GPU" },
                { value: "custom", label: "Parcial", disabled: true },
              ]}
              onChange={(v) => editor.setField([section, "gpu_layers"], v === "cpu" ? 0 : 999)}
            />
          </div>
          {total && (
            <div className="flex h-4 gap-px" data-testid="layer-strip">
              {Array.from({ length: total }, (_, i) => (
                <div key={i} className={cn("flex-1 rounded-[2px]", i < onGpu ? "bg-indigo-500/70" : "bg-slate-600/50")} title={`Capa ${i + 1}: ${i < onGpu ? "GPU" : "CPU"}`} />
              ))}
            </div>
          )}
          {total && !all && value > 0 && <Hint tone="warn">Las capas en CPU van mucho más lentas y ocupan RAM del host.</Hint>}
        </div>
      </FieldRow>
    </Panel>
  )
}

/** Fracción de la GPU que reservan vLLM / SGLang al arrancar. */
function FractionPanel({ editor, readOnly, gpus }: SectionProps) {
  const field = editor.fieldByKey.get("vllm.gpu_memory_utilization") ?? editor.fieldByKey.get("sglang.mem_fraction_static")
  if (!field) return null
  const raw = getPath(editor.draft!.extraConfig, field.path)
  const value = typeof raw === "number" ? raw : Number(field.default ?? 0.9)
  const gpu = gpus.find((g) => g.index === (editor.draft!.preferredGpu ?? 0)) ?? gpus[0]
  return (
    <Panel title="Memoria reservada por el motor" subtitle="El motor se queda con esta parte de cada GPU al arrancar, la use o no.">
      <FieldRow field={field} editor={editor} readOnly={readOnly} compact>
        <div className="space-y-2">
          <div className="flex flex-wrap items-center gap-3">
            <input
              type="range"
              min={0.05}
              max={0.98}
              step={0.01}
              value={value}
              disabled={readOnly}
              onChange={(e) => editor.setField(field.path, Number(e.target.value))}
              className="w-64"
              aria-label={field.label}
            />
            <span className="tabular-nums">{Math.round(value * 100)}%</span>
            <Segmented
              size="sm"
              disabled={readOnly}
              value={[0.75, 0.85, 0.9, 0.95].includes(value) ? value : -1}
              options={[0.75, 0.85, 0.9, 0.95].map((v) => ({ value: v, label: `${Math.round(v * 100)}%` }))}
              onChange={(v) => editor.setField(field.path, v)}
            />
          </div>
          {gpu && (
            <div className="flex h-3 overflow-hidden rounded bg-muted">
              <div className="bg-indigo-500/70" style={{ width: `${value * 100}%` }} title={`Motor: ${fmtMb(gpu.totalVramMb * value)}`} />
            </div>
          )}
          {gpu && <Hint>En la #{gpu.index}: {fmtMb(gpu.totalVramMb * value)} para el motor y {fmtMb(gpu.totalVramMb * (1 - value))} para todo lo demás.</Hint>}
        </div>
      </FieldRow>
    </Panel>
  )
}
