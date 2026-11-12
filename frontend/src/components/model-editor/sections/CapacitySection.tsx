import { useEffect, useMemo, useState } from "react"
import { Link, useSearchParams } from "react-router-dom"
import { Plus, Star } from "lucide-react"
import { modelEditorApi } from "@/api/modelEditor"
import { GpuBadge } from "@/components/dashboard/GpuBadge"
import { fmtContext } from "@/components/dashboard/format"
import { cn } from "@/lib/cn"
import { getPath } from "../draft"
import { GpuMemoryBar, fmtMb } from "../PreviewPanel"
import type { Json } from "../types"
import { Callout, Hint, Panel, Segmented } from "../ui"
import type { LadderRung } from "../usePresetLadder"
import { ContextSection } from "./ContextSection"
import { FractionPanel, GpuLayersPanel, GpuSection } from "./GpuSection"
import { SchemaFields } from "../SchemaField"
import { PolicyPanel, ProfileLadder } from "./PresetsSection"
import type { SectionProps } from "./types"

/** Familias de backend por cómo gestionan la memoria. */
export type CapacityFamily = "llama" | "encoder" | "engine" | "ollama" | "fixed"

export function capacityFamily(
  backend: string | undefined,
  isLlm: boolean,
  caps?: Record<string, unknown> | null,
): CapacityFamily {
  const encoderOnly = Boolean(caps && (caps.embeddings || caps.rerank || caps.pooling) && !caps.chat)
  if (backend === "llama_cpp" || backend === "llamacpp_mks" || backend === "bitnet") return encoderOnly ? "encoder" : "llama"
  if (backend === "vllm" || backend === "sglang" || backend === "tensorrt_llm") return "engine"
  if (backend === "ollama") return "ollama"
  return isLlm ? "engine" : "fixed"
}

const FAMILY_TEXT: Record<CapacityFamily, string> = {
  llama:
    "llama.cpp reparte los pesos por capas entre las GPUs y reserva la KV según el contexto (por slot o en un pool compartido). Cada peldaño es una variante con su contexto, concurrencia y GPUs.",
  engine:
    "El motor reserva de golpe una fracción de cada GPU; los pesos van dentro y el resto es caché KV en páginas, que se reparte entre la longitud máxima y las secuencias simultáneas.",
  ollama:
    "Ollama es un único servidor para todos sus modelos: su paralelismo y su contexto máximo son globales (Ajustes) y él decide en qué GPU va cada capa. Por modelo solo se puede bajar el contexto.",
  fixed: "Este backend no tiene contexto: su memoria depende del modelo y de cuántas peticiones atiende a la vez.",
  encoder:
    "Embeddings y rerank no generan: el contexto solo tiene que cubrir la entrada más larga y la KV es pequeña. Lo que importa es cuántas peticiones atiende a la vez, la entrada máxima y en qué GPU va. No usa presets.",
}

/** Contexto configurado en la base, según el backend. */
function baseContext(extra: Record<string, Json>, backend: string, fallback: number | null): number | null {
  const pick = (path: string[]) => {
    const v = getPath(extra, path)
    return typeof v === "number" ? v : null
  }
  if (backend === "llama_cpp" || backend === "llamacpp_mks") return pick(["llama_cpp", "ctx_size"]) ?? fallback
  if (backend === "bitnet") return pick(["bitnet", "ctx_size"]) ?? fallback
  if (backend === "vllm") return pick(["vllm", "max_model_len"]) ?? fallback
  if (backend === "sglang") return pick(["sglang", "context_length"]) ?? fallback
  if (backend === "ollama") return pick(["ollama", "num_ctx"]) ?? fallback
  return null
}

/**
 * Capacidad: GPUs, contexto y presets en una sola vista. Arriba, los peldaños
 * (la base, que es lo que se carga sin preset, y cada preset); abajo, el lienzo
 * del peldaño elegido según cómo gestiona la memoria su backend.
 */
export function CapacitySection(props: SectionProps & { ladder: { rungs: LadderRung[]; profile: import("@/types").ModelProfile | null; reload: () => Promise<void> } }) {
  const { editor, readOnly, gpus, liveVramByGpu, ladder } = props
  const [search, setSearch] = useSearchParams()
  const schema = editor.schema
  const backend = schema?.backend ?? ""
  const family = capacityFamily(backend, Boolean(schema?.isLlm), editor.model?.capabilities as Record<string, unknown> | undefined)
  const canPreset = family !== "ollama" && family !== "encoder" && Boolean(schema?.isLlm) && ladder.profile != null
  const allPresets = ladder.profile?.runtimePresets ?? []
  // Con presets activos la base nunca se carga sola (toda carga elige un
  // preset, también "Cargar"): los peldaños son los presets y la base queda
  // como "Común a todos". Sin presets, la base es la configuración.
  const hasPresets = canPreset && allPresets.some((p) => p.enabled)
  const defaultKey = (allPresets.find((p) => p.isDefault && p.enabled) ?? allPresets.find((p) => p.enabled))?.presetKey
  const requested = search.get("r")
  const selected =
    requested === "new" && canPreset && !readOnly
      ? "new"
      : requested && allPresets.some((p) => p.presetKey === requested)
        ? requested
        : requested === "common" && hasPresets
          ? "common"
          : hasPresets && defaultKey
            ? defaultKey
            : "base"
  const select = (key: string) =>
    setSearch(
      (q) => {
        if (!key || key === "base") q.delete("r")
        else q.set("r", key)
        return q
      },
      { replace: true },
    )

  if (!editor.draft || !schema) return null
  return (
    <div className="space-y-4" data-testid="capacity-section">
      <Hint>{FAMILY_TEXT[family]}</Hint>
      <RungSelector {...props} family={family} selected={selected} onSelect={select} canPreset={canPreset && !readOnly} hasPresets={hasPresets} />

      {selected === "base" ? (
        <BaseCanvas {...props} family={family} hasPresets={false} />
      ) : selected === "common" ? (
        <SharedCanvas {...props} family={family} />
      ) : ladder.profile ? (
        <ProfileLadder
          key={`${ladder.profile.profileId}-${selected}`}
          profile={ladder.profile}
          gpus={gpus}
          readOnly={readOnly}
          nativeContext={Number(editor.model?.capabilities?.native_context_length || 0) || null}
          liveVramByGpu={liveVramByGpu}
          focusKey={selected}
          onChanged={async () => {
            await ladder.reload()
            await editor.refreshLive()
          }}
          onFocusDone={(key) => select(key ?? "")}
        />
      ) : null}

      {canPreset && ladder.profile && allPresets.length > 0 && (
        <PolicyPanel profile={ladder.profile} readOnly={readOnly} onChanged={ladder.reload} />
      )}
      {canPreset && ladder.profile && allPresets.length === 0 && selected === "base" && (
        <Callout tone="info">
          Sin presets, el modelo siempre carga con la base. Una escalera deja que las peticiones pequeñas usen poca memoria y solo las grandes
          paguen el contexto alto: crea peldaños con «Nuevo peldaño».
        </Callout>
      )}
    </div>
  )
}

function RungSelector({
  editor,
  gpus,
  ladder,
  family,
  selected,
  onSelect,
  canPreset,
  hasPresets,
}: SectionProps & {
  ladder: { rungs: LadderRung[]; profile: import("@/types").ModelProfile | null }
  family: CapacityFamily
  selected: string
  onSelect: (key: string) => void
  canPreset: boolean
  hasPresets: boolean
}) {
  const p = editor.preview
  const base = useMemo(() => {
    const extra = (editor.draft?.extraConfig ?? {}) as Record<string, Json>
    const fallback = editor.fieldByKey.get("ollama.num_ctx")?.default
    const ctx = baseContext(extra, editor.schema?.backend ?? "", typeof fallback === "number" ? fallback : null)
    const total = p?.estimate?.model_loading_memory_mb ?? null
    return { ctx, total, gpus: p?.gpuIndices ?? [] }
  }, [editor.draft, editor.schema, editor.fieldByKey, p])
  // Peldaños: todos los presets del perfil (también los desactivados, en gris).
  const presets = ladder.profile?.runtimePresets ?? []
  const resolved = new Map(ladder.rungs.map((r) => [r.preset.presetKey, r.resolution]))
  const sorted = [...presets].sort(
    (a, b) => (a.contextMode === "max" ? 1 : 0) - (b.contextMode === "max" ? 1 : 0) || (a.contextTokens ?? 0) - (b.contextTokens ?? 0),
  )
  const Item = ({
    k,
    title,
    ctx,
    parallel,
    gpuList,
    mb,
    fits,
    star,
    disabled,
    hint,
  }: {
    k: string
    title: string
    ctx: number | null
    parallel?: number | null
    gpuList: number[]
    mb: number | null
    fits: boolean | null
    star?: boolean
    disabled?: boolean
    hint?: string
  }) => (
    <button
      type="button"
      onClick={() => onSelect(k)}
      title={hint}
      data-rung={k}
      aria-pressed={selected === k}
      className={cn(
        "min-w-[150px] flex-1 rounded-xl border p-2.5 text-left transition-colors",
        selected === k ? "border-primary bg-primary/10 ring-1 ring-primary/40" : "border-border hover:bg-muted/30",
        disabled && "opacity-50",
      )}
    >
      <div className="flex items-center gap-1.5 text-sm font-semibold">
        {fits === false ? <span className="text-red-300">✗</span> : fits ? <span className="text-emerald-300">✓</span> : null}
        <span className="truncate">{title}</span>
        {star && <Star size={11} className="text-amber-300" fill="currentColor" />}
      </div>
      <div className="mt-0.5 text-[11px] tabular-nums text-muted-foreground">
        {ctx ? fmtContext(ctx) : "contexto auto"}
        {parallel ? ` · p${parallel}` : ""}
        {mb ? ` · ${fmtMb(mb)}` : ""}
      </div>
      <div className="mt-1 flex flex-wrap gap-1">
        {gpuList.map((g) => (
          <GpuBadge key={g} index={g} gpus={gpus} />
        ))}
      </div>
    </button>
  )
  const baseFits = (() => {
    if (!p?.gpus?.length) return null
    return p.gpus.every((s) => {
      const g = gpus.find((x) => x.index === s.index)
      return g ? s.mb <= g.totalVramMb : true
    })
  })()
  return (
    <Panel
      title={family === "ollama" || !editor.schema?.isLlm ? "Configuración" : "Peldaños"}
      subtitle={
        hasPresets
          ? "Cada carga usa uno de estos peldaños (el ★ si la petición no pide otro). Cada uno fija su contexto, concurrencia, KV y GPUs; lo demás es común a todos."
          : undefined
      }
    >
      <div className="flex flex-wrap gap-2" data-testid="rung-selector">
        {!hasPresets && (
          <Item
            k="base"
            title="Configuración"
            ctx={base.ctx}
            gpuList={base.gpus}
            mb={base.total}
            fits={baseFits}
            hint="Se guarda con el botón Guardar de la página."
          />
        )}
        {sorted.map((preset) => {
          const r = resolved.get(preset.presetKey) ?? null
          return (
            <Item
              key={preset.presetKey}
              k={preset.presetKey}
              title={preset.label}
              ctx={r?.effectiveContext ?? preset.contextTokens}
              parallel={r?.effectiveParallelism ?? preset.parallelism}
              gpuList={r?.targetGpus ?? preset.preferredGpus ?? []}
              mb={r?.estimatedVramMb ?? null}
              fits={preset.enabled ? (r?.fits ?? null) : null}
              star={preset.isDefault}
              disabled={!preset.enabled}
              hint="Los presets se guardan al momento."
            />
          )
        })}
        {hasPresets && (
          <button
            type="button"
            onClick={() => onSelect("common")}
            aria-pressed={selected === "common"}
            data-rung="common"
            title="Lo que comparten todos los peldaños. Se guarda con el botón Guardar de la página."
            className={cn(
              "min-w-[150px] rounded-xl border border-dashed p-2.5 text-left text-xs",
              selected === "common" ? "border-primary bg-primary/10" : "border-border text-muted-foreground hover:bg-muted/30",
            )}
          >
            <span className="block text-sm font-semibold text-foreground">Común a todos</span>
            capas, modo de la KV, descarga a CPU…
          </button>
        )}
        {canPreset && (
          <button
            type="button"
            onClick={() => onSelect("new")}
            aria-pressed={selected === "new"}
            className={cn(
              "flex min-w-[120px] items-center justify-center gap-1 rounded-xl border border-dashed p-2.5 text-xs",
              selected === "new" ? "border-primary bg-primary/10" : "border-border text-muted-foreground hover:bg-muted/30",
            )}
          >
            <Plus size={13} /> Nuevo peldaño
          </button>
        )}
      </div>
    </Panel>
  )
}

/** Lo que se carga sin preset: memoria por GPU y los controles del backend. */
function BaseCanvas(props: SectionProps & { family: CapacityFamily; hasPresets: boolean }) {
  const { editor, gpus, liveVramByGpu, family, hasPresets } = props
  const shares = new Map((editor.preview?.gpus ?? []).map((g) => [g.index, g]))
  const quiet = { ...props, presets: [] }
  return (
    <div className="space-y-4">
      <Panel
        title="Memoria por GPU"
        subtitle={
          family === "ollama"
            ? "Estimación: Ollama decide en qué GPU va cada capa."
            : hasPresets
              ? "De la base. La de cada preset se ve al elegirlo."
              : undefined
        }
      >
        <div className="space-y-2" data-testid="base-memory">
          {gpus.map((g) => (
            <GpuMemoryBar key={g.index} gpu={g} share={shares.get(g.index)} ownLiveMb={liveVramByGpu[String(g.index)] ?? 0} />
          ))}
          {!gpus.length && <Hint>Sin datos de GPU.</Hint>}
        </div>
      </Panel>
      {family === "ollama" && <OllamaGlobalCard />}
      {family === "encoder" && <EncoderPanel {...quiet} />}
      <GpuSection {...quiet} />
      {family !== "fixed" && family !== "encoder" && <ContextSection {...quiet} />}
    </div>
  )
}

/** Opciones que cada preset sustituye: con presets no se muestran en lo común. */
const PRESET_OWNED = [
  "llama_cpp.ctx_size",
  "llama_cpp.parallel_slots",
  "llama_cpp.cache_type_k",
  "llama_cpp.cache_type_v",
  "llama_cpp.disabled_gpus",
  "llama_cpp.split_mode",
  "llama_cpp.split_strategy",
  "llama_cpp.tensor_split",
  "llama_cpp.main_gpu",
  "llama_cpp.kv_unified",
  "llama_cpp.gpu_layers",
  "bitnet.ctx_size",
  "bitnet.parallel",
  "bitnet.cache_type_k",
  "bitnet.cache_type_v",
  "bitnet.gpu_layers",
  "bitnet.kv_unified",
  "vllm.max_model_len",
  "vllm.kv_cache_dtype",
  "vllm.gpu_memory_utilization",
  "sglang.context_length",
  "sglang.mem_fraction_static",
]

/** Lo que comparten todos los peldaños (la base, cuando hay presets). */
function SharedCanvas(props: SectionProps & { family: CapacityFamily }) {
  const { editor, readOnly, family } = props
  const kvField = editor.fieldByKey.get("llama_cpp.kv_unified") ?? editor.fieldByKey.get("bitnet.kv_unified")
  const unified = kvField ? getPath(editor.draft!.extraConfig, kvField.path) === true : false
  return (
    <div className="space-y-4" data-testid="shared-canvas">
      <Callout tone="info">
        Cada peldaño fija su contexto, sus peticiones simultáneas, la precisión de la KV y sus GPUs. Aquí está lo que comparten todos;
        se guarda con el botón Guardar de la página.
      </Callout>
      {family === "llama" && <GpuLayersPanel {...props} />}
      {kvField && (
        <Panel title="Modo de la KV" subtitle="Cómo usan la memoria de contexto los slots de cualquier peldaño.">
          <Segmented
            size="sm"
            disabled={readOnly}
            value={unified ? "pool" : "slots"}
            options={[
              { value: "slots", label: "Por slot", title: "Cada petición tiene su trozo fijo del contexto del peldaño" },
              { value: "pool", label: "Pool compartido", title: "Todas comparten el contexto del peldaño mientras quede sitio" },
            ]}
            onChange={(v) => editor.setField(kvField.path, v === "pool" ? true : undefined)}
          />
          <Hint>
            {unified
              ? "Con pool, el contexto de un peldaño es el total compartido: una petición puede usarlo entero si las demás no lo ocupan."
              : "Por slot, el contexto de un peldaño se reparte a partes iguales entre sus peticiones simultáneas."}
          </Hint>
        </Panel>
      )}
      {family === "engine" && <FractionPanel {...props} />}
      <SchemaFields title="Colocación común" editor={editor} group="gpus" exclude={PRESET_OWNED} readOnly={readOnly} showAdvanced />
      <SchemaFields title="Contexto común" editor={editor} group="context" exclude={PRESET_OWNED} readOnly={readOnly} showAdvanced />
    </div>
  )
}

/** Embeddings / rerank en llama.cpp: simultáneas y entrada máxima. El batch no
 *  hace falta tocarlo: en modo embeddings llama-server lo ajusta a la entrada
 *  (en producción pasan textos de ~12k tokens con el micro-batch por defecto). */
function EncoderPanel({ editor, readOnly }: SectionProps) {
  const sec = (getPath(editor.draft!.extraConfig, ["llama_cpp"]) ?? {}) as Record<string, Json>
  const num = (key: string, fallback: number) => {
    const v = sec[key]
    if (typeof v === "number") return v
    const d = editor.fieldByKey.get(`llama_cpp.${key}`)?.default
    return typeof d === "number" ? d : fallback
  }
  const slots = Math.max(1, num("parallel_slots", 1))
  const ctx = num("ctx_size", 4096)
  const unified = sec.kv_unified === true
  const perInput = unified ? ctx : Math.floor(ctx / slots)
  const set = (key: string, value: number) => editor.setField(["llama_cpp", key], value)
  return (
    <Panel title="Rendimiento del encoder" subtitle="Se guarda con el botón Guardar de la página.">
      <div className="grid gap-5 md:grid-cols-2" data-testid="encoder-panel">
        <div>
          <p className="mb-1 text-xs font-medium">Peticiones simultáneas</p>
          <Segmented
            size="sm"
            disabled={readOnly}
            value={[1, 2, 4, 8, 16].includes(slots) ? slots : -1}
            options={[1, 2, 4, 8, 16].map((n) => ({ value: n, label: String(n) }))}
            onChange={(n) => {
              // Mantiene la entrada máxima por petición al cambiar los slots.
              set("parallel_slots", n)
              if (!unified) set("ctx_size", perInput * n)
            }}
          />
          <Hint>Más ayuda con ráfagas de peticiones; cada una añade su trozo de KV.</Hint>
        </div>
        <div>
          <p className="mb-1 text-xs font-medium">Entrada máxima por petición</p>
          <Segmented
            size="sm"
            disabled={readOnly}
            value={[512, 2048, 8192, 32768].includes(perInput) ? perInput : -1}
            options={[512, 2048, 8192, 32768].map((n) => ({ value: n, label: fmtContext(n) }))}
            onChange={(n) => set("ctx_size", unified ? n : n * slots)}
          />
          <Hint>Ahora {fmtContext(perInput)} tokens por texto.</Hint>
        </div>
      </div>
    </Panel>
  )
}

/** Paralelismo y contexto máximo del servidor Ollama (globales). */
export function OllamaGlobalCard() {
  const [data, setData] = useState<Awaited<ReturnType<typeof modelEditorApi.ollamaSettings>> | null>(null)
  useEffect(() => {
    modelEditorApi
      .ollamaSettings()
      .then(setData)
      .catch(() => setData(null))
  }, [])
  return (
    <Panel
      title="Servidor Ollama (global)"
      subtitle="Afecta a todos los modelos de Ollama. Se cambia en Ajustes y aplicarlo reinicia Ollama."
      actions={
        <Link to="/settings" className="text-xs text-sky-300 hover:underline">
          Cambiar en Ajustes
        </Link>
      }
    >
      {data ? (
        <div className="flex flex-wrap gap-6 text-sm" data-testid="ollama-global">
          <div>
            <p className="text-[11px] text-muted-foreground">Peticiones simultáneas por modelo</p>
            <p className="text-xl font-semibold tabular-nums">{data.numParallel}</p>
          </div>
          <div>
            <p className="text-[11px] text-muted-foreground">Contexto máximo</p>
            <p className="text-xl font-semibold tabular-nums">{fmtContext(data.maxContext)}</p>
          </div>
          <div className="self-center text-xs text-muted-foreground">
            KV reservada por modelo: contexto × {data.numParallel} peticiones.
          </div>
          {data.pending && (
            <Callout tone="warn">
              El servidor sigue con {data.running?.numParallel ?? "?"} y {fmtContext(data.running?.maxContext ?? null)}: falta aplicar los cambios en
              Ajustes.
            </Callout>
          )}
        </div>
      ) : (
        <Hint>No se pudo leer la configuración del servidor Ollama.</Hint>
      )}
    </Panel>
  )
}
