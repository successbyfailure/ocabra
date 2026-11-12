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
import { Callout, Hint, Panel } from "../ui"
import type { LadderRung } from "../usePresetLadder"
import { ContextSection } from "./ContextSection"
import { GpuSection } from "./GpuSection"
import { PolicyPanel, ProfileLadder } from "./PresetsSection"
import type { SectionProps } from "./types"

/** Familias de backend por cómo gestionan la memoria. */
export type CapacityFamily = "llama" | "engine" | "ollama" | "fixed"

export function capacityFamily(backend: string | undefined, isLlm: boolean): CapacityFamily {
  if (backend === "llama_cpp" || backend === "llamacpp_mks" || backend === "bitnet") return "llama"
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
  const family = capacityFamily(backend, Boolean(schema?.isLlm))
  const canPreset = family !== "ollama" && Boolean(schema?.isLlm) && ladder.profile != null
  const allPresets = ladder.profile?.runtimePresets ?? []
  const requested = search.get("r")
  const selected =
    requested === "new" && canPreset && !readOnly
      ? "new"
      : requested && allPresets.some((p) => p.presetKey === requested)
        ? requested
        : "base"
  const select = (key: string) =>
    setSearch(
      (q) => {
        if (key === "base") q.delete("r")
        else q.set("r", key)
        return q
      },
      { replace: true },
    )

  if (!editor.draft || !schema) return null
  return (
    <div className="space-y-4" data-testid="capacity-section">
      <Hint>{FAMILY_TEXT[family]}</Hint>
      <RungSelector {...props} family={family} selected={selected} onSelect={select} canPreset={canPreset && !readOnly} />

      {selected === "base" ? (
        <BaseCanvas {...props} family={family} hasPresets={allPresets.some((p) => p.enabled)} />
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
          onFocusDone={(key) => select(key ?? "base")}
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
}: SectionProps & {
  ladder: { rungs: LadderRung[]; profile: import("@/types").ModelProfile | null }
  family: CapacityFamily
  selected: string
  onSelect: (key: string) => void
  canPreset: boolean
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
        presets.length
          ? "La base es lo que se carga sin preset; cada preset la sustituye en contexto, concurrencia, KV y GPUs. Elige uno para verlo y editarlo."
          : undefined
      }
    >
      <div className="flex flex-wrap gap-2" data-testid="rung-selector">
        <Item
          k="base"
          title={presets.length ? "Base (sin preset)" : "Configuración"}
          ctx={base.ctx}
          gpuList={base.gpus}
          mb={base.total}
          fits={baseFits}
          hint="Se guarda con el botón Guardar de la página."
        />
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
      <GpuSection {...quiet} />
      {family !== "fixed" && <ContextSection {...quiet} />}
    </div>
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
