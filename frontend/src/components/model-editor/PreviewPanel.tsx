import { useState } from "react"
import { ChevronDown, ChevronRight, Loader2, RefreshCw, TerminalSquare } from "lucide-react"
import { gpuColor } from "@/components/dashboard/GpuBadge"
import { shortGpuName } from "@/components/dashboard/gpuNames"
import { fmtContext } from "@/components/dashboard/format"
import { cn } from "@/lib/cn"
import type { GPUState } from "@/types"
import { formatValue } from "./draft"
import type { PreviewGpu } from "./types"
import type { LadderRung } from "./usePresetLadder"
import { GpuBadge } from "@/components/dashboard/GpuBadge"
import type { ModelEditorState } from "./useModelEditor"
import { Callout, Chip } from "./ui"

/** Colores de las partes de la huella de memoria, iguales en toda la página. */
export const PART_META: Record<string, { label: string; color: string }> = {
  weights: { label: "pesos", color: "#6366f1" },
  kv: { label: "KV", color: "#0ea5e9" },
  rs_cache: { label: "rs_cache", color: "#14b8a6" },
  fixed: { label: "fijo", color: "#64748b" },
  draft: { label: "draft", color: "#f59e0b" },
  overhead: { label: "extra", color: "#94a3b8" },
  model: { label: "modelo", color: "#6366f1" },
  preset: { label: "con el preset", color: "#6366f1" },
}

export function fmtMb(mb: number | null | undefined): string {
  if (mb == null) return "—"
  return mb >= 1024 ? `${(mb / 1024).toFixed(mb >= 10240 ? 0 : 1)} GB` : `${Math.round(mb)} MB`
}

/** Barra de una GPU: lo que ocupa el resto, la huella prevista de este modelo
 *  por partes y el hueco libre. */
export function GpuMemoryBar({
  gpu,
  share,
  ownLiveMb,
  compact,
}: {
  gpu: GPUState
  share: PreviewGpu | undefined
  ownLiveMb: number
  compact?: boolean
}) {
  const total = Math.max(1, gpu.totalVramMb)
  const others = Math.max(0, gpu.usedVramMb - ownLiveMb)
  const mine = share?.mb ?? 0
  const room = total - others
  const fits = mine <= room
  const fitsAlone = mine <= total
  const pct = (mb: number) => `${Math.min(100, (mb / total) * 100)}%`
  const color = gpuColor(gpu.index)
  return (
    <div className="space-y-1" data-gpu-bar={gpu.index}>
      <div className="flex items-center justify-between gap-2 text-[11px]">
        <span className="flex items-center gap-1.5 font-medium" style={{ color }}>
          <span className="h-2 w-2 rounded-full" style={{ background: color }} />#{gpu.index} {shortGpuName(gpu.name)}
        </span>
        <span className="tabular-nums text-muted-foreground">
          {share ? (
            <>
              <span className={fits ? "text-emerald-300" : fitsAlone ? "text-amber-300" : "text-red-300"}>{fmtMb(mine)}</span> / {fmtMb(room)} libres
            </>
          ) : (
            <>sin usar · {fmtMb(room)} libres</>
          )}
        </span>
      </div>
      <div className={cn("relative flex overflow-hidden rounded-md bg-muted/50", compact ? "h-2.5" : "h-4")}>
        <div className="h-full bg-slate-600/70" style={{ width: pct(others) }} title={`Otros: ${fmtMb(others)}`} />
        {share &&
          Object.entries(Object.keys(share.breakdown).length ? share.breakdown : { model: share.mb }).map(([part, mb]) => (
            <div
              key={part}
              className="h-full border-l border-black/20"
              style={{ width: pct(mb), background: PART_META[part]?.color ?? "#a78bfa" }}
              title={`${PART_META[part]?.label ?? part}: ${fmtMb(mb)}`}
            />
          ))}
        {share && !fits && (
          <div className="absolute inset-y-0 right-0 w-1 bg-red-500" title="No cabe con lo que hay cargado" />
        )}
      </div>
    </div>
  )
}

export function PreviewPanel({
  editor,
  gpus,
  liveVramByGpu,
  presets = [],
}: {
  editor: ModelEditorState
  gpus: GPUState[]
  liveVramByGpu: Record<string, number>
  presets?: LadderRung[]
}) {
  const [showCmd, setShowCmd] = useState(false)
  const [showChanges, setShowChanges] = useState(true)
  const p = editor.preview
  const est = p?.estimate
  // Encaje de la base (sin preset): lo que el borrador ocuparía en cada GPU
  // frente a lo que dejan libre los demás. Las barras viven en Capacidad.
  const shares = new Map((p?.gpus ?? []).map((g) => [g.index, g]))
  const used = gpus.filter((g) => shares.has(g.index))
  const verdict = (() => {
    if (!p || !est || !used.length) return null
    const fitsNow = used.every((g) => (shares.get(g.index)?.mb ?? 0) <= g.totalVramMb - Math.max(0, g.usedVramMb - (liveVramByGpu[String(g.index)] ?? 0)))
    const fitsEmpty = used.every((g) => (shares.get(g.index)?.mb ?? 0) <= g.totalVramMb)
    if (fitsNow) return { tone: "ok" as const, text: "cabe ahora" }
    if (fitsEmpty) return { tone: "warn" as const, text: "cabe desalojando otros" }
    return { tone: "error" as const, text: "no cabe" }
  })()
  const reloads = editor.changes.filter((c) => c.reload).length
  const loaded = editor.model?.status === "loaded"
  const requested = est?.requested_context_length ?? null

  return (
    <aside className="space-y-3 text-sm" data-testid="preview-panel">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Vista previa</h3>
        {editor.previewLoading ? <Loader2 size={14} className="animate-spin text-muted-foreground" /> : <RefreshCw size={12} className="text-muted-foreground/50" />}
      </div>

      <div className="space-y-1.5 rounded-xl border border-border bg-card p-3" data-testid="preview-fit">
        <div className="text-xs font-medium">Encaje</div>
        {verdict ? (
          <div className="flex flex-wrap items-center gap-1.5 text-[11px]" data-testid="preview-base">
            <span className={verdict.tone === "ok" ? "text-emerald-300" : verdict.tone === "warn" ? "text-amber-300" : "text-red-300"}>
              {verdict.tone === "error" ? "✗" : "✓"}
            </span>
            <span className="font-medium">{presets.length ? "Base" : "Esta configuración"}</span>
            {editor.schema?.isLlm && requested ? <span className="tabular-nums text-muted-foreground">{fmtContext(requested)}</span> : null}
            {est?.model_loading_memory_mb ? <span className="tabular-nums text-muted-foreground">· {fmtMb(est.model_loading_memory_mb)}</span> : null}
            <span className="text-muted-foreground">· {verdict.text}</span>
            {used.map((g) => (
              <GpuBadge key={g.index} index={g.index} gpus={gpus} />
            ))}
          </div>
        ) : (
          <p className="text-[11px] text-muted-foreground">{editor.previewLoading ? "Calculando…" : "Sin estimación de memoria."}</p>
        )}
        {presets.map(({ preset, resolution }) => (
          <div key={preset.presetKey} className="flex flex-wrap items-center gap-1.5 text-[11px]">
            <span className={resolution?.fits === false ? "text-red-300" : "text-emerald-300"}>{resolution?.fits === false ? "✗" : "✓"}</span>
            <span className="font-medium">{preset.label}</span>
            {preset.isDefault && <span className="text-amber-300">★</span>}
            <span className="tabular-nums text-muted-foreground">{fmtContext(resolution?.effectiveContext ?? preset.contextTokens)}</span>
            <span className="text-muted-foreground">· p{resolution?.effectiveParallelism ?? preset.parallelism}</span>
            {resolution?.estimatedVramMb ? <span className="tabular-nums text-muted-foreground">· {fmtMb(resolution.estimatedVramMb)}</span> : null}
            {(resolution?.targetGpus ?? preset.preferredGpus ?? []).map((g) => (
              <GpuBadge key={g} index={g} gpus={gpus} />
            ))}
          </div>
        ))}
        {est?.warning && <p className="text-[11px] text-amber-300">{est.warning}</p>}
        <p className="text-[10.5px] text-muted-foreground">Las barras por GPU están en Capacidad.</p>
      </div>

      {p?.errors?.length ? (
        <Callout tone="error">
          <p className="mb-1 font-medium">Hay opciones no válidas</p>
          <ul className="list-disc pl-4">
            {p.errors.map((e) => (
              <li key={e.key}>
                {e.label}: {e.message}
              </li>
            ))}
          </ul>
        </Callout>
      ) : null}

      <div className="rounded-xl border border-border bg-card p-3">
        <button type="button" className="flex w-full items-center justify-between text-xs font-medium" onClick={() => setShowChanges((v) => !v)}>
          <span className="flex items-center gap-1">
            {showChanges ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
            Cambios pendientes ({editor.changes.length})
          </span>
          {reloads > 0 && loaded && <Chip tone="warn">{reloads} requieren recarga</Chip>}
        </button>
        {showChanges && (
          <ul className="mt-2 space-y-1" data-testid="pending-changes">
            {editor.changes.length === 0 && <li className="text-[11px] text-muted-foreground">Sin cambios.</li>}
            {editor.changes.map((c) => (
              <li key={c.key} className="text-[11px]">
                <span className="font-medium text-foreground">{c.label}</span>{" "}
                <span className="text-muted-foreground">
                  {c.key === "schedules" ? "modificados" : `${formatValue(c.from)} → ${formatValue(c.to)}`}
                </span>
                {c.reload && <span className="ml-1 text-amber-300/80">⟳</span>}
              </li>
            ))}
          </ul>
        )}
      </div>

      {p?.launch?.argv && (
        <div className="rounded-xl border border-border bg-card p-3">
          <button type="button" className="flex w-full items-center gap-1 text-xs font-medium" onClick={() => setShowCmd((v) => !v)}>
            {showCmd ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
            <TerminalSquare size={12} /> Comando de arranque
          </button>
          {showCmd && (
            <pre className="mt-2 max-h-64 overflow-auto whitespace-pre-wrap break-all rounded-md bg-black/40 p-2 font-mono text-[10.5px] leading-relaxed text-slate-300" data-testid="launch-command">
              {p.launch.env?.CUDA_VISIBLE_DEVICES !== undefined ? `CUDA_VISIBLE_DEVICES=${p.launch.env.CUDA_VISIBLE_DEVICES} ` : ""}
              {formatArgv(p.launch.argv)}
            </pre>
          )}
        </div>
      )}
      {p?.launch?.error && <p className="text-[11px] text-amber-300">Sin comando: {p.launch.error}</p>}
    </aside>
  )
}

/** Un flag por línea para que se lea. */
export function formatArgv(argv: string[]): string {
  const lines: string[] = []
  for (const arg of argv) {
    if (arg.startsWith("--") || lines.length === 0) lines.push(arg)
    else lines[lines.length - 1] += ` ${arg}`
  }
  return lines.join(" \\\n  ")
}

/** En pantallas estrechas la vista previa vive en una barra inferior que se
 *  despliega: resumen (cambios, errores, si cabe) y el panel completo al tocar. */
export function MobilePreviewBar(props: Parameters<typeof PreviewPanel>[0]) {
  const [open, setOpen] = useState(false)
  const { editor } = props
  const errors = editor.previewErrors.length
  const changes = editor.changes.length
  const reloads = editor.changes.filter((c) => c.reload).length
  return (
    <div className="fixed inset-x-0 bottom-0 z-30 lg:hidden" data-testid="mobile-preview">
      {open && (
        <div className="max-h-[70vh] overflow-y-auto border-t border-border bg-background p-3 shadow-2xl">
          <PreviewPanel {...props} />
        </div>
      )}
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between gap-2 border-t border-border bg-card px-4 py-2 text-xs"
        aria-expanded={open}
      >
        <span className="font-medium">Vista previa</span>
        <span className="flex items-center gap-2">
          {errors > 0 && <Chip tone="error">{errors} con error</Chip>}
          <Chip tone={changes ? "info" : "muted"}>{changes} cambio{changes === 1 ? "" : "s"}</Chip>
          {reloads > 0 && <Chip tone="warn">⟳ {reloads}</Chip>}
          {open ? <ChevronDown size={14} /> : <ChevronRight size={14} className="-rotate-90" />}
        </span>
      </button>
    </div>
  )
}
