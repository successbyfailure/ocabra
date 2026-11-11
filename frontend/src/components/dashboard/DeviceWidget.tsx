import { Activity, Gauge, MemoryStick, Zap, type LucideIcon } from "lucide-react"
import { AreaSpark } from "@/components/gpu/AreaSpark"
import { METRIC, pct, statusColor, tempColor } from "@/components/gpu/metrics"
import { useGpuStore, type GpuHistoryPoint } from "@/stores/gpuStore"
import type {
  GPUState,
  HostStats,
  ModelActivity,
  ModelPhaseInfo,
  ModelState,
  OllamaRuntimeInfo,
  ServerPower,
} from "@/types"
import { gpuColor } from "./GpuBadge"
import { gpuLabel } from "./gpuNames"
import { StatePill, cardState } from "./ModelLiveCard"
import { BackendIcon } from "./BackendIcon"
import { fmtTps } from "./format"

const MODEL_HUES = ["#2a78d6", "#1baf7a", "#b07cf0", "#e0a93b", "#4ec9d6", "#d4679a"] as const
const gb = (mb: number) => (mb / 1024).toFixed(1)
const EMPTY_HISTORY: GpuHistoryPoint[] = []

interface BarSegment {
  key: string
  pct: number
  color: string
  title: string
  pending?: boolean
}

/** Fila de métrica: icono · barra · valor · mini tendencia. */
function MetricRow({
  icon: Icon,
  label,
  value,
  barPct,
  color,
  segments,
  secondaryPct,
  secondaryTitle,
  history,
}: {
  icon: LucideIcon
  label: string
  value: string
  barPct: number
  color: string
  segments?: BarSegment[]
  secondaryPct?: number
  secondaryTitle?: string
  history?: number[]
}) {
  return (
    <div className="grid grid-cols-[16px_64px_1fr_auto_56px] items-center gap-2 text-[11.5px]">
      <Icon size={14} style={{ color }} aria-hidden />
      <span className="text-muted-foreground">{label}</span>
      <div className="space-y-0.5">
      <div className="relative h-2.5 overflow-hidden rounded-full bg-muted" title={`${label}: ${value}`}>
        {segments && segments.length > 0 ? (
          <span className="absolute inset-0 flex">
            {segments.map((s) => (
              <span
                key={s.key}
                className="h-full border-r border-background/50"
                style={{
                  width: `${s.pct}%`,
                  background: s.pending
                    ? `repeating-linear-gradient(135deg, ${s.color}88 0 3px, transparent 3px 6px)`
                    : s.color,
                }}
                title={s.title}
              />
            ))}
          </span>
        ) : (
          <span className="absolute inset-y-0 left-0" style={{ width: `${barPct}%`, background: color }} />
        )}
      </div>
      {secondaryPct != null && secondaryPct > 0 && (
        <div className="relative h-1 overflow-hidden rounded-full bg-muted/60" title={secondaryTitle}>
          <span className="absolute inset-y-0 left-0" style={{ width: `${secondaryPct}%`, background: METRIC.memLocked }} />
        </div>
      )}
      </div>
      <span className="text-right font-medium tabular-nums">{value}</span>
      <div className="h-4">{history && history.length > 1 && <AreaSpark data={history} color={color} height={16} />}</div>
    </div>
  )
}

/** Modelos cargados en un dispositivo, con su actividad en vivo. */
function DeviceModels({
  models,
  activity,
  ollamaRuntime = {},
  emptyLabel,
  vramOnDevice,
  ramOf,
  modelStates = {},
  stuckThreshold = 300,
}: {
  models: ModelState[]
  activity: Record<string, ModelActivity>
  ollamaRuntime?: Record<string, OllamaRuntimeInfo>
  emptyLabel: string
  modelStates?: Record<string, ModelPhaseInfo>
  stuckThreshold?: number
  /** MB del modelo en este dispositivo, si se conoce (si no, el total). */
  vramOnDevice?: (m: ModelState) => number | undefined
  /** RAM del host en vivo del modelo, si se mide (Strata). */
  ramOf?: (m: ModelState) => number | undefined
}) {
  if (models.length === 0) return <p className="text-[11.5px] text-muted-foreground">{emptyLabel}</p>
  return (
    <div className="flex flex-col gap-1">
      {models.map((m, i) => {
        const act = activity[m.modelId]
        const queued = act?.queuedCount ?? act?.admission?.queuedRequests ?? 0
        const running = Math.max(0, (act?.inFlight ?? 0) - queued)
        const rt = m.backendType === "ollama" ? ollamaRuntime[m.modelId.replace(/^ollama\//, "")] : undefined
        return (
          <div key={m.modelId} className="rounded-lg bg-muted/45 px-2 py-1 text-[11.5px]">
            <div className="flex items-center justify-between gap-2">
              <span className="flex min-w-0 items-center gap-1.5">
                <span className="h-2 w-2 shrink-0 rounded-sm" style={{ background: MODEL_HUES[i % MODEL_HUES.length] }} />
                <BackendIcon backendType={m.backendType} showLabel={false} />
                <span className="truncate font-medium">{m.displayName || m.modelId}</span>
                <StatePill compact state={cardState(m, act, modelStates[m.modelId], stuckThreshold)} />
              </span>
              <span className="flex shrink-0 items-center gap-2 tabular-nums text-muted-foreground">
                {running > 0 && <span className="text-emerald-400">{running} en curso</span>}
                {queued > 0 && <span className="text-amber-400">{queued} en cola</span>}
                {act?.prefillTokensPerSecondTotal ? <span title="Prefill ahora">⇢ {fmtTps(act.prefillTokensPerSecondTotal)}</span> : null}
                {act?.generationTokensPerSecondTotal ? (
                  <span title="Generación ahora">✎ {fmtTps(act.generationTokensPerSecondTotal)}</span>
                ) : null}
                {m.vramUsedMb > 0 &&
                  (() => {
                    // "aquí/total" solo para los repartidos entre GPUs; en uno
                    // de una sola GPU el total ya es lo de esta.
                    const here = (m.currentGpu ?? []).length > 1 ? vramOnDevice?.(m) : undefined
                    return here != null && here !== m.vramUsedMb ? (
                      <span title={`${gb(here)} GB en esta GPU de ${gb(m.vramUsedMb)} GB en total`}>
                        {gb(here)}/{gb(m.vramUsedMb)} GB
                      </span>
                    ) : (
                      <span>{gb(m.vramUsedMb)} GB</span>
                    )
                  })()}
                {(ramOf?.(m) ?? m.ramUsedMb) > 0 && <span>{gb(ramOf?.(m) ?? m.ramUsedMb)} GB RAM</span>}
              </span>
            </div>
            {rt && (
              <p className="mt-0.5 text-[10px] text-muted-foreground">
                Ollama: {rt.gpuPct}% GPU{rt.cpuPct > 0 ? ` · ${rt.cpuPct}% CPU` : ""}
              </p>
            )}
          </div>
        )
      })}
    </div>
  )
}

function deviceTotals(models: ModelState[], activity: Record<string, ModelActivity>) {
  let running = 0
  let queued = 0
  let prefill = 0
  let generation = 0
  for (const m of models) {
    const act = activity[m.modelId]
    if (!act) continue
    const q = act.queuedCount ?? act.admission?.queuedRequests ?? 0
    queued += q
    running += Math.max(0, act.inFlight - q)
    prefill += act.prefillTokensPerSecondTotal ?? 0
    generation += act.generationTokensPerSecondTotal ?? 0
  }
  return { running, queued, prefill, generation }
}

function TotalsLine({ totals }: { totals: ReturnType<typeof deviceTotals> }) {
  return (
    <div className="grid grid-cols-4 gap-1.5 text-center">
      {[
        ["en curso", String(totals.running)],
        ["en cola", String(totals.queued)],
        ["prefill", fmtTps(totals.prefill)],
        ["generación", fmtTps(totals.generation)],
      ].map(([label, value]) => (
        <div key={label} className="rounded-md bg-muted/45 px-1.5 py-1">
          <span className="block text-[9.5px] font-semibold uppercase tracking-wide text-muted-foreground">{label}</span>
          <span className="block text-[12px] font-semibold tabular-nums">{value}</span>
        </div>
      ))}
    </div>
  )
}

function Header({
  kicker,
  title,
  tempC,
  color,
}: {
  kicker: string
  title: string
  tempC?: number | null
  color?: string
}) {
  return (
    <div className="mb-3 flex items-start justify-between gap-2">
      <div className="min-w-0">
        <p className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">{kicker}</p>
        <h3 className="flex items-center gap-1.5 truncate text-[15px] font-semibold tracking-tight" style={color ? { color } : undefined}>
          {color && <span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: color }} />}
          {title}
        </h3>
      </div>
      {tempC != null && (
        <span className="inline-flex shrink-0 items-center gap-1.5 rounded-full border border-border bg-muted/50 px-2.5 py-1 text-[12.5px] font-semibold tabular-nums">
          <span className="h-[7px] w-[7px] rounded-full" style={{ background: tempColor(tempC) }} />
          {tempC.toFixed(0)}°C
        </span>
      )}
    </div>
  )
}

export function GpuWidget({
  gpu,
  models,
  activity,
  vramByGpu = {},
  modelStates,
  stuckThreshold,
}: {
  gpu: GPUState
  models: ModelState[]
  activity: Record<string, ModelActivity>
  /** VRAM medida por NVML de cada modelo en cada GPU (backend). */
  vramByGpu?: Record<string, Record<string, number>>
  modelStates?: Record<string, ModelPhaseInfo>
  stuckThreshold?: number
}) {
  const hist = useGpuStore((s) => s.history[gpu.index] ?? EMPTY_HISTORY).slice(-40)
  const vramPct = pct(gpu.usedVramMb, gpu.totalVramMb)
  const lockedPct = pct(gpu.lockedVramMb, gpu.totalVramMb)
  const powPct = gpu.powerLimitW > 0 ? (gpu.powerDrawW / gpu.powerLimitW) * 100 : 0
  const gpuModels = models.filter((m) => m.status === "loaded" && (m.currentGpu ?? []).includes(gpu.index))

  // VRAM por consumidor: lo que NVML mide para los procesos del modelo en
  // ESTA tarjeta. Solo si no hay medida (Ollama comparte proceso entre
  // modelos) se reparte el total a partes iguales, y se dice.
  const segments: BarSegment[] = []
  let attributed = 0
  gpuModels.forEach((m, i) => {
    const measured = vramByGpu[m.modelId]?.[String(gpu.index)]
    const n = (m.currentGpu ?? []).length || 1
    const share = measured ?? m.vramUsedMb / n
    attributed += share
    segments.push({
      key: m.modelId,
      pct: pct(share, gpu.totalVramMb),
      color: MODEL_HUES[i % MODEL_HUES.length],
      title:
        measured != null
          ? `${m.displayName} — ${gb(share)} GB en esta GPU (medido)`
          : n > 1
            ? `${m.displayName} — ~${gb(share)} GB (sin medida por GPU: reparto estimado a partes iguales entre ${n} GPUs)`
            : `${m.displayName} — ${gb(share)} GB`,
    })
  })
  const others = gpu.usedVramMb - attributed
  if (segments.length > 0 && others > gpu.totalVramMb * 0.005) {
    segments.push({ key: "otros", pct: pct(others, gpu.totalVramMb), color: "#6b7280", title: `${gb(others)} GB fuera de los modelos de oCabra (driver, otros procesos)` })
  } else if (segments.length > 0 && others < -gpu.totalVramMb * 0.005) {
    segments.push({ key: "reservado", pct: 0, color: METRIC.memLocked, title: `${gb(-others)} GB reservados sin residir`, pending: true })
  }
  const hot = gpu.temperatureC >= 78 || vramPct >= 90

  return (
    <article
      className={`rounded-2xl border border-t-4 bg-card p-4 shadow-sm ${hot ? "border-red-500/60" : "border-border"}`}
      style={{ borderTopColor: gpuColor(gpu.index) }}
      data-gpu={gpu.index}
    >
      <Header kicker="GPU" title={gpuLabel(gpu.index, [gpu])} tempC={gpu.temperatureC} color={gpuColor(gpu.index)} />
      <div className="space-y-1.5">
        <MetricRow icon={Zap} label="Potencia" value={`${Math.round(gpu.powerDrawW)}/${Math.round(gpu.powerLimitW)} W`} barPct={powPct} color={METRIC.power} history={hist.map((p) => p.powerPct)} />
        <MetricRow icon={Gauge} label="Uso" value={`${gpu.utilizationPct.toFixed(0)}%`} barPct={gpu.utilizationPct} color={METRIC.util} history={hist.map((p) => p.util)} />
        <MetricRow
          icon={MemoryStick}
          label="VRAM"
          value={`${gb(gpu.usedVramMb)}/${gb(gpu.totalVramMb)} GB`}
          barPct={vramPct}
          color={statusColor(vramPct, 85, 95) === "#16a34a" ? METRIC.mem : statusColor(vramPct, 85, 95)}
          segments={segments}
          secondaryPct={lockedPct}
          secondaryTitle={`${gb(gpu.lockedVramMb)} GB bloqueados`}
          history={hist.map((p) => p.vramPct)}
        />
      </div>
      {gpu.lockedVramMb > 0 && (
        <p className="mt-1 text-[10.5px] text-muted-foreground">
          <span className="mr-1 inline-block h-1 w-3 rounded-full align-middle" style={{ background: METRIC.memLocked }} />
          VRAM bloqueada: {gb(gpu.lockedVramMb)} GB
        </p>
      )}

      <div className="mt-3 space-y-2 border-t border-border pt-3">
        <TotalsLine totals={deviceTotals(gpuModels, activity)} />
        <DeviceModels
          models={gpuModels}
          activity={activity}
          emptyLabel="Sin modelos en esta GPU"
          vramOnDevice={(m) => vramByGpu[m.modelId]?.[String(gpu.index)]}
          modelStates={modelStates}
          stuckThreshold={stuckThreshold}
        />
      </div>

      <details className="group mt-3 border-t border-border">
        <summary className="flex cursor-pointer list-none items-center justify-between pt-2.5 text-[11.5px] font-medium text-muted-foreground [&::-webkit-details-marker]:hidden">
          <span>Procesos ({gpu.processes.length})</span>
          <span className="text-muted-foreground/70 transition-transform group-open:rotate-180">▾</span>
        </summary>
        <div className="flex flex-col gap-1 pt-2">
          {gpu.processes.length === 0 ? (
            <p className="text-[11.5px] text-muted-foreground">Sin procesos en la GPU</p>
          ) : (
            gpu.processes.map((p) => (
              <div key={`${p.processType}-${p.pid}`} className="flex items-center justify-between gap-2 rounded-lg bg-muted/50 px-2.5 py-1 text-[11px]">
                <span className="min-w-0 truncate font-mono">{p.pid} · {p.processName ?? "desconocido"}</span>
                <span className="shrink-0 tabular-nums text-muted-foreground">{p.processType} · {p.usedVramMb.toLocaleString()} MB</span>
              </div>
            ))
          )}
        </div>
      </details>
    </article>
  )
}

export interface HostHistoryPoint {
  t: number
  cpuPct: number
  memPct: number
  powerW: number | null
}

export function CpuWidget({
  stats,
  serverPower,
  history,
  models,
  activity,
  ollamaRuntime,
  modelStates,
  stuckThreshold,
  ramByModel = {},
}: {
  stats: HostStats
  serverPower: ServerPower | null
  history: HostHistoryPoint[]
  models: ModelState[]
  activity: Record<string, ModelActivity>
  ollamaRuntime: Record<string, OllamaRuntimeInfo>
  modelStates?: Record<string, ModelPhaseInfo>
  stuckThreshold?: number
  /** RAM del host en vivo por modelo (Strata y otros que la usan a propósito). */
  ramByModel?: Record<string, number>
}) {
  const cpuPower = serverPower?.cpuPowerW ?? null
  // Parte de la RAM de cada modelo en la barra; el resto, "otros".
  const ramOf = (m: ModelState) => ramByModel[m.modelId] ?? (m.ramUsedMb > 0 ? m.ramUsedMb : undefined)
  const ramSegments: BarSegment[] = []
  let ramAttributed = 0
  models.forEach((m, i) => {
    const mb = ramOf(m)
    if (!mb || stats.memTotalMb <= 0) return
    ramAttributed += mb
    ramSegments.push({
      key: m.modelId,
      pct: pct(mb, stats.memTotalMb),
      color: MODEL_HUES[i % MODEL_HUES.length],
      title: `${m.displayName} — ${gb(mb)} GB de RAM`,
    })
  })
  if (ramSegments.length > 0 && stats.memUsedMb > ramAttributed) {
    ramSegments.push({
      key: "otros",
      pct: pct(stats.memUsedMb - ramAttributed, stats.memTotalMb),
      color: "#6b7280",
      title: `${gb(stats.memUsedMb - ramAttributed)} GB del resto del sistema`,
    })
  }
  const powerHistory = history.map((p) => p.powerW ?? 0)
  // Escala real: el máximo de la CPU (ajuste, RAPL o TDP del modelo). Solo si
  // no se conoce, el pico observado, y entonces se dice.
  const powerMax = stats.cpuPowerMaxW ?? null
  const peak = powerMax ?? Math.max(cpuPower ?? 0, ...powerHistory, 1)
  const powerScaleLabel =
    powerMax != null
      ? stats.cpuPowerMaxSource === "tdp"
        ? `TDP ${Math.round(powerMax)} W`
        : stats.cpuPowerMaxSource === "rapl"
          ? `límite ${Math.round(powerMax)} W`
          : `máx. ${Math.round(powerMax)} W`
      : "escala: pico observado"
  return (
    <article className="rounded-2xl border border-border bg-card p-4 shadow-sm" data-device="cpu">
      <Header
        kicker="CPU"
        title={`${(stats.cpuModel ?? "CPU").replace(/^AMD |^Intel\(R\) |\s*\d+-Core Processor$/g, "")} · ${stats.cpuCountPhysical}C/${stats.cpuCount}T`}
        tempC={serverPower?.cpuTempC ?? null}
      />
      <div className="space-y-1.5">
        <MetricRow
          icon={Zap}
          label="Potencia"
          value={cpuPower != null ? `${Math.round(cpuPower)}/${powerMax != null ? Math.round(powerMax) : "?"} W` : "—"}
          barPct={cpuPower != null ? Math.min(100, (cpuPower / peak) * 100) : 0}
          color={METRIC.power}
          history={powerHistory}
        />
        <MetricRow icon={Gauge} label="Uso" value={`${stats.cpuPct.toFixed(0)}%`} barPct={stats.cpuPct} color={METRIC.util} history={history.map((p) => p.cpuPct)} />
        <MetricRow
          icon={MemoryStick}
          label="RAM"
          value={`${gb(stats.memUsedMb)}/${gb(stats.memTotalMb)} GB`}
          barPct={stats.memPct}
          color={METRIC.mem}
          segments={ramSegments.length > 0 ? ramSegments : undefined}
          history={history.map((p) => p.memPct)}
        />
        <MetricRow
          icon={Activity}
          label="Swap"
          value={`${gb(stats.swapUsedMb)}/${gb(stats.swapTotalMb)} GB`}
          barPct={stats.swapTotalMb > 0 ? stats.swapPct : 0}
          color={METRIC.memLocked}
        />
      </div>
      <p className="mt-1.5 text-[10.5px] tabular-nums text-muted-foreground">
        Load average (1/5/15 min): {stats.loadAvg1m} / {stats.loadAvg5m} / {stats.loadAvg15m}
        <span className="ml-2" title="Contra qué se mide la barra de potencia">
          · potencia sobre {powerScaleLabel}
        </span>
      </p>
      <div className="mt-3 space-y-2 border-t border-border pt-3">
        <TotalsLine totals={deviceTotals(models, activity)} />
        <DeviceModels
          models={models}
          activity={activity}
          ollamaRuntime={ollamaRuntime}
          emptyLabel="Ningún modelo corriendo en CPU"
          ramOf={ramOf}
          modelStates={modelStates}
          stuckThreshold={stuckThreshold}
        />
      </div>
    </article>
  )
}
