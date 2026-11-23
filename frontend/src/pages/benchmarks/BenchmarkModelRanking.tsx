import { useMemo, useState } from "react"
import { Activity, BadgeCheck, CircleGauge, Cpu, Info, SlidersHorizontal, Sparkles, Trophy } from "lucide-react"
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts"
import type { BenchmarkRun } from "@/types"
import {
  buildModelRanking,
  DEFAULT_CUSTOM_WEIGHTS,
  normalizeWeights,
  RANKING_PROFILES,
  rankingFilterOptions,
  type ModelRankingRow,
  type RankingFactor,
  type RankingProfileId,
  type RankingWeights,
} from "./modelRanking"

const FACTOR_LABELS: Record<RankingFactor, string> = {
  corrala: "Corrala",
  escriba: "Escriba",
  bfcl: "BFCL",
  ifeval: "IFEval",
  reliability: "Fiabilidad",
  performance: "Rendimiento",
  efficiency: "Eficiencia GPU",
}

const DIMENSION_LABELS = {
  corrala: "Corrala",
  escriba: "Escriba",
  bfcl: "BFCL",
  ifeval: "IFEval",
}

export function paretoIds(rows: ModelRankingRow[]): Set<string> {
  return new Set(rows.filter((row) => row.eligible && !rows.some((candidate) => (
    candidate.eligible
    && candidate.id !== row.id
    && candidate.quality >= row.quality
    && candidate.performance >= row.performance
    && (candidate.quality > row.quality || candidate.performance > row.performance)
  ))).map((row) => row.id))
}

function pct(value: number): string {
  return `${value.toFixed(1)}%`
}

function shortModel(value: string): string {
  return value.length > 31 ? `${value.slice(0, 29)}…` : value
}

function ScoreExplanation({ row, weights }: { row: ModelRankingRow; weights: RankingWeights }) {
  const normalized = normalizeWeights(weights)
  return (
    <details>
      <summary className="cursor-pointer text-xs text-primary hover:underline">Cómo se calcula</summary>
      <div className="mt-2 grid gap-1.5 rounded-md bg-muted/30 p-2 text-[11px]">
        {(Object.keys(FACTOR_LABELS) as RankingFactor[]).map((factor) => (
          <div key={factor} className="flex items-center justify-between gap-3">
            <span className="text-muted-foreground">{FACTOR_LABELS[factor]} · {(normalized[factor] * 100).toFixed(0)}%</span>
            <span className="font-mono">+{(row.contributions[factor] * 100).toFixed(1)}</span>
          </div>
        ))}
        <div className="mt-1 flex items-center justify-between border-t border-border pt-1.5 font-medium">
          <span>Total</span><span className="font-mono">{row.score.toFixed(1)}</span>
        </div>
      </div>
    </details>
  )
}

function RankingTable({ rows, weights, pareto }: {
  rows: ModelRankingRow[]
  weights: RankingWeights
  pareto: Set<string>
}) {
  return (
    <div className="overflow-x-auto rounded-lg border border-border">
      <table className="w-full min-w-[1040px] text-left text-xs">
        <thead className="bg-muted/30 text-muted-foreground">
          <tr>
            <th className="px-3 py-2">#</th>
            <th className="px-3 py-2">Modelo + backend</th>
            <th className="px-3 py-2 text-right">Puntuación</th>
            <th className="px-3 py-2 text-right">Calidad</th>
            <th className="px-3 py-2 text-right">Fiabilidad</th>
            <th className="px-3 py-2 text-right">Rendimiento</th>
            <th className="px-3 py-2 text-right">Eficiencia</th>
            <th className="px-3 py-2">Cobertura</th>
            <th className="px-3 py-2">Entorno</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={row.id} className={`border-t border-border/60 ${row.eligible ? "" : "opacity-60"}`}>
              <td className="px-3 py-3 align-top font-mono text-muted-foreground">{row.eligible ? index + 1 : "—"}</td>
              <td className="px-3 py-3 align-top">
                <p className="font-mono font-medium">{row.model}</p>
                <p className="mt-0.5 text-muted-foreground">{row.backend} · perfil {row.profile}</p>
                <div className="mt-1.5 flex flex-wrap gap-1">
                  {pareto.has(row.id) && <span className="rounded-full bg-violet-500/15 px-1.5 py-0.5 text-[10px] text-violet-300">frontera calidad–velocidad</span>}
                  {!row.eligible && <span className="rounded-full bg-amber-500/15 px-1.5 py-0.5 text-[10px] text-amber-300">sin cobertura mínima</span>}
                </div>
              </td>
              <td className="px-3 py-3 text-right align-top">
                <p className={`text-base font-semibold ${index === 0 && row.eligible ? "text-emerald-400" : ""}`}>{row.score.toFixed(1)}</p>
                <ScoreExplanation row={row} weights={weights} />
              </td>
              <td className="px-3 py-3 text-right align-top tabular-nums">{pct(row.quality)}</td>
              <td className="px-3 py-3 text-right align-top tabular-nums">{pct(row.reliability)}</td>
              <td className="px-3 py-3 text-right align-top tabular-nums">{pct(row.performance)}</td>
              <td className="px-3 py-3 text-right align-top tabular-nums">{pct(row.efficiency)}</td>
              <td className="px-3 py-3 align-top">
                <p className="font-medium">{row.dimensions.length}/4 · {pct(row.coverage * 100)}</p>
                <p className="mt-1 text-[10px] text-muted-foreground">{row.dimensions.map((item) => DIMENSION_LABELS[item]).join(" · ")}</p>
              </td>
              <td className="px-3 py-3 align-top text-[10px] text-muted-foreground">
                <p>{row.versions.join(" · ")}</p>
                <p className="mt-1">{row.hardware.join(" · ")}</p>
                <details className="mt-1"><summary className="cursor-pointer">{row.sources.length} fuentes</summary><div className="mt-1 space-y-1">{row.sources.map((source) => <p key={`${source.dimension}-${source.runId}`}>{DIMENSION_LABELS[source.dimension]} · {source.suite} · {source.runId.slice(0, 8)}</p>)}</div></details>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function BenchmarkModelRanking({ runs }: { runs: BenchmarkRun[] }) {
  const [profileId, setProfileId] = useState<RankingProfileId>("balanced")
  const [customWeights, setCustomWeights] = useState<RankingWeights>(DEFAULT_CUSTOM_WEIGHTS)
  const [ocabraVersion, setOcabraVersion] = useState("")
  const [hardware, setHardware] = useState("")
  const options = useMemo(() => rankingFilterOptions(runs), [runs])
  const activeProfile = RANKING_PROFILES.find((profile) => profile.id === profileId)
  const weights = activeProfile?.weights ?? customWeights
  const rows = useMemo(() => buildModelRanking(runs, weights, { ocabraVersion, hardware }), [hardware, ocabraVersion, runs, weights])
  const pareto = useMemo(() => paretoIds(rows), [rows])
  const eligible = rows.filter((row) => row.eligible)
  const leader = eligible[0]
  const practicalTies = new Set(eligible.filter((row) => leader && leader.score - row.score < 1).map((row) => row.id))
  const fastest = [...eligible].sort((left, right) => right.performance - left.performance)[0]
  const mostReliable = [...eligible].sort((left, right) => right.reliability - left.reliability)[0]
  const chartRows = rows.slice(0, 12).map((row) => ({ ...row, label: shortModel(row.model) }))
  const normalized = normalizeWeights(weights)

  if (rows.length === 0 && options.versions.length === 0) return null

  return (
    <section className="overflow-hidden rounded-xl border border-border bg-card">
      <div className="flex flex-wrap items-end justify-between gap-4 border-b border-border px-4 py-4">
        <div>
          <p className="text-xs font-medium uppercase tracking-wider text-primary">Ranking global</p>
          <h2 className="mt-1 flex items-center gap-2 text-lg font-semibold"><Trophy size={18} /> Mejores modelos para producción</h2>
          <p className="mt-1 max-w-2xl text-xs text-muted-foreground">Compara cada perfil con su backend. La calidad ausente resta puntos y hacen falta al menos dos baterías para entrar en la clasificación.</p>
        </div>
        <div className="flex flex-wrap gap-2">
          <label className="space-y-1 text-xs text-muted-foreground"><span className="block">Versión oCabra</span><select value={ocabraVersion} onChange={(event) => setOcabraVersion(event.target.value)} className="rounded-md border border-border bg-background px-2.5 py-2 text-sm text-foreground"><option value="">Todas</option>{options.versions.map((value) => <option key={value} value={value}>{value}</option>)}</select></label>
          <label className="space-y-1 text-xs text-muted-foreground"><span className="block">Hardware</span><select value={hardware} onChange={(event) => setHardware(event.target.value)} className="max-w-64 rounded-md border border-border bg-background px-2.5 py-2 text-sm text-foreground"><option value="">Todo el hardware</option>{options.hardware.map((value) => <option key={value} value={value}>{value}</option>)}</select></label>
        </div>
      </div>

      <div className="space-y-5 p-4">
        <div>
          <p className="mb-2 flex items-center gap-1.5 text-xs font-medium uppercase tracking-wide text-muted-foreground"><SlidersHorizontal size={13} /> Objetivo</p>
          <div className="flex flex-wrap gap-2">
            {RANKING_PROFILES.map((profile) => <button key={profile.id} type="button" onClick={() => setProfileId(profile.id)} title={profile.description} className={`rounded-full border px-3 py-1.5 text-xs ${profileId === profile.id ? "border-primary bg-primary/10 text-primary" : "border-border text-muted-foreground hover:text-foreground"}`}>{profile.label}</button>)}
            <button type="button" onClick={() => setProfileId("custom")} className={`rounded-full border px-3 py-1.5 text-xs ${profileId === "custom" ? "border-primary bg-primary/10 text-primary" : "border-border text-muted-foreground hover:text-foreground"}`}>Personalizado</button>
          </div>
          <p className="mt-2 text-xs text-muted-foreground">{activeProfile?.description ?? "Ajusta cada peso; la puntuación los normaliza automáticamente al 100%."}</p>
        </div>

        {profileId === "custom" && <div className="grid gap-3 rounded-lg border border-border bg-background p-3 sm:grid-cols-2 xl:grid-cols-4">{(Object.keys(FACTOR_LABELS) as RankingFactor[]).map((factor) => <label key={factor} className="text-xs"><span className="flex justify-between"><span>{FACTOR_LABELS[factor]}</span><span className="font-mono text-muted-foreground">{(normalized[factor] * 100).toFixed(0)}%</span></span><input type="range" min="0" max="50" step="1" value={customWeights[factor]} onChange={(event) => setCustomWeights((current) => ({ ...current, [factor]: Number(event.target.value) }))} className="mt-1 w-full accent-primary" /></label>)}</div>}

        {rows.length === 0 ? <div className="rounded-lg border border-dashed border-border py-10 text-center"><Info size={22} className="mx-auto text-muted-foreground" /><p className="mt-2 text-sm">No hay resultados para esta cohorte.</p><p className="text-xs text-muted-foreground">Prueba otra versión o configuración de hardware.</p></div> : <>
          <div className="grid gap-3 sm:grid-cols-3">
            {[
              { icon: <Sparkles size={15} />, label: "Mejor equilibrio", row: leader, value: leader?.score },
              { icon: <Activity size={15} />, label: "Mejor rendimiento", row: fastest, value: fastest?.performance },
              { icon: <BadgeCheck size={15} />, label: "Mayor fiabilidad", row: mostReliable, value: mostReliable?.reliability },
            ].map(({ icon, label, row, value }) => <div key={label} className="rounded-lg border border-border bg-background p-3"><p className="flex items-center gap-1.5 text-xs text-muted-foreground">{icon}{label}</p>{row ? <><p className="mt-2 truncate font-mono text-sm font-semibold" title={row.model}>{row.model}</p><p className="text-xs text-muted-foreground">{row.backend}</p><p className="mt-1 text-lg font-semibold text-emerald-400">{typeof value === "number" ? value.toFixed(1) : "—"}</p></> : <p className="mt-3 text-sm text-muted-foreground">Faltan resultados comparables</p>}</div>)}
          </div>

          <div className="grid gap-4 xl:grid-cols-2">
            <div className="rounded-lg border border-border bg-background p-3">
              <h3 className="flex items-center gap-1.5 text-sm font-medium"><CircleGauge size={15} /> Clasificación ponderada</h3>
              <p className="text-xs text-muted-foreground">La cobertura incompleta reduce la puntuación total.</p>
              <div className="mt-3" style={{ height: Math.max(280, chartRows.length * 34) }}><ResponsiveContainer width="100%" height="100%"><BarChart data={chartRows} layout="vertical" margin={{ left: 4, right: 24 }}><CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="hsl(var(--border))" /><XAxis type="number" domain={[0, 100]} tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }} axisLine={false} tickLine={false} /><YAxis type="category" dataKey="label" width={180} tick={{ fontSize: 10, fill: "hsl(var(--foreground))" }} axisLine={false} tickLine={false} /><Tooltip formatter={(value) => [`${Number(value).toFixed(1)} / 100`, "Puntuación"]} contentStyle={{ background: "hsl(var(--popover))", border: "1px solid hsl(var(--border))", borderRadius: 8, fontSize: 12 }} /><Bar dataKey="score" radius={[0, 5, 5, 0]}>{chartRows.map((row, index) => <Cell key={row.id} fill={!row.eligible ? "hsl(var(--muted-foreground) / 0.25)" : index === 0 ? "hsl(var(--primary))" : "hsl(var(--primary) / 0.45)"} />)}</Bar></BarChart></ResponsiveContainer></div>
            </div>

            <div className="rounded-lg border border-border bg-background p-3">
              <h3 className="flex items-center gap-1.5 text-sm font-medium"><Cpu size={15} /> Calidad frente a rendimiento</h3>
              <p className="text-xs text-muted-foreground">Los puntos destacados no están dominados en ambos ejes.</p>
              <div className="mt-3 h-[280px]"><ResponsiveContainer width="100%" height="100%"><ScatterChart margin={{ top: 8, right: 16, bottom: 10, left: 0 }}><CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" /><XAxis type="number" dataKey="quality" name="Calidad" domain={[0, 100]} unit="%" tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }} /><YAxis type="number" dataKey="performance" name="Rendimiento" domain={[0, 100]} unit="%" tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }} /><ZAxis range={[80, 80]} /><Tooltip cursor={{ strokeDasharray: "3 3" }} formatter={(value, name) => [`${Number(value).toFixed(1)}%`, name]} labelFormatter={(_, payload) => payload[0]?.payload?.model ?? ""} contentStyle={{ background: "hsl(var(--popover))", border: "1px solid hsl(var(--border))", borderRadius: 8, fontSize: 12 }} /><Scatter data={eligible}>{eligible.map((row) => <Cell key={row.id} fill={pareto.has(row.id) ? "hsl(var(--primary))" : "hsl(var(--muted-foreground) / 0.55)"} />)}</Scatter></ScatterChart></ResponsiveContainer></div>
            </div>
          </div>

          {practicalTies.size > 1 && <div className="rounded-lg border border-amber-500/20 bg-amber-500/5 p-3 text-xs text-amber-200"><Info size={14} className="mr-1.5 inline" />Los {practicalTies.size} primeros están separados por menos de un punto: trátalos como un empate práctico hasta repetir las baterías.</div>}
          <RankingTable rows={rows} weights={weights} pareto={pareto} />
          <div className="rounded-lg border border-blue-500/20 bg-blue-500/5 p-3 text-xs text-muted-foreground"><Info size={14} className="mr-1.5 inline text-blue-300" />La calidad conserva la escala absoluta del benchmark. Rendimiento y eficiencia se normalizan dentro de cada ejecución para no comparar directamente tiempos tomados en cargas diferentes. Usa los filtros de versión y GPU cuando necesites una cohorte estricta.</div>
        </>}
      </div>
    </section>
  )
}
