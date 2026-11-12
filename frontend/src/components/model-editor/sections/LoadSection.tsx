import { useMemo, useRef, useState } from "react"
import { Anchor, Flame, Hourglass, Plus, Trash2 } from "lucide-react"
import { cn } from "@/lib/cn"
import type { EvictionSchedule, LoadPolicy } from "@/types"
import { getPath } from "../draft"
import { FieldRow, SchemaFields } from "../SchemaField"
import { Callout, Hint, Panel, Segmented, Toggle, inputCls } from "../ui"
import type { SectionProps } from "./types"
import { LoadCostHint } from "./SummaryInsights"

const POLICIES: {
  value: LoadPolicy
  title: string
  icon: typeof Hourglass
  text: string
  timeline: { label: string; tone: "load" | "on" | "off" }[]
}[] = [
  {
    value: "on_demand",
    title: "Bajo demanda",
    icon: Hourglass,
    text: "Se carga con la primera petición y se descarga tras un rato sin uso. Deja la GPU libre para otros.",
    timeline: [
      { label: "petición", tone: "load" },
      { label: "en uso", tone: "on" },
      { label: "inactivo → fuera", tone: "off" },
    ],
  },
  {
    value: "warm",
    title: "Caliente",
    icon: Flame,
    text: "Se carga con la primera petición y se queda. Solo sale por falta de memoria o por un horario.",
    timeline: [
      { label: "petición", tone: "load" },
      { label: "se queda cargado", tone: "on" },
    ],
  },
  {
    value: "pin",
    title: "Fijado",
    icon: Anchor,
    text: "Se carga al arrancar el servidor y se recarga solo si se cae o lo desalojan.",
    timeline: [
      { label: "arranque", tone: "load" },
      { label: "siempre cargado", tone: "on" },
    ],
  },
]

const TONE = { load: "bg-violet-500/60", on: "bg-emerald-500/60", off: "bg-slate-500/40" }

const IDLE_PRESETS = [
  { value: 60, label: "1 min" },
  { value: 300, label: "5 min" },
  { value: 900, label: "15 min" },
  { value: 3600, label: "1 h" },
  { value: 0, label: "Nunca" },
]

function fmtSeconds(s: number): string {
  if (s <= 0) return "nunca"
  if (s < 60) return `${s} s`
  if (s < 3600) return `${Math.round(s / 60)} min`
  return `${(s / 3600).toFixed(s % 3600 ? 1 : 0)} h`
}

export function LoadSection(props: SectionProps) {
  const { editor, readOnly } = props
  const d = editor.draft
  if (!d) return null
  const keepAlive = editor.fieldByKey.get("keep_alive_seconds")
  const keepAliveRaw = getPath(d.extraConfig, ["keep_alive_seconds"])
  const globalIdle = Number(keepAlive?.default ?? 300)
  const effectiveIdle = typeof keepAliveRaw === "number" ? keepAliveRaw : globalIdle

  return (
    <div className="space-y-4">
      <Panel title="¿Cuándo está en memoria?" subtitle="Política de carga del despliegue.">
        <div className="grid gap-3 md:grid-cols-3" data-testid="load-policy">
          {POLICIES.map((p) => {
            const active = d.loadPolicy === p.value
            const Icon = p.icon
            return (
              <button
                key={p.value}
                type="button"
                disabled={readOnly}
                aria-pressed={active}
                onClick={() => editor.update({ loadPolicy: p.value, pinPermanent: p.value === "pin" ? d.pinPermanent : false })}
                className={cn(
                  "rounded-xl border p-3 text-left transition-colors disabled:cursor-not-allowed",
                  active ? "border-primary bg-primary/10 ring-1 ring-primary/40" : "border-border hover:border-primary/40 hover:bg-muted/30",
                )}
              >
                <div className="flex items-center gap-2 text-sm font-semibold">
                  <Icon size={16} className={active ? "text-primary" : "text-muted-foreground"} /> {p.title}
                </div>
                <p className="mt-1 text-[11.5px] leading-snug text-muted-foreground">{p.text}</p>
                <div className="mt-2 flex h-5 overflow-hidden rounded-md text-[9.5px] font-medium text-white/90">
                  {p.timeline.map((t, i) => (
                    <div key={i} className={cn("flex flex-1 items-center justify-center truncate px-1", TONE[t.tone])}>
                      {t.label}
                    </div>
                  ))}
                </div>
              </button>
            )
          })}
        </div>
        <LoadCostHint insights={props.insights} policy={d.loadPolicy} />
        {d.loadPolicy === "pin" && (
          <div className="mt-3 rounded-lg border border-border bg-muted/20 p-3">
            <Toggle
              checked={d.pinPermanent}
              disabled={readOnly}
              onChange={(v) => editor.update({ pinPermanent: v })}
              label={<span className="font-medium">Pin duro</span>}
              testId="pin-permanent"
            />
            <Hint>
              Sin pin duro, si falta memoria el modelo puede salir como último recurso y vuelve a cargarse cuando hay sitio. Con pin
              duro solo lo descarga un administrador.
            </Hint>
          </div>
        )}
      </Panel>

      {keepAlive && (
        <Panel
          title="Descargar tras inactividad"
          subtitle={d.loadPolicy === "on_demand" ? `Ahora: ${fmtSeconds(effectiveIdle)} sin peticiones.` : "Solo se aplica a los modelos bajo demanda."}
        >
          <div className={cn(d.loadPolicy !== "on_demand" && "opacity-50")}>
            <FieldRow field={keepAlive} editor={editor} readOnly={readOnly}>
              <div className="flex flex-wrap items-center gap-3">
                <Segmented
                  size="sm"
                  disabled={readOnly}
                  value={typeof keepAliveRaw === "number" ? keepAliveRaw : -1}
                  options={[
                    { value: -1, label: `Global (${fmtSeconds(globalIdle)})` },
                    ...IDLE_PRESETS.map((o) => ({ value: o.value, label: o.label })),
                  ]}
                  onChange={(v) => editor.setField(["keep_alive_seconds"], v === -1 ? undefined : v)}
                />
                <input
                  type="number"
                  min={0}
                  disabled={readOnly}
                  className={cn(inputCls, "w-28")}
                  value={typeof keepAliveRaw === "number" ? keepAliveRaw : ""}
                  placeholder={String(globalIdle)}
                  onChange={(e) => editor.setField(["keep_alive_seconds"], e.target.value === "" ? undefined : Math.max(0, Number(e.target.value)))}
                />
                <span className="text-xs text-muted-foreground">segundos</span>
              </div>
              <IdleTimeline seconds={effectiveIdle} />
            </FieldRow>
          </div>
        </Panel>
      )}

      <SchedulesPanel {...props} />

      <SchemaFields editor={editor} group="load" exclude={["keep_alive_seconds"]} readOnly={readOnly} showAdvanced />
    </div>
  )
}

function IdleTimeline({ seconds }: { seconds: number }) {
  const never = seconds <= 0
  return (
    <div className="mt-2 flex items-center gap-2 text-[10.5px] text-muted-foreground">
      <span className="rounded bg-violet-500/30 px-1.5 py-0.5 text-violet-200">última petición</span>
      <div className="relative h-1.5 flex-1 overflow-hidden rounded-full bg-muted">
        <div className={cn("absolute inset-y-0 left-0 rounded-full", never ? "w-full bg-emerald-500/50" : "w-3/4 bg-gradient-to-r from-emerald-500/60 to-slate-500/40")} />
      </div>
      <span className={cn("rounded px-1.5 py-0.5", never ? "bg-emerald-500/20 text-emerald-200" : "bg-slate-500/30")}>
        {never ? "sigue cargado" : `descarga a los ${fmtSeconds(seconds)}`}
      </span>
    </div>
  )
}

// Días en el orden de la semana española; el valor es el de cron (0 = domingo).
const DAYS = [
  { value: 1, short: "L", long: "lunes" },
  { value: 2, short: "M", long: "martes" },
  { value: 3, short: "X", long: "miércoles" },
  { value: 4, short: "J", long: "jueves" },
  { value: 5, short: "V", long: "viernes" },
  { value: 6, short: "S", long: "sábado" },
  { value: 0, short: "D", long: "domingo" },
]

const pad = (n: number) => String(n).padStart(2, "0")
const toMin = (t: string) => {
  const [h, m] = t.split(":").map(Number)
  return (h || 0) * 60 + (m || 0)
}

/** Cada franja pintada en la rejilla: (fila de día, minuto inicio, minuto fin). */
function windowCells(s: EvictionSchedule): { day: number; from: number; to: number }[] {
  const start = toMin(s.start)
  const end = toMin(s.end)
  const out: { day: number; from: number; to: number }[] = []
  for (const day of s.days) {
    if (end > start) out.push({ day, from: start, to: end })
    else {
      out.push({ day, from: start, to: 1440 })
      out.push({ day: (day + 1) % 7, from: 0, to: end })
    }
  }
  return out
}

export function describeSchedule(s: EvictionSchedule): string {
  const names = DAYS.filter((d) => s.days.includes(d.value)).map((d) => d.long)
  const days = names.length === 7 ? "todos los días" : names.length ? names.join(", ") : "ningún día"
  const wraps = toMin(s.end) <= toMin(s.start)
  return `Fuera de memoria ${days} de ${s.start} a ${s.end}${wraps ? " (del día siguiente)" : ""}`
}

function SchedulesPanel({ editor, readOnly }: SectionProps) {
  const d = editor.draft!
  const schedules = d.schedules
  const [drag, setDrag] = useState<{ day: number; from: number; to: number } | null>(null)
  const dragging = useRef(false)
  const setSchedules = (next: EvictionSchedule[]) => editor.update({ schedules: next })
  const cells = useMemo(() => schedules.flatMap((s, i) => windowCells(s).map((c) => ({ ...c, i, enabled: s.enabled }))), [schedules])

  const finishDrag = () => {
    if (!dragging.current || !drag) return
    dragging.current = false
    const from = Math.min(drag.from, drag.to)
    const to = Math.max(drag.from, drag.to) + 1
    setSchedules([
      ...schedules,
      {
        id: `franja-${Date.now().toString(36)}`,
        days: [drag.day],
        start: `${pad(from)}:00`,
        end: to >= 24 ? "00:00" : `${pad(to)}:00`,
        enabled: true,
      },
    ])
    setDrag(null)
  }

  const policyNote = {
    on_demand: "Bajo demanda: se descarga al empezar la franja; si llega una petición dentro, vuelve a cargarse.",
    warm: "Caliente: se descarga al empezar la franja y se vuelve a cargar al acabar.",
    pin: "Fijado: se descarga al empezar, no se recarga dentro de la franja y vuelve a cargarse al acabar.",
  }[d.loadPolicy]

  return (
    <Panel
      title="Horarios"
      subtitle="Franjas en las que el modelo deja la GPU libre. Arrastra sobre la rejilla para crear una."
      actions={
        !readOnly && (
          <button
            type="button"
            className="inline-flex items-center gap-1 rounded-md border border-border px-2 py-1 text-xs hover:bg-muted"
            onClick={() =>
              setSchedules([
                ...schedules,
                { id: `franja-${Date.now().toString(36)}`, days: [1, 2, 3, 4, 5], start: "09:00", end: "14:00", enabled: true },
              ])
            }
          >
            <Plus size={12} /> Añadir franja
          </button>
        )
      }
    >
      <div className="select-none overflow-x-auto" onMouseLeave={finishDrag} onMouseUp={finishDrag} data-testid="schedule-grid">
        <div className="grid min-w-[560px]" style={{ gridTemplateColumns: "28px repeat(24, 1fr)" }}>
          <div />
          {Array.from({ length: 24 }, (_, h) => (
            <div key={h} className="text-center text-[9px] text-muted-foreground">
              {h % 3 === 0 ? h : ""}
            </div>
          ))}
          {DAYS.map((day) => (
            <div key={day.value} className="contents">
              <div className="flex items-center text-[11px] font-medium text-muted-foreground">{day.short}</div>
              {Array.from({ length: 24 }, (_, h) => {
                const covered = cells.filter((c) => c.day === day.value && c.from < (h + 1) * 60 && c.to > h * 60)
                const inDrag = drag && drag.day === day.value && h >= Math.min(drag.from, drag.to) && h <= Math.max(drag.from, drag.to)
                const on = covered.some((c) => c.enabled)
                const off = covered.length > 0 && !on
                return (
                  <div
                    key={h}
                    data-cell={`${day.value}-${h}`}
                    className={cn(
                      "m-px h-5 rounded-[3px] border border-border/40",
                      on && "border-orange-500/40 bg-orange-500/50",
                      off && "bg-[repeating-linear-gradient(45deg,transparent,transparent_3px,rgba(148,163,184,0.35)_3px,rgba(148,163,184,0.35)_5px)]",
                      inDrag && "bg-sky-500/50",
                      !readOnly && "cursor-crosshair hover:bg-muted",
                    )}
                    onMouseDown={() => {
                      if (readOnly) return
                      dragging.current = true
                      setDrag({ day: day.value, from: h, to: h })
                    }}
                    onMouseEnter={() => {
                      if (dragging.current && drag && drag.day === day.value) setDrag({ ...drag, to: h })
                    }}
                  />
                )
              })}
            </div>
          ))}
        </div>
      </div>

      {schedules.length > 0 && <Hint tone="info">{policyNote}</Hint>}

      <div className="mt-3 space-y-2">
        {schedules.map((s, i) => (
          <div key={s.id} className={cn("rounded-lg border border-border p-2", !s.enabled && "opacity-60")} data-testid="schedule-row">
            <div className="flex flex-wrap items-center gap-2">
              <Toggle
                checked={s.enabled}
                disabled={readOnly}
                onChange={(v) => setSchedules(schedules.map((x, j) => (j === i ? { ...x, enabled: v } : x)))}
              />
              <div className="flex gap-0.5">
                {DAYS.map((day) => {
                  const on = s.days.includes(day.value)
                  return (
                    <button
                      key={day.value}
                      type="button"
                      title={day.long}
                      disabled={readOnly}
                      data-day={day.value}
                      aria-pressed={on}
                      onClick={() => {
                        const days = on ? s.days.filter((x) => x !== day.value) : [...s.days, day.value]
                        if (days.length) setSchedules(schedules.map((x, j) => (j === i ? { ...x, days: days.sort() } : x)))
                      }}
                      className={cn(
                        "h-6 w-6 rounded text-[11px] font-medium",
                        on ? "bg-orange-500/70 text-white" : "bg-muted text-muted-foreground hover:bg-muted/70",
                      )}
                    >
                      {day.short}
                    </button>
                  )
                })}
              </div>
              <input
                type="time"
                disabled={readOnly}
                className={cn(inputCls, "w-28")}
                value={s.start}
                onChange={(e) => setSchedules(schedules.map((x, j) => (j === i ? { ...x, start: e.target.value } : x)))}
              />
              <span className="text-xs text-muted-foreground">a</span>
              <input
                type="time"
                disabled={readOnly}
                className={cn(inputCls, "w-28")}
                value={s.end}
                onChange={(e) => setSchedules(schedules.map((x, j) => (j === i ? { ...x, end: e.target.value } : x)))}
              />
              {!readOnly && (
                <button
                  type="button"
                  className="ml-auto rounded p-1 text-muted-foreground hover:bg-red-500/10 hover:text-red-300"
                  title="Quitar franja"
                  onClick={() => setSchedules(schedules.filter((_, j) => j !== i))}
                >
                  <Trash2 size={14} />
                </button>
              )}
            </div>
            <p className="mt-1 text-[11px] text-muted-foreground">{describeSchedule(s)}</p>
          </div>
        ))}
        {schedules.length === 0 && <Callout tone="info">Sin horarios: el modelo sigue su política de carga todo el día.</Callout>}
      </div>
    </Panel>
  )
}
