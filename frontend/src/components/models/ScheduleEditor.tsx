import { useMemo } from "react"
import { Plus, Trash2 } from "lucide-react"
import type { EvictionSchedule } from "@/types"

interface ScheduleEditorProps {
  value: EvictionSchedule[]
  onChange: (next: EvictionSchedule[]) => void
}

const WEEK_DAYS = ["L", "M", "X", "J", "V", "S", "D"]

function summarizeDays(days: number[]) {
  const labels = days
    .sort((a, b) => a - b)
    .map((day) => ["domingo", "lunes", "martes", "miercoles", "jueves", "viernes", "sabado"][day])
  if (labels.length === 0) return "sin dias"
  if (labels.length === 7) return "todos los dias"
  return labels.join(", ")
}

export function ScheduleEditor({ value, onChange }: ScheduleEditorProps) {
  const preview = useMemo(() => {
    if (value.length === 0) return "Sin ventanas programadas"
    const first = value[0]
    return `Los ${summarizeDays(first.days)} de ${first.start} a ${first.end} este modelo se descargara automaticamente`
  }, [value])

  const addSchedule = () => {
    onChange([
      ...value,
      {
        id: `schedule-${Date.now()}`,
        days: [1],
        start: "02:00",
        end: "06:00",
        enabled: true,
      },
    ])
  }

  const patchSchedule = (id: string, patch: Partial<EvictionSchedule>) => {
    onChange(value.map((schedule) => (schedule.id === id ? { ...schedule, ...patch } : schedule)))
  }

  const toggleDay = (scheduleId: string, day: number) => {
    const schedule = value.find((item) => item.id === scheduleId)
    if (!schedule) return
    const hasDay = schedule.days.includes(day)
    const days = hasDay ? schedule.days.filter((item) => item !== day) : [...schedule.days, day]
    patchSchedule(scheduleId, { days })
  }

  return (
    <div className="space-y-3 rounded-lg border border-border bg-background/60 p-3">
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-medium">Ventanas de eviccion</h4>
        <button
          type="button"
          onClick={addSchedule}
          className="inline-flex items-center gap-1 rounded-md border border-border px-2 py-1 text-xs hover:bg-muted"
        >
          <Plus size={14} />
          Anadir
        </button>
      </div>

      {value.length === 0 && <p className="text-xs text-muted-foreground">No hay schedules definidos.</p>}

      {value.map((schedule) => (
        <div key={schedule.id} className="space-y-2 rounded-md border border-border p-2">
          <div className="flex items-center justify-between">
            <label className="inline-flex items-center gap-2 text-xs text-muted-foreground">
              <input
                type="checkbox"
                checked={schedule.enabled}
                onChange={(event) => patchSchedule(schedule.id, { enabled: event.target.checked })}
              />
              Habilitado
            </label>
            <button
              type="button"
              onClick={() => onChange(value.filter((item) => item.id !== schedule.id))}
              className="rounded-md p-1 text-red-300 hover:bg-red-500/10"
              aria-label="Delete schedule"
            >
              <Trash2 size={14} />
            </button>
          </div>

          <div className="flex flex-wrap gap-1">
            {WEEK_DAYS.map((day, idx) => {
              const active = schedule.days.includes(idx)
              return (
                <button
                  key={`${schedule.id}-${day}`}
                  type="button"
                  onClick={() => toggleDay(schedule.id, idx)}
                  className={`h-7 w-7 rounded text-xs font-semibold ${
                    active ? "bg-primary/30 text-primary" : "bg-muted text-muted-foreground"
                  }`}
                >
                  {day}
                </button>
              )
            })}
          </div>

          <div className="grid grid-cols-2 gap-2">
            <label className="text-xs text-muted-foreground">
              Inicio
              <input
                type="time"
                value={schedule.start}
                onChange={(event) => patchSchedule(schedule.id, { start: event.target.value })}
                className="mt-1 w-full rounded-md border border-border bg-background px-2 py-1"
              />
            </label>
            <label className="text-xs text-muted-foreground">
              Fin
              <input
                type="time"
                value={schedule.end}
                onChange={(event) => patchSchedule(schedule.id, { end: event.target.value })}
                className="mt-1 w-full rounded-md border border-border bg-background px-2 py-1"
              />
            </label>
          </div>
        </div>
      ))}

      <p className="text-xs text-muted-foreground">{preview}</p>
    </div>
  )
}
