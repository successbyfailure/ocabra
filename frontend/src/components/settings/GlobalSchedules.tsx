import { useEffect, useState } from "react"
import { toast } from "sonner"
import { ScheduleEditor } from "@/components/models/ScheduleEditor"
import type { EvictionSchedule, ServerConfig } from "@/types"

interface GlobalSchedulesProps {
  config: ServerConfig
  onSave: (patch: Partial<ServerConfig>) => Promise<void>
}

export function GlobalSchedules({ config, onSave }: GlobalSchedulesProps) {
  const [schedules, setSchedules] = useState<EvictionSchedule[]>(config.globalSchedules)

  useEffect(() => {
    setSchedules(config.globalSchedules)
  }, [config.globalSchedules])

  const save = async () => {
    try {
      await onSave({ globalSchedules: schedules })
      toast.success("Schedules globales guardados")
    } catch {
      // page-level toast is shown in Settings
    }
  }

  return (
    <section className="space-y-3 rounded-lg border border-border bg-card p-4">
      <h2 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">Schedules globales</h2>
      <p className="text-xs text-muted-foreground">
        Precedencia: schedule de modelo individual mayor que schedule global.
      </p>
      <ScheduleEditor value={schedules} onChange={setSchedules} />
      <button
        type="button"
        onClick={() => void save()}
        className="rounded-md bg-primary px-3 py-2 text-sm font-medium text-primary-foreground"
      >
        Guardar schedules
      </button>
    </section>
  )
}
