export interface DateRangeValue {
  from: string
  to: string
  preset: "1h" | "24h" | "7d" | "30d" | "custom"
}

interface DateRangePickerProps {
  value: DateRangeValue
  onChange: (next: DateRangeValue) => void
}

function toIsoLocal(date: Date): string {
  const pad = (num: number) => String(num).padStart(2, "0")
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(date.getHours())}:${pad(date.getMinutes())}`
}

function presetToRange(preset: DateRangeValue["preset"]): DateRangeValue {
  const now = new Date()
  const from = new Date(now)
  if (preset === "1h") from.setHours(now.getHours() - 1)
  if (preset === "24h") from.setDate(now.getDate() - 1)
  if (preset === "7d") from.setDate(now.getDate() - 7)
  if (preset === "30d") from.setDate(now.getDate() - 30)
  return {
    preset,
    from: toIsoLocal(from),
    to: toIsoLocal(now),
  }
}

export function defaultDateRange(): DateRangeValue {
  return presetToRange("24h")
}

export function DateRangePicker({ value, onChange }: DateRangePickerProps) {
  return (
    <div className="grid gap-2 rounded-lg border border-border bg-card p-3 md:grid-cols-3">
      <label className="text-xs text-muted-foreground">
        Preset
        <select
          value={value.preset}
          onChange={(event) => {
            const preset = event.target.value as DateRangeValue["preset"]
            if (preset === "custom") {
              onChange({ ...value, preset })
              return
            }
            onChange(presetToRange(preset))
          }}
          className="mt-1 w-full rounded-md border border-border bg-background px-2 py-1.5 text-sm"
        >
          <option value="1h">Ultima 1h</option>
          <option value="24h">Ultimas 24h</option>
          <option value="7d">Ultimos 7d</option>
          <option value="30d">Ultimos 30d</option>
          <option value="custom">Custom</option>
        </select>
      </label>

      <label className="text-xs text-muted-foreground">
        From
        <input
          type="datetime-local"
          value={value.from}
          onChange={(event) => onChange({ ...value, preset: "custom", from: event.target.value })}
          className="mt-1 w-full rounded-md border border-border bg-background px-2 py-1.5 text-sm"
        />
      </label>

      <label className="text-xs text-muted-foreground">
        To
        <input
          type="datetime-local"
          value={value.to}
          onChange={(event) => onChange({ ...value, preset: "custom", to: event.target.value })}
          className="mt-1 w-full rounded-md border border-border bg-background px-2 py-1.5 text-sm"
        />
      </label>
    </div>
  )
}
