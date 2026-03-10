interface SearchFiltersProps {
  task: string
  size: string
  gated: string
  onTaskChange: (value: string) => void
  onSizeChange: (value: string) => void
  onGatedChange: (value: string) => void
}

export function SearchFilters({ task, size, gated, onTaskChange, onSizeChange, onGatedChange }: SearchFiltersProps) {
  return (
    <div className="grid gap-2 rounded-lg border border-border bg-card p-3 md:grid-cols-3">
      <label className="text-xs text-muted-foreground">
        Task
        <select
          value={task}
          onChange={(event) => onTaskChange(event.target.value)}
          className="mt-1 w-full rounded-md border border-border bg-background px-2 py-1.5 text-sm"
        >
          <option value="">Todas</option>
          <option value="text-generation">text-generation</option>
          <option value="image-generation">image-generation</option>
          <option value="automatic-speech-recognition">automatic-speech-recognition</option>
          <option value="text-to-speech">text-to-speech</option>
        </select>
      </label>

      <label className="text-xs text-muted-foreground">
        Tamano
        <select
          value={size}
          onChange={(event) => onSizeChange(event.target.value)}
          className="mt-1 w-full rounded-md border border-border bg-background px-2 py-1.5 text-sm"
        >
          <option value="">Todos</option>
          <option value="small">&lt; 4 GB</option>
          <option value="medium">4 - 12 GB</option>
          <option value="large">&gt; 12 GB</option>
        </select>
      </label>

      <label className="text-xs text-muted-foreground">
        Gated
        <select
          value={gated}
          onChange={(event) => onGatedChange(event.target.value)}
          className="mt-1 w-full rounded-md border border-border bg-background px-2 py-1.5 text-sm"
        >
          <option value="">Todos</option>
          <option value="false">Abiertos</option>
          <option value="true">Gated</option>
        </select>
      </label>
    </div>
  )
}
