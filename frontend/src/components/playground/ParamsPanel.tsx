export interface PlaygroundParams {
  temperature: number
  maxTokens: number
  topP: number
  systemPrompt: string
  responseFormat: "text" | "json" | "verbose_json"
}

interface ParamsPanelProps {
  params: PlaygroundParams
  onChange: (next: PlaygroundParams) => void
}

export function ParamsPanel({ params, onChange }: ParamsPanelProps) {
  return (
    <aside className="space-y-4 rounded-lg border border-border bg-card p-4">
      <h2 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">Params</h2>

      <label className="block text-sm text-muted-foreground">
        temperature: {params.temperature.toFixed(2)}
        <input
          type="range"
          min={0}
          max={2}
          step={0.01}
          value={params.temperature}
          onChange={(event) => onChange({ ...params, temperature: Number(event.target.value) })}
          className="mt-2 w-full"
        />
      </label>

      <label className="block text-sm text-muted-foreground">
        max_tokens
        <input
          type="number"
          min={1}
          max={8192}
          value={params.maxTokens}
          onChange={(event) => onChange({ ...params, maxTokens: Number(event.target.value) })}
          className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2"
        />
      </label>

      <label className="block text-sm text-muted-foreground">
        top_p: {params.topP.toFixed(2)}
        <input
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={params.topP}
          onChange={(event) => onChange({ ...params, topP: Number(event.target.value) })}
          className="mt-2 w-full"
        />
      </label>

      <label className="block text-sm text-muted-foreground">
        system_prompt
        <textarea
          value={params.systemPrompt}
          onChange={(event) => onChange({ ...params, systemPrompt: event.target.value })}
          className="mt-1 min-h-24 w-full rounded-md border border-border bg-background px-3 py-2"
        />
      </label>

      <label className="block text-sm text-muted-foreground">
        response_format
        <select
          value={params.responseFormat}
          onChange={(event) => onChange({ ...params, responseFormat: event.target.value as PlaygroundParams["responseFormat"] })}
          className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2"
        >
          <option value="text">text</option>
          <option value="json">json</option>
          <option value="verbose_json">verbose_json</option>
        </select>
      </label>
    </aside>
  )
}
