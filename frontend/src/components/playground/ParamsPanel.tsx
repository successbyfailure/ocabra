import * as Tooltip from "@radix-ui/react-tooltip"
import { HelpCircle } from "lucide-react"

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

function ParamTooltip({ text }: { text: string }) {
  return (
    <Tooltip.Root>
      <Tooltip.Trigger asChild>
        <button type="button" className="ml-1 text-muted-foreground/60 hover:text-muted-foreground">
          <HelpCircle size={12} />
        </button>
      </Tooltip.Trigger>
      <Tooltip.Portal>
        <Tooltip.Content
          className="z-50 max-w-xs rounded-md border border-border bg-popover px-3 py-1.5 text-xs shadow-md"
          sideOffset={4}
        >
          {text}
          <Tooltip.Arrow className="fill-border" />
        </Tooltip.Content>
      </Tooltip.Portal>
    </Tooltip.Root>
  )
}

export function ParamsPanel({ params, onChange }: ParamsPanelProps) {
  return (
    <Tooltip.Provider delayDuration={200}>
      <aside className="space-y-4 rounded-lg border border-border bg-card p-4">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">Params</h2>

        <div className="space-y-1">
          <div className="flex items-center text-sm text-muted-foreground">
            <span>temperature</span>
            <ParamTooltip text="Controla la aleatoriedad. Valores altos = mas creativo, bajos = mas determinista" />
            <span className="ml-auto text-xs font-mono text-primary">{params.temperature.toFixed(2)}</span>
          </div>
          <input
            type="range"
            min={0}
            max={2}
            step={0.01}
            value={params.temperature}
            onChange={(event) => onChange({ ...params, temperature: Number(event.target.value) })}
            className="w-full accent-primary h-2 rounded-lg cursor-pointer"
          />
          <div className="flex justify-between text-[10px] text-muted-foreground">
            <span>0</span>
            <span>2</span>
          </div>
        </div>

        <div className="space-y-1">
          <div className="flex items-center text-sm text-muted-foreground">
            <span>max_tokens</span>
            <ParamTooltip text="Numero maximo de tokens en la respuesta" />
          </div>
          <input
            type="number"
            min={1}
            max={8192}
            value={params.maxTokens}
            onChange={(event) => onChange({ ...params, maxTokens: Number(event.target.value) })}
            className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
          />
        </div>

        <div className="space-y-1">
          <div className="flex items-center text-sm text-muted-foreground">
            <span>top_p</span>
            <ParamTooltip text="Nucleus sampling. Limita a los tokens mas probables que sumen este porcentaje" />
            <span className="ml-auto text-xs font-mono text-primary">{params.topP.toFixed(2)}</span>
          </div>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={params.topP}
            onChange={(event) => onChange({ ...params, topP: Number(event.target.value) })}
            className="w-full accent-primary h-2 rounded-lg cursor-pointer"
          />
          <div className="flex justify-between text-[10px] text-muted-foreground">
            <span>0</span>
            <span>1</span>
          </div>
        </div>

        <div className="space-y-1">
          <div className="flex items-center text-sm text-muted-foreground">
            <span>system_prompt</span>
            <ParamTooltip text="Instrucciones iniciales para el modelo" />
          </div>
          <textarea
            value={params.systemPrompt}
            onChange={(event) => onChange({ ...params, systemPrompt: event.target.value })}
            className="min-h-24 w-full rounded-md border border-border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
          />
        </div>

        <div className="space-y-1">
          <div className="flex items-center text-sm text-muted-foreground">
            <span>response_format</span>
            <ParamTooltip text="Formato de la respuesta del modelo" />
          </div>
          <select
            value={params.responseFormat}
            onChange={(event) => onChange({ ...params, responseFormat: event.target.value as PlaygroundParams["responseFormat"] })}
            className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
          >
            <option value="text">text</option>
            <option value="json">json</option>
            <option value="verbose_json">verbose_json</option>
          </select>
        </div>
      </aside>
    </Tooltip.Provider>
  )
}
