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
  disableSystemPrompt?: boolean
  /** Max context length of the underlying model. Used to cap max_tokens. */
  modelContextLength?: number | null
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

export function ParamsPanel({
  params,
  onChange,
  disableSystemPrompt = false,
  modelContextLength = null,
}: ParamsPanelProps) {
  // Cap max_tokens to a sensible upper bound. Modern context windows reach
  // 128K-256K, but a single response of more than ~8K tokens is rarely useful
  // and risks exceeding the model's window once the prompt is added.
  // Hard ceiling: min(model context, 8192). Prompts will eat into the rest.
  const ABSOLUTE_CEILING = 8192
  const ctxCap = modelContextLength && modelContextLength > 0
    ? Math.min(modelContextLength, ABSOLUTE_CEILING)
    : ABSOLUTE_CEILING
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
            <ParamTooltip
              text={
                modelContextLength
                  ? `Tokens maximos en la respuesta. Limite: ${ctxCap.toLocaleString()} (modelo: ${modelContextLength.toLocaleString()} de contexto)`
                  : "Tokens maximos en la respuesta"
              }
            />
            <span className="ml-auto text-[10px] font-mono text-muted-foreground">
              max {ctxCap.toLocaleString()}
            </span>
          </div>
          <input
            type="number"
            min={1}
            max={ctxCap}
            value={params.maxTokens}
            onChange={(event) => {
              const raw = Number(event.target.value)
              const clamped = Number.isFinite(raw)
                ? Math.max(1, Math.min(raw, ctxCap))
                : params.maxTokens
              onChange({ ...params, maxTokens: clamped })
            }}
            className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
          />
          {params.maxTokens > ctxCap && (
            <p className="text-[10px] text-amber-400">
              Tope ajustado a {ctxCap.toLocaleString()}.
            </p>
          )}
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
            <ParamTooltip
              text={
                disableSystemPrompt
                  ? "El agente seleccionado fuerza su propio system prompt. Este campo está deshabilitado."
                  : "Instrucciones iniciales para el modelo"
              }
            />
          </div>
          <textarea
            value={disableSystemPrompt ? "(impuesto por el agente)" : params.systemPrompt}
            onChange={(event) => onChange({ ...params, systemPrompt: event.target.value })}
            disabled={disableSystemPrompt}
            className="min-h-24 w-full rounded-md border border-border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 disabled:opacity-60"
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
