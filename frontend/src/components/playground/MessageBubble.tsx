import { Bot, BrainCircuit, ChevronDown, Cog, User } from "lucide-react"
import ReactMarkdown from "react-markdown"
import rehypeHighlight from "rehype-highlight"

export interface ChatToolCall {
  id: string
  name: string
  args: string
  kind: "tool" | "subagent"
  status?: "pending" | "ok" | "timeout" | "schema_error" | "mcp_error"
  alias?: string
  toolName?: string
  durationMs?: number
  error?: string | null
  /** Truncated text result returned by the tool / subagent. */
  resultSummary?: string | null
  /** Tool calls performed inside a subagent's own loop (only set when kind=subagent). */
  childCalls?: ChatToolCall[]
}

export interface ChatMessage {
  id: string
  role: "user" | "assistant" | "tool"
  content: string
  apiContent?: string
  reasoning?: string
  image?: string
  toolCalls?: ChatToolCall[]
}

interface MessageBubbleProps {
  message: ChatMessage
  streaming?: boolean
}

export function MessageBubble({ message, streaming }: MessageBubbleProps) {
  const isUser = message.role === "user"
  const hasContent = Boolean(message.content.trim())
  const hasReasoning = Boolean(message.reasoning?.trim())
  const toolCalls = message.toolCalls ?? []
  const hasToolCalls = toolCalls.length > 0
  const showEmptyPlaceholder = !hasContent && !hasReasoning && !hasToolCalls && !streaming

  return (
    <article
      className={`max-w-[92%] rounded-2xl border shadow-sm ${
        isUser
          ? "ml-auto border-primary/30 bg-primary/[0.08]"
          : "border-border bg-card/95"
      }`}
    >
      <header className="flex items-center justify-between border-b border-border/60 px-4 py-3">
        <div className="flex items-center gap-2">
          <span
            className={`inline-flex h-8 w-8 items-center justify-center rounded-full border ${
              isUser
                ? "border-primary/30 bg-primary/15 text-primary"
                : "border-sky-500/30 bg-sky-500/10 text-sky-300"
            }`}
          >
            {isUser ? <User size={15} /> : <Bot size={15} />}
          </span>
          <div>
            <p className="text-sm font-medium capitalize">{message.role}</p>
            <p className="text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
              {hasReasoning || hasToolCalls ? "agent trace" : "message"}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2 text-[11px] text-muted-foreground">
          {hasReasoning && <Chip label="Thought" tone="amber" />}
          {hasToolCalls && <Chip label={`${toolCalls.length} Calls`} tone="cyan" />}
          {streaming && <Chip label="Live" tone="emerald" />}
        </div>
      </header>

      <div className="space-y-3 px-4 py-3">
        {message.image && (
          <img
            src={message.image}
            alt="Vision input"
            className="max-h-72 rounded-xl border border-border object-contain"
          />
        )}

        {/*
          Render order is chronological: reasoning + tool calls happen *before*
          the final assistant response, so they appear above it in the bubble.
        */}
        {hasReasoning && (
          <details className="group rounded-xl border border-amber-500/25 bg-amber-500/[0.05] p-3" open>
            <summary className="flex cursor-pointer list-none items-center justify-between gap-3">
              <div className="flex items-center gap-2">
                <BrainCircuit size={15} className="text-amber-300" />
                <span className="text-sm font-medium text-amber-100">Pensamiento</span>
              </div>
              <ChevronDown
                size={15}
                className="text-amber-200 transition-transform group-open:rotate-180"
              />
            </summary>
            <div className="mt-3 whitespace-pre-wrap text-sm leading-6 text-amber-50/90">
              {message.reasoning}
            </div>
          </details>
        )}

        {hasToolCalls && (
          <details className="group rounded-xl border border-cyan-500/25 bg-cyan-500/[0.05] p-3" open>
            <summary className="flex cursor-pointer list-none items-center justify-between gap-3">
              <div className="flex items-center gap-2">
                <Cog size={15} className="text-cyan-300" />
                <span className="text-sm font-medium text-cyan-100">Tools y subagentes</span>
              </div>
              <ChevronDown
                size={15}
                className="text-cyan-200 transition-transform group-open:rotate-180"
              />
            </summary>
            <div className="mt-3 space-y-3">
              {toolCalls.map((toolCall) => (
                <ToolTraceCard key={toolCall.id || `${toolCall.kind}-${toolCall.name}-${toolCall.args}`} toolCall={toolCall} />
              ))}
            </div>
          </details>
        )}

        {hasContent && (
          <section className="rounded-xl border border-border/70 bg-background/60 p-3">
            <p className="mb-2 text-[11px] font-medium uppercase tracking-[0.18em] text-muted-foreground">
              Respuesta
            </p>
            <div className="prose prose-invert max-w-none text-sm">
              <ReactMarkdown rehypePlugins={[rehypeHighlight]}>{message.content}</ReactMarkdown>
              {streaming && (
                <span className="ml-1 inline-block h-4 w-2 animate-pulse rounded-sm bg-primary align-middle" />
              )}
            </div>
          </section>
        )}

        {showEmptyPlaceholder && (
          <p className="text-sm italic text-muted-foreground">(sin contenido)</p>
        )}
      </div>
    </article>
  )
}

function ToolTraceCard({ toolCall, depth = 0 }: { toolCall: ChatToolCall; depth?: number }) {
  const label = toolCall.kind === "subagent" ? "Subagente" : "Tool"
  const tone = toolCall.kind === "subagent" ? "border-fuchsia-500/25 bg-fuchsia-500/[0.06]" : "border-cyan-500/20 bg-background/40"
  const displayName =
    toolCall.kind === "subagent" && toolCall.toolName
      ? `agent/${toolCall.toolName}`
      : toolCall.toolName || toolCall.name
  const hasArgs = toolCall.args.trim().length > 0
  const hasResult = Boolean(toolCall.resultSummary && toolCall.resultSummary.trim())
  const childCalls = toolCall.childCalls ?? []
  const hasChildren = childCalls.length > 0

  return (
    <section className={`rounded-xl border p-3 ${tone}`}>
      <div className="flex flex-wrap items-center gap-2">
        <Chip label={label} tone={toolCall.kind === "subagent" ? "fuchsia" : "cyan"} />
        <p className="text-sm font-medium">{displayName}</p>
        <div className="ml-auto flex flex-wrap items-center gap-2 text-[11px] text-muted-foreground">
          <Chip label={formatStatus(toolCall.status)} tone={statusTone(toolCall.status)} />
          {typeof toolCall.durationMs === "number" && toolCall.durationMs > 0 && (
            <span>{toolCall.durationMs} ms</span>
          )}
          {toolCall.alias && toolCall.alias !== "agent" && <span>via {toolCall.alias}</span>}
          {hasChildren && <span>{childCalls.length} pasos internos</span>}
        </div>
      </div>

      {hasArgs && (
        <div className="mt-3">
          <p className="mb-1 text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
            Args
          </p>
          <pre className="overflow-x-auto whitespace-pre-wrap rounded-lg border border-border/70 bg-background/70 p-3 text-xs text-foreground/90">
            {toolCall.args}
          </pre>
        </div>
      )}

      {hasResult && (
        <div className="mt-3">
          <p className="mb-1 text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
            Resultado
          </p>
          <pre className="overflow-x-auto whitespace-pre-wrap rounded-lg border border-border/70 bg-background/70 p-3 text-xs text-foreground/80">
            {toolCall.resultSummary}
          </pre>
        </div>
      )}

      {toolCall.error && (
        <div className="mt-3 rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-xs text-rose-100">
          {toolCall.error}
        </div>
      )}

      {hasChildren && depth < 3 && (
        <details className="group mt-3 rounded-lg border border-fuchsia-500/20 bg-fuchsia-500/[0.04] p-2" open={depth === 0}>
          <summary className="flex cursor-pointer list-none items-center justify-between gap-2">
            <span className="text-[11px] font-medium uppercase tracking-[0.18em] text-fuchsia-200">
              Pasos del subagente ({childCalls.length})
            </span>
            <ChevronDown
              size={13}
              className="text-fuchsia-200 transition-transform group-open:rotate-180"
            />
          </summary>
          <div className="mt-2 space-y-2 border-l border-fuchsia-500/20 pl-3">
            {childCalls.map((child, idx) => (
              <ToolTraceCard
                key={child.id || `${child.kind}-${child.name}-${idx}`}
                toolCall={child}
                depth={depth + 1}
              />
            ))}
          </div>
        </details>
      )}
    </section>
  )
}

function Chip({
  label,
  tone,
}: {
  label: string
  tone: "amber" | "cyan" | "emerald" | "fuchsia" | "slate" | "rose"
}) {
  const tones = {
    amber: "border-amber-500/30 bg-amber-500/10 text-amber-100",
    cyan: "border-cyan-500/30 bg-cyan-500/10 text-cyan-100",
    emerald: "border-emerald-500/30 bg-emerald-500/10 text-emerald-100",
    fuchsia: "border-fuchsia-500/30 bg-fuchsia-500/10 text-fuchsia-100",
    rose: "border-rose-500/30 bg-rose-500/10 text-rose-100",
    slate: "border-border bg-background/60 text-muted-foreground",
  }
  return (
    <span className={`inline-flex rounded-full border px-2 py-0.5 text-[11px] font-medium ${tones[tone]}`}>
      {label}
    </span>
  )
}

function formatStatus(status: ChatToolCall["status"]): string {
  switch (status) {
    case "ok":
      return "ok"
    case "timeout":
      return "timeout"
    case "schema_error":
      return "schema"
    case "mcp_error":
      return "error"
    case "pending":
      return "pending"
    default:
      return "pending"
  }
}

function statusTone(status: ChatToolCall["status"]): "amber" | "emerald" | "rose" {
  if (status === "ok") return "emerald"
  if (status === "timeout" || status === "schema_error" || status === "mcp_error") return "rose"
  return "amber"
}
