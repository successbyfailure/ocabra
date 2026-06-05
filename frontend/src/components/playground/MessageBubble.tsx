import { AudioLines, Bot, BrainCircuit, ChevronDown, Cog, User } from "lucide-react"
import { useState, type ReactNode } from "react"
import ReactMarkdown from "react-markdown"
import rehypeHighlight from "rehype-highlight"
import type { ChatAudioFormat } from "@/types"

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

export interface ChatAudioAttachment {
  name: string
  format: ChatAudioFormat
  durationSec?: number | null
}

export interface ChatMessage {
  id: string
  role: "user" | "assistant" | "tool"
  content: string
  apiContent?: string
  reasoning?: string
  image?: string
  audio?: ChatAudioAttachment
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

        {message.audio && (
          <div className="inline-flex items-center gap-2 rounded-lg border border-violet-500/30 bg-violet-500/10 px-3 py-1.5 text-xs text-violet-100">
            <AudioLines size={14} className="shrink-0" aria-hidden="true" />
            <span className="truncate font-medium">{message.audio.name}</span>
            <span className="text-violet-100/70">{message.audio.format.toUpperCase()}</span>
            {typeof message.audio.durationSec === "number" && message.audio.durationSec > 0 && (
              <span className="text-violet-100/70">{formatDuration(message.audio.durationSec)}</span>
            )}
          </div>
        )}

        {/*
          Render order is chronological: reasoning + tool calls happen *before*
          the final assistant response, so they appear above it in the bubble.
        */}
        {hasReasoning && (
          <Collapsible
            tone="amber"
            icon={<BrainCircuit size={15} className="text-amber-300" />}
            title="Pensamiento"
            preview={previewText(message.reasoning ?? "")}
            forceOpen={streaming}
          >
            <div className="mt-3 whitespace-pre-wrap text-sm leading-6 text-amber-50/90">
              {message.reasoning}
            </div>
          </Collapsible>
        )}

        {hasToolCalls && (
          <Collapsible
            tone="cyan"
            icon={<Cog size={15} className="text-cyan-300" />}
            title="Tools y subagentes"
            preview={toolCallsPreview(toolCalls)}
            forceOpen={streaming}
          >
            <div className="mt-3 space-y-3">
              {toolCalls.map((toolCall) => (
                <ToolTraceCard
                  key={toolCall.id || `${toolCall.kind}-${toolCall.name}-${toolCall.args}`}
                  toolCall={toolCall}
                  streaming={streaming}
                />
              ))}
            </div>
          </Collapsible>
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

function ToolTraceCard({
  toolCall,
  depth = 0,
  streaming = false,
}: {
  toolCall: ChatToolCall
  depth?: number
  streaming?: boolean
}) {
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
        <CollapsibleBlock
          label="Args"
          preview={previewText(toolCall.args)}
          forceOpen={streaming}
        >
          <pre className="mt-2 overflow-x-auto whitespace-pre-wrap rounded-lg border border-border/70 bg-background/70 p-3 text-xs text-foreground/90">
            {toolCall.args}
          </pre>
        </CollapsibleBlock>
      )}

      {hasResult && (
        <CollapsibleBlock
          label="Resultado"
          preview={previewText(toolCall.resultSummary ?? "")}
          forceOpen={streaming}
        >
          <pre className="mt-2 overflow-x-auto whitespace-pre-wrap rounded-lg border border-border/70 bg-background/70 p-3 text-xs text-foreground/80">
            {toolCall.resultSummary}
          </pre>
        </CollapsibleBlock>
      )}

      {toolCall.error && (
        <div className="mt-3 rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-xs text-rose-100">
          {toolCall.error}
        </div>
      )}

      {hasChildren && depth < 3 && (
        <div className="mt-3">
          <Collapsible
            tone="fuchsia"
            compact
            title={`Pasos del subagente (${childCalls.length})`}
            preview={subagentStepsPreview(childCalls)}
            forceOpen={streaming}
          >
            <div className="mt-2 space-y-2 border-l border-fuchsia-500/20 pl-3">
              {childCalls.map((child, idx) => (
                <ToolTraceCard
                  key={child.id || `${child.kind}-${child.name}-${idx}`}
                  toolCall={child}
                  depth={depth + 1}
                  streaming={streaming}
                />
              ))}
            </div>
          </Collapsible>
        </div>
      )}
    </section>
  )
}

type CollapsibleTone = "amber" | "cyan" | "fuchsia"

const collapsibleTones: Record<
  CollapsibleTone,
  { container: string; title: string; chevron: string; preview: string }
> = {
  amber: {
    container: "border-amber-500/25 bg-amber-500/[0.05]",
    title: "text-amber-100",
    chevron: "text-amber-200",
    preview: "text-amber-100/70",
  },
  cyan: {
    container: "border-cyan-500/25 bg-cyan-500/[0.05]",
    title: "text-cyan-100",
    chevron: "text-cyan-200",
    preview: "text-cyan-100/70",
  },
  fuchsia: {
    container: "border-fuchsia-500/20 bg-fuchsia-500/[0.04]",
    title: "text-fuchsia-200",
    chevron: "text-fuchsia-200",
    preview: "text-fuchsia-100/70",
  },
}

function Collapsible({
  tone,
  icon,
  title,
  preview,
  forceOpen = false,
  compact = false,
  children,
}: {
  tone: CollapsibleTone
  icon?: ReactNode
  title: string
  preview?: string
  forceOpen?: boolean
  compact?: boolean
  children: ReactNode
}) {
  const [pinned, setPinned] = useState(false)
  const [hovered, setHovered] = useState(false)
  const open = forceOpen || pinned || hovered
  const styles = collapsibleTones[tone]
  const padding = compact ? "p-2" : "p-3"
  const titleSize = compact
    ? "text-[11px] font-medium uppercase tracking-[0.18em]"
    : "text-sm font-medium"

  return (
    <div
      className={`rounded-xl border ${padding} ${styles.container}`}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <button
        type="button"
        onClick={() => setPinned((v) => !v)}
        aria-expanded={open}
        className="flex w-full cursor-pointer items-center justify-between gap-3 text-left"
      >
        <div className="flex min-w-0 items-center gap-2">
          {icon}
          <span className={`${titleSize} ${styles.title}`}>{title}</span>
          {!open && preview && (
            <span className={`truncate text-xs ${styles.preview}`}>· {preview}</span>
          )}
        </div>
        <ChevronDown
          size={compact ? 13 : 15}
          className={`shrink-0 transition-transform ${styles.chevron} ${open ? "rotate-180" : ""}`}
        />
      </button>
      {open && <div>{children}</div>}
    </div>
  )
}

function CollapsibleBlock({
  label,
  preview,
  forceOpen = false,
  children,
}: {
  label: string
  preview?: string
  forceOpen?: boolean
  children: ReactNode
}) {
  const [pinned, setPinned] = useState(false)
  const [hovered, setHovered] = useState(false)
  const open = forceOpen || pinned || hovered

  return (
    <div
      className="mt-3"
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <button
        type="button"
        onClick={() => setPinned((v) => !v)}
        aria-expanded={open}
        className="flex w-full cursor-pointer items-center justify-between gap-3 text-left"
      >
        <div className="flex min-w-0 items-center gap-2">
          <span className="text-[11px] font-medium uppercase tracking-[0.18em] text-muted-foreground">
            {label}
          </span>
          {!open && preview && (
            <span className="truncate text-xs text-muted-foreground/80">· {preview}</span>
          )}
        </div>
        <ChevronDown
          size={13}
          className={`shrink-0 text-muted-foreground transition-transform ${open ? "rotate-180" : ""}`}
        />
      </button>
      {open && children}
    </div>
  )
}

function previewText(value: string, maxLen = 120): string {
  const flat = value.replace(/\s+/g, " ").trim()
  if (flat.length <= maxLen) return flat
  return flat.slice(0, maxLen).trimEnd() + "…"
}

function toolCallsPreview(calls: ChatToolCall[]): string {
  if (calls.length === 0) return ""
  const names = calls
    .map((c) => (c.kind === "subagent" && c.toolName ? `agent/${c.toolName}` : c.toolName || c.name))
    .filter(Boolean)
  const head = names.slice(0, 3).join(", ")
  const extra = names.length > 3 ? ` +${names.length - 3}` : ""
  return previewText(`${head}${extra}`, 100)
}

function subagentStepsPreview(children: ChatToolCall[]): string {
  return toolCallsPreview(children)
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

function formatDuration(seconds: number): string {
  const total = Math.max(0, Math.round(seconds))
  const m = Math.floor(total / 60)
  const s = total % 60
  return `${m}:${s.toString().padStart(2, "0")}`
}
