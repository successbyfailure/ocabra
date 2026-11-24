// Parsers puros del stream de chat del Playground (REF-18).
//
// Extraído de ChatInterface.tsx para poder probarlo sin montar el componente:
// desenrolla el SSE de /v1/chat/completions (incluidos los eventos
// ``event: ocabra.tool_*`` propios de oCabra) y el NDJSON de /api/chat de
// Ollama, y reúne el texto, el razonamiento y las llamadas a herramientas.
// Sin React ni fetch: las funciones reciben una ``Response`` ya abierta.

import type { ChatMessage, ChatToolCall } from "@/components/playground/MessageBubble"

export interface ResponseContent {
  displayContent: string
  apiContent: string
  reasoning: string
  toolCalls: ChatToolCall[]
}

export interface AssistantPatch {
  contentAppend?: string
  apiContentAppend?: string
  reasoningAppend?: string
  toolCalls?: ChatToolCall[]
  toolResult?: ToolResultPatch
}

export interface ToolResultPatch {
  id: string
  alias?: string
  toolName?: string
  status?: ChatToolCall["status"]
  durationMs?: number
  error?: string | null
  resultSummary?: string | null
  childCalls?: ChatToolCall[]
}

export function finalizeAssistantMessage(
  message: ChatMessage,
  responseContent: ResponseContent,
): ChatMessage {
  const content = message.content || responseContent.displayContent
  const reasoning = message.reasoning || responseContent.reasoning
  const toolCalls = mergeToolCalls(message.toolCalls ?? [], responseContent.toolCalls)
  return {
    ...message,
    content,
    apiContent: message.apiContent || responseContent.apiContent,
    reasoning,
    toolCalls,
  }
}

export function updateAssistantMessage(
  messages: ChatMessage[],
  assistantId: string,
  patch: AssistantPatch,
): ChatMessage[] {
  return messages.map((msg) => {
    if (msg.id !== assistantId) return msg
    const nextToolCalls = patch.toolResult
      ? mergeToolResult(msg.toolCalls ?? [], patch.toolResult)
      : mergeToolCalls(msg.toolCalls ?? [], patch.toolCalls ?? [])
    return {
      ...msg,
      content: `${msg.content}${patch.contentAppend ?? ""}`,
      apiContent: `${msg.apiContent ?? ""}${patch.apiContentAppend ?? ""}`,
      reasoning: `${msg.reasoning ?? ""}${patch.reasoningAppend ?? ""}`,
      toolCalls: nextToolCalls,
    }
  })
}

export async function readOpenAIChatStream(
  response: Response,
  onPatch: (patch: AssistantPatch) => void,
): Promise<ResponseContent> {
  if (!response.body) throw new Error("Streaming no soportado por el navegador")

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ""
  let result: ResponseContent = {
    displayContent: "",
    apiContent: "",
    reasoning: "",
    toolCalls: [],
  }

  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    let boundary = buffer.indexOf("\n\n")
    while (boundary !== -1) {
      const rawEvent = buffer.slice(0, boundary)
      buffer = buffer.slice(boundary + 2)
      boundary = buffer.indexOf("\n\n")

      const event = parseSseEvent(rawEvent)
      if (!event) continue
      if (event.data === "[DONE]") return result

      if (event.name === "ocabra.tool_result") {
        const patch = parseToolResultEvent(event.data)
        if (patch) {
          onPatch({ toolResult: patch })
          result = {
            ...result,
            toolCalls: mergeToolResult(result.toolCalls, patch),
          }
        }
        continue
      }

      const parsed = JSON.parse(event.data) as {
        choices?: Array<{
          delta?: Record<string, unknown>
          message?: Record<string, unknown>
        }>
        error?: { message?: string }
      }

      if (parsed.error?.message) throw new Error(parsed.error.message)

      const choice = parsed.choices?.[0]
      const delta = choice?.delta ?? choice?.message ?? {}
      const content = extractTextContent(delta.content)
      const reasoning = extractReasoningContent(delta)
      const toolCalls = extractToolCalls(delta.tool_calls)

      if (content || reasoning || toolCalls.length > 0) {
        onPatch({
          contentAppend: content,
          apiContentAppend: content,
          reasoningAppend: reasoning,
          toolCalls,
        })
        result = {
          displayContent: `${result.displayContent}${content}`,
          apiContent: `${result.apiContent}${content}`,
          reasoning: `${result.reasoning}${reasoning}`,
          toolCalls: mergeToolCalls(result.toolCalls, toolCalls),
        }
      }
    }
  }

  return result
}

export async function readOllamaChatStream(
  response: Response,
  onPatch: (patch: AssistantPatch) => void,
): Promise<string> {
  if (!response.body) throw new Error("Streaming no soportado por el navegador")

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ""
  let fullText = ""

  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    let newline = buffer.indexOf("\n")
    while (newline !== -1) {
      const line = buffer.slice(0, newline).trim()
      buffer = buffer.slice(newline + 1)
      newline = buffer.indexOf("\n")
      if (!line) continue

      const parsed = JSON.parse(line) as {
        error?: string
        done?: boolean
        message?: { content?: unknown }
      }

      if (parsed.error) throw new Error(parsed.error)
      if (parsed.done) return fullText

      const content = typeof parsed.message?.content === "string" ? parsed.message.content : ""
      if (content) {
        fullText += content
        onPatch({ contentAppend: content, apiContentAppend: content })
      }
    }
  }

  return fullText
}

export function parseSseEvent(rawEvent: string): { name: string; data: string } | null {
  const lines = rawEvent.split("\n")
  let name = "message"
  const dataLines: string[] = []

  for (const line of lines) {
    if (line.startsWith("event:")) {
      name = line.slice(6).trim()
      continue
    }
    if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trimStart())
    }
  }

  if (dataLines.length === 0) return null
  return { name, data: dataLines.join("\n") }
}

export function parseToolResultEvent(data: string): ToolResultPatch | null {
  const parsed = JSON.parse(data) as Record<string, unknown>
  const id = typeof parsed.tool_call_id === "string" ? parsed.tool_call_id : ""
  if (!id) return null
  const childCallsRaw = Array.isArray(parsed.child_tool_calls) ? parsed.child_tool_calls : []
  return {
    id,
    alias: typeof parsed.alias === "string" ? parsed.alias : undefined,
    toolName: typeof parsed.tool_name === "string" ? parsed.tool_name : undefined,
    status: isToolStatus(parsed.status) ? parsed.status : undefined,
    durationMs: typeof parsed.duration_ms === "number" ? parsed.duration_ms : undefined,
    error: typeof parsed.error === "string" ? parsed.error : null,
    resultSummary:
      typeof parsed.result_summary === "string" ? parsed.result_summary : null,
    childCalls: childCallsRaw.map((raw, idx) => parseChildCall(raw, idx)),
  }
}

function parseChildCall(raw: unknown, idx: number): ChatToolCall {
  const r = (raw ?? {}) as Record<string, unknown>
  const alias = typeof r.alias === "string" ? r.alias : ""
  const toolName = typeof r.tool_name === "string" ? r.tool_name : ""
  const kind = inferToolKind({ alias, toolName })
  const status = isToolStatus(r.status) ? r.status : "ok"
  return {
    id: `child-${idx}-${alias}-${toolName}`,
    name: toolName,
    args: "",
    kind,
    alias,
    toolName,
    status,
    durationMs: typeof r.duration_ms === "number" ? r.duration_ms : undefined,
    error: typeof r.error === "string" ? r.error : null,
    resultSummary: typeof r.result_summary === "string" ? r.result_summary : null,
  }
}

export function extractToolCalls(toolCallsRaw: unknown): ChatToolCall[] {
  if (!Array.isArray(toolCallsRaw)) return []
  return toolCallsRaw
    .map((toolCall, index) => {
      const raw = toolCall as {
        id?: unknown
        function?: { name?: unknown; arguments?: unknown }
      }
      const name = typeof raw.function?.name === "string" ? raw.function.name : `tool_${index + 1}`
      const args =
        typeof raw.function?.arguments === "string"
          ? raw.function.arguments
          : raw.function?.arguments == null
            ? ""
            : JSON.stringify(raw.function.arguments, null, 2)
      return {
        id: typeof raw.id === "string" && raw.id ? raw.id : `${name}-${index}`,
        name,
        args,
        kind: inferToolKind({ name }),
        status: "pending" as const,
      }
    })
    .filter((toolCall) => toolCall.name)
}

export function extractOpenAIResponseContent(payload: unknown): ResponseContent {
  const choice = (payload as { choices?: Array<{ message?: Record<string, unknown> }> })?.choices?.[0]
  const message = choice?.message ?? {}
  const content = extractTextContent(message.content)
  const fallbackContent =
    typeof message.content === "string" && message.content.trim() ? message.content : ""
  const reasoning = extractReasoningContent(message)
  const toolCalls = extractToolCalls(message.tool_calls)
  return {
    displayContent: content || fallbackContent,
    apiContent: content || fallbackContent,
    reasoning,
    toolCalls,
  }
}

export function extractOllamaResponseContent(payload: unknown): ResponseContent {
  const message = (payload as { message?: Record<string, unknown> })?.message ?? {}
  if (typeof message.content === "string") {
    return {
      displayContent: message.content,
      apiContent: message.content,
      reasoning: "",
      toolCalls: [],
    }
  }
  if (Array.isArray(message.content)) {
    const content = extractTextContent(message.content)
    return {
      displayContent: content,
      apiContent: content,
      reasoning: "",
      toolCalls: [],
    }
  }
  return {
    displayContent: "",
    apiContent: "",
    reasoning: "",
    toolCalls: [],
  }
}

function extractReasoningContent(value: unknown): string {
  if (!isRecord(value)) return ""
  return (
    extractTextContent(value.reasoning_content) ||
    extractTextContent(value.reasoning) ||
    extractTextContent(value.thinking) ||
    ""
  )
}

function extractTextContent(value: unknown): string {
  if (typeof value === "string") return value
  if (typeof value === "number" || typeof value === "boolean") return String(value)
  if (Array.isArray(value)) {
    return value
      .map((part) => extractTextContent(part))
      .filter(Boolean)
      .join("\n")
  }
  if (!isRecord(value)) return ""

  if (typeof value.text === "string") return value.text
  if (typeof value.content === "string") return value.content
  if (Array.isArray(value.content)) return extractTextContent(value.content)
  if (typeof value.reasoning_content === "string") return value.reasoning_content
  if (typeof value.reasoning === "string") return value.reasoning

  if (value.type === "text" && typeof value.text === "string") return value.text

  return ""
}

export function mergeToolCalls(existing: ChatToolCall[], incoming: ChatToolCall[]): ChatToolCall[] {
  if (incoming.length === 0) return existing
  const merged = [...existing]
  for (const nextCall of incoming) {
    const index = merged.findIndex((item) => item.id === nextCall.id)
    if (index === -1) {
      merged.push(nextCall)
      continue
    }
    merged[index] = mergeToolCall(merged[index], nextCall)
  }
  return merged
}

export function mergeToolResult(existing: ChatToolCall[], patch: ToolResultPatch): ChatToolCall[] {
  const index = existing.findIndex((item) => item.id === patch.id)
  if (index === -1) {
    return [
      ...existing,
      {
        id: patch.id,
        name: patch.toolName || patch.alias || "tool",
        args: "",
        kind: inferToolKind({ alias: patch.alias, toolName: patch.toolName }),
        alias: patch.alias,
        toolName: patch.toolName,
        status: patch.status ?? "pending",
        durationMs: patch.durationMs,
        error: patch.error ?? null,
        resultSummary: patch.resultSummary ?? null,
        childCalls: patch.childCalls ?? [],
      },
    ]
  }

  const current = existing[index]
  const updated: ChatToolCall = {
    ...current,
    kind: inferToolKind({
      alias: patch.alias ?? current.alias,
      toolName: patch.toolName ?? current.toolName,
      name: current.name,
    }),
    alias: patch.alias ?? current.alias,
    toolName: patch.toolName ?? current.toolName,
    status: patch.status ?? current.status,
    durationMs: patch.durationMs ?? current.durationMs,
    error: patch.error ?? current.error ?? null,
    resultSummary:
      patch.resultSummary !== undefined ? patch.resultSummary : current.resultSummary ?? null,
    childCalls:
      patch.childCalls !== undefined && patch.childCalls.length > 0
        ? patch.childCalls
        : current.childCalls,
  }
  return existing.map((item, itemIndex) => (itemIndex === index ? updated : item))
}

function mergeToolCall(current: ChatToolCall, incoming: ChatToolCall): ChatToolCall {
  return {
    ...current,
    ...incoming,
    args: mergeProgressiveText(current.args, incoming.args),
    kind: inferToolKind({
      alias: incoming.alias ?? current.alias,
      toolName: incoming.toolName ?? current.toolName,
      name: incoming.name || current.name,
    }),
    status: incoming.status ?? current.status,
    alias: incoming.alias ?? current.alias,
    toolName: incoming.toolName ?? current.toolName,
    durationMs: incoming.durationMs ?? current.durationMs,
    error: incoming.error ?? current.error ?? null,
  }
}

function mergeProgressiveText(current: string, incoming: string): string {
  if (!incoming) return current
  if (!current) return incoming
  if (incoming.startsWith(current)) return incoming
  if (current.startsWith(incoming)) return current
  return `${current}${incoming}`
}

function inferToolKind(input: {
  alias?: string
  toolName?: string
  name?: string
}): "tool" | "subagent" {
  if (input.alias === "agent") return "subagent"
  if (input.name?.startsWith("delegate_") || input.name?.startsWith("delegate-")) return "subagent"
  if (input.toolName?.startsWith("agent/")) return "subagent"
  return "tool"
}

function isToolStatus(value: unknown): value is NonNullable<ChatToolCall["status"]> {
  return (
    value === "pending" ||
    value === "ok" ||
    value === "timeout" ||
    value === "schema_error" ||
    value === "mcp_error"
  )
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null
}

export function isEmptyResponseContent(value: ResponseContent): boolean {
  return !value.displayContent.trim() && !value.reasoning.trim() && value.toolCalls.length === 0
}
