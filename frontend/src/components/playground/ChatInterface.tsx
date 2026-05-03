import { useMemo, useRef, useState } from "react"
import { Copy, ImagePlus, Send } from "lucide-react"
import { toast } from "sonner"
import {
  MessageBubble,
  type ChatMessage,
  type ChatToolCall,
} from "@/components/playground/MessageBubble"
import type { PlaygroundParams } from "@/components/playground/ParamsPanel"
import type { BackendType } from "@/types"

interface ChatInterfaceProps {
  modelId: string
  backendType: BackendType | null
  params: PlaygroundParams
}

interface ResponseContent {
  displayContent: string
  apiContent: string
  reasoning: string
  toolCalls: ChatToolCall[]
}

interface AssistantPatch {
  contentAppend?: string
  apiContentAppend?: string
  reasoningAppend?: string
  toolCalls?: ChatToolCall[]
  toolResult?: ToolResultPatch
}

interface ToolResultPatch {
  id: string
  alias?: string
  toolName?: string
  status?: ChatToolCall["status"]
  durationMs?: number
  error?: string | null
  resultSummary?: string | null
  childCalls?: ChatToolCall[]
}

export function ChatInterface({ modelId, backendType, params }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState("")
  const [sending, setSending] = useState(false)
  const [streamingMessageId, setStreamingMessageId] = useState<string | null>(null)
  const [dragImage, setDragImage] = useState<string | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  const buildOpenAIMessages = (userText: string) => {
    const history = messages.map((msg) => {
      if (msg.role === "assistant") {
        return { role: "assistant" as const, content: msg.apiContent ?? msg.content }
      }
      return { role: "user" as const, content: msg.content }
    })
    const userContent = dragImage
      ? [
          { type: "text", text: userText || "Describe la imagen" },
          { type: "image_url", image_url: { url: dragImage } },
        ]
      : userText
    return [
      { role: "system" as const, content: params.systemPrompt },
      ...history,
      { role: "user" as const, content: userContent },
    ]
  }

  const curlPreview = useMemo(() => {
    const origin = typeof window !== "undefined" ? window.location.origin : "http://localhost:8000"
    const body = JSON.stringify({
      model: modelId,
      messages: buildOpenAIMessages(input || "Hello"),
      temperature: params.temperature,
      max_tokens: params.maxTokens,
      top_p: params.topP,
      stream: true,
    })
    return `curl -X POST ${origin}/v1/chat/completions \\\n  -H 'Content-Type: application/json' \\\n  -H 'Authorization: Bearer YOUR_API_KEY' \\\n  -d '${body}'`
  }, [input, modelId, params.maxTokens, params.systemPrompt, params.temperature, params.topP, messages, dragImage])

  const sendMessage = async () => {
    if (!modelId) {
      toast.error("Selecciona un modelo")
      return
    }
    const text = input.trim()
    if (!text && !dragImage) return

    const userMessage: ChatMessage = {
      id: `msg-${Date.now()}`,
      role: "user",
      content: text || "[imagen]",
      image: dragImage ?? undefined,
    }

    setMessages((prev) => [...prev, userMessage])
    setInput("")
    setSending(true)
    const assistantId = `msg-${Date.now()}-assistant`
    try {
      const useOllama = backendType === "ollama"
      const useStream = true
      setStreamingMessageId(assistantId)
      setMessages((prev) => [
        ...prev,
        {
          id: assistantId,
          role: "assistant",
          content: "",
          apiContent: "",
          reasoning: "",
          toolCalls: [],
        },
      ])

      const requestBody = JSON.stringify(
        useOllama
          ? {
              model: modelId,
              stream: useStream,
              messages: buildOpenAIMessages(text),
              options: {
                temperature: params.temperature,
                top_p: params.topP,
                num_predict: params.maxTokens,
              },
            }
          : {
              model: modelId,
              messages: buildOpenAIMessages(text),
              temperature: params.temperature,
              max_tokens: params.maxTokens,
              top_p: params.topP,
              stream: useStream,
            },
      )

      const response = await fetch(useOllama ? "/api/chat" : "/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: requestBody,
      })
      if (!response.ok) {
        const err = await response.json().catch(() => ({}))
        const detail = err?.detail
        const message =
          err?.error?.message ??
          (typeof detail === "string" ? detail : detail?.error?.message) ??
          err?.message ??
          `HTTP ${response.status}`
        throw new Error(typeof message === "string" ? message : JSON.stringify(message))
      }

      let responseContent = useOllama
        ? {
            displayContent: await readOllamaChatStream(response, (patch) => {
              setMessages((prev) => updateAssistantMessage(prev, assistantId, patch))
            }),
            apiContent: "",
            reasoning: "",
            toolCalls: [],
          }
        : await readOpenAIChatStream(response, (patch) => {
            setMessages((prev) => updateAssistantMessage(prev, assistantId, patch))
          })

      if (!useOllama && isEmptyResponseContent(responseContent)) {
        responseContent = await retryOpenAINonStreaming(requestBody)
      }

      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantId
            ? finalizeAssistantMessage(msg, responseContent)
            : msg,
        ),
      )
      setDragImage(null)
    } catch (err) {
      setMessages((prev) => prev.filter((msg) => msg.id !== assistantId))
      toast.error(err instanceof Error ? err.message : "Error en chat")
    } finally {
      setSending(false)
      setStreamingMessageId(null)
    }
  }

  const onDropImage = (file: File) => {
    const reader = new FileReader()
    reader.onload = () => {
      setDragImage(String(reader.result ?? ""))
      toast.success("Imagen adjuntada")
    }
    reader.readAsDataURL(file)
  }

  return (
    <div className="flex h-full flex-col rounded-lg border border-border bg-card">
      <div
        className="flex-1 space-y-4 overflow-y-auto p-3"
        onDragOver={(event) => event.preventDefault()}
        onDrop={(event) => {
          event.preventDefault()
          const file = event.dataTransfer.files[0]
          if (file?.type.startsWith("image/")) {
            onDropImage(file)
          }
        }}
      >
        {messages.length === 0 && (
          <p className="text-sm text-muted-foreground">Escribe un prompt o arrastra una imagen para vision.</p>
        )}
        {messages.map((message) => (
          <MessageBubble
            key={message.id}
            message={message}
            streaming={message.id === streamingMessageId}
          />
        ))}
      </div>

      {dragImage && (
        <div className="border-t border-border px-3 py-2">
          <div className="inline-flex items-center gap-2 rounded-md border border-primary/40 bg-primary/10 px-2 py-1 text-xs">
            <ImagePlus size={14} />
            Imagen adjuntada
          </div>
        </div>
      )}

      <div className="border-t border-border p-3">
        <textarea
          value={input}
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={(event) => {
            if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
              event.preventDefault()
              void sendMessage()
            }
          }}
          placeholder="Escribe tu mensaje..."
          className="mb-2 min-h-20 w-full rounded-md border border-border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
        />
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => fileRef.current?.click()}
              className="rounded-md border border-border px-3 py-2 text-xs hover:bg-muted"
            >
              Vision input
            </button>
            <button
              type="button"
              onClick={async () => {
                await navigator.clipboard.writeText(curlPreview)
                toast.success("curl copiado")
              }}
              className="inline-flex items-center gap-1 rounded-md border border-border px-3 py-2 text-xs hover:bg-muted"
            >
              <Copy size={14} />
              Copy as OpenAI API call
            </button>
          </div>
          <button
            type="button"
            onClick={() => void sendMessage()}
            disabled={sending || !modelId}
            className="inline-flex items-center gap-2 rounded-md bg-primary px-5 py-2.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
          >
            <Send size={14} />
            {sending ? "Enviando..." : "Enviar"}
            <span className="ml-0.5 text-xs text-primary-foreground/50">⌘↵</span>
          </button>
        </div>
      </div>

      <input
        ref={fileRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(event) => {
          const file = event.target.files?.[0]
          if (file) onDropImage(file)
        }}
      />
    </div>
  )
}

async function retryOpenAINonStreaming(requestBody: string): Promise<ResponseContent> {
  const parsedBody = JSON.parse(requestBody) as Record<string, unknown>
  const retryBody = JSON.stringify({ ...parsedBody, stream: false })
  const response = await fetch("/v1/chat/completions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: retryBody,
  })
  if (!response.ok) {
    return {
      displayContent: "",
      apiContent: "",
      reasoning: "",
      toolCalls: [],
    }
  }
  return extractOpenAIResponseContent(await response.json())
}

function finalizeAssistantMessage(
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

function updateAssistantMessage(
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

async function readOpenAIChatStream(
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

async function readOllamaChatStream(
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

function parseSseEvent(rawEvent: string): { name: string; data: string } | null {
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

function parseToolResultEvent(data: string): ToolResultPatch | null {
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

function extractToolCalls(toolCallsRaw: unknown): ChatToolCall[] {
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

function extractOpenAIResponseContent(payload: unknown): ResponseContent {
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

function extractOllamaResponseContent(payload: unknown): ResponseContent {
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

function mergeToolCalls(existing: ChatToolCall[], incoming: ChatToolCall[]): ChatToolCall[] {
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

function mergeToolResult(existing: ChatToolCall[], patch: ToolResultPatch): ChatToolCall[] {
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

function isEmptyResponseContent(value: ResponseContent): boolean {
  return !value.displayContent.trim() && !value.reasoning.trim() && value.toolCalls.length === 0
}

export {
  extractOpenAIResponseContent,
  extractOllamaResponseContent,
  parseSseEvent,
  parseToolResultEvent,
}
