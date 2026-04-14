import { useMemo, useRef, useState } from "react"
import { Copy, ImagePlus, Send } from "lucide-react"
import { toast } from "sonner"
import { MessageBubble, type ChatMessage } from "@/components/playground/MessageBubble"
import type { PlaygroundParams } from "@/components/playground/ParamsPanel"
import type { BackendType } from "@/types"

interface ChatInterfaceProps {
  modelId: string
  backendType: BackendType | null
  params: PlaygroundParams
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
        return { role: "assistant" as const, content: msg.content }
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
      setStreamingMessageId(assistantId)
      setMessages((prev) => [
        ...prev,
        {
          id: assistantId,
          role: "assistant",
          content: "",
        },
      ])

      const response = await fetch(useOllama ? "/api/chat" : "/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(
          useOllama
            ? {
                model: modelId,
                stream: true,
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
                stream: true,
              },
        ),
      })
      if (!response.ok) {
        const err = await response.json().catch(() => ({}))
        throw new Error(String(err?.error?.message ?? err?.detail ?? `HTTP ${response.status}`))
      }
      const content = useOllama
        ? await readOllamaChatStream(response, (chunk) => {
            setMessages((prev) => updateAssistantMessage(prev, assistantId, chunk))
          })
        : await readOpenAIChatStream(response, (chunk) => {
            setMessages((prev) => updateAssistantMessage(prev, assistantId, chunk))
          })

      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantId
            ? { ...msg, content: msg.content || content || "(sin contenido)" }
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
        className="flex-1 space-y-3 overflow-y-auto p-3"
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
            <span className="text-xs text-primary-foreground/50 ml-0.5">⌘↵</span>
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

function updateAssistantMessage(messages: ChatMessage[], assistantId: string, chunk: string): ChatMessage[] {
  return messages.map((msg) =>
    msg.id === assistantId
      ? { ...msg, content: `${msg.content}${chunk}` }
      : msg,
  )
}

async function readOpenAIChatStream(
  response: Response,
  onChunk: (chunk: string) => void,
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

    let boundary = buffer.indexOf("\n\n")
    while (boundary !== -1) {
      const rawEvent = buffer.slice(0, boundary)
      buffer = buffer.slice(boundary + 2)
      boundary = buffer.indexOf("\n\n")

      const dataLines = rawEvent
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trim())

      if (dataLines.length === 0) continue
      const payload = dataLines.join("\n")
      if (payload === "[DONE]") {
        return fullText
      }

      const parsed = JSON.parse(payload) as {
        choices?: Array<{
          delta?: { content?: unknown }
          message?: { content?: unknown }
        }>
        error?: { message?: string }
      }

      if (parsed.error?.message) throw new Error(parsed.error.message)

      const choice = parsed.choices?.[0]
      const content =
        typeof choice?.delta?.content === "string"
          ? choice.delta.content
          : extractMessageContent(choice?.message?.content)

      if (content) {
        fullText += content
        onChunk(content)
      }
    }
  }

  return fullText
}

async function readOllamaChatStream(
  response: Response,
  onChunk: (chunk: string) => void,
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
        onChunk(content)
      }
    }
  }

  return fullText
}

function extractMessageContent(contentRaw: unknown): string {
  if (typeof contentRaw === "string") return contentRaw
  if (!Array.isArray(contentRaw)) return ""
  return contentRaw
    .map((part: { text?: string; type?: string }) =>
      part?.type === "text" && typeof part.text === "string" ? part.text : "",
    )
    .join("\n")
}
