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

  const curlPreview = useMemo(
    () =>
      `curl -X POST http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '${JSON.stringify({
        model: modelId,
        messages: buildOpenAIMessages(input || "Hello"),
        temperature: params.temperature,
        max_tokens: params.maxTokens,
        top_p: params.topP,
      })}'`,
    [input, modelId, params.maxTokens, params.systemPrompt, params.temperature, params.topP, messages, dragImage],
  )

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
    try {
      const useOllama = backendType === "ollama"
      const response = await fetch(useOllama ? "/api/chat" : "/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(
          useOllama
            ? {
                model: modelId,
                stream: false,
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
                stream: false,
              },
        ),
      })
      if (!response.ok) {
        const err = await response.json().catch(() => ({}))
        throw new Error(String(err?.error?.message ?? err?.detail ?? `HTTP ${response.status}`))
      }
      const data = await response.json()
      const content = useOllama
        ? String(data?.message?.content ?? "")
        : extractOpenAIContent(data)
      const choice = data?.choices?.[0]?.message
      const toolCall = Array.isArray(choice?.tool_calls) ? choice.tool_calls[0] : undefined
      setMessages((prev) => [
        ...prev,
        {
          id: `msg-${Date.now()}-assistant`,
          role: "assistant",
          content: content || "(sin contenido)",
          toolCall: toolCall
            ? {
                name: String(toolCall.function?.name ?? "tool"),
                args: String(toolCall.function?.arguments ?? "{}"),
                result: "(pendiente de resultado)",
              }
            : undefined,
        },
      ])
      setDragImage(null)
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Error en chat")
    } finally {
      setSending(false)
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
    <div className="flex h-[70vh] flex-col rounded-lg border border-border bg-card">
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
          <MessageBubble key={message.id} message={message} />
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
          placeholder="Escribe tu mensaje..."
          className="mb-2 min-h-24 w-full rounded-md border border-border bg-background px-3 py-2"
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
            className="inline-flex items-center gap-2 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground"
          >
            <Send size={14} /> {sending ? "Enviando..." : "Enviar"}
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

function extractOpenAIContent(data: unknown): string {
  const payload = (data ?? {}) as { choices?: Array<{ message?: { content?: unknown } }> }
  const choice = payload.choices?.[0]?.message
  const contentRaw = choice?.content
  if (typeof contentRaw === "string") return contentRaw
  if (!Array.isArray(contentRaw)) return ""
  return contentRaw
    .map((part: { text?: string; type?: string }) =>
      part?.type === "text" && typeof part.text === "string" ? part.text : "",
    )
    .join("\n")
}
