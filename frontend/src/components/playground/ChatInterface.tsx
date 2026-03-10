import { useMemo, useRef, useState } from "react"
import { Copy, ImagePlus, Send } from "lucide-react"
import { toast } from "sonner"
import { MessageBubble, type ChatMessage } from "@/components/playground/MessageBubble"
import type { PlaygroundParams } from "@/components/playground/ParamsPanel"

interface ChatInterfaceProps {
  modelId: string
  params: PlaygroundParams
}

export function ChatInterface({ modelId, params }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState("")
  const [streamingId, setStreamingId] = useState<string | null>(null)
  const [dragImage, setDragImage] = useState<string | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  const curlPreview = useMemo(
    () =>
      `curl -X POST http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '${JSON.stringify({
        model: modelId,
        messages: [{ role: "user", content: input || "Hello" }],
        temperature: params.temperature,
        max_tokens: params.maxTokens,
        top_p: params.topP,
      })}'`,
    [input, modelId, params.maxTokens, params.temperature, params.topP],
  )

  const streamAssistant = (content: string, base: ChatMessage) => {
    setStreamingId(base.id)
    let index = 0
    const timer = window.setInterval(() => {
      index += 2
      setMessages((prev) => prev.map((msg) => (msg.id === base.id ? { ...msg, content: content.slice(0, index) } : msg)))
      if (index >= content.length) {
        window.clearInterval(timer)
        setStreamingId(null)
      }
    }, 22)
  }

  const sendMessage = () => {
    const text = input.trim()
    if (!text && !dragImage) return

    const userMessage: ChatMessage = {
      id: `msg-${Date.now()}`,
      role: "user",
      content: text || "[imagen]",
      image: dragImage ?? undefined,
    }

    const assistantMessage: ChatMessage = {
      id: `msg-${Date.now()}-assistant`,
      role: "assistant",
      content: "",
    }

    const withTool = text.toLowerCase().includes("tool")
    const response = withTool
      ? "He ejecutado una llamada de herramienta y este es el resultado."
      : `Respuesta simulada de **${modelId}** con temperature=${params.temperature.toFixed(2)}.`

    setMessages((prev) => [
      ...prev,
      userMessage,
      {
        ...assistantMessage,
        toolCall: withTool
          ? {
              name: "search_local_models",
              args: '{"limit": 5}',
              result: '["mistral-7b", "llama3-8b"]',
            }
          : undefined,
      },
    ])

    streamAssistant(response, assistantMessage)
    setInput("")
    setDragImage(null)
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
          <MessageBubble key={message.id} message={message} streaming={streamingId === message.id} />
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
            onClick={sendMessage}
            className="inline-flex items-center gap-2 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground"
          >
            <Send size={14} /> Enviar
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
