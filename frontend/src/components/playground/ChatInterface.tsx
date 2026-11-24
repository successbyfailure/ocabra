import { useEffect, useMemo, useRef, useState } from "react"
import { toast } from "sonner"
import { api, type DurationEstimateResult } from "@/api/client"
import {
  type ChatAudioAttachment,
  type ChatMessage,
} from "@/components/playground/MessageBubble"
import type { PlaygroundParams } from "@/components/playground/ParamsPanel"
import { ChatComposer } from "@/components/playground/chat/ChatComposer"
import { ChatMessageList } from "@/components/playground/chat/ChatMessageList"
import { useChatRecorder } from "@/components/playground/chat/useChatRecorder"
import type { BackendType, ChatContentPart, ChatRequestMessage } from "@/types"
import {
  detectAudioFormat,
  fileToBase64,
  probeAudioDurationSec,
  type AudioAttachment,
} from "@/lib/chatAudio"
import {
  extractOpenAIResponseContent,
  finalizeAssistantMessage,
  isEmptyResponseContent,
  readOllamaChatStream,
  readOpenAIChatStream,
  updateAssistantMessage,
  type AssistantPatch,
  type ResponseContent,
} from "@/lib/chatStream"
import { playgroundFetch } from "@/lib/playgroundNode"

interface ChatInterfaceProps {
  modelId: string
  backendType: BackendType | null
  params: PlaygroundParams
  /** Max sequence length of the underlying model (prompt + completion). */
  modelContextLength?: number | null
  /** True if the selected model declares native audio input capability. */
  audioInputCapable?: boolean
}

export function ChatInterface({
  modelId,
  backendType,
  params,
  modelContextLength = null,
  audioInputCapable = false,
}: ChatInterfaceProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState("")
  const [sending, setSending] = useState(false)
  const [streamingMessageId, setStreamingMessageId] = useState<string | null>(null)
  const [dragImage, setDragImage] = useState<string | null>(null)
  const [audioAttachment, setAudioAttachment] = useState<AudioAttachment | null>(null)
  // Becomes true 3 s after a request starts if no token has streamed back —
  // a heuristic for "model is cold-loading or compiling kernels".
  const [slowResponse, setSlowResponse] = useState(false)
  const firstByteRef = useRef(false)
  // Aborts the in-flight chat request when the user hits "Detener".
  const abortRef = useRef<AbortController | null>(null)

  const stopGeneration = () => abortRef.current?.abort()

  const { recording, toggle: toggleRecording } = useChatRecorder({
    audioInputCapable,
    onRecorded: setAudioAttachment,
  })

  // Drive the "cargando modelo" hint. Reset on each request.
  useEffect(() => {
    if (!sending) {
      setSlowResponse(false)
      return
    }
    firstByteRef.current = false
    const handle = window.setTimeout(() => {
      if (!firstByteRef.current) setSlowResponse(true)
    }, 3000)
    return () => window.clearTimeout(handle)
  }, [sending])

  // Attribution shown when the last request went through a Router.
  const [lastRouterInfo, setLastRouterInfo] = useState<{ router: string; target: string } | null>(null)

  // Duration estimate. Fires 400 ms after the last change to the outgoing
  // request so we don't hammer the estimator with a call per keystroke.
  const [estimate, setEstimate] = useState<DurationEstimateResult | null>(null)
  const [estimateLoading, setEstimateLoading] = useState(false)
  useEffect(() => {
    if (!modelId) {
      setEstimate(null)
      return
    }
    if (sending) return  // frozen while the request is in flight
    const previewMessages = buildOpenAIMessages(input || "Hola")
    const body: Record<string, unknown> = {
      model: modelId,
      messages: previewMessages,
      max_tokens: params.maxTokens,
    }
    const handle = window.setTimeout(async () => {
      setEstimateLoading(true)
      try {
        const result = await api.estimate.predict(modelId, body)
        setEstimate(result)
      } catch {
        setEstimate(null)
      } finally {
        setEstimateLoading(false)
      }
    }, 400)
    return () => window.clearTimeout(handle)
  }, [modelId, input, params.maxTokens, params.systemPrompt, messages.length, dragImage, audioAttachment, sending])

  const buildOpenAIMessages = (userText: string): ChatRequestMessage[] => {
    const history: ChatRequestMessage[] = messages.map((msg) => {
      if (msg.role === "assistant") {
        return { role: "assistant", content: msg.apiContent ?? msg.content }
      }
      return { role: "user", content: msg.content }
    })

    let userContent: string | ChatContentPart[]
    if (dragImage || audioAttachment) {
      const parts: ChatContentPart[] = []
      const fallbackText = audioAttachment
        ? "Transcribe / responde al audio"
        : "Describe la imagen"
      parts.push({ type: "text", text: userText || fallbackText })
      if (dragImage) {
        parts.push({ type: "image_url", image_url: { url: dragImage } })
      }
      if (audioAttachment) {
        parts.push({
          type: "input_audio",
          input_audio: {
            data: audioAttachment.dataBase64,
            format: audioAttachment.format,
          },
        })
      }
      userContent = parts
    } else {
      userContent = userText
    }

    return [
      { role: "system", content: params.systemPrompt },
      ...history,
      { role: "user", content: userContent },
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
  }, [input, modelId, params.maxTokens, params.systemPrompt, params.temperature, params.topP, messages, dragImage, audioAttachment])

  const sendMessage = async () => {
    if (!modelId) {
      toast.error("Selecciona un modelo")
      return
    }
    const text = input.trim()
    if (!text && !dragImage && !audioAttachment) return
    if (audioAttachment && !audioInputCapable) {
      toast.error("El modelo seleccionado no acepta audio como entrada")
      return
    }

    // Rough client-side check: ~4 chars/token (OpenAI estimate). Counts the
    // system prompt + history + the new user turn. If the total + max_tokens
    // is going to exceed the model's context window the server will return
    // a 400 anyway, but giving immediate feedback is much nicer than a
    // generic 500 once vLLM rejects the request.
    if (modelContextLength && modelContextLength > 0) {
      const charBudget =
        (params.systemPrompt?.length ?? 0) +
        messages.reduce((sum, m) => sum + (m.apiContent ?? m.content ?? "").length, 0) +
        text.length +
        (dragImage ? 4096 : 0) // image tokens are huge; rough placeholder
      const estPromptTokens = Math.ceil(charBudget / 4)
      const total = estPromptTokens + params.maxTokens
      if (total > modelContextLength) {
        toast.error(
          `Demasiados tokens. Estimado ${estPromptTokens.toLocaleString()} prompt + ` +
            `${params.maxTokens.toLocaleString()} max_tokens = ${total.toLocaleString()} > ` +
            `${modelContextLength.toLocaleString()} contexto del modelo. ` +
            `Reduce max_tokens o limpia el historial.`,
          { duration: 8000 },
        )
        return
      }
    }

    const fallbackContent = audioAttachment ? "[audio]" : "[imagen]"
    const userMessage: ChatMessage = {
      id: `msg-${Date.now()}`,
      role: "user",
      content: text || fallbackContent,
      image: dragImage ?? undefined,
      audio: audioAttachment
        ? ({
            name: audioAttachment.name,
            format: audioAttachment.format,
            durationSec: audioAttachment.durationSec,
          } satisfies ChatAudioAttachment)
        : undefined,
    }

    setMessages((prev) => [...prev, userMessage])
    setInput("")
    setSending(true)
    const controller = new AbortController()
    abortRef.current = controller
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

      const response = await playgroundFetch(useOllama ? "/api/chat" : "/v1/chat/completions", {
        method: "POST",
        // Opt into oCabra tool-progress SSE events (event: ocabra.tool_*) so the
        // agent tool cards render. Plain OpenAI clients omit this and get a
        // standards-clean stream.
        headers: { "Content-Type": "application/json", "X-Ocabra-Stream-Events": "true" },
        body: requestBody,
        signal: controller.signal,
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

      // Bloque 20 — surface router redirects. When the request resolved a
      // Router profile, the backend echoes X-Ocabra-Router (the router
      // profile_id) and X-Ocabra-Router-Target (the actual worker profile).
      const routerId = response.headers.get("X-Ocabra-Router")
      const routerTarget = response.headers.get("X-Ocabra-Router-Target")
      if (routerId && routerTarget && routerTarget !== routerId) {
        setLastRouterInfo({ router: routerId, target: routerTarget })
      } else {
        setLastRouterInfo(null)
      }

      const onPatch = (patch: AssistantPatch) => {
        if (!firstByteRef.current) {
          firstByteRef.current = true
          setSlowResponse(false)
        }
        setMessages((prev) => updateAssistantMessage(prev, assistantId, patch))
      }
      let responseContent = useOllama
        ? {
            displayContent: await readOllamaChatStream(response, onPatch),
            apiContent: "",
            reasoning: "",
            toolCalls: [],
          }
        : await readOpenAIChatStream(response, onPatch)

      if (!useOllama && isEmptyResponseContent(responseContent) && !controller.signal.aborted) {
        responseContent = await retryOpenAINonStreaming(requestBody, controller.signal)
      }

      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantId
            ? finalizeAssistantMessage(msg, responseContent)
            : msg,
        ),
      )
      setDragImage(null)
      setAudioAttachment(null)
    } catch (err) {
      // User hit "Detener": keep whatever streamed so far, no error toast.
      if (controller.signal.aborted || (err instanceof DOMException && err.name === "AbortError")) {
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantId && !(msg.content || msg.reasoning || (msg.toolCalls?.length ?? 0))
              ? { ...msg, content: "_(detenido)_" }
              : msg,
          ),
        )
        toast.info("Generación detenida")
      } else {
        setMessages((prev) => prev.filter((msg) => msg.id !== assistantId))
        toast.error(err instanceof Error ? err.message : "Error en chat")
      }
    } finally {
      abortRef.current = null
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

  const onAttachAudioFile = async (file: File) => {
    if (!audioInputCapable) {
      toast.error("El modelo seleccionado no acepta audio como entrada")
      return
    }
    try {
      const format = detectAudioFormat(file)
      if (!format) {
        toast.error("Formato de audio no soportado")
        return
      }
      const dataBase64 = await fileToBase64(file)
      const durationSec = await probeAudioDurationSec(file).catch(() => null)
      setAudioAttachment({ name: file.name, format, dataBase64, durationSec })
      toast.success("Audio adjuntado")
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Error leyendo audio")
    }
  }

  return (
    <div className="flex h-full flex-col rounded-lg border border-border bg-card">
      <ChatMessageList
        messages={messages}
        streamingMessageId={streamingMessageId}
        slowResponse={slowResponse}
        lastRouterInfo={lastRouterInfo}
        onImageDrop={onDropImage}
        onAudioDrop={(file) => void onAttachAudioFile(file)}
      />
      <ChatComposer
        input={input}
        onInputChange={setInput}
        onSend={() => void sendMessage()}
        onStop={stopGeneration}
        sending={sending}
        modelId={modelId}
        audioInputCapable={audioInputCapable}
        recording={recording}
        dragImage={dragImage}
        audioAttachment={audioAttachment}
        onRemoveImage={() => setDragImage(null)}
        onRemoveAudio={() => setAudioAttachment(null)}
        onImageSelected={onDropImage}
        onAudioSelected={(file) => void onAttachAudioFile(file)}
        onToggleRecord={toggleRecording}
        curlPreview={curlPreview}
        estimate={estimate}
        estimateLoading={estimateLoading}
      />
    </div>
  )
}

async function retryOpenAINonStreaming(
  requestBody: string,
  signal?: AbortSignal,
): Promise<ResponseContent> {
  const parsedBody = JSON.parse(requestBody) as Record<string, unknown>
  const retryBody = JSON.stringify({ ...parsedBody, stream: false })
  const response = await playgroundFetch("/v1/chat/completions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: retryBody,
    signal,
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
