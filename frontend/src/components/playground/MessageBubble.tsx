import ReactMarkdown from "react-markdown"
import rehypeHighlight from "rehype-highlight"
import { ToolCallRenderer } from "@/components/playground/ToolCallRenderer"

export interface ChatMessage {
  id: string
  role: "user" | "assistant" | "tool"
  content: string
  image?: string
  toolCall?: {
    name: string
    args: string
    result: string
  }
}

interface MessageBubbleProps {
  message: ChatMessage
  streaming?: boolean
}

export function MessageBubble({ message, streaming }: MessageBubbleProps) {
  const isUser = message.role === "user"

  return (
    <div className={`max-w-[90%] rounded-lg border px-3 py-2 ${isUser ? "ml-auto border-primary/40 bg-primary/10" : "border-border bg-card"}`}>
      <p className="mb-1 text-xs uppercase tracking-wide text-muted-foreground">{message.role}</p>
      {message.image && (
        <img src={message.image} alt="Vision input" className="mb-2 max-h-64 rounded-md border border-border" />
      )}
      <div className="prose prose-invert max-w-none text-sm">
        <ReactMarkdown rehypePlugins={[rehypeHighlight]}>{message.content}</ReactMarkdown>
        {streaming && <span className="ml-1 inline-block h-4 w-2 animate-pulse bg-primary align-middle" />}
      </div>
      {message.toolCall && (
        <ToolCallRenderer name={message.toolCall.name} args={message.toolCall.args} result={message.toolCall.result} />
      )}
    </div>
  )
}
