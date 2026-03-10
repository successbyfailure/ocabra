interface ToolCallRendererProps {
  name: string
  args: string
  result: string
}

export function ToolCallRenderer({ name, args, result }: ToolCallRendererProps) {
  return (
    <div className="mt-2 rounded-md border border-cyan-500/30 bg-cyan-500/10 p-2 text-xs">
      <p className="font-semibold text-cyan-100">tool: {name}</p>
      <p className="mt-1 whitespace-pre-wrap text-cyan-50/80">args: {args}</p>
      <p className="mt-1 whitespace-pre-wrap text-cyan-50/80">result: {result}</p>
    </div>
  )
}
