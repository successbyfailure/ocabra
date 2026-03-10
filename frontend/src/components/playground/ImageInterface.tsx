import { useState } from "react"
import { Download, Shuffle } from "lucide-react"
import { toast } from "sonner"
import type { PlaygroundParams } from "@/components/playground/ParamsPanel"

interface ImageResult {
  id: string
  url: string
  prompt: string
}

interface ImageInterfaceProps {
  modelId: string
  params: PlaygroundParams
}

function createPlaceholder(prompt: string, seed: number, width: number, height: number): string {
  const svg = `<svg xmlns='http://www.w3.org/2000/svg' width='${width}' height='${height}'><defs><linearGradient id='g' x1='0' y1='0' x2='1' y2='1'><stop offset='0%' stop-color='#0ea5e9'/><stop offset='100%' stop-color='#14b8a6'/></linearGradient></defs><rect width='100%' height='100%' fill='url(#g)'/><text x='24' y='48' fill='white' font-family='monospace' font-size='20'>${prompt.slice(0, 52)}</text><text x='24' y='80' fill='white' font-family='monospace' font-size='13'>seed ${seed}</text></svg>`
  return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`
}

export function ImageInterface({ modelId, params }: ImageInterfaceProps) {
  const [prompt, setPrompt] = useState("")
  const [negativePrompt, setNegativePrompt] = useState("")
  const [steps, setSteps] = useState(30)
  const [guidance, setGuidance] = useState(7)
  const [width, setWidth] = useState(1024)
  const [height, setHeight] = useState(1024)
  const [seed, setSeed] = useState(42)
  const [results, setResults] = useState<ImageResult[]>([])

  const generate = () => {
    if (!prompt.trim()) return
    const id = `img-${Date.now()}`
    const url = createPlaceholder(prompt, seed, width, height)
    setResults((prev) => [{ id, url, prompt }, ...prev])
    toast.success(`Imagen generada con ${modelId} (${params.responseFormat})`)
  }

  return (
    <div className="space-y-4 rounded-lg border border-border bg-card p-4">
      <div className="grid gap-2">
        <label className="text-sm text-muted-foreground">
          Prompt
          <textarea
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            className="mt-1 min-h-20 w-full rounded-md border border-border bg-background px-3 py-2"
          />
        </label>
        <label className="text-sm text-muted-foreground">
          Negative prompt
          <textarea
            value={negativePrompt}
            onChange={(event) => setNegativePrompt(event.target.value)}
            className="mt-1 min-h-16 w-full rounded-md border border-border bg-background px-3 py-2"
          />
        </label>
      </div>

      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        <label className="text-sm text-muted-foreground">
          steps {steps}
          <input type="range" min={10} max={80} value={steps} onChange={(event) => setSteps(Number(event.target.value))} className="mt-2 w-full" />
        </label>
        <label className="text-sm text-muted-foreground">
          guidance {guidance.toFixed(1)}
          <input type="range" min={1} max={20} step={0.1} value={guidance} onChange={(event) => setGuidance(Number(event.target.value))} className="mt-2 w-full" />
        </label>
        <label className="text-sm text-muted-foreground">
          width
          <select value={width} onChange={(event) => setWidth(Number(event.target.value))} className="mt-1 w-full rounded-md border border-border bg-background px-2 py-2">
            <option value={512}>512</option>
            <option value={768}>768</option>
            <option value={1024}>1024</option>
            <option value={1280}>1280</option>
          </select>
        </label>
        <label className="text-sm text-muted-foreground">
          height
          <select value={height} onChange={(event) => setHeight(Number(event.target.value))} className="mt-1 w-full rounded-md border border-border bg-background px-2 py-2">
            <option value={512}>512</option>
            <option value={768}>768</option>
            <option value={1024}>1024</option>
            <option value={1280}>1280</option>
          </select>
        </label>
      </div>

      <div className="flex flex-wrap items-end gap-3">
        <label className="text-sm text-muted-foreground">
          Seed
          <input
            type="number"
            value={seed}
            onChange={(event) => setSeed(Number(event.target.value))}
            className="mt-1 w-32 rounded-md border border-border bg-background px-3 py-2"
          />
        </label>
        <button
          type="button"
          onClick={() => setSeed(Math.floor(Math.random() * 999_999))}
          className="inline-flex items-center gap-1 rounded-md border border-border px-3 py-2 text-sm hover:bg-muted"
        >
          <Shuffle size={14} />
          Random
        </button>
        <button
          type="button"
          onClick={generate}
          className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground"
        >
          Generar
        </button>
      </div>

      <div className="grid gap-3 md:grid-cols-2">
        {results.map((item) => (
          <figure key={item.id} className="rounded-md border border-border bg-background/60 p-2">
            <img src={item.url} alt={item.prompt} className="w-full rounded" />
            <figcaption className="mt-2 flex items-center justify-between gap-2">
              <span className="line-clamp-1 text-xs text-muted-foreground">{item.prompt}</span>
              <a
                href={item.url}
                download={`${item.id}.png`}
                className="inline-flex items-center gap-1 rounded-md border border-border px-2 py-1 text-xs hover:bg-muted"
              >
                <Download size={12} />
                Descargar
              </a>
            </figcaption>
          </figure>
        ))}
      </div>
    </div>
  )
}
