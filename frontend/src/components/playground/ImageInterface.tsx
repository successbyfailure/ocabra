import { useRef, useState } from "react"
import { Download, ImagePlus, Shuffle, Upload, X } from "lucide-react"
import { toast } from "sonner"
import type { PlaygroundParams } from "@/components/playground/ParamsPanel"

type Mode = "generate" | "edit"

interface ImageResult {
  id: string
  url: string
  prompt: string
}

interface ImageInterfaceProps {
  modelId: string
  params: PlaygroundParams
}

export function ImageInterface({ modelId, params }: ImageInterfaceProps) {
  const [mode, setMode] = useState<Mode>("generate")
  const [prompt, setPrompt] = useState("")
  const [negativePrompt, setNegativePrompt] = useState("")
  const [steps, setSteps] = useState(30)
  const [guidance, setGuidance] = useState(7)
  const [width, setWidth] = useState(1024)
  const [height, setHeight] = useState(1024)
  const [seed, setSeed] = useState(42)
  const [strength, setStrength] = useState(0.75)
  const [results, setResults] = useState<ImageResult[]>([])
  const [generating, setGenerating] = useState(false)

  // Edit-mode inputs
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [imagePreview, setImagePreview] = useState<string | null>(null)
  const [maskFile, setMaskFile] = useState<File | null>(null)
  const [maskPreview, setMaskPreview] = useState<string | null>(null)
  const imageInputRef = useRef<HTMLInputElement>(null)
  const maskInputRef = useRef<HTMLInputElement>(null)

  const acceptImage = (file: File | undefined, kind: "image" | "mask") => {
    if (!file) return
    const url = URL.createObjectURL(file)
    if (kind === "image") {
      if (imagePreview) URL.revokeObjectURL(imagePreview)
      setImageFile(file)
      setImagePreview(url)
    } else {
      if (maskPreview) URL.revokeObjectURL(maskPreview)
      setMaskFile(file)
      setMaskPreview(url)
    }
  }

  const clearImage = (kind: "image" | "mask") => {
    if (kind === "image") {
      if (imagePreview) URL.revokeObjectURL(imagePreview)
      setImageFile(null)
      setImagePreview(null)
      if (imageInputRef.current) imageInputRef.current.value = ""
    } else {
      if (maskPreview) URL.revokeObjectURL(maskPreview)
      setMaskFile(null)
      setMaskPreview(null)
      if (maskInputRef.current) maskInputRef.current.value = ""
    }
  }

  const submit = async () => {
    if (!modelId) {
      toast.error("Selecciona un modelo")
      return
    }
    if (!prompt.trim()) {
      toast.error("El prompt no puede estar vacio")
      return
    }
    if (mode === "edit" && !imageFile) {
      toast.error("Sube una imagen base para editar")
      return
    }

    setGenerating(true)
    try {
      let response: Response
      if (mode === "generate") {
        response = await fetch("/v1/images/generations", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model: modelId,
            prompt: prompt.trim(),
            negative_prompt: negativePrompt.trim() || undefined,
            size: `${width}x${height}`,
            n: 1,
            num_inference_steps: steps,
            guidance_scale: guidance,
            seed,
          }),
        })
      } else {
        const fd = new FormData()
        fd.append("model", modelId)
        fd.append("prompt", prompt.trim())
        if (negativePrompt.trim()) fd.append("negative_prompt", negativePrompt.trim())
        fd.append("num_inference_steps", String(steps))
        fd.append("guidance_scale", String(guidance))
        fd.append("strength", String(strength))
        fd.append("seed", String(seed))
        fd.append("n", "1")
        if (imageFile) fd.append("image", imageFile, imageFile.name)
        if (maskFile) fd.append("mask", maskFile, maskFile.name)
        response = await fetch("/v1/images/edits", { method: "POST", body: fd })
      }

      if (!response.ok) {
        const err = await response.json().catch(() => ({}))
        throw new Error(String(err?.error?.message ?? err?.detail ?? `HTTP ${response.status}`))
      }
      const payload = await response.json()
      const images = Array.isArray(payload?.data) ? payload.data : []
      const next = images
        .map((item: { b64_json?: string; url?: string }, idx: number) => {
          if (item?.b64_json) {
            return {
              id: `img-${Date.now()}-${idx}`,
              url: `data:image/png;base64,${item.b64_json}`,
              prompt: prompt.trim(),
            }
          }
          if (item?.url) {
            return { id: `img-${Date.now()}-${idx}`, url: item.url, prompt: prompt.trim() }
          }
          return null
        })
        .filter((item: ImageResult | null): item is ImageResult => item !== null)
      if (next.length === 0) {
        throw new Error("El backend no devolvio imagenes")
      }
      setResults((prev) => [...next, ...prev])
      toast.success(
        mode === "edit"
          ? `Imagen editada con ${modelId}`
          : `Imagen generada con ${modelId} (${params.responseFormat})`,
      )
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Error procesando imagen")
    } finally {
      setGenerating(false)
    }
  }

  const editAvailable = mode === "edit"

  return (
    <div className="space-y-4 rounded-lg border border-border bg-card p-4">
      <div className="flex items-center gap-2">
        <div className="inline-flex rounded-md border border-border bg-background p-0.5 text-xs">
          <button
            type="button"
            onClick={() => setMode("generate")}
            className={`rounded px-3 py-1.5 transition ${
              mode === "generate" ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:bg-muted"
            }`}
          >
            Generar
          </button>
          <button
            type="button"
            onClick={() => setMode("edit")}
            className={`rounded px-3 py-1.5 transition ${
              mode === "edit" ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:bg-muted"
            }`}
          >
            Editar
          </button>
        </div>
        {mode === "edit" && (
          <span className="text-xs text-muted-foreground">
            Sube una imagen base y, opcionalmente, una mascara (alfa transparente = zona a editar).
          </span>
        )}
      </div>

      {editAvailable && (
        <div className="grid gap-3 md:grid-cols-2">
          <ImageDropTile
            label="Imagen base"
            previewUrl={imagePreview}
            onPick={() => imageInputRef.current?.click()}
            onClear={() => clearImage("image")}
            required
          />
          <ImageDropTile
            label="Mascara (opcional)"
            previewUrl={maskPreview}
            onPick={() => maskInputRef.current?.click()}
            onClear={() => clearImage("mask")}
          />
          <input
            ref={imageInputRef}
            type="file"
            accept="image/png,image/jpeg,image/webp"
            className="hidden"
            onChange={(e) => acceptImage(e.target.files?.[0] ?? undefined, "image")}
          />
          <input
            ref={maskInputRef}
            type="file"
            accept="image/png"
            className="hidden"
            onChange={(e) => acceptImage(e.target.files?.[0] ?? undefined, "mask")}
          />
        </div>
      )}

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
        {mode === "generate" ? (
          <>
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
          </>
        ) : (
          <label className="text-sm text-muted-foreground md:col-span-2">
            strength {strength.toFixed(2)}
            <input
              type="range"
              min={0.1}
              max={1.0}
              step={0.05}
              value={strength}
              onChange={(event) => setStrength(Number(event.target.value))}
              className="mt-2 w-full"
            />
            <span className="mt-1 block text-xs text-muted-foreground/80">
              0 = preserva original. 1 = reimagina por completo.
            </span>
          </label>
        )}
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
          onClick={() => void submit()}
          disabled={generating || !modelId}
          className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground disabled:opacity-50"
        >
          {generating
            ? mode === "edit" ? "Editando..." : "Generando..."
            : mode === "edit" ? "Editar" : "Generar"}
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

interface ImageDropTileProps {
  label: string
  previewUrl: string | null
  onPick: () => void
  onClear: () => void
  required?: boolean
}

function ImageDropTile({ label, previewUrl, onPick, onClear, required }: ImageDropTileProps) {
  return (
    <div className="rounded-md border border-dashed border-border bg-background/60 p-3">
      <div className="mb-2 flex items-center justify-between gap-2">
        <span className="text-xs font-medium text-muted-foreground">
          {label}{required ? " *" : ""}
        </span>
        {previewUrl ? (
          <button
            type="button"
            onClick={onClear}
            className="inline-flex items-center gap-1 rounded border border-border px-2 py-0.5 text-[11px] text-muted-foreground hover:bg-muted"
          >
            <X size={11} />
            Quitar
          </button>
        ) : null}
      </div>
      {previewUrl ? (
        <button
          type="button"
          onClick={onPick}
          className="block w-full"
          title="Cambiar imagen"
        >
          <img src={previewUrl} alt={label} className="mx-auto max-h-48 rounded" />
        </button>
      ) : (
        <button
          type="button"
          onClick={onPick}
          className="flex h-32 w-full flex-col items-center justify-center gap-2 rounded border border-dashed border-border/60 text-xs text-muted-foreground hover:bg-muted/50"
        >
          {label.startsWith("Mascara") ? <ImagePlus size={20} /> : <Upload size={20} />}
          <span>Click para subir</span>
        </button>
      )}
    </div>
  )
}
