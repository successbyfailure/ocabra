import { useCallback, useEffect, useRef, useState } from "react"
import {
  Copy,
  Download,
  Edit3,
  ImagePlus,
  RotateCcw,
  Shuffle,
  Upload,
  X,
} from "lucide-react"
import { toast } from "sonner"
import type { PlaygroundParams } from "@/components/playground/ParamsPanel"

type Mode = "generate" | "edit"

interface ImageResult {
  id: string
  url: string
  prompt: string
  mode: Mode
  seed: number
  steps: number
  guidance: number
  strength?: number
  width: number
  height: number
  durationMs?: number
}

interface AspectPreset {
  key: string
  label: string
  ratio: string
  width: number
  height: number
}

// Presets tuned to Qwen-Image family: cuadrado, retrato, apaisado y las dos
// panorámicas oficiales del paper (1328x768, 928x1664, etc.). Los tamaños
// están dentro del envelope que el modelo entrena sin degradar composición.
const ASPECT_PRESETS: AspectPreset[] = [
  { key: "1_1", label: "Cuadrado", ratio: "1:1", width: 1024, height: 1024 },
  { key: "4_3", label: "Foto H", ratio: "4:3", width: 1152, height: 864 },
  { key: "3_4", label: "Foto V", ratio: "3:4", width: 864, height: 1152 },
  { key: "3_2", label: "Cine H", ratio: "3:2", width: 1216, height: 832 },
  { key: "2_3", label: "Cine V", ratio: "2:3", width: 832, height: 1216 },
  { key: "16_9", label: "Ancha", ratio: "16:9", width: 1344, height: 768 },
  { key: "9_16", label: "Alta", ratio: "9:16", width: 768, height: 1344 },
]

const MAX_REFS = 3

async function fileFromUrl(url: string, name = "input.png"): Promise<File> {
  const res = await fetch(url)
  const blob = await res.blob()
  return new File([blob], name, { type: blob.type || "image/png" })
}

async function fileFromClipboard(): Promise<File | null> {
  const nav = navigator as Navigator & {
    clipboard?: { read?: () => Promise<ClipboardItem[]> }
  }
  if (!nav.clipboard?.read) return null
  const items = await nav.clipboard.read()
  for (const item of items) {
    for (const type of item.types) {
      if (type.startsWith("image/")) {
        const blob = await item.getType(type)
        return new File([blob], "pasted.png", { type })
      }
    }
  }
  return null
}

interface ImageInterfaceProps {
  modelId: string
  params: PlaygroundParams
  canGenerate?: boolean
  canEdit?: boolean
  supportsMultiRef?: boolean
  editDraft?: { url: string; prompt: string } | null
  onDraftConsumed?: () => void
  onSendToEdit?: (url: string, prompt: string) => void
}

export function ImageInterface({
  modelId,
  params,
  canGenerate = true,
  canEdit = true,
  supportsMultiRef = false,
  editDraft,
  onDraftConsumed,
  onSendToEdit,
}: ImageInterfaceProps) {
  const [mode, setMode] = useState<Mode>(canGenerate ? "generate" : "edit")

  useEffect(() => {
    // Both false = the model isn't loaded yet (capabilities default to false
    // until the worker reports back). Skip forcing a mode — the empty
    // interface is fine and the user still sees generate/edit toggles.
    if (!canEdit && !canGenerate) return
    if (!canEdit && mode === "edit") setMode("generate")
    else if (!canGenerate && mode === "generate") setMode("edit")
  }, [canEdit, canGenerate, mode])

  const [prompt, setPrompt] = useState("")
  const [negativePrompt, setNegativePrompt] = useState("")
  const [steps, setSteps] = useState(30)
  const [guidance, setGuidance] = useState(4)
  const [aspectKey, setAspectKey] = useState<string>("1_1")
  const [width, setWidth] = useState(1024)
  const [height, setHeight] = useState(1024)
  const [n, setN] = useState(1)
  const [seed, setSeed] = useState(42)
  const [strength, setStrength] = useState(0.75)
  const [results, setResults] = useState<ImageResult[]>([])
  const [generating, setGenerating] = useState(false)

  // Edit-mode inputs — hasta 3 imágenes de referencia en modelos que lo
  // soportan (Qwen-Image-Edit-Plus). El primer slot es la imagen principal.
  const [images, setImages] = useState<Array<{ file: File; preview: string } | null>>([
    null,
    null,
    null,
  ])
  const [maskFile, setMaskFile] = useState<File | null>(null)
  const [maskPreview, setMaskPreview] = useState<string | null>(null)
  const imageInputRefs = useRef<Array<HTMLInputElement | null>>([null, null, null])
  const maskInputRef = useRef<HTMLInputElement>(null)
  const rootRef = useRef<HTMLDivElement>(null)

  const applyPreset = useCallback((key: string) => {
    setAspectKey(key)
    if (key === "custom") return
    const preset = ASPECT_PRESETS.find((p) => p.key === key)
    if (!preset) return
    setWidth(preset.width)
    setHeight(preset.height)
  }, [])

  const setImageAt = useCallback((idx: number, file: File | null) => {
    setImages((prev) => {
      const next = [...prev]
      const old = next[idx]
      if (old?.preview) URL.revokeObjectURL(old.preview)
      next[idx] = file ? { file, preview: URL.createObjectURL(file) } : null
      return next
    })
  }, [])

  // Consume "usar como base de edit" drafts sent from another tab / from the
  // gallery. The parent hands us a `{url, prompt}` object once; we load the
  // image, apply the prompt, then notify the parent to clear its slot.
  useEffect(() => {
    if (!editDraft) return
    let cancelled = false
    setMode("edit")
    ;(async () => {
      try {
        const file = await fileFromUrl(editDraft.url, "input.png")
        if (cancelled) return
        setImageAt(0, file)
        if (editDraft.prompt) {
          setPrompt((prev) => (prev.trim() ? prev : editDraft.prompt))
        }
      } catch (err) {
        if (!cancelled) toast.error("No se pudo cargar la imagen como base")
        console.error(err)
      } finally {
        if (!cancelled) onDraftConsumed?.()
      }
    })()
    return () => {
      cancelled = true
    }
  }, [editDraft, setImageAt, onDraftConsumed])

  // Ctrl/Cmd+V mientras el foco está en el componente pega la imagen del
  // portapapeles al primer slot libre (o al principal si todos ocupados).
  useEffect(() => {
    const el = rootRef.current
    if (!el) return
    const onPaste = (evt: ClipboardEvent) => {
      if (mode !== "edit") return
      const item = Array.from(evt.clipboardData?.items ?? []).find((i) =>
        i.type.startsWith("image/"),
      )
      if (!item) return
      const file = item.getAsFile()
      if (!file) return
      evt.preventDefault()
      const nextIdx = images.findIndex((slot, idx) => slot === null && (idx === 0 || supportsMultiRef))
      setImageAt(nextIdx === -1 ? 0 : nextIdx, file)
      toast.success("Imagen pegada")
    }
    el.addEventListener("paste", onPaste as EventListener)
    return () => el.removeEventListener("paste", onPaste as EventListener)
  }, [mode, images, supportsMultiRef, setImageAt])

  const pasteFromClipboard = async (idx: number) => {
    try {
      const file = await fileFromClipboard()
      if (!file) {
        toast.error("No hay imagen en el portapapeles")
        return
      }
      setImageAt(idx, file)
    } catch (err) {
      toast.error("El navegador no da acceso al portapapeles")
      console.error(err)
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
    if (mode === "edit" && !images[0]) {
      toast.error("Sube una imagen base para editar")
      return
    }

    const perImageSeed = seed
    setGenerating(true)
    const t0 = performance.now()
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
            n: Math.max(1, Math.min(4, n)),
            num_inference_steps: steps,
            guidance_scale: guidance,
            seed: perImageSeed,
            response_format: "url",
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
        fd.append("seed", String(perImageSeed))
        fd.append("n", String(Math.max(1, Math.min(4, n))))
        fd.append("response_format", "url")
        const primary = images[0]!
        fd.append("image", primary.file, primary.file.name)
        if (supportsMultiRef) {
          for (let i = 1; i < images.length; i++) {
            const ref = images[i]
            if (ref) fd.append(`image_ref_${i}`, ref.file, ref.file.name)
          }
        }
        if (maskFile) fd.append("mask", maskFile, maskFile.name)
        response = await fetch("/v1/images/edits", { method: "POST", body: fd })
      }

      if (!response.ok) {
        const err = await response.json().catch(() => ({}))
        throw new Error(String(err?.error?.message ?? err?.detail ?? `HTTP ${response.status}`))
      }
      const payload = await response.json()
      const items = Array.isArray(payload?.data) ? payload.data : []
      const durationMs = Math.round(performance.now() - t0)
      const next: ImageResult[] = items
        .map((item: { b64_json?: string; url?: string }, idx: number) => {
          const url = item?.b64_json
            ? `data:image/png;base64,${item.b64_json}`
            : item?.url
          if (!url) return null
          return {
            id: `img-${Date.now()}-${idx}`,
            url,
            prompt: prompt.trim(),
            mode,
            seed: perImageSeed,
            steps,
            guidance,
            strength: mode === "edit" ? strength : undefined,
            width,
            height,
            durationMs: idx === 0 ? durationMs : undefined,
          }
        })
        .filter((x: ImageResult | null): x is ImageResult => x !== null)
      if (next.length === 0) throw new Error("El backend no devolvio imagenes")
      setResults((prev) => [...next, ...prev])
      toast.success(
        mode === "edit"
          ? `${next.length} imagen(es) editada(s) en ${(durationMs / 1000).toFixed(1)}s`
          : `${next.length} imagen(es) generada(s) en ${(durationMs / 1000).toFixed(1)}s (${params.responseFormat})`,
      )
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Error procesando imagen")
    } finally {
      setGenerating(false)
    }
  }

  const editAvailable = mode === "edit"
  const activeRefCount = images.filter(Boolean).length
  const visibleRefSlots = supportsMultiRef ? MAX_REFS : 1

  const copyToClipboard = async (value: string, label: string) => {
    try {
      await navigator.clipboard.writeText(value)
      toast.success(`${label} copiado`)
    } catch {
      toast.error("No se pudo copiar")
    }
  }

  const useAsEditBase = async (result: ImageResult) => {
    if (canEdit) {
      try {
        const file = await fileFromUrl(result.url, `input-${result.id}.png`)
        setMode("edit")
        setImageAt(0, file)
        if (!prompt.trim()) setPrompt(result.prompt)
        toast.success("Imagen cargada en el editor")
      } catch (err) {
        toast.error("No se pudo cargar la imagen")
        console.error(err)
      }
      return
    }
    if (onSendToEdit) onSendToEdit(result.url, result.prompt)
    else toast.error("No hay modelo de edición seleccionado")
  }

  return (
    <div
      ref={rootRef}
      tabIndex={0}
      className="space-y-4 rounded-lg border border-border bg-card p-4 focus:outline-none"
    >
      <div className="flex items-center gap-2">
        <div className="inline-flex rounded-md border border-border bg-background p-0.5 text-xs">
          <button
            type="button"
            onClick={() => setMode("generate")}
            disabled={!canGenerate}
            title={canGenerate ? undefined : "Este modelo solo edita, no genera desde texto"}
            className={`rounded px-3 py-1.5 transition ${
              mode === "generate" ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:bg-muted"
            } ${canGenerate ? "" : "cursor-not-allowed opacity-50 hover:bg-transparent"}`}
          >
            Generar
          </button>
          <button
            type="button"
            onClick={() => setMode("edit")}
            disabled={!canEdit}
            title={canEdit ? undefined : "Este modelo no admite edicion de imagenes"}
            className={`rounded px-3 py-1.5 transition ${
              mode === "edit" ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:bg-muted"
            } ${canEdit ? "" : "cursor-not-allowed opacity-50 hover:bg-transparent"}`}
          >
            Editar
          </button>
        </div>
        {mode === "edit" && (
          <span className="text-xs text-muted-foreground">
            {supportsMultiRef
              ? `Sube hasta ${MAX_REFS} imagenes de referencia (pega con Ctrl+V o arrastra). ${activeRefCount}/${MAX_REFS} activas.`
              : "Sube una imagen base (pega con Ctrl+V o arrastra) y opcionalmente una mascara PNG con alfa."}
          </span>
        )}
      </div>

      {editAvailable && (
        <div className={`grid gap-3 ${supportsMultiRef ? "md:grid-cols-2 lg:grid-cols-4" : "md:grid-cols-2"}`}>
          {Array.from({ length: visibleRefSlots }, (_, i) => (
            <ImageDropTile
              key={i}
              label={i === 0 ? "Imagen base" : `Ref ${i + 1}`}
              previewUrl={images[i]?.preview ?? null}
              onPick={() => imageInputRefs.current[i]?.click()}
              onClear={() => setImageAt(i, null)}
              onPaste={() => void pasteFromClipboard(i)}
              onDropFile={(f) => setImageAt(i, f)}
              required={i === 0}
            />
          ))}
          <ImageDropTile
            label="Mascara"
            previewUrl={maskPreview}
            onPick={() => maskInputRef.current?.click()}
            onClear={() => {
              if (maskPreview) URL.revokeObjectURL(maskPreview)
              setMaskFile(null)
              setMaskPreview(null)
              if (maskInputRef.current) maskInputRef.current.value = ""
            }}
            onDropFile={(f) => {
              if (maskPreview) URL.revokeObjectURL(maskPreview)
              setMaskFile(f)
              setMaskPreview(URL.createObjectURL(f))
            }}
            hint="Alfa transparente = zona a editar"
          />
          {Array.from({ length: visibleRefSlots }, (_, i) => (
            <input
              key={i}
              ref={(el) => { imageInputRefs.current[i] = el }}
              type="file"
              accept="image/png,image/jpeg,image/webp"
              className="hidden"
              onChange={(e) => {
                const f = e.target.files?.[0]
                if (f) setImageAt(i, f)
              }}
            />
          ))}
          <input
            ref={maskInputRef}
            type="file"
            accept="image/png"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0]
              if (!f) return
              if (maskPreview) URL.revokeObjectURL(maskPreview)
              setMaskFile(f)
              setMaskPreview(URL.createObjectURL(f))
            }}
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

      {mode === "generate" && (
        <div className="space-y-2">
          <div className="flex flex-wrap items-center gap-1.5">
            {ASPECT_PRESETS.map((p) => (
              <button
                key={p.key}
                type="button"
                onClick={() => applyPreset(p.key)}
                className={`rounded-md border px-2.5 py-1 text-xs transition-colors ${
                  aspectKey === p.key
                    ? "border-primary bg-primary/10 text-primary"
                    : "border-border text-muted-foreground hover:bg-muted"
                }`}
                title={`${p.width}×${p.height}`}
              >
                {p.label} <span className="text-[10px] opacity-70">{p.ratio}</span>
              </button>
            ))}
            <button
              type="button"
              onClick={() => setAspectKey("custom")}
              className={`rounded-md border px-2.5 py-1 text-xs transition-colors ${
                aspectKey === "custom"
                  ? "border-primary bg-primary/10 text-primary"
                  : "border-border text-muted-foreground hover:bg-muted"
              }`}
            >
              Custom
            </button>
          </div>
          {aspectKey === "custom" && (
            <div className="grid gap-3 md:grid-cols-2">
              <label className="text-sm text-muted-foreground">
                width
                <input
                  type="number"
                  min={256}
                  max={2048}
                  step={8}
                  value={width}
                  onChange={(e) => setWidth(Math.max(256, Math.min(2048, Number(e.target.value) || 1024)))}
                  className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2"
                />
              </label>
              <label className="text-sm text-muted-foreground">
                height
                <input
                  type="number"
                  min={256}
                  max={2048}
                  step={8}
                  value={height}
                  onChange={(e) => setHeight(Math.max(256, Math.min(2048, Number(e.target.value) || 1024)))}
                  className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2"
                />
              </label>
            </div>
          )}
        </div>
      )}

      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        <label className="text-sm text-muted-foreground">
          steps {steps}
          <input type="range" min={10} max={80} value={steps} onChange={(event) => setSteps(Number(event.target.value))} className="mt-2 w-full" />
        </label>
        <label className="text-sm text-muted-foreground">
          guidance {guidance.toFixed(1)}
          <input type="range" min={1} max={20} step={0.1} value={guidance} onChange={(event) => setGuidance(Number(event.target.value))} className="mt-2 w-full" />
        </label>
        {mode === "edit" && (
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
          n
          <input
            type="number"
            min={1}
            max={4}
            value={n}
            onChange={(event) => setN(Math.max(1, Math.min(4, Number(event.target.value) || 1)))}
            className="mt-1 w-20 rounded-md border border-border bg-background px-3 py-2"
          />
        </label>
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
            <figcaption className="mt-2 space-y-2 text-xs text-muted-foreground">
              <div className="line-clamp-2 break-words" title={item.prompt}>{item.prompt}</div>
              <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-muted-foreground/80">
                <span className={item.mode === "edit" ? "text-primary" : ""}>{item.mode}</span>
                <span>{item.width}×{item.height}</span>
                <span>steps {item.steps}</span>
                <span>guidance {item.guidance.toFixed(1)}</span>
                {item.strength !== undefined && <span>strength {item.strength.toFixed(2)}</span>}
                <span>seed {item.seed}</span>
                {item.durationMs !== undefined && <span>{(item.durationMs / 1000).toFixed(1)}s</span>}
              </div>
              <div className="flex flex-wrap gap-1">
                <a
                  href={item.url}
                  download={`${item.id}.png`}
                  className="inline-flex items-center gap-1 rounded-md border border-border px-2 py-1 text-xs hover:bg-muted"
                >
                  <Download size={12} />
                  Descargar
                </a>
                {(canEdit || onSendToEdit) && (
                  <button
                    type="button"
                    onClick={() => void useAsEditBase(item)}
                    className="inline-flex items-center gap-1 rounded-md border border-border px-2 py-1 text-xs hover:bg-muted"
                    title="Cargar esta imagen como base de edicion"
                  >
                    <Edit3 size={12} />
                    Editar
                  </button>
                )}
                <button
                  type="button"
                  onClick={() => setSeed(item.seed)}
                  className="inline-flex items-center gap-1 rounded-md border border-border px-2 py-1 text-xs hover:bg-muted"
                  title="Reutilizar seed"
                >
                  <RotateCcw size={12} />
                  Seed
                </button>
                <button
                  type="button"
                  onClick={() => setPrompt(item.prompt)}
                  className="inline-flex items-center gap-1 rounded-md border border-border px-2 py-1 text-xs hover:bg-muted"
                  title="Cargar el prompt"
                >
                  <Copy size={12} />
                  Prompt
                </button>
                <button
                  type="button"
                  onClick={() => void copyToClipboard(String(item.seed), "seed")}
                  className="inline-flex items-center gap-1 rounded-md border border-border px-2 py-1 text-xs hover:bg-muted"
                  title="Copiar seed al portapapeles"
                >
                  <Copy size={12} />
                  Copy seed
                </button>
              </div>
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
  onPaste?: () => void
  onDropFile?: (file: File) => void
  required?: boolean
  hint?: string
}

function ImageDropTile({
  label,
  previewUrl,
  onPick,
  onClear,
  onPaste,
  onDropFile,
  required,
  hint,
}: ImageDropTileProps) {
  const [dragOver, setDragOver] = useState(false)
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer?.files?.[0]
    if (!file || !file.type.startsWith("image/")) return
    onDropFile?.(file)
  }
  return (
    <div
      onDragOver={(e) => {
        e.preventDefault()
        setDragOver(true)
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
      className={`rounded-md border border-dashed p-3 transition-colors ${
        dragOver ? "border-primary bg-primary/5" : "border-border bg-background/60"
      }`}
    >
      <div className="mb-2 flex items-center justify-between gap-2">
        <span className="text-xs font-medium text-muted-foreground">
          {label}{required ? " *" : ""}
        </span>
        <div className="flex items-center gap-1">
          {onPaste && (
            <button
              type="button"
              onClick={onPaste}
              className="inline-flex items-center gap-1 rounded border border-border px-2 py-0.5 text-[11px] text-muted-foreground hover:bg-muted"
              title="Pegar del portapapeles (Ctrl+V)"
            >
              <Copy size={11} />
              Paste
            </button>
          )}
          {previewUrl && (
            <button
              type="button"
              onClick={onClear}
              className="inline-flex items-center gap-1 rounded border border-border px-2 py-0.5 text-[11px] text-muted-foreground hover:bg-muted"
            >
              <X size={11} />
              Quitar
            </button>
          )}
        </div>
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
          <span>Click, arrastra aqui o Ctrl+V</span>
          {hint && <span className="text-[10px] text-muted-foreground/70">{hint}</span>}
        </button>
      )}
    </div>
  )
}
