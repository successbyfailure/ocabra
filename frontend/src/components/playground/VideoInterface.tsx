import { useCallback, useEffect, useRef, useState } from "react"
import { Download, RotateCcw, Shuffle, Trash2 } from "lucide-react"
import { toast } from "sonner"
import { ImageDropTile } from "@/components/playground/ImageInterface"
import { fileFromClipboard, fileFromUrl } from "@/lib/imageFiles"
import { playgroundFetch } from "@/lib/playgroundNode"

interface InputImage {
  file: File
  preview: string
}

/** Imagen enviada desde otra pestaña (p. ej. «Animar» en Imagen). */
export interface VideoDraft {
  url: string
  prompt: string
}

/** Objeto ``video`` de /v1/videos (API de vídeo estilo OpenAI). */
export interface VideoJob {
  id: string
  model: string
  status: "queued" | "in_progress" | "completed" | "failed"
  progress: number
  created_at: number
  completed_at: number | null
  expires_at: number | null
  seconds: string
  size: string
  seed: number | null
  error: { code?: string; message?: string } | null
  queue_position?: number
}

interface SizePreset {
  key: string
  label: string
  ratio: string
  width: number
  height: number
}

// LTX-2.5 genera a media resolución y escala x2: el servidor redondea a
// múltiplos de 64. 1280x704 es el tamaño de la plantilla oficial y el mayor
// que cabe en 24 GB; los demás rondan el mismo número de píxeles o menos.
const SIZE_PRESETS: SizePreset[] = [
  { key: "16_9", label: "Ancha", ratio: "16:9", width: 1280, height: 704 },
  { key: "16_9_s", label: "Ancha rápida", ratio: "16:9", width: 960, height: 512 },
  { key: "9_16", label: "Vertical", ratio: "9:16", width: 704, height: 1280 },
  { key: "1_1", label: "Cuadrado", ratio: "1:1", width: 960, height: 960 },
  { key: "4_3", label: "Foto", ratio: "4:3", width: 1088, height: 832 },
]

const POLL_MS = 2500
const HISTORY_LIMIT = 12

const STATUS_LABEL: Record<VideoJob["status"], string> = {
  queued: "En cola",
  in_progress: "Generando",
  completed: "Listo",
  failed: "Error",
}

async function errorMessage(response: Response): Promise<string> {
  const err = await response.json().catch(() => ({}))
  return String(
    err?.detail?.error?.message ?? err?.error?.message ?? err?.detail ?? `HTTP ${response.status}`,
  )
}

function contentUrl(id: string, variant: "video" | "thumbnail" = "video"): string {
  return `/v1/videos/${encodeURIComponent(id)}/content${variant === "thumbnail" ? "?variant=thumbnail" : ""}`
}

function elapsed(job: VideoJob, now: number): string {
  const end = job.completed_at ?? now / 1000
  const s = Math.max(0, Math.round(end - job.created_at))
  return s >= 60 ? `${Math.floor(s / 60)}m ${s % 60}s` : `${s}s`
}

// "auto": sin ``size``, el servidor toma la proporción de la imagen inicial.
const AUTO_SIZE = "auto"

export function VideoInterface({
  modelId,
  draft = null,
  onDraftConsumed,
}: {
  modelId: string
  draft?: VideoDraft | null
  onDraftConsumed?: () => void
}) {
  const [prompt, setPrompt] = useState("")
  const [negativePrompt, setNegativePrompt] = useState("")
  const [seconds, setSeconds] = useState(5)
  const [sizeKey, setSizeKey] = useState("16_9")
  const [seed, setSeed] = useState<string>("")
  const [startImage, setStartImage] = useState<InputImage | null>(null)
  const [endImage, setEndImage] = useState<InputImage | null>(null)
  const startInput = useRef<HTMLInputElement | null>(null)
  const endInput = useRef<HTMLInputElement | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [jobs, setJobs] = useState<VideoJob[]>([])
  const [prompts, setPrompts] = useState<Record<string, string>>({})
  const [now, setNow] = useState(() => Date.now())
  const mounted = useRef(true)

  const size = SIZE_PRESETS.find((p) => p.key === sizeKey) ?? SIZE_PRESETS[0]!
  const autoSize = sizeKey === AUTO_SIZE && startImage !== null

  const setImage = useCallback(
    (which: "start" | "end", file: File | null) => {
      const setter = which === "start" ? setStartImage : setEndImage
      setter((prev) => {
        if (prev) URL.revokeObjectURL(prev.preview)
        return file ? { file, preview: URL.createObjectURL(file) } : null
      })
      // Con imagen inicial, el formato por defecto pasa a ser el de la imagen.
      if (which === "start") setSizeKey((key) => (file ? AUTO_SIZE : key === AUTO_SIZE ? "16_9" : key))
    },
    [],
  )

  const pasteInto = async (which: "start" | "end") => {
    try {
      const file = await fileFromClipboard()
      if (file) setImage(which, file)
      else toast.error("No hay ninguna imagen en el portapapeles")
    } catch {
      toast.error("El navegador no deja leer el portapapeles")
    }
  }

  // «Animar» desde la pestaña Imagen: la imagen llega como primer fotograma.
  // El callback va por ref para que un padre que lo recrea en cada render no
  // relance la carga.
  const consumed = useRef(onDraftConsumed)
  useEffect(() => {
    consumed.current = onDraftConsumed
  }, [onDraftConsumed])
  useEffect(() => {
    if (!draft) return
    let cancelled = false
    void (async () => {
      try {
        const file = await fileFromUrl(draft.url, "start.png")
        if (cancelled) return
        setImage("start", file)
        setPrompt((current) => current || draft.prompt)
      } catch {
        toast.error("No se pudo cargar la imagen")
      } finally {
        consumed.current?.()
      }
    })()
    return () => {
      cancelled = true
    }
  }, [draft, setImage])
  const pending = jobs.filter((j) => j.status === "queued" || j.status === "in_progress")

  // Historial: los vídeos se guardan 24 h en el servidor, así que al volver a
  // la pestaña se ven los recientes aunque se generasen en otra sesión.
  const loadHistory = useCallback(async () => {
    try {
      const response = await playgroundFetch(`/v1/videos?limit=${HISTORY_LIMIT}`)
      if (!response.ok) return
      const payload = await response.json()
      if (mounted.current && Array.isArray(payload?.data)) setJobs(payload.data as VideoJob[])
    } catch {
      // sin historial: la pestaña sigue funcionando
    }
  }, [])

  useEffect(() => {
    mounted.current = true
    void loadHistory()
    return () => {
      mounted.current = false
    }
  }, [loadHistory])

  // Sondeo de los trabajos pendientes.
  const pendingIds = pending.map((j) => j.id).join(",")
  useEffect(() => {
    if (!pendingIds) return
    const timer = window.setInterval(async () => {
      setNow(Date.now())
      const ids = pendingIds.split(",")
      const updates = await Promise.all(
        ids.map(async (id) => {
          try {
            const r = await playgroundFetch(`/v1/videos/${encodeURIComponent(id)}`)
            return r.ok ? ((await r.json()) as VideoJob) : null
          } catch {
            return null
          }
        }),
      )
      if (!mounted.current) return
      setJobs((prev) =>
        prev.map((job) => {
          const next = updates.find((u) => u?.id === job.id)
          if (!next) return job
          if (job.status !== next.status && next.status === "completed") {
            toast.success(`Vídeo listo en ${elapsed(next, Date.now())}`)
          } else if (job.status !== next.status && next.status === "failed") {
            toast.error(next.error?.message ?? "La generación falló")
          }
          return next
        }),
      )
    }, POLL_MS)
    return () => window.clearInterval(timer)
  }, [pendingIds])

  const submit = async () => {
    if (!modelId) {
      toast.error("Selecciona un modelo de vídeo")
      return
    }
    if (!prompt.trim()) {
      toast.error("Describe la escena (y el sonido) en el prompt")
      return
    }
    if (endImage && !startImage) {
      toast.error("Para usar imagen final hace falta también la inicial")
      return
    }
    setSubmitting(true)
    try {
      const fields: Record<string, string> = {
        model: modelId,
        prompt: prompt.trim(),
        seconds: String(seconds),
      }
      if (!autoSize) fields.size = `${size.width}x${size.height}`
      if (negativePrompt.trim()) fields.negative_prompt = negativePrompt.trim()
      if (seed.trim() !== "" && Number.isFinite(Number(seed))) fields.seed = String(Number(seed))
      let init: RequestInit
      if (startImage) {
        // Con imágenes, multipart: es como lo envía el SDK de OpenAI.
        const fd = new FormData()
        for (const [key, value] of Object.entries(fields)) fd.append(key, value)
        fd.append("input_reference", startImage.file, startImage.file.name)
        if (endImage) fd.append("last_frame", endImage.file, endImage.file.name)
        init = { method: "POST", body: fd }
      } else {
        const body: Record<string, unknown> = { ...fields }
        if (fields.seed) body.seed = Number(fields.seed)
        init = {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        }
      }
      const response = await playgroundFetch("/v1/videos", init)
      if (!response.ok) throw new Error(await errorMessage(response))
      const job = (await response.json()) as VideoJob
      setPrompts((prev) => ({ ...prev, [job.id]: prompt.trim() }))
      setJobs((prev) => [job, ...prev])
      toast.success("Vídeo en cola")
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "No se pudo encolar el vídeo")
    } finally {
      setSubmitting(false)
    }
  }

  const remove = async (job: VideoJob) => {
    try {
      const response = await playgroundFetch(`/v1/videos/${encodeURIComponent(job.id)}`, {
        method: "DELETE",
      })
      if (!response.ok) throw new Error(await errorMessage(response))
      setJobs((prev) => prev.filter((j) => j.id !== job.id))
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "No se pudo borrar")
    }
  }

  return (
    <div className="h-full space-y-4 overflow-y-auto rounded-lg border border-border bg-card p-4">
      <div className="grid gap-2">
        <label className="text-sm text-muted-foreground">
          Prompt
          <textarea
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            placeholder="Plano, sujeto, acción, luz, movimiento de cámara… y lo que se oye: el modelo genera también el sonido."
            className="mt-1 min-h-24 w-full rounded-md border border-border bg-background px-3 py-2"
          />
        </label>
        <label className="text-sm text-muted-foreground">
          Negative prompt
          <textarea
            value={negativePrompt}
            onChange={(event) => setNegativePrompt(event.target.value)}
            placeholder="Opcional"
            className="mt-1 min-h-12 w-full rounded-md border border-border bg-background px-3 py-2"
          />
        </label>
      </div>

      <div className="grid gap-3 md:grid-cols-2">
        <ImageDropTile
          label="Imagen inicial (opcional)"
          previewUrl={startImage?.preview ?? null}
          onPick={() => startInput.current?.click()}
          onClear={() => setImage("start", null)}
          onPaste={() => void pasteInto("start")}
          onDropFile={(f) => setImage("start", f)}
          hint="Primer fotograma: el vídeo la anima siguiendo el prompt"
        />
        <ImageDropTile
          label="Imagen final (opcional)"
          previewUrl={endImage?.preview ?? null}
          onPick={() => endInput.current?.click()}
          onClear={() => setImage("end", null)}
          onPaste={() => void pasteInto("end")}
          onDropFile={(f) => setImage("end", f)}
          hint="Último fotograma: el vídeo va de la inicial a esta"
        />
        {(["start", "end"] as const).map((which) => (
          <input
            key={which}
            ref={which === "start" ? startInput : endInput}
            type="file"
            accept="image/png,image/jpeg,image/webp"
            className="hidden"
            data-testid={`video-${which}-input`}
            onChange={(e) => {
              const f = e.target.files?.[0]
              if (f) setImage(which, f)
              e.target.value = ""
            }}
          />
        ))}
      </div>

      <div className="flex flex-wrap items-center gap-1.5">
        {startImage && (
          <button
            type="button"
            onClick={() => setSizeKey(AUTO_SIZE)}
            className={`rounded-md border px-2.5 py-1 text-xs transition-colors ${
              autoSize
                ? "border-primary bg-primary/10 text-primary"
                : "border-border text-muted-foreground hover:bg-muted"
            }`}
            title="La proporción de la imagen inicial"
          >
            Auto <span className="text-[10px] opacity-70">imagen</span>
          </button>
        )}
        {SIZE_PRESETS.map((p) => (
          <button
            key={p.key}
            type="button"
            onClick={() => setSizeKey(p.key)}
            className={`rounded-md border px-2.5 py-1 text-xs transition-colors ${
              sizeKey === p.key
                ? "border-primary bg-primary/10 text-primary"
                : "border-border text-muted-foreground hover:bg-muted"
            }`}
            title={`${p.width}×${p.height}`}
          >
            {p.label} <span className="text-[10px] opacity-70">{p.ratio}</span>
          </button>
        ))}
        <span className="ml-1 text-xs text-muted-foreground">
          {autoSize ? "según la imagen" : `${size.width}×${size.height}`}
        </span>
      </div>

      <div className="flex flex-wrap items-end gap-3">
        <label className="w-56 text-sm text-muted-foreground">
          Duración {seconds} s
          <input
            type="range"
            min={1}
            max={10}
            value={seconds}
            onChange={(event) => setSeconds(Number(event.target.value))}
            className="mt-2 w-full"
          />
        </label>
        <label className="text-sm text-muted-foreground">
          Seed
          <input
            type="number"
            value={seed}
            placeholder="aleatoria"
            onChange={(event) => setSeed(event.target.value)}
            className="mt-1 w-36 rounded-md border border-border bg-background px-3 py-2"
          />
        </label>
        <button
          type="button"
          onClick={() => setSeed(String(Math.floor(Math.random() * 2 ** 31)))}
          className="inline-flex items-center gap-1 rounded-md border border-border px-3 py-2 text-sm hover:bg-muted"
        >
          <Shuffle size={14} />
          Random
        </button>
        <button
          type="button"
          onClick={() => void submit()}
          disabled={submitting || !modelId}
          className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground disabled:opacity-50"
        >
          {submitting
            ? "Enviando..."
            : endImage
              ? "Generar transición"
              : startImage
                ? "Animar imagen"
                : "Generar vídeo"}
        </button>
      </div>
      <p className="text-xs text-muted-foreground">
        Se genera un vídeo a la vez y ocupa la GPU grande entera: ~65 s para 5 s a 1280×704 con el
        modelo cargado, ~1,5-2 min si hay que cargarlo. Los vídeos se borran a las 24 h.
      </p>

      <div className="grid gap-3 md:grid-cols-2">
        {jobs.map((job) => (
          <VideoCard
            key={job.id}
            job={job}
            prompt={prompts[job.id]}
            now={now}
            onReuseSeed={(s) => setSeed(String(s))}
            onDelete={() => void remove(job)}
          />
        ))}
      </div>
    </div>
  )
}

function VideoCard({
  job,
  prompt,
  now,
  onReuseSeed,
  onDelete,
}: {
  job: VideoJob
  prompt?: string
  now: number
  onReuseSeed: (seed: number) => void
  onDelete: () => void
}) {
  const busy = job.status === "queued" || job.status === "in_progress"
  const [w, h] = job.size.split("x").map(Number)
  return (
    <figure className="rounded-md border border-border bg-background/60 p-2" data-testid="video-card">
      {job.status === "completed" ? (
        <video
          src={contentUrl(job.id)}
          poster={contentUrl(job.id, "thumbnail")}
          controls
          preload="none"
          className="w-full rounded bg-black"
        />
      ) : (
        <div
          className="flex w-full flex-col items-center justify-center gap-2 rounded bg-muted/40 p-4 text-sm text-muted-foreground"
          style={{ aspectRatio: w && h ? `${w} / ${h}` : "16 / 9" }}
        >
          {busy ? (
            <>
              <span>
                {job.status === "queued" && job.queue_position
                  ? `En cola (${job.queue_position - 1} delante)`
                  : STATUS_LABEL[job.status]}
                {" · "}
                {elapsed(job, now)}
              </span>
              <div className="h-2 w-3/4 overflow-hidden rounded-full bg-border">
                <div
                  className="h-full bg-primary transition-all"
                  style={{ width: `${Math.max(2, job.progress)}%` }}
                  role="progressbar"
                  aria-valuenow={job.progress}
                />
              </div>
              <span className="text-xs">{job.progress}%</span>
            </>
          ) : (
            <span className="text-center text-destructive">
              {job.error?.message ?? "La generación falló"}
            </span>
          )}
        </div>
      )}
      <figcaption className="mt-2 space-y-2 text-xs text-muted-foreground">
        {prompt && (
          <div className="line-clamp-2 break-words" title={prompt}>
            {prompt}
          </div>
        )}
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-muted-foreground/80">
          <span>{STATUS_LABEL[job.status]}</span>
          <span>{job.size}</span>
          <span>{job.seconds} s</span>
          {job.seed !== null && <span>seed {job.seed}</span>}
          {job.status === "completed" && <span>{elapsed(job, now)}</span>}
          <span>{job.model}</span>
        </div>
        {!busy && (
          <div className="flex flex-wrap gap-1">
            {job.status === "completed" && (
              <a
                href={contentUrl(job.id)}
                download={`${job.id}.mp4`}
                className="inline-flex items-center gap-1 rounded-md border border-border px-2 py-1 text-xs hover:bg-muted"
              >
                <Download size={12} />
                Descargar
              </a>
            )}
            {job.seed !== null && (
              <button
                type="button"
                onClick={() => onReuseSeed(job.seed!)}
                className="inline-flex items-center gap-1 rounded-md border border-border px-2 py-1 text-xs hover:bg-muted"
                title="Reutilizar seed"
              >
                <RotateCcw size={12} />
                Seed
              </button>
            )}
            <button
              type="button"
              onClick={onDelete}
              className="inline-flex items-center gap-1 rounded-md border border-border px-2 py-1 text-xs hover:bg-muted"
            >
              <Trash2 size={12} />
              Borrar
            </button>
          </div>
        )}
      </figcaption>
    </figure>
  )
}
