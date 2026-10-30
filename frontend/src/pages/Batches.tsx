import { useEffect, useMemo, useRef, useState } from "react"
import {
  AlertCircle,
  Ban,
  Download,
  FileAudio,
  ListChecks,
  Loader2,
  Plus,
  RefreshCw,
  Upload,
  X,
} from "lucide-react"
import { toast } from "sonner"
import { api } from "@/api/client"
import type { BatchEndpoint, BatchStatus, OpenAIBatchJob } from "@/types"

const ENDPOINTS: { value: BatchEndpoint; label: string }[] = [
  { value: "/v1/chat/completions", label: "Chat completions" },
  { value: "/v1/embeddings", label: "Embeddings" },
  { value: "/v1/completions", label: "Completions" },
  { value: "/v1/images/generations", label: "Images generations" },
  { value: "/v1/audio/speech", label: "Audio speech (TTS)" },
  { value: "/v1/audio/transcriptions", label: "Audio transcriptions (extensión oCabra)" },
]

const CANCELLABLE: BatchStatus[] = ["validating", "in_progress", "finalizing"]

function statusPill(status: BatchStatus) {
  const map: Record<BatchStatus, string> = {
    validating: "border-sky-500/40 bg-sky-500/10 text-sky-600 dark:text-sky-400",
    in_progress: "border-sky-500/40 bg-sky-500/10 text-sky-600 dark:text-sky-400",
    finalizing: "border-amber-500/40 bg-amber-500/10 text-amber-600 dark:text-amber-400",
    completed: "border-emerald-500/40 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400",
    failed: "border-destructive/40 bg-destructive/10 text-destructive",
    expired: "border-border bg-muted/40 text-muted-foreground",
    cancelling: "border-amber-500/40 bg-amber-500/10 text-amber-600 dark:text-amber-400",
    cancelled: "border-border bg-muted/40 text-muted-foreground",
  }
  return (
    <span className={`inline-flex items-center rounded border px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide ${map[status]}`}>
      {status}
    </span>
  )
}

function fmtTs(ts: number | null): string {
  if (!ts) return "—"
  return new Date(ts * 1000).toLocaleString()
}

export function Batches() {
  const [batches, setBatches] = useState<OpenAIBatchJob[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showCreate, setShowCreate] = useState(false)
  const [detailBatch, setDetailBatch] = useState<OpenAIBatchJob | null>(null)
  const [busyId, setBusyId] = useState<string | null>(null)

  const refresh = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await api.batches.list({ limit: 50 })
      setBatches(res.data)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void refresh()
    const handle = window.setInterval(() => void refresh(), 10_000)
    return () => window.clearInterval(handle)
  }, [])

  const cancelBatch = async (batchId: string) => {
    setBusyId(batchId)
    try {
      await api.batches.cancel(batchId)
      toast.success("Cancelación solicitada")
      await refresh()
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Error cancelando el lote")
    } finally {
      setBusyId(null)
    }
  }

  return (
    <div className="space-y-4 p-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold">Lotes</h1>
          <p className="text-sm text-muted-foreground">
            Peticiones asíncronas vía <code>/v1/batches</code>. Ceden slots a sesiones
            interactivas automáticamente; ver <a href="/settings" className="underline">prioridad</a>{" "}
            para los parámetros de preempción.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setShowCreate(true)}
            className="inline-flex items-center gap-1 rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground hover:bg-primary/90"
          >
            <Plus size={14} />
            Nuevo lote
          </button>
          <button
            type="button"
            onClick={() => void refresh()}
            disabled={loading}
            className="inline-flex items-center gap-1 rounded-md border border-border px-3 py-1.5 text-sm hover:bg-muted disabled:opacity-50"
          >
            <RefreshCw size={14} className={loading ? "animate-spin" : ""} />
            Actualizar
          </button>
        </div>
      </div>

      {error && (
        <div className="flex items-start gap-2 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
          <AlertCircle size={16} className="mt-0.5" />
          {error}
        </div>
      )}

      {batches.length === 0 && !loading && !error && (
        <div className="rounded-md border border-border bg-muted/20 px-4 py-10 text-center text-sm text-muted-foreground">
          <ListChecks size={20} className="mx-auto mb-2 opacity-60" />
          No hay lotes todavía. Usa <strong>Nuevo lote</strong> para crear uno.
        </div>
      )}

      <div className="grid gap-3">
        {batches.map((batch) => (
          <div key={batch.id} className="rounded-lg border border-border bg-card p-4">
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <code className="font-mono text-sm font-semibold">{batch.id}</code>
                  {statusPill(batch.status)}
                  <span className="rounded border border-border/60 bg-muted/40 px-1.5 py-0.5 text-[10px] text-muted-foreground">
                    {batch.endpoint}
                  </span>
                </div>
                <div className="mt-1 text-xs text-muted-foreground">
                  creado {fmtTs(batch.created_at)} · expira {fmtTs(batch.expires_at)}
                </div>
                <div className="mt-1 text-xs">
                  {batch.request_counts.completed}/{batch.request_counts.total} completadas
                  {batch.request_counts.failed > 0 && (
                    <span className="text-destructive"> · {batch.request_counts.failed} fallidas</span>
                  )}
                </div>
              </div>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => setDetailBatch(batch)}
                  className="rounded-md border border-border px-2.5 py-1 text-xs hover:bg-muted"
                >
                  Ver detalle
                </button>
                {CANCELLABLE.includes(batch.status) && (
                  <button
                    type="button"
                    disabled={busyId === batch.id}
                    onClick={() => void cancelBatch(batch.id)}
                    className="inline-flex items-center gap-1 rounded-md border border-destructive/40 px-2.5 py-1 text-xs text-destructive hover:bg-destructive/10 disabled:opacity-50"
                  >
                    {busyId === batch.id ? <Loader2 size={12} className="animate-spin" /> : <Ban size={12} />}
                    Cancelar
                  </button>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>

      {showCreate && (
        <CreateBatchModal
          onClose={() => setShowCreate(false)}
          onCreated={async () => {
            setShowCreate(false)
            await refresh()
          }}
        />
      )}

      {detailBatch && (
        <BatchDetailModal batch={detailBatch} onClose={() => setDetailBatch(null)} />
      )}
    </div>
  )
}

interface CreateBatchModalProps {
  onClose: () => void
  onCreated: () => Promise<void>
}

function CreateBatchModal({ onClose, onCreated }: CreateBatchModalProps) {
  const [endpoint, setEndpoint] = useState<BatchEndpoint>("/v1/chat/completions")
  const [file, setFile] = useState<File | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [audioFile, setAudioFile] = useState<File | null>(null)
  const [audioUploading, setAudioUploading] = useState(false)
  const [uploadedFileIds, setUploadedFileIds] = useState<{ filename: string; fileId: string }[]>([])
  const fileInputRef = useRef<HTMLInputElement | null>(null)

  const handleSubmit = async () => {
    if (!file) {
      toast.error("Selecciona un fichero JSONL")
      return
    }
    setSubmitting(true)
    try {
      const uploaded = await api.batches.uploadFile(file, "batch")
      const batch = await api.batches.create({ inputFileId: uploaded.id, endpoint })
      toast.success(`Lote ${batch.id} creado (${batch.request_counts.total} peticiones)`)
      await onCreated()
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Error creando el lote")
    } finally {
      setSubmitting(false)
    }
  }

  const uploadAudio = async () => {
    if (!audioFile) return
    setAudioUploading(true)
    try {
      const uploaded = await api.batches.uploadFile(audioFile, "batch_input_audio")
      setUploadedFileIds((prev) => [...prev, { filename: audioFile.name, fileId: uploaded.id }])
      setAudioFile(null)
      toast.success(`Subido como ${uploaded.id}`)
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Error subiendo audio")
    } finally {
      setAudioUploading(false)
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4" onClick={onClose}>
      <div
        className="w-full max-w-lg rounded-lg border border-border bg-card p-5 shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-4 flex items-start justify-between gap-3">
          <div>
            <h2 className="text-lg font-semibold">Nuevo lote</h2>
            <p className="mt-1 text-xs text-muted-foreground">
              Un fichero JSONL con una petición por línea (<code>{"{custom_id, method, url, body}"}</code>),
              estilo OpenAI Batches.
            </p>
          </div>
          <button type="button" onClick={onClose} className="rounded p-1 text-muted-foreground hover:bg-muted">
            <X size={16} />
          </button>
        </div>

        <div className="space-y-3">
          <div>
            <label className="mb-1 block text-xs font-medium">Endpoint</label>
            <select
              value={endpoint}
              onChange={(e) => setEndpoint(e.target.value as BatchEndpoint)}
              className="w-full rounded-md border border-input bg-background px-2 py-1.5 text-sm"
            >
              {ENDPOINTS.map((e) => (
                <option key={e.value} value={e.value}>{e.label}</option>
              ))}
            </select>
          </div>

          {endpoint === "/v1/audio/transcriptions" && (
            <div className="rounded-md border border-amber-500/30 bg-amber-500/5 p-3">
              <div className="mb-2 flex items-center gap-1.5 text-xs font-medium text-amber-600 dark:text-amber-400">
                <FileAudio size={14} />
                Extensión oCabra: sube el audio primero, referencia el <code>file_id</code> en
                <code className="ml-1">body.file_id</code> de tu línea JSONL
              </div>
              <div className="flex items-center gap-2">
                <input
                  type="file"
                  accept="audio/*"
                  onChange={(e) => setAudioFile(e.target.files?.[0] ?? null)}
                  className="flex-1 text-xs"
                />
                <button
                  type="button"
                  disabled={!audioFile || audioUploading}
                  onClick={() => void uploadAudio()}
                  className="inline-flex items-center gap-1 rounded-md border border-border bg-background px-2 py-1 text-xs hover:bg-muted disabled:opacity-50"
                >
                  {audioUploading ? <Loader2 size={12} className="animate-spin" /> : <Upload size={12} />}
                  Subir
                </button>
              </div>
              {uploadedFileIds.length > 0 && (
                <ul className="mt-2 space-y-1">
                  {uploadedFileIds.map((f) => (
                    <li key={f.fileId} className="flex items-center justify-between gap-2 text-[10px]">
                      <span className="truncate text-muted-foreground">{f.filename}</span>
                      <code
                        className="cursor-pointer truncate rounded bg-muted px-1 py-0.5 font-mono"
                        title="Clic para copiar"
                        onClick={() => {
                          void navigator.clipboard.writeText(f.fileId)
                          toast.success("file_id copiado")
                        }}
                      >
                        {f.fileId}
                      </code>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )}

          <div>
            <label className="mb-1 block text-xs font-medium">Fichero JSONL</label>
            <input
              ref={fileInputRef}
              type="file"
              accept=".jsonl,application/json,text/plain"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              className="w-full text-xs"
            />
          </div>
        </div>

        <div className="mt-5 flex items-center justify-end gap-2">
          <button type="button" onClick={onClose} className="rounded-md border border-border px-3 py-1.5 text-sm hover:bg-muted">
            Cancelar
          </button>
          <button
            type="button"
            disabled={submitting || !file}
            onClick={() => void handleSubmit()}
            className="inline-flex items-center gap-1 rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
          >
            {submitting ? <Loader2 size={14} className="animate-spin" /> : <Plus size={14} />}
            Crear lote
          </button>
        </div>
      </div>
    </div>
  )
}

interface BatchLineResult {
  id: string
  custom_id: string | null
  response: { status_code: number; request_id: string; body: unknown } | null
  error: { code?: string; message: string } | null
}

function isAudioArtifact(body: unknown): body is { file_id: string; content_type: string } {
  return (
    typeof body === "object" &&
    body !== null &&
    "file_id" in body &&
    "content_type" in body &&
    String((body as Record<string, unknown>).content_type).startsWith("audio/")
  )
}

function imagesFromBody(body: unknown): string[] {
  if (typeof body !== "object" || body === null || !("data" in body)) return []
  const data = (body as Record<string, unknown>).data
  if (!Array.isArray(data)) return []
  return data
    .map((item) => (typeof item === "object" && item !== null ? (item as Record<string, unknown>).b64_json : null))
    .filter((v): v is string => typeof v === "string")
}

function BatchDetailModal({ batch, onClose }: { batch: OpenAIBatchJob; onClose: () => void }) {
  const [outputLines, setOutputLines] = useState<BatchLineResult[] | null>(null)
  const [errorLines, setErrorLines] = useState<BatchLineResult[] | null>(null)
  const [loadingOutput, setLoadingOutput] = useState(false)
  const [loadError, setLoadError] = useState<string | null>(null)

  const loadFile = async (fileId: string): Promise<BatchLineResult[]> => {
    const res = await fetch(api.batches.fileContentUrl(fileId))
    if (!res.ok) throw new Error(`No se pudo descargar ${fileId}`)
    const text = await res.text()
    return text
      .split("\n")
      .map((l) => l.trim())
      .filter(Boolean)
      .map((l) => JSON.parse(l) as BatchLineResult)
  }

  useEffect(() => {
    let cancelled = false
    const run = async () => {
      setLoadingOutput(true)
      setLoadError(null)
      try {
        if (batch.output_file_id) {
          const lines = await loadFile(batch.output_file_id)
          if (!cancelled) setOutputLines(lines)
        }
        if (batch.error_file_id) {
          const lines = await loadFile(batch.error_file_id)
          if (!cancelled) setErrorLines(lines)
        }
      } catch (e) {
        if (!cancelled) setLoadError(e instanceof Error ? e.message : String(e))
      } finally {
        if (!cancelled) setLoadingOutput(false)
      }
    }
    void run()
    return () => {
      cancelled = true
    }
  }, [batch.output_file_id, batch.error_file_id])

  const results = useMemo(() => outputLines ?? [], [outputLines])

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4" onClick={onClose}>
      <div
        className="flex max-h-[85vh] w-full max-w-3xl flex-col rounded-lg border border-border bg-card p-5 shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-3 flex items-start justify-between gap-3">
          <div>
            <h2 className="flex items-center gap-2 text-lg font-semibold">
              <code className="font-mono text-base">{batch.id}</code>
              {statusPill(batch.status)}
            </h2>
            <p className="mt-1 text-xs text-muted-foreground">
              {batch.endpoint} · {batch.request_counts.completed}/{batch.request_counts.total} completadas
              {batch.request_counts.failed > 0 && `, ${batch.request_counts.failed} fallidas`}
            </p>
          </div>
          <button type="button" onClick={onClose} className="rounded p-1 text-muted-foreground hover:bg-muted">
            <X size={16} />
          </button>
        </div>

        <div className="mb-3 flex items-center gap-2">
          {batch.output_file_id && (
            <a
              href={api.batches.fileContentUrl(batch.output_file_id)}
              className="inline-flex items-center gap-1 rounded-md border border-border px-2 py-1 text-xs hover:bg-muted"
              download
            >
              <Download size={12} />
              output.jsonl
            </a>
          )}
          {batch.error_file_id && (
            <a
              href={api.batches.fileContentUrl(batch.error_file_id)}
              className="inline-flex items-center gap-1 rounded-md border border-destructive/40 px-2 py-1 text-xs text-destructive hover:bg-destructive/10"
              download
            >
              <Download size={12} />
              errors.jsonl
            </a>
          )}
        </div>

        <div className="flex-1 overflow-y-auto space-y-2">
          {loadingOutput && (
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Loader2 size={14} className="animate-spin" />
              Cargando resultados…
            </div>
          )}
          {loadError && (
            <div className="rounded-md border border-destructive/40 bg-destructive/10 px-2 py-1.5 text-xs text-destructive">
              {loadError}
            </div>
          )}
          {!loadingOutput && !batch.output_file_id && !batch.error_file_id && (
            <p className="text-sm text-muted-foreground">Aún no hay resultados (lote en curso o sin salida).</p>
          )}

          {results.map((line) => {
            const body = line.response?.body
            const audio = isAudioArtifact(body) ? body : null
            const images = imagesFromBody(body)
            return (
              <div key={line.id} className="rounded-md border border-border/60 bg-muted/10 p-2">
                <div className="mb-1 flex items-center gap-2 text-[10px] text-muted-foreground">
                  <code>{line.custom_id ?? line.id}</code>
                  {line.response && (
                    <span className="rounded border border-border/60 px-1 text-[10px]">
                      {line.response.status_code}
                    </span>
                  )}
                  {line.error && (
                    <span className="rounded border border-destructive/40 px-1 text-[10px] text-destructive">
                      error
                    </span>
                  )}
                </div>
                {line.error ? (
                  <p className="text-xs text-destructive">{line.error.message}</p>
                ) : audio ? (
                  <audio controls className="w-full" src={api.batches.fileContentUrl(audio.file_id)} />
                ) : images.length > 0 ? (
                  <div className="flex flex-wrap gap-2">
                    {images.map((b64, i) => (
                      <img
                        key={i}
                        src={`data:image/png;base64,${b64}`}
                        alt={`${line.custom_id ?? line.id} #${i}`}
                        className="h-24 w-24 rounded object-cover"
                      />
                    ))}
                  </div>
                ) : (
                  <pre className="max-h-40 overflow-auto text-[10px]">{JSON.stringify(body, null, 2)}</pre>
                )}
              </div>
            )
          })}

          {errorLines && errorLines.length > 0 && (
            <div className="mt-2 border-t border-border/60 pt-2">
              <p className="mb-1 text-xs font-medium text-destructive">Errores ({errorLines.length})</p>
              {errorLines.map((line) => (
                <div key={line.id} className="rounded-md border border-destructive/30 bg-destructive/5 p-2 text-xs">
                  <code className="text-[10px] text-muted-foreground">{line.custom_id ?? line.id}</code>
                  <p className="text-destructive">{line.error?.message}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
