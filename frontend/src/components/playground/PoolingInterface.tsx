import { useRef, useState } from "react"
import { toast } from "sonner"

import { playgroundFetch } from "@/lib/playgroundNode"
interface PoolingInterfaceProps {
  modelId: string
  scoreCapable: boolean
  rerankCapable: boolean
  classificationCapable: boolean
}

export function PoolingInterface({
  modelId,
  scoreCapable,
  rerankCapable,
  classificationCapable,
}: PoolingInterfaceProps) {
  const [input, setInput] = useState("oCabra maximiza compatibilidad real de vLLM.")
  const [scoreQuery, setScoreQuery] = useState("GPU scheduling for vLLM")
  const [scoreDocument, setScoreDocument] = useState("vLLM support for pooling models")
  const [rerankQuery, setRerankQuery] = useState("best vLLM retrieval backend")
  const [rerankDocuments, setRerankDocuments] = useState(
    "vLLM supports pooling models\nCross-encoders help rerank passages\nDiffusers serves image generation",
  )
  const [classificationInput, setClassificationInput] = useState(
    "This routing layer improves retrieval quality.",
  )
  const [poolingResult, setPoolingResult] = useState<string>("")
  const [scoreResult, setScoreResult] = useState<string>("")
  const [rerankResult, setRerankResult] = useState<string>("")
  const [classificationResult, setClassificationResult] = useState<string>("")
  const [runningPooling, setRunningPooling] = useState(false)
  const [runningScore, setRunningScore] = useState(false)
  const [runningRerank, setRunningRerank] = useState(false)
  const [runningClassification, setRunningClassification] = useState(false)
  const abortRef = useRef<AbortController | null>(null)
  const stopRequest = () => abortRef.current?.abort()
  const isAbort = (err: unknown) =>
    (abortRef.current?.signal.aborted ?? false) ||
    (err instanceof DOMException && err.name === "AbortError")

  const runPooling = async () => {
    if (!modelId) {
      toast.error("Selecciona un modelo")
      return
    }
    if (!input.trim()) return
    setRunningPooling(true)
    const controller = new AbortController()
    abortRef.current = controller
    try {
      const response = await playgroundFetch("/v1/pooling", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({
          model: modelId,
          input: input.trim(),
        }),
      })
      if (!response.ok) {
        const err = await response.json().catch(() => ({}))
        throw new Error(String(err?.error?.message ?? err?.detail ?? `HTTP ${response.status}`))
      }
      const payload = await response.json()
      setPoolingResult(JSON.stringify(payload, null, 2))
    } catch (err) {
      if (isAbort(err)) toast.info("Petición detenida")
      else toast.error(err instanceof Error ? err.message : "Error en pooling")
    } finally {
      abortRef.current = null
      setRunningPooling(false)
    }
  }

  const runScore = async () => {
    if (!modelId) {
      toast.error("Selecciona un modelo")
      return
    }
    setRunningScore(true)
    const controller = new AbortController()
    abortRef.current = controller
    try {
      const response = await playgroundFetch("/v1/score", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({
          model: modelId,
          queries: scoreQuery,
          documents: scoreDocument,
        }),
      })
      if (!response.ok) {
        const err = await response.json().catch(() => ({}))
        throw new Error(String(err?.error?.message ?? err?.detail ?? `HTTP ${response.status}`))
      }
      const payload = await response.json()
      setScoreResult(JSON.stringify(payload, null, 2))
    } catch (err) {
      if (isAbort(err)) toast.info("Petición detenida")
      else toast.error(err instanceof Error ? err.message : "Error en score")
    } finally {
      abortRef.current = null
      setRunningScore(false)
    }
  }

  const runRerank = async () => {
    if (!modelId) {
      toast.error("Selecciona un modelo")
      return
    }
    const documents = rerankDocuments
      .split("\n")
      .map((item) => item.trim())
      .filter(Boolean)
    if (!rerankQuery.trim() || documents.length === 0) return
    setRunningRerank(true)
    const controller = new AbortController()
    abortRef.current = controller
    try {
      const response = await playgroundFetch("/v1/rerank", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({
          model: modelId,
          query: rerankQuery.trim(),
          documents,
        }),
      })
      if (!response.ok) {
        const err = await response.json().catch(() => ({}))
        throw new Error(String(err?.error?.message ?? err?.detail ?? `HTTP ${response.status}`))
      }
      const payload = await response.json()
      setRerankResult(JSON.stringify(payload, null, 2))
    } catch (err) {
      if (isAbort(err)) toast.info("Petición detenida")
      else toast.error(err instanceof Error ? err.message : "Error en rerank")
    } finally {
      abortRef.current = null
      setRunningRerank(false)
    }
  }

  const runClassification = async () => {
    if (!modelId) {
      toast.error("Selecciona un modelo")
      return
    }
    if (!classificationInput.trim()) return
    setRunningClassification(true)
    const controller = new AbortController()
    abortRef.current = controller
    try {
      const response = await playgroundFetch("/v1/classify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({
          model: modelId,
          input: classificationInput.trim(),
        }),
      })
      if (!response.ok) {
        const err = await response.json().catch(() => ({}))
        throw new Error(String(err?.error?.message ?? err?.detail ?? `HTTP ${response.status}`))
      }
      const payload = await response.json()
      setClassificationResult(JSON.stringify(payload, null, 2))
    } catch (err) {
      if (isAbort(err)) toast.info("Petición detenida")
      else toast.error(err instanceof Error ? err.message : "Error en classification")
    } finally {
      abortRef.current = null
      setRunningClassification(false)
    }
  }

  return (
    <div className="space-y-4 rounded-lg border border-border bg-card p-4">
      <section className="space-y-3">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">Pooling</h2>
        <textarea
          value={input}
          onChange={(event) => setInput(event.target.value)}
          className="min-h-28 w-full rounded-md border border-border bg-background px-3 py-2"
        />
        <button
          type="button"
          onClick={() => void runPooling()}
          disabled={runningPooling || !modelId}
          className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground"
        >
          {runningPooling ? "Ejecutando..." : "Ejecutar pooling"}
        </button>
        {runningPooling && <StopButton onClick={stopRequest} />}
        {poolingResult && (
          <pre className="overflow-x-auto rounded-md border border-border bg-background/60 p-3 text-xs text-muted-foreground">
            {poolingResult}
          </pre>
        )}
      </section>

      <section className="space-y-3 border-t border-border pt-3">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">Score</h2>
        {!scoreCapable && (
          <p className="text-xs text-muted-foreground">
            El modelo no declara capacidad `score`; este bloque puede fallar aunque `pooling` funcione.
          </p>
        )}
        <p className="text-xs text-muted-foreground">
          oCabra usa `queries` y `documents` como contrato principal; mantiene compatibilidad con el alias legado `text_1`/`text_2`.
        </p>
        <input
          value={scoreQuery}
          onChange={(event) => setScoreQuery(event.target.value)}
          className="w-full rounded-md border border-border bg-background px-3 py-2"
          placeholder="query"
        />
        <input
          value={scoreDocument}
          onChange={(event) => setScoreDocument(event.target.value)}
          className="w-full rounded-md border border-border bg-background px-3 py-2"
          placeholder="document"
        />
        <button
          type="button"
          onClick={() => void runScore()}
          disabled={runningScore || !modelId}
          className="rounded-md border border-border px-4 py-2 text-sm hover:bg-muted"
        >
          {runningScore ? "Calculando..." : "Calcular score"}
        </button>
        {runningScore && <StopButton onClick={stopRequest} />}
        {scoreResult && (
          <pre className="overflow-x-auto rounded-md border border-border bg-background/60 p-3 text-xs text-muted-foreground">
            {scoreResult}
          </pre>
        )}
      </section>

      <section className="space-y-3 border-t border-border pt-3">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">Rerank</h2>
        {!rerankCapable && (
          <p className="text-xs text-muted-foreground">
            El modelo no declara capacidad `rerank`; este bloque depende de soporte real del worker.
          </p>
        )}
        <input
          value={rerankQuery}
          onChange={(event) => setRerankQuery(event.target.value)}
          className="w-full rounded-md border border-border bg-background px-3 py-2"
          placeholder="consulta"
        />
        <textarea
          value={rerankDocuments}
          onChange={(event) => setRerankDocuments(event.target.value)}
          className="min-h-28 w-full rounded-md border border-border bg-background px-3 py-2"
          placeholder="un documento por linea"
        />
        <button
          type="button"
          onClick={() => void runRerank()}
          disabled={runningRerank || !modelId}
          className="rounded-md border border-border px-4 py-2 text-sm hover:bg-muted"
        >
          {runningRerank ? "Reordenando..." : "Ejecutar rerank"}
        </button>
        {runningRerank && <StopButton onClick={stopRequest} />}
        {rerankResult && (
          <pre className="overflow-x-auto rounded-md border border-border bg-background/60 p-3 text-xs text-muted-foreground">
            {rerankResult}
          </pre>
        )}
      </section>

      <section className="space-y-3 border-t border-border pt-3">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">Classification</h2>
        {!classificationCapable && (
          <p className="text-xs text-muted-foreground">
            El modelo no declara capacidad `classification`; este bloque puede no estar disponible.
          </p>
        )}
        <textarea
          value={classificationInput}
          onChange={(event) => setClassificationInput(event.target.value)}
          className="min-h-24 w-full rounded-md border border-border bg-background px-3 py-2"
          placeholder="texto a clasificar"
        />
        <button
          type="button"
          onClick={() => void runClassification()}
          disabled={runningClassification || !modelId}
          className="rounded-md border border-border px-4 py-2 text-sm hover:bg-muted"
        >
          {runningClassification ? "Clasificando..." : "Ejecutar classification"}
        </button>
        {runningClassification && <StopButton onClick={stopRequest} />}
        {classificationResult && (
          <pre className="overflow-x-auto rounded-md border border-border bg-background/60 p-3 text-xs text-muted-foreground">
            {classificationResult}
          </pre>
        )}
      </section>
    </div>
  )
}

function StopButton({ onClick }: { onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="ml-2 rounded-md border border-rose-500/50 bg-rose-500/15 px-4 py-2 text-sm font-medium text-rose-100 hover:bg-rose-500/25"
    >
      Detener
    </button>
  )
}
