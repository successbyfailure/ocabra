import { useState } from "react"
import { toast } from "sonner"

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

  const runPooling = async () => {
    if (!modelId) {
      toast.error("Selecciona un modelo")
      return
    }
    if (!input.trim()) return
    setRunningPooling(true)
    try {
      const response = await fetch("/v1/pooling", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
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
      toast.error(err instanceof Error ? err.message : "Error en pooling")
    } finally {
      setRunningPooling(false)
    }
  }

  const runScore = async () => {
    if (!modelId) {
      toast.error("Selecciona un modelo")
      return
    }
    setRunningScore(true)
    try {
      const response = await fetch("/v1/score", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
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
      toast.error(err instanceof Error ? err.message : "Error en score")
    } finally {
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
    try {
      const response = await fetch("/v1/rerank", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
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
      toast.error(err instanceof Error ? err.message : "Error en rerank")
    } finally {
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
    try {
      const response = await fetch("/v1/classify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
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
      toast.error(err instanceof Error ? err.message : "Error en classification")
    } finally {
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
        {classificationResult && (
          <pre className="overflow-x-auto rounded-md border border-border bg-background/60 p-3 text-xs text-muted-foreground">
            {classificationResult}
          </pre>
        )}
      </section>
    </div>
  )
}
