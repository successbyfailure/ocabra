import { useEffect, useMemo, useState } from "react"
import { AlertTriangle, MessageSquarePlus, SlidersHorizontal } from "lucide-react"
import { Link } from "react-router-dom"
import { toast } from "sonner"
import { api } from "@/api/client"
import { AudioInterface } from "@/components/playground/AudioInterface"
import { ChatInterface } from "@/components/playground/ChatInterface"
import { ImageInterface } from "@/components/playground/ImageInterface"
import { ModelSelector } from "@/components/playground/ModelSelector"
import { ParamsPanel, type PlaygroundParams } from "@/components/playground/ParamsPanel"
import { PoolingInterface } from "@/components/playground/PoolingInterface"
import type { ModelState } from "@/types"

function detectMode(model: ModelState | null): "chat" | "image" | "audio" | "pooling" {
  if (!model) return "chat"
  if (model.capabilities.imageGeneration) return "image"
  if (model.capabilities.audioTranscription || model.capabilities.tts) return "audio"
  if (model.capabilities.pooling || model.capabilities.embeddings) return "pooling"
  return "chat"
}

const DEFAULT_PARAMS: PlaygroundParams = {
  temperature: 0.7,
  maxTokens: 1024,
  topP: 0.9,
  systemPrompt: "You are a helpful assistant.",
  responseFormat: "text",
}

export function Playground() {
  const [loading, setLoading] = useState(true)
  const [models, setModels] = useState<ModelState[]>([])
  const [selectedModelId, setSelectedModelId] = useState("")
  const [params, setParams] = useState<PlaygroundParams>(DEFAULT_PARAMS)
  const [showParams, setShowParams] = useState(() =>
    typeof window !== "undefined" && window.innerWidth >= 1280,
  )
  const [chatKey, setChatKey] = useState(0)

  useEffect(() => {
    let active = true
    const load = async () => {
      try {
        const data = await api.models.list()
        if (!active) return
        const sorted = [...data].sort((a, b) => {
          const aLoaded = a.status === "loaded" ? 0 : 1
          const bLoaded = b.status === "loaded" ? 0 : 1
          if (aLoaded !== bLoaded) return aLoaded - bLoaded
          return a.displayName.localeCompare(b.displayName)
        })
        const loadedFirst = sorted.find((item) => item.status === "loaded")
        setModels(sorted)
        setSelectedModelId((prev) => {
          if (prev && sorted.some((item) => item.modelId === prev)) return prev
          return loadedFirst?.modelId || sorted[0]?.modelId || ""
        })
      } catch (err) {
        toast.error(err instanceof Error ? err.message : "No se pudieron cargar los modelos")
      } finally {
        if (active) setLoading(false)
      }
    }

    void load()
    const timer = window.setInterval(() => {
      void load()
    }, 30_000)

    return () => {
      active = false
      window.clearInterval(timer)
    }
  }, [])

  const selectedModel = useMemo(
    () => models.find((item) => item.modelId === selectedModelId) ?? null,
    [models, selectedModelId],
  )

  const mode = detectMode(selectedModel)

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-2xl font-semibold">Playground</h1>
        <p className="text-muted-foreground">Prueba chat, pooling, imagen y audio por capacidad del modelo.</p>
      </div>

      {loading ? (
        <div className="space-y-2" role="status" aria-label="Cargando modelos">
          <div className="h-16 animate-pulse rounded-md bg-muted" />
          <div className="h-80 animate-pulse rounded-md bg-muted" />
        </div>
      ) : (
        <>
          <div className="flex flex-wrap items-center gap-2">
            <div className="flex-1 min-w-0">
              <ModelSelector models={models} selectedModelId={selectedModelId} onSelect={setSelectedModelId} />
            </div>
            <button
              type="button"
              onClick={() => setChatKey((k) => k + 1)}
              className="inline-flex items-center gap-1.5 rounded-md border border-border px-3 py-1.5 text-sm text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
            >
              <MessageSquarePlus size={14} />
              Nueva conversacion
            </button>
            <button
              type="button"
              onClick={() => setShowParams((p) => !p)}
              className={`inline-flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-sm transition-colors ${
                showParams
                  ? "border-primary/40 bg-primary/10 text-primary"
                  : "border-border text-muted-foreground hover:bg-muted hover:text-foreground"
              }`}
              title={showParams ? "Ocultar parametros" : "Mostrar parametros"}
            >
              <SlidersHorizontal size={14} />
              <span className="hidden sm:inline">Params</span>
            </button>
          </div>

          {selectedModel && selectedModel.status !== "loaded" && (
            <div role="alert" className="flex items-start gap-3 rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-3 text-sm text-amber-100">
              <AlertTriangle size={16} className="mt-0.5 shrink-0 text-amber-400" aria-hidden="true" />
              <div>
                <span className="font-medium">Modelo no cargado</span>
                {" — "}estado actual: <span className="font-mono">{selectedModel.status}</span>.
                {" "}La primera llamada lo cargara automaticamente (puede tardar).{" "}
                <Link to="/models" className="underline underline-offset-2 hover:text-amber-50">
                  Gestionar modelos
                </Link>
              </div>
            </div>
          )}

          <div className={`grid gap-4 transition-all duration-200 ${showParams ? "xl:grid-cols-[minmax(0,1fr)_320px]" : ""}`}>
            <section className="h-[calc(100vh-16rem)] min-h-[400px]">
              {mode === "chat" && (
                <ChatInterface
                  key={chatKey}
                  modelId={selectedModelId}
                  backendType={selectedModel?.backendType ?? null}
                  params={params}
                />
              )}
              {mode === "pooling" && (
                <PoolingInterface
                  modelId={selectedModelId}
                  scoreCapable={Boolean(selectedModel?.capabilities.score)}
                  rerankCapable={Boolean(selectedModel?.capabilities.rerank)}
                  classificationCapable={Boolean(selectedModel?.capabilities.classification)}
                />
              )}
              {mode === "image" && <ImageInterface modelId={selectedModelId} params={params} />}
              {mode === "audio" && (
                <AudioInterface
                  modelId={selectedModelId}
                  params={params}
                  canTranscribe={Boolean(selectedModel?.capabilities.audioTranscription)}
                  canTTS={Boolean(selectedModel?.capabilities.tts)}
                />
              )}
            </section>
            {showParams && <ParamsPanel params={params} onChange={setParams} />}
          </div>
        </>
      )}
    </div>
  )
}
