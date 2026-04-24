import { useEffect, useMemo, useState } from "react"
import { AlertTriangle, MessageSquarePlus, SlidersHorizontal, Sparkles } from "lucide-react"
import { Link } from "react-router-dom"
import * as Tooltip from "@radix-ui/react-tooltip"
import { toast } from "sonner"
import { api } from "@/api/client"
import { useAgentsStore } from "@/stores/agentsStore"
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

  const agents = useAgentsStore((s) => s.agents)
  const fetchAgents = useAgentsStore((s) => s.fetchAll)

  const selectedAgent = useMemo(() => {
    if (!selectedModelId.startsWith("agent/")) return null
    const slug = selectedModelId.slice("agent/".length)
    return agents.find((a) => a.slug === slug) ?? null
  }, [agents, selectedModelId])

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
    void fetchAgents()
    const timer = window.setInterval(() => {
      void load()
    }, 30_000)

    return () => {
      active = false
      window.clearInterval(timer)
    }
  }, [fetchAgents])

  const selectedModel = useMemo(
    () => models.find((item) => item.modelId === selectedModelId) ?? null,
    [models, selectedModelId],
  )

  // When an agent is selected its base model/profile determines capabilities; for now
  // agents are chat-only so we force "chat" mode.
  const mode = selectedAgent ? "chat" : detectMode(selectedModel)

  // When an agent is active, the server forces its system prompt. Pass a placeholder
  // so the client build doesn't leak the old system prompt to /v1/chat/completions.
  const effectiveParams = selectedAgent
    ? { ...params, systemPrompt: "" }
    : params

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
              <ModelSelector
                models={models}
                selectedModelId={selectedModelId}
                onSelect={setSelectedModelId}
                agents={agents}
              />
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

          {selectedAgent && (
            <Tooltip.Provider delayDuration={200}>
              <Tooltip.Root>
                <Tooltip.Trigger asChild>
                  <div
                    role="status"
                    className="flex cursor-help items-start gap-2 rounded-md border border-primary/40 bg-primary/10 px-3 py-2 text-sm text-primary"
                  >
                    <Sparkles size={16} className="mt-0.5 shrink-0" />
                    <span>
                      Powered by agent:{" "}
                      <code className="font-mono">agent/{selectedAgent.slug}</code>. El system
                      prompt y las tools los impone el agente.
                    </span>
                  </div>
                </Tooltip.Trigger>
                <Tooltip.Portal>
                  <Tooltip.Content
                    side="bottom"
                    className="z-50 max-w-md whitespace-pre-wrap rounded-md border border-border bg-popover p-3 text-xs shadow-md"
                  >
                    <p className="mb-1 font-semibold">System prompt</p>
                    <p className="font-mono text-[11px] text-muted-foreground">
                      {selectedAgent.systemPrompt.slice(0, 600)}
                      {selectedAgent.systemPrompt.length > 600 ? "..." : ""}
                    </p>
                    <Tooltip.Arrow className="fill-border" />
                  </Tooltip.Content>
                </Tooltip.Portal>
              </Tooltip.Root>
            </Tooltip.Provider>
          )}

          {!selectedAgent && selectedModel && selectedModel.status !== "loaded" && (
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
                  params={effectiveParams}
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
              {mode === "image" && <ImageInterface modelId={selectedModelId} params={effectiveParams} />}
              {mode === "audio" && (
                <AudioInterface
                  modelId={selectedModelId}
                  params={effectiveParams}
                  canTranscribe={Boolean(selectedModel?.capabilities.audioTranscription)}
                  canTTS={Boolean(selectedModel?.capabilities.tts)}
                />
              )}
            </section>
            {showParams && (
              <ParamsPanel
                params={params}
                onChange={setParams}
                disableSystemPrompt={Boolean(selectedAgent)}
              />
            )}
          </div>
        </>
      )}
    </div>
  )
}
