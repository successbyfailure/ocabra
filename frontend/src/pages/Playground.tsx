import { useEffect, useMemo, useState } from "react"
import {
  AlertTriangle,
  AudioLines,
  Image as ImageIcon,
  Layers,
  MessageSquare,
  MessageSquarePlus,
  Radio,
  SlidersHorizontal,
  Sparkles,
} from "lucide-react"
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
import { RealtimeInterface } from "@/components/playground/RealtimeInterface"
import type { ModelState } from "@/types"

type TaskTab = "text" | "image" | "audio" | "embeddings" | "realtime"

const TAB_DEFS: Array<{
  id: TaskTab
  label: string
  icon: typeof MessageSquare
  matches: (m: ModelState) => boolean
  emptyHint: string
}> = [
  {
    id: "text",
    label: "Texto",
    icon: MessageSquare,
    matches: (m) => m.capabilities.chat || m.capabilities.completion,
    emptyHint: "No hay modelos de chat/completion cargados. Carga uno en /models o selecciona un agente.",
  },
  {
    id: "image",
    label: "Imagen",
    icon: ImageIcon,
    matches: (m) => m.capabilities.imageGeneration || m.capabilities.imageEditing,
    emptyHint: "No hay modelos de imagen disponibles. Instala qwen-image-2.1 o qwen-image-edit-plus.",
  },
  {
    id: "audio",
    label: "Audio",
    icon: AudioLines,
    matches: (m) => m.capabilities.audioTranscription || m.capabilities.tts,
    emptyHint: "No hay modelos de audio (STT/TTS) disponibles.",
  },
  {
    id: "embeddings",
    label: "Embeddings",
    icon: Layers,
    matches: (m) => m.capabilities.embeddings || m.capabilities.pooling || m.capabilities.rerank,
    emptyHint: "No hay modelos de embeddings/rerank disponibles.",
  },
  {
    id: "realtime",
    label: "Realtime",
    icon: Radio,
    matches: () => true,
    emptyHint: "",
  },
]

function suggestInitialTab(model: ModelState | null): TaskTab {
  if (!model) return "text"
  if (model.capabilities.imageGeneration || model.capabilities.imageEditing) return "image"
  if (model.capabilities.audioTranscription || model.capabilities.tts) return "audio"
  if (model.capabilities.embeddings || model.capabilities.pooling || model.capabilities.rerank) return "embeddings"
  return "text"
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
  // ?model=... lets other pages (Agents "Use in Playground") deep-link a
  // preselected model. Task tab is inferred from it below on first load.
  const [selectedModelId, setSelectedModelId] = useState(() => {
    if (typeof window === "undefined") return ""
    return new URLSearchParams(window.location.search).get("model") ?? ""
  })
  const [params, setParams] = useState<PlaygroundParams>(DEFAULT_PARAMS)
  const [showParams, setShowParams] = useState(() =>
    typeof window !== "undefined" && window.innerWidth >= 1280,
  )
  const [chatKey, setChatKey] = useState(0)
  const [tab, setTab] = useState<TaskTab | null>(null)
  const [imageDraft, setImageDraft] = useState<{ url: string; prompt: string } | null>(null)

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
          if (prev?.startsWith("agent/")) return prev
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

  const agentBaseModel = useMemo(() => {
    if (!selectedAgent?.baseModelId) return null
    return models.find((item) => item.modelId === selectedAgent.baseModelId) ?? null
  }, [models, selectedAgent])

  const effectiveModel = selectedAgent ? agentBaseModel : selectedModel

  // First-load auto-select: pick the tab that matches the initially selected
  // model. After that, the user's explicit tab selection is honoured.
  useEffect(() => {
    if (tab !== null) return
    if (!models.length && !selectedModel) return
    setTab(selectedAgent ? "text" : suggestInitialTab(selectedModel))
  }, [tab, models.length, selectedModel, selectedAgent])

  // Compatible model list for the active tab. Agent selections stay valid on
  // the Text tab because agents force chat.
  const activeTab = tab ?? "text"
  const filteredModels = useMemo(() => {
    if (activeTab === "realtime") return models
    const def = TAB_DEFS.find((t) => t.id === activeTab)!
    return models.filter(def.matches)
  }, [activeTab, models])

  const tabHasSelection = useMemo(() => {
    if (selectedAgent && activeTab === "text") return true
    return filteredModels.some((m) => m.modelId === selectedModelId)
  }, [filteredModels, selectedAgent, selectedModelId, activeTab])

  // When the tab is switched and the current model doesn't match, prefer a
  // compatible one (loaded first) so the interface has something to talk to.
  useEffect(() => {
    if (tab === null || tab === "realtime") return
    if (tabHasSelection) return
    const first = filteredModels.find((m) => m.status === "loaded") ?? filteredModels[0]
    if (first) setSelectedModelId(first.modelId)
  }, [tab, tabHasSelection, filteredModels])

  const effectiveParams = selectedAgent
    ? { ...params, systemPrompt: "" }
    : params

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-2xl font-semibold">Playground</h1>
        <p className="text-muted-foreground">Elige tarea, selecciona modelo compatible y trabaja.</p>
      </div>

      {loading ? (
        <div className="space-y-2" role="status" aria-label="Cargando modelos">
          <div className="h-16 animate-pulse rounded-md bg-muted" />
          <div className="h-80 animate-pulse rounded-md bg-muted" />
        </div>
      ) : (
        <>
          <div className="flex flex-wrap items-center gap-1 rounded-md border border-border bg-card p-0.5">
            {TAB_DEFS.map((def) => {
              const Icon = def.icon
              const count = def.id === "realtime"
                ? null
                : models.filter(def.matches).length
              const active = activeTab === def.id
              return (
                <button
                  key={def.id}
                  type="button"
                  onClick={() => setTab(def.id)}
                  className={`inline-flex items-center gap-1.5 rounded-sm px-3 py-1.5 text-sm transition-colors ${
                    active
                      ? "bg-primary text-primary-foreground"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  <Icon size={14} />
                  {def.label}
                  {count !== null && count > 0 && (
                    <span className={`ml-1 rounded-full px-1.5 text-[10px] ${
                      active ? "bg-primary-foreground/20" : "bg-muted"
                    }`}>{count}</span>
                  )}
                </button>
              )
            })}
          </div>

          {activeTab === "realtime" ? (
            <RealtimeInterface models={models} />
          ) : (
            <>
              <div className="flex flex-wrap items-center gap-2">
                <div className="flex-1 min-w-0">
                  <ModelSelector
                    models={filteredModels}
                    selectedModelId={selectedModelId}
                    onSelect={setSelectedModelId}
                    agents={activeTab === "text" ? agents : []}
                  />
                </div>
                {activeTab === "text" && (
                  <button
                    type="button"
                    onClick={() => setChatKey((k) => k + 1)}
                    className="inline-flex items-center gap-1.5 rounded-md border border-border px-3 py-1.5 text-sm text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
                  >
                    <MessageSquarePlus size={14} />
                    Nueva conversacion
                  </button>
                )}
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

              {selectedAgent && activeTab === "text" && (
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

              {effectiveModel && effectiveModel.status !== "loaded" && (
                <div role="alert" className="flex items-start gap-3 rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-3 text-sm text-amber-100">
                  <AlertTriangle size={16} className="mt-0.5 shrink-0 text-amber-400" aria-hidden="true" />
                  <div>
                    <span className="font-medium">
                      {selectedAgent ? "Modelo del agente no cargado" : "Modelo no cargado"}
                    </span>
                    {" — "}estado actual:{" "}
                    <span className="font-mono">{effectiveModel.status}</span>
                    {selectedAgent && (
                      <>
                        {" ("}
                        <span className="font-mono">{effectiveModel.displayName}</span>
                        {")"}
                      </>
                    )}
                    .{" "}La primera llamada lo cargara automaticamente (puede tardar 1-2 min en frio).{" "}
                    <Link to="/models" className="underline underline-offset-2 hover:text-amber-50">
                      Gestionar modelos
                    </Link>
                  </div>
                </div>
              )}

              {filteredModels.length === 0 && !selectedAgent && (
                <div
                  role="status"
                  className="rounded-md border border-border bg-card px-3 py-4 text-sm text-muted-foreground"
                >
                  {TAB_DEFS.find((t) => t.id === activeTab)?.emptyHint}
                </div>
              )}

              <div className={`grid gap-4 transition-all duration-200 ${showParams ? "xl:grid-cols-[minmax(0,1fr)_320px]" : ""}`}>
                <section className="h-[calc(100vh-16rem)] min-h-[400px]">
                  {activeTab === "text" && (
                    <ChatInterface
                      key={chatKey}
                      modelId={selectedModelId}
                      backendType={selectedModel?.backendType ?? null}
                      params={effectiveParams}
                      modelContextLength={effectiveModel?.capabilities.contextLength ?? null}
                      audioInputCapable={Boolean(effectiveModel?.capabilities.audioInput)}
                    />
                  )}
                  {activeTab === "embeddings" && (
                    <PoolingInterface
                      modelId={selectedModelId}
                      scoreCapable={Boolean(selectedModel?.capabilities.score)}
                      rerankCapable={Boolean(selectedModel?.capabilities.rerank)}
                      classificationCapable={Boolean(selectedModel?.capabilities.classification)}
                    />
                  )}
                  {activeTab === "image" && (
                    <ImageInterface
                      modelId={selectedModelId}
                      params={effectiveParams}
                      canGenerate={Boolean(selectedModel?.capabilities.imageGeneration)}
                      canEdit={Boolean(selectedModel?.capabilities.imageEditing)}
                      supportsMultiRef={selectedModelId === "qwen-image-edit-plus" || selectedModelId.endsWith("/qwen-image-edit-plus")}
                      editDraft={imageDraft}
                      onDraftConsumed={() => setImageDraft(null)}
                      onSendToEdit={(url, prompt) => setImageDraft({ url, prompt })}
                    />
                  )}
                  {activeTab === "audio" && (
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
                    modelContextLength={effectiveModel?.capabilities.contextLength ?? null}
                  />
                )}
              </div>
            </>
          )}
        </>
      )}
    </div>
  )
}
