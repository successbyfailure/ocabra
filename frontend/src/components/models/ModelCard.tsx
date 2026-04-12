import { useState } from "react"
import * as Tooltip from "@radix-ui/react-tooltip"
import {
  ChevronDown,
  ChevronRight,
  Cpu,
  Pencil,
  Pin,
  PinOff,
  Play,
  Plus,
  Server,
  Share2,
  Square,
  Trash2,
} from "lucide-react"
import { LoadPolicyBadge } from "@/components/models/LoadPolicyBadge"
import { ModelStatusBadge } from "@/components/models/ModelStatusBadge"
import { CATEGORY_COLORS } from "@/components/models/ProfileModal"
import { formatTokenCount, getModelContextSummary, getVllmConfig } from "@/lib/modelContext"
import type { ModelProfile, ModelState, ProfileCategory } from "@/types"

interface ModelCardProps {
  model: ModelState
  busy?: boolean
  onLoad: (modelId: string) => void
  onUnload: (modelId: string) => void
  onTogglePin: (model: ModelState) => void
  onConfigure: (model: ModelState) => void
  onDelete: (model: ModelState) => void
  onCompile?: (model: ModelState) => void
  onEditProfile: (model: ModelState, profile: ModelProfile | null) => void
  onToggleProfileEnabled: (profile: ModelProfile) => void
}

function modelType(model: ModelState): string {
  if (model.capabilities.imageGeneration) return "image"
  if (model.capabilities.audioTranscription || model.capabilities.tts) return "audio"
  if (model.capabilities.pooling || model.capabilities.embeddings) return "pooling"
  return "llm"
}

function formatDiskSize(bytes: number | null): string {
  if (bytes == null || bytes <= 0) return "-"
  const gb = bytes / (1024 ** 3)
  if (gb >= 1) return `${gb.toFixed(2)} GB`
  const mb = bytes / (1024 ** 2)
  return `${mb.toFixed(0)} MB`
}

function summarizeError(model: ModelState): { message: string; suggestTrustRemote: boolean } | null {
  const error = model.errorMessage?.trim()
  if (!error) return null

  const trustRemoteEnabled = Boolean(getVllmConfig(model)?.trustRemoteCode)
  const suggestTrustRemote =
    error.includes("trust_remote_code") ||
    error.includes("--trust-remote-code") ||
    error.includes("custom tokenizer") ||
    error.includes("Tokenizer class")

  if (suggestTrustRemote) {
    if (trustRemoteEnabled) {
      return {
        message:
          "El modelo sigue fallando aunque 'trust remote code' ya esta activado. Parece una incompatibilidad real del tokenizer/modelo con esta version de vLLM o transformers.",
        suggestTrustRemote: false,
      }
    }

    return {
      message: "Este modelo parece requerir trust_remote_code. Abre Configurar y activa 'trust remote code'.",
      suggestTrustRemote: true,
    }
  }

  const compact = error.split("\n")[0]?.trim() || "Error al cargar el modelo."
  return {
    message: compact.length > 220 ? `${compact.slice(0, 217)}...` : compact,
    suggestTrustRemote: false,
  }
}

function CategoryBadge({ category }: { category: ProfileCategory }) {
  const colors = CATEGORY_COLORS[category] ?? "bg-muted text-muted-foreground border-border"
  return (
    <span className={`inline-block rounded border px-1.5 py-0.5 text-[10px] font-medium uppercase ${colors}`}>
      {category}
    </span>
  )
}

export function ModelCard({
  model,
  busy,
  onLoad,
  onUnload,
  onTogglePin,
  onConfigure,
  onDelete,
  onCompile,
  onEditProfile,
  onToggleProfileEnabled,
}: ModelCardProps) {
  const [expanded, setExpanded] = useState(false)
  const isLoaded = model.status === "loaded" || model.status === "loading"
  const isUnloaded = model.status === "unloaded" || model.status === "unloading"
  const errorHint = model.status === "error" ? summarizeError(model) : null
  const context = getModelContextSummary(model)
  const profiles = model.profiles ?? []
  const profileCount = profiles.length

  return (
    <>
      <tr className="border-b border-border/60 text-sm">
        <td className="px-3 py-3">
          <div className="flex items-center gap-2">
            {profileCount > 0 && (
              <button
                type="button"
                onClick={() => setExpanded(!expanded)}
                className="shrink-0 rounded p-0.5 hover:bg-muted"
                aria-label={expanded ? "Colapsar perfiles" : "Expandir perfiles"}
              >
                {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
              </button>
            )}
            <div>
              <div className="flex items-center gap-1.5">
                <span className={`font-medium${model.federation?.remote ? " opacity-75" : ""}`}>{model.displayName}</span>
                {model.federation?.remote && (
                  <Tooltip.Root>
                    <Tooltip.Trigger asChild>
                      <span className="inline-flex items-center gap-1 rounded-full border border-violet-500/30 bg-violet-500/20 px-1.5 py-0.5 text-[10px] font-medium text-violet-200">
                        <Share2 size={9} />
                        Remote
                      </span>
                    </Tooltip.Trigger>
                    <Tooltip.Portal>
                      <Tooltip.Content className="z-50 rounded-md border border-border bg-popover px-3 py-1.5 text-xs shadow-md" sideOffset={4}>
                        Modelo federado desde nodo: {model.federation.nodeName || model.federation.nodeId}
                        <Tooltip.Arrow className="fill-border" />
                      </Tooltip.Content>
                    </Tooltip.Portal>
                  </Tooltip.Root>
                )}
              </div>
              <div className="text-xs text-muted-foreground">{model.modelId}</div>
            </div>
          </div>
        </td>
        <td className="px-3 py-3 text-muted-foreground">{modelType(model)}</td>
        <td className="px-3 py-3 text-muted-foreground">{model.backendType}</td>
        <td className="px-3 py-3">
          <LoadPolicyBadge policy={model.loadPolicy} />
        </td>
        <td className="px-3 py-3 text-muted-foreground">
          {model.currentGpu.join(", ") || (model.preferredGpu ?? "-")}
        </td>
        <td className="px-3 py-3 text-muted-foreground">
          <div>{formatTokenCount(context.nativeContext)}</div>
          <div className="text-xs opacity-70">nativo</div>
        </td>
        <td className="px-3 py-3 text-muted-foreground">
          <div>{formatTokenCount(context.configuredContext)}</div>
          <div className="text-xs opacity-70">configurado</div>
        </td>
        <td className="px-3 py-3 text-muted-foreground">
          <div>{formatTokenCount(context.maxInputTokens)} / {formatTokenCount(context.maxOutputTokens)}</div>
          <div className="text-xs opacity-70">input / output</div>
        </td>
        <td className="px-3 py-3 text-muted-foreground">{model.vramUsedMb.toLocaleString()} MB</td>
        <td className="px-3 py-3 text-muted-foreground">{formatDiskSize(model.diskSizeBytes)}</td>
        <td className="px-3 py-3">
          <ModelStatusBadge status={model.status} />
        </td>
        <td className="px-3 py-3 text-center text-xs text-muted-foreground">
          {profileCount}
        </td>
        <td className="px-3 py-3">
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={() => onLoad(model.modelId)}
              disabled={busy || isLoaded}
              className="rounded-md border border-emerald-500/40 p-1.5 text-emerald-200 disabled:opacity-40"
              title="Load"
            >
              <Play size={14} />
            </button>
            <button
              type="button"
              onClick={() => onUnload(model.modelId)}
              disabled={busy || isUnloaded}
              className="rounded-md border border-red-500/40 p-1.5 text-red-200 disabled:opacity-40"
              title="Unload"
            >
              <Square size={14} />
            </button>
            <button
              type="button"
              onClick={() => onTogglePin(model)}
              disabled={busy}
              className="rounded-md border border-border p-1.5 text-foreground hover:bg-muted disabled:opacity-40"
              title={model.loadPolicy === "pin" ? "Unpin" : "Pin"}
            >
              {model.loadPolicy === "pin" ? <PinOff size={14} /> : <Pin size={14} />}
            </button>
            <button
              type="button"
              onClick={() => onConfigure(model)}
              disabled={busy}
              className="rounded-md border border-border p-1.5 text-foreground hover:bg-muted disabled:opacity-40"
              title="Configure"
            >
              <Pencil size={14} />
            </button>
            {onCompile && model.backendType === "vllm" && (
              <button
                type="button"
                onClick={() => onCompile(model)}
                disabled={busy}
                className="rounded-md border border-purple-500/40 p-1.5 text-purple-300 hover:bg-purple-500/10 disabled:opacity-40"
                title="Compilar engine TensorRT-LLM"
              >
                <Cpu size={14} />
              </button>
            )}
            <button
              type="button"
              onClick={() => onDelete(model)}
              disabled={busy}
              className="rounded-md border border-red-500/40 p-1.5 text-red-200 disabled:opacity-40"
              title="Delete"
            >
              <Trash2 size={14} />
            </button>
          </div>
        </td>
      </tr>
      {errorHint && (
        <tr className="border-b border-border/60 bg-red-500/5 text-sm">
          <td colSpan={13} className="px-3 py-3">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <p className="text-red-200">{errorHint.message}</p>
              {errorHint.suggestTrustRemote && (
                <button
                  type="button"
                  onClick={() => onConfigure(model)}
                  className="rounded-md border border-amber-500/40 px-3 py-1.5 text-xs text-amber-100 hover:bg-amber-500/10"
                >
                  Configurar trust remote code
                </button>
              )}
            </div>
          </td>
        </tr>
      )}
      {expanded && profiles.map((profile) => {
        const hasOverrides =
          profile.loadOverrides !== null &&
          Object.keys(profile.loadOverrides).length > 0
        return (
          <tr key={profile.profileId} className="border-b border-border/30 bg-muted/10 text-sm">
            <td className="py-2 pl-10 pr-3">
              <div className="font-mono text-xs">{profile.profileId}</div>
            </td>
            <td className="px-3 py-2">
              <span className="text-xs text-muted-foreground">
                {profile.displayName || "-"}
              </span>
            </td>
            <td className="px-3 py-2">
              <CategoryBadge category={profile.category} />
            </td>
            <td className="px-3 py-2">
              <button
                type="button"
                role="switch"
                aria-checked={profile.enabled}
                onClick={() => onToggleProfileEnabled(profile)}
                className={`relative h-4 w-7 rounded-full transition-colors ${
                  profile.enabled ? "bg-emerald-500" : "bg-muted"
                }`}
                title={profile.enabled ? "Desactivar" : "Activar"}
              >
                <span
                  className={`absolute top-0.5 block h-3 w-3 rounded-full bg-white transition-transform ${
                    profile.enabled ? "translate-x-3.5" : "translate-x-0.5"
                  }`}
                />
              </button>
            </td>
            <td className="px-3 py-2">
              {profile.isDefault && (
                <span className="rounded border border-sky-500/40 bg-sky-500/10 px-1.5 py-0.5 text-[10px] font-medium text-sky-300">
                  default
                </span>
              )}
            </td>
            <td className="px-3 py-2" colSpan={2}>
              <Tooltip.Provider delayDuration={200}>
                <Tooltip.Root>
                  <Tooltip.Trigger asChild>
                    <span className="inline-flex cursor-help items-center gap-1 text-xs text-muted-foreground">
                      {hasOverrides ? (
                        <>
                          <Server size={12} className="text-amber-400" /> Dedicado
                        </>
                      ) : (
                        <>
                          <Share2 size={12} className="text-emerald-400" /> Compartido
                        </>
                      )}
                    </span>
                  </Tooltip.Trigger>
                  <Tooltip.Portal>
                    <Tooltip.Content className="z-[60] max-w-xs rounded-md border border-border bg-popover px-3 py-1.5 text-xs shadow-md" sideOffset={4}>
                      {hasOverrides
                        ? "Este perfil tiene load_overrides distintos, lo que crea un worker GPU separado."
                        : "Este perfil comparte el worker del modelo base (sin overrides de carga)."}
                      <Tooltip.Arrow className="fill-border" />
                    </Tooltip.Content>
                  </Tooltip.Portal>
                </Tooltip.Root>
              </Tooltip.Provider>
            </td>
            <td className="px-3 py-2" colSpan={4} />
            <td className="px-3 py-2" />
            <td className="px-3 py-2">
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => onEditProfile(model, profile)}
                  className="rounded-md border border-border p-1 text-foreground hover:bg-muted"
                  title="Editar perfil"
                >
                  <Pencil size={12} />
                </button>
              </div>
            </td>
          </tr>
        )
      })}
      {expanded && (
        <tr className="border-b border-border/60 bg-muted/5 text-sm">
          <td colSpan={13} className="py-2 pl-10 pr-3">
            <button
              type="button"
              onClick={() => onEditProfile(model, null)}
              className="flex items-center gap-1.5 rounded-md border border-dashed border-border px-2.5 py-1 text-xs text-muted-foreground hover:border-primary hover:text-primary"
            >
              <Plus size={12} /> Anadir perfil
            </button>
          </td>
        </tr>
      )}
    </>
  )
}
