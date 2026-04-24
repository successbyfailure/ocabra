import { useState } from "react"
import * as DropdownMenu from "@radix-ui/react-dropdown-menu"
import * as Tooltip from "@radix-ui/react-tooltip"
import {
  ChevronDown,
  ChevronRight,
  Cpu,
  MoreHorizontal,
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
              {(model.createdAt || model.updatedAt) && (
                <div className="text-[10px] text-muted-foreground/60 mt-0.5">
                  {model.createdAt && `Registrado ${new Date(model.createdAt).toLocaleDateString()}`}
                  {model.updatedAt && model.createdAt !== model.updatedAt && ` · Actualizado ${new Date(model.updatedAt).toLocaleDateString()}`}
                </div>
              )}
            </div>
          </div>
        </td>
        <td className="px-3 py-3 text-muted-foreground">{modelType(model)}</td>
        <td className="px-3 py-3 text-muted-foreground hidden lg:table-cell">{model.backendType}</td>
        <td className="px-3 py-3">
          <LoadPolicyBadge policy={model.loadPolicy} />
        </td>
        <td className="px-3 py-3 text-muted-foreground">
          {model.currentGpu.join(", ") || (model.preferredGpu ?? "-")}
        </td>
        <td className="px-3 py-3 text-muted-foreground hidden xl:table-cell">
          <div>{formatTokenCount(context.nativeContext)}</div>
          <div className="text-xs opacity-70">nativo</div>
        </td>
        <td className="px-3 py-3 text-muted-foreground hidden xl:table-cell">
          <div>{formatTokenCount(context.configuredContext)}</div>
          <div className="text-xs opacity-70">configurado</div>
        </td>
        <td className="px-3 py-3 text-muted-foreground hidden xl:table-cell">
          <div>{formatTokenCount(context.maxInputTokens)} / {formatTokenCount(context.maxOutputTokens)}</div>
          <div className="text-xs opacity-70">input / output</div>
        </td>
        <td className="px-3 py-3 text-muted-foreground">{model.vramUsedMb.toLocaleString()} MB</td>
        <td className="px-3 py-3 text-muted-foreground hidden lg:table-cell">{formatDiskSize(model.diskSizeBytes)}</td>
        <td className="px-3 py-3">
          <ModelStatusBadge status={model.status} />
        </td>
        <td className="px-3 py-3 text-center text-xs text-muted-foreground">
          {profileCount}
        </td>
        <td className="px-3 py-3">
          <div className="flex items-center gap-1.5">
            {/* Primary actions: Load / Unload */}
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

            {/* Secondary actions: dropdown */}
            <DropdownMenu.Root>
              <DropdownMenu.Trigger asChild>
                <button
                  type="button"
                  disabled={busy}
                  className="rounded-md border border-border p-1.5 text-muted-foreground hover:bg-muted hover:text-foreground disabled:opacity-40"
                  title="Mas acciones"
                >
                  <MoreHorizontal size={14} />
                </button>
              </DropdownMenu.Trigger>
              <DropdownMenu.Portal>
                <DropdownMenu.Content
                  className="z-50 min-w-[180px] rounded-md border border-border bg-card p-1 shadow-md"
                  sideOffset={4}
                  align="end"
                >
                  <DropdownMenu.Item
                    className="flex items-center gap-2 rounded-sm px-2 py-1.5 text-sm cursor-pointer outline-none hover:bg-muted focus:bg-muted"
                    onSelect={() => onTogglePin(model)}
                  >
                    {model.loadPolicy === "pin" ? <PinOff size={14} /> : <Pin size={14} />}
                    {model.loadPolicy === "pin" ? "Unpin" : "Pin"}
                  </DropdownMenu.Item>
                  <DropdownMenu.Item
                    className="flex items-center gap-2 rounded-sm px-2 py-1.5 text-sm cursor-pointer outline-none hover:bg-muted focus:bg-muted"
                    onSelect={() => onConfigure(model)}
                  >
                    <Pencil size={14} />
                    Configurar
                  </DropdownMenu.Item>
                  {onCompile && model.backendType === "vllm" && (
                    <DropdownMenu.Item
                      className="flex items-center gap-2 rounded-sm px-2 py-1.5 text-sm cursor-pointer outline-none hover:bg-muted focus:bg-muted text-purple-300"
                      onSelect={() => onCompile(model)}
                    >
                      <Cpu size={14} />
                      Compilar TRT
                    </DropdownMenu.Item>
                  )}
                  <DropdownMenu.Separator className="my-1 h-px bg-border" />
                  <DropdownMenu.Item
                    className="flex items-center gap-2 rounded-sm px-2 py-1.5 text-sm cursor-pointer outline-none hover:bg-muted focus:bg-muted text-red-400"
                    onSelect={() => onDelete(model)}
                  >
                    <Trash2 size={14} />
                    Eliminar
                  </DropdownMenu.Item>
                </DropdownMenu.Content>
              </DropdownMenu.Portal>
            </DropdownMenu.Root>
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
          <tr key={profile.profileId} className="border-b border-border/30 border-l-2 border-l-primary/30 bg-muted/10 text-sm">
            <td className="py-2 pl-12 pr-3">
              <div className="font-mono text-xs">{profile.profileId}</div>
            </td>
            <td className="px-3 py-2">
              <span className="text-xs text-muted-foreground">
                {profile.displayName || "-"}
              </span>
            </td>
            <td className="px-3 py-2 hidden lg:table-cell">
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
            <td className="px-3 py-2 hidden xl:table-cell" colSpan={2}>
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
            <td className="px-3 py-2 hidden xl:table-cell" colSpan={4} />
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
        <tr className="border-b border-border/60 border-l-2 border-l-primary/30 bg-muted/5 text-sm">
          <td colSpan={13} className="py-2 pl-12 pr-3">
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
