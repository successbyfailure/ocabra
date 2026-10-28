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
  RefreshCw,
  Share2,
  Square,
  Trash2,
} from "lucide-react"
import { CapabilityBadge } from "@/components/common/CapabilityBadge"
import { BackendBadge } from "@/components/models/BackendBadge"
import { LoadPolicyBadge } from "@/components/models/LoadPolicyBadge"
import { ModelStatusBadge } from "@/components/models/ModelStatusBadge"
import { CATEGORY_COLORS } from "@/components/models/ProfileModal"
import { formatTokenCount, getModelContextSummary, getVllmConfig, type ModelCategory } from "@/lib/modelContext"
import type { ModelProfile, ModelState, ProfileCategory } from "@/types"

interface ModelGridCardProps {
  model: ModelState
  category: ModelCategory
  busy?: boolean
  onLoad: (modelId: string) => void
  onUnload: (modelId: string) => void
  onTogglePin: (model: ModelState) => void
  onConfigure: (model: ModelState) => void
  onDelete: (model: ModelState) => void
  onCompile?: (model: ModelState) => void
  onUpdate?: (model: ModelState) => void
  onEditProfile: (model: ModelState, profile: ModelProfile | null) => void
  onToggleProfileEnabled: (profile: ModelProfile) => void
}

function formatDiskSize(bytes: number | null): string {
  if (bytes == null || bytes <= 0) return "-"
  const gb = bytes / 1024 ** 3
  if (gb >= 1) return `${gb.toFixed(2)} GB`
  const mb = bytes / 1024 ** 2
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
    message: compact.length > 160 ? `${compact.slice(0, 157)}...` : compact,
    suggestTrustRemote: false,
  }
}

function CategoryPill({ category }: { category: ProfileCategory }) {
  const colors = CATEGORY_COLORS[category] ?? "bg-muted text-muted-foreground border-border"
  return (
    <span className={`inline-block rounded border px-1.5 py-0.5 text-[10px] font-medium uppercase ${colors}`}>
      {category}
    </span>
  )
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0">
      <div className="truncate text-sm text-foreground">{value}</div>
      <div className="text-[10px] uppercase tracking-wide text-muted-foreground/70">{label}</div>
    </div>
  )
}

export function ModelGridCard({
  model,
  category,
  busy,
  onLoad,
  onUnload,
  onTogglePin,
  onConfigure,
  onDelete,
  onCompile,
  onUpdate,
  onEditProfile,
  onToggleProfileEnabled,
}: ModelGridCardProps) {
  const [expanded, setExpanded] = useState(false)
  const isLoaded = model.status === "loaded" || model.status === "loading"
  const isUnloaded = model.status === "unloaded" || model.status === "unloading"
  const errorHint = model.status === "error" ? summarizeError(model) : null
  const context = getModelContextSummary(model)
  const profiles = model.profiles ?? []
  const profileCount = profiles.length
  const gpuLabel = model.currentGpu.length > 0 ? model.currentGpu.join(", ") : (model.preferredGpu ?? null)

  return (
    <div className="flex flex-col gap-3 rounded-lg border border-border bg-card p-4 transition-colors hover:border-primary/40">
      {/* Header */}
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="flex items-center gap-1.5">
            <span className={`truncate font-medium${model.federation?.remote ? " opacity-75" : ""}`}>
              {profileCount > 1 ? model.modelId : model.displayName}
            </span>
            {model.federation?.remote && (
              <Tooltip.Root>
                <Tooltip.Trigger asChild>
                  <span className="inline-flex shrink-0 items-center gap-1 rounded-full border border-violet-500/30 bg-violet-500/20 px-1.5 py-0.5 text-[10px] font-medium text-violet-200">
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
          <div className="truncate text-xs text-muted-foreground">
            {profileCount > 1 ? `${profileCount} perfiles` : model.modelId}
          </div>
        </div>
        <ModelStatusBadge status={model.status} />
      </div>

      {/* Badges row */}
      <div className="flex flex-wrap items-center gap-1.5">
        <BackendBadge backendType={model.backendType} />
        <LoadPolicyBadge policy={model.loadPolicy} />
        {model.capabilities.vision && <CapabilityBadge capability="vision" />}
        {model.capabilities.tools && <CapabilityBadge capability="tools" />}
        {model.capabilities.reasoning && <CapabilityBadge capability="reasoning" />}
        {model.capabilities.audioInput && <CapabilityBadge capability="audioInput" />}
        {model.capabilities.audioOutput && <CapabilityBadge capability="audioOutput" />}
        {model.capabilities.videoInput && <CapabilityBadge capability="videoInput" />}
      </div>

      {/* Stats grid */}
      <div className="grid grid-cols-2 gap-x-3 gap-y-2 rounded-md bg-muted/30 p-2.5">
        <Stat label="GPU" value={gpuLabel != null ? String(gpuLabel) : "-"} />
        <Stat label="VRAM" value={`${model.vramUsedMb.toLocaleString()} MB`} />
        {category === "chat" ? (
          <>
            <Stat label="Ctx nativo" value={formatTokenCount(context.nativeContext)} />
            <Stat label="Ctx config" value={formatTokenCount(context.configuredContext)} />
          </>
        ) : (
          <Stat label="Tamaño" value={formatDiskSize(model.diskSizeBytes)} />
        )}
      </div>

      {errorHint && (
        <div className="rounded-md border border-red-500/30 bg-red-500/5 px-2.5 py-2 text-xs text-red-200">
          <p>{errorHint.message}</p>
          {errorHint.suggestTrustRemote && (
            <button
              type="button"
              onClick={() => onConfigure(model)}
              className="mt-1.5 rounded-md border border-amber-500/40 px-2 py-1 text-[11px] text-amber-100 hover:bg-amber-500/10"
            >
              Configurar trust remote code
            </button>
          )}
        </div>
      )}

      {/* Profiles (collapsible) */}
      {profileCount > 0 && (
        <div>
          <button
            type="button"
            onClick={() => setExpanded((v) => !v)}
            className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
          >
            {expanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
            {profileCount} perfil{profileCount === 1 ? "" : "es"}
          </button>
          {expanded && (
            <div className="mt-2 space-y-1.5 border-l-2 border-primary/30 pl-2.5">
              {profiles.map((profile) => (
                <div key={profile.profileId} className="flex items-center justify-between gap-2 text-xs">
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-1.5">
                      <span className="truncate font-mono text-[11px]">{profile.profileId}</span>
                      <CategoryPill category={profile.category} />
                      {profile.isDefault && (
                        <span className="shrink-0 rounded border border-sky-500/40 bg-sky-500/10 px-1 py-0.5 text-[9px] font-medium text-sky-300">
                          default
                        </span>
                      )}
                      {profile.routingTargets && profile.routingTargets.length > 0 && (
                        <span className="shrink-0 rounded border border-primary/40 bg-primary/10 px-1 py-0.5 text-[9px] font-medium uppercase text-primary">
                          router
                        </span>
                      )}
                    </div>
                  </div>
                  <div className="flex shrink-0 items-center gap-1.5">
                    <button
                      type="button"
                      role="switch"
                      aria-checked={profile.enabled}
                      onClick={() => onToggleProfileEnabled(profile)}
                      className={`relative h-3.5 w-6 rounded-full transition-colors ${
                        profile.enabled ? "bg-emerald-500" : "bg-muted"
                      }`}
                      title={profile.enabled ? "Desactivar" : "Activar"}
                    >
                      <span
                        className={`absolute top-0.5 block h-2.5 w-2.5 rounded-full bg-white transition-transform ${
                          profile.enabled ? "translate-x-3" : "translate-x-0.5"
                        }`}
                      />
                    </button>
                    <button
                      type="button"
                      onClick={() => onEditProfile(model, profile)}
                      className="rounded border border-border p-0.5 hover:bg-muted"
                      title="Editar perfil"
                    >
                      <Pencil size={11} />
                    </button>
                  </div>
                </div>
              ))}
              <button
                type="button"
                onClick={() => onEditProfile(model, null)}
                className="flex items-center gap-1 rounded-md border border-dashed border-border px-2 py-1 text-[11px] text-muted-foreground hover:border-primary hover:text-primary"
              >
                <Plus size={11} /> Añadir perfil
              </button>
            </div>
          )}
        </div>
      )}

      {/* Actions footer */}
      <div className="mt-auto flex items-center gap-1.5 border-t border-border/60 pt-3">
        <button
          type="button"
          onClick={() => onLoad(model.modelId)}
          disabled={busy || isLoaded}
          className="inline-flex items-center gap-1 rounded-md border border-emerald-500/40 px-2 py-1.5 text-xs text-emerald-200 disabled:opacity-40"
          title="Load"
        >
          <Play size={13} /> Cargar
        </button>
        <button
          type="button"
          onClick={() => onUnload(model.modelId)}
          disabled={busy || isUnloaded}
          className="inline-flex items-center gap-1 rounded-md border border-red-500/40 px-2 py-1.5 text-xs text-red-200 disabled:opacity-40"
          title="Unload"
        >
          <Square size={13} /> Descargar
        </button>

        <div className="ml-auto">
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
                  className="flex cursor-pointer items-center gap-2 rounded-sm px-2 py-1.5 text-sm outline-none hover:bg-muted focus:bg-muted"
                  onSelect={() => onTogglePin(model)}
                >
                  {model.loadPolicy === "pin" ? <PinOff size={14} /> : <Pin size={14} />}
                  {model.loadPolicy === "pin" ? "Unpin" : "Pin"}
                </DropdownMenu.Item>
                <DropdownMenu.Item
                  className="flex cursor-pointer items-center gap-2 rounded-sm px-2 py-1.5 text-sm outline-none hover:bg-muted focus:bg-muted"
                  onSelect={() => onConfigure(model)}
                >
                  <Pencil size={14} />
                  Configurar
                </DropdownMenu.Item>
                {onCompile && model.backendType === "vllm" && (
                  <DropdownMenu.Item
                    className="flex cursor-pointer items-center gap-2 rounded-sm px-2 py-1.5 text-sm text-purple-300 outline-none hover:bg-muted focus:bg-muted"
                    onSelect={() => onCompile(model)}
                  >
                    <Cpu size={14} />
                    Compilar TRT
                  </DropdownMenu.Item>
                )}
                {onUpdate && (
                  <DropdownMenu.Item
                    className="flex cursor-pointer items-center gap-2 rounded-sm px-2 py-1.5 text-sm outline-none hover:bg-muted focus:bg-muted"
                    onSelect={() => onUpdate(model)}
                  >
                    <RefreshCw size={14} />
                    Actualizar
                  </DropdownMenu.Item>
                )}
                <DropdownMenu.Separator className="my-1 h-px bg-border" />
                <DropdownMenu.Item
                  className="flex cursor-pointer items-center gap-2 rounded-sm px-2 py-1.5 text-sm text-red-400 outline-none hover:bg-muted focus:bg-muted"
                  onSelect={() => onDelete(model)}
                >
                  <Trash2 size={14} />
                  Eliminar
                </DropdownMenu.Item>
              </DropdownMenu.Content>
            </DropdownMenu.Portal>
          </DropdownMenu.Root>
        </div>
      </div>
    </div>
  )
}
