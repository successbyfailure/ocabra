import { Pencil, Pin, PinOff, Play, Square, Trash2 } from "lucide-react"
import { LoadPolicyBadge } from "@/components/models/LoadPolicyBadge"
import { ModelStatusBadge } from "@/components/models/ModelStatusBadge"
import type { ModelState } from "@/types"

interface ModelCardProps {
  model: ModelState
  busy?: boolean
  onLoad: (modelId: string) => void
  onUnload: (modelId: string) => void
  onTogglePin: (model: ModelState) => void
  onConfigure: (model: ModelState) => void
  onDelete: (model: ModelState) => void
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

  const trustRemoteEnabled = Boolean(
    (model.extraConfig?.vllm as { trustRemoteCode?: boolean } | undefined)?.trustRemoteCode,
  )
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

export function ModelCard({ model, busy, onLoad, onUnload, onTogglePin, onConfigure, onDelete }: ModelCardProps) {
  const isLoaded = model.status === "loaded" || model.status === "loading"
  const isUnloaded = model.status === "unloaded" || model.status === "unloading"
  const errorHint = model.status === "error" ? summarizeError(model) : null

  return (
    <>
      <tr className="border-b border-border/60 text-sm">
        <td className="px-3 py-3">
          <div className="font-medium">{model.displayName}</div>
          <div className="text-xs text-muted-foreground">{model.modelId}</div>
        </td>
        <td className="px-3 py-3 text-muted-foreground">{modelType(model)}</td>
        <td className="px-3 py-3 text-muted-foreground">{model.backendType}</td>
        <td className="px-3 py-3">
          <LoadPolicyBadge policy={model.loadPolicy} />
        </td>
        <td className="px-3 py-3 text-muted-foreground">
          {model.currentGpu.join(", ") || (model.preferredGpu ?? "-")}
        </td>
        <td className="px-3 py-3 text-muted-foreground">{model.vramUsedMb.toLocaleString()} MB</td>
        <td className="px-3 py-3 text-muted-foreground">{formatDiskSize(model.diskSizeBytes)}</td>
        <td className="px-3 py-3">
          <ModelStatusBadge status={model.status} />
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
          <td colSpan={9} className="px-3 py-3">
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
    </>
  )
}
