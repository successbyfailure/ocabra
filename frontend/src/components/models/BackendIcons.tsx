import * as Tooltip from "@radix-ui/react-tooltip"
import type { CompatibleBackend, ModelDeployment } from "@/types"

/**
 * One backend of a model as shown by {@link BackendIcons}: either an existing
 * deployment or a backend that could serve the same file.
 */
export interface BackendIconItem {
  backend: string
  /** Serves the model's public name. */
  isDefault?: boolean
  /** A deployment exists for this backend (vs. merely compatible). */
  deployed?: boolean
  /** Backend module installed on this node. */
  installed?: boolean
  status?: CompatibleBackend["status"]
  reason?: string
  /** Deployment id, when deployed. */
  modelId?: string
}

/** Short labels so a row of 3–4 backends stays on one line. */
const SHORT_LABELS: Record<string, string> = {
  llamacpp_mks: "llama.cpp mks",
  llama_cpp: "llama.cpp",
  whisper_cpp: "whisper.cpp",
  tensorrt_llm: "TensorRT",
  strata: "Strata",
}

export function backendLabel(backend: string): string {
  return SHORT_LABELS[backend] ?? backend
}

/**
 * Deployments first (default on top), then the compatible backends that are
 * not deployed yet. A backend appearing in both lists is shown once.
 */
export function mergeBackendItems(
  deployments: ModelDeployment[],
  compatible: CompatibleBackend[] = [],
): BackendIconItem[] {
  const byBackend = new Map(compatible.map((entry) => [entry.backend, entry]))
  const items: BackendIconItem[] = [...deployments]
    .sort((a, b) => Number(b.isDefault) - Number(a.isDefault))
    .map((dep) => {
      const compat = byBackend.get(dep.backend)
      return {
        backend: dep.backend,
        isDefault: dep.isDefault,
        deployed: true,
        installed: dep.installed,
        status: compat?.status,
        reason: compat?.reason,
        modelId: dep.modelId,
      }
    })
  const seen = new Set(items.map((item) => item.backend))
  for (const entry of compatible) {
    if (seen.has(entry.backend)) continue
    seen.add(entry.backend)
    items.push({
      backend: entry.backend,
      deployed: Boolean(entry.deployed),
      installed: entry.installed,
      status: entry.status,
      reason: entry.reason,
    })
  }
  return items
}

/** Compatible backends from a registry search, without deployments. */
export function compatibleToItems(compatible: CompatibleBackend[] = []): BackendIconItem[] {
  return mergeBackendItems([], compatible)
}

function describe(item: BackendIconItem): string {
  const parts = [backendLabel(item.backend)]
  if (item.isDefault) parts.push("por defecto")
  if (item.deployed) parts.push("desplegado")
  else parts.push(item.status === "probable" ? "probablemente compatible" : "compatible")
  if (item.installed === false) parts.push("no instalado")
  if (item.reason) parts.push(item.reason)
  return parts.join(" · ")
}

function itemState(item: BackendIconItem): string {
  if (item.installed === false) return "not-installed"
  if (item.deployed) return item.isDefault ? "default" : "deployed"
  return item.status === "probable" ? "probable" : "compatible"
}

function itemClass(item: BackendIconItem): string {
  const base = "inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 text-xs leading-4"
  if (item.installed === false) {
    return `${base} border-dashed border-border bg-transparent text-muted-foreground/60 line-through decoration-muted-foreground/40`
  }
  if (item.deployed) {
    return item.isDefault
      ? `${base} border-amber-400/60 bg-amber-400/15 font-medium text-amber-100`
      : `${base} border-sky-500/40 bg-sky-500/15 text-sky-100`
  }
  if (item.status === "probable") {
    return `${base} border-dashed border-border bg-transparent text-muted-foreground/80 opacity-70`
  }
  return `${base} border-dashed border-emerald-500/40 bg-transparent text-emerald-200/90`
}

interface BackendIconsProps {
  items: BackendIconItem[]
  /** Called when a deployed backend is clicked (e.g. open its config). */
  onSelect?: (item: BackendIconItem) => void
  /** Text shown when there are no items. Omit to render nothing. */
  emptyLabel?: string
  className?: string
}

/**
 * Compact row of backend badges for a model: ★ marks the default deployment,
 * solid = deployed, dashed = compatible but not deployed, faded with "?" =
 * probable (tooltip carries the reason), struck-through = backend not
 * installed on this node.
 */
export function BackendIcons({ items, onSelect, emptyLabel, className = "" }: BackendIconsProps) {
  if (items.length === 0) {
    return emptyLabel ? <span className="text-xs text-muted-foreground">{emptyLabel}</span> : null
  }
  return (
    <Tooltip.Provider delayDuration={200}>
      <div className={`flex flex-wrap items-center gap-1 ${className}`} role="list" aria-label="Backends">
        {items.map((item) => {
          const label = describe(item)
          const clickable = Boolean(onSelect && item.deployed)
          const content = (
            <>
              {item.isDefault && <span aria-hidden>★</span>}
              <span>{backendLabel(item.backend)}</span>
              {!item.deployed && item.status === "probable" && <span aria-hidden>?</span>}
            </>
          )
          return (
            <Tooltip.Root key={item.backend}>
              <Tooltip.Trigger asChild>
                {clickable ? (
                  <button
                    type="button"
                    role="listitem"
                    aria-label={label}
                    data-backend={item.backend}
                    data-state-kind={itemState(item)}
                    onClick={() => onSelect?.(item)}
                    className={`${itemClass(item)} cursor-pointer hover:brightness-125`}
                  >
                    {content}
                  </button>
                ) : (
                  <span
                    role="listitem"
                    aria-label={label}
                    data-backend={item.backend}
                    data-state-kind={itemState(item)}
                    className={itemClass(item)}
                  >
                    {content}
                  </span>
                )}
              </Tooltip.Trigger>
              <Tooltip.Portal>
                <Tooltip.Content
                  className="z-50 max-w-xs rounded-md border border-border bg-popover px-3 py-1.5 text-xs shadow-md"
                  sideOffset={4}
                >
                  {label}
                  <Tooltip.Arrow className="fill-border" />
                </Tooltip.Content>
              </Tooltip.Portal>
            </Tooltip.Root>
          )
        })}
      </div>
    </Tooltip.Provider>
  )
}
