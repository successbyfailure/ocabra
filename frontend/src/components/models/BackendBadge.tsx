import * as Tooltip from "@radix-ui/react-tooltip"
import { useBackendsStore } from "@/stores/backendsStore"

interface BackendBadgeProps {
  backendType: string
}

const STATUS_STYLES: Record<string, { className: string; label: (st: string) => string }> = {
  installed: {
    className: "border-emerald-500/40 bg-emerald-500/15 text-emerald-200",
    label: () => "instalado",
  },
  installing: {
    className: "border-amber-500/40 bg-amber-500/15 text-amber-200",
    label: () => "instalando",
  },
  uninstalling: {
    className: "border-amber-500/40 bg-amber-500/15 text-amber-200",
    label: () => "desinstalando",
  },
  not_installed: {
    className: "border-rose-500/40 bg-rose-500/15 text-rose-200",
    label: () => "no instalado",
  },
  error: {
    className: "border-rose-500/40 bg-rose-500/15 text-rose-200",
    label: () => "error",
  },
}

const BUILTIN_STYLE = {
  className: "border-slate-500/40 bg-slate-500/15 text-slate-200",
  label: () => "built-in",
}

const UNKNOWN_STYLE = {
  className: "border-border bg-muted text-muted-foreground",
  label: () => "desconocido",
}

/**
 * Badge that shows a model's backend name plus a status pill so the user knows
 * whether unloading the backend would break this model. Reads from
 * `backendsStore` so it stays live with WebSocket updates.
 *
 * Closes Deuda #5 of bloque 15: hint to the user about which backend a model
 * uses, and surface install state at-a-glance.
 */
export function BackendBadge({ backendType }: BackendBadgeProps) {
  const backend = useBackendsStore((s) =>
    s.backends.find((b) => b.backendType === backendType),
  )

  // Built-in (always-available) backends like ollama have install_source="built-in".
  const isBuiltIn = backend?.installSource === "built-in"
  const status = backend?.installStatus
  const style = isBuiltIn
    ? BUILTIN_STYLE
    : status && STATUS_STYLES[status]
      ? STATUS_STYLES[status]
      : UNKNOWN_STYLE

  const tooltip = backend
    ? `${backend.displayName ?? backendType} · ${style.label(status ?? "")}` +
      (backend.installedVersion ? ` · v${backend.installedVersion}` : "")
    : backendType

  return (
    <Tooltip.Root>
      <Tooltip.Trigger asChild>
        <span
          className={`inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 text-xs ${style.className}`}
        >
          <span className="font-medium">{backendType}</span>
          <span className="h-1.5 w-1.5 rounded-full bg-current opacity-70" aria-hidden />
        </span>
      </Tooltip.Trigger>
      <Tooltip.Portal>
        <Tooltip.Content
          className="z-50 rounded-md border border-border bg-popover px-3 py-1.5 text-xs shadow-md"
          sideOffset={4}
        >
          {tooltip}
          <Tooltip.Arrow className="fill-border" />
        </Tooltip.Content>
      </Tooltip.Portal>
    </Tooltip.Root>
  )
}
