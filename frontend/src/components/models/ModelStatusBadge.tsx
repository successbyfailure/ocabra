import type { RuntimePhase } from "@/types"

interface ModelStatusBadgeProps {
  status: RuntimePhase
}

const STATUS_LABELS: Record<RuntimePhase, string> = {
  discovered: "Discovered",
  configured: "Configured",
  loading: "Loading",
  loaded: "Loaded",
  unloading: "Unloading",
  unloaded: "Unloaded",
  error: "Error",
  upsizing: "Ampliando",
  downsizing: "Reduciendo",
}

export function ModelStatusBadge({ status }: ModelStatusBadgeProps) {
  let classes = "border-slate-500/40 bg-slate-500/20 text-slate-200"

  if (status === "loaded") {
    classes = "border-emerald-500/40 bg-emerald-500/20 text-emerald-200"
  } else if (
    status === "loading" ||
    status === "unloading" ||
    status === "upsizing" ||
    status === "downsizing"
  ) {
    classes = "border-amber-500/40 bg-amber-500/20 text-amber-100 animate-pulse"
  } else if (status === "error") {
    classes = "border-red-500/40 bg-red-500/20 text-red-100"
  }

  return (
    <span className={`inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-medium ${classes}`}>
      {STATUS_LABELS[status]}
    </span>
  )
}
