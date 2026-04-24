import type { BackendInstallStatus } from "@/types"

interface BackendStatusBadgeProps {
  status: BackendInstallStatus
  progressPct?: number | null
}

const STATUS_LABELS: Record<BackendInstallStatus, string> = {
  not_installed: "No instalado",
  installing: "Instalando",
  installed: "Instalado",
  uninstalling: "Desinstalando",
  error: "Error",
  "built-in": "Integrado",
}

export function BackendStatusBadge({ status, progressPct }: BackendStatusBadgeProps) {
  let classes = "border-slate-500/40 bg-slate-500/20 text-slate-200"

  if (status === "installed") {
    classes = "border-emerald-500/40 bg-emerald-500/20 text-emerald-200"
  } else if (status === "installing") {
    classes = "border-violet-500/40 bg-violet-500/20 text-violet-200 animate-pulse"
  } else if (status === "uninstalling") {
    classes = "border-amber-500/40 bg-amber-500/20 text-amber-100 animate-pulse"
  } else if (status === "error") {
    classes = "border-red-500/40 bg-red-500/20 text-red-100"
  } else if (status === "built-in") {
    classes = "border-sky-500/40 bg-sky-500/20 text-sky-200"
  }

  let label = STATUS_LABELS[status]
  if (status === "installing" && typeof progressPct === "number") {
    label = `Instalando ${Math.round(progressPct * 100)}%`
  }

  return (
    <span
      className={`inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-medium ${classes}`}
    >
      {label}
    </span>
  )
}
