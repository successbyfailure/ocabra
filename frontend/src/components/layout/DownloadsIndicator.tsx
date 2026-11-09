import { useMemo } from "react"
import { AlertTriangle, Download } from "lucide-react"
import { useDownloadStore, visibleDownloadJobs } from "@/stores/downloadStore"

/**
 * Píldora de la cabecera para las descargas: sale mientras haya alguna en
 * curso, en cola o fallida en la última hora, y abre o cierra el panel de
 * descargas (`DownloadQueue`), que es quien consulta la lista, sigue el
 * progreso y permite cancelar. Sustituye a la sección "Descargas activas"
 * del dashboard y a la barra plegada que el panel dejaba flotando.
 */
export function DownloadsIndicator() {
  const jobs = useDownloadStore((s) => s.jobs)
  const open = useDownloadStore((s) => s.queueOpen)
  const setOpen = useDownloadStore((s) => s.setQueueOpen)

  const visible = useMemo(() => visibleDownloadJobs(jobs), [jobs])
  const active = visible.filter((j) => j.status === "queued" || j.status === "downloading")
  const failed = visible.filter((j) => j.status === "failed")
  if (active.length === 0 && failed.length === 0) return null

  const downloading = active.filter((j) => j.status === "downloading")
  const avg = downloading.length
    ? downloading.reduce((acc, j) => acc + (j.progressPct ?? 0), 0) / downloading.length
    : 0
  const label =
    active.length > 0
      ? `${active.length} descarga${active.length === 1 ? "" : "s"}${downloading.length > 0 ? ` · ${avg.toFixed(0)}%` : ""}`
      : `${failed.length} fallida${failed.length === 1 ? "" : "s"}`

  return (
    <button
      type="button"
      onClick={() => setOpen(!open)}
      aria-expanded={open}
      data-testid="downloads-indicator"
      title={open ? "Cerrar el panel de descargas" : "Abrir el panel de descargas"}
      className={`hidden items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium transition-colors sm:inline-flex ${
        active.length > 0
          ? "bg-blue-500/15 text-blue-300 hover:bg-blue-500/25"
          : "bg-red-500/15 text-red-300 hover:bg-red-500/25"
      }`}
    >
      {active.length > 0 ? <Download size={11} className="animate-pulse" /> : <AlertTriangle size={11} />}
      {label}
      {active.length > 0 && failed.length > 0 && <span className="text-red-300"> · {failed.length} fallida{failed.length === 1 ? "" : "s"}</span>}
    </button>
  )
}
