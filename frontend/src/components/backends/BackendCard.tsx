import { useEffect, useState, type ReactNode } from "react"
import { Link } from "react-router"
import * as DropdownMenu from "@radix-ui/react-dropdown-menu"
import {
  ChevronDown,
  Download,
  FileText,
  Cpu,
  Hammer,
  Package,
  RefreshCw,
  Trash2,
  X,
} from "lucide-react"
import { BackendStatusBadge } from "@/components/backends/BackendStatusBadge"
import { BackendGlyph } from "@/components/dashboard/BackendIcon"
import { InstallProgressBar } from "@/components/backends/InstallProgressBar"
import { api } from "@/api/client"
import { formatDate, formatMb } from "@/lib/format"
import type { BackendInstallMethod, BackendModuleState, BackendPublishedInfo } from "@/types"

interface BackendCardProps {
  backend: BackendModuleState
  onInstall: (backendType: string, method: BackendInstallMethod) => void
  onCancelInstall: (backendType: string) => void
  onUninstall: (backend: BackendModuleState) => void
  onUpdate?: (backend: BackendModuleState) => void
  onViewLogs: (backend: BackendModuleState) => void
  /** Controles propios del backend (p. ej. el daemon y los ajustes de Ollama). */
  extra?: ReactNode
  /** Avisa a la página de lo que hay publicado (para el resumen de arriba). */
  onOffer?: (backendType: string, info: BackendPublishedInfo | null) => void
}

const formatSize = (mb: number | null | undefined): string =>
  mb == null || mb <= 0 ? "-" : formatMb(mb)

/** Installed vs published package versions, differences highlighted. */
function PackageVersions({
  installed,
  published,
}: {
  installed: Record<string, string>
  published: Record<string, string> | null
}) {
  const [open, setOpen] = useState(false)
  const names = Array.from(new Set([...Object.keys(installed), ...Object.keys(published ?? {})])).sort()
  if (names.length === 0) return null
  const differing = published ? names.filter((n) => installed[n] !== published[n]).length : 0
  return (
    <div className="mt-3">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
        aria-expanded={open}
        data-testid="packages-toggle"
      >
        <ChevronDown size={12} className={open ? "" : "-rotate-90"} />
        Paquetes ({names.length})
        {differing > 0 && <span className="text-amber-600 dark:text-amber-300"> · {differing} distintos del precompilado</span>}
      </button>
      {open && (
        <div className="mt-1.5 overflow-x-auto">
          <table className="w-full table-fixed text-[11px]">
            <thead>
              <tr className="text-left text-muted-foreground">
                <th className="w-[38%] font-normal">Paquete</th>
                <th className="font-normal">Instalado</th>
                {published && <th className="font-normal">Publicado</th>}
              </tr>
            </thead>
            <tbody className="font-mono">
              {names.map((n) => {
                const a = installed[n]
                const b = published?.[n]
                const differs = published != null && a !== b
                return (
                  <tr key={n} className={differs ? "text-amber-600 dark:text-amber-300" : ""}>
                    <td className="truncate pr-2" title={n}>{n}</td>
                    <td className="truncate pr-2" title={a}>{a ?? "—"}</td>
                    {published && <td className="truncate" title={b}>{b ?? "—"}</td>}
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

/** Icono grande del backend en un recuadro: el logo de marca si lo tiene
 *  (Ollama, vLLM, NVIDIA, Hugging Face) o el de su familia. */
export function BackendTile({ backendType, size = 44 }: { backendType: string; size?: number }) {
  return (
    <div
      className="flex shrink-0 items-center justify-center rounded-xl border border-border bg-muted/50 text-foreground/85"
      style={{ width: size, height: size }}
      data-backend-icon={backendType}
    >
      <BackendGlyph backendType={backendType} size={Math.round(size * 0.5)} />
    </div>
  )
}

export function BackendCard({
  backend,
  onInstall,
  onCancelInstall,
  onUninstall,
  onUpdate,
  onViewLogs,
  extra,
  onOffer,
}: BackendCardProps) {
  const [installMenuOpen, setInstallMenuOpen] = useState(false)
  const [offer, setOffer] = useState<BackendPublishedInfo | null>(null)
  const status = backend.installStatus
  const published = offer?.published ?? null

  // What the OCI repository offers, re-read whenever the install changes.
  useEffect(() => {
    if (status === "built-in" || status === "installing" || status === "uninstalling") return
    let cancelled = false
    api.backends
      .published(backend.backendType)
      .then((info) => {
        if (cancelled) return
        setOffer(info)
        onOffer?.(backend.backendType, info)
      })
      .catch(() => !cancelled && setOffer(null))
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [backend.backendType, status, backend.ociDigest])

  const displaySize =
    status === "installed" || status === "uninstalling"
      ? formatSize(backend.actualSizeMb ?? backend.estimatedSizeMb)
      : `~${formatSize(backend.estimatedSizeMb)}`

  return (
    <div className="flex h-full flex-col rounded-lg border border-border bg-card p-5" data-backend-card={backend.backendType} id={`backend-${backend.backendType}`}>
      {/* Header */}
      <div className="flex items-start justify-between gap-3">
        <div className="flex min-w-0 flex-1 items-center gap-3">
          <BackendTile backendType={backend.backendType} />
          <div className="min-w-0">
            <h3 className="truncate text-lg font-semibold text-foreground">{backend.displayName}</h3>
            <p className="mt-0.5 font-mono text-xs text-muted-foreground">{backend.backendType}</p>
          </div>
        </div>
        <BackendStatusBadge
          status={status}
          progressPct={status === "installing" ? backend.installProgress : null}
        />
      </div>

      {/* Description */}
      <p className="mt-3 text-sm text-muted-foreground">
        {backend.description || "Sin descripcion disponible."}
      </p>

      {/* Tags */}
      {backend.tags.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-1.5">
          {backend.tags.map((tag) => (
            <span
              key={tag}
              className="rounded border border-border bg-muted/40 px-1.5 py-0.5 text-[10px] font-medium uppercase text-muted-foreground"
            >
              {tag}
            </span>
          ))}
        </div>
      )}

      {/* Meta info */}
      <dl className="mt-4 grid grid-cols-2 gap-x-3 gap-y-1.5 text-xs">
        <div className="flex justify-between">
          <dt className="text-muted-foreground">Tamaño</dt>
          <dd className="font-medium">{displaySize}</dd>
        </div>
        {backend.installedVersion && (
          <div className="flex justify-between">
            <dt className="text-muted-foreground">Versión</dt>
            <dd className="truncate font-medium">{backend.installedVersion}</dd>
          </div>
        )}
        {backend.installedAt && (
          <div className="flex justify-between">
            <dt className="text-muted-foreground">Instalado</dt>
            <dd className="font-medium">{formatDate(backend.installedAt)}</dd>
          </div>
        )}
        {backend.installSource && (
          <div className="flex justify-between">
            <dt className="text-muted-foreground">Fuente</dt>
            <dd className="font-medium">
              {{ oci: "precompilado", source: "código", "built-in": "integrado" }[backend.installSource] ??
                backend.installSource}
            </dd>
          </div>
        )}
        {backend.gitRef && (
          <div className="flex justify-between">
            <dt className="text-muted-foreground">Rama/commit</dt>
            <dd className="truncate font-mono font-medium">{backend.gitRef}</dd>
          </div>
        )}
        <div className="flex justify-between">
          <dt className="text-muted-foreground">Modelos cargados</dt>
          <dd className="font-medium">{backend.modelsLoaded}</dd>
        </div>
      </dl>

      {/* Prebuilt image on offer */}
      {published && status !== "built-in" && (
        <div className="mt-3 flex items-center gap-1.5 rounded-md border border-border bg-muted/30 px-2 py-1.5 text-xs text-muted-foreground">
          <Package size={12} />
          <span>
            Precompilado publicado {published.created ? formatDate(published.created) : ""} ·{" "}
            {formatSize(published.sizeBytes / 1048576)}
            {published.gpuArchs ? ` · GPUs sm ${published.gpuArchs}` : ""}
          </span>
          {offer?.updateAvailable && (
            <span className="ml-auto font-medium text-emerald-600 dark:text-emerald-300">
              actualización disponible
            </span>
          )}
        </div>
      )}
      {(Object.keys(backend.packages).length > 0 || published) && status === "installed" && (
        <PackageVersions
          installed={backend.packages}
          published={published && Object.keys(published.packages).length ? published.packages : null}
        />
      )}

      {/* Error banner */}
      {backend.error && status !== "installing" && (
        <div className="mt-3 rounded-md border border-red-500/40 bg-red-500/10 p-2 text-xs text-red-100">
          <span className="font-semibold">Error:</span> {backend.error}
        </div>
      )}

      {/* Progress bar during install */}
      {status === "installing" && (
        <div className="mt-4">
          <InstallProgressBar
            progress={backend.installProgress}
            detail={backend.installDetail}
          />
        </div>
      )}

      {extra && <div className="mt-4 space-y-4 border-t border-border pt-4">{extra}</div>}

      <div className="flex-1" />

      {/* Action buttons */}
      <div className="mt-4 flex flex-wrap items-center gap-2">
        {status === "not_installed" && (
          <DropdownMenu.Root open={installMenuOpen} onOpenChange={setInstallMenuOpen}>
            <DropdownMenu.Trigger asChild>
              <button
                type="button"
                className="inline-flex items-center gap-1.5 rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground hover:bg-primary/90"
              >
                <Download size={14} />
                Instalar
                <ChevronDown size={14} />
              </button>
            </DropdownMenu.Trigger>
            <DropdownMenu.Portal>
              <DropdownMenu.Content
                className="z-50 min-w-[180px] rounded-md border border-border bg-card p-1 shadow-md"
                sideOffset={4}
                align="start"
              >
                <DropdownMenu.Item
                  disabled={!published}
                  className="flex cursor-pointer items-center gap-2 rounded-sm px-2 py-1.5 text-sm outline-none hover:bg-muted focus:bg-muted data-[disabled]:cursor-not-allowed data-[disabled]:opacity-50"
                  onSelect={() => onInstall(backend.backendType, "oci")}
                >
                  <Download size={14} />
                  {published ? "Precompilado (recomendado)" : "Precompilado (no publicado)"}
                </DropdownMenu.Item>
                <DropdownMenu.Item
                  className="flex cursor-pointer items-center gap-2 rounded-sm px-2 py-1.5 text-sm outline-none hover:bg-muted focus:bg-muted"
                  onSelect={() => onInstall(backend.backendType, "source")}
                >
                  <Hammer size={14} />
                  Desde el código
                </DropdownMenu.Item>
              </DropdownMenu.Content>
            </DropdownMenu.Portal>
          </DropdownMenu.Root>
        )}

        {status === "installing" && (
          <button
            type="button"
            onClick={() => onCancelInstall(backend.backendType)}
            className="inline-flex items-center gap-1.5 rounded-md border border-border px-3 py-1.5 text-sm text-muted-foreground hover:bg-muted hover:text-foreground"
          >
            <X size={14} />
            Cancelar
          </button>
        )}

        {status === "installed" && (
          <>
            <button
              type="button"
              onClick={() => onUninstall(backend)}
              className="inline-flex items-center gap-1.5 rounded-md border border-red-500/40 px-3 py-1.5 text-sm text-red-200 hover:bg-red-500/10"
            >
              <Trash2 size={14} />
              Desinstalar
            </button>
            <button
              type="button"
              onClick={() => onViewLogs(backend)}
              className="inline-flex items-center gap-1.5 rounded-md border border-border px-3 py-1.5 text-sm text-muted-foreground hover:bg-muted hover:text-foreground"
            >
              <FileText size={14} />
              Ver logs
            </button>
            {backend.backendType === "tensorrt_llm" && (
              <Link
                to="/engines"
                className="inline-flex items-center gap-1.5 rounded-md border border-border px-3 py-1.5 text-sm text-muted-foreground hover:bg-muted hover:text-foreground"
                data-testid="trt-engines-link"
              >
                <Cpu size={14} />
                Engines compilados
              </Link>
            )}
            {(offer?.updateAvailable || backend.hasUpdate) && onUpdate && (
              <button
                type="button"
                onClick={() => onUpdate(backend)}
                className="inline-flex items-center gap-1.5 rounded-md border border-emerald-500/40 bg-emerald-500/10 px-3 py-1.5 text-sm text-emerald-100 hover:bg-emerald-500/20"
              >
                <RefreshCw size={14} />
                Actualizar
              </button>
            )}
          </>
        )}

        {status === "error" && (
          <>
            <button
              type="button"
              onClick={() => onInstall(backend.backendType, "auto")}
              className="inline-flex items-center gap-1.5 rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-1.5 text-sm text-amber-100 hover:bg-amber-500/20"
            >
              <RefreshCw size={14} />
              Reintentar
            </button>
            <button
              type="button"
              onClick={() => onViewLogs(backend)}
              className="inline-flex items-center gap-1.5 rounded-md border border-border px-3 py-1.5 text-sm text-muted-foreground hover:bg-muted hover:text-foreground"
            >
              <FileText size={14} />
              Ver log
            </button>
          </>
        )}

        {status === "uninstalling" && (
          <span className="text-xs text-muted-foreground">Desinstalando...</span>
        )}

        {status === "built-in" && (
          <span className="text-xs text-muted-foreground">
            Backend integrado · siempre disponible
          </span>
        )}
      </div>
    </div>
  )
}
