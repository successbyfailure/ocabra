import { useState } from "react"
import * as DropdownMenu from "@radix-ui/react-dropdown-menu"
import {
  ChevronDown,
  Download,
  FileText,
  Hammer,
  RefreshCw,
  Trash2,
  X,
} from "lucide-react"
import { BackendStatusBadge } from "@/components/backends/BackendStatusBadge"
import { InstallProgressBar } from "@/components/backends/InstallProgressBar"
import type { BackendInstallMethod, BackendModuleState } from "@/types"

interface BackendCardProps {
  backend: BackendModuleState
  onInstall: (backendType: string, method: BackendInstallMethod) => void
  onCancelInstall: (backendType: string) => void
  onUninstall: (backend: BackendModuleState) => void
  onUpdate?: (backend: BackendModuleState) => void
  onViewLogs: (backend: BackendModuleState) => void
}

function formatSize(mb: number | null | undefined): string {
  if (mb == null || mb <= 0) return "-"
  if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`
  return `${Math.round(mb)} MB`
}

function formatDate(iso: string | null): string {
  if (!iso) return "-"
  try {
    return new Date(iso).toLocaleDateString()
  } catch {
    return iso
  }
}

export function BackendCard({
  backend,
  onInstall,
  onCancelInstall,
  onUninstall,
  onUpdate,
  onViewLogs,
}: BackendCardProps) {
  const [installMenuOpen, setInstallMenuOpen] = useState(false)
  const status = backend.installStatus

  const displaySize =
    status === "installed" || status === "uninstalling"
      ? formatSize(backend.actualSizeMb ?? backend.estimatedSizeMb)
      : `~${formatSize(backend.estimatedSizeMb)}`

  return (
    <div className="flex h-full flex-col rounded-lg border border-border bg-card p-5">
      {/* Header */}
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <h3 className="truncate text-lg font-semibold text-foreground">
            {backend.displayName}
          </h3>
          <p className="mt-0.5 text-xs text-muted-foreground">{backend.backendType}</p>
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
            <dd className="font-medium">{backend.installSource}</dd>
          </div>
        )}
        <div className="flex justify-between">
          <dt className="text-muted-foreground">Modelos cargados</dt>
          <dd className="font-medium">{backend.modelsLoaded}</dd>
        </div>
      </dl>

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
                  className="flex cursor-pointer items-center gap-2 rounded-sm px-2 py-1.5 text-sm outline-none hover:bg-muted focus:bg-muted"
                  onSelect={() => onInstall(backend.backendType, "oci")}
                >
                  <Download size={14} />
                  Desde OCI (pre-built)
                </DropdownMenu.Item>
                <DropdownMenu.Item
                  className="flex cursor-pointer items-center gap-2 rounded-sm px-2 py-1.5 text-sm outline-none hover:bg-muted focus:bg-muted"
                  onSelect={() => onInstall(backend.backendType, "source")}
                >
                  <Hammer size={14} />
                  Desde source
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
            {backend.hasUpdate && onUpdate && (
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
              onClick={() => onInstall(backend.backendType, "oci")}
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
