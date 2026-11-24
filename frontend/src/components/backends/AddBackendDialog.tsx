import { useEffect, useState } from "react"
import * as Dialog from "@radix-ui/react-dialog"
import { Download, Hammer, Search, X } from "lucide-react"
import { api } from "@/api/client"
import { BackendTile } from "@/components/backends/BackendCard"
import { formatMb } from "@/lib/format"
import type { BackendInstallMethod, BackendModuleState, BackendPublishedInfo } from "@/types"

const formatSize = (mb: number | null | undefined): string =>
  mb == null || mb <= 0 ? "?" : formatMb(mb)

/** Catálogo de backends que se pueden instalar: los que aún no están. Cada
 *  uno dice qué hace, cuánto ocupa y si hay precompilado publicado (lo
 *  recomendado: tarda minutos en vez de compilar). */
export function AddBackendDialog({
  open,
  onOpenChange,
  available,
  onInstall,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  available: BackendModuleState[]
  onInstall: (backendType: string, method: BackendInstallMethod) => void
}) {
  const [query, setQuery] = useState("")
  const q = query.trim().toLowerCase()
  const shown = q
    ? available.filter((b) =>
        [b.displayName, b.backendType, b.description, ...b.tags].some((t) => t?.toLowerCase().includes(q)),
      )
    : available
  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-40 bg-black/60" />
        <Dialog.Content
          className="fixed left-1/2 top-1/2 z-50 flex max-h-[85vh] w-[95vw] max-w-3xl -translate-x-1/2 -translate-y-1/2 flex-col rounded-lg border border-border bg-card p-5"
          data-testid="add-backend-dialog"
        >
          <div className="flex items-start justify-between gap-3">
            <div>
              <Dialog.Title className="text-lg font-semibold">Añadir backend</Dialog.Title>
              <Dialog.Description className="mt-1 text-sm text-muted-foreground">
                Los que aún no están instalados. El precompilado es lo recomendado: se descarga en minutos; desde el código hay
                que compilar.
              </Dialog.Description>
            </div>
            <Dialog.Close className="rounded-md p-1 text-muted-foreground hover:bg-muted" aria-label="Cerrar">
              <X size={16} />
            </Dialog.Close>
          </div>
          <div className="relative mt-3">
            <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Buscar por nombre, uso o etiqueta…"
              className="w-full rounded-md border border-border bg-background py-1.5 pl-9 pr-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
            />
          </div>
          <div className="mt-3 flex-1 space-y-2 overflow-auto pr-1">
            {available.length === 0 ? (
              <p className="py-8 text-center text-sm text-muted-foreground">Están todos instalados.</p>
            ) : shown.length === 0 ? (
              <p className="py-8 text-center text-sm text-muted-foreground">Ninguno coincide con «{query}».</p>
            ) : (
              shown.map((b) => (
                <CatalogRow
                  key={b.backendType}
                  backend={b}
                  open={open}
                  onInstall={(method) => {
                    onInstall(b.backendType, method)
                    onOpenChange(false)
                  }}
                />
              ))
            )}
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  )
}

function CatalogRow({
  backend,
  open,
  onInstall,
}: {
  backend: BackendModuleState
  open: boolean
  onInstall: (method: BackendInstallMethod) => void
}) {
  const [offer, setOffer] = useState<BackendPublishedInfo | null | undefined>(undefined)
  useEffect(() => {
    if (!open) return
    let cancelled = false
    api.backends
      .published(backend.backendType)
      .then((info) => !cancelled && setOffer(info))
      .catch(() => !cancelled && setOffer(null))
    return () => {
      cancelled = true
    }
  }, [backend.backendType, open])
  const published = offer?.published ?? null
  return (
    <div className="flex flex-wrap items-center gap-3 rounded-lg border border-border p-3" data-catalog={backend.backendType}>
      <BackendTile backendType={backend.backendType} size={40} />
      <div className="min-w-0 flex-[1_1_260px]">
        <div className="flex flex-wrap items-baseline gap-x-2">
          <span className="font-semibold">{backend.displayName}</span>
          <span className="font-mono text-[11px] text-muted-foreground">{backend.backendType}</span>
        </div>
        <p className="text-xs text-muted-foreground">{backend.description || "Sin descripción."}</p>
        <div className="mt-1 flex flex-wrap items-center gap-1.5 text-[10.5px] text-muted-foreground">
          {backend.tags.map((t) => (
            <span key={t} className="rounded border border-border bg-muted/40 px-1.5 py-0.5 font-medium uppercase">
              {t}
            </span>
          ))}
          <span>~{formatSize(backend.estimatedSizeMb)}</span>
          {offer === undefined ? <span>· buscando precompilado…</span> : published ? <span>· precompilado disponible</span> : <span>· sin precompilado</span>}
        </div>
        {backend.error && <p className="mt-1 text-[11px] text-red-300">Último intento: {backend.error}</p>}
      </div>
      <div className="flex shrink-0 gap-2">
        <button
          type="button"
          disabled={!published}
          onClick={() => onInstall("oci")}
          title={published ? "Descarga la imagen publicada" : "No hay precompilado publicado para este backend"}
          className="inline-flex items-center gap-1.5 rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-40"
        >
          <Download size={14} /> Precompilado
        </button>
        <button
          type="button"
          onClick={() => onInstall("source")}
          className="inline-flex items-center gap-1.5 rounded-md border border-border px-3 py-1.5 text-sm text-muted-foreground hover:bg-muted hover:text-foreground"
        >
          <Hammer size={14} /> Desde el código
        </button>
      </div>
    </div>
  )
}
