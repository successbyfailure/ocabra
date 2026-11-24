// Primitivas y ayudantes compartidos de la página de Federación (REF-18).
// Extraídos de pages/Federation.tsx para repartir la página en tarjetas y
// diálogos sin duplicar estilos ni helpers.

import * as Dialog from "@radix-ui/react-dialog"
import { toast } from "sonner"

export const inputCls = "mt-1 w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
export const btnCls =
  "inline-flex items-center gap-1.5 rounded-md border border-border px-2.5 py-1.5 text-xs hover:bg-muted disabled:opacity-50"
export const primaryBtnCls =
  "inline-flex items-center gap-1.5 rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground disabled:opacity-50"

export function timeAgo(value: string | number | null | undefined): string {
  if (value == null) return "nunca"
  const ms = typeof value === "number" ? value * 1000 : Date.parse(value)
  if (Number.isNaN(ms)) return "nunca"
  const s = Math.max(0, Math.floor((Date.now() - ms) / 1000))
  if (s < 60) return `hace ${s}s`
  const m = Math.floor(s / 60)
  if (m < 60) return `hace ${m} min`
  const h = Math.floor(m / 60)
  if (h < 24) return `hace ${h} h`
  return `hace ${Math.floor(h / 24)} d`
}

export function errMsg(err: unknown): string {
  return err instanceof Error ? err.message : "Error"
}

export async function copy(text: string) {
  try {
    await navigator.clipboard.writeText(text)
    toast.success("Copiado")
  } catch {
    toast.error("No se pudo copiar")
  }
}

export function Pill({ tone, children }: { tone: "ok" | "bad" | "warn" | "muted" | "info"; children: React.ReactNode }) {
  const cls = {
    ok: "border-emerald-500/30 bg-emerald-500/15 text-emerald-600 dark:text-emerald-300",
    bad: "border-red-500/30 bg-red-500/15 text-red-600 dark:text-red-300",
    warn: "border-amber-500/30 bg-amber-500/15 text-amber-700 dark:text-amber-300",
    info: "border-sky-500/30 bg-sky-500/15 text-sky-700 dark:text-sky-300",
    muted: "border-border bg-muted text-muted-foreground",
  }[tone]
  return (
    <span className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] font-medium ${cls}`}>
      {children}
    </span>
  )
}

export function Toggle({
  checked,
  disabled,
  onChange,
  label,
}: {
  checked: boolean
  disabled?: boolean
  onChange: (v: boolean) => void
  label: string
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className={`relative inline-flex h-5 w-9 shrink-0 rounded-full border-2 border-transparent transition-colors disabled:opacity-50 ${
        checked ? "bg-primary" : "bg-muted"
      }`}
    >
      <span
        className={`pointer-events-none inline-block h-4 w-4 rounded-full bg-white shadow transition ${
          checked ? "translate-x-4" : "translate-x-0"
        }`}
      />
    </button>
  )
}

export function ModalShell({
  open,
  onOpenChange,
  title,
  description,
  children,
}: {
  open: boolean
  onOpenChange: (v: boolean) => void
  title: string
  description?: string
  children: React.ReactNode
}) {
  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-40 bg-black/60" />
        <Dialog.Content className="fixed left-1/2 top-1/2 z-50 max-h-[90vh] w-[95vw] max-w-lg -translate-x-1/2 -translate-y-1/2 overflow-y-auto rounded-lg border border-border bg-card p-5">
          <Dialog.Title className="text-lg font-semibold">{title}</Dialog.Title>
          {description && (
            <Dialog.Description className="mt-1 text-sm text-muted-foreground">{description}</Dialog.Description>
          )}
          <div className="mt-4">{children}</div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  )
}
