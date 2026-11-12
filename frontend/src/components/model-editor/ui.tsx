import type { ReactNode } from "react"
import { RotateCcw } from "lucide-react"
import { cn } from "@/lib/cn"

/** Bloque de una sección: título, subtítulo opcional y contenido. */
export function Panel({
  title,
  subtitle,
  actions,
  children,
  className,
}: {
  title?: ReactNode
  subtitle?: ReactNode
  actions?: ReactNode
  children: ReactNode
  className?: string
}) {
  return (
    <section className={cn("rounded-2xl border border-border bg-card p-4 shadow-sm", className)}>
      {(title || actions) && (
        <header className="mb-3 flex flex-wrap items-start justify-between gap-2">
          <div>
            {title && <h3 className="text-sm font-semibold text-foreground">{title}</h3>}
            {subtitle && <p className="mt-0.5 text-xs text-muted-foreground">{subtitle}</p>}
          </div>
          {actions && <div className="flex items-center gap-2">{actions}</div>}
        </header>
      )}
      {children}
    </section>
  )
}

export interface SegmentOption<T> {
  value: T
  label: ReactNode
  title?: string
  disabled?: boolean
}

/** Botones excluyentes. */
export function Segmented<T extends string | number | boolean | null>({
  value,
  options,
  onChange,
  disabled,
  size = "md",
  testId,
}: {
  value: T
  options: SegmentOption<T>[]
  onChange: (value: T) => void
  disabled?: boolean
  size?: "sm" | "md"
  testId?: string
}) {
  return (
    <div className="inline-flex flex-wrap gap-1 rounded-lg border border-border bg-muted/30 p-0.5" data-testid={testId}>
      {options.map((opt) => {
        const active = opt.value === value
        return (
          <button
            key={String(opt.value)}
            type="button"
            title={opt.title}
            disabled={disabled || opt.disabled}
            aria-pressed={active}
            onClick={() => onChange(opt.value)}
            className={cn(
              "rounded-md font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40",
              size === "sm" ? "px-2 py-0.5 text-[11px]" : "px-2.5 py-1 text-xs",
              active ? "bg-primary text-primary-foreground shadow-sm" : "text-muted-foreground hover:bg-muted hover:text-foreground",
            )}
          >
            {opt.label}
          </button>
        )
      })}
    </div>
  )
}

/** Indicador de herencia: el valor sale del global o está fijado aquí. */
export function InheritBadge({ set, onReset, disabled }: { set: boolean; onReset?: () => void; disabled?: boolean }) {
  if (!set) {
    return (
      <span className="rounded-full border border-border px-1.5 py-px text-[10px] text-muted-foreground" title="No está fijado: se usa el valor por defecto">
        por defecto
      </span>
    )
  }
  return (
    <span className="inline-flex items-center gap-1 rounded-full border border-sky-500/40 bg-sky-500/10 px-1.5 py-px text-[10px] text-sky-300">
      fijado
      {onReset && !disabled && (
        <button type="button" onClick={onReset} title="Volver al valor por defecto" className="hover:text-sky-100">
          <RotateCcw size={10} />
        </button>
      )}
    </span>
  )
}

export function Hint({ children, tone = "muted" }: { children: ReactNode; tone?: "muted" | "warn" | "error" | "info" }) {
  const cls = {
    muted: "text-muted-foreground",
    warn: "text-amber-300",
    error: "text-red-300",
    info: "text-sky-300",
  }[tone]
  return <p className={cn("text-[11px] leading-snug", cls)}>{children}</p>
}

export function Callout({ children, tone = "warn" }: { children: ReactNode; tone?: "warn" | "error" | "info" | "ok" }) {
  const cls = {
    warn: "border-amber-500/40 bg-amber-500/10 text-amber-200",
    error: "border-red-500/40 bg-red-500/10 text-red-200",
    info: "border-sky-500/40 bg-sky-500/10 text-sky-200",
    ok: "border-emerald-500/40 bg-emerald-500/10 text-emerald-200",
  }[tone]
  return <div className={cn("rounded-lg border px-3 py-2 text-xs", cls)}>{children}</div>
}

export function Chip({ children, tone = "muted", title }: { children: ReactNode; tone?: "muted" | "ok" | "warn" | "error" | "info"; title?: string }) {
  const cls = {
    muted: "border-border bg-muted/40 text-muted-foreground",
    ok: "border-emerald-500/40 bg-emerald-500/10 text-emerald-200",
    warn: "border-amber-500/40 bg-amber-500/10 text-amber-200",
    error: "border-red-500/40 bg-red-500/10 text-red-200",
    info: "border-sky-500/40 bg-sky-500/10 text-sky-200",
  }[tone]
  return (
    <span title={title} className={cn("inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] font-medium", cls)}>
      {children}
    </span>
  )
}

export const inputCls =
  "w-full rounded-md border border-border bg-background px-2 py-1 text-sm text-foreground outline-none focus:border-primary disabled:opacity-50"

export function Toggle({
  checked,
  onChange,
  disabled,
  label,
  testId,
}: {
  checked: boolean
  onChange: (value: boolean) => void
  disabled?: boolean
  label?: ReactNode
  testId?: string
}) {
  return (
    <label className={cn("inline-flex cursor-pointer items-center gap-2 text-sm", disabled && "cursor-not-allowed opacity-50")}>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        disabled={disabled}
        data-testid={testId}
        onClick={() => onChange(!checked)}
        className={cn(
          "relative h-5 w-9 rounded-full transition-colors",
          checked ? "bg-primary" : "bg-muted-foreground/30",
        )}
      >
        <span className={cn("absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-all", checked ? "left-[18px]" : "left-0.5")} />
      </button>
      {label}
    </label>
  )
}
