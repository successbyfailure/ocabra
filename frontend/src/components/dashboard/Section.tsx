import { useState } from "react"
import { ChevronDown, ChevronRight } from "lucide-react"

/** Sección plegable del dashboard. */
export function Section({
  title,
  badge,
  defaultOpen = true,
  children,
}: {
  title: string
  badge?: React.ReactNode
  defaultOpen?: boolean
  children: React.ReactNode
}) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <section className="space-y-3">
      <button type="button" onClick={() => setOpen(!open)} className="flex w-full items-center gap-2 text-left">
        {open ? <ChevronDown size={16} className="text-muted-foreground" /> : <ChevronRight size={16} className="text-muted-foreground" />}
        <h2 className="text-lg font-semibold">{title}</h2>
        {badge}
      </button>
      {open && children}
    </section>
  )
}
