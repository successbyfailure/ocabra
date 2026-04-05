import { Menu, X } from "lucide-react"

interface HeaderProps {
  sidebarOpen: boolean
  onToggleSidebar: () => void
}

export function Header({ sidebarOpen, onToggleSidebar }: HeaderProps) {
  return (
    <div className="mb-4 flex lg:hidden">
      <button
        type="button"
        onClick={onToggleSidebar}
        className="inline-flex items-center gap-2 rounded-md border border-border px-3 py-1.5 text-sm text-muted-foreground hover:bg-muted"
        aria-label="Toggle sidebar"
      >
        {sidebarOpen ? <X size={16} /> : <Menu size={16} />}
        Menu
      </button>
    </div>
  )
}
