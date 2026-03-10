import { Menu, X } from "lucide-react"

interface HeaderProps {
  sidebarOpen: boolean
  onToggleSidebar: () => void
}

export function Header({ sidebarOpen, onToggleSidebar }: HeaderProps) {
  return (
    <header className="mb-6 flex items-center justify-between border-b border-border pb-4">
      <div>
        <h1 className="text-lg font-semibold text-foreground">oCabra Control Plane</h1>
        <p className="text-sm text-muted-foreground">Multi-GPU model orchestration</p>
      </div>

      <button
        type="button"
        onClick={onToggleSidebar}
        className="inline-flex items-center gap-2 rounded-md border border-border px-3 py-2 text-sm text-muted-foreground hover:bg-muted lg:hidden"
        aria-label="Toggle sidebar"
      >
        {sidebarOpen ? <X size={16} /> : <Menu size={16} />}
        Menu
      </button>
    </header>
  )
}
