import { NavLink } from "react-router-dom"
import {
  LayoutDashboard,
  Boxes,
  Search,
  MessageSquare,
  BarChart2,
  Settings,
} from "lucide-react"
import { clsx } from "clsx"

interface SidebarProps {
  open: boolean
  onClose: () => void
}

const NAV_ITEMS = [
  { to: "/dashboard", label: "Dashboard",  icon: LayoutDashboard },
  { to: "/models",    label: "Models",     icon: Boxes },
  { to: "/explore",   label: "Explore",    icon: Search },
  { to: "/playground",label: "Playground", icon: MessageSquare },
  { to: "/stats",     label: "Stats",      icon: BarChart2 },
  { to: "/settings",  label: "Settings",   icon: Settings },
]

export function Sidebar({ open, onClose }: SidebarProps) {
  return (
    <aside
      className={clsx(
        "fixed inset-y-0 left-0 z-30 w-56 -translate-x-full flex-col border-r border-border bg-card transition-transform lg:static lg:translate-x-0",
        open && "translate-x-0",
      )}
    >
      {/* Logo */}
      <div className="flex items-center gap-2 px-5 py-5 border-b border-border">
        <span className="text-xl font-bold tracking-tight text-foreground">oCabra</span>
        <span className="text-xs text-muted-foreground mt-0.5">AI Server</span>
      </div>

      {/* Nav */}
      <nav className="flex-1 px-2 py-4 space-y-1">
        {NAV_ITEMS.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            onClick={onClose}
            className={({ isActive }) =>
              clsx(
                "flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-colors",
                isActive
                  ? "bg-primary/10 text-primary"
                  : "text-muted-foreground hover:bg-muted hover:text-foreground",
              )
            }
          >
            <Icon size={16} />
            {label}
          </NavLink>
        ))}
      </nav>

      {/* Version */}
      <div className="px-5 py-3 border-t border-border">
        <span className="text-xs text-muted-foreground">v0.1.0</span>
      </div>
    </aside>
  )
}
