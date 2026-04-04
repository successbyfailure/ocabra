import { NavLink, useNavigate } from "react-router-dom"
import {
  LayoutDashboard,
  Boxes,
  Cpu,
  Search,
  MessageSquare,
  BarChart2,
  Settings,
  LogOut,
} from "lucide-react"
import { clsx } from "clsx"
import { useAuthStore } from "@/stores/authStore"
import { api } from "@/api/client"

interface SidebarProps {
  open: boolean
  onClose: () => void
}

const ROLE_LABELS: Record<string, string> = {
  user: "Usuario",
  model_manager: "Manager",
  system_admin: "Administrador",
}

interface NavItem {
  to: string
  label: string
  icon: React.ElementType
  minRole?: "model_manager" | "system_admin"
}

const NAV_ITEMS: NavItem[] = [
  { to: "/dashboard", label: "Dashboard",  icon: LayoutDashboard },
  { to: "/models",    label: "Models",     icon: Boxes },
  { to: "/engines",   label: "Engines",    icon: Cpu,            minRole: "model_manager" },
  { to: "/explore",   label: "Explore",    icon: Search,         minRole: "model_manager" },
  { to: "/playground",label: "Playground", icon: MessageSquare },
  { to: "/stats",     label: "Stats",      icon: BarChart2 },
  { to: "/settings",  label: "Settings",   icon: Settings },
]

export function Sidebar({ open, onClose }: SidebarProps) {
  const user = useAuthStore((s) => s.user)
  const hasRole = useAuthStore((s) => s.hasRole)
  const logout = useAuthStore((s) => s.logout)
  const navigate = useNavigate()

  const visibleItems = NAV_ITEMS.filter(
    (item) => !item.minRole || hasRole(item.minRole),
  )

  async function handleLogout() {
    try {
      await api.auth.logout()
    } catch {
      // Ignore errors — session may already be gone
    }
    logout()
    navigate("/login", { replace: true })
  }

  return (
    <aside
      className={clsx(
        "fixed inset-y-0 left-0 z-30 w-56 flex flex-col border-r border-border bg-card transition-transform lg:static lg:translate-x-0",
        open ? "translate-x-0" : "-translate-x-full",
      )}
    >
      {/* Logo */}
      <div className="flex items-center gap-2 px-5 py-5 border-b border-border">
        <span className="text-xl font-bold tracking-tight text-foreground">oCabra</span>
        <span className="text-xs text-muted-foreground mt-0.5">AI Server</span>
      </div>

      {/* Nav */}
      <nav className="flex-1 px-2 py-4 space-y-1">
        {visibleItems.map(({ to, label, icon: Icon }) => (
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

      {/* User info + logout */}
      <div className="border-t border-border px-4 py-3 space-y-2">
        {user && (
          <div className="flex items-center gap-3">
            {/* Avatar initial */}
            <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary/20 text-sm font-semibold text-primary uppercase">
              {user.username.charAt(0)}
            </div>
            <div className="min-w-0 flex-1">
              <p className="truncate text-sm font-medium text-foreground">{user.username}</p>
              <p className="truncate text-xs text-muted-foreground">
                {ROLE_LABELS[user.role] ?? user.role}
              </p>
            </div>
          </div>
        )}
        <button
          type="button"
          onClick={handleLogout}
          className="flex w-full items-center gap-2 rounded-md px-3 py-2 text-sm text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
        >
          <LogOut size={14} />
          Cerrar sesión
        </button>
        <span className="block text-xs text-muted-foreground px-1">v0.1.0</span>
      </div>
    </aside>
  )
}
