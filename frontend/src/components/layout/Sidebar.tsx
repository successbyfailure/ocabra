import { NavLink } from "react-router-dom"
import {
  LayoutDashboard,
  Boxes,
  Cpu,
  Search,
  MessageSquare,
  BarChart2,
  ScrollText,
  Settings,
  Users,
  UsersRound,
  Key,
  BookOpen,
  ChevronsLeft,
  ChevronsRight,
  Package,
  Sparkles,
  Plug,
} from "lucide-react"
import * as Tooltip from "@radix-ui/react-tooltip"
import { clsx } from "clsx"
import { useAuthStore } from "@/stores/authStore"

interface SidebarProps {
  open: boolean
  onClose: () => void
  collapsed: boolean
  onToggleCollapse: () => void
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
  external?: boolean
}

const NAV_ITEMS: NavItem[] = [
  { to: "/dashboard", label: "Dashboard",  icon: LayoutDashboard },
  { to: "/models",    label: "Models",     icon: Boxes },
  { to: "/agents",    label: "Agents",     icon: Sparkles,       minRole: "model_manager" },
  { to: "/mcp-servers", label: "MCP Servers", icon: Plug,        minRole: "model_manager" },
  { to: "/engines",   label: "Engines",    icon: Cpu,            minRole: "model_manager" },
  { to: "/explore",   label: "Explore",    icon: Search,         minRole: "model_manager" },
  { to: "/playground",label: "Playground", icon: MessageSquare },
  { to: "/stats",     label: "Stats",      icon: BarChart2 },
  { to: "/logs",      label: "Logs",       icon: ScrollText },
  { to: "/api-keys",  label: "API Keys",   icon: Key },
  { to: "/settings",  label: "Settings",   icon: Settings },
  { to: "/docs",      label: "API Docs",   icon: BookOpen, external: true },
]

const ADMIN_NAV_ITEMS: NavItem[] = [
  { to: "/backends", label: "Backends", icon: Package,     minRole: "system_admin" },
  { to: "/users",    label: "Usuarios", icon: Users,       minRole: "system_admin" },
  { to: "/groups",   label: "Grupos",   icon: UsersRound,  minRole: "system_admin" },
]

function NavItemLink({
  item,
  collapsed,
  onClose,
}: {
  item: NavItem
  collapsed: boolean
  onClose: () => void
}) {
  const { to, label, icon: Icon, external } = item

  const link = external ? (
    <a
      href={to}
      target="_blank"
      rel="noopener noreferrer"
      className={clsx(
        "flex items-center gap-3 rounded-md text-sm font-medium transition-colors",
        collapsed ? "justify-center px-2 py-2" : "px-3 py-2",
        "text-muted-foreground hover:bg-muted hover:text-foreground",
      )}
    >
      <Icon size={16} className="shrink-0" />
      {!collapsed && label}
    </a>
  ) : (
    <NavLink
      to={to}
      onClick={onClose}
      className={({ isActive }) =>
        clsx(
          "flex items-center gap-3 rounded-md text-sm font-medium transition-colors",
          collapsed ? "justify-center px-2 py-2" : "px-3 py-2",
          isActive
            ? "bg-primary/10 text-primary"
            : "text-muted-foreground hover:bg-muted hover:text-foreground",
        )
      }
    >
      <Icon size={16} className="shrink-0" />
      {!collapsed && label}
    </NavLink>
  )

  if (collapsed) {
    return (
      <Tooltip.Root>
        <Tooltip.Trigger asChild>{link}</Tooltip.Trigger>
        <Tooltip.Portal>
          <Tooltip.Content
            side="right"
            sideOffset={8}
            className="z-50 rounded-md border border-border bg-popover px-2.5 py-1 text-xs shadow-md"
          >
            {label}
            <Tooltip.Arrow className="fill-border" />
          </Tooltip.Content>
        </Tooltip.Portal>
      </Tooltip.Root>
    )
  }

  return link
}

export function Sidebar({ open, onClose, collapsed, onToggleCollapse }: SidebarProps) {
  const user = useAuthStore((s) => s.user)
  const hasRole = useAuthStore((s) => s.hasRole)

  const visibleItems = NAV_ITEMS.filter(
    (item) => !item.minRole || hasRole(item.minRole),
  )

  const visibleAdminItems = ADMIN_NAV_ITEMS.filter(
    (item) => !item.minRole || hasRole(item.minRole),
  )

  return (
    <Tooltip.Provider delayDuration={200}>
      <aside
        className={clsx(
          "fixed inset-y-0 left-0 z-30 flex flex-col border-r border-border bg-card transition-all duration-200 lg:static lg:translate-x-0",
          collapsed ? "w-16" : "w-56",
          open ? "translate-x-0" : "-translate-x-full",
        )}
      >
        {/* Logo */}
        <div className={clsx(
          "flex items-center border-b border-border",
          collapsed ? "justify-center px-2 py-5" : "gap-2 px-5 py-5",
        )}>
          <span className="text-xl font-bold tracking-tight text-foreground">
            {collapsed ? "oC" : "oCabra"}
          </span>
          {!collapsed && (
            <span className="text-xs text-muted-foreground mt-0.5">AI Server</span>
          )}
        </div>

        {/* Nav */}
        <nav className="flex-1 px-2 py-4 space-y-1 overflow-y-auto">
          {visibleItems.map((item) => (
            <NavItemLink key={item.to} item={item} collapsed={collapsed} onClose={onClose} />
          ))}

          {/* Admin section */}
          {visibleAdminItems.length > 0 && (
            <div className="pt-3">
              {!collapsed && (
                <p className="px-3 pb-1 text-xs font-semibold uppercase tracking-widest text-muted-foreground/60">
                  Admin
                </p>
              )}
              {visibleAdminItems.map((item) => (
                <NavItemLink key={item.to} item={item} collapsed={collapsed} onClose={onClose} />
              ))}
            </div>
          )}
        </nav>

        {/* User info (only when expanded) + collapse toggle */}
        <div className="border-t border-border px-2 py-3 space-y-2">
          {!collapsed && user && (
            <div className="flex items-center gap-3 px-2">
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

          {/* Collapse toggle (desktop only) */}
          <button
            type="button"
            onClick={onToggleCollapse}
            className={clsx(
              "hidden lg:flex w-full items-center gap-2 rounded-md px-3 py-2 text-sm text-muted-foreground transition-colors hover:bg-muted hover:text-foreground",
              collapsed && "justify-center px-2",
            )}
            title={collapsed ? "Expandir sidebar" : "Colapsar sidebar"}
          >
            {collapsed ? <ChevronsRight size={14} /> : <ChevronsLeft size={14} />}
            {!collapsed && "Colapsar"}
          </button>

          {!collapsed && (
            <span className="block text-xs text-muted-foreground px-3">v0.5.0</span>
          )}
        </div>
      </aside>
    </Tooltip.Provider>
  )
}
