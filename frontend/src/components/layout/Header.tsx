import { useLocation, useNavigate } from "react-router-dom"
import { LogOut, Menu, Search, X } from "lucide-react"
import { useAuthStore } from "@/stores/authStore"
import { api } from "@/api/client"

interface HeaderProps {
  sidebarOpen: boolean
  onToggleSidebar: () => void
  connected: boolean
}

const PAGE_TITLES: Record<string, string> = {
  "/dashboard": "Dashboard",
  "/models": "Models",
  "/engines": "Engines",
  "/explore": "Explore",
  "/playground": "Playground",
  "/stats": "Stats",
  "/logs": "Logs",
  "/api-keys": "API Keys",
  "/settings": "Settings",
  "/users": "Usuarios",
  "/groups": "Grupos",
}

export function Header({ sidebarOpen, onToggleSidebar, connected }: HeaderProps) {
  const location = useLocation()
  const navigate = useNavigate()
  const user = useAuthStore((s) => s.user)
  const logout = useAuthStore((s) => s.logout)

  const pageTitle = PAGE_TITLES[location.pathname] ?? ""

  async function handleLogout() {
    try {
      await api.auth.logout()
    } catch {
      // session may already be gone
    }
    logout()
    navigate("/login", { replace: true })
  }

  return (
    <header className="mb-4 flex h-10 items-center justify-between gap-4">
      {/* Left: mobile toggle + page title */}
      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={onToggleSidebar}
          className="inline-flex items-center gap-2 rounded-md border border-border px-2 py-1.5 text-sm text-muted-foreground hover:bg-muted lg:hidden"
          aria-label="Toggle sidebar"
        >
          {sidebarOpen ? <X size={16} /> : <Menu size={16} />}
        </button>
        <h1 className="text-lg font-semibold text-foreground hidden sm:block">{pageTitle}</h1>
      </div>

      {/* Center: search placeholder (desktop only) */}
      <div className="hidden md:flex flex-1 max-w-md mx-4">
        <div className="relative w-full">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
          <input
            type="text"
            placeholder="Buscar modelos, logs..."
            className="w-full rounded-md border border-border bg-background pl-9 pr-3 py-1.5 text-sm text-foreground placeholder:text-muted-foreground/60 focus:outline-none focus:ring-2 focus:ring-primary/50"
            readOnly
          />
        </div>
      </div>

      {/* Right: connection status + user */}
      <div className="flex items-center gap-3">
        <span
          role="status"
          aria-live="polite"
          className={`hidden sm:inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium ${
            connected
              ? "bg-emerald-500/15 text-emerald-300"
              : "bg-amber-500/15 text-amber-300"
          }`}
        >
          <span className={`h-1.5 w-1.5 rounded-full ${connected ? "bg-emerald-400" : "bg-amber-400 animate-pulse"}`} />
          {connected ? "Live" : "Reconectando"}
        </span>

        {user && (
          <div className="hidden lg:flex items-center gap-2">
            <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-primary/20 text-xs font-semibold text-primary uppercase">
              {user.username.charAt(0)}
            </div>
            <span className="text-sm text-muted-foreground">{user.username}</span>
            <button
              type="button"
              onClick={() => void handleLogout()}
              className="rounded-md p-1.5 text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
              title="Cerrar sesion"
            >
              <LogOut size={14} />
            </button>
          </div>
        )}
      </div>
    </header>
  )
}
