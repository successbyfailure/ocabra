import { useState, type ReactNode } from "react"
import { Header } from "./Header"
import { Sidebar } from "./Sidebar"
import { DownloadQueue } from "@/components/downloads/DownloadQueue"
import { useWebSocket } from "@/hooks/useWebSocket"
import { useLocation } from "react-router-dom"
import { ArrowLeft, MonitorSmartphone } from "lucide-react"
import { getRemoteNode, leaveRemoteNode, REMOTE_HIDDEN_ROUTES, type RemoteNode } from "@/lib/remoteNode"

interface LayoutProps {
  children: ReactNode
}

function getInitialCollapsed(): boolean {
  try {
    return localStorage.getItem("sidebar-collapsed") === "true"
  } catch {
    return false
  }
}

function RemoteBanner({ remote, connected }: { remote: RemoteNode; connected: boolean }) {
  return (
    <div className="sticky top-0 z-30 -mx-4 -mt-4 mb-4 flex flex-wrap items-center gap-3 border-b border-sky-500/40 bg-sky-500/15 px-4 py-2 text-sm lg:-mx-6 lg:-mt-6 lg:px-6">
      <MonitorSmartphone size={16} className="text-sky-700 dark:text-sky-300" />
      <span>
        Estás viendo el nodo <span className="font-semibold">{remote.name}</span>
        {!connected && <span className="ml-2 text-xs text-amber-700 dark:text-amber-300">(sin respuesta)</span>}
      </span>
      <span className="text-xs text-muted-foreground">
        Lo que hagas aquí se aplica en ese nodo, a través de la federación.
      </span>
      <button
        type="button"
        className="ml-auto inline-flex items-center gap-1.5 rounded-md border border-border bg-background px-2.5 py-1 text-xs hover:bg-muted"
        onClick={() => leaveRemoteNode()}
      >
        <ArrowLeft size={12} /> Volver a este servidor
      </button>
    </div>
  )
}

export function Layout({ children }: LayoutProps) {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [collapsed, setCollapsed] = useState(getInitialCollapsed)
  const { connected } = useWebSocket()
  const remote = getRemoteNode()
  const location = useLocation()

  const toggleCollapse = () => {
    setCollapsed((prev) => {
      const next = !prev
      try {
        localStorage.setItem("sidebar-collapsed", String(next))
      } catch {
        // ignore storage errors
      }
      return next
    })
  }

  return (
    <div className="flex h-screen overflow-hidden bg-background">
      {sidebarOpen && (
        <button
          type="button"
          className="fixed inset-0 z-20 bg-black/50 lg:hidden"
          onClick={() => setSidebarOpen(false)}
          aria-label="Close sidebar overlay"
        />
      )}

      <Sidebar
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        collapsed={collapsed}
        onToggleCollapse={toggleCollapse}
      />

      <main className="relative z-10 flex-1 overflow-y-auto p-4 lg:p-6">
        {remote && <RemoteBanner remote={remote} connected={connected} />}
        <Header
          sidebarOpen={sidebarOpen}
          onToggleSidebar={() => setSidebarOpen((isOpen) => !isOpen)}
          connected={connected}
        />
        {remote && REMOTE_HIDDEN_ROUTES.has(location.pathname) ? (
          <div className="mt-6 rounded-lg border border-dashed border-border px-6 py-10 text-center text-sm text-muted-foreground">
            Esta sección no está disponible en un nodo remoto: usuarios, grupos, claves, ajustes y
            federación los decide cada servidor.
          </div>
        ) : (
          children
        )}
      </main>
      <DownloadQueue />
    </div>
  )
}
