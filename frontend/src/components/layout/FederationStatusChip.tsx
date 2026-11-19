import { useEffect, useState } from "react"
import { useNavigate } from "react-router"
import { Cloud, Network } from "lucide-react"
import { api } from "@/api/client"
import { useAuthStore } from "@/stores/authStore"
import { getRemoteNode } from "@/lib/remoteNode"
import type { CloudNode, FederationPeer } from "@/types"

const POLL_MS = 30_000

/** Minutes until a running cloud node shuts itself down (TTL or idle), if known. */
export function minutesLeft(node: CloudNode, now = Date.now()): number | null {
  const ends: number[] = []
  if (node.terminates_at) ends.push(new Date(node.terminates_at).getTime())
  if (node.idle_shutdown_minutes) {
    const since = node.last_active_at ?? node.started_at ?? node.created_at
    if (since) ends.push(new Date(since).getTime() + node.idle_shutdown_minutes * 60_000)
  }
  if (!ends.length) return null
  return Math.max(0, Math.round((Math.min(...ends) - now) / 60_000))
}

/**
 * Federation at a glance in the header: nodes online and, when cloud nodes
 * are running, what they cost per hour and when the first one shuts down —
 * a paid pod used to be visible only on the Federation page.
 */
export function FederationStatusChip() {
  const isAdmin = useAuthStore((s) => s.hasRole("system_admin"))
  const navigate = useNavigate()
  const [peers, setPeers] = useState<FederationPeer[] | null>(null)
  const [nodes, setNodes] = useState<CloudNode[]>([])

  useEffect(() => {
    if (!isAdmin || getRemoteNode()) return
    let cancelled = false
    const load = async () => {
      const [p, n] = await Promise.all([
        api.federation.getPeers().catch(() => null),
        api.federation.cloud.nodes().catch(() => [] as CloudNode[]),
      ])
      if (!cancelled) {
        setPeers(p)
        setNodes(n)
      }
    }
    void load()
    const id = window.setInterval(() => void load(), POLL_MS)
    return () => {
      cancelled = true
      window.clearInterval(id)
    }
  }, [isAdmin])

  const enabled = (peers ?? []).filter((p) => p.enabled)
  if (!isAdmin || peers == null || enabled.length === 0) return null
  const online = enabled.filter((p) => p.online).length
  const running = nodes.filter((n) => n.status !== "terminated" && n.status !== "error")
  const perHour = running.reduce((acc, n) => acc + (n.cost_per_hr ?? 0), 0)
  const left = running.map((n) => minutesLeft(n)).filter((m): m is number => m != null)
  const soonest = left.length ? Math.min(...left) : null

  return (
    <button
      type="button"
      onClick={() => navigate("/federation")}
      className="inline-flex items-center gap-1.5 rounded-md border border-border px-2 py-1 text-xs text-muted-foreground hover:bg-muted hover:text-foreground"
      title="Federación: nodos en línea y nodos en la nube en marcha"
    >
      <Network size={13} className={online ? "text-emerald-500" : "text-red-500"} />
      <span>
        {online}/{enabled.length} nodos
      </span>
      {running.length > 0 && (
        <span className="inline-flex items-center gap-1 border-l border-border pl-1.5 text-amber-700 dark:text-amber-300">
          <Cloud size={13} />
          {running.length} en la nube · ${perHour.toFixed(2)}/h
          {soonest != null && ` · se apaga en ${soonest < 60 ? `${soonest} min` : `${Math.floor(soonest / 60)} h ${soonest % 60} min`}`}
        </span>
      )}
    </button>
  )
}
