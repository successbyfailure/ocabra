import { useEffect, useState } from "react"
import { Network } from "lucide-react"
import { api } from "@/api/client"
import type { FederationPeer } from "@/types"
import { Section } from "./Section"

/**
 * Peers de federación, versión básica: es todo lo que el nodo sabe de un
 * peer remoto (heartbeat, GPUs con su VRAM libre, modelos cargados y una carga
 * agregada). No hay peticiones por petición ni t/s del remoto.
 */
export function FederationSection() {
  const [peers, setPeers] = useState<FederationPeer[] | null>(null)

  useEffect(() => {
    let active = true
    const load = async () => {
      try {
        const data = await api.federation.getPeers()
        if (active) setPeers(data)
      } catch {
        if (active) setPeers([])
      }
    }
    void load()
    const id = window.setInterval(() => void load(), 30_000)
    return () => {
      active = false
      window.clearInterval(id)
    }
  }, [])

  const enabled = (peers ?? []).filter((p) => p.enabled)
  if (enabled.length === 0) return null

  return (
    <Section title="Federación" badge={<span className="ml-1 text-xs text-muted-foreground">{enabled.filter((p) => p.online).length}/{enabled.length} online</span>}>
    <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
      {enabled.map((peer) => {
        const loaded = peer.models.filter((m) => m.status === "loaded")
        return (
          <article key={peer.peer_id} className="rounded-2xl border border-border bg-card p-3.5 text-[12px] shadow-sm">
            <div className="flex items-center justify-between gap-2">
              <span className="flex min-w-0 items-center gap-1.5">
                <Network size={14} className="text-muted-foreground" />
                <span className="truncate font-semibold">{peer.name}</span>
              </span>
              <span
                className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-medium ${
                  peer.online ? "bg-emerald-500/15 text-emerald-300" : "bg-red-500/15 text-red-300"
                }`}
              >
                <span className={`h-1.5 w-1.5 rounded-full ${peer.online ? "bg-emerald-400" : "bg-red-400"}`} />
                {peer.online ? `online${peer.latency_ms != null ? ` · ${Math.round(peer.latency_ms)} ms` : ""}` : "offline"}
              </span>
            </div>
            {peer.online && (
              <>
                <p className="mt-1.5 text-muted-foreground">
                  {peer.load.active_requests} peticiones en curso · uso GPU medio {peer.load.gpu_utilization_avg_pct.toFixed(0)}%
                </p>
                <div className="mt-2 space-y-1">
                  {peer.gpus.map((g) => {
                    const used = g.total_vram_mb - g.free_vram_mb
                    const pct = g.total_vram_mb > 0 ? (used / g.total_vram_mb) * 100 : 0
                    return (
                      <div key={g.index} className="grid grid-cols-[1fr_auto] items-center gap-2">
                        <div className="h-1.5 overflow-hidden rounded-full bg-muted" title={`${g.name}: ${(used / 1024).toFixed(1)} de ${(g.total_vram_mb / 1024).toFixed(1)} GB`}>
                          <span className="block h-full bg-emerald-500" style={{ width: `${pct}%` }} />
                        </div>
                        <span className="truncate text-[11px] text-muted-foreground">{g.name}</span>
                      </div>
                    )
                  })}
                </div>
                <p className="mt-2 truncate text-muted-foreground" title={loaded.map((m) => m.model_id).join(", ")}>
                  {loaded.length} modelo{loaded.length === 1 ? "" : "s"} cargado{loaded.length === 1 ? "" : "s"}
                  {loaded.length > 0 ? `: ${loaded.slice(0, 3).map((m) => m.model_id).join(", ")}${loaded.length > 3 ? "…" : ""}` : ""}
                </p>
              </>
            )}
          </article>
        )
      })}
    </div>
    </Section>
  )
}
