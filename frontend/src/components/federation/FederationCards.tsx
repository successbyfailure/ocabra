// Tarjetas de la página de Federación (REF-18): nodo activo, solicitud
// saliente y solicitud entrante. Extraídas de pages/Federation.tsx.

import { useState } from "react"
import {
  ArrowDownLeft,
  ArrowUpRight,
  Check,
  Cpu,
  Globe,
  Inbox,
  Loader2,
  MonitorSmartphone,
  Network,
  RefreshCw,
  Send,
  Settings2,
  Trash2,
  Wifi,
  WifiOff,
  X,
  Zap,
} from "lucide-react"
import { toast } from "sonner"
import { api } from "@/api/client"
import { rulesSummary } from "@/components/federation/RulesEditor"
import { btnCls, errMsg, inputCls, Pill, primaryBtnCls, timeAgo } from "@/components/federation/shared"
import { enterRemoteNode } from "@/lib/remoteNode"
import type { FederationJoinRequest, FederationPeer } from "@/types"

function connectionLabel(peer: FederationPeer): { icon: React.ReactNode; text: string } {
  if (peer.connection_mode === "tunnel_in") return { icon: <ArrowDownLeft size={11} />, text: "túnel entrante" }
  if (peer.connection_mode === "tunnel_out") return { icon: <ArrowUpRight size={11} />, text: "túnel saliente" }
  return { icon: <Globe size={11} />, text: "HTTP directo" }
}

function StatusPill({ peer }: { peer: FederationPeer }) {
  if (!peer.enabled) return <Pill tone="muted">Desactivado</Pill>
  if (peer.online)
    return (
      <Pill tone="ok">
        <Wifi size={11} /> Online
      </Pill>
    )
  if (peer.connection_mode === "tunnel_out" && peer.dial_state === "rejected")
    return <Pill tone="bad">Rechazado por el otro nodo</Pill>
  if (peer.connection_mode === "tunnel_out" && peer.dial_state && peer.dial_state !== "connected")
    return (
      <Pill tone="warn">
        <Loader2 size={11} className="animate-spin" /> Conectando
      </Pill>
    )
  if (peer.connection_mode === "tunnel_in" && !peer.tunnel_connected)
    return (
      <Pill tone="warn">
        <WifiOff size={11} /> Esperando a que conecte
      </Pill>
    )
  return (
    <Pill tone="bad">
      <WifiOff size={11} /> Offline
    </Pill>
  )
}

export function PeerCard({
  peer,
  onChanged,
  onEditRules,
}: {
  peer: FederationPeer
  onChanged: () => void
  onEditRules: (peer: FederationPeer) => void
}) {
  const [busy, setBusy] = useState<string | null>(null)
  const [editing, setEditing] = useState(false)
  const [notes, setNotes] = useState(peer.notes ?? "")
  const conn = connectionLabel(peer)
  const loaded = peer.models.filter((m) => m.status.toLowerCase() === "loaded")

  const run = async (key: string, fn: () => Promise<unknown>, ok?: string) => {
    setBusy(key)
    try {
      await fn()
      if (ok) toast.success(ok)
      onChanged()
    } catch (err) {
      toast.error(errMsg(err))
    } finally {
      setBusy(null)
    }
  }

  const test = () =>
    run("test", async () => {
      const r = await api.federation.testPeer(peer.peer_id)
      if (r.success) toast.success(`Responde en ${r.latency_ms ?? "?"} ms`)
      else toast.error(r.error ?? "No responde")
    })

  const remove = () => {
    const what =
      peer.mode === "swarm"
        ? `¿Sacar a ${peer.name} del enjambre? Dejará de usar tus modelos y tú los suyos.`
        : `¿Quitar ${peer.name}?`
    if (!window.confirm(what)) return
    void run("delete", () => api.federation.deletePeer(peer.peer_id), "Nodo quitado")
  }

  return (
    <div className="flex flex-col rounded-lg border border-border bg-card p-4">
      <div className="flex flex-wrap items-center gap-2">
        <Network size={16} className="text-muted-foreground" />
        <span className="font-semibold">{peer.name}</span>
        <StatusPill peer={peer} />
        {peer.privacy_mode === "zero_data" && (
          <span
            className="rounded-full border border-emerald-500/40 px-2 py-0.5 text-[11px] font-medium text-emerald-700 dark:text-emerald-300"
            title="Este nodo no guarda nada de las peticiones que sirve ni de quién las hace"
          >
            0 data
          </span>
        )}
        <Pill tone="muted">
          {conn.icon} {conn.text}
        </Pill>
        {peer.mode === "swarm" && <Pill tone="info">enjambre</Pill>}
        {peer.mode === "rules" && <Pill tone="info">mis reglas</Pill>}
      </div>

      {editing ? (
        <div className="mt-2 flex gap-2">
          <input className={`${inputCls} mt-0`} value={notes} onChange={(e) => setNotes(e.target.value)} placeholder="De quién es, para qué…" />
          <button
            type="button"
            className={btnCls}
            onClick={() =>
              void run("notes", () => api.federation.updatePeer(peer.peer_id, { notes: notes.trim() || null }), "Guardado").then(() =>
                setEditing(false),
              )
            }
          >
            <Check size={12} />
          </button>
          <button type="button" className={btnCls} onClick={() => setEditing(false)}>
            <X size={12} />
          </button>
        </div>
      ) : (
        <button
          type="button"
          className="mt-1 text-left text-xs text-muted-foreground hover:text-foreground"
          onClick={() => {
            setNotes(peer.notes ?? "")
            setEditing(true)
          }}
        >
          {peer.notes || "Añadir notas…"}
        </button>
      )}

      <div className="mt-3 space-y-2 text-xs">
        {(peer.mode === "swarm" || peer.mode === "rules") && peer.local_access_level && (
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-muted-foreground">Le sirvo como:</span>
            <span className="font-medium">{peer.mode === "swarm" ? "Enjambre" : "Mis reglas"}</span>
            <span className="text-muted-foreground" title="Lo que le aplicas">
              {peer.mode === "swarm" ? "puede hacerlo todo" : rulesSummary(peer.effective_rules)}
            </span>
            <button type="button" className={`${btnCls} ml-auto`} onClick={() => onEditRules(peer)}>
              <Settings2 size={12} /> Configurar
            </button>
          </div>
        )}
        {peer.their_rules && (
          <div className="text-muted-foreground" title="Lo que este nodo te aplica a ti">
            Sus reglas contigo: {rulesSummary(peer.their_rules)}
          </div>
        )}
        {peer.connection_mode === "http" && <div className="truncate text-muted-foreground">{peer.url}</div>}
        {peer.connection_mode === "tunnel_out" && <div className="truncate text-muted-foreground">→ {peer.url}</div>}
        {peer.tunnel && (
          <div className="text-muted-foreground">
            {peer.tunnel.remote_version ? `v${peer.tunnel.remote_version} · ` : ""}conectado {timeAgo(peer.tunnel.connected_at)} ·{" "}
            {peer.tunnel.requests_sent} peticiones enviadas · {peer.tunnel.requests_served} atendidas
          </div>
        )}
        {peer.last_error && !peer.online && <div className="text-red-600 dark:text-red-300">{peer.last_error}</div>}

        {peer.gpus.length > 0 && (
          <div className="space-y-1">
            {peer.gpus.map((g) => {
              const used = Math.max(0, g.total_vram_mb - g.free_vram_mb)
              const pct = g.total_vram_mb > 0 ? (used / g.total_vram_mb) * 100 : 0
              return (
                <div key={g.index} className="flex items-center gap-2">
                  <Cpu size={12} className="text-muted-foreground" />
                  <span className="w-28 truncate">{g.name}</span>
                  <div className="h-1.5 flex-1 rounded bg-muted">
                    <div className="h-1.5 rounded bg-primary" style={{ width: `${pct}%` }} />
                  </div>
                  <span className="tabular-nums text-muted-foreground">
                    {(used / 1024).toFixed(1)}/{(g.total_vram_mb / 1024).toFixed(0)} GB
                  </span>
                </div>
              )
            })}
          </div>
        )}

        <div>
          <span className="text-muted-foreground">Cargados: </span>
          {loaded.length === 0 ? (
            <span className="text-muted-foreground">ninguno</span>
          ) : (
            loaded.map((m) => (
              <span key={m.model_id} className="mr-1 inline-block rounded bg-muted px-1.5 py-0.5 font-mono text-[11px]">
                {m.model_id}
              </span>
            ))
          )}
        </div>
        {peer.online && (
          <div className="text-muted-foreground">
            {peer.load.active_requests} peticiones activas · GPU {peer.load.gpu_utilization_avg_pct.toFixed(0)} % · heartbeat{" "}
            {timeAgo(peer.last_heartbeat)}
          </div>
        )}
      </div>

      <div className="mt-auto flex flex-wrap gap-2 pt-3">
        <button type="button" className={btnCls} disabled={busy !== null} onClick={() => void test()}>
          {busy === "test" ? <Loader2 size={12} className="animate-spin" /> : <Zap size={12} />} Probar
        </button>
        <button
          type="button"
          className={btnCls}
          disabled={busy !== null}
          onClick={() =>
            void run("toggle", () => api.federation.updatePeer(peer.peer_id, { enabled: !peer.enabled }))
          }
        >
          {peer.enabled ? "Desactivar" : "Activar"}
        </button>
        {peer.access_level === "full" && peer.online && (
          <button
            type="button"
            className={btnCls}
            title="Ver y gestionar ese nodo desde aquí"
            onClick={() => enterRemoteNode({ id: peer.peer_id, name: peer.name })}
          >
            <MonitorSmartphone size={12} /> Abrir panel
          </button>
        )}
        <button type="button" className={`${btnCls} text-red-600 dark:text-red-300`} disabled={busy !== null} onClick={remove}>
          <Trash2 size={12} /> Quitar
        </button>
      </div>
    </div>
  )
}

export function OutgoingCard({ peer, onChanged }: { peer: FederationPeer; onChanged: () => void }) {
  const [busy, setBusy] = useState(false)
  const pending = peer.join_state === "pending"
  return (
    <div className="flex flex-wrap items-center gap-3 rounded-lg border border-border bg-card p-3 text-sm">
      <Send size={16} className="text-muted-foreground" />
      <div className="min-w-0">
        <div>
          Solicitud a <span className="font-medium">{peer.name}</span>{" "}
          <span className="text-xs text-muted-foreground">({peer.url})</span>
        </div>
        <div className="text-xs text-muted-foreground">
          {pending ? (
            <>
              Esperando a que su admin la acepte · código de verificación{" "}
              <span className="font-mono font-semibold text-foreground">{peer.join_verification_code ?? "—"}</span>
            </>
          ) : peer.join_state === "denied" ? (
            "Denegada por el otro nodo"
          ) : (
            "Caducada o retirada"
          )}
        </div>
      </div>
      <div className="ml-auto flex gap-2">
        {pending && (
          <button
            type="button"
            className={btnCls}
            disabled={busy}
            onClick={async () => {
              setBusy(true)
              try {
                await api.federation.joinCheck(peer.peer_id)
                onChanged()
              } catch (err) {
                toast.error(errMsg(err))
              } finally {
                setBusy(false)
              }
            }}
          >
            <RefreshCw size={12} className={busy ? "animate-spin" : ""} /> Comprobar
          </button>
        )}
        <button
          type="button"
          className={btnCls}
          onClick={async () => {
            try {
              await api.federation.deletePeer(peer.peer_id)
              onChanged()
            } catch (err) {
              toast.error(errMsg(err))
            }
          }}
        >
          {pending ? "Retirar" : "Descartar"}
        </button>
      </div>
    </div>
  )
}

export function RequestCard({
  req,
  onAccept,
  onChanged,
}: {
  req: FederationJoinRequest
  onAccept: (r: FederationJoinRequest) => void
  onChanged: () => void
}) {
  const [busy, setBusy] = useState(false)
  return (
    <div className="rounded-lg border border-amber-500/40 bg-amber-500/5 p-4">
      <div className="flex flex-wrap items-center gap-2">
        <Inbox size={16} className="text-amber-600 dark:text-amber-300" />
        <span className="font-semibold">{req.remote_node_name}</span>
        {req.remote_version && <Pill tone="muted">v{req.remote_version}</Pill>}
        <Pill tone="info">enjambre</Pill>
        <span className="text-xs text-muted-foreground">
          {req.source_ip ? `desde ${req.source_ip} · ` : ""}
          {timeAgo(req.created_at)}
        </span>
      </div>
      {req.gpus.length > 0 && (
        <div className="mt-1 text-xs text-muted-foreground">
          {req.gpus.map((g) => `${g.name ?? "GPU"}${g.total_vram_mb ? ` (${Math.round(g.total_vram_mb / 1024)} GB)` : ""}`).join(" · ")}
        </div>
      )}
      {req.message && <p className="mt-2 text-sm">«{req.message}»</p>}
      <div className="mt-2 flex flex-wrap items-center gap-2">
        <span className="text-xs text-muted-foreground">
          Código de verificación <span className="font-mono font-semibold text-foreground">{req.verification_code}</span>
        </span>
        <div className="ml-auto flex gap-2">
          <button
            type="button"
            className={btnCls}
            disabled={busy}
            onClick={async () => {
              setBusy(true)
              try {
                await api.federation.denyJoinRequest(req.id)
                toast.success("Solicitud denegada")
                onChanged()
              } catch (err) {
                toast.error(errMsg(err))
              } finally {
                setBusy(false)
              }
            }}
          >
            <X size={12} /> Denegar
          </button>
          <button type="button" className={primaryBtnCls} disabled={busy} onClick={() => onAccept(req)}>
            <Check size={14} /> Aceptar…
          </button>
        </div>
      </div>
    </div>
  )
}
