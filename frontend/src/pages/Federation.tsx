import { useCallback, useEffect, useMemo, useState } from "react"
import { Check, Link2, Loader2, Network, Plus, Ticket } from "lucide-react"
import { toast } from "sonner"
import { api } from "@/api/client"
import { NodeRulesCard, PeerRulesDialog } from "@/components/federation/RulesEditor"
import { CloudSection } from "@/components/federation/CloudSection"
import {
  AcceptDialog,
  InviteDialog,
  JoinDialog,
  ManualPeerDialog,
} from "@/components/federation/FederationDialogs"
import { OutgoingCard, PeerCard, RequestCard } from "@/components/federation/FederationCards"
import { btnCls, errMsg, primaryBtnCls, Toggle } from "@/components/federation/shared"
import type {
  FederationInvite,
  FederationJoinRequest,
  FederationNodeInfo,
  FederationPeer,
} from "@/types"

/**
 * Federación — página propia (antes era una pestaña de Settings).
 *
 * Dos casos de uso hoy: el nodo de otra persona en otra ubicación y nodos
 * temporales en la nube. Basta con que UNO de los dos lados tenga URL
 * alcanzable: el otro abre un túnel saliente (wss) y por él viajan las
 * peticiones en los dos sentidos.
 *
 * Modos: "Soy parte del enjambre" (mutuo, control completo) es el único
 * disponible; "Mi server mis reglas" llegará cuando sepamos qué conviene
 * dejar elegir.
 *
 * REF-18: las tarjetas, los diálogos y las primitivas viven en
 * components/federation/ (FederationCards, FederationDialogs, shared).
 */

const POLL_MS = 5_000

export function Federation() {
  const [node, setNode] = useState<FederationNodeInfo | null>(null)
  const [peers, setPeers] = useState<FederationPeer[]>([])
  const [requests, setRequests] = useState<FederationJoinRequest[]>([])
  const [invites, setInvites] = useState<FederationInvite[]>([])
  const [loading, setLoading] = useState(true)
  const [nodeName, setNodeName] = useState("")
  const [savingName, setSavingName] = useState(false)
  const [joinOpen, setJoinOpen] = useState(false)
  const [rulesPeer, setRulesPeer] = useState<FederationPeer | null>(null)
  const [inviteOpen, setInviteOpen] = useState(false)
  const [manualOpen, setManualOpen] = useState(false)
  const [accepting, setAccepting] = useState<FederationJoinRequest | null>(null)

  const load = useCallback(async () => {
    try {
      const info = await api.federation.getNode()
      setNode(info)
      if (info.enabled) {
        const [p, r, i] = await Promise.all([
          api.federation.getPeers(),
          api.federation.getJoinRequests(),
          api.federation.getInvites(),
        ])
        setPeers(p)
        setRequests(r)
        setInvites(i)
      } else {
        setPeers([])
        setRequests([])
        setInvites([])
      }
    } catch (err) {
      toast.error(errMsg(err))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void load()
    const id = window.setInterval(() => void load(), POLL_MS)
    return () => window.clearInterval(id)
  }, [load])

  useEffect(() => {
    if (node?.node_name != null) setNodeName(node.node_name)
  }, [node?.node_name])

  const patchConfig = async (patch: Record<string, unknown>) => {
    try {
      await api.config.patch(patch)
      await load()
    } catch (err) {
      toast.error(errMsg(err))
    }
  }

  const active = useMemo(() => peers.filter((p) => p.join_state == null), [peers])
  const outgoing = useMemo(() => peers.filter((p) => p.join_state != null), [peers])
  const usableInvites = useMemo(() => invites.filter((i) => i.usable), [invites])

  if (loading && !node) {
    return (
      <div className="flex items-center gap-2 p-6 text-muted-foreground">
        <Loader2 size={16} className="animate-spin" /> Cargando…
      </div>
    )
  }

  const enabled = node?.enabled ?? false

  return (
    <div className="space-y-5 pb-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold">Federación</h1>
          <p className="text-muted-foreground">
            Une varios oCabra para compartir modelos y GPUs. Con que uno tenga URL alcanzable basta.
          </p>
        </div>
        {enabled && (
          <div className="flex flex-wrap gap-2">
            <button type="button" className={btnCls} onClick={() => setInviteOpen(true)}>
              <Ticket size={13} /> Invitar nodo
            </button>
            <button type="button" className={primaryBtnCls} onClick={() => setJoinOpen(true)}>
              <Link2 size={14} /> Unirse a otra federación
            </button>
          </div>
        )}
      </div>

      {/* Este nodo */}
      <div className="rounded-lg border border-border bg-card p-4">
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-2">
            <Toggle
              label="Federación activada"
              checked={enabled}
              onChange={(v) => void patchConfig({ federationEnabled: v })}
            />
            <span className="text-sm font-medium">{enabled ? "Federación activada" : "Federación desactivada"}</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-muted-foreground">Nombre de este nodo</span>
            <input
              className="w-44 rounded-md border border-border bg-background px-2 py-1 text-sm"
              value={nodeName}
              onChange={(e) => setNodeName(e.target.value)}
              placeholder="mi-ocabra"
            />
            {nodeName !== (node?.node_name ?? "") && (
              <button
                type="button"
                className={btnCls}
                disabled={savingName}
                onClick={async () => {
                  setSavingName(true)
                  await patchConfig({ federationNodeName: nodeName.trim() })
                  setSavingName(false)
                }}
              >
                <Check size={12} /> Guardar
              </button>
            )}
          </div>
          {enabled && (
            <div className="flex items-center gap-2">
              <Toggle
                label="Aceptar solicitudes de unión"
                checked={node?.accept_join_requests ?? false}
                onChange={(v) => void patchConfig({ federationAcceptJoinRequests: v })}
              />
              <span className="text-sm">Aceptar solicitudes de unión</span>
            </div>
          )}
        </div>
        <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
          <span>
            URL pública: {node?.public_url ? <span className="font-mono">{node.public_url}</span> : "sin configurar (PUBLIC_URL)"}
          </span>
          <span>v{node?.version}</span>
          {node?.node_id && <span className="font-mono">id {node.node_id.slice(0, 8)}</span>}
        </div>
        {enabled && !node?.accept_join_requests && (
          <p className="mt-2 text-xs text-muted-foreground">
            Con las solicitudes desactivadas, otros nodos solo pueden entrar con una invitación.
          </p>
        )}
      </div>

      {enabled && peers.some((p) => p.mode === "rules") && <NodeRulesCard />}
      <PeerRulesDialog peer={rulesPeer} onOpenChange={(v) => !v && setRulesPeer(null)} onSaved={() => void load()} />

      {!enabled ? (
        <div className="rounded-lg border border-dashed border-border bg-card/50 px-6 py-10 text-center">
          <Network size={32} className="mx-auto text-muted-foreground/50" />
          <p className="mt-2 text-sm font-medium text-muted-foreground">La federación está desactivada</p>
          <p className="mt-1 text-xs text-muted-foreground">Actívala para invitar nodos o unirte a otra federación.</p>
        </div>
      ) : (
        <>
          {requests.length > 0 && (
            <section className="space-y-2">
              <h2 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">
                Solicitudes pendientes ({requests.length})
              </h2>
              {requests.map((r) => (
                <RequestCard key={r.id} req={r} onAccept={setAccepting} onChanged={() => void load()} />
              ))}
            </section>
          )}

          {outgoing.length > 0 && (
            <section className="space-y-2">
              <h2 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">Mis solicitudes</h2>
              {outgoing.map((p) => (
                <OutgoingCard key={p.peer_id} peer={p} onChanged={() => void load()} />
              ))}
            </section>
          )}

          <section className="space-y-2">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">
                Nodos ({active.length})
              </h2>
              <button type="button" className={btnCls} onClick={() => setManualOpen(true)}>
                <Plus size={12} /> Añadir por URL (avanzado)
              </button>
            </div>
            {active.length === 0 ? (
              <div className="rounded-lg border border-dashed border-border px-6 py-8 text-center text-sm text-muted-foreground">
                Aún no hay nodos. Invita a uno o únete a otra federación.
              </div>
            ) : (
              <div className="grid gap-3 lg:grid-cols-2">
                {active.map((p) => (
                  <PeerCard
                    key={p.peer_id}
                    peer={p}
                    onChanged={() => void load()}
                    onEditRules={setRulesPeer}
                  />
                ))}
              </div>
            )}
          </section>

          <CloudSection />

          {usableInvites.length > 0 && (
            <section className="space-y-2">
              <h2 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">Invitaciones activas</h2>
              <div className="divide-y divide-border rounded-lg border border-border bg-card">
                {usableInvites.map((inv) => (
                  <div key={inv.id} className="flex flex-wrap items-center gap-3 px-4 py-2 text-sm">
                    <Ticket size={14} className="text-muted-foreground" />
                    <span className="font-medium">{inv.label}</span>
                    <span className="text-xs text-muted-foreground">
                      {inv.uses}/{inv.max_uses} usos
                      {inv.expires_at ? ` · caduca ${new Date(inv.expires_at).toLocaleString()}` : ""}
                    </span>
                    <button
                      type="button"
                      className={`${btnCls} ml-auto`}
                      onClick={async () => {
                        try {
                          await api.federation.revokeInvite(inv.id)
                          void load()
                        } catch (err) {
                          toast.error(errMsg(err))
                        }
                      }}
                    >
                      Revocar
                    </button>
                  </div>
                ))}
              </div>
            </section>
          )}
        </>
      )}

      <JoinDialog open={joinOpen} onOpenChange={setJoinOpen} onDone={() => void load()} />
      <InviteDialog open={inviteOpen} onOpenChange={setInviteOpen} onDone={() => void load()} />
      <ManualPeerDialog open={manualOpen} onOpenChange={setManualOpen} onDone={() => void load()} />
      <AcceptDialog request={accepting} onOpenChange={(v) => !v && setAccepting(null)} onDone={() => void load()} />
    </div>
  )
}
