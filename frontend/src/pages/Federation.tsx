import { useCallback, useEffect, useMemo, useState } from "react"
import * as Dialog from "@radix-ui/react-dialog"
import {
  ArrowDownLeft,
  ArrowUpRight,
  Check,
  Copy,
  Cpu,
  Globe,
  Inbox,
  Link2,
  Loader2,
  MonitorSmartphone,
  Network,
  Plus,
  RefreshCw,
  Send,
  Ticket,
  Trash2,
  Wifi,
  WifiOff,
  X,
  Zap,
} from "lucide-react"
import { toast } from "sonner"
import { api } from "@/api/client"
import { CloudSection } from "@/components/federation/CloudSection"
import { enterRemoteNode } from "@/lib/remoteNode"
import type {
  FederationInvite,
  FederationJoinRequest,
  FederationNodeInfo,
  FederationPeer,
  FederationPeerCreate,
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
 */

const POLL_MS = 5_000

const inputCls = "mt-1 w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
const btnCls =
  "inline-flex items-center gap-1.5 rounded-md border border-border px-2.5 py-1.5 text-xs hover:bg-muted disabled:opacity-50"
const primaryBtnCls =
  "inline-flex items-center gap-1.5 rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground disabled:opacity-50"

function timeAgo(value: string | number | null | undefined): string {
  if (value == null) return "nunca"
  const ms = typeof value === "number" ? value * 1000 : Date.parse(value)
  if (Number.isNaN(ms)) return "nunca"
  const s = Math.max(0, Math.floor((Date.now() - ms) / 1000))
  if (s < 60) return `hace ${s}s`
  const m = Math.floor(s / 60)
  if (m < 60) return `hace ${m} min`
  const h = Math.floor(m / 60)
  if (h < 24) return `hace ${h} h`
  return `hace ${Math.floor(h / 24)} d`
}

function errMsg(err: unknown): string {
  return err instanceof Error ? err.message : "Error"
}

async function copy(text: string) {
  try {
    await navigator.clipboard.writeText(text)
    toast.success("Copiado")
  } catch {
    toast.error("No se pudo copiar")
  }
}

function Pill({ tone, children }: { tone: "ok" | "bad" | "warn" | "muted" | "info"; children: React.ReactNode }) {
  const cls = {
    ok: "border-emerald-500/30 bg-emerald-500/15 text-emerald-600 dark:text-emerald-300",
    bad: "border-red-500/30 bg-red-500/15 text-red-600 dark:text-red-300",
    warn: "border-amber-500/30 bg-amber-500/15 text-amber-700 dark:text-amber-300",
    info: "border-sky-500/30 bg-sky-500/15 text-sky-700 dark:text-sky-300",
    muted: "border-border bg-muted text-muted-foreground",
  }[tone]
  return (
    <span className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] font-medium ${cls}`}>
      {children}
    </span>
  )
}

function Toggle({
  checked,
  disabled,
  onChange,
  label,
}: {
  checked: boolean
  disabled?: boolean
  onChange: (v: boolean) => void
  label: string
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className={`relative inline-flex h-5 w-9 shrink-0 rounded-full border-2 border-transparent transition-colors disabled:opacity-50 ${
        checked ? "bg-primary" : "bg-muted"
      }`}
    >
      <span
        className={`pointer-events-none inline-block h-4 w-4 rounded-full bg-white shadow transition ${
          checked ? "translate-x-4" : "translate-x-0"
        }`}
      />
    </button>
  )
}

function ModalShell({
  open,
  onOpenChange,
  title,
  description,
  children,
}: {
  open: boolean
  onOpenChange: (v: boolean) => void
  title: string
  description?: string
  children: React.ReactNode
}) {
  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-40 bg-black/60" />
        <Dialog.Content className="fixed left-1/2 top-1/2 z-50 max-h-[90vh] w-[95vw] max-w-lg -translate-x-1/2 -translate-y-1/2 overflow-y-auto rounded-lg border border-border bg-card p-5">
          <Dialog.Title className="text-lg font-semibold">{title}</Dialog.Title>
          {description && (
            <Dialog.Description className="mt-1 text-sm text-muted-foreground">{description}</Dialog.Description>
          )}
          <div className="mt-4">{children}</div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  )
}

// ── Modo de federación ──────────────────────────────────────────

function ModePicker() {
  return (
    <div className="grid gap-2 sm:grid-cols-2">
      <div className="rounded-md border-2 border-primary bg-primary/5 p-3">
        <div className="text-sm font-medium">Soy parte del enjambre</div>
        <p className="mt-1 text-xs text-muted-foreground">
          Mutuo y con control completo: cada nodo usa los modelos del otro y puede cargarlos, descargarlos o
          bajarlos.
        </p>
      </div>
      <div className="cursor-not-allowed rounded-md border border-dashed border-border p-3 opacity-60">
        <div className="text-sm font-medium">Mi server mis reglas</div>
        <p className="mt-1 text-xs text-muted-foreground">Eliges qué ofreces y qué permites. Próximamente.</p>
      </div>
    </div>
  )
}

// ── Diálogos ────────────────────────────────────────────────────

function JoinDialog({
  open,
  onOpenChange,
  onDone,
}: {
  open: boolean
  onOpenChange: (v: boolean) => void
  onDone: () => void
}) {
  const [hubUrl, setHubUrl] = useState("")
  const [message, setMessage] = useState("")
  const [invite, setInvite] = useState("")
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    if (open) {
      setHubUrl("")
      setMessage("")
      setInvite("")
    }
  }, [open])

  const submit = async () => {
    if (!hubUrl.trim()) {
      toast.error("Indica la URL del otro oCabra")
      return
    }
    setBusy(true)
    try {
      const peer = await api.federation.join({
        hub_url: hubUrl.trim(),
        message: message.trim() || undefined,
        invite_code: invite.trim() || undefined,
      })
      toast.success(
        peer.join_state === "pending"
          ? `Solicitud enviada. Código de verificación: ${peer.join_verification_code ?? "—"}`
          : "Unido: conectando el túnel…",
      )
      onOpenChange(false)
      onDone()
    } catch (err) {
      toast.error(errMsg(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <ModalShell
      open={open}
      onOpenChange={onOpenChange}
      title="Unirse a otra federación"
      description="Este nodo no necesita URL pública: abre un túnel saliente hacia el otro y las peticiones viajan por él en los dos sentidos."
    >
      <div className="space-y-3">
        <label className="block text-sm text-muted-foreground">
          URL del otro oCabra
          <input
            className={inputCls}
            placeholder="https://ocabra.ejemplo.org"
            value={hubUrl}
            onChange={(e) => setHubUrl(e.target.value)}
          />
        </label>
        <label className="block text-sm text-muted-foreground">
          Mensaje para su admin <span className="text-xs">(opcional)</span>
          <textarea
            className={`${inputCls} min-h-16`}
            placeholder="Quién eres y qué aportas"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
          />
        </label>
        <label className="block text-sm text-muted-foreground">
          Código de invitación <span className="text-xs">(si te lo han dado, entra sin esperar aprobación)</span>
          <input
            className={inputCls}
            placeholder="ocinv_…"
            value={invite}
            onChange={(e) => setInvite(e.target.value)}
          />
        </label>
        <div>
          <div className="mb-1 text-sm text-muted-foreground">Modo</div>
          <ModePicker />
        </div>
      </div>
      <div className="mt-4 flex justify-end gap-2">
        <Dialog.Close className={btnCls}>Cancelar</Dialog.Close>
        <button type="button" className={primaryBtnCls} disabled={busy} onClick={() => void submit()}>
          {busy ? <Loader2 size={14} className="animate-spin" /> : <Send size={14} />}
          Enviar solicitud
        </button>
      </div>
    </ModalShell>
  )
}

function InviteDialog({
  open,
  onOpenChange,
  onDone,
}: {
  open: boolean
  onOpenChange: (v: boolean) => void
  onDone: () => void
}) {
  const [label, setLabel] = useState("")
  const [hours, setHours] = useState("24")
  const [uses, setUses] = useState("1")
  const [busy, setBusy] = useState(false)
  const [created, setCreated] = useState<FederationInvite | null>(null)

  useEffect(() => {
    if (open) {
      setLabel("")
      setHours("24")
      setUses("1")
      setCreated(null)
    }
  }, [open])

  const submit = async () => {
    if (!label.trim()) {
      toast.error("Ponle un nombre (p. ej. «nodo de Fulano»)")
      return
    }
    setBusy(true)
    try {
      const inv = await api.federation.createInvite({
        label: label.trim(),
        max_uses: Math.max(1, Number(uses) || 1),
        expires_in_hours: Number(hours) > 0 ? Number(hours) : null,
      })
      setCreated(inv)
      onDone()
    } catch (err) {
      toast.error(errMsg(err))
    } finally {
      setBusy(false)
    }
  }

  const joinUrl = created?.join_url || window.location.origin
  const envSnippet = created?.code
    ? `FEDERATION_JOIN_URL=${joinUrl}\nFEDERATION_JOIN_INVITE=${created.code}`
    : ""

  return (
    <ModalShell
      open={open}
      onOpenChange={onOpenChange}
      title="Invitar un nodo"
      description="Quien use la invitación entra sin que tengas que aprobarla. Úsala para nodos de confianza o para nodos en la nube."
    >
      {created?.code ? (
        <div className="space-y-3 text-sm">
          <p className="text-amber-600 dark:text-amber-300">
            Copia el código ahora: no se volverá a mostrar.
          </p>
          <div>
            <div className="text-xs text-muted-foreground">En el otro nodo: Federación → Unirse, con esta URL y código</div>
            <div className="mt-1 flex items-center gap-2 rounded-md border border-border bg-muted/40 px-3 py-2 font-mono text-xs">
              <span className="truncate">{joinUrl}</span>
              <button type="button" className="ml-auto" onClick={() => void copy(joinUrl)} aria-label="Copiar URL">
                <Copy size={13} />
              </button>
            </div>
            <div className="mt-1 flex items-center gap-2 rounded-md border border-border bg-muted/40 px-3 py-2 font-mono text-xs">
              <span className="truncate">{created.code}</span>
              <button type="button" className="ml-auto" onClick={() => void copy(created.code!)} aria-label="Copiar código">
                <Copy size={13} />
              </button>
            </div>
          </div>
          <div>
            <div className="text-xs text-muted-foreground">O, para un nodo desatendido, en su .env</div>
            <pre className="mt-1 overflow-x-auto rounded-md border border-border bg-muted/40 px-3 py-2 text-xs">{envSnippet}</pre>
            <button type="button" className={`${btnCls} mt-1`} onClick={() => void copy(envSnippet)}>
              <Copy size={12} /> Copiar
            </button>
          </div>
          <div className="flex justify-end">
            <Dialog.Close className={primaryBtnCls}>Hecho</Dialog.Close>
          </div>
        </div>
      ) : (
        <>
          <div className="space-y-3">
            <label className="block text-sm text-muted-foreground">
              Nombre
              <input className={inputCls} value={label} onChange={(e) => setLabel(e.target.value)} placeholder="nodo de Fulano" />
            </label>
            <div className="grid grid-cols-2 gap-3">
              <label className="block text-sm text-muted-foreground">
                Caduca en (horas)
                <input className={inputCls} type="number" min={1} value={hours} onChange={(e) => setHours(e.target.value)} />
              </label>
              <label className="block text-sm text-muted-foreground">
                Usos
                <input className={inputCls} type="number" min={1} max={100} value={uses} onChange={(e) => setUses(e.target.value)} />
              </label>
            </div>
            <div>
              <div className="mb-1 text-sm text-muted-foreground">Modo</div>
              <ModePicker />
            </div>
          </div>
          <div className="mt-4 flex justify-end gap-2">
            <Dialog.Close className={btnCls}>Cancelar</Dialog.Close>
            <button type="button" className={primaryBtnCls} disabled={busy} onClick={() => void submit()}>
              {busy ? <Loader2 size={14} className="animate-spin" /> : <Ticket size={14} />}
              Crear invitación
            </button>
          </div>
        </>
      )}
    </ModalShell>
  )
}

function AcceptDialog({
  request,
  onOpenChange,
  onDone,
}: {
  request: FederationJoinRequest | null
  onOpenChange: (v: boolean) => void
  onDone: () => void
}) {
  const [name, setName] = useState("")
  const [notes, setNotes] = useState("")
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    if (request) {
      setName(request.remote_node_name)
      setNotes("")
    }
  }, [request])

  const submit = async () => {
    if (!request) return
    setBusy(true)
    try {
      await api.federation.acceptJoinRequest(request.id, {
        name: name.trim() || undefined,
        notes: notes.trim() || undefined,
      })
      toast.success(`${name || request.remote_node_name} aceptado: se conectará en unos segundos`)
      onOpenChange(false)
      onDone()
    } catch (err) {
      toast.error(errMsg(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <ModalShell
      open={request !== null}
      onOpenChange={onOpenChange}
      title="Aceptar en el enjambre"
      description="Federación mutua con control completo: este nodo usará sus modelos y él los tuyos, y ambos podréis cargar y descargar modelos en el otro."
    >
      {request && (
        <div className="space-y-3">
          <div className="rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-sm">
            Comprueba con su admin, por otro canal, que ve el código{" "}
            <span className="font-mono font-semibold">{request.verification_code}</span>.
          </div>
          <label className="block text-sm text-muted-foreground">
            Nombre del nodo aquí
            <input className={inputCls} value={name} onChange={(e) => setName(e.target.value)} />
          </label>
          <label className="block text-sm text-muted-foreground">
            Notas <span className="text-xs">(de quién es, para qué)</span>
            <textarea className={`${inputCls} min-h-16`} value={notes} onChange={(e) => setNotes(e.target.value)} />
          </label>
        </div>
      )}
      <div className="mt-4 flex justify-end gap-2">
        <Dialog.Close className={btnCls}>Cancelar</Dialog.Close>
        <button type="button" className={primaryBtnCls} disabled={busy} onClick={() => void submit()}>
          {busy ? <Loader2 size={14} className="animate-spin" /> : <Check size={14} />}
          Aceptar
        </button>
      </div>
    </ModalShell>
  )
}

function ManualPeerDialog({
  open,
  onOpenChange,
  onDone,
}: {
  open: boolean
  onOpenChange: (v: boolean) => void
  onDone: () => void
}) {
  const [form, setForm] = useState<FederationPeerCreate>({ name: "", url: "", api_key: "", access_level: "inference" })
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    if (open) setForm({ name: "", url: "", api_key: "", access_level: "inference" })
  }, [open])

  const submit = async () => {
    if (!form.name.trim() || !form.url.trim() || !form.api_key.trim()) {
      toast.error("Nombre, URL y API key son obligatorios")
      return
    }
    setBusy(true)
    try {
      await api.federation.addPeer({ ...form, name: form.name.trim(), url: form.url.trim(), api_key: form.api_key.trim() })
      toast.success("Nodo añadido")
      onOpenChange(false)
      onDone()
    } catch (err) {
      toast.error(errMsg(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <ModalShell
      open={open}
      onOpenChange={onOpenChange}
      title="Añadir nodo por URL"
      description="Modo antiguo, de un solo sentido: este nodo llama al otro con una API key que te haya dado. El otro necesita URL alcanzable."
    >
      <div className="space-y-3">
        <label className="block text-sm text-muted-foreground">
          Nombre
          <input className={inputCls} value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />
        </label>
        <label className="block text-sm text-muted-foreground">
          URL
          <input className={inputCls} placeholder="http://10.0.0.5:8484" value={form.url} onChange={(e) => setForm({ ...form, url: e.target.value })} />
        </label>
        <label className="block text-sm text-muted-foreground">
          API key
          <input className={inputCls} type="password" value={form.api_key} onChange={(e) => setForm({ ...form, api_key: e.target.value })} />
        </label>
        <label className="block text-sm text-muted-foreground">
          Acceso
          <select
            className={inputCls}
            value={form.access_level}
            onChange={(e) => setForm({ ...form, access_level: e.target.value as "inference" | "full" })}
          >
            <option value="inference">Solo inferencia</option>
            <option value="full">Completo (cargar/descargar modelos)</option>
          </select>
        </label>
      </div>
      <div className="mt-4 flex justify-end gap-2">
        <Dialog.Close className={btnCls}>Cancelar</Dialog.Close>
        <button type="button" className={primaryBtnCls} disabled={busy} onClick={() => void submit()}>
          {busy ? <Loader2 size={14} className="animate-spin" /> : <Plus size={14} />}
          Añadir
        </button>
      </div>
    </ModalShell>
  )
}

// ── Tarjetas ────────────────────────────────────────────────────

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

function PeerCard({ peer, onChanged }: { peer: FederationPeer; onChanged: () => void }) {
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
        <Pill tone="muted">
          {conn.icon} {conn.text}
        </Pill>
        {peer.mode === "swarm" && <Pill tone="info">enjambre</Pill>}
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

function OutgoingCard({ peer, onChanged }: { peer: FederationPeer; onChanged: () => void }) {
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

function RequestCard({
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

// ── Página ──────────────────────────────────────────────────────

export function Federation() {
  const [node, setNode] = useState<FederationNodeInfo | null>(null)
  const [peers, setPeers] = useState<FederationPeer[]>([])
  const [requests, setRequests] = useState<FederationJoinRequest[]>([])
  const [invites, setInvites] = useState<FederationInvite[]>([])
  const [loading, setLoading] = useState(true)
  const [nodeName, setNodeName] = useState("")
  const [savingName, setSavingName] = useState(false)
  const [joinOpen, setJoinOpen] = useState(false)
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
                  <PeerCard key={p.peer_id} peer={p} onChanged={() => void load()} />
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
