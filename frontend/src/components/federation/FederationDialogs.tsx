// Diálogos de la página de Federación (REF-18): unirse, invitar, aceptar y
// añadir por URL. Extraídos de pages/Federation.tsx.

import { useEffect, useState } from "react"
import * as Dialog from "@radix-ui/react-dialog"
import { Check, Copy, Loader2, Plus, Send, Ticket } from "lucide-react"
import { toast } from "sonner"
import { api } from "@/api/client"
import { btnCls, copy, errMsg, inputCls, ModalShell, primaryBtnCls } from "@/components/federation/shared"
import type { FederationInvite, FederationJoinRequest, FederationPeerCreate } from "@/types"

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
      <div className="rounded-md border border-dashed border-border p-3">
        <div className="text-sm font-medium">Mi servidor, mis reglas</div>
        <p className="mt-1 text-xs text-muted-foreground">
          Eliges qué ofreces y qué permites. Se cambia después en la tarjeta del nodo («Le sirvo como»).
        </p>
      </div>
    </div>
  )
}

export function JoinDialog({
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

export function InviteDialog({
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

export function AcceptDialog({
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

export function ManualPeerDialog({
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
