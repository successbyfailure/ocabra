import { useEffect, useState } from "react"
import * as Dialog from "@radix-ui/react-dialog"
import { Loader2, Scale } from "lucide-react"
import { toast } from "sonner"
import { api } from "@/api/client"
import type { FederationPeer, FederationRules, FederationRulesUsage } from "@/types"

const inputCls = "w-24 rounded-md border border-border bg-background px-2 py-1 text-sm"
const btnCls =
  "inline-flex items-center gap-1.5 rounded-md border border-border px-2.5 py-1 text-xs hover:bg-muted disabled:opacity-50"

const SWITCHES: { key: keyof FederationRules; label: string; help: string }[] = [
  { key: "local_first", label: "Yo primero", help: "Mis peticiones van antes que las suyas" },
  { key: "allow_loads", label: "Puede cargar modelos", help: "Si no, solo usa lo que ya está cargado aquí" },
  { key: "allow_evict", label: "Puede expulsar modelos en uso", help: "Si no, sus cargas no echan un modelo que estoy usando" },
  { key: "allow_downloads", label: "Puede descargar modelos", help: "Descargas e instalación de backends en este nodo" },
  { key: "allow_admin", label: "Puede administrar", help: "Si no, el panel remoto es de solo lectura" },
]

/** ``PeerRules`` defaults on the server (core/federation_rules.py). */
const PEER_RULES_DEFAULTS: FederationRules = {
  local_first: false,
  allow_loads: true,
  allow_evict: true,
  max_requests_per_hour: null,
  max_wh_per_month: null,
  allow_downloads: true,
  allow_admin: true,
  offered_models: null,
  hours: null,
}

/** One-line summary of a set of rules, for cards. */
export function rulesSummary(r: FederationRules | null | undefined): string {
  if (!r) return "todo permitido"
  const parts: string[] = []
  if (r.local_first) parts.push("yo primero")
  if (!r.allow_loads) parts.push("solo lo cargado")
  if (!r.allow_evict) parts.push("sin expulsar")
  if (!r.allow_downloads) parts.push("sin descargas")
  if (!r.allow_admin) parts.push("solo lectura")
  if (r.max_requests_per_hour) parts.push(`${r.max_requests_per_hour} pet/h`)
  if (r.max_wh_per_month) parts.push(`${r.max_wh_per_month} Wh/mes`)
  if (r.offered_models) parts.push(`${r.offered_models.length} modelos`)
  if (r.hours) parts.push(`de ${r.hours[0]}:00 a ${r.hours[1]}:00`)
  return parts.length ? parts.join(" · ") : "todo permitido"
}

/**
 * Form for a full set of rules. ``value`` is what applies now; ``onChange``
 * gets the edited set.
 */
export function RulesForm({
  value,
  onChange,
  models,
}: {
  value: FederationRules
  onChange: (r: FederationRules) => void
  models: string[]
}) {
  const set = (patch: Partial<FederationRules>) => onChange({ ...value, ...patch })
  const [modelFilter, setModelFilter] = useState("")
  const offered = value.offered_models
  return (
    <div className="space-y-3 text-sm">
      <div className="grid gap-2 sm:grid-cols-2">
        {SWITCHES.map((s) => (
          <label key={s.key} className="flex cursor-pointer items-start gap-2 rounded-md border border-border p-2">
            <input
              type="checkbox"
              className="mt-0.5 accent-primary"
              checked={Boolean(value[s.key])}
              onChange={(e) => set({ [s.key]: e.target.checked } as Partial<FederationRules>)}
            />
            <span>
              <span className="font-medium">{s.label}</span>
              <span className="block text-xs text-muted-foreground">{s.help}</span>
            </span>
          </label>
        ))}
      </div>
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <label className="flex items-center gap-2">
          Peticiones por hora
          <input
            className={inputCls}
            type="number"
            min={0}
            placeholder="sin límite"
            value={value.max_requests_per_hour ?? ""}
            onChange={(e) => set({ max_requests_per_hour: Number(e.target.value) || null })}
          />
        </label>
        <label className="flex items-center gap-2">
          Wh al mes
          <input
            className={inputCls}
            type="number"
            min={0}
            placeholder="sin límite"
            value={value.max_wh_per_month ?? ""}
            onChange={(e) => set({ max_wh_per_month: Number(e.target.value) || null })}
          />
        </label>
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            className="accent-primary"
            checked={value.hours != null}
            onChange={(e) => set({ hours: e.target.checked ? [22, 8] : null })}
          />
          Solo de
          <input
            className="w-14 rounded-md border border-border bg-background px-1 py-1 text-sm disabled:opacity-50"
            type="number"
            min={0}
            max={23}
            disabled={value.hours == null}
            value={value.hours?.[0] ?? 22}
            onChange={(e) => set({ hours: [Number(e.target.value) % 24, value.hours?.[1] ?? 8] })}
          />
          a
          <input
            className="w-14 rounded-md border border-border bg-background px-1 py-1 text-sm disabled:opacity-50"
            type="number"
            min={0}
            max={23}
            disabled={value.hours == null}
            value={value.hours?.[1] ?? 8}
            onChange={(e) => set({ hours: [value.hours?.[0] ?? 22, Number(e.target.value) % 24] })}
          />
          h
        </label>
      </div>
      <div className="rounded-md border border-border">
        <div className="flex flex-wrap items-center gap-2 border-b border-border px-2 py-1.5">
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              className="accent-primary"
              checked={offered == null}
              onChange={(e) => set({ offered_models: e.target.checked ? null : [] })}
            />
            Ofrecer todos los modelos
          </label>
          {offered != null && (
            <input
              className="ml-auto w-36 rounded-md border border-border bg-background px-2 py-0.5 text-xs"
              placeholder="Filtrar…"
              value={modelFilter}
              onChange={(e) => setModelFilter(e.target.value)}
            />
          )}
        </div>
        {offered != null && (
          <div className="max-h-40 overflow-y-auto px-2 py-1">
            {models
              .filter((m) => !modelFilter || m.toLowerCase().includes(modelFilter.toLowerCase()))
              .map((m) => (
                <label key={m} className="flex items-center gap-2 py-0.5 text-xs">
                  <input
                    type="checkbox"
                    className="accent-primary"
                    checked={offered.includes(m)}
                    onChange={(e) =>
                      set({
                        offered_models: e.target.checked ? [...offered, m] : offered.filter((x) => x !== m),
                      })
                    }
                  />
                  <span className="font-mono">{m}</span>
                </label>
              ))}
          </div>
        )}
      </div>
    </div>
  )
}

/** Keys of *next* that differ from *base*: what to store as an adjustment. */
function diff(next: FederationRules, base: FederationRules): Partial<FederationRules> {
  const out: Partial<FederationRules> = {}
  for (const k of Object.keys(next) as (keyof FederationRules)[]) {
    if (JSON.stringify(next[k]) !== JSON.stringify(base[k])) {
      ;(out as Record<string, unknown>)[k] = next[k]
    }
  }
  return out
}

/** Our models' public ids, for the "offered models" list. */
function useModelIds(open: boolean): string[] {
  const [ids, setIds] = useState<string[]>([])
  useEffect(() => {
    if (!open) return
    api.models
      .list()
      .then((ms) => setIds(ms.filter((m) => !m.federation?.remote).map((m) => m.modelId).sort()))
      .catch(() => setIds([]))
  }, [open])
  return ids
}

/** This node's defaults for peers in «Mi servidor, mis reglas». */
export function NodeRulesCard() {
  const [rules, setRules] = useState<FederationRules | null>(null)
  const [saved, setSaved] = useState<FederationRules | null>(null)
  const [base, setBase] = useState<FederationRules | null>(null)
  const [busy, setBusy] = useState(false)
  const models = useModelIds(true)

  useEffect(() => {
    api.federation
      .getRules()
      .then((r) => {
        setRules(r.effective)
        setSaved(r.effective)
        // What rules mode starts from; the node defaults are stored as the
        // differences from it (PUT replaces them whole).
        setBase({ ...PEER_RULES_DEFAULTS, ...r.base } as FederationRules)
      })
      .catch(() => setRules(null))
  }, [])

  if (!rules || !saved || !base) return null
  const dirty = JSON.stringify(rules) !== JSON.stringify(saved)
  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="mb-1 flex items-center gap-2">
        <Scale size={15} className="text-muted-foreground" />
        <span className="text-sm font-semibold">Reglas para los nodos en «Mi servidor, mis reglas»</span>
      </div>
      <p className="mb-3 text-xs text-muted-foreground">
        Lo que pueden hacer aquí los nodos que pongas en ese modo (los del enjambre lo pueden todo). Cada nodo
        puede tener además sus propios ajustes. Una petición que no cumple se rechaza y su nodo la resuelve por
        su cuenta.
      </p>
      <RulesForm value={rules} onChange={setRules} models={models} />
      <div className="mt-3 flex justify-end">
        <button
          type="button"
          disabled={!dirty || busy}
          className="rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground disabled:opacity-50"
          onClick={async () => {
            setBusy(true)
            try {
              const out = await api.federation.setRules(diff(rules, base))
              setRules(out.effective)
              setSaved(out.effective)
              toast.success("Reglas guardadas")
            } catch (err) {
              toast.error(err instanceof Error ? err.message : String(err))
            } finally {
              setBusy(false)
            }
          }}
        >
          {busy ? <Loader2 size={14} className="animate-spin" /> : null} Guardar
        </button>
      </div>
    </div>
  )
}

/** Two ways of serving a peer (the swarm is «rules» with everything allowed). */
type ServeMode = "swarm" | "rules"

/**
 * How this node serves *peer*: as part of the swarm (it may do everything) or
 * under this node's rules, with the adjustments that only apply to it and what
 * it has used. One save writes the mode and the rules together.
 */
export function PeerRulesDialog({
  peer,
  onOpenChange,
  onSaved,
}: {
  peer: FederationPeer | null
  onOpenChange: (v: boolean) => void
  onSaved: () => void
}) {
  const open = peer !== null
  const [mode, setMode] = useState<ServeMode>("swarm")
  const [nodeRules, setNodeRules] = useState<FederationRules | null>(null)
  const [rules, setRules] = useState<FederationRules | null>(null)
  const [usage, setUsage] = useState<FederationRulesUsage | null>(null)
  const [busy, setBusy] = useState(false)
  const models = useModelIds(open)

  useEffect(() => {
    if (!peer) return
    setMode(peer.mode === "rules" ? "rules" : "swarm")
    setRules(null)
    api.federation
      .getRules()
      .then((r) => {
        setNodeRules(r.effective)
        setRules({ ...r.effective, ...(peer.rules ?? {}) } as FederationRules)
      })
      .catch(() => setNodeRules(null))
    api.federation.rulesUsage(peer.peer_id).then(setUsage).catch(() => setUsage(null))
  }, [peer])

  const save = async (resetRules: boolean) => {
    if (!peer) return
    if (mode === "rules" && !resetRules && (!rules || !nodeRules)) return
    setBusy(true)
    try {
      const adjustments = !resetRules && rules && nodeRules ? diff(rules, nodeRules) : {}
      await api.federation.updatePeer(peer.peer_id, {
        mode,
        ...(mode === "rules" && { rules: Object.keys(adjustments).length ? adjustments : null }),
      })
      toast.success(
        mode === "swarm"
          ? `${peer.name} es parte del enjambre: puede hacerlo todo`
          : resetRules
            ? `${peer.name} se sirve con las reglas del nodo`
            : `${peer.name} se sirve con tus reglas`,
      )
      onOpenChange(false)
      onSaved()
    } catch (err) {
      toast.error(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-40 bg-black/60" />
        <Dialog.Content className="fixed left-1/2 top-1/2 z-50 max-h-[90vh] w-[95vw] max-w-2xl -translate-x-1/2 -translate-y-1/2 overflow-y-auto rounded-lg border border-border bg-card p-5">
          <Dialog.Title className="text-lg font-semibold">Cómo le sirvo a {peer?.name}</Dialog.Title>
          <Dialog.Description className="mt-1 text-sm text-muted-foreground">
            Lo que {peer?.name} puede hacer con este nodo cuando te pide algo.
          </Dialog.Description>

          <div className="mt-4 grid gap-2 sm:grid-cols-2" role="radiogroup" aria-label="Le sirvo como">
            {(
              [
                ["swarm", "Enjambre", "Puede hacerlo todo: usar, cargar y expulsar modelos, descargar y administrar."],
                ["rules", "Mis reglas", "Se le sirve con las reglas de este nodo, ajustadas para él si quieres."],
              ] as const
            ).map(([value, label, help]) => (
              <button
                key={value}
                type="button"
                role="radio"
                aria-checked={mode === value}
                onClick={() => setMode(value)}
                className={`rounded-md border p-3 text-left ${
                  mode === value ? "border-primary bg-primary/10" : "border-border hover:bg-muted"
                }`}
              >
                <div className="text-sm font-medium">{label}</div>
                <div className="mt-0.5 text-xs text-muted-foreground">{help}</div>
              </button>
            ))}
          </div>

          {mode === "rules" && (
            <div className="mt-4 space-y-3">
              <p className="text-xs text-muted-foreground">
                Parte de las reglas de este nodo; aquí cambias solo lo que sea distinto para {peer?.name}.
              </p>
              {usage && (
                <p className="text-xs text-muted-foreground">
                  Uso: {usage.requests_last_hour} peticiones en la última hora
                  {usage.max_requests_per_hour ? ` de ${usage.max_requests_per_hour}` : ""} · {usage.wh_this_month} Wh
                  este mes{usage.max_wh_per_month ? ` de ${usage.max_wh_per_month}` : ""}
                </p>
              )}
              {rules ? (
                <RulesForm value={rules} onChange={setRules} models={models} />
              ) : (
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Loader2 size={14} className="animate-spin" /> Cargando…
                </div>
              )}
            </div>
          )}

          <div className="mt-4 flex justify-end gap-2">
            <Dialog.Close className={btnCls}>Cancelar</Dialog.Close>
            {mode === "rules" && (
              <button type="button" className={btnCls} disabled={busy || !peer} onClick={() => void save(true)}>
                Usar las del nodo
              </button>
            )}
            <button
              type="button"
              disabled={busy || !peer || (mode === "rules" && (!rules || !nodeRules))}
              className="rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground disabled:opacity-50"
              onClick={() => void save(false)}
            >
              {busy ? <Loader2 size={14} className="animate-spin" /> : "Guardar"}
            </button>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  )
}
