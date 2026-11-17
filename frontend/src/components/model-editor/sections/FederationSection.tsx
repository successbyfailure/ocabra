import { useCallback, useEffect, useMemo, useState } from "react"
import { ArrowDown, ArrowUp, Loader2, Plus, X } from "lucide-react"
import { toast } from "sonner"
import { api } from "@/api/client"
import type { ModelFederationStrategy } from "@/types"
import { Callout, Chip, Hint, Panel, Segmented, Toggle, inputCls } from "../ui"
import type { SectionProps } from "./types"

type Strategy = ModelFederationStrategy["strategy"]
type Peer = ModelFederationStrategy["peers"][number]

const LOCAL = "local"

/** How a node has this model, for the chip next to it. */
export function nodeState(node: string, data: ModelFederationStrategy): { label: string; tone: "ok" | "info" | "warn" | "muted" } {
  if (node === LOCAL) return data.local ? { label: "lo tiene", tone: "ok" } : { label: "no lo tiene", tone: "warn" }
  const peer = findPeer(node, data.peers)
  if (!peer) return { label: "nodo desconocido", tone: "warn" }
  if (!peer.online) return { label: "desconectado", tone: "muted" }
  if (peer.model === "loaded") return { label: "cargado", tone: "ok" }
  if (peer.model === "cold") return { label: "en frío: lo carga", tone: "info" }
  return { label: "no lo ofrece", tone: "warn" }
}

function findPeer(node: string, peers: Peer[]): Peer | undefined {
  const n = node.toLowerCase()
  return peers.find((p) => p.name.toLowerCase() === n || p.peer_id.toLowerCase() === n)
}

const STRATEGY_HELP: Record<Strategy, string> = {
  auto: "Lo caliente gana: si está cargado aquí se sirve aquí; si solo un vecino lo tiene cargado, va a ese vecino.",
  local_only: "Nunca sale de este nodo, aunque un vecino lo tenga cargado y aquí haya que cargarlo.",
  prefer: "Va al primer nodo de la lista que esté en línea y lo tenga, aunque esté en frío (lo carga allí).",
}

/**
 * Reparto: where this model's requests are served when the node is in a
 * federation. A property of the model, not of a rung of its ladder.
 */
export function FederationSection(props: SectionProps) {
  const { editor, readOnly } = props
  const modelId = editor.model?.model_id ?? null
  const [data, setData] = useState<ModelFederationStrategy | null>(null)
  const [unavailable, setUnavailable] = useState<string | null>(null)
  const [strategy, setStrategy] = useState<Strategy>("auto")
  const [nodes, setNodes] = useState<string[]>([])
  const [fallback, setFallback] = useState(true)
  const [adding, setAdding] = useState("")
  const [saving, setSaving] = useState(false)

  const apply = (d: ModelFederationStrategy) => {
    setData(d)
    setStrategy(d.strategy)
    setNodes(d.nodes)
    setFallback(d.fallback)
  }

  const load = useCallback(async () => {
    if (!modelId) return
    try {
      apply(await api.federation.getStrategy(modelId))
      setUnavailable(null)
    } catch (err) {
      setUnavailable(err instanceof Error ? err.message : String(err))
    }
  }, [modelId])

  useEffect(() => {
    void load()
  }, [load])

  const dirty =
    data != null &&
    (strategy !== data.strategy || fallback !== data.fallback || nodes.join("\u0000") !== data.nodes.join("\u0000"))

  const addable = useMemo(() => {
    if (!data) return [] as { value: string; label: string }[]
    const taken = new Set(nodes.map((n) => n.toLowerCase()))
    const out: { value: string; label: string }[] = []
    if (!taken.has(LOCAL)) out.push({ value: LOCAL, label: "Este nodo" })
    for (const p of data.peers) {
      if (taken.has(p.name.toLowerCase()) || taken.has(p.peer_id.toLowerCase())) continue
      out.push({ value: p.name, label: p.name })
    }
    return out
  }, [data, nodes])

  if (unavailable) {
    return (
      <Callout tone="info">
        Este nodo no está en una federación (o no tienes permiso para cambiar el reparto), así que el modelo siempre se
        sirve aquí.
      </Callout>
    )
  }
  if (!data) {
    return (
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <Loader2 size={14} className="animate-spin" /> Cargando…
      </div>
    )
  }

  const move = (i: number, d: -1 | 1) =>
    setNodes((ns) => {
      const j = i + d
      if (j < 0 || j >= ns.length) return ns
      const next = [...ns]
      ;[next[i], next[j]] = [next[j], next[i]]
      return next
    })

  const save = async () => {
    if (!modelId) return
    if (strategy === "prefer" && nodes.length === 0) {
      toast.error("Elige al menos un nodo")
      return
    }
    setSaving(true)
    try {
      apply(await api.federation.setStrategy(modelId, { strategy, nodes: strategy === "prefer" ? nodes : [], fallback }))
      toast.success("Reparto guardado")
    } catch (err) {
      toast.error(err instanceof Error ? err.message : String(err))
    } finally {
      setSaving(false)
    }
  }

  const peersWithIt = data.peers.filter((p) => p.model != null)
  return (
    <div className="space-y-4">
      <Panel
        title="Dónde se sirve"
        subtitle="Entre este nodo y los de la federación que tienen el modelo. Se aplica a todas sus peticiones, sea cual sea el peldaño."
        actions={
          !readOnly && (
            <button
              type="button"
              disabled={!dirty || saving}
              onClick={() => void save()}
              className="inline-flex items-center gap-1.5 rounded-md bg-primary px-3 py-1.5 text-xs text-primary-foreground disabled:opacity-50"
            >
              {saving && <Loader2 size={12} className="animate-spin" />} Guardar
            </button>
          )
        }
      >
        <div className="space-y-3">
          <Segmented<Strategy>
            value={strategy}
            disabled={readOnly}
            testId="federation-strategy"
            options={[
              { value: "auto", label: "Auto" },
              { value: "local_only", label: "Solo este nodo" },
              { value: "prefer", label: "Preferir nodos" },
            ]}
            onChange={(v) => {
              setStrategy(v)
              if (v === "prefer" && nodes.length === 0 && peersWithIt[0]) setNodes([peersWithIt[0].name, ...(data.local ? [LOCAL] : [])])
            }}
          />
          <Hint>{STRATEGY_HELP[strategy]}</Hint>

          {strategy === "prefer" && (
            <div className="space-y-2">
              {nodes.length === 0 && <Hint tone="warn">Añade al menos un nodo.</Hint>}
              <ol className="space-y-1">
                {nodes.map((node, i) => {
                  const st = nodeState(node, data)
                  return (
                    <li key={node} className="flex items-center gap-2 rounded-md border border-border px-2 py-1 text-sm">
                      <span className="w-5 text-xs text-muted-foreground">{i + 1}.</span>
                      <span className="font-medium">{node === LOCAL ? "Este nodo" : node}</span>
                      <Chip tone={st.tone}>{st.label}</Chip>
                      {!readOnly && (
                        <span className="ml-auto flex gap-1">
                          <button type="button" aria-label="Subir" className="rounded p-1 hover:bg-muted disabled:opacity-30" disabled={i === 0} onClick={() => move(i, -1)}>
                            <ArrowUp size={12} />
                          </button>
                          <button type="button" aria-label="Bajar" className="rounded p-1 hover:bg-muted disabled:opacity-30" disabled={i === nodes.length - 1} onClick={() => move(i, 1)}>
                            <ArrowDown size={12} />
                          </button>
                          <button type="button" aria-label="Quitar" className="rounded p-1 hover:bg-muted" onClick={() => setNodes((ns) => ns.filter((n) => n !== node))}>
                            <X size={12} />
                          </button>
                        </span>
                      )}
                    </li>
                  )
                })}
              </ol>
              {!readOnly && addable.length > 0 && (
                <div className="flex items-center gap-2">
                  <select className={`${inputCls} w-56`} value={adding} onChange={(e) => setAdding(e.target.value)} aria-label="Nodo a añadir">
                    <option value="">Añadir nodo…</option>
                    {addable.map((o) => (
                      <option key={o.value} value={o.value}>
                        {o.label}
                      </option>
                    ))}
                  </select>
                  <button
                    type="button"
                    disabled={!adding}
                    onClick={() => {
                      setNodes((ns) => [...ns, adding])
                      setAdding("")
                    }}
                    className="inline-flex items-center gap-1 rounded-md border border-border px-2 py-1 text-xs hover:bg-muted disabled:opacity-50"
                  >
                    <Plus size={12} /> Añadir
                  </button>
                </div>
              )}
              <Toggle
                checked={fallback}
                disabled={readOnly}
                onChange={setFallback}
                label={<span className="text-xs">Si ninguno puede servirlo, seguir como Auto (si no, responde 503)</span>}
              />
            </div>
          )}
        </div>
      </Panel>

      <Panel title="Quién lo tiene">
        {data.peers.length === 0 ? (
          <Hint>No hay nodos en la federación.</Hint>
        ) : (
          <ul className="space-y-1 text-sm">
            <li className="flex items-center gap-2">
              <span className="font-medium">Este nodo</span>
              <Chip tone={data.local ? "ok" : "warn"}>{data.local ? "lo tiene" : "no lo tiene"}</Chip>
            </li>
            {data.peers.map((p) => {
              const st = nodeState(p.name, data)
              return (
                <li key={p.peer_id} className="flex items-center gap-2">
                  <span className="font-medium">{p.name}</span>
                  <Chip tone={st.tone}>{st.label}</Chip>
                </li>
              )
            })}
          </ul>
        )}
      </Panel>
    </div>
  )
}
