import { useEffect, useState } from "react"
import { Network } from "lucide-react"
import { api } from "@/api/client"
import { getRemoteNode } from "@/lib/remoteNode"
import { setPlaygroundNode, usePlaygroundNode } from "@/lib/playgroundNode"

/**
 * Which federation node runs the Playground's requests: automatic (normal
 * routing), this server, or a peer that is online. Hidden when there are no
 * peers (or they can't be listed with this role).
 */
export function NodeSelector() {
  const node = usePlaygroundNode()
  const [peers, setPeers] = useState<string[] | null>(null)

  useEffect(() => {
    if (getRemoteNode()) return
    api.federation
      .getPeers()
      .then((list) => setPeers(list.filter((p) => p.enabled && p.online).map((p) => p.name)))
      .catch(() => setPeers(null))
  }, [])

  if (!peers || peers.length === 0) {
    return null
  }
  const options = ["auto", "local", ...peers]
  return (
    <label
      className="inline-flex items-center gap-1.5 rounded-md border border-border px-2 py-1 text-sm text-muted-foreground"
      title="Nodo de la federación que ejecuta las peticiones del Playground (cabecera X-Ocabra-Node)"
    >
      <Network size={14} />
      <select
        className="bg-transparent text-foreground outline-none"
        value={options.includes(node) ? node : "auto"}
        onChange={(e) => setPlaygroundNode(e.target.value)}
        aria-label="Nodo"
      >
        <option value="auto">Nodo: automático</option>
        <option value="local">Nodo: este servidor</option>
        {peers.map((p) => (
          <option key={p} value={p}>
            Nodo: {p}
          </option>
        ))}
      </select>
    </label>
  )
}
