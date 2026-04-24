import { useEffect, useMemo, useState, type FormEvent } from "react"
import * as Dialog from "@radix-ui/react-dialog"
import { AlertCircle, Pencil, Plus, RefreshCw, Trash2, X, Zap } from "lucide-react"
import { toast } from "sonner"
import { api } from "@/api/client"
import { mcpApi } from "@/api/mcp"
import { useIsAdmin, useIsModelManager } from "@/hooks/useAuth"
import { useMCPStore } from "@/stores/mcpStore"
import type {
  Group,
} from "@/types"
import type {
  MCPAuthType,
  MCPServer,
  MCPServerCreate,
  MCPTransport,
  MCPToolSpec,
} from "@/types/agents"

const TRANSPORTS: MCPTransport[] = ["http", "sse", "stdio"]
const AUTH_TYPES: MCPAuthType[] = ["none", "api_key", "bearer", "basic", "oauth2"]

function HealthBadge({ server }: { server: MCPServer }) {
  const map: Record<MCPServer["healthStatus"], { label: string; cls: string }> = {
    healthy: { label: "healthy", cls: "bg-emerald-500/20 text-emerald-200" },
    unhealthy: { label: "unhealthy", cls: "bg-red-500/20 text-red-200" },
    unknown: { label: "unknown", cls: "bg-muted text-muted-foreground" },
  }
  const entry = map[server.healthStatus]
  return (
    <span className={`rounded-md px-2 py-0.5 text-xs ${entry.cls}`} title={server.lastError ?? undefined}>
      {entry.label}
    </span>
  )
}

interface ServerFormProps {
  open: boolean
  onClose: () => void
  initial: MCPServer | null
  groups: Group[]
  isAdmin: boolean
}

function ServerFormModal({ open, onClose, initial, groups, isAdmin }: ServerFormProps) {
  const create = useMCPStore((s) => s.create)
  const update = useMCPStore((s) => s.update)

  const [alias, setAlias] = useState("")
  const [displayName, setDisplayName] = useState("")
  const [description, setDescription] = useState("")
  const [transport, setTransport] = useState<MCPTransport>("http")
  const [url, setUrl] = useState("")
  const [command, setCommand] = useState("")
  const [argsText, setArgsText] = useState("")
  const [envText, setEnvText] = useState("")
  const [authType, setAuthType] = useState<MCPAuthType>("none")
  const [authValue, setAuthValue] = useState("")
  const [allowedToolsText, setAllowedToolsText] = useState("")
  const [allowedToolsSet, setAllowedToolsSet] = useState<Set<string>>(new Set())
  const [groupId, setGroupId] = useState<string>("")
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    if (!open) return
    if (initial) {
      setAlias(initial.alias)
      setDisplayName(initial.displayName)
      setDescription(initial.description ?? "")
      setTransport(initial.transport)
      setUrl(initial.url ?? "")
      setCommand(initial.command ?? "")
      setArgsText((initial.args ?? []).join(" "))
      setEnvText(
        initial.env
          ? Object.entries(initial.env)
              .map(([k, v]) => `${k}=${v}`)
              .join("\n")
          : "",
      )
      setAuthType(initial.authType)
      setAuthValue("")
      const allowed = initial.allowedTools ?? []
      setAllowedToolsText(allowed.join(", "))
      setAllowedToolsSet(new Set(allowed))
      setGroupId(initial.groupId ?? "")
    } else {
      setAlias("")
      setDisplayName("")
      setDescription("")
      setTransport("http")
      setUrl("")
      setCommand("")
      setArgsText("")
      setEnvText("")
      setAuthType("none")
      setAuthValue("")
      setAllowedToolsText("")
      setAllowedToolsSet(new Set())
      setGroupId("")
    }
    setErr(null)
  }, [open, initial])

  const cachedTools = initial?.toolsCache ?? null
  const useChecklist = (cachedTools?.length ?? 0) > 0

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    if (!alias.trim() || !displayName.trim()) {
      setErr("Alias y nombre son obligatorios.")
      return
    }
    if (transport === "stdio" && !isAdmin && !initial) {
      setErr("Sólo system_admin puede crear servidores stdio.")
      return
    }
    let allowedTools: string[] | null = null
    if (useChecklist) {
      allowedTools = [...allowedToolsSet]
      if (allowedTools.length === 0) allowedTools = null
    } else {
      const parsed = allowedToolsText
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean)
      allowedTools = parsed.length > 0 ? parsed : null
    }
    const argsList =
      transport === "stdio"
        ? argsText
            .split(/\s+/)
            .map((s) => s.trim())
            .filter(Boolean)
        : null
    let envObj: Record<string, string> | null = null
    if (transport === "stdio" && envText.trim()) {
      envObj = {}
      for (const line of envText.split("\n")) {
        const idx = line.indexOf("=")
        if (idx <= 0) continue
        const k = line.slice(0, idx).trim()
        const v = line.slice(idx + 1).trim()
        if (k) envObj[k] = v
      }
    }

    const payload: MCPServerCreate = {
      alias: alias.trim(),
      displayName: displayName.trim(),
      description: description.trim() || null,
      transport,
      url: transport === "stdio" ? null : url.trim() || null,
      command: transport === "stdio" ? command.trim() || null : null,
      args: argsList,
      env: envObj,
      authType,
      authValue: authValue || null,
      allowedTools,
      groupId: groupId || null,
    }

    setBusy(true)
    setErr(null)
    try {
      if (initial) {
        await update(initial.id, payload)
        toast.success(`Servidor "${displayName}" actualizado`)
      } else {
        await create(payload)
        toast.success(`Servidor "${displayName}" creado`)
      }
      onClose()
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Error al guardar")
    } finally {
      setBusy(false)
    }
  }

  return (
    <Dialog.Root open={open} onOpenChange={(v) => !v && onClose()}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-40 bg-black/50" />
        <Dialog.Content className="fixed left-1/2 top-1/2 z-50 w-full max-w-2xl max-h-[90vh] -translate-x-1/2 -translate-y-1/2 overflow-auto rounded-lg border border-border bg-card p-6 shadow-xl">
          <div className="mb-4 flex items-center justify-between">
            <Dialog.Title className="text-lg font-semibold">
              {initial ? `Editar ${initial.alias}` : "Nuevo MCP server"}
            </Dialog.Title>
            <Dialog.Close asChild>
              <button type="button" className="rounded p-1 hover:bg-muted" onClick={onClose}>
                <X size={16} />
              </button>
            </Dialog.Close>
          </div>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="grid gap-3 md:grid-cols-2">
              <div>
                <label className="mb-1 block text-sm font-medium" htmlFor="mcp-alias">
                  Alias *
                </label>
                <input
                  id="mcp-alias"
                  type="text"
                  value={alias}
                  onChange={(e) => setAlias(e.target.value)}
                  placeholder="github"
                  className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                  disabled={Boolean(initial)}
                />
              </div>
              <div>
                <label className="mb-1 block text-sm font-medium" htmlFor="mcp-display-name">
                  Display name *
                </label>
                <input
                  id="mcp-display-name"
                  type="text"
                  value={displayName}
                  onChange={(e) => setDisplayName(e.target.value)}
                  className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                />
              </div>
            </div>

            <div>
              <label className="mb-1 block text-sm font-medium" htmlFor="mcp-desc">
                Descripción
              </label>
              <textarea
                id="mcp-desc"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                rows={2}
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
              />
            </div>

            <div className="grid gap-3 md:grid-cols-2">
              <div>
                <label className="mb-1 block text-sm font-medium" htmlFor="mcp-transport">
                  Transport *
                </label>
                <select
                  id="mcp-transport"
                  value={transport}
                  onChange={(e) => setTransport(e.target.value as MCPTransport)}
                  className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                >
                  {TRANSPORTS.map((t) => (
                    <option key={t} value={t} disabled={t === "stdio" && !isAdmin}>
                      {t}
                      {t === "stdio" && !isAdmin ? " (system_admin only)" : ""}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="mb-1 block text-sm font-medium" htmlFor="mcp-group">
                  Grupo
                </label>
                <select
                  id="mcp-group"
                  value={groupId}
                  onChange={(e) => setGroupId(e.target.value)}
                  className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                >
                  <option value="">— sin grupo —</option>
                  {groups.map((g) => (
                    <option key={g.id} value={g.id}>
                      {g.name}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {(transport === "http" || transport === "sse") && (
              <div>
                <label className="mb-1 block text-sm font-medium" htmlFor="mcp-url">
                  URL *
                </label>
                <input
                  id="mcp-url"
                  type="url"
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  placeholder="https://mcp.example.com/github"
                  className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                />
              </div>
            )}

            {transport === "stdio" && (
              <div className="space-y-3 rounded-md border border-border bg-background/60 p-3">
                <div>
                  <label className="mb-1 block text-sm font-medium" htmlFor="mcp-cmd">
                    Command *
                  </label>
                  <input
                    id="mcp-cmd"
                    type="text"
                    value={command}
                    onChange={(e) => setCommand(e.target.value)}
                    placeholder="uvx"
                    className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm font-mono"
                  />
                </div>
                <div>
                  <label className="mb-1 block text-sm font-medium" htmlFor="mcp-args">
                    Args (separados por espacio)
                  </label>
                  <input
                    id="mcp-args"
                    type="text"
                    value={argsText}
                    onChange={(e) => setArgsText(e.target.value)}
                    placeholder="mcp-server-filesystem /data"
                    className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm font-mono"
                  />
                </div>
                <div>
                  <label className="mb-1 block text-sm font-medium" htmlFor="mcp-env">
                    Env (KEY=value, uno por línea)
                  </label>
                  <textarea
                    id="mcp-env"
                    value={envText}
                    onChange={(e) => setEnvText(e.target.value)}
                    rows={3}
                    className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm font-mono"
                  />
                </div>
              </div>
            )}

            <div className="grid gap-3 md:grid-cols-2">
              <div>
                <label className="mb-1 block text-sm font-medium" htmlFor="mcp-auth-type">
                  Auth type
                </label>
                <select
                  id="mcp-auth-type"
                  value={authType}
                  onChange={(e) => setAuthType(e.target.value as MCPAuthType)}
                  className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                >
                  {AUTH_TYPES.map((a) => (
                    <option key={a} value={a}>
                      {a}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="mb-1 block text-sm font-medium" htmlFor="mcp-auth-value">
                  Auth value {initial ? "(dejar vacío para no cambiar)" : ""}
                </label>
                <input
                  id="mcp-auth-value"
                  type="password"
                  value={authValue}
                  onChange={(e) => setAuthValue(e.target.value)}
                  autoComplete="new-password"
                  disabled={authType === "none"}
                  className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm font-mono"
                />
              </div>
            </div>

            <div>
              <label className="mb-1 block text-sm font-medium">Allowed tools</label>
              {useChecklist ? (
                <div className="max-h-48 overflow-y-auto rounded-md border border-border bg-background/60 p-2">
                  {(cachedTools ?? []).map((tool: MCPToolSpec) => {
                    const checked = allowedToolsSet.has(tool.name)
                    return (
                      <label
                        key={tool.name}
                        className="flex items-start gap-2 rounded px-2 py-1 hover:bg-muted/60"
                      >
                        <input
                          type="checkbox"
                          checked={checked}
                          onChange={(ev) => {
                            const next = new Set(allowedToolsSet)
                            if (ev.target.checked) next.add(tool.name)
                            else next.delete(tool.name)
                            setAllowedToolsSet(next)
                          }}
                          className="mt-0.5"
                        />
                        <div className="min-w-0">
                          <p className="font-mono text-xs">{tool.name}</p>
                          <p className="truncate text-xs text-muted-foreground">{tool.description}</p>
                        </div>
                      </label>
                    )
                  })}
                  <p className="mt-1 text-xs text-muted-foreground">
                    Sin selección = todas las tools están permitidas.
                  </p>
                </div>
              ) : (
                <>
                  <input
                    type="text"
                    value={allowedToolsText}
                    onChange={(e) => setAllowedToolsText(e.target.value)}
                    placeholder="tool_one, tool_two"
                    className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm font-mono"
                  />
                  <p className="mt-1 text-xs text-muted-foreground">
                    Refresca las tools para habilitar la checklist. Vacío = todas las tools.
                  </p>
                </>
              )}
            </div>

            {err && (
              <div className="rounded-md border border-red-500/40 bg-red-500/10 p-2 text-sm text-red-200">
                {err}
              </div>
            )}

            <div className="flex justify-end gap-2">
              <button
                type="button"
                onClick={onClose}
                className="rounded-md border border-border px-4 py-2 text-sm hover:bg-muted"
              >
                Cancelar
              </button>
              <button
                type="submit"
                disabled={busy}
                className="rounded-md bg-primary px-4 py-2 text-sm text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
              >
                {busy ? "Guardando..." : initial ? "Guardar" : "Crear"}
              </button>
            </div>
          </form>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  )
}

export function MCPServers() {
  const servers = useMCPStore((s) => s.servers)
  const loading = useMCPStore((s) => s.loading)
  const usingMock = useMCPStore((s) => s.usingMock)
  const error = useMCPStore((s) => s.error)
  const fetchAll = useMCPStore((s) => s.fetchAll)
  const refresh = useMCPStore((s) => s.refresh)
  const remove = useMCPStore((s) => s.remove)

  const isAdmin = useIsAdmin()
  const isManager = useIsModelManager()

  const [modalOpen, setModalOpen] = useState(false)
  const [editing, setEditing] = useState<MCPServer | null>(null)
  const [groups, setGroups] = useState<Group[]>([])
  const [pendingTestId, setPendingTestId] = useState<string | null>(null)

  useEffect(() => {
    void fetchAll()
    void api.groups
      .list()
      .then(setGroups)
      .catch(() => setGroups([]))
  }, [fetchAll])

  const openCreate = () => {
    setEditing(null)
    setModalOpen(true)
  }
  const openEdit = (s: MCPServer) => {
    setEditing(s)
    setModalOpen(true)
  }

  const handleTest = async (server: MCPServer) => {
    setPendingTestId(server.id)
    try {
      if (usingMock) {
        await new Promise((r) => window.setTimeout(r, 300))
        toast.success(`${server.alias}: healthy (mock) · ${server.toolsCache?.length ?? 0} tools`)
      } else {
        const res = await mcpApi.test(server.id)
        if (res.healthy) {
          toast.success(`${server.alias}: healthy · ${res.toolsCount} tools`)
        } else {
          toast.error(`${server.alias}: ${res.error ?? "unhealthy"}`)
        }
      }
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Test fallido")
    } finally {
      setPendingTestId(null)
    }
  }

  const handleRefresh = async (server: MCPServer) => {
    try {
      await refresh(server.id)
      toast.success(`${server.alias}: tools refrescadas`)
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Refresh fallido")
    }
  }

  const handleDelete = async (server: MCPServer) => {
    if (!window.confirm(`¿Borrar MCP server "${server.alias}"?`)) return
    try {
      await remove(server.id)
      toast.success(`${server.alias} borrado`)
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Error al borrar")
    }
  }

  const sortedGroups = useMemo(() => [...groups].sort((a, b) => a.name.localeCompare(b.name)), [groups])

  return (
    <div className="space-y-4 pb-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold">MCP Servers</h1>
          <p className="text-muted-foreground">
            Servidores MCP registrados que los agentes pueden consumir.
          </p>
        </div>
        <div className="flex gap-2">
          <button
            type="button"
            onClick={() => void fetchAll()}
            className="inline-flex items-center gap-1.5 rounded-md border border-border px-3 py-1.5 text-sm text-muted-foreground hover:bg-muted hover:text-foreground"
          >
            <RefreshCw size={14} />
            Recargar
          </button>
          {isManager && (
            <button
              type="button"
              onClick={openCreate}
              className="inline-flex items-center gap-1.5 rounded-md bg-primary px-3 py-1.5 text-sm text-primary-foreground hover:bg-primary/90"
            >
              <Plus size={14} />
              Nuevo
            </button>
          )}
        </div>
      </div>

      {usingMock && (
        <div className="flex items-start gap-2 rounded-md border border-amber-500/40 bg-amber-500/10 p-3 text-sm text-amber-100">
          <AlertCircle size={16} className="mt-0.5 shrink-0" />
          <div>
            <span className="font-semibold">Modo mock:</span> la API{" "}
            <code className="font-mono">/ocabra/mcp-servers</code> todavía no existe. Se muestran
            datos de ejemplo hasta que Stream A mergee.
          </div>
        </div>
      )}

      {error && !usingMock && (
        <div className="flex items-start gap-2 rounded-md border border-red-500/40 bg-red-500/10 p-3 text-sm text-red-100">
          <AlertCircle size={16} className="mt-0.5 shrink-0" />
          <div>
            <span className="font-semibold">Error:</span> {error}
          </div>
        </div>
      )}

      {loading && servers.length === 0 ? (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          {Array.from({ length: 3 }).map((_, idx) => (
            <div
              key={idx}
              className="h-40 animate-pulse rounded-lg border border-border bg-muted/30"
            />
          ))}
        </div>
      ) : servers.length === 0 ? (
        <div className="rounded-md border border-dashed border-border p-10 text-center text-sm text-muted-foreground">
          No hay MCP servers registrados.
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          {servers.map((server) => (
            <div
              key={server.id}
              className="flex flex-col gap-3 rounded-lg border border-border bg-card p-4"
            >
              <div className="flex items-start justify-between gap-2">
                <div className="min-w-0">
                  <div className="flex items-center gap-2">
                    <h3 className="truncate text-base font-semibold">{server.displayName}</h3>
                    <HealthBadge server={server} />
                  </div>
                  <p className="truncate font-mono text-xs text-muted-foreground">
                    {server.alias} · {server.transport}
                  </p>
                </div>
                <div className="flex shrink-0 gap-1">
                  {isManager && (
                    <button
                      type="button"
                      onClick={() => openEdit(server)}
                      className="rounded-md border border-border p-1.5 text-muted-foreground hover:bg-muted hover:text-foreground"
                      title="Editar"
                    >
                      <Pencil size={14} />
                    </button>
                  )}
                  {isManager && (
                    <button
                      type="button"
                      onClick={() => void handleDelete(server)}
                      className="rounded-md border border-red-500/40 p-1.5 text-red-200 hover:bg-red-500/10"
                      title="Borrar"
                    >
                      <Trash2 size={14} />
                    </button>
                  )}
                </div>
              </div>

              {server.description && (
                <p className="text-sm text-muted-foreground">{server.description}</p>
              )}

              <div className="flex flex-wrap gap-2 text-xs text-muted-foreground">
                <span className="rounded-md bg-muted px-2 py-0.5">
                  {server.toolsCache?.length ?? 0} tools cacheadas
                </span>
                {server.allowedTools && server.allowedTools.length > 0 && (
                  <span className="rounded-md bg-muted px-2 py-0.5">
                    Allowlist: {server.allowedTools.length}
                  </span>
                )}
                {server.groupName && (
                  <span className="rounded-md bg-muted px-2 py-0.5">{server.groupName}</span>
                )}
              </div>

              {server.lastError && (
                <p className="truncate rounded-md bg-red-500/10 px-2 py-1 font-mono text-xs text-red-200">
                  {server.lastError}
                </p>
              )}

              <div className="flex gap-2 border-t border-border pt-3">
                <button
                  type="button"
                  onClick={() => void handleRefresh(server)}
                  className="inline-flex items-center gap-1 rounded-md border border-border px-3 py-1 text-sm text-muted-foreground hover:bg-muted hover:text-foreground"
                >
                  <RefreshCw size={12} />
                  Refresh tools
                </button>
                <button
                  type="button"
                  onClick={() => void handleTest(server)}
                  disabled={pendingTestId === server.id}
                  className="inline-flex items-center gap-1 rounded-md border border-primary/40 bg-primary/10 px-3 py-1 text-sm text-primary hover:bg-primary/20 disabled:opacity-50"
                >
                  <Zap size={12} />
                  {pendingTestId === server.id ? "Probando..." : "Test connection"}
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      <ServerFormModal
        open={modalOpen}
        onClose={() => setModalOpen(false)}
        initial={editing}
        groups={sortedGroups}
        isAdmin={isAdmin}
      />
    </div>
  )
}
