import { useEffect, useMemo, useState, type FormEvent } from "react"
import * as Dialog from "@radix-ui/react-dialog"
import { AlertCircle, Pencil, Plus, RefreshCw, Sparkles, Trash2, X, Zap } from "lucide-react"
import { toast } from "sonner"
import { api } from "@/api/client"
import { agentsApi } from "@/api/agents"
import { useIsModelManager } from "@/hooks/useAuth"
import { useAgentsStore } from "@/stores/agentsStore"
import { useMCPStore } from "@/stores/mcpStore"
import type { Group, ModelProfile, ModelState } from "@/types"
import type {
  Agent,
  AgentCreate,
  AgentMCPBinding,
  AgentRequireApproval,
  AgentToolChoice,
  MCPServer,
} from "@/types/agents"

const TOOL_CHOICES: AgentToolChoice[] = ["auto", "required", "none"]
const APPROVAL_MODES: AgentRequireApproval[] = ["never", "always"]
const MAX_SYSTEM_PROMPT = 8000

interface AgentFormProps {
  open: boolean
  onClose: () => void
  initial: Agent | null
  models: ModelState[]
  servers: MCPServer[]
  groups: Group[]
}

function AgentFormModal({ open, onClose, initial, models, servers, groups }: AgentFormProps) {
  const create = useAgentsStore((s) => s.create)
  const update = useAgentsStore((s) => s.update)

  const [slug, setSlug] = useState("")
  const [displayName, setDisplayName] = useState("")
  const [description, setDescription] = useState("")
  const [baseModelId, setBaseModelId] = useState<string>("")
  const [profileId, setProfileId] = useState<string>("")
  const [profiles, setProfiles] = useState<ModelProfile[]>([])
  const [systemPrompt, setSystemPrompt] = useState("")
  const [toolChoice, setToolChoice] = useState<AgentToolChoice>("auto")
  const [requireApproval, setRequireApproval] = useState<AgentRequireApproval>("never")
  const [maxToolHops, setMaxToolHops] = useState(8)
  const [toolTimeout, setToolTimeout] = useState(60)
  const [groupId, setGroupId] = useState<string>("")
  const [bindings, setBindings] = useState<AgentMCPBinding[]>([])
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    if (!open) return
    if (initial) {
      setSlug(initial.slug)
      setDisplayName(initial.displayName)
      setDescription(initial.description ?? "")
      setBaseModelId(initial.baseModelId ?? "")
      setProfileId(initial.profileId ?? "")
      setSystemPrompt(initial.systemPrompt)
      setToolChoice(initial.toolChoiceDefault)
      setRequireApproval(initial.requireApproval)
      setMaxToolHops(initial.maxToolHops)
      setToolTimeout(initial.toolTimeoutSeconds)
      setGroupId(initial.groupId ?? "")
      setBindings(initial.mcpServers)
    } else {
      setSlug("")
      setDisplayName("")
      setDescription("")
      setBaseModelId("")
      setProfileId("")
      setSystemPrompt("")
      setToolChoice("auto")
      setRequireApproval("never")
      setMaxToolHops(8)
      setToolTimeout(60)
      setGroupId("")
      setBindings([])
    }
    setErr(null)
  }, [open, initial])

  // Load profiles when base model changes.
  useEffect(() => {
    if (!open || !baseModelId) {
      setProfiles([])
      return
    }
    let active = true
    api.profiles
      .listByModel(baseModelId)
      .then((list) => {
        if (active) setProfiles(list)
      })
      .catch(() => {
        if (active) setProfiles([])
      })
    return () => {
      active = false
    }
  }, [open, baseModelId])

  const bindingMap = useMemo(() => {
    const m = new Map<string, AgentMCPBinding>()
    bindings.forEach((b) => m.set(b.mcpServerId, b))
    return m
  }, [bindings])

  function toggleServer(serverId: string) {
    const existing = bindingMap.get(serverId)
    if (existing) {
      setBindings(bindings.filter((b) => b.mcpServerId !== serverId))
    } else {
      setBindings([...bindings, { mcpServerId: serverId, allowedTools: null }])
    }
  }

  function toggleTool(serverId: string, toolName: string) {
    const existing = bindingMap.get(serverId)
    if (!existing) return
    const current = existing.allowedTools ?? []
    const has = current.includes(toolName)
    const nextTools = has ? current.filter((t) => t !== toolName) : [...current, toolName]
    // allowedTools=null → inherit (all); [] → explicit no-tools? Use null when we clear.
    const nextBinding: AgentMCPBinding = {
      mcpServerId: serverId,
      allowedTools: nextTools.length === 0 ? null : nextTools,
    }
    setBindings(bindings.map((b) => (b.mcpServerId === serverId ? nextBinding : b)))
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    if (!slug.trim() || !displayName.trim()) {
      setErr("Slug y nombre son obligatorios.")
      return
    }
    if (!baseModelId && !profileId) {
      setErr("Selecciona un base_model o un profile.")
      return
    }
    if (baseModelId && profileId) {
      setErr("Sólo se puede elegir uno: base_model o profile.")
      return
    }
    if (!systemPrompt.trim()) {
      setErr("El system prompt es obligatorio.")
      return
    }
    if (systemPrompt.length > MAX_SYSTEM_PROMPT) {
      setErr(`System prompt demasiado largo (${systemPrompt.length} > ${MAX_SYSTEM_PROMPT}).`)
      return
    }

    const payload: AgentCreate = {
      slug: slug.trim(),
      displayName: displayName.trim(),
      description: description.trim() || null,
      baseModelId: baseModelId || null,
      profileId: profileId || null,
      systemPrompt,
      toolChoiceDefault: toolChoice,
      maxToolHops,
      toolTimeoutSeconds: toolTimeout,
      requireApproval,
      requestDefaults: null,
      groupId: groupId || null,
      mcpServers: bindings,
    }

    setBusy(true)
    setErr(null)
    try {
      if (initial) {
        await update(initial.slug, payload)
        toast.success(`Agent "${displayName}" actualizado`)
      } else {
        await create(payload)
        toast.success(`Agent "${displayName}" creado`)
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
        <Dialog.Content className="fixed left-1/2 top-1/2 z-50 w-full max-w-3xl max-h-[92vh] -translate-x-1/2 -translate-y-1/2 overflow-auto rounded-lg border border-border bg-card p-6 shadow-xl">
          <div className="mb-4 flex items-center justify-between">
            <Dialog.Title className="text-lg font-semibold">
              {initial ? `Editar ${initial.slug}` : "Nuevo agent"}
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
                <label className="mb-1 block text-sm font-medium" htmlFor="agent-slug">
                  Slug *
                </label>
                <input
                  id="agent-slug"
                  type="text"
                  value={slug}
                  onChange={(e) => setSlug(e.target.value)}
                  placeholder="research-bot"
                  disabled={Boolean(initial)}
                  className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm font-mono"
                />
              </div>
              <div>
                <label className="mb-1 block text-sm font-medium" htmlFor="agent-name">
                  Display name *
                </label>
                <input
                  id="agent-name"
                  type="text"
                  value={displayName}
                  onChange={(e) => setDisplayName(e.target.value)}
                  className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                />
              </div>
            </div>

            <div>
              <label className="mb-1 block text-sm font-medium" htmlFor="agent-desc">
                Descripción
              </label>
              <textarea
                id="agent-desc"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                rows={2}
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
              />
            </div>

            <div className="grid gap-3 md:grid-cols-2">
              <div>
                <label className="mb-1 block text-sm font-medium" htmlFor="agent-base-model">
                  Base model
                </label>
                <select
                  id="agent-base-model"
                  value={baseModelId}
                  onChange={(e) => {
                    setBaseModelId(e.target.value)
                    setProfileId("")
                  }}
                  className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                >
                  <option value="">— elegir —</option>
                  {models.map((m) => (
                    <option key={m.modelId} value={m.modelId}>
                      {m.displayName}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="mb-1 block text-sm font-medium" htmlFor="agent-profile">
                  Profile (alternativa)
                </label>
                <select
                  id="agent-profile"
                  value={profileId}
                  onChange={(e) => {
                    setProfileId(e.target.value)
                    if (e.target.value) setBaseModelId("")
                  }}
                  className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                  disabled={!baseModelId && profiles.length === 0}
                >
                  <option value="">— ninguno —</option>
                  {profiles.map((p) => (
                    <option key={p.profileId} value={p.profileId}>
                      {p.displayName}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div>
              <label className="mb-1 block text-sm font-medium" htmlFor="agent-prompt">
                System prompt *{" "}
                <span className="font-normal text-muted-foreground">
                  ({systemPrompt.length}/{MAX_SYSTEM_PROMPT})
                </span>
              </label>
              <textarea
                id="agent-prompt"
                value={systemPrompt}
                onChange={(e) => setSystemPrompt(e.target.value)}
                rows={6}
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm font-mono"
              />
            </div>

            <div className="grid gap-3 md:grid-cols-3">
              <div>
                <label className="mb-1 block text-sm font-medium" htmlFor="agent-tool-choice">
                  Tool choice default
                </label>
                <select
                  id="agent-tool-choice"
                  value={toolChoice}
                  onChange={(e) => setToolChoice(e.target.value as AgentToolChoice)}
                  className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                >
                  {TOOL_CHOICES.map((v) => (
                    <option key={v} value={v}>
                      {v}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="mb-1 block text-sm font-medium" htmlFor="agent-approval">
                  Require approval
                </label>
                <select
                  id="agent-approval"
                  value={requireApproval}
                  onChange={(e) => setRequireApproval(e.target.value as AgentRequireApproval)}
                  className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                >
                  {APPROVAL_MODES.map((v) => (
                    <option key={v} value={v}>
                      {v}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="mb-1 block text-sm font-medium" htmlFor="agent-group">
                  Grupo
                </label>
                <select
                  id="agent-group"
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

            <div className="grid gap-3 md:grid-cols-2">
              <div>
                <label className="mb-1 block text-sm font-medium" htmlFor="agent-hops">
                  Max tool hops: <span className="font-mono">{maxToolHops}</span>
                </label>
                <input
                  id="agent-hops"
                  type="range"
                  min={1}
                  max={20}
                  step={1}
                  value={maxToolHops}
                  onChange={(e) => setMaxToolHops(Number(e.target.value))}
                  className="w-full"
                />
              </div>
              <div>
                <label className="mb-1 block text-sm font-medium" htmlFor="agent-timeout">
                  Tool timeout: <span className="font-mono">{toolTimeout}s</span>
                </label>
                <input
                  id="agent-timeout"
                  type="range"
                  min={5}
                  max={300}
                  step={5}
                  value={toolTimeout}
                  onChange={(e) => setToolTimeout(Number(e.target.value))}
                  className="w-full"
                />
              </div>
            </div>

            <div>
              <label className="mb-2 block text-sm font-medium">MCP servers</label>
              {servers.length === 0 ? (
                <p className="rounded-md border border-dashed border-border p-3 text-sm text-muted-foreground">
                  No hay MCP servers registrados todavía.
                </p>
              ) : (
                <div className="space-y-2">
                  {servers.map((server) => {
                    const binding = bindingMap.get(server.id)
                    const selected = Boolean(binding)
                    const tools = server.toolsCache ?? []
                    const allowed = binding?.allowedTools ?? null
                    return (
                      <div
                        key={server.id}
                        className="rounded-md border border-border bg-background/60 p-3"
                      >
                        <label className="flex items-center gap-2">
                          <input
                            type="checkbox"
                            checked={selected}
                            onChange={() => toggleServer(server.id)}
                          />
                          <span className="font-medium">{server.displayName}</span>
                          <span className="font-mono text-xs text-muted-foreground">
                            ({server.alias})
                          </span>
                          <span className="ml-auto text-xs text-muted-foreground">
                            {tools.length} tools
                          </span>
                        </label>
                        {selected && tools.length > 0 && (
                          <div className="mt-2 grid gap-1 pl-6 md:grid-cols-2">
                            {tools.map((tool) => {
                              const checked = allowed === null || allowed.includes(tool.name)
                              return (
                                <label
                                  key={tool.name}
                                  className="flex items-start gap-2 rounded px-2 py-1 text-sm hover:bg-muted/60"
                                >
                                  <input
                                    type="checkbox"
                                    checked={checked}
                                    onChange={() => toggleTool(server.id, tool.name)}
                                    className="mt-0.5"
                                  />
                                  <span className="min-w-0">
                                    <span className="block font-mono text-xs">{tool.name}</span>
                                    <span className="block truncate text-xs text-muted-foreground">
                                      {tool.description}
                                    </span>
                                  </span>
                                </label>
                              )
                            })}
                          </div>
                        )}
                        {selected && tools.length === 0 && (
                          <p className="mt-2 pl-6 text-xs text-muted-foreground">
                            No hay tools cacheadas — el agente hereda las permitidas del server.
                          </p>
                        )}
                      </div>
                    )
                  })}
                </div>
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

export function Agents() {
  const agents = useAgentsStore((s) => s.agents)
  const loading = useAgentsStore((s) => s.loading)
  const usingMock = useAgentsStore((s) => s.usingMock)
  const error = useAgentsStore((s) => s.error)
  const fetchAll = useAgentsStore((s) => s.fetchAll)
  const remove = useAgentsStore((s) => s.remove)

  const servers = useMCPStore((s) => s.servers)
  const fetchServers = useMCPStore((s) => s.fetchAll)

  const isManager = useIsModelManager()

  const [modalOpen, setModalOpen] = useState(false)
  const [editing, setEditing] = useState<Agent | null>(null)
  const [models, setModels] = useState<ModelState[]>([])
  const [groups, setGroups] = useState<Group[]>([])
  const [pendingTestSlug, setPendingTestSlug] = useState<string | null>(null)

  useEffect(() => {
    void fetchAll()
    void fetchServers()
    void api.models.list().then(setModels).catch(() => setModels([]))
    void api.groups.list().then(setGroups).catch(() => setGroups([]))
  }, [fetchAll, fetchServers])

  const serverNameById = useMemo(() => {
    const m = new Map<string, string>()
    servers.forEach((s) => m.set(s.id, s.alias))
    return m
  }, [servers])

  const openCreate = () => {
    setEditing(null)
    setModalOpen(true)
  }
  const openEdit = (a: Agent) => {
    setEditing(a)
    setModalOpen(true)
  }

  const handleTest = async (agent: Agent) => {
    setPendingTestSlug(agent.slug)
    try {
      if (usingMock) {
        await new Promise((r) => window.setTimeout(r, 300))
        toast.success(`${agent.slug}: healthy (mock) · ${agent.mcpServers.length} servers`)
      } else {
        const res = await agentsApi.test(agent.slug)
        if (res.healthy) {
          toast.success(`${agent.slug}: healthy · ${res.toolsCount} tools`)
        } else {
          const firstErr = res.errors[0] ?? res.servers.find((s) => !s.healthy)?.error ?? "unhealthy"
          toast.error(`${agent.slug}: ${firstErr}`)
        }
      }
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Test fallido")
    } finally {
      setPendingTestSlug(null)
    }
  }

  const handleDelete = async (agent: Agent) => {
    if (!window.confirm(`¿Borrar agent "${agent.slug}"?`)) return
    try {
      await remove(agent.slug)
      toast.success(`${agent.slug} borrado`)
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Error al borrar")
    }
  }

  return (
    <div className="space-y-4 pb-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="flex items-center gap-2 text-2xl font-semibold">
            <Sparkles size={20} className="text-primary" />
            Agents
          </h1>
          <p className="text-muted-foreground">
            Agentes = base_model + system prompt + MCP tools. Se invocan como{" "}
            <code className="rounded bg-muted px-1 font-mono">agent/&lt;slug&gt;</code>.
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
            <code className="font-mono">/ocabra/agents</code> todavía no existe. Se muestran datos
            de ejemplo hasta que Stream A mergee.
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

      {loading && agents.length === 0 ? (
        <div className="space-y-3">
          {Array.from({ length: 3 }).map((_, idx) => (
            <div key={idx} className="h-32 animate-pulse rounded-lg border border-border bg-muted/30" />
          ))}
        </div>
      ) : agents.length === 0 ? (
        <div className="rounded-md border border-dashed border-border p-10 text-center text-sm text-muted-foreground">
          No hay agents definidos.
        </div>
      ) : (
        <div className="space-y-3">
          {agents.map((agent) => (
            <div
              key={agent.slug}
              className="flex flex-col gap-3 rounded-lg border border-border bg-card p-4"
            >
              <div className="flex items-start justify-between gap-2">
                <div className="min-w-0">
                  <h3 className="flex items-center gap-2 text-base font-semibold">
                    <Sparkles size={16} className="text-primary" />
                    {agent.displayName}
                    <code className="rounded bg-muted px-1.5 py-0.5 font-mono text-xs text-muted-foreground">
                      agent/{agent.slug}
                    </code>
                  </h3>
                  {agent.description && (
                    <p className="mt-1 text-sm text-muted-foreground">{agent.description}</p>
                  )}
                </div>
                <div className="flex shrink-0 gap-1">
                  {isManager && (
                    <button
                      type="button"
                      onClick={() => openEdit(agent)}
                      className="rounded-md border border-border p-1.5 text-muted-foreground hover:bg-muted hover:text-foreground"
                      title="Editar"
                    >
                      <Pencil size={14} />
                    </button>
                  )}
                  {isManager && (
                    <button
                      type="button"
                      onClick={() => void handleDelete(agent)}
                      className="rounded-md border border-red-500/40 p-1.5 text-red-200 hover:bg-red-500/10"
                      title="Borrar"
                    >
                      <Trash2 size={14} />
                    </button>
                  )}
                </div>
              </div>

              <div className="flex flex-wrap gap-2 text-xs text-muted-foreground">
                {agent.baseModelId && (
                  <span className="rounded-md bg-muted px-2 py-0.5 font-mono">
                    {agent.baseModelId}
                  </span>
                )}
                {agent.profileId && (
                  <span className="rounded-md bg-muted px-2 py-0.5 font-mono">
                    profile: {agent.profileId}
                  </span>
                )}
                <span className="rounded-md bg-muted px-2 py-0.5">
                  hops {agent.maxToolHops} · {agent.toolTimeoutSeconds}s
                </span>
                <span className="rounded-md bg-muted px-2 py-0.5">
                  approval: {agent.requireApproval}
                </span>
                <span className="rounded-md bg-muted px-2 py-0.5">
                  tool_choice: {agent.toolChoiceDefault}
                </span>
                {agent.groupName && (
                  <span className="rounded-md bg-muted px-2 py-0.5">{agent.groupName}</span>
                )}
              </div>

              {agent.mcpServers.length > 0 && (
                <div className="flex flex-wrap gap-1 text-xs">
                  <span className="text-muted-foreground">MCP:</span>
                  {agent.mcpServers.map((binding) => (
                    <span
                      key={binding.mcpServerId}
                      className="rounded-md border border-border bg-background/60 px-2 py-0.5 font-mono"
                    >
                      {serverNameById.get(binding.mcpServerId) ?? binding.mcpServerId}
                      {binding.allowedTools ? ` (${binding.allowedTools.length})` : ""}
                    </span>
                  ))}
                </div>
              )}

              <div className="flex gap-2 border-t border-border pt-3">
                <button
                  type="button"
                  onClick={() => void handleTest(agent)}
                  disabled={pendingTestSlug === agent.slug}
                  className="inline-flex items-center gap-1 rounded-md border border-primary/40 bg-primary/10 px-3 py-1 text-sm text-primary hover:bg-primary/20 disabled:opacity-50"
                >
                  <Zap size={12} />
                  {pendingTestSlug === agent.slug ? "Probando..." : "Test"}
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      <AgentFormModal
        open={modalOpen}
        onClose={() => setModalOpen(false)}
        initial={editing}
        models={models}
        servers={servers}
        groups={groups}
      />
    </div>
  )
}
