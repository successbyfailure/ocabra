# oCabra — Plan de Agentes + MCP

Última actualización: 2026-04-24
Autor inicial: Claude (revisado con usuario 2026-04-24)

**Estado**: Plan listo para repartir. Ninguna fase iniciada.

---

## Concepto

Añadir a oCabra dos primitivas nuevas:

1. **MCP servers registrados**: oCabra actúa como *cliente* de servidores MCP externos
   (http/sse/stdio), los descubre, cachea sus tools y los expone al LLM en
   `/v1/chat/completions` en formato OpenAI tool-calling.
2. **Agentes**: un agente = `base_model` (o `profile`) + `system_prompt` +
   conjunto de MCP servers/tools permitidos + parámetros del tool-loop.
   Se invoca igual que un modelo: `model="agent/<slug>"` en `/v1/chat/completions`.

**Stateless**: sin historial persistido. Cada request del cliente trae el contexto
completo; oCabra ejecuta el tool-loop dentro del turno y devuelve un único
`assistant` message final (que puede contener `tool_calls` si `require_approval != "never"`,
pero por defecto se ejecutan hasta terminar).

**Sin sub-agentes v1**: un agente no puede invocar a otro agente. Queda marcado
como extensión futura (el schema ya reserva espacio con `max_tool_hops`).

**ACL**: CRUD de agentes y MCP servers sólo para `model_manager` y `system_admin`.
Invocación de un agente sigue el modelo de grupos existente (igual que `/v1/models`).

---

## Inspiración: qué copiamos de LiteLLM (sin importarlo)

Ver decisiones razonadas en la conversación del 2026-04-24. Resumen:

- **Namespace por servidor**: tools expuestas al LLM como `{alias}_{tool_name}` para
  evitar colisiones.
- **`allowed_tools` por request**: además del ACL persistido del agente, el caller
  puede restringir en cada llamada (mínimo privilegio).
- **`require_approval`** (`never` | `always` | lista de tools): controla el tool-loop.
- **`tools/list` cacheado con TTL** e invalidación explícita.
- **Dos superficies**: `/v1/chat/completions` (consumo LLM) y `/mcp/{alias}`
  (passthrough para clientes MCP puros — opcional, ver Fase 6).
- **Header-based per-request auth** (`x-mcp-{alias}-{header}`): inyecta credenciales
  del usuario final sin guardarlas en BD.
- **Audit granular**: tabla `tool_call_stats` ligada a `request_stats`.
- **Límites de seguridad**: `max_tool_hops`, `tool_timeout_seconds`, args redactados
  en logs, schema validation antes de invocar.

**Lo que NO copiamos**: LiteLLM Proxy completo, modelo de "teams" duplicado,
`/v1/responses` como primitiva (priorizamos `/v1/chat/completions`).

---

## Arquitectura

```
┌───────────────────────────────────────────────────────────────────┐
│  Cliente OpenAI-compat                                            │
│  POST /v1/chat/completions                                        │
│  { "model": "agent/research-bot", "messages": [...] }             │
└───────────────┬───────────────────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────────────────────────┐
│  api/openai/chat.py (existente, parcheado)                        │
│  ├─ resolve_profile() ← ya existe                                 │
│  ├─ resolve_agent()  ← NUEVO: detecta "agent/…" y delega          │
│  └─ si agent: delega a AgentExecutor                              │
└───────────────┬───────────────────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────────────────────────┐
│  agents/executor.py — Tool-loop                                    │
│                                                                    │
│  1. Cargar AgentSpec (cache)                                       │
│  2. Enriquecer request: system_prompt, tools                       │
│  3. Descubrir tools MCP (mcp_registry.get_tools(agent))            │
│  4. Loop:                                                          │
│     ├─ WorkerPool.forward(request)                                 │
│     ├─ Si no hay tool_calls → return assistant final              │
│     ├─ Si hops > max_tool_hops → error / truncate                  │
│     ├─ Validar args vs schema                                      │
│     ├─ Ejecutar tools en paralelo (mcp_registry.call_tool)         │
│     ├─ Registrar tool_call_stats                                   │
│     └─ Append tool results, repetir                                │
└───────┬────────────────────────┬──────────────────────────────────┘
        │                        │
        ▼                        ▼
┌──────────────────┐      ┌────────────────────────────────────────┐
│  WorkerPool       │      │  agents/mcp_registry.py                │
│  (existente)      │      │                                        │
│  Ejecuta el LLM   │      │  ├─ mcp_clients: {alias: MCPClient}    │
│  base_model       │      │  ├─ tool_cache: {alias: [tools]}       │
└──────────────────┘      │  ├─ get_tools_for_agent(agent)         │
                           │  ├─ call_tool(alias, name, args,       │
                           │  │             request_headers)         │
                           │  └─ health_check()                     │
                           └──────────┬─────────────────────────────┘
                                      │
                                      ▼
                           ┌───────────────────────────────────────┐
                           │  mcp SDK oficial (Anthropic)          │
                           │  httpx / SSE / asyncio.subprocess     │
                           └───────────────────────────────────────┘
```

### Decisiones de diseño clave

| Decisión | Elegido | Alternativa descartada |
|----------|---------|-----------------------|
| Cliente MCP | `mcp` SDK oficial de Anthropic | `litellm.experimental_mcp_client` (trae toda la dep de litellm) |
| Invocación del agente | `model="agent/<slug>"` en `/v1/chat/completions` | Endpoint nuevo `/v1/agents/…` (fragmentaría el contrato OpenAI) |
| Estado | Stateless, sin tabla de conversaciones | Persistir historial (contradice CLAUDE.md) |
| Ejecución de tools | Paralela con `asyncio.gather` + per-tool timeout | Secuencial (latencia innecesaria cuando hay varios tool_calls en un turno) |
| Cifrado de auth_value | Fernet (ya existe en federation) | Plaintext + TLS sólo |
| Transport `stdio` | Sólo para `system_admin`, con `cwd` y `env` sanitizados | Prohibido (útil para servidores locales de confianza) |
| `/v1/responses` | Fuera de scope v1 | Implementar (poca adopción fuera de Anthropic) |

---

## Modelo de datos

### Tabla `mcp_servers` (migración 0017)

```python
class MCPServer(Base):
    __tablename__ = "mcp_servers"

    id: Mapped[UUID] = primary_key
    alias: Mapped[str] = unique, indexed          # "github", "filesystem"
    display_name: Mapped[str]
    description: Mapped[str | None]
    transport: Mapped[str]                         # "http" | "sse" | "stdio"

    # HTTP/SSE
    url: Mapped[str | None]

    # stdio (sólo admin puede crear estos)
    command: Mapped[str | None]                    # e.g. "uvx"
    args: Mapped[list[str] | None]                 # JSONB
    env: Mapped[dict[str, str] | None]             # JSONB, NO se loguea

    # Auth
    auth_type: Mapped[str]                         # "none" | "api_key" | "bearer" | "basic" | "oauth2"
    auth_value_encrypted: Mapped[bytes | None]     # Fernet; header name + value serializados
    # Para OAuth2 (v2, futuro):
    oauth_config: Mapped[dict | None]              # issuer, client_id, scopes...

    # ACL server-level
    allowed_tools: Mapped[list[str] | None]        # None = todas; list = allowlist
    group_id: Mapped[UUID | None] = FK(groups.id, ondelete="SET NULL")

    # Runtime
    tools_cache: Mapped[list[dict] | None]         # JSONB: [{name, description, input_schema}]
    tools_cache_updated_at: Mapped[datetime | None]
    last_error: Mapped[str | None]
    health_status: Mapped[str]                     # "unknown" | "healthy" | "unhealthy"

    # Audit
    created_by: Mapped[UUID] = FK(users.id)
    created_at: Mapped[datetime]
    updated_at: Mapped[datetime]
```

### Tabla `agents` (migración 0017)

```python
class Agent(Base):
    __tablename__ = "agents"

    id: Mapped[UUID] = primary_key
    slug: Mapped[str] = unique, indexed            # "research-bot"
    display_name: Mapped[str]
    description: Mapped[str | None]

    # Modelo base: exclusivamente uno de los dos
    base_model_id: Mapped[str | None] = FK(model_configs.model_id, ondelete="CASCADE")
    profile_id: Mapped[str | None] = FK(model_profiles.profile_id, ondelete="CASCADE")

    system_prompt: Mapped[str]                     # TEXT

    # Tool-loop params
    tool_choice_default: Mapped[str]               # "auto" | "required" | "none"
    max_tool_hops: Mapped[int] = default(8)
    tool_timeout_seconds: Mapped[int] = default(60)
    require_approval: Mapped[str]                  # "never" | "always"

    # Sampling defaults (opcional, override del request)
    request_defaults: Mapped[dict | None]          # JSONB: {temperature, top_p, max_tokens, ...}

    # ACL
    group_id: Mapped[UUID | None] = FK(groups.id, ondelete="SET NULL")

    # Audit
    created_by: Mapped[UUID] = FK(users.id)
    created_at: Mapped[datetime]
    updated_at: Mapped[datetime]
```

### Tabla `agent_mcp_servers` (many-to-many con filtro fino)

```python
class AgentMCPServer(Base):
    __tablename__ = "agent_mcp_servers"

    agent_id: Mapped[UUID] = FK(agents.id, ondelete="CASCADE"), primary_key
    mcp_server_id: Mapped[UUID] = FK(mcp_servers.id, ondelete="CASCADE"), primary_key
    # Allowlist de tools de ESE servidor para ESE agente (más restrictivo que server.allowed_tools)
    allowed_tools: Mapped[list[str] | None]        # None = hereda del server
```

### Tabla `tool_call_stats` (migración 0017)

```python
class ToolCallStat(Base):
    __tablename__ = "tool_call_stats"

    id: Mapped[UUID] = primary_key
    request_stat_id: Mapped[UUID | None] = FK(request_stats.id, ondelete="CASCADE"), indexed
    agent_id: Mapped[UUID | None] = FK(agents.id, ondelete="SET NULL")
    mcp_server_alias: Mapped[str]
    tool_name: Mapped[str]
    tool_args_redacted: Mapped[dict]               # JSONB, con redact_fields aplicados
    result_summary: Mapped[str | None]             # truncado a 4KB
    duration_ms: Mapped[int]
    status: Mapped[str]                            # "ok" | "timeout" | "schema_error" | "mcp_error"
    error: Mapped[str | None]
    hop_index: Mapped[int]                         # 0..max_tool_hops
    created_at: Mapped[datetime]
```

### Extensión de `request_stats`

Campos opcionales (migración 0017):
- `agent_id: UUID | None` + índice
- `parent_request_id: UUID | None` + índice — apunta al request_stat raíz del
  agente cuando esta fila es un hop intermedio. El root tiene `parent_request_id = NULL`
  y `agent_id` rellenado. Los hops hijos heredan `agent_id` y rellenan `parent_request_id`.

No se rompe compatibilidad (ambos NULL para requests no-agente).

---

## Contratos (añadir a `docs/CONTRACTS.md`)

### 1. MCPClientInterface

```python
# backend/ocabra/agents/mcp_client.py

from abc import ABC, abstractmethod

@dataclass
class MCPTool:
    name: str
    description: str
    input_schema: dict  # JSON Schema

@dataclass
class MCPToolResult:
    content: list[dict]  # blocks: text, image, resource
    is_error: bool
    raw: dict            # original MCP response

class MCPClientInterface(ABC):

    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...

    @abstractmethod
    async def list_tools(self) -> list[MCPTool]: ...

    @abstractmethod
    async def call_tool(
        self,
        name: str,
        arguments: dict,
        *,
        timeout_seconds: float,
        extra_headers: dict[str, str] | None = None,
    ) -> MCPToolResult: ...

    @abstractmethod
    async def health_check(self) -> bool: ...
```

Implementaciones: `HttpMCPClient`, `SseMCPClient`, `StdioMCPClient`.

### 2. AgentExecutor

```python
@dataclass
class AgentExecutorResult:
    openai_response: dict            # payload final /v1/chat/completions
    hops_used: int
    tool_calls: list[ToolCallRecord] # para stats

class AgentExecutor:
    async def run(
        self,
        agent: AgentSpec,
        messages: list[dict],
        request_options: dict,       # temperature, etc.
        user_ctx: UserContext,
        per_request_headers: dict[str, str] | None = None,
    ) -> AgentExecutorResult: ...

    async def run_stream(
        self, ...
    ) -> AsyncIterator[dict]:        # SSE chunks OpenAI-compat
        """Streaming: emite deltas del LLM y eventos ocabra.tool_call custom."""
```

### 3. Formato OpenAI ↔ MCP

**MCP tool → OpenAI tool schema**:
```python
{
    "type": "function",
    "function": {
        "name": f"{alias}_{mcp_tool.name}",           # namespaced
        "description": mcp_tool.description,
        "parameters": mcp_tool.input_schema,           # JSON Schema pasa tal cual
    }
}
```

**OpenAI tool_call → MCP call**:
```python
name = openai_tc["function"]["name"]          # "github_create_issue"
alias, tool_name = name.split("_", 1)         # "github", "create_issue"
arguments = json.loads(openai_tc["function"]["arguments"])
# → mcp_registry.call_tool(alias, tool_name, arguments, ...)
```

**MCP result → OpenAI tool message**:
```python
{
    "role": "tool",
    "tool_call_id": openai_tc["id"],
    "content": _flatten_mcp_content(mcp_result.content),  # concat text blocks
}
```

Bloques no-texto (image/resource) → se codifican según capacidades del modelo base
(vision-capable → image_url content; resto → `[image omitted]` textual).

---

## API REST nueva

Todos los endpoints bajo `/ocabra/*`, con auth interna. Rol requerido documentado por endpoint.

### MCP servers

| Método | Ruta | Rol mínimo | Descripción |
|--------|------|-----------|-------------|
| `GET`  | `/ocabra/mcp-servers` | `model_manager` | Lista todos. Redacta `auth_value`, `env`. |
| `POST` | `/ocabra/mcp-servers` | `model_manager` (http/sse), `system_admin` (stdio) | Crea. |
| `GET`  | `/ocabra/mcp-servers/{id}` | `model_manager` | Detalle (sin secretos). |
| `PATCH`| `/ocabra/mcp-servers/{id}` | `model_manager` | Actualiza. |
| `DELETE`| `/ocabra/mcp-servers/{id}` | `model_manager` | Borra. Falla si algún agente lo usa y `force=false`. |
| `POST` | `/ocabra/mcp-servers/{id}/refresh` | `model_manager` | Fuerza `tools/list`, actualiza cache. |
| `GET`  | `/ocabra/mcp-servers/{id}/tools` | `model_manager` | Tools cacheadas. |
| `POST` | `/ocabra/mcp-servers/{id}/test` | `model_manager` | Dry-run de conexión, devuelve `{healthy, tools_count, error}`. |

### Agents

| Método | Ruta | Rol mínimo | Descripción |
|--------|------|-----------|-------------|
| `GET`  | `/ocabra/agents` | `user` | Lista agentes accesibles por el grupo del caller. |
| `POST` | `/ocabra/agents` | `model_manager` | Crea. |
| `GET`  | `/ocabra/agents/{slug}` | `user` (si accesible) | Detalle. |
| `PATCH`| `/ocabra/agents/{slug}` | `model_manager` | Actualiza. |
| `DELETE`| `/ocabra/agents/{slug}` | `model_manager` | Borra. |
| `POST` | `/ocabra/agents/{slug}/test` | `model_manager` | Dry-run: valida base_model cargable, tools resolubles, servidores healthy. |

### Inventario de modelos

`GET /v1/models` y `GET /ocabra/models` añaden los agentes como entradas con
`id = f"agent/{slug}"` y `owned_by = "ocabra-agent"`. Filtrado por grupo igual que modelos.

### Invocación

`POST /v1/chat/completions` acepta:

- `model: "agent/<slug>"` — dispara el tool-loop.
- `tools`: si el caller pasa tools adicionales, se **concatenan** con namespace
  `caller_*` (el agente usa `{alias}_*` para MCP). Colisión de nombres → 400.
- `tool_choice`: override puntual del `tool_choice_default` del agente.
- Header `x-ocabra-allowed-tools: tool1,tool2`: allowlist por request (intersección
  con la del agente y la del server).
- Header `x-ocabra-require-approval: never|always`: override por request. Sólo se
  acepta si el agente no es más restrictivo (p.ej. agente `never` permite override
  a `always`; agente `always` no permite override a `never`).
- Headers `x-mcp-{alias}-{header}`: passthrough a ese servidor MCP concreto
  (no puede sobrescribir headers estáticos configurados en el server).

`POST /api/chat` (Ollama-compat) acepta los mismos parámetros equivalentes.
`stream: true` soportado — ver Fase 3.

---

## Fases

### Fase 1 — Schema + registry (sin tool-loop)

Objetivo: poder listar/CRUD MCP servers y agentes, refrescar tools cacheadas, sin
que aún afecte a `/v1/chat/completions`.

**Entregables**:
- Migración `0017_agents_mcp.py` (4 tablas + índices + campo `agent_id` en `request_stats`)
- `backend/ocabra/db/agents.py`, `backend/ocabra/db/mcp.py` (modelos SQLAlchemy)
- `backend/ocabra/schemas/agents.py`, `.../mcp.py` (Pydantic)
- `backend/ocabra/agents/mcp_client.py` (interface + impl http/sse/stdio usando `mcp` SDK)
- `backend/ocabra/agents/mcp_registry.py` (singleton, pool de clients, cache de tools con TTL `mcp_tools_cache_ttl_seconds=300`, invalidación explícita)
- `backend/ocabra/api/internal/mcp_servers.py` (CRUD + refresh + test)
- `backend/ocabra/api/internal/agents.py` (CRUD + test)
- `ocabra.config.settings` nuevas:
  - `mcp_tools_cache_ttl_seconds: int = 300`
  - `mcp_default_tool_timeout_seconds: int = 60`
  - `mcp_max_concurrent_tool_calls: int = 8`
  - `mcp_result_max_bytes: int = 262144` (256 KB)
  - `mcp_stdio_allowed: bool = True`
- Cifrado de `auth_value` y `env` con Fernet (reutilizar helper de federation).
- Tests: CRUD endpoints (permisos por rol), mock MCP server para `list_tools`/`call_tool`, cifrado, validación de Pydantic.

**Criterio de aceptación**: `POST /ocabra/mcp-servers` → `POST /ocabra/mcp-servers/{id}/refresh` → `GET /ocabra/mcp-servers/{id}/tools` devuelve tools. Usuario `user` recibe 403 en CRUD.

### Fase 2 — AgentExecutor non-streaming (OpenAI + Ollama)

Objetivo: `model="agent/<slug>"` funciona end-to-end sin streaming en ambas APIs.

**Entregables**:
- `backend/ocabra/agents/executor.py` con `AgentExecutor.run()`
- `backend/ocabra/agents/resolver.py` — `resolve_agent(model_id) -> AgentSpec | None`
- Parche en `backend/ocabra/api/openai/chat.py`: tras `resolve_profile`, probar `resolve_agent` y delegar a executor
- Parche en `backend/ocabra/api/ollama/chat.py`: mismo flujo, adapter Ollama↔OpenAI antes y después del executor (el executor siempre habla en formato OpenAI internamente)
- Conversión OpenAI↔MCP en `backend/ocabra/agents/translation.py`
- Validación de args con `jsonschema` contra `input_schema` antes de invocar
- Ejecución paralela de tool_calls dentro de un mismo turno (`asyncio.gather` con semáforo `mcp_max_concurrent_tool_calls`)
- Enforcement de `max_tool_hops`, `tool_timeout_seconds`
- Soporte completo de `require_approval`:
  - `"never"`: bucle hasta no haber tool_calls o `max_tool_hops`
  - `"always"`: ejecuta hasta el primer `tool_calls`, devuelve con `finish_reason: "tool_calls"` sin ejecutarlos; el cliente reenvía un turno con los resultados y el executor continúa (reconoce que ya hay resultados en los mensajes y procede)
  - Override por header `x-ocabra-require-approval` respetando restrictividad del agente
- Propagación de `parent_request_id` al WorkerPool en cada hop (el worker lo persiste en `request_stats`); root request guarda `agent_id`
- Agregación de coste: helper que suma tokens de toda la cadena cuando se consulta una request raíz
- Redacción de args en logs según `redact_fields` (v1: lista fija `["authorization", "password", "token", "api_key", "secret"]`)
- Persistencia en `tool_call_stats` + link a `request_stats` (root)
- Concatenación de tools del caller con namespace `caller_*`; rechazo con 400 si colisiona
- Tests:
  - tool-loop (0 hops, 1 hop, N hops, timeout, schema_error, parallel tools)
  - `require_approval=always`: primer turno devuelve tool_calls; reenvío con resultados continúa
  - Ollama adapter: request Ollama con `model=agent/...` devuelve response Ollama válida
  - Propagación de `parent_request_id` (suma de tokens por cadena correcta)
  - Colisión de nombres entre `caller_*` y tools del agente → 400

**Criterio de aceptación**: 
1. Agente con MCP filesystem local ejecuta "lista el dir X" vía OpenAI y vía Ollama y devuelve una response válida en cada formato.
2. Stats acumula tokens de todos los hops bajo el `agent_id`.
3. `require_approval=always` funciona con handshake de dos turnos.

### Fase 3 — Streaming

Objetivo: `stream: true` con eventos OpenAI-compat para deltas del LLM y tool_calls.

**Entregables**:
- `AgentExecutor.run_stream()` — emite SSE chunks del LLM, entre hops emite un
  chunk con `choices[].delta.tool_calls[]` (formato OpenAI) y tras ejecutar el
  tool emite un `role: "tool"` chunk *fuera* del standard (evento custom
  `ocabra.tool_result`) que clientes que no lo entienden ignoran. La response
  final sigue siendo OpenAI-compat.
- Integración con Langfuse: un solo trace por request con spans por hop.

**Criterio de aceptación**: cliente OpenAI Python SDK consume streaming sin error;
clientes con UI rica (nuestro frontend) muestran los tool_calls en vivo.

### Fase 4 — UI

**Entregables**:
- Página `frontend/src/pages/Agents.tsx` (rol `model_manager`+):
  - Lista de agentes con CRUD
  - Form: slug, base_model (selector reusa `ModelPicker`), system_prompt (textarea
    con contador), selección de MCP servers con checklist de tools, sliders para
    `max_tool_hops` / `tool_timeout_seconds`
  - Botón "Test" que dispara `/ocabra/agents/{slug}/test`
- Página `frontend/src/pages/MCPServers.tsx` (rol `model_manager`+):
  - Lista con estado de salud (badge) y count de tools
  - Form: alias, transport, url/command, auth (campo password), allowed_tools
  - Botón "Refresh tools" y "Test connection"
- Entrada en sidebar (visible sólo para rol `model_manager`+)
- Cliente tipado en `frontend/src/api/agents.ts` y `.../mcp.ts`
- Zustand store `frontend/src/stores/agentsStore.ts` con WebSocket listener para
  eventos `agent_updated` / `mcp_server_health_changed`

**Criterio de aceptación**: admin puede crear un agente desde la UI y usarlo
inmediatamente desde el Playground seleccionándolo en el selector de modelo.

### Fase 5 — Playground + Stats

**Entregables**:
- Playground: dropdown de modelo muestra agentes como sección aparte con icono ✨.
  Al seleccionarlo, oculta controles que el agente fuerza (system prompt no editable,
  tools controladas por el agente).
- Stats: tab nueva "Agents" con:
  - Top agents por request count
  - Top tools por invocaciones, latencia p50/p95, error rate
  - Drill-down por agente → lista de tool_calls recientes
- Card en Dashboard: "Active agents (last hour)"

### Fase 6 — MCP passthrough (OPCIONAL, decidir tras Fase 5)

Objetivo: exponer los MCP servers de oCabra a clientes MCP puros (Claude Desktop,
Cursor) con la auth de oCabra.

`POST /mcp/{alias}` que:
- Autentica con API key oCabra
- Verifica que el grupo del caller tenga acceso al server
- Forwardea al MCP server real añadiendo/reescribiendo auth
- Registra en `tool_call_stats` con `agent_id=null`, `api_key` rellenada

No bloquea ningún caso de uso actual. Pospuesto hasta ver demanda.

---

## Seguridad — checklist obligatoria

Cada PR de este bloque debe confirmar:

- [ ] `auth_value` y `env` de mcp_servers cifrados en BD (Fernet), nunca loggeados
- [ ] Transport `stdio`: rol `system_admin` requerido en create/update
- [ ] Validación JSON Schema de args antes de llamar al tool; args inválidos → 400 al LLM como `tool` message con error, no excepción al cliente
- [ ] Timeout por tool_call (`tool_timeout_seconds`); expiración → `tool` message con error
- [ ] `max_tool_hops` enforced; exceder → response final con `finish_reason: "tool_hop_limit"`
- [ ] Redacción de campos sensibles en `tool_args_redacted` (lista configurable en settings)
- [ ] `result_summary` truncado a `mcp_result_max_bytes`
- [ ] Rate limit a nivel de API key sobre total de tool_calls por minuto (reutilizar infra existente si la hay, si no `mcp_tool_calls_per_minute: int = 60`)
- [ ] `stdio` subprocess corre con `cwd` bajo `/data/mcp/<alias>/`, `env` sin heredar variables del proceso padre, PATH explícito
- [ ] `stdio` subprocess sometido a `BackendProcessManager` (health check, auto-restart, limit de restarts)
- [ ] Ningún endpoint admite CORS fuera de localhost (igual que resto de `/ocabra/*`)
- [ ] Intersección de allowlists: `server.allowed_tools` ∩ `agent_mcp_server.allowed_tools` ∩ `x-ocabra-allowed-tools` header
- [ ] Headers `x-mcp-{alias}-*` del caller **nunca** sobrescriben headers estáticos configurados en el server (cliente no puede saltarse auth)
- [ ] Header `x-ocabra-require-approval` no permite *rebajar* restrictividad del agente (agente `always` no puede pasar a `never`)
- [ ] Colisión de nombres entre tools del caller (`caller_*`) y del agente (`{alias}_*`) → 400, no se silencia
- [ ] Un borrado de MCP server invalida su cache y fuerza reload de agentes que lo usaban

---

## Dependencias nuevas

```toml
# backend/pyproject.toml
mcp = ">=0.9.0"          # SDK oficial Anthropic
jsonschema = ">=4.21.0"  # validación de args (si no está ya)
```

No se añade `litellm` como dep (ya existe sólo para `litellm_sync`, se mantiene).

---

## Reparto entre agentes paralelos (sugerido)

| Stream | Fases | Ficheros principales | Bloqueado por |
|--------|-------|---------------------|---------------|
| **A — DB + Registry** | 1 | `alembic/0017_*`, `db/agents.py`, `db/mcp.py`, `agents/mcp_client.py`, `agents/mcp_registry.py`, `api/internal/mcp_servers.py`, `api/internal/agents.py` | — |
| **B — Executor** | 2, 3 | `agents/executor.py`, `agents/translation.py`, `agents/resolver.py`, parche en `api/openai/chat.py` | A (schema + registry) |
| **C — Frontend** | 4, 5 | `pages/Agents.tsx`, `pages/MCPServers.tsx`, `api/agents.ts`, `api/mcp.ts`, stores | A (endpoints REST) — puede empezar con mock fallback como hizo Fase 5 de bloque 15 |
| **D — Observabilidad** | 2.5 | Extensión de Langfuse tracer, panel Stats tab Agents | B (executor emitiendo eventos) |

Streams A y C pueden arrancar en paralelo desde el día 1. B arranca cuando A mergea
schema + registry. D arranca cuando B emite su primer tool_call_stat.

---

## Regla de documentación (obligatoria para todos los streams)

Al cerrar cualquier fase, PR o sesión de trabajo sobre este plan, **cada stream debe actualizar este documento** con tres secciones:

1. **Avances**: entregables concretos (ficheros nuevos/modificados, endpoints expuestos, migraciones aplicadas, tests añadidos, comandos validados).
2. **Deudas técnicas**: lo que quedó con mock/stub, lo que requiere refactor posterior, los atajos tomados conscientemente, los TODOs dejados en el código.
3. **Cuestiones pendientes**: decisiones que requieren input del usuario, ambigüedades encontradas, bloqueos para otros streams.

Si no se actualiza esta sección, el trabajo no se considera cerrado aunque el código esté mergeado. Los agentes deben incluirlo en su reporte final y editar este fichero antes de terminar.

---

## Registro de decisiones y deudas

*(Añadir entradas a medida que las fases progresen, como en `modular-backends-plan.md`.)*

### 2026-04-25 — Validación de los entregables de Streams A y C

**Backend (Stream A, ya en `main`)**:
- `pytest tests/agents/` (vía contenedor `ocabra-api-1`): **46/49 tests pasan, 3 fallan**:
  - `test_create_http_ok_for_model_manager` y `test_create_stdio_ok_for_admin`: `MCPServerOut.health_status` rechaza `None` con `literal_error`. La columna tiene `server_default="unknown"` pero el `.scalar_one()` post-INSERT devuelve `None` en el modelo Pydantic. Fix: o bien `health_status: Literal[...] | None = "unknown"` en schema, o `default="unknown"` en `MCPServer.health_status` ORM (no sólo `server_default`), o `INSERT ... RETURNING` que rellene el valor.
  - `test_decrypt_rejects_tampered`: token Fernet manipulado no levanta `ValueError` (silenciado por try/except en `_decrypt_optional` o similar). Revisar: si la API se documenta como "raise on tamper", quitar el except; si se documenta como "return None on tamper", actualizar el test.
- `ruff check`: 33 errores (28× `B008` *function-call-in-default-argument* — patrón `Depends()`, mismo problema que `api/internal/models.py` preexistente con 24× B008 — **no es regresión de Stream A**; 4× `F401`; 1× `I001`).
- `ruff format --check`: 10 ficheros requieren reformat.

**Frontend (Stream C, worktree `agent-a4edf4f5`)**:
- `npm run lint`: ✅ pasa
- `npm run build` (`tsc -b && vite build`): ✅ pasa, 2729 módulos transformados
- `npx vitest run`: 21/25 tests pasan. Los **2 tests nuevos** (`Agents.test.tsx`, `MCPServers.test.tsx`) **pasan**. Los 4 fallos son preexistentes en `main` (`Dashboard`, `ExploreFlow`, `GpuCard`, `Settings`) — **no introducidos por Stream C**.
- Symlink `node_modules` → `../../../../frontend/node_modules` necesario para validar (worktree no instaló deps).

**Deudas que cerrar antes de Stream B**:
- [ ] Fix `health_status=None` en `MCPServerOut` o en el insert (bug bloqueante para crear cualquier MCP server por API)
- [ ] Fix Fernet tampering test (decisión: ¿levanta o devuelve None?)
- [ ] (Opcional) `ruff format` sobre los 10 ficheros nuevos
- [ ] (Opcional) Limpiar 4× F401 + 1× I001 en tests/agents/
- [ ] (Opcional) Refactor B008 a `Annotated[X, Depends(...)]` — afecta también a routers preexistentes; tarea separada si se acomete
- [ ] Mergear Stream C (`worktree-agent-a4edf4f5` → `main`); ya validado, no necesita cambios

---

### 2026-04-24 — Stream A (DB + Registry, Fase 1) entregado

**Rama**: worktree `agent-a545a2ab` (4 commits sobre `86b70b7`, worktree en `/docker/ocabra/.claude/worktrees/agent-a545a2ab`).

**Avances**:
- Migración `0017_agents_mcp.py` con 4 tablas (`mcp_servers`, `agents`, `agent_mcp_servers`, `tool_call_stats`) + campos `agent_id` y `parent_request_id` en `request_stats` + CHECK constraint `ck_agents_exactly_one_base`.
- ORM: `db/mcp.py`, `db/agents.py`, extensión de `db/stats.py`.
- Schemas Pydantic v2: `schemas/mcp.py`, `schemas/agents.py`.
- Cliente MCP (`agents/mcp_client.py`) con interface + 3 impls (http/sse/stdio) importando el SDK `mcp` de forma perezosa.
- Registry MCP (`agents/mcp_registry.py`) con pool de clients, cache TTL, invalidación, health checks, carga desde BD al arranque.
- Routers CRUD: `api/internal/mcp_servers.py` y `api/internal/agents.py` con ACL por rol (stdio exige `system_admin` en create/update/**delete**).
- Inventario: agentes aparecen en `/v1/models` y `/ocabra/models` como `agent/<slug>`.
- Lifespan de `MCPRegistry` wireado en `main.py`.
- Deps añadidas: `mcp>=0.9.0`, `jsonschema>=4.21.0`.
- Settings: 5 nuevas `mcp_*` en `config.py`.
- 44 test cases en `tests/agents/` (CRUD, client mock, registry, inventory, conftest con `FakeSessionFactory`).
- Cifrado Fernet para `auth_value` y `env` (reutiliza helper de federation).

**Deudas técnicas**:
- **Validación no ejecutada**: `ruff`, `pytest` y `alembic upgrade head` bloqueados por el sandbox del harness. **Bloqueante para merge**: ejecutar desde el worktree `cd backend && ruff check . && ruff format --check . && pytest tests/agents/` y `alembic upgrade head` contra BD de test.
- Header merge (`_HeaderMerger`) ya implementado en el cliente — adelanta parte de Fase 2 (Stream B). No refactorizar, sólo consumir.
- `env_encrypted` no estaba en el plan original (sí en la checklist implícita); plan actualizado implícitamente, pero conviene reflejarlo en la sección "Modelo de datos" si se rehace el doc.

**Cuestiones pendientes / decisiones no consultadas**:
- `env` cifrado con Fernet (no explícito en el plan; confirmar criterio — probablemente OK, alinea con la checklist de seguridad).
- `group_id = NULL` en `agents` ⇒ agente público (visible para todos los usuarios). Mismo patrón que modelos sin restricción. **Confirmar con usuario** si se prefiere que NULL signifique "sólo admins".
- DELETE de `mcp_servers` con `transport=stdio` requiere `system_admin` (plan sólo lo pedía en create/update). Aplicado por simetría de privilegios.
- CHECK constraint SQL duplica la validación Pydantic de "exactamente uno de base_model_id o profile_id" — consciente, defensa en profundidad.
- SDK `mcp` importado perezosamente: tests y dev local funcionan sin él instalado. En prod la dep existe en `pyproject.toml`.

---

### 2026-04-24 — Stream C (Frontend, Fases 4+5) entregado

**Rama**: `worktree-agent-a4edf4f5` (6 commits sobre `main`, worktree en `/docker/ocabra/.claude/worktrees/agent-a4edf4f5`).

**Avances**:
- Páginas `/agents` y `/mcp-servers` (rol `model_manager`+) con CRUD completo, selector de MCP servers con allowlist de tools por agente, botones Test/Refresh.
- Cliente tipado `api/agents.ts` y `api/mcp.ts` + stores Zustand con mock fallback al detectar 404/501.
- Playground integra agentes en el dropdown (sección "✨ Agents") y bloquea el campo system_prompt cuando se elige uno.
- Stats: tab "Agents" con top agents, top tools, drill-down a tool_calls recientes (mock).
- Dashboard: card "Active agents (last hour)" para `model_manager`+.
- WebSocket hook extendido con `agent_updated` y `mcp_server_health_changed`.
- Transport `stdio` visible pero deshabilitado para `model_manager` sin admin.
- Smoke tests: `__tests__/Agents.test.tsx` y `MCPServers.test.tsx` (render + banner de mock).
- Sidebar con entradas nuevas y rutas `/agents`, `/mcp-servers` protegidas por rol en `App.tsx`.

**Deudas técnicas**:
- **`npm run lint` y `npm run build` no ejecutados**: el worktree no tenía `node_modules` y el sandbox bloqueó `npm ci` + crear symlink. Auditoría manual realizada pero no sustituye a `tsc -b`. **Bloquea merge hasta validar**: ejecutar desde el worktree `cd frontend && npm install && npm run lint && npm run build && npm test`.
- Mocks pendientes de Stream A (marcados con `// TODO: remove mock once Stream A merges`):
  - CRUD `/ocabra/agents` y `/ocabra/mcp-servers`
  - Eventos WS `agent_updated` y `mcp_server_health_changed`
- Mocks pendientes de Stream A/B (ver "Deudas abiertas"):
  - `/ocabra/stats/by-agent` y `/ocabra/stats/tool-calls`
- Edge-case UX aceptado para v1: editar un agente cuyo `profile_id` no tiene `base_model_id` asociado deja el picker de profile vacío hasta que el usuario seleccione un modelo base primero.
- Limitación UX: `allowed_tools = null` (hereda) vs `[]` (ninguno explícito) no son distinguibles en el modal — desmarcar todo vuelve a `null`. Documentar en la Fase 4 del README de agentes si se añade.

**Cuestiones pendientes**:
- Validación humana de la UI en dev server (`npm run dev`) antes de mergear. La auditoría estática cubre tipos e imports pero no regresiones visuales ni WebSocket-in-UI.
- Decidir si `agentsApi`/`mcpApi` se quedan como módulos paralelos al gran objeto `api` en `client.ts` (decisión tomada para no inflarlo más) o se consolidan en `api.agents`/`api.mcp` en Fase 5+.
- Los endpoints de stats (`/ocabra/stats/by-agent`, `/ocabra/stats/tool-calls`) no están especificados en detalle — Stream A/B debe diseñar su forma antes de que C pueda quitar el mock. Sugerencia: `{ agents: [{agent_id, slug, requests, tokens_total, avg_hops}], tool_calls: [...] }` reusando el agregador de `cost_calculator`.

### Decisiones confirmadas

- **2026-04-24**: usar `mcp` SDK oficial en lugar de `litellm.experimental_mcp_client`. Razón: evitar traer toda la dep de litellm; traducción OpenAI↔MCP son ~30 líneas propias.
- **2026-04-24**: stateless, sin tabla de conversaciones. Razón: CLAUDE.md lo prohíbe y simplifica el blast radius.
- **2026-04-24**: sin sub-agentes en v1. Razón: abre recursión, coste y complejidad; no hay demanda concreta aún.
- **2026-04-24**: CRUD de agentes y mcp_servers sólo para `model_manager`+. Razón: un MCP server con auth propia es una credencial compartida; tratarlo como config de modelo.
- **2026-04-24**: invocación por `model="agent/<slug>"`. Razón: reusar el contrato OpenAI existente en vez de fragmentar la API.

### Deudas abiertas

- [ ] **Stats endpoints para agents**: `GET /ocabra/stats/by-agent` (top agents + top tools con p50/p95/error rate) y `GET /ocabra/stats/tool-calls?agent_id=...` (últimos tool calls para drill-down). **Frontend (Stream C) ya los consume contra mock fallback**; Stream A/B debe exponerlos para que la tab "Agents" de Stats y la card "Active agents (last hour)" del Dashboard dejen de mostrar datos de ejemplo.
- [ ] OAuth2 discovery en MCP servers (LiteLLM lo soporta; v1 sólo api_key/bearer/basic).
- [ ] `/v1/responses` compat (v1 sólo `/v1/chat/completions` y Ollama `/api/chat`).
- [ ] Sub-agentes (un tool de tipo `invoke_agent` que llame a otro `AgentExecutor`).
- [ ] `require_approval: per-tool` (lista de tools que requieren approval; v1 sólo `never`/`always`).
- [ ] UI: drag-and-drop para ordenar el orden de presentación de tools al LLM.
- [ ] Passthrough `/mcp/{alias}` (Fase 6 opcional).
- [ ] Rate limit distribuido (Redis) — v1 puede ser in-process si no hay cluster.

---

## Decisiones resueltas con el usuario (2026-04-24)

1. **`require_approval` configurable** por agente: soportamos `"never"` (auto-exec, default)
   y `"always"` (devuelve la response con `tool_calls` pendientes y el cliente debe
   reenviar los resultados en un turno nuevo). Impacto:
   - Campo `require_approval` en `agents` ya previsto.
   - En `"always"`: executor emite el chunk del LLM con tool_calls y termina el turno
     con `finish_reason: "tool_calls"` (estándar OpenAI).
   - En `"never"`: ejecuta el bucle hasta `max_tool_hops` o ausencia de tool_calls.
   - Override por request: campo `tool_choice` ya lo cubre parcialmente; añadimos
     también soporte al header `x-ocabra-require-approval: never|always` para forzar
     por request (sólo si el agente lo permite — ver Fase 2).
   - `"per-tool"` (lista de tools que requieren aprobación) queda como deuda v2.

2. **Tools extra del caller**: se concatenan con las del agente. Namespace `caller_*`
   para evitar colisión con las MCP (que usan `{alias}_*`). Si el caller intenta
   registrar un tool cuyo nombre ya existe en el agente, se rechaza con 400.

3. **Ollama también**: agentes disponibles en `/api/chat` de Ollama con
   `model="agent/<slug>"`. Mismo `AgentExecutor`, adapter en
   `backend/ocabra/api/ollama/chat.py` que traduce formato Ollama↔OpenAI antes
   y después del executor. Scope añadido a Fase 2.

4. **Coste y tokens**: *todos* los tokens procesados por el LLM cuentan. Cada hop
   es una request al `WorkerPool`, cada una registra su `request_stat` hijo con
   `parent_request_id` apuntando al request_stat raíz del agente. El `cost_calculator`
   agrega los hijos. Esto ya funciona con el modelo actual de `request_stats`;
   sólo hay que añadir:
   - Campo `parent_request_id: UUID | None` en `request_stats` (migración 0017).
   - Campo `agent_id` en `request_stats` (ya previsto).
   - `AgentExecutor` propaga `parent_request_id` en cada call al worker.
   - UI Stats suma tokens de toda la cadena cuando se filtra por agente o por
     request raíz.
