# oCabra — Plan de Federación (Modo Federado)

Última actualización: 2026-04-12

---

## Concepto

Permitir que múltiples instancias de oCabra (cada una autónoma, con su propio Postgres,
Redis, backends y GPUs) se conecten entre sí para formar una **federación peer-to-peer**.

Cada nodo:
- Funciona completamente por sí mismo (sin depender de otros nodos).
- Conoce los modelos cargados en los nodos remotos conectados.
- Puede **proxificar peticiones de inferencia** a modelos remotos de forma transparente.
- Aplica **load balancing** cuando un mismo modelo está disponible en múltiples nodos.
- Mantiene su propia auth, API keys y base de datos independientes.

### Ejemplo concreto

| Nodo | GPU | Modelos típicos |
|------|-----|-----------------|
| `nodo-A` (principal) | RTX 3090 + RTX 3060 | Qwen3-32B, Whisper-large, SDXL |
| `nodo-B` (secundario) | GTX 1660 (6 GB) | Qwen2.5-3B, Whisper-small, Kokoro TTS |

Un usuario conectado a `nodo-B` pide `Qwen3-32B` → `nodo-B` no lo tiene localmente →
lo proxifica a `nodo-A` → el usuario recibe la respuesta como si fuera local.

Un usuario conectado a `nodo-A` pide `Qwen2.5-3B` que está cargado en ambos nodos →
`nodo-A` aplica load balancing y elige el nodo con menos carga.

---

## Decisiones de diseño

| Decisión | Elección | Razón |
|----------|----------|-------|
| Topología | Peer-to-peer (sin coordinador central) | Cada nodo debe funcionar solo |
| Base de datos | Independiente por nodo | Autonomía, tolera desconexión |
| Descubrimiento | Configuración explícita de peers | Simple, predecible, seguro |
| Protocolo inter-nodo | HTTPS REST + SSE (reutilizar API existente) | Sin dependencias nuevas |
| Auth inter-nodo | API key dedicada por peer | Cada nodo controla qué acceso da |
| Servicios interactivos | NO se federan (v1) | Complejidad alta, poco beneficio |
| Descarga remota de modelos | Controlada por nivel de acceso del peer | Configurable por nodo |

---

## Arquitectura

```
┌─────────────────────────────────┐     HTTPS + API key     ┌─────────────────────────────────┐
│          nodo-A                 │◄────────────────────────►│          nodo-B                 │
│                                 │                          │                                 │
│  ┌──────────┐  ┌─────────────┐  │   Heartbeat (30s)        │  ┌──────────┐  ┌─────────────┐  │
│  │ GPU Mgr  │  │ Model Mgr   │  │◄────────────────────────►│  │ GPU Mgr  │  │ Model Mgr   │  │
│  └──────────┘  └─────────────┘  │   + inventario modelos   │  └──────────┘  └─────────────┘  │
│                                 │                          │                                 │
│  ┌──────────────────────────┐   │                          │  ┌──────────────────────────┐   │
│  │    FederationManager     │   │                          │  │    FederationManager     │   │
│  │  - peers[]               │   │                          │  │  - peers[]               │   │
│  │  - remote_models cache   │   │                          │  │  - remote_models cache   │   │
│  │  - proxy_request()       │   │                          │  │  - proxy_request()       │   │
│  │  - load_balancer()       │   │                          │  │  - load_balancer()       │   │
│  └──────────────────────────┘   │                          │  └──────────────────────────┘   │
│                                 │                          │                                 │
│  ┌──────┐ ┌──────┐ ┌────────┐  │                          │  ┌──────┐ ┌────────┐           │
│  │ PG   │ │Redis │ │Workers │  │                          │  │ PG   │ │Workers │           │
│  └──────┘ └──────┘ └────────┘  │                          │  └──────┘ └────────┘           │
└─────────────────────────────────┘                          └─────────────────────────────────┘
```

---

## Fases de implementación

### Fase 1 — Registro de peers y heartbeat

**Objetivo:** Cada nodo conoce la existencia y estado de sus peers.

#### 1.1 — Configuración de peers

Nuevo setting en `config.py`:

```python
# Lista de peers conocidos
federation_peers: list[dict] = []
# Ejemplo:
# [
#   {
#     "name": "nodo-B",
#     "url": "https://nodo-b.local:8000",
#     "api_key": "sk-ocabra-...",
#     "access_level": "inference"  # "inference" | "full"
#   }
# ]

# Identificador único de este nodo (auto-generado si vacío)
federation_node_id: str = ""
federation_node_name: str = ""

# Intervalo de heartbeat en segundos
federation_heartbeat_interval: int = 30

# Habilitar modo federado
federation_enabled: bool = False
```

`access_level` controla qué puede hacer el peer:
- `inference`: solo proxificar peticiones de inferencia a modelos cargados.
- `full`: además puede disparar descargas de modelos, ver GPUs, etc.

#### 1.2 — Endpoint de heartbeat

Nuevo endpoint en la API interna:

```
GET /ocabra/federation/heartbeat
Authorization: Bearer sk-ocabra-...

Response 200:
{
  "node_id": "uuid",
  "node_name": "nodo-A",
  "version": "0.5.0",
  "uptime_seconds": 3600,
  "gpus": [
    {"index": 0, "name": "RTX 3060", "total_vram_mb": 12288, "free_vram_mb": 4096}
  ],
  "models": [
    {
      "model_id": "vllm/Qwen3-32B",
      "status": "LOADED",
      "capabilities": {"chat": true, "streaming": true, ...},
      "profiles": ["qwen3-32b", "qwen3-32b-creative"]
    }
  ],
  "load": {
    "active_requests": 3,
    "gpu_utilization_avg_pct": 45.0
  }
}
```

#### 1.3 — FederationManager (core)

Nuevo módulo `backend/ocabra/core/federation.py`:

```python
class PeerState:
    peer_id: str
    name: str
    url: str
    api_key: str
    access_level: str  # "inference" | "full"
    last_heartbeat: datetime | None
    online: bool
    gpus: list[dict]
    models: list[dict]  # modelos cargados con sus capabilities
    load: dict           # active_requests, gpu_util avg

class FederationManager:
    _peers: dict[str, PeerState]

    async def start()       # Arranca heartbeat loop
    async def stop()        # Para el loop
    async def _heartbeat_loop()  # Polling periódico a cada peer

    def get_online_peers() -> list[PeerState]
    def get_remote_models() -> dict[str, list[PeerState]]
        # model_id → lista de peers que lo tienen cargado

    def find_best_peer(model_id: str) -> PeerState | None
        # Load balancing: elige peer con menos carga
```

#### 1.4 — Tabla de peers (persistencia)

```python
class FederationPeer(Base):
    __tablename__ = "federation_peers"
    id: UUID (PK)
    name: str
    url: str
    api_key_encrypted: str  # cifrada con Fernet (key derivada de SECRET_KEY)
    access_level: str       # "inference" | "full"
    enabled: bool
    created_at: datetime
    updated_at: datetime
```

Esto permite gestionar peers desde la UI (añadir, eliminar, habilitar/deshabilitar)
sin reiniciar el servidor. La config de `config.py` sirve como seed inicial.

#### Entregables Fase 1
- [ ] `backend/ocabra/core/federation.py` — FederationManager
- [ ] `backend/ocabra/db/federation.py` — FederationPeer model
- [ ] Migración Alembic para `federation_peers`
- [ ] Settings de federación en `config.py`
- [ ] `GET /ocabra/federation/heartbeat` endpoint
- [ ] `GET /ocabra/federation/peers` — listar peers y su estado
- [ ] `POST/DELETE /ocabra/federation/peers` — CRUD de peers
- [ ] Heartbeat loop con reconexión y backoff exponencial
- [ ] Tests: heartbeat, peer online/offline, reconexión

---

### Fase 2 — Proxy de inferencia transparente

**Objetivo:** Una petición a un modelo remoto se resuelve automáticamente.

#### 2.1 — Resolución de modelos federados

Extender la resolución de modelos en las capas OpenAI y Ollama:

```
1. ¿El modelo/perfil existe localmente y está LOADED? → servir local
2. ¿El modelo existe localmente pero está UNLOADED? → cargar local (comportamiento actual)
3. ¿El modelo existe en un peer remoto y está LOADED? → proxificar al peer
4. → Error 404
```

El paso 3 es nuevo. Se activa solo si `federation_enabled = True`.

#### 2.2 — Proxy HTTP para requests no-streaming

```python
# En FederationManager
async def proxy_request(
    peer: PeerState,
    path: str,        # "/v1/chat/completions"
    body: dict,
    headers: dict,
    timeout: float = 300.0
) -> Response:
    """
    Reenvía la petición al peer remoto usando su API key.
    Reescribe el header Authorization con la key del peer.
    """
```

#### 2.3 — Proxy SSE para streaming

```python
async def proxy_stream(
    peer: PeerState,
    path: str,
    body: dict,
    headers: dict,
) -> AsyncIterator[bytes]:
    """
    Abre conexión SSE al peer remoto y retransmite chunks.
    Cierra la conexión upstream si el cliente se desconecta.
    """
```

#### 2.4 — Load balancing

Cuando un modelo está disponible en múltiples ubicaciones (local + N peers):

```python
def select_target(model_id: str) -> "local" | PeerState:
    """
    Estrategia de selección:
    1. Si el modelo está cargado localmente Y en peers remotos:
       - Calcular score de carga para cada candidato:
         score = active_requests * 10 + gpu_utilization_avg_pct
       - Seleccionar el de menor score
       - Preferencia local (bias): restar 5 puntos al score local
         (evitar latencia de red cuando la carga es similar)
    2. Si solo está en remoto: elegir peer con menor carga
    3. Si solo está en local: servir local
    """
```

#### 2.5 — Request stats federadas

Cuando un request se proxifica a un peer remoto:
- El nodo local registra un `request_stat` con `source = "federation_proxy"`
  y un campo `remote_node_id` para trazabilidad.
- El nodo remoto registra su propio `request_stat` normalmente (con el `api_key`
  del peer, no del usuario original).
- No se intenta sincronizar stats entre nodos.

#### Entregables Fase 2
- [ ] Middleware de resolución federada en OpenAI router
- [ ] Middleware de resolución federada en Ollama router
- [ ] `proxy_request()` y `proxy_stream()` en FederationManager
- [ ] Load balancer con scoring y bias local
- [ ] Campo `remote_node_id` en `request_stats` (migración)
- [ ] Timeout y error handling: peer caído mid-request → retry en otro peer o error
- [ ] Tests: proxy non-streaming, proxy streaming, load balancing, failover

---

### Fase 3 — Inventario federado en `/v1/models`

**Objetivo:** Los endpoints de listado muestran modelos locales + remotos.

#### 3.1 — `/v1/models` incluye modelos remotos

```json
{
  "data": [
    {
      "id": "qwen3-32b",
      "object": "model",
      "owned_by": "nodo-A",
      "federation": {
        "remote": true,
        "node_name": "nodo-A",
        "node_id": "uuid"
      }
    },
    {
      "id": "qwen2.5-3b",
      "object": "model",
      "owned_by": "local"
    }
  ]
}
```

#### 3.2 — `/api/tags` (Ollama) incluye modelos remotos

Mismo patrón: modelos remotos aparecen con metadata indicando su nodo de origen.

#### 3.3 — `/ocabra/models` (admin) muestra inventario federado

Vista admin con columna "Nodo" para cada modelo. Los modelos remotos se muestran
como read-only (no se pueden configurar desde otro nodo a menos que `access_level = full`).

#### 3.4 — Deduplicación de modelos

Si el mismo modelo (mismo `backend/model_id` canónico) está en múltiples nodos,
aparece **una sola vez** en `/v1/models` con metadata indicando los nodos disponibles.
El load balancer decide a cuál enviar.

#### Entregables Fase 3
- [ ] Merge de inventario remoto en `/v1/models`
- [ ] Merge de inventario remoto en `/api/tags`
- [ ] Indicadores de nodo en `/ocabra/models`
- [ ] Deduplicación por `model_id` canónico
- [ ] Tests: listado con peers online/offline, deduplicación

---

### Fase 4 — UI de federación

**Objetivo:** Gestionar la federación desde el dashboard.

#### 4.1 — Página/sección "Federation" en Settings

- Lista de peers con estado (online/offline), última conexión, latencia.
- Botón "Añadir peer" con formulario (nombre, URL, API key, nivel de acceso).
- Toggle habilitar/deshabilitar peer.
- Botón "Test connection" que verifica el heartbeat.

#### 4.2 — Indicadores en la página Models

- Badge "Remote" o icono de nodo en modelos que vienen de peers.
- Tooltip con nombre del nodo y latencia.
- En el detalle de modelo: lista de nodos donde está disponible.

#### 4.3 — Dashboard federado

- Panel resumen con: nodos online, GPUs totales del cluster, VRAM total/libre.
- En el feed de requests: indicar cuáles fueron proxificadas.

#### 4.4 — WebSocket de eventos de federación

Extender el WS existente (`/ocabra/ws`) con eventos:
- `peer_online` / `peer_offline`
- `remote_model_loaded` / `remote_model_unloaded`

#### Entregables Fase 4
- [ ] Sección Federation en Settings (frontend)
- [ ] CRUD de peers desde la UI
- [ ] Badges de nodo en Models page
- [ ] Panel de cluster en Dashboard
- [ ] Eventos WS de federación
- [ ] Tests E2E de la UI de federación

---

### Fase 5 — Operaciones remotas (access_level = "full")

**Objetivo:** Nodos con acceso `full` pueden gestionar modelos en peers remotos.

#### 5.1 — Carga/descarga remota

Desde la UI de `nodo-A`, cargar o descargar un modelo en `nodo-B`.
Reutiliza los endpoints existentes — el FederationManager proxifica
`POST /ocabra/models/{id}/load` al peer.

#### 5.2 — Descarga de modelos en nodo remoto

Disparar `POST /ocabra/downloads` en un peer para que descargue un modelo
desde HuggingFace. El progreso se retransmite via SSE.

#### 5.3 — Monitorización de GPUs remotas

Ver las GPUs del peer con sus stats (temp, VRAM, utilización) en la UI local.
El heartbeat ya trae un resumen; para detalle, proxy a `/ocabra/gpus`.

#### Entregables Fase 5
- [ ] Proxy de operaciones admin a peers con `access_level = full`
- [ ] UI para load/unload remoto
- [ ] UI para descargas remotas con progreso SSE
- [ ] Vista de GPUs remotas
- [ ] Tests: operaciones remotas, permisos, error handling

---

## Protocolo inter-nodo (resumen)

| Operación | Método | Endpoint en el peer | Access level |
|-----------|--------|---------------------|--------------|
| Heartbeat | GET | `/ocabra/federation/heartbeat` | inference |
| Inferencia | POST | `/v1/chat/completions`, `/v1/embeddings`, etc. | inference |
| Stream | POST | `/v1/chat/completions` (SSE) | inference |
| Listar modelos | GET | `/v1/models` | inference |
| Cargar modelo | POST | `/ocabra/models/{id}/load` | full |
| Descargar modelo | POST | `/ocabra/downloads` | full |
| GPUs detalle | GET | `/ocabra/gpus` | full |

**Autenticación:** Cada petición inter-nodo lleva `Authorization: Bearer {api_key_del_peer}`.
El peer valida la key con su auth system normal. Se recomienda crear un usuario
dedicado `federation` con rol `system_admin` (para `full`) o `user` (para `inference`).

---

## Consideraciones de seguridad

1. **API keys de peers cifradas** en BD (Fernet, key derivada de `SECRET_KEY`).
2. **No reenviar credenciales del usuario original** al peer — el peer solo ve la
   API key del nodo federado.
3. **Rate limiting por peer** para evitar que un nodo sobrecargue a otro.
4. **TLS obligatorio** para peers en WAN. En LAN, configurable.
5. **Validación de certificados** configurable (`federation_verify_ssl: bool = True`).

---

## Consideraciones de red

1. **Timeout de proxy** configurable: `federation_proxy_timeout_s: int = 300`.
2. **Retry**: si un peer falla mid-request y hay otro peer con el modelo,
   reintentar en el otro (solo para requests no-streaming o antes del primer chunk).
3. **Circuit breaker**: si un peer falla N heartbeats consecutivos, marcarlo offline
   y no intentar proxy hasta que vuelva. Backoff exponencial en reconexión.
4. **Latencia**: el heartbeat mide RTT. El load balancer puede usar la latencia
   como factor adicional en el scoring para peers WAN.

---

## Impacto en el código existente

| Módulo | Cambio |
|--------|--------|
| `config.py` | Nuevos settings `federation_*` |
| `db/` | Nueva tabla `federation_peers` |
| `core/` | Nuevo `federation.py` con FederationManager |
| `api/internal/` | Nuevo router `federation.py` |
| `api/openai/` | Hook de resolución federada antes de `ensure_loaded()` |
| `api/ollama/` | Idem |
| `api/internal/models.py` | Merge de inventario remoto |
| `db/stats.py` | Campo `remote_node_id` en `request_stats` |
| `api/internal/ws.py` | Eventos de federación |
| `frontend/` | Sección Federation en Settings, badges en Models |
| `main.py` | Inicializar FederationManager en lifespan |

**NO se modifican:** backends, workers, GPU manager, scheduler, service manager.

---

## Orden de ejecución recomendado

```
Fase 1 — Peers + heartbeat          (fundación, sin impacto en inferencia)
Fase 2 — Proxy de inferencia        (funcionalidad core, el valor principal)
Fase 3 — Inventario federado        (UX: ver modelos remotos en listados)
Fase 4 — UI de federación           (gestión visual)
Fase 5 — Operaciones remotas        (gestión avanzada, solo para access_level=full)
```

Fases 1-2 son el **MVP funcional**. Un usuario puede conectar dos nodos y hacer
inferencia transparente en modelos remotos. Fases 3-5 son mejora progresiva.

---

## Fuera de alcance (v1)

- Federación de servicios interactivos (ComfyUI, HunyuanVideo, etc.)
- Replicación automática de modelos entre nodos
- Descubrimiento automático de peers (mDNS, etc.)
- Consenso distribuido / elección de líder
- Base de datos compartida entre nodos
- Migración en caliente de modelos entre nodos
- Realtime API WebSocket federado (requiere relay de WebSocket, complejidad alta)
