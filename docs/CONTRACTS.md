# Contratos de interfaces entre módulos

Este documento define las interfaces que deben respetar todos los agentes.
**Consulta este fichero antes de implementar cualquier integración entre módulos.**
Si necesitas cambiar un contrato, documenta el cambio aquí antes de implementarlo.

---

## 1. BackendInterface — Contrato de backends de inferencia

Todo backend (vLLM, Diffusers, Whisper, TTS) debe implementar esta interfaz abstracta.

```python
# backend/ocabra/backends/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Any

@dataclass
class BackendCapabilities:
    chat: bool = False
    completion: bool = False
    tools: bool = False
    vision: bool = False
    embeddings: bool = False
    reasoning: bool = False
    image_generation: bool = False
    audio_transcription: bool = False
    tts: bool = False
    streaming: bool = False
    context_length: int = 0

@dataclass
class WorkerInfo:
    backend_type: str          # "vllm" | "bitnet" | "diffusers" | "whisper" | "tts" | "ollama"
    model_id: str
    gpu_indices: list[int]
    port: int
    pid: int
    vram_used_mb: int

class BackendInterface(ABC):

    @abstractmethod
    async def load(self, model_id: str, gpu_indices: list[int], **kwargs) -> WorkerInfo:
        """Carga el modelo. Retorna WorkerInfo con el proceso activo."""

    @abstractmethod
    async def unload(self, model_id: str) -> None:
        """Descarga el modelo y libera recursos."""

    @abstractmethod
    async def health_check(self, model_id: str) -> bool:
        """Retorna True si el worker está listo para recibir peticiones."""

    @abstractmethod
    async def get_capabilities(self, model_id: str) -> BackendCapabilities:
        """Detecta y retorna las capacidades del modelo."""

    @abstractmethod
    async def get_vram_estimate_mb(self, model_id: str) -> int:
        """Estima la VRAM necesaria antes de cargar el modelo."""

    @abstractmethod
    async def forward_request(self, model_id: str, path: str, body: dict) -> Any:
        """Reenvía una petición al worker. Retorna la respuesta."""

    @abstractmethod
    async def forward_stream(self, model_id: str, path: str, body: dict) -> AsyncIterator[bytes]:
        """Reenvía una petición streaming al worker."""
```

---

## 2. GPUManager — Contrato de estado de GPU

```python
# Producido por: core/gpu_manager.py
# Consumido por: core/scheduler.py, api/internal/gpus.py, stats/gpu_power.py

@dataclass
class GPUState:
    index: int
    name: str
    total_vram_mb: int
    free_vram_mb: int
    used_vram_mb: int
    utilization_pct: float       # 0-100
    temperature_c: float
    power_draw_w: float
    power_limit_w: float
    locked_vram_mb: int          # VRAM reservada por modelos cargados

# Publicado en Redis como evento cada N segundos:
# Canal: "gpu:stats"
# Payload: list[GPUState] serializado como JSON
```

---

## 3. ModelManager — Contrato de estado de modelo

```python
# Producido por: core/model_manager.py
# Consumido por: api/internal/models.py, api/openai/models.py, api/ollama/tags.py

from enum import Enum

class ModelStatus(str, Enum):
    DISCOVERED = "discovered"    # Conocido pero no configurado
    CONFIGURED = "configured"    # Configurado, no cargado
    LOADING = "loading"
    LOADED = "loaded"
    UNLOADING = "unloading"
    UNLOADED = "unloaded"
    ERROR = "error"

class LoadPolicy(str, Enum):
    PIN = "pin"             # Siempre cargado, inmune a idle eviction
    WARM = "warm"           # Cargado bajo demanda, no baja por idle
    ON_DEMAND = "on_demand" # Cargado bajo demanda, baja por idle

@dataclass
class ModelState:
    model_id: str               # e.g. "mistral-7b-instruct"
    display_name: str
    backend_type: str           # "vllm" | "bitnet" | "diffusers" | "whisper" | "tts" | "ollama"
    status: ModelStatus
    load_policy: LoadPolicy
    auto_reload: bool           # Recargar automáticamente tras eviction
    preferred_gpu: int | None   # None = usa el default del servidor
    current_gpu: list[int]      # GPUs actuales (puede ser más de una)
    vram_used_mb: int
    capabilities: BackendCapabilities
    last_request_at: datetime | None
    loaded_at: datetime | None
    worker_info: WorkerInfo | None

# Publicado en Redis como evento en cada cambio de estado:
# Canal: "model:events"
# Payload: {"event": "status_changed", "model_id": str, "new_status": ModelStatus}
```

---

## 4. WorkerPool — Contrato de registro de workers

```python
# backend/ocabra/core/worker_pool.py

class WorkerPool:
    async def register_backend(self, backend_type: str, backend: BackendInterface) -> None:
        """Registra un backend disponible."""

    async def get_worker(self, model_id: str) -> WorkerInfo | None:
        """Retorna el WorkerInfo del modelo si está cargado."""

    async def assign_port(self) -> int:
        """Asigna un puerto libre del rango configurado (default: 18000-19000)."""

    async def release_port(self, port: int) -> None:
        """Libera un puerto."""
```

---

## 5. API Interna oCabra — Contratos REST

Base URL: `/ocabra/`
Todos los endpoints retornan `Content-Type: application/json`.
Errores siguen el formato: `{"detail": str, "code": str}`.

### 5.1 Modelos

```
GET    /ocabra/models                    → list[ModelState]
GET    /ocabra/models/{model_id}         → ModelState
POST   /ocabra/models/{model_id}/load    → ModelState
POST   /ocabra/models/{model_id}/unload  → {"ok": true}
PATCH  /ocabra/models/{model_id}         → ModelState
  body: {
    load_policy?: "pin"|"warm"|"on_demand",
    preferred_gpu?: int|null,
    auto_reload?: bool,
    display_name?: str
  }
DELETE /ocabra/models/{model_id}         → {"ok": true}  # elimina config y ficheros
```

### 5.2 GPUs

```
GET /ocabra/gpus               → list[GPUState]
GET /ocabra/gpus/{index}       → GPUState
GET /ocabra/gpus/{index}/stats?window=5m → GPUStatHistory
```

### 5.3 Descargas

```
GET    /ocabra/downloads                 → list[DownloadJob]
POST   /ocabra/downloads                 → DownloadJob
  body: { source: "huggingface"|"ollama", model_ref: str }
DELETE /ocabra/downloads/{job_id}        → {"ok": true}  # cancela descarga
GET    /ocabra/downloads/{job_id}/stream → SSE stream de progreso
```

### 5.4 Registry

```
GET /ocabra/registry/hf/search?q=str&task=str&limit=20  → list[HFModelCard]
GET /ocabra/registry/hf/{repo_id}                        → HFModelDetail
GET /ocabra/registry/ollama/search?q=str                 → list[OllamaModelCard]
GET /ocabra/registry/local                               → list[LocalModel]
```

### 5.5 Stats

```
GET /ocabra/stats/requests?from=ISO&to=ISO&model_id=str  → RequestStats
GET /ocabra/stats/tokens?from=ISO&to=ISO                 → TokenStats
GET /ocabra/stats/energy?from=ISO&to=ISO                 → EnergyStats
GET /ocabra/stats/performance?model_id=str               → PerformanceStats
```

### 5.6 Config

```
GET   /ocabra/config           → ServerConfig
PATCH /ocabra/config           → ServerConfig
POST  /ocabra/config/litellm/sync → {"synced_models": int}
```

### 5.7 Servicios interactivos

```
GET  /ocabra/services                    → list[ServiceState]
GET  /ocabra/services/{service_id}       → ServiceState
POST /ocabra/services/{service_id}/refresh → ServiceState
POST /ocabra/services/{service_id}/touch   → ServiceState
  body: {
    runtime_loaded?: bool,
    active_model_ref?: str|null,
    detail?: str|null
  }
PATCH /ocabra/services/{service_id}/runtime → ServiceState
  body: {
    runtime_loaded: bool,
    active_model_ref?: str|null,
    detail?: str|null
  }
POST /ocabra/services/{service_id}/unload → ServiceState
```

`service_id` iniciales:
- `hunyuan`
- `comfyui`
- `a1111`

`ServiceState`:

```python
@dataclass
class ServiceState:
    service_id: str               # "hunyuan" | "comfyui" | "a1111"
    service_type: str             # "hunyuan3d" | "comfyui" | "automatic1111"
    display_name: str
    base_url: str                 # URL interna Docker
    ui_base_path: str             # path proxied, e.g. "/comfy"
    health_path: str
    unload_path: str | None
    preferred_gpu: int | None
    idle_unload_after_seconds: int
    service_alive: bool           # proceso/UI accesible por healthcheck
    runtime_loaded: bool          # hay runtime/pesos cargados en VRAM
    status: str                   # "unknown" | "idle" | "active" | "unreachable"
    active_model_ref: str | None
    last_activity_at: datetime | None
    last_health_check_at: datetime | None
    last_unload_at: datetime | None
    detail: str | None
    extra: dict
```

### 5.8 Tiempo real (WebSocket)

```
WS /ocabra/ws
```

Eventos emitidos por el servidor (JSON):
```json
{"type": "gpu_stats",    "data": [GPUState, ...]}
{"type": "model_event",  "data": {"event": str, "model_id": str, "status": str}}
{"type": "service_event", "data": {"event": str, "service_id": str, "status": str, "service": ServiceState}}
{"type": "download_progress", "data": {"job_id": str, "pct": float, "speed_mb_s": float}}
{"type": "system_alert", "data": {"level": "warn"|"error", "message": str}}
```

---

## 6. Redis — Keys y canales

```
# Canales pub/sub
gpu:stats                   → GPUState[] cada 2s
model:events                → ModelEvent en cada cambio
service:events              → ServiceEvent en cada cambio de servicio
download:progress:{job_id}  → DownloadProgress cada 500ms

# Keys (state)
model:state:{model_id}      → ModelState (JSON, TTL: none)
service:state:{service_id}  → ServiceState (JSON, TTL: none)
gpu:state:{index}           → GPUState (JSON, TTL: 5s)
download:job:{job_id}       → DownloadJob (JSON)

# Queues (listas Redis)
queue:load                  → {model_id, priority, requested_at}
queue:unload                → {model_id, reason}
```

---

## 7. Esquema de base de datos — Tablas principales

```sql
-- Configuración de modelos
model_configs (
  id            UUID PRIMARY KEY,
  model_id      TEXT UNIQUE NOT NULL,
  display_name  TEXT,
  backend_type  TEXT NOT NULL,
  load_policy   TEXT NOT NULL DEFAULT 'on_demand',
  auto_reload   BOOLEAN DEFAULT false,
  preferred_gpu INTEGER,
  extra_config  JSONB,          -- parámetros específicos del backend
  created_at    TIMESTAMPTZ,
  updated_at    TIMESTAMPTZ
)

-- Schedules de evicción
eviction_schedules (
  id          UUID PRIMARY KEY,
  model_id    TEXT REFERENCES model_configs(model_id),  -- NULL = global
  cron_expr   TEXT NOT NULL,    -- expresión cron
  action      TEXT NOT NULL,    -- 'evict_warm' | 'evict_all' | 'reload'
  enabled     BOOLEAN DEFAULT true
)

-- Stats por request
request_stats (
  id              UUID PRIMARY KEY,
  model_id        TEXT NOT NULL,
  gpu_index       INTEGER,
  started_at      TIMESTAMPTZ NOT NULL,
  duration_ms     INTEGER,
  input_tokens    INTEGER,
  output_tokens   INTEGER,
  energy_wh       FLOAT,        -- vatios-hora estimados
  error           TEXT
)

-- Stats de GPU (serie temporal, agregada por minuto)
gpu_stats (
  recorded_at      TIMESTAMPTZ NOT NULL,
  gpu_index        INTEGER NOT NULL,
  utilization_pct  FLOAT,
  vram_used_mb     INTEGER,
  power_draw_w     FLOAT,
  temperature_c    FLOAT,
  PRIMARY KEY (recorded_at, gpu_index)
)

-- Config del servidor
server_config (
  key    TEXT PRIMARY KEY,
  value  JSONB NOT NULL
)
```

---

## 8. Variables de entorno (contratos de configuración)

```bash
# Base de datos
DATABASE_URL=postgresql+asyncpg://user:pass@postgres:5432/ocabra

# Redis
REDIS_URL=redis://redis:6379/0

# Modelos
MODELS_DIR=/data/models          # Carpeta raíz donde se guardan los modelos
HF_TOKEN=                        # Token HuggingFace (opcional, para modelos privados/gated)

# GPUs
DEFAULT_GPU_INDEX=1              # GPU preferida por defecto (0=3060, 1=3090)
WORKER_PORT_RANGE_START=18000
WORKER_PORT_RANGE_END=19000

# LiteLLM
LITELLM_BASE_URL=http://litellm:4000
LITELLM_ADMIN_KEY=

# Energía
ENERGY_COST_EUR_KWH=0.15        # Para estimación de coste

# Servidor
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8000
```
