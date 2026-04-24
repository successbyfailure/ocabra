# Contratos de interfaces entre módulos

Este documento define las interfaces que deben respetar todos los agentes.
**Consulta este fichero antes de implementar cualquier integración entre módulos.**
Si necesitas cambiar un contrato, documenta el cambio aquí antes de implementarlo.

---

## 1. BackendInterface — Contrato de backends de inferencia

Todo backend de inferencia (vLLM, Diffusers, Whisper, TTS, Ollama, llama.cpp, SGLang, TensorRT-LLM, BitNet, ACE-Step) debe implementar esta interfaz abstracta.

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
    pooling: bool = False
    rerank: bool = False
    classification: bool = False
    score: bool = False
    reasoning: bool = False
    image_generation: bool = False
    audio_transcription: bool = False
    tts: bool = False
    music_generation: bool = False
    streaming: bool = False
    context_length: int = 0

@dataclass
class WorkerInfo:
    backend_type: str           # Backend prefix in model_id: "vllm" | "bitnet" | "acestep" | "diffusers" | "whisper" | "tts" | "ollama" | "llama_cpp" | "sglang" | "tensorrt_llm"
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
    processes: list[GPUProcessInfo]



@dataclass
class GPUProcessInfo:
    pid: int
    process_name: str | None
    process_type: str            # "compute" | "graphics"
    used_vram_mb: int

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
    model_id: str               # Canonical: "backend/model" (e.g. "vllm/mistral-7b-instruct")
                            # Formato legacy sin prefijo ya no es válido.
    backend_model_id: str       # Backend-native id (without prefix), e.g. "mistral-7b-instruct"
    display_name: str
    backend_type: str           # Backend prefix in model_id: "vllm" | "bitnet" | "acestep" | "diffusers" | "whisper" | "tts" | "ollama" | "llama_cpp" | "sglang" | "tensorrt_llm"
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
    error_message: str | None
    extra_config: dict

# Publicado en Redis como evento en cada cambio de estado:
# Canal: "model:events"
# Payload: {"event": "status_changed", "model_id": str, "new_status": ModelStatus}
```

---

## 4. WorkerPool — Contrato de registro de workers

```python
# backend/ocabra/core/worker_pool.py

class WorkerPool:
    def register_backend(self, backend_type: str, backend: BackendInterface) -> None:
        """Registra un backend disponible."""

    def register_disabled_backend(self, backend_type: str, reason: str) -> None:
        """Marca un backend como deshabilitado."""

    async def get_backend(self, backend_type: str) -> BackendInterface:
        """Retorna el backend registrado o falla si está deshabilitado/no existe."""

    def get_worker(self, model_id: str) -> WorkerInfo | None:
        """Retorna el WorkerInfo del modelo si está cargado."""

    async def assign_port(self) -> int:
        """Asigna un puerto libre del rango configurado (default: 18000-19000)."""

    def set_worker(self, model_id: str, info: WorkerInfo) -> None:
        """Registra un worker cargado."""

    def remove_worker(self, model_id: str) -> None:
        """Elimina un worker cargado y libera su puerto."""

    def release_port(self, port: int) -> None:
        """Libera un puerto."""

    async def forward_request(self, model_id: str, path: str, body: dict) -> Any:
        """Reenvía una petición HTTP al worker."""

    async def forward_stream(self, model_id: str, path: str, body: dict) -> AsyncIterator[bytes]:
        """Reenvía una petición streaming al worker."""
```

---

## 5. API Interna oCabra — Contratos REST

Base URL: `/ocabra/`
Todos los endpoints retornan `Content-Type: application/json`.
Errores internos de `/ocabra/*` siguen el formato estándar de FastAPI `{"detail": ...}`.
Los endpoints de compatibilidad OpenAI y Ollama usan sus propios envelopes de error.

### 5.1 Modelos

```
GET    /ocabra/models                    → list[ModelState]  # model_id is canonical backend/model; no legacy bare-name fallback at this boundary
GET    /ocabra/models/storage            → ModelsStorageStats

Nota OpenAI `/v1/*`: el campo `model` acepta `model_id` canónico y también `backend_model_id` como alias. Si hay múltiples coincidencias de alias, se usa la primera.

GET    /ocabra/models/{model_id}         → ModelState
POST   /ocabra/models/{model_id}/load    → ModelState
POST   /ocabra/models/{model_id}/unload  → ModelState
POST   /ocabra/models/{model_id}/memory-estimate → ModelMemoryEstimate
  body: {
    preferred_gpu?: int|null,
    extra_config?: dict|null,
    run_probe?: bool
  }
PATCH  /ocabra/models/{model_id}         → ModelState
  body: {
    load_policy?: "pin"|"warm"|"on_demand",
    preferred_gpu?: int|null,
    idle_timeout_seconds?: int|null,
    auto_reload?: bool,
    display_name?: str,
    extra_config?: dict
  }
DELETE /ocabra/models/{model_id}         → {"ok": true, "deleted_path": str|null}  # elimina config y ficheros
```

`ModelState` REST payload:
- `model_id`, `backend_model_id`, `display_name`, `backend_type`
- `status`, `load_policy`, `auto_reload`, `preferred_gpu`, `current_gpu`, `vram_used_mb`
- `capabilities`, `last_request_at`, `loaded_at`, `error_message`, `extra_config`
- `disk_size_bytes` se añade en `/ocabra/models` y `/ocabra/models/{model_id}` cuando se puede estimar

`ModelMemoryEstimate` REST payload:
- `backend_type`, `preferred_gpu`, `source`
- `estimated_total_mb` cuando aplica
- `estimated_weights_mb` para backends basados en pesos
- `estimated_engine_mb_per_gpu` y `engine_present` para `tensorrt_llm`
- `estimated_kv_cache_mb`, `estimated_max_context_length`, `maximum_concurrency` cuando el backend puede derivarlos
- `status: "ok" | "warning" | "error"` si se ejecuta probe runtime
- `warning`, `error`, `notes` como señales diagnósticas legibles

Semántica:
- `source="heuristic"` implica cálculo rápido sin arrancar runtime.
- `source="runtime_probe"` implica validación real con el engine; se usa sobre todo en `vllm`.
- El endpoint no persiste cambios por sí mismo; solo evalúa la configuración propuesta.

### 5.2 GPUs

```
GET /ocabra/gpus               → list[GPUState]
GET /ocabra/gpus/{index}       → GPUState
GET /ocabra/gpus/{index}/stats?window=5m|1h|24h → list[GPUStat]
```

### 5.3 Descargas

```
GET    /ocabra/downloads                 → list[DownloadJob]
POST   /ocabra/downloads                 → DownloadJob
  body: {
    source: "huggingface"|"ollama"|"bitnet",
    model_ref: str,
    artifact?: str|null,
    register_config?: dict|null
  }
DELETE /ocabra/downloads/{job_id}        → {"ok": true}  # cancela descarga
GET    /ocabra/downloads/{job_id}/stream → SSE stream de progreso
```

### 5.4 Registry

```
GET /ocabra/registry/hf/search?q=str&task=str&limit=20  → list[HFModelCard]
GET /ocabra/registry/hf/{repo_id}                        → HFModelDetail
GET /ocabra/registry/hf/{repo_id}/variants              → list[HFModelVariant]
GET /ocabra/registry/bitnet/search?q=str&limit=20       → list[HFModelCard]
GET /ocabra/registry/bitnet/{repo_id}/variants          → list[HFModelVariant]
GET /ocabra/registry/ollama/search?q=str                 → list[OllamaModelCard]
GET /ocabra/registry/ollama/{model_name}/variants       → list[OllamaModelVariant]
GET /ocabra/registry/local                               → list[LocalModel]
```

### 5.5 Stats

```
GET /ocabra/stats/requests?from=ISO&to=ISO&model_id=str   → RequestStats
GET /ocabra/stats/tokens?from=ISO&to=ISO&model_id=str     → TokenStats
GET /ocabra/stats/energy?from=ISO&to=ISO                  → EnergyStats
GET /ocabra/stats/performance?from=ISO&to=ISO&model_id=str → PerformanceStats
GET /ocabra/stats/overview?from=ISO&to=ISO&model_id=str   → OverviewStats
```

`RequestStats`:
- `totalRequests`, `errorRate`, `avgDurationMs`, `p50DurationMs`, `p95DurationMs`
- `series`: `[{ timestamp, count }]`

`TokenStats`:
- `totalInputTokens`, `totalOutputTokens`
- `byBackend`: `[{ backendType, inputTokens, outputTokens }]`
- `series`: `[{ timestamp, inputTokens, outputTokens }]`

`EnergyStats`:
- `totalKwh`, `estimatedCostEur`
- `byGpu`: `[{ gpuIndex, totalKwh, powerDrawW }]`

`PerformanceStats`:
- `byModel`: `[{ modelId, backendType, requestKinds, totalRequests, avgLatencyMs, p95LatencyMs, requestsPerMinute, tokensPerSecond, totalInputTokens, totalOutputTokens, tokenizedRequests, errorCount, uptimePct, loadCount, avgLoadMs, p95LoadMs, lastLoadMs }]`

`OverviewStats`:
- `totalRequests`, `totalErrors`, `avgDurationMs`, `tokenizedRequests`, `totalInputTokens`, `totalOutputTokens`
- `byBackend`: `[{ backendType, totalRequests, errorRate, avgLatencyMs, p95LatencyMs }]`
- `byRequestKind`: `[{ requestKind, totalRequests, errorRate, avgLatencyMs, p95LatencyMs }]`

### 5.6 Config

```
GET   /ocabra/config             → ServerConfig
PATCH /ocabra/config             → ServerConfig
POST  /ocabra/config/litellm/sync → {"synced_models": int, "errors": list[str]}
```

`ServerConfig` (campos relevantes):
- `defaultGpuIndex`, `idleTimeoutSeconds`, `idleEvictionCheckIntervalSeconds`
- `vramBufferMb`, `vramPressureThresholdPct`, `logLevel`
- `litellmBaseUrl`, `litellmAdminKey` (enmascarada), `litellmAutoSync`
- `energyCostEurKwh`, `modelsDir`, `downloadDir`, `maxTemperatureC`
- `globalSchedules`: `list[EvictionSchedule]` persistida en la tabla `eviction_schedules` y reconstruida por `GET /ocabra/config`
- `modelsDir` proviene de `MODELS_DIR` y es de solo lectura en runtime
- `downloadDir` y `maxTemperatureC` siguen como overrides de proceso en `request.app.state.config_overrides`
- `globalSchedules` se traduce internamente a dos filas por ventana en `eviction_schedules`: una acción `evict_all` al inicio y una acción `reload` al final
- `check_schedule_reloads()` exige `auto_reload=True` además de `pin`/`warm`
- `GET /ocabra/config` y `PATCH /ocabra/config` usan claves camelCase; no hay fallback legacy a snake_case o `localStorage`.

### 5.7 Servicios interactivos

```
GET  /ocabra/services                    → list[ServiceState]
GET  /ocabra/services/{service_id}       → ServiceState
POST /ocabra/services/{service_id}/refresh → ServiceState
POST /ocabra/services/{service_id}/start   → ServiceState
PATCH /ocabra/services/{service_id}         → ServiceState
  body: {
    enabled: bool
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
- `acestep`


Campos relevantes de `ServiceState`:
- `enabled: bool`
- `status`: `"unknown" | "idle" | "active" | "unreachable" | "disabled"`

`ServiceState`:

```python
@dataclass
class ServiceState:
    service_id: str               # "hunyuan" | "comfyui" | "a1111" | "acestep"
    service_type: str             # "hunyuan3d" | "comfyui" | "automatic1111" | "acestep"
    display_name: str
    base_url: str                 # URL interna Docker
    ui_url: str                   # path proxied, e.g. "/comfy"
    health_path: str
    runtime_check_path: str | None
    runtime_check_key: str
    runtime_check_model_key: str | None
    unload_path: str | None
    unload_method: str
    unload_payload: dict | None
    post_unload_flush_path: str | None
    docker_container_name: str | None
    runtime_loaded_when_alive: bool
    preferred_gpu: int | None
    idle_unload_after_seconds: int
    idle_action: str              # "stop" | "restart"
    enabled: bool                # servicio habilitado para uso en oCabra
    service_alive: bool           # proceso/UI accesible por healthcheck
    runtime_loaded: bool          # hay runtime/pesos cargados en VRAM
    status: str                   # "unknown" | "idle" | "active" | "unreachable" | "disabled"
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
```

---

## 5.8 OpenAI Files API

```
POST   /v1/files                     → FileObject
  form: file (multipart), purpose: "batch"|"batch_output"|"user_data"
  Sube un fichero al almacenamiento interno. Para batches, el fichero
  debe ser JSONL con propósito "batch".

GET    /v1/files                     → {object: "list", data: [FileObject]}
  query: purpose? (filtro opcional), limit? (1-10000, default 100)

GET    /v1/files/{file_id}           → FileObject
  file_id acepta formato "file-<uuid>" o UUID directo.

GET    /v1/files/{file_id}/content   → binary (application/octet-stream)
  Descarga el contenido del fichero.

DELETE /v1/files/{file_id}           → {id, object: "file", deleted: true}
  No permite borrar ficheros en uso por un batch activo (409).
```

`FileObject`:
```json
{
  "id": "file-<uuid>",
  "object": "file",
  "bytes": 1234,
  "created_at": 1700000000,
  "filename": "batch_input.jsonl",
  "purpose": "batch",
  "status": "uploaded",
  "status_details": null
}
```

### 5.9 OpenAI Batches API

```
POST   /v1/batches                   → BatchObject
  body: {
    input_file_id: "file-<uuid>",
    endpoint: "/v1/chat/completions"|"/v1/embeddings"|"/v1/completions",
    completion_window: "24h",
    metadata?: dict
  }
  Crea un batch job. El fichero de input debe ser JSONL con purpose="batch".
  Cada línea: {"custom_id": str, "method": "POST", "url": str, "body": dict}

GET    /v1/batches                   → {object: "list", data: [BatchObject]}
  query: after? (cursor pagination), limit? (1-100, default 20)

GET    /v1/batches/{batch_id}        → BatchObject

POST   /v1/batches/{batch_id}/cancel → BatchObject
  Transiciona a "cancelling" → "cancelled". No afecta batches terminales.
```

`BatchObject`:
```json
{
  "id": "batch_<uuid>",
  "object": "batch",
  "endpoint": "/v1/chat/completions",
  "status": "validating|in_progress|finalizing|completed|failed|cancelled|cancelling",
  "input_file_id": "file-<uuid>",
  "output_file_id": "file-<uuid>|null",
  "error_file_id": "file-<uuid>|null",
  "completion_window": "24h",
  "request_counts": {"total": N, "completed": N, "failed": N},
  "created_at": unix_ts,
  "in_progress_at": unix_ts|null,
  "completed_at": unix_ts|null,
  "expires_at": unix_ts|null,
  "metadata": dict|null
}
```

Formato JSONL de salida (output_file_id):
```json
{"id": "batch_req_<uuid>", "custom_id": "req1", "response": {"status_code": 200, "request_id": "...", "body": {...}}, "error": null}
```

Configuración:
- `OPENAI_FILES_DIR` — directorio de almacenamiento (default `/data/openai_files`)
- `BATCH_MAX_CONCURRENCY` — peticiones concurrentes por batch (default 4)
- `BATCH_REQUEST_TIMEOUT_SECONDS` — timeout por petición (default 600)
- `BATCH_POLL_INTERVAL_SECONDS` — intervalo de polling del procesador (default 5)

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
service:overrides           → overrides persistidos de enable/disable
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
  backend_type    TEXT,
  request_kind    TEXT,
  endpoint_path   TEXT,
  status_code     INTEGER,
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

model_load_stats (
  id              UUID PRIMARY KEY,
  model_id        TEXT NOT NULL,
  backend_type    TEXT,
  started_at      TIMESTAMPTZ NOT NULL,
  finished_at     TIMESTAMPTZ,
  duration_ms     INTEGER,
  gpu_count       INTEGER,
  gpu_indices     TEXT
)

-- Config del servidor
server_config (
  key    TEXT PRIMARY KEY,
  value  JSONB NOT NULL
)

-- OpenAI Files API
openai_files (
  id              UUID PRIMARY KEY,
  user_id         UUID NOT NULL REFERENCES users(id),
  filename        TEXT NOT NULL,
  bytes           BIGINT NOT NULL,
  purpose         TEXT NOT NULL,           -- 'batch', 'batch_output', 'user_data'
  storage_path    TEXT NOT NULL,
  status          TEXT NOT NULL DEFAULT 'uploaded',
  status_details  TEXT,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
)

-- OpenAI Batches API
openai_batches (
  id                  UUID PRIMARY KEY,
  user_id             UUID NOT NULL REFERENCES users(id),
  api_key_id          UUID REFERENCES api_keys(id),
  endpoint            TEXT NOT NULL,
  input_file_id       UUID NOT NULL REFERENCES openai_files(id),
  completion_window   TEXT NOT NULL DEFAULT '24h',
  status              TEXT NOT NULL DEFAULT 'validating',
  output_file_id      UUID REFERENCES openai_files(id),
  error_file_id       UUID REFERENCES openai_files(id),
  errors              JSONB,
  request_total       INTEGER NOT NULL DEFAULT 0,
  request_completed   INTEGER NOT NULL DEFAULT 0,
  request_failed      INTEGER NOT NULL DEFAULT 0,
  batch_metadata      JSONB,
  created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
  in_progress_at      TIMESTAMPTZ,
  expires_at          TIMESTAMPTZ,
  finalizing_at       TIMESTAMPTZ,
  completed_at        TIMESTAMPTZ,
  failed_at           TIMESTAMPTZ,
  cancelling_at       TIMESTAMPTZ,
  cancelled_at        TIMESTAMPTZ,
  expired_at          TIMESTAMPTZ
)
```

---

## 8. Model Profiles — Contrato de perfiles de modelo

### 8.1 ModelProfile (DB → ProfileRegistry → API)

```python
# backend/ocabra/db/model_config.py

class ModelProfile(Base):
    __tablename__ = "model_profiles"

    profile_id: str           # PK, slug sin /, lowercase, solo guiones y alfanuméricos
    base_model_id: str        # FK model_configs.model_id, ON DELETE CASCADE
    display_name: str
    description: str | None
    category: str             # "llm" | "tts" | "stt" | "image" | "music"
    load_overrides: dict      # JSONB, merge con extra_config del modelo base al cargar
    request_defaults: dict    # JSONB, inyectados como defaults en body de request
    assets: dict              # JSONB, {"voice_ref": "/data/profiles/{profile_id}/reference.wav", ...}
    enabled: bool             # default True
    is_default: bool          # default False, max 1 por base_model_id
    created_at: datetime
    updated_at: datetime
```

Validaciones:
- `profile_id`: regex `^[a-z0-9]+(-[a-z0-9]+)*$` (slug, no `/`, no espacios, no mayúsculas).
- `is_default`: si se marca True, cualquier otro perfil del mismo `base_model_id` con `is_default=True` se desactiva automáticamente.
- `base_model_id`: debe existir en `model_configs.model_id`. Cascada al borrar el modelo.

### 8.2 Resolución de perfiles (ProfileRegistry → OpenAI/Ollama API)

```python
# backend/ocabra/core/profile_registry.py

async def resolve_profile(profile_id: str) -> tuple[ModelProfile, ModelState]:
    """
    1. Buscar profile_id en ProfileRegistry (cache + BD)
    2. Si no existe o disabled → HTTPException 404
    3. ensure_loaded(base_model_id, load_overrides=profile.load_overrides)
    4. Retornar (profile, state)
    """

async def forward_with_profile(
    profile: ModelProfile,
    state: ModelState,
    body: dict,
) -> Any:
    """
    1. merged = {**profile.request_defaults, **body}  # body del cliente prevalece
    2. Inyectar assets (voice_ref, lora_path, etc.) en merged
    3. forward_request(state.worker_key, path, merged)
    """
```

### 8.3 Worker key (ModelManager)

```
worker_key = f"{base_model_id}:{hash(sorted(load_overrides.items()))}"
```

- `load_overrides` vacío o idéntico entre perfiles → mismo worker (compartido).
- `load_overrides` diferente → worker separado (dedicado).
- Esto permite que perfiles como `chat` y `chat-creative` (que solo difieren en `request_defaults`) compartan worker, mientras que `chat` y `chat-long` (que difieren en `load_overrides`) usen workers separados.

### 8.4 Assets de perfil (ProfileRegistry → filesystem)

```
Asset storage: /data/profiles/{profile_id}/{filename}
Upload: POST /ocabra/profiles/{profile_id}/assets (multipart)
Delete: DELETE /ocabra/profiles/{profile_id}/assets/{asset_key}
```

Seguridad:
- Path traversal: validar que la ruta resultante esté dentro de `/data/profiles/{profile_id}/`.
- `profile_id` ya está validado como slug (sin `/`, sin `..`).
- Filenames sanitizados antes de escribir al disco.

### 8.5 Backend Chatterbox

```python
# backend/ocabra/backends/chatterbox_backend.py

class ChatterboxBackend(BackendInterface):
    backend_type = "chatterbox"
    # Capabilities: tts=True, streaming=True
    # VRAM estimate: 4096 MB (Turbo) | 8192 MB (Full)
```

Worker endpoints (`backend/workers/chatterbox_worker.py`):
```
GET  /health                       → {"status": "ok"}
GET  /info                         → {"model": str, "languages": list[str], "supports_voice_clone": bool}
GET  /voices                       → {"voices": [...], "supports_voice_clone": true}
POST /synthesize                   → audio bytes (Content-Type según formato solicitado)
POST /synthesize/stream            → chunked audio (Transfer-Encoding: chunked)
```

Contratos de seguridad:
- `voice_ref`: path controlado por oCabra (asset del perfil), NO ruta libre del cliente.
- El worker recibe la ruta resuelta desde el backend, nunca un path enviado directamente por el usuario final.

### 8.6 Exposición pública de perfiles

```
/v1/models                          → solo lista perfiles habilitados (profile_id como "id")
/v1/models/{id}                     → solo acepta profile_id
/v1/chat/completions, /v1/audio/speech, etc. → model= es profile_id
/ocabra/models                      → lista modelos internos con profiles[] anidado (admin only)
```

Legacy fallback (configurable `LEGACY_MODEL_ID_FALLBACK`, default `true` en v0.6, `false` en v0.7):
- Si `model=` contiene `/` y coincide con un `model_id` canónico:
  1. Buscar perfil default del modelo.
  2. Si existe → usar ese perfil + emitir deprecation warning en logs.
  3. Si no hay perfil default → 404.
- Si `LEGACY_MODEL_ID_FALLBACK=false` → 404 directamente para IDs con `/`.

### 8.7 Endpoints REST de perfiles

Base URL: `/ocabra/`

```
GET    /ocabra/profiles                              → list[ModelProfile]
GET    /ocabra/profiles/{profile_id}                 → ModelProfile
POST   /ocabra/profiles                              → ModelProfile
  body: {
    profile_id: str,          # slug requerido
    base_model_id: str,       # FK a model_configs.model_id
    display_name: str,
    description?: str|null,
    category: "llm"|"tts"|"stt"|"image"|"music",
    load_overrides?: dict,
    request_defaults?: dict,
    enabled?: bool,
    is_default?: bool
  }
PATCH  /ocabra/profiles/{profile_id}                 → ModelProfile
  body: {
    display_name?: str,
    description?: str|null,
    category?: str,
    load_overrides?: dict,
    request_defaults?: dict,
    enabled?: bool,
    is_default?: bool
  }
DELETE /ocabra/profiles/{profile_id}                 → {"ok": true}
POST   /ocabra/profiles/{profile_id}/assets          → ModelProfile  (multipart upload)
DELETE /ocabra/profiles/{profile_id}/assets/{asset_key} → ModelProfile
```

### 8.8 Redis — Keys y canales adicionales para perfiles

```
# Canal pub/sub
profile:events              → {"event": "created"|"updated"|"deleted", "profile_id": str}

# Key (cache)
profile:state:{profile_id}  → ModelProfile (JSON, TTL: none)
```

---

## 9. Variables de entorno (contratos de configuración)

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
