# Plan operativo — Bloque 11: Resiliencia de backends y gestión avanzada de recursos

Fecha: 2026-04-07
Estado: pendiente
Origen: análisis comparativo con LocalAI (gRPC process isolation, WatchDog LRU+VRAM)
y AnythingLLM (context window lookup). Ver `docs/ROADMAP.md` Bloque 11.

## Alcance

Incluye:
- 11.1 Evicción inteligente LRU + umbral de VRAM
- 11.2 Aislamiento de backends por subproceso con auto-restart
- 11.3 Interfaz unificada multi-modal en `BackendInterface`
- 11.4 Busy timeout / health watchdog para requests colgadas

Queda fuera:
- Federación multi-nodo / clustering
- gRPC formal (usamos HTTP contra workers, no cambiamos el protocolo)
- Migración de workers existentes a un runtime diferente
- Config YAML por modelo (`models.d/`) y gallery de modelos (ideas futuras)

## Dependencias entre sub-bloques

```
11.3 (interfaz unificada)  →  se puede hacer independiente, sin dependencias
11.1 (evicción LRU+VRAM)  →  se puede hacer independiente
11.4 (busy timeout)        →  se puede hacer independiente
11.2 (aislamiento proceso) →  se beneficia de 11.4 (reutiliza watchdog), pero no bloquea
```

Orden recomendado: 11.3 → 11.1 → 11.4 → 11.2 (de menor a mayor riesgo).

## Principios de implementación

1. No romper la API pública ni los contratos existentes en `docs/CONTRACTS.md`.
2. Todos los cambios deben ser backwards-compatible con los backends actuales.
3. Los nuevos settings deben seguir el patrón de `config.py` (Pydantic `Field`, env var).
4. Async en todo. Nunca `time.sleep()`.
5. Tests unitarios para cada feature nueva. Mocks de `gpu_manager` y `worker_pool`.

---

## 11.1 — Evicción LRU + umbral de VRAM

### Contexto: qué existe hoy

- `config.py` ya tiene `vram_pressure_threshold_pct=90.0` y `vram_buffer_mb=512`.
- `model_manager.py` ya tiene `check_idle_evictions()` que evicta ON_DEMAND por idle timeout.
- `model_manager.py` tiene `_in_flight` dict y `begin_request()`/`end_request()` para tracking.
- `gpu_manager.py` tiene `get_free_vram(gpu_index)` que retorna free - locked - buffer.
- `gpu_manager.py` tiene `get_all_states()` que retorna `list[GPUState]` con VRAM por GPU.
- `ModelState` tiene `last_request_at: float | None` (timestamp del último uso).
- La evicción por presión actual NO es proactiva: se dispara solo cuando `find_gpu_for_model()`
  falla con `InsufficientVRAMError`. No hay loop de fondo que vigile VRAM.

### Qué hay que hacer

#### 1. Añadir setting `vram_eviction_threshold`

**Fichero:** `backend/ocabra/config.py`, clase `Settings`

```python
vram_eviction_threshold: float = Field(
    default=0.90,
    description="Fracción de VRAM total (0.0-1.0) a partir de la cual se evictan modelos WARM/ON_DEMAND por LRU. "
                "Diferente de vram_pressure_threshold_pct (que es % y se usa en el scheduler).",
    ge=0.0, le=1.0,
)
```

Nota: `vram_pressure_threshold_pct` ya existe (90.0, en porcentaje). El nuevo setting es
redundante intencionalmente para separar la lógica del scheduler (reactiva) de la del
watchdog (proactiva). Si se prefiere unificar, renombrar el existente y eliminar este.

#### 2. Añadir LRU tracking al `ModelManager`

**Fichero:** `backend/ocabra/core/model_manager.py`

El campo `ModelState.last_request_at` ya existe pero solo se actualiza vía Redis.
Hay que asegurar que `end_request()` actualice `_states[model_id].last_request_at = time.time()`
en memoria (no solo Redis), para que el LRU sea instantáneo.

Añadir método:

```python
def _get_eviction_candidates(self, gpu_index: int) -> list[str]:
    """Retorna model_ids cargados en gpu_index, ordenados por LRU (más antiguo primero).
    Solo incluye modelos con load_policy WARM u ON_DEMAND (nunca PIN).
    Excluye modelos con _in_flight > 0 (están sirviendo requests)."""
```

#### 3. Pre-load VRAM check con evicción preventiva

**Fichero:** `backend/ocabra/core/model_manager.py`, método `_load_model()`

Antes de llamar a `backend.load()`, insertar:

```python
# Después de obtener gpu_index y vram_estimate:
available = await self._gpu_manager.get_free_vram(gpu_index)
if vram_estimate > available:
    freed = await self._evict_for_space(gpu_index, vram_estimate - available)
    if freed < (vram_estimate - available):
        raise InsufficientVRAMError(...)
```

Nuevo método:

```python
async def _evict_for_space(self, gpu_index: int, needed_mb: int) -> int:
    """Evicta modelos LRU de gpu_index hasta liberar needed_mb.
    Retorna MB liberados. Respeta: no evicta PIN ni modelos con in_flight > 0.
    Llama a self.unload() para cada modelo evictado con reason='vram_pressure'."""
```

#### 4. Background watchdog loop

**Fichero:** `backend/ocabra/core/model_manager.py`

Nuevo método lanzado en `start()` como `asyncio.create_task`:

```python
async def _vram_watchdog(self) -> None:
    """Loop cada idle_eviction_check_interval_seconds (reusar el intervalo existente).
    Para cada GPU:
      1. Obtener GPUState vía gpu_manager.get_all_states()
      2. Calcular ratio = used_vram_mb / total_vram_mb
      3. Si ratio > settings.vram_eviction_threshold:
         - Llamar _evict_for_space(gpu_index, mb_a_liberar)
         - mb_a_liberar = used_vram_mb - (total_vram_mb * vram_eviction_threshold)
         - Log warning con modelo evictado y VRAM liberada
    """
```

#### 5. Tests

**Fichero:** `backend/tests/test_vram_eviction.py` (nuevo)

Tests necesarios:
- `test_eviction_candidates_excludes_pin` — modelos PIN nunca aparecen como candidatos
- `test_eviction_candidates_excludes_busy` — modelos con in_flight > 0 no se evictan
- `test_eviction_order_is_lru` — el modelo con last_request_at más antiguo se evicta primero
- `test_preload_evicts_when_insufficient_vram` — al cargar un modelo, se evictan WARM si no hay espacio
- `test_preload_fails_if_cannot_free_enough` — si todo es PIN, raise InsufficientVRAMError
- `test_watchdog_evicts_above_threshold` — simular VRAM al 95%, verificar que se evicta
- `test_watchdog_does_nothing_below_threshold` — simular VRAM al 80%, verificar que no pasa nada

Mock de `gpu_manager` con `GPUState` fijos. Mock de backends con `unload()` que no hace nada.

---

## 11.2 — Aislamiento por proceso con auto-restart

### Contexto: qué existe hoy

- Los backends de IA (vLLM, Diffusers, Whisper, TTS, etc.) ya se ejecutan como subprocesos
  FastAPI lanzados desde cada `*_backend.py` vía `subprocess.Popen` o `asyncio.create_subprocess_exec`.
- `WorkerPool._workers` trackea `WorkerInfo(pid, port, gpu_indices, vram_used_mb)`.
- `BackendInterface.health_check(model_id)` existe pero cada backend lo implementa a su manera
  (la mayoría hace `GET /health` al worker).
- Si un worker muere (OOM, segfault), el modelo queda en estado LOADED pero el port está muerto.
  No hay detección automática ni restart.
- `ServiceManager` tiene health checks para servicios externos (Hunyuan, ComfyUI) pero NO
  para los workers internos de modelos.

### Qué hay que hacer

#### 1. `BackendProcessManager` — health monitor para workers

**Fichero nuevo:** `backend/ocabra/core/backend_process_manager.py`

```python
class BackendProcessManager:
    """Monitoriza workers activos y reinicia los que mueren.

    No reemplaza la lógica de launch de cada backend.
    Se limita a:
    - Polling periódico de salud de cada worker registrado
    - Detección de proceso muerto (pid check + HTTP health)
    - Transición del modelo a ERROR si el worker muere
    - Auto-restart opcional con backoff exponencial
    """

    def __init__(self, model_manager, worker_pool, settings):
        self._model_manager = model_manager
        self._worker_pool = worker_pool
        self._settings = settings
        self._restart_counts: dict[str, int] = {}  # model_id -> consecutive restarts
        self._max_restarts: int = 3
        self._backoff_base_seconds: float = 5.0

    async def start(self) -> None:
        """Lanza el loop de health check como asyncio.Task."""

    async def stop(self) -> None:
        """Cancela el loop."""

    async def _health_loop(self) -> None:
        """Cada worker_health_check_interval_seconds (nuevo setting, default 10s):
        Para cada (model_id, worker_info) en worker_pool._workers:
          1. Comprobar si el PID sigue vivo: os.kill(pid, 0) o /proc/{pid}/status
          2. Si PID muerto → _handle_worker_death(model_id)
          3. Si PID vivo → HTTP GET http://127.0.0.1:{port}/health con timeout 5s
          4. Si health falla → incrementar contador de fallos consecutivos
          5. Si fallos consecutivos >= 3 → _handle_worker_death(model_id)
          6. Si health OK → resetear contador de fallos
        """

    async def _handle_worker_death(self, model_id: str) -> None:
        """1. Log error con model_id, PID, razón
        2. Marcar modelo como ERROR vía model_manager
        3. Limpiar worker de worker_pool
        4. Liberar VRAM lock en gpu_manager
        5. Emitir system_alert via WebSocket
        6. Si auto_restart_workers=True (setting) y _restart_counts[model_id] < _max_restarts:
             - await asyncio.sleep(backoff_base * 2**restart_count)
             - Intentar model_manager.ensure_loaded(model_id)
             - Incrementar _restart_counts[model_id]
           Si no:
             - Log que se ha alcanzado el límite de restarts, requiere intervención manual
        """
```

#### 2. Settings nuevos

**Fichero:** `backend/ocabra/config.py`

```python
worker_health_check_interval_seconds: int = Field(default=10, ge=1)
auto_restart_workers: bool = Field(default=True)
max_worker_restarts: int = Field(default=3, ge=0, description="Restarts consecutivos antes de rendirse")
worker_restart_backoff_seconds: float = Field(default=5.0, ge=1.0)
```

#### 3. Integración en `main.py`

**Fichero:** `backend/ocabra/main.py`, en el bloque de startup:

```python
backend_process_manager = BackendProcessManager(model_manager, worker_pool, settings)
await backend_process_manager.start()
# En shutdown:
await backend_process_manager.stop()
```

#### 4. Tests

**Fichero:** `backend/tests/test_backend_process_manager.py` (nuevo)

Tests necesarios:
- `test_detects_dead_pid` — mock PID que no existe, verifica transición a ERROR
- `test_detects_health_failure` — mock HTTP timeout 3 veces, verifica transición a ERROR
- `test_auto_restart_on_death` — mock muerte + re-load exitoso, verifica modelo vuelve a LOADED
- `test_max_restarts_exceeded` — 4 muertes consecutivas, verifica que NO intenta restart #4
- `test_backoff_increases` — verificar que sleep entre restarts crece exponencialmente
- `test_healthy_worker_resets_failure_count` — fallo → OK → fallo cuenta desde 0
- `test_vram_freed_on_death` — verificar que gpu_manager.unlock_vram se llama

---

## 11.3 — Interfaz unificada multi-modal en BackendInterface

### Contexto: qué existe hoy

- `BackendInterface` en `backends/base.py` define 7 métodos abstractos.
- `BackendCapabilities` ya es un dataclass con 19 flags booleanos:
  `chat`, `completions`, `embeddings`, `vision`, `function_calling`, `json_mode`,
  `streaming`, `images`, `audio_transcription`, `audio_speech`, `audio_voices`,
  `audio_speech_stream`, `pooling`, `rerank`, más `context_length`, `model_name`, etc.
- `get_capabilities()` ya existe y cada backend lo implementa.
- El routing actual en las APIs (ej: `api/openai/chat.py`) resuelve el backend por tipo
  hardcodeado (vía `model_config.backend_type`), NO por capabilities.

### Qué hay que hacer

#### 1. Añadir métodos opcionales tipados a `BackendInterface`

**Fichero:** `backend/ocabra/backends/base.py`

NO hacer abstractos — defaults que lanzan `NotImplementedError` con mensaje claro:

```python
class BackendInterface(ABC):
    # ... métodos abstractos existentes (load, unload, health_check, etc.) ...

    # --- Métodos por modalidad (opcionales, override solo si el backend soporta) ---

    async def generate_text(self, model_id: str, messages: list[dict], **kwargs) -> dict:
        """Chat/completions. Override en backends de texto (vllm, llama_cpp, sglang, ollama)."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support text generation")

    async def stream_text(self, model_id: str, messages: list[dict], **kwargs) -> AsyncIterator[bytes]:
        """Streaming text generation."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support text streaming")

    async def generate_embeddings(self, model_id: str, input: list[str], **kwargs) -> dict:
        """Embeddings. Override en backends con soporte (vllm, ollama)."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support embeddings")

    async def generate_image(self, model_id: str, prompt: str, **kwargs) -> bytes:
        """Image generation. Override en diffusers, acestep."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support image generation")

    async def transcribe(self, model_id: str, audio: bytes, **kwargs) -> dict:
        """STT. Override en whisper."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support transcription")

    async def synthesize_speech(self, model_id: str, text: str, **kwargs) -> bytes:
        """TTS. Override en tts, chatterbox, voxtral."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support speech synthesis")

    async def rerank(self, model_id: str, query: str, documents: list[str], **kwargs) -> dict:
        """Reranking. Override en backends con soporte."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support reranking")
```

**IMPORTANTE:** Estos métodos son la API futura. En esta fase NO se migran los backends
a usarlos. Los backends siguen usando `forward_request()`/`forward_stream()` como hoy.
La migración gradual es trabajo futuro. Lo que sí se hace es:

#### 2. Añadir `ModalityType` enum y método de clase `supported_modalities()`

**Fichero:** `backend/ocabra/backends/base.py`

```python
class ModalityType(str, Enum):
    TEXT_GENERATION = "text_generation"
    EMBEDDINGS = "embeddings"
    IMAGE_GENERATION = "image_generation"
    AUDIO_TRANSCRIPTION = "audio_transcription"
    AUDIO_SPEECH = "audio_speech"
    RERANKING = "reranking"

class BackendInterface(ABC):
    @classmethod
    def supported_modalities(cls) -> set[ModalityType]:
        """Cada backend override esto declarando qué modalidades soporta.
        Se usa en routing y validación. Default: set vacío."""
        return set()
```

#### 3. Implementar `supported_modalities()` en cada backend

**Ficheros:** Cada `backend/ocabra/backends/*_backend.py`

Ejemplo para `vllm_backend.py`:
```python
@classmethod
def supported_modalities(cls) -> set[ModalityType]:
    return {ModalityType.TEXT_GENERATION, ModalityType.EMBEDDINGS}
```

Ejemplo para `whisper_backend.py`:
```python
@classmethod
def supported_modalities(cls) -> set[ModalityType]:
    return {ModalityType.AUDIO_TRANSCRIPTION}
```

Lista completa de mappings:
| Backend | Modalities |
|---------|-----------|
| vllm | TEXT_GENERATION, EMBEDDINGS |
| llama_cpp | TEXT_GENERATION, EMBEDDINGS |
| sglang | TEXT_GENERATION, EMBEDDINGS |
| ollama | TEXT_GENERATION, EMBEDDINGS |
| bitnet | TEXT_GENERATION |
| diffusers | IMAGE_GENERATION |
| acestep | IMAGE_GENERATION (audio, pero se expone como imagen/media) |
| whisper | AUDIO_TRANSCRIPTION |
| tts | AUDIO_SPEECH |
| chatterbox | AUDIO_SPEECH |
| voxtral | AUDIO_SPEECH |

#### 4. Helper de routing por capability en `WorkerPool`

**Fichero:** `backend/ocabra/core/worker_pool.py`

```python
def get_backends_for_modality(self, modality: ModalityType) -> list[str]:
    """Retorna nombres de backends registrados que soportan la modalidad dada.
    Excluye backends disabled."""
    return [
        name for name, backend in self._backends.items()
        if name not in self._disabled_backends
        and modality in backend.supported_modalities()
    ]

def supports_modality(self, backend_name: str, modality: ModalityType) -> bool:
    """Check rápido para validación en API layers."""
    backend = self._backends.get(backend_name)
    return backend is not None and modality in backend.supported_modalities()
```

**NO cambiar el routing actual de las APIs en esta fase.** Solo exponer los helpers.
La migración del routing hardcodeado a routing por capability es trabajo futuro.

#### 5. Tests

**Fichero:** `backend/tests/test_backend_modalities.py` (nuevo)

- `test_all_backends_declare_modalities` — cada backend en `_backends` tiene `supported_modalities()` no vacío
- `test_vllm_supports_text_and_embeddings` — check específico
- `test_whisper_supports_only_transcription` — check específico
- `test_get_backends_for_modality_text` — retorna vllm, llama_cpp, sglang, ollama
- `test_get_backends_for_modality_excludes_disabled` — disabled no aparece
- `test_unsupported_modality_method_raises` — llamar `generate_image()` en vllm → NotImplementedError

---

## 11.4 — Busy timeout / health watchdog

### Contexto: qué existe hoy

- `model_manager._in_flight: dict[str, int]` cuenta requests activas por modelo.
- `begin_request(model_id)` / `end_request(model_id)` se llaman desde las capas API.
- `worker_pool.forward_request()` usa `httpx` con timeout 300s hardcodeado.
- No hay tracking de cuándo empezó cada request individual (solo el count).
- No hay acción si una request excede el timeout — httpx simplemente lanza `ReadTimeout`.

### Qué hay que hacer

#### 1. Request tracking con timestamps

**Fichero:** `backend/ocabra/core/model_manager.py`

Cambiar `_in_flight: dict[str, int]` a un tracking más rico:

```python
@dataclass
class ActiveRequest:
    request_id: str        # uuid4
    model_id: str
    started_at: float      # time.time()

# En ModelManager.__init__:
self._active_requests: dict[str, ActiveRequest] = {}  # request_id -> ActiveRequest

def begin_request(self, model_id: str) -> str:
    """Retorna request_id. El caller debe pasarlo a end_request()."""
    request_id = str(uuid.uuid4())
    self._active_requests[request_id] = ActiveRequest(
        request_id=request_id,
        model_id=model_id,
        started_at=time.time(),
    )
    # Mantener compatibilidad: seguir actualizando _in_flight count
    self._in_flight[model_id] = self._in_flight.get(model_id, 0) + 1
    return request_id

def end_request(self, model_id: str, request_id: str | None = None) -> None:
    """Si request_id es None, compatibilidad legacy (solo decrementa count)."""
    if request_id and request_id in self._active_requests:
        del self._active_requests[request_id]
    self._in_flight[model_id] = max(0, self._in_flight.get(model_id, 0) - 1)
```

**IMPORTANTE:** Los callers actuales de `begin_request()`/`end_request()` están en los
endpoints de las APIs (`api/openai/chat.py`, `api/ollama/generate.py`, etc.). Hay que
actualizar cada caller para capturar el `request_id` retornado y pasarlo a `end_request()`.
Buscar todos los usos con: `grep -rn "begin_request\|end_request" backend/ocabra/api/`

#### 2. Settings

**Fichero:** `backend/ocabra/config.py`

```python
busy_timeout_seconds: int = Field(
    default=300,
    ge=30,
    description="Timeout máximo para una request individual. Si se supera, el modelo se marca ERROR.",
)
busy_timeout_action: str = Field(
    default="mark_error",
    description="Acción al exceder busy_timeout: 'mark_error' (marca ERROR, no mata worker) "
                "o 'restart_worker' (marca ERROR y reinicia worker). "
                "'restart_worker' requiere auto_restart_workers=True (11.2).",
)
```

#### 3. Busy watchdog loop

**Fichero:** `backend/ocabra/core/model_manager.py`

Nuevo método lanzado en `start()` como `asyncio.create_task`:

```python
async def _busy_watchdog(self) -> None:
    """Loop cada 10 segundos.
    Para cada ActiveRequest en _active_requests.values():
      elapsed = time.time() - request.started_at
      if elapsed > settings.busy_timeout_seconds:
        1. Log error: f"Request {request.request_id} for {request.model_id} exceeded busy timeout ({elapsed:.0f}s)"
        2. Eliminar de _active_requests
        3. Decrementar _in_flight
        4. Si busy_timeout_action == 'mark_error':
             - Transicionar modelo a ERROR
             - Emitir system_alert via WebSocket
        5. Si busy_timeout_action == 'restart_worker':
             - Llamar self.unload(model_id, reason='busy_timeout')
             - El BackendProcessManager (11.2) se encargará del restart si está activo
    """
```

#### 4. Métricas

**Fichero:** `backend/ocabra/core/model_manager.py`

Añadir contador simple:

```python
# En __init__:
self._timeout_counts: dict[str, int] = {}  # model_id -> count

# En _busy_watchdog cuando detecta timeout:
self._timeout_counts[model_id] = self._timeout_counts.get(model_id, 0) + 1
```

Exponer en el endpoint de stats existente (`api/internal/stats.py` o similar).

#### 5. Actualizar callers de begin_request/end_request

**Ficheros a modificar** (buscar con grep):
- `backend/ocabra/api/openai/chat.py`
- `backend/ocabra/api/openai/completions.py`
- `backend/ocabra/api/openai/embeddings.py`
- `backend/ocabra/api/openai/images.py`
- `backend/ocabra/api/openai/audio.py`
- `backend/ocabra/api/openai/pooling.py`
- `backend/ocabra/api/ollama/chat.py`
- `backend/ocabra/api/ollama/generate.py`
- `backend/ocabra/api/ollama/embeddings.py`

Patrón de cambio en cada endpoint:

```python
# Antes:
model_manager.begin_request(model_id)
try:
    result = await worker_pool.forward_request(model_id, path, body)
finally:
    model_manager.end_request(model_id)

# Después:
request_id = model_manager.begin_request(model_id)
try:
    result = await worker_pool.forward_request(model_id, path, body)
finally:
    model_manager.end_request(model_id, request_id)
```

La firma antigua `end_request(model_id)` sigue funcionando (request_id=None) para
compatibilidad, pero todos los callers deben actualizarse.

#### 6. Tests

**Fichero:** `backend/tests/test_busy_timeout.py` (nuevo)

- `test_begin_request_returns_id` — verifica que retorna string UUID
- `test_end_request_clears_active` — request_id desaparece de _active_requests
- `test_end_request_legacy_compat` — sin request_id, solo decrementa count
- `test_watchdog_detects_timeout` — request con started_at hace 400s, busy_timeout=300, verifica ERROR
- `test_watchdog_ignores_fresh_requests` — request con 10s, no pasa nada
- `test_timeout_count_increments` — verificar _timeout_counts
- `test_mark_error_action` — busy_timeout_action='mark_error', modelo transiciona a ERROR
- `test_restart_worker_action` — busy_timeout_action='restart_worker', se llama unload()

---

## Checklist de ficheros afectados (resumen)

| Fichero | Sub-bloques | Tipo de cambio |
|---------|-------------|----------------|
| `backend/ocabra/config.py` | 11.1, 11.2, 11.4 | Añadir settings |
| `backend/ocabra/core/model_manager.py` | 11.1, 11.4 | Evicción LRU, busy watchdog, ActiveRequest |
| `backend/ocabra/backends/base.py` | 11.3 | ModalityType, métodos opcionales, supported_modalities() |
| `backend/ocabra/backends/*_backend.py` (todos) | 11.3 | Implementar supported_modalities() |
| `backend/ocabra/core/worker_pool.py` | 11.3 | Helpers de routing por capability |
| `backend/ocabra/core/backend_process_manager.py` | 11.2 | **NUEVO** — health monitor + auto-restart |
| `backend/ocabra/main.py` | 11.2 | Instanciar y arrancar BackendProcessManager |
| `backend/ocabra/api/openai/*.py` | 11.4 | Actualizar begin/end_request con request_id |
| `backend/ocabra/api/ollama/*.py` | 11.4 | Actualizar begin/end_request con request_id |
| `backend/tests/test_vram_eviction.py` | 11.1 | **NUEVO** |
| `backend/tests/test_backend_process_manager.py` | 11.2 | **NUEVO** |
| `backend/tests/test_backend_modalities.py` | 11.3 | **NUEVO** |
| `backend/tests/test_busy_timeout.py` | 11.4 | **NUEVO** |
