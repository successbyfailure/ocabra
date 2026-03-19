# Plan de implementación: BitNet backend

**Fecha:** 2026-03-19
**Objetivo:** Añadir soporte para modelos Microsoft BitNet (1.58-bit / ternarios) con máximo rendimiento usando `bitnet.cpp` (`llama-server`).

---

## Contexto y decisiones de arquitectura

### ¿Por qué bitnet.cpp?

BitNet no puede ejecutarse de forma eficiente vía HuggingFace Transformers ni vLLM — ambos carecen de los kernels ternarios especializados. El único runtime con rendimiento real es **bitnet.cpp** (fork de llama.cpp con kernels SIMD/AVX2/NEON para operaciones ternarias).

### Patrón de integración

`BitnetBackend` sigue **exactamente el mismo patrón** que `VLLMBackend`:

- Subproceso por modelo que expone `llama-server` en un puerto local
- API OpenAI-compatible en `/v1/chat/completions`, `/v1/completions`, `/v1/models`
- Proxy httpx desde oCabra hacia el worker
- Health check en `GET /health`
- SIGTERM → SIGKILL para shutdown

No se usa `ServiceManager` (ese patrón es para servicios Docker externos como ComfyUI/A1111). BitNet son subprocesos directos del contenedor `api`, como vLLM.

### Rendimiento en este hardware (RTX 3060 12GB + RTX 3090 24GB)

| Configuración | RAM | VRAM consumida | Throughput approx. |
|---|---|---|---|
| CPU-only (`--ngl 0`) | ~400 MB RAM | 0 MB | 5-7 tok/s |
| GPU offload parcial (`--ngl 16`) | ~200 MB RAM | ~200 MB VRAM | 10-15 tok/s |
| GPU offload total (`--ngl 99`) | ~100 MB RAM | ~400 MB VRAM | 20-30 tok/s |

**Recomendación**: CPU-only por defecto (`--ngl 0`). El modelo BitNet 2B es tan pequeño que no merece ocupar VRAM de las GPUs (3060/3090 con 12/24 GB) que se necesitan para vLLM. El usuario puede activar GPU offload por modelo vía `extra_config`.

### Formato de modelo: GGUF i2_s

Los modelos BitNet se distribuyen en formato GGUF con cuantización `i2_s` (ternary). El fichero de referencia es:

```
microsoft/bitnet-b1.58-2B-4T-gguf → ggml-model-i2_s.gguf
```

La detección se basa en:
1. **Nombre de fichero**: contiene `i2_s` o `bitnet`
2. **Metadata GGUF**: `general.architecture == "bitnet"` (en los primeros bytes del header)

---

## Ficheros a crear / modificar

### CREAR

| Fichero | Descripción |
|---|---|
| `backend/ocabra/backends/bitnet_backend.py` | Implementación completa de `BitnetBackend` |
| `backend/scripts/build_bitnet.sh` | Script de compilación de bitnet.cpp (usado en Dockerfile) |

### MODIFICAR

| Fichero | Cambio |
|---|---|
| `backend/Dockerfile` | Añadir stage builder para compilar bitnet.cpp y copiar binario |
| `backend/ocabra/config.py` | Añadir settings `bitnet_*` |
| `backend/ocabra/registry/local_scanner.py` | Detectar GGUF i2_s → `backend_type="bitnet"` |
| `backend/ocabra/backends/__init__.py` | Importar/registrar `BitnetBackend` |
| `docker-compose.yml` | Añadir variables de entorno `BITNET_*` |
| `.env.example` | Documentar las nuevas variables |
| `docs/CONTRACTS.md` | Añadir `"bitnet"` a los valores válidos de `backend_type` |

---

## Paso 1 — Dockerfile: compilar bitnet.cpp (multi-stage)

**Fichero:** `backend/Dockerfile`

Añadir un stage `bitnet-builder` antes del stage `base`. Este stage:
1. Clona `https://github.com/microsoft/BitNet` (tag estable o commit fijo)
2. Instala cmake + build tools
3. Compila el target `llama-server` con soporte CUDA y AVX2
4. El binario resultante (`llama-server`) se copia al stage final en `/usr/local/bin/bitnet-server`

```dockerfile
# ── Stage: bitnet-builder ──────────────────────────────────────────────────
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS bitnet-builder

ARG BITNET_COMMIT=main

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias Python de bitnet.cpp (setup_env.py)
RUN pip3 install --no-cache-dir huggingface-hub numpy

WORKDIR /build
RUN git clone --depth 1 --branch ${BITNET_COMMIT} \
    https://github.com/microsoft/BitNet.git .

# Compilar llama-server con CUDA + AVX2 para máximo rendimiento
RUN cmake -B build \
    -DGGML_CUDA=ON \
    -DGGML_AVX2=ON \
    -DGGML_F16C=ON \
    -DGGML_FMA=ON \
    -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --target llama-server -j$(nproc)
```

En el stage `base` / `production`:
```dockerfile
COPY --from=bitnet-builder /build/build/bin/llama-server /usr/local/bin/bitnet-server
RUN chmod +x /usr/local/bin/bitnet-server
```

**Flags de compilación explicados:**
- `GGML_CUDA=ON` — habilita kernels CUDA para GPU offload opcional
- `GGML_AVX2=ON` — instrucciones vectoriales AVX2 para CPU (x86_64)
- `GGML_F16C=ON` + `GGML_FMA=ON` — FP16 conversion y FMA para mayor throughput en CPU

---

## Paso 2 — `BitnetBackend` (nuevo fichero)

**Fichero:** `backend/ocabra/backends/bitnet_backend.py`

```
Responsabilidades:
- load(): lanzar bitnet-server como subproceso asyncio
- unload(): SIGTERM → SIGKILL
- health_check(): GET http://127.0.0.1:{port}/health
- get_capabilities(): chat=True, completion=True, streaming=True, context_length configurable
- get_vram_estimate_mb(): 0 si CPU-only, estimación proporcional si --ngl > 0
- forward_request(): proxy httpx POST no-streaming
- forward_stream(): proxy httpx streaming SSE
```

### Comando de lanzamiento

```
bitnet-server \
  --model <ruta_al_gguf> \
  --host 127.0.0.1 \
  --port <puerto> \
  --ctx-size <bitnet_ctx_size>   # default: 4096
  --threads <bitnet_threads>     # default: $(nproc)
  --batch-size <bitnet_batch_size>  # default: 512
  -ngl <bitnet_gpu_layers>       # default: 0 (CPU-only)
  --parallel <bitnet_parallel>   # default: 1
  [--flash-attn]                 # si bitnet_flash_attn=True
  [--mlock]                      # si bitnet_mlock=True — fijar RAM, evitar swap
  --served-model-name <model_id>
```

### Variables de entorno del subproceso

```python
env = {
    **os.environ,
    "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in gpu_indices) if gpu_layers > 0 else "",
    "CUDA_DEVICE_ORDER": settings.cuda_device_order,
}
```

Si `gpu_layers == 0` (CPU-only), pasar `CUDA_VISIBLE_DEVICES=""` para que el proceso no inicialice CUDA en absoluto — más rápido en arranque.

### Registro de proceso

```python
self._processes: dict[str, tuple[asyncio.subprocess.Process, int]] = {}
```

Idéntico a `VLLMBackend`.

### Timeout de arranque

`llama-server` arranca mucho más rápido que vLLM (segundos vs minutos). Usar `_STARTUP_TIMEOUT_S = 30`.

### VRAM estimate

```python
async def get_vram_estimate_mb(self, model_id: str) -> int:
    gpu_layers = self._get_gpu_layers(model_id)
    if gpu_layers == 0:
        return 0  # CPU-only, no consume VRAM
    # 0.4 GB total para 2B4T, proporcional a capas offloaded
    # bitnet-b1.58-2B-4T tiene 32 transformer layers
    total_model_mb = 400
    total_layers = 32
    return int(total_model_mb * min(gpu_layers, total_layers) / total_layers)
```

### Capabilities

BitNet solo hace chat y completion (no embeddings, no vision, no tools, no reasoning):

```python
BackendCapabilities(
    chat=True,
    completion=True,
    tools=False,
    vision=False,
    embeddings=False,
    streaming=True,
    context_length=ctx_size,  # del extra_config o settings.bitnet_ctx_size
)
```

### Manejo de extra_config

Los parámetros por-modelo se leen de `extra_config["bitnet"]` (mismo patrón que `_get_vllm_option`):

```python
def _get_bitnet_option(self, extra_config: dict, key: str, default: Any) -> Any:
    bitnet_cfg = extra_config.get("bitnet")
    if isinstance(bitnet_cfg, dict) and key in bitnet_cfg:
        return bitnet_cfg[key]
    return extra_config.get(key, default)
```

---

## Paso 3 — Settings (`config.py`)

**Fichero:** `backend/ocabra/config.py`

Añadir en la clase `Settings`:

```python
# BitNet (bitnet.cpp llama-server)
bitnet_server_bin: str = "/usr/local/bin/bitnet-server"  # ruta al binario compilado
bitnet_gpu_layers: int = 0           # 0 = CPU-only (máxima eficiencia BitNet)
bitnet_ctx_size: int = 4096          # context window
bitnet_threads: int | None = None    # None = auto (usa nproc)
bitnet_batch_size: int = 512         # batch de procesamiento de prompt
bitnet_ubatch_size: int = 128        # microbatch de generación
bitnet_parallel: int = 1             # slots paralelos (1 = mayor eficiencia CPU)
bitnet_flash_attn: bool = False      # Flash Attention (experimental en llama-server)
bitnet_mlock: bool = True            # Fijar pesos en RAM, evitar swap
bitnet_startup_timeout_s: int = 30   # Timeout de arranque (mucho más rápido que vLLM)
```

Añadir validadores `_empty_string_to_none` para `bitnet_threads`.

---

## Paso 4 — LocalScanner: detección de GGUF BitNet

**Fichero:** `backend/ocabra/registry/local_scanner.py`

El scanner actual asigna `backend_type="vllm"` a todos los GGUF. Necesita distinguir:

### Estrategia de detección

**Nivel 1 — Heurística por nombre de fichero** (rápido, sin leer el fichero):
- Nombre contiene `i2_s` → BitNet
- Nombre contiene `bitnet` (case-insensitive) → BitNet

**Nivel 2 — GGUF header metadata** (definitivo, lee los primeros KB del fichero):

El formato GGUF tiene un header con magic `GGUF`, versión, y metadata key-value. La clave `general.architecture` = `"bitnet"` identifica inequívocamente los modelos BitNet.

```python
def _is_bitnet_gguf(self, path: Path) -> bool:
    """Detect BitNet GGUF by filename heuristic or GGUF header metadata."""
    # Level 1: filename heuristic (fast, no file read)
    stem_lower = path.stem.lower()
    if "i2_s" in stem_lower or "bitnet" in stem_lower:
        return True
    # Level 2: read GGUF general.architecture from header
    return self._read_gguf_architecture(path) == "bitnet"

def _read_gguf_architecture(self, path: Path) -> str | None:
    """
    Read general.architecture from GGUF header metadata.

    GGUF format (v3):
      - magic: 4 bytes (b"GGUF")
      - version: uint32
      - tensor_count: uint64
      - metadata_kv_count: uint64
      - metadata_kv pairs: key_len(uint64) + key(bytes) + value_type(uint32) + value
    """
    GGUF_MAGIC = b"GGUF"
    VALUE_TYPE_STRING = 8

    try:
        with path.open("rb") as f:
            if f.read(4) != GGUF_MAGIC:
                return None
            version = int.from_bytes(f.read(4), "little")
            if version not in (2, 3):
                return None
            _tensor_count = int.from_bytes(f.read(8), "little")
            kv_count = int.from_bytes(f.read(8), "little")

            for _ in range(min(kv_count, 64)):  # scan first 64 keys only
                key_len = int.from_bytes(f.read(8), "little")
                if key_len > 256:
                    return None
                key = f.read(key_len).decode("utf-8", errors="replace")
                val_type = int.from_bytes(f.read(4), "little")

                if val_type == VALUE_TYPE_STRING:
                    str_len = int.from_bytes(f.read(8), "little")
                    if str_len > 1024:
                        return None
                    value = f.read(str_len).decode("utf-8", errors="replace")
                    if key == "general.architecture":
                        return value
                else:
                    # Skip non-string values (variable width — stop scanning)
                    # Cannot reliably skip without full type table; bail out
                    break
    except Exception:
        return None
    return None
```

Cambio en `_scan_sync`:
```python
if path.is_file() and path.suffix.lower() == ".gguf":
    is_bitnet = self._is_bitnet_gguf(path)
    models.append(
        LocalModel(
            model_ref=path.stem,
            path=str(path),
            source="gguf",
            backend_type="bitnet" if is_bitnet else "vllm",
            size_gb=path.stat().st_size / (1024**3),
        )
    )
```

---

## Paso 5 — Registro del backend

**Fichero:** `backend/ocabra/backends/__init__.py`
(y dondequiera que se registren los backends en `worker_pool.py` / `model_manager.py`)

Registrar `BitnetBackend` bajo la clave `"bitnet"` en el mismo lugar donde se registra `VLLMBackend`.

---

## Paso 6 — Variables de entorno

**Fichero:** `docker-compose.yml` — añadir en el servicio `api`:

```yaml
BITNET_GPU_LAYERS: ${BITNET_GPU_LAYERS:-0}
BITNET_CTX_SIZE: ${BITNET_CTX_SIZE:-4096}
BITNET_THREADS: ${BITNET_THREADS:-}
BITNET_BATCH_SIZE: ${BITNET_BATCH_SIZE:-512}
BITNET_PARALLEL: ${BITNET_PARALLEL:-1}
BITNET_MLOCK: ${BITNET_MLOCK:-true}
```

**Fichero:** `.env.example`:

```bash
# BitNet (bitnet.cpp llama-server)
# BITNET_GPU_LAYERS=0        # 0 = CPU-only (máx eficiencia). Subir para GPU offload parcial.
# BITNET_CTX_SIZE=4096       # Context window (bitnet-b1.58-2B-4T soporta hasta 4096)
# BITNET_THREADS=            # Auto (usa todos los cores). Ejemplo: 16
# BITNET_BATCH_SIZE=512      # Batch de procesamiento de prompt
# BITNET_PARALLEL=1          # Slots paralelos. Aumentar con CPU potente.
# BITNET_MLOCK=true          # Fijar pesos en RAM para evitar swap
```

---

## Paso 7 — Actualizar CONTRACTS.md

En la sección `WorkerInfo` y `ModelState.backend_type`, añadir `"bitnet"` a los valores válidos:

```
backend_type: str  # "vllm" | "diffusers" | "whisper" | "tts" | "ollama" | "bitnet"
```

---

## Paso 8 — Frontend: badge BitNet

**Fichero:** `frontend/src/components/models/ModelBadges.tsx`

Añadir un badge visual para modelos BitNet. Sugerencia de estilo: badge amarillo/dorado con texto `1.58-bit` o `BitNet`.

---

## Orden de implementación

```
Paso 1  Dockerfile multi-stage (bitnet-builder stage)
   │
   ├── Paso 2  BitnetBackend (bitnet_backend.py)
   ├── Paso 3  Settings (config.py)
   │
   ├── Paso 4  LocalScanner (detección i2_s)
   ├── Paso 5  Registro del backend (__init__.py / worker_pool)
   │
   ├── Paso 6  Docker env vars (docker-compose.yml + .env.example)
   ├── Paso 7  CONTRACTS.md
   └── Paso 8  Frontend badge
```

Pasos 2-8 son independientes entre sí y pueden implementarse en paralelo una vez que el Dockerfile compile correctamente (Paso 1).

---

## Notas de rendimiento avanzadas

### Thread tuning para CPU inference

En servidores multi-core, `--threads` debe igualar los **P-cores físicos** (sin HT):
- Si el servidor tiene 16 cores / 32 threads: `BITNET_THREADS=16`
- Más threads no ayuda por la naturaleza secuencial de la generación autoregresiva

### mlock = imprescindible en producción

`--mlock` fija los pesos en RAM física. Sin él, el sistema puede hacer swap del modelo durante picos de actividad, añadiendo latencias de 100-1000x. Activado por defecto.

### Parallel slots en CPU

`--parallel 1` maximiza la velocidad de generación por petición. Si se necesita throughput concurrente (múltiples usuarios simultáneos), subir a 2-4, pero la velocidad por petición cae proporcionalmente.

### GPU offload `--ngl`

BitNet 2B tiene 32 transformer layers. Con `--ngl 32` (full GPU), se espera ~20-30 tok/s. Con GPU offload parcial, el cuello de botella es la transferencia CPU↔GPU por capa. Recomendado: o CPU-only (0) o full GPU (32), no valores intermedios.

---

## Modelos compatibles

| Modelo | GGUF | Tamaño | Contexto | Notas |
|---|---|---|---|---|
| `microsoft/bitnet-b1.58-2B-4T-gguf` | `ggml-model-i2_s.gguf` | ~0.4 GB | 4096 | Modelo principal |
| Futuras variantes BitNet | `*i2_s*.gguf` | Variable | Variable | Detección automática |

Los modelos se colocan en `$MODELS_DIR/bitnet/` o en cualquier subdirectorio de `$MODELS_DIR` — el scanner los detectará por el header GGUF.

---

## Tests necesarios

- `tests/backends/test_bitnet_backend.py`:
  - `test_load_spawns_process` — mock de `asyncio.create_subprocess_exec`
  - `test_unload_sends_sigterm`
  - `test_health_check_returns_true_on_200`
  - `test_cpu_only_empty_cuda_visible_devices`
  - `test_gpu_offload_sets_cuda_devices`
  - `test_vram_estimate_zero_cpu_only`
  - `test_vram_estimate_proportional_gpu_layers`

- `tests/registry/test_local_scanner.py`:
  - `test_i2s_gguf_detected_as_bitnet` — fichero con `i2_s` en nombre
  - `test_gguf_header_bitnet_architecture` — fichero con header GGUF válido
  - `test_regular_gguf_detected_as_vllm` — GGUF sin indicadores BitNet

---

## Preguntas abiertas

1. **Versión de bitnet.cpp a fijar**: ¿usar `main` o anclar a un commit? Recomendado anclar a un commit estable para builds reproducibles.
2. **Soporte para futuros modelos BitNet más grandes** (7B, 70B): `get_vram_estimate_mb` debería leer el número real de capas del header GGUF en lugar de asumir 32.
3. **¿Añadir endpoint `/ocabra/registry/local?backend=bitnet`** para filtrar por tipo en la UI? Ya funciona con el `backend_type` en `LocalModel`.

---

## Mejoras de concurrencia (aplicables a todos los backends)

Identificadas al revisar cómo funciona la concurrencia actual en oCabra. Se implementan en el mismo stream que BitNet o en un stream separado.

### Diagnóstico actual

| Aspecto | Estado |
|---|---|
| vLLM acepta peticiones concurrentes | ✅ (continuous batching, `max_num_seqs=16`) |
| FastAPI + httpx proxy es async | ✅ (no bloquea entre peticiones) |
| Race condition en carga bajo demanda | ✅ ya resuelto — `ModelManager._load_model` usa `asyncio.Lock` por `model_id` con early-return si `status == LOADED` |
| Connection pooling en httpx | ❌ nuevo cliente TCP por petición |
| Backpressure para backends no-concurrentes | ❌ sin semáforo para Diffusers/Whisper/TTS/BitNet |

---

### Mejora A — httpx connection pool en `WorkerPool`

**Fichero:** `backend/ocabra/core/worker_pool.py`

**Problema:** `forward_request` y `forward_stream` crean `httpx.AsyncClient()` por cada petición, abriendo y cerrando una conexión TCP cada vez. Con concurrencia alta (ej. 10 req/s a vLLM) el overhead de handshake TCP es significativo.

**Solución:** Un `httpx.AsyncClient` persistente por puerto de worker, creado en `set_worker()` y cerrado en `remove_worker()`.

```python
class WorkerPool:
    def __init__(self) -> None:
        self._backends: dict[str, BackendInterface] = {}
        self._workers: dict[str, WorkerInfo] = {}
        self._used_ports: set[int] = set()
        self._clients: dict[str, httpx.AsyncClient] = {}   # nuevo: pool por model_id

    def set_worker(self, model_id: str, info: WorkerInfo) -> None:
        self._workers[model_id] = info
        self._used_ports.add(info.port)
        # Crear cliente persistente con pool de conexiones
        if info.backend_type not in ("ollama",):
            base_url = f"http://127.0.0.1:{info.port}"
            self._clients[model_id] = httpx.AsyncClient(
                base_url=base_url,
                timeout=300.0,
                limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
            )

    def remove_worker(self, model_id: str) -> None:
        info = self._workers.pop(model_id, None)
        if info:
            self._used_ports.discard(info.port)
        client = self._clients.pop(model_id, None)
        if client:
            # Cerrar async — llamar desde unload async
            asyncio.get_event_loop().create_task(client.aclose())

    async def forward_request(self, model_id: str, path: str, body: dict) -> Any:
        worker = self._workers.get(model_id)
        if not worker:
            raise KeyError(f"No worker found for model '{model_id}'")
        if worker.backend_type == "ollama":
            # Ollama sigue con cliente efímero (servicio externo, distinto host)
            base = settings.ollama_base_url.rstrip("/")
            async with httpx.AsyncClient(timeout=300.0) as client:
                resp = await client.post(f"{base}{path}", json=body)
                resp.raise_for_status()
                return resp.json()
        client = self._clients[model_id]
        resp = await client.post(path, json=body)
        resp.raise_for_status()
        return resp.json()

    async def forward_stream(self, model_id: str, path: str, body: dict) -> AsyncIterator[bytes]:
        worker = self._workers.get(model_id)
        if not worker:
            raise KeyError(f"No worker found for model '{model_id}'")
        if worker.backend_type == "ollama":
            base = settings.ollama_base_url.rstrip("/")
            async with httpx.AsyncClient(timeout=300.0) as client:
                async with client.stream("POST", f"{base}{path}", json=body) as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk
            return
        client = self._clients[model_id]
        async with client.stream("POST", path, json=body) as resp:
            async for chunk in resp.aiter_bytes():
                yield chunk
```

**Beneficio:** Las conexiones TCP se reutilizan (keep-alive). Para vLLM bajo carga sostenida, elimina la latencia de establecimiento de conexión en cada petición (~1-5 ms por handshake TCP localhost).

**Nota:** `httpx.AsyncClient` es thread-safe para uso concurrente cuando se comparte entre corutinas asyncio — no necesita lock externo.

---

### Mejora B — Semáforo de concurrencia para backends de inferencia única

**Problema:** Diffusers, Whisper, TTS y BitNet (CPU-only) no procesan varias peticiones en paralelo — la GPU/CPU ejecuta inferencia de forma secuencial. Sin límite en oCabra, N peticiones concurrentes se reenvían simultáneamente al worker, que las procesa en serie internamente pero acumula N conexiones abiertas y N buffers en memoria.

**Solución:** Un `asyncio.Semaphore` configurable por backend en `WorkerPool`, que limita las peticiones en vuelo para backends no-concurrentes. Las peticiones en exceso esperan en la cola asyncio (no se rechazan) con un timeout opcional que devuelve 503.

**Ficheros:**
- `backend/ocabra/core/worker_pool.py` — añadir semáforos
- `backend/ocabra/config.py` — añadir settings de límite

**Settings a añadir en `config.py`:**

```python
# Concurrency limits por backend (0 = sin límite)
diffusers_max_concurrent: int = 1
whisper_max_concurrent: int = 2
tts_max_concurrent: int = 2
bitnet_max_concurrent: int = 4   # llama-server puede encolar internamente
concurrency_queue_timeout_s: int = 120  # segundos antes de devolver 503
```

**Lógica en `WorkerPool`:**

```python
# Backends con concurrencia limitada (configurables)
_CONCURRENCY_LIMITS: dict[str, str] = {
    "diffusers": "diffusers_max_concurrent",
    "whisper":   "whisper_max_concurrent",
    "tts":       "tts_max_concurrent",
    "bitnet":    "bitnet_max_concurrent",
}

class WorkerPool:
    def __init__(self) -> None:
        ...
        self._semaphores: dict[str, asyncio.Semaphore] = {}

    def register_backend(self, backend_type: str, backend: BackendInterface) -> None:
        self._backends[backend_type] = backend
        limit_key = _CONCURRENCY_LIMITS.get(backend_type)
        if limit_key:
            limit = getattr(settings, limit_key, 0)
            if limit > 0:
                self._semaphores[backend_type] = asyncio.Semaphore(limit)

    async def forward_request(self, model_id: str, path: str, body: dict) -> Any:
        worker = self._workers.get(model_id)
        sem = self._semaphores.get(worker.backend_type) if worker else None
        if sem:
            try:
                await asyncio.wait_for(sem.acquire(), timeout=settings.concurrency_queue_timeout_s)
            except asyncio.TimeoutError:
                raise RuntimeError(
                    f"Backend '{worker.backend_type}' concurrency queue timeout "
                    f"({settings.concurrency_queue_timeout_s}s)"
                )
            try:
                return await self._do_forward_request(model_id, path, body)
            finally:
                sem.release()
        return await self._do_forward_request(model_id, path, body)
```

El mismo patrón aplica a `forward_stream` (adquiere el semáforo antes del primer chunk, lo libera cuando el generador termina/es cancelado).

**Comportamiento observable:**
- vLLM: sin semáforo (maneja concurrencia internamente)
- Ollama: sin semáforo (maneja concurrencia internamente)
- Diffusers: máx 1 petición activa → las demás esperan en cola asyncio
- Whisper/TTS: máx 2 peticiones activas
- BitNet: máx 4 peticiones activas (llama-server puede pre-procesar prompts solapados)

---

### Tests adicionales para las mejoras de concurrencia

**`tests/core/test_worker_pool_concurrency.py`:**
- `test_connection_pool_reuses_client` — mismo objeto `AsyncClient` en dos llamadas consecutivas
- `test_client_closed_on_remove_worker`
- `test_semaphore_limits_diffusers_concurrency` — lanzar N tareas concurrentes, verificar que solo 1 está activa simultáneamente
- `test_semaphore_timeout_raises_503` — simular worker lento, verificar que la cola expira correctamente
- `test_vllm_no_semaphore` — vLLM no tiene semáforo, N peticiones pasan sin esperar
