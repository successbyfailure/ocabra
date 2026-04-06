# oCabra — Plan de Implementación

## Concepto

Servidor de modelos de IA de alto rendimiento compatible con las APIs de OpenAI y Ollama.
Gestiona múltiples GPUs, carga modelos bajo demanda, y sirve cualquier tipo de modelo
(LLM, imagen, audio, TTS, multimodal). Incluye sistema de autenticación propio (JWT + API keys),
grupos de acceso a modelos, y gateway para servicios interactivos de generación.
LiteLLM Proxy puede usarse opcionalmente como capa adicional de enrutamiento/rate-limiting.

---

## Hardware objetivo

| GPU | VRAM | TDP |
|-----|------|-----|
| RTX 3060 (GPU 0) | 12 GB | 170 W |
| RTX 3090 (GPU 1) | 24 GB | 370 W |

---

## Stack tecnológico

| Capa | Tecnología |
|------|-----------|
| Backend API | Python 3.11, FastAPI, Pydantic v2 |
| ORM / DB | SQLAlchemy 2.0 async, Alembic, PostgreSQL |
| Cache / Queue | Redis |
| GPU monitoring | pynvml (NVML bindings) |
| Backend LLM | vLLM + llama.cpp + SGLang (+ TensorRT-LLM opcional) |
| Backend Imagen | Diffusers + Accelerate |
| Backend Audio | faster-whisper |
| Backend TTS | Transformers (Qwen3-TTS, Kokoro) |
| Frontend | React 18 + TypeScript + Vite + TailwindCSS + shadcn/ui |
| Charts | Recharts |
| Estado frontend | Zustand |
| Tiempo real | WebSockets + SSE (FastAPI nativo) |
| Contenedores | Docker Compose con perfiles |
| Frontend serve | Nginx |
| Reverse proxy | Caddy |

## Estado actual (2026-04-06)

**Todas las fases del plan original (0–5) y el roadmap completo están implementados y en producción.**
**Versión: 0.5.0**

El backlog de refactorización y hardening de seguridad está cerrado (ver `docs/REFACTOR_PLAN.md`).
El trabajo restante (menor) está en `docs/ROADMAP.md`.

### Implementado en código

- Fases 0–5 completas.
- IDs canónicas de modelo `backend/model`, con alias `backend_model_id` en OpenAI `/v1/*`.
- Backends first-class: `vllm`, `diffusers`, `whisper`, `tts`, `voxtral`, `ollama`, `llama_cpp`, `sglang`, `tensorrt_llm`, `bitnet`, `acestep`.
- **Auth system completo**: JWT (cookie HTTP-only), API keys por usuario (`sk-ocabra-...`), 3 roles (`user`, `model_manager`, `system_admin`), grupos de acceso a modelos con `group_id` en API keys y request stats.
- Modo sin key configurable por separado para OpenAI y Ollama (anonymous → solo grupo default).
- **Settings persistidos en BD** (`server_config`): `PATCH /ocabra/config` persiste en PostgreSQL; `.env` solo establece valores iniciales.
- **Gateway de servicios** (`gateway/`): proxy para HunyuanVideo, ComfyUI, A1111, AceStep con directorio autenticado.
- **Stats ampliadas**: endpoints `recent`, `by-user`, `by-group`, `my`, `my-group`; admin puede crear keys para otros usuarios.
- **Langfuse**: Integración de observabilidad LLM opcional (desactivada por defecto). Trazas para streaming y non-streaming.
- **Voice Pipeline completo**:
  - TTS con encoding real (MP3/WAV/PCM/FLAC) y streaming por frases
  - Voxtral TTS backend (vllm-omni)
  - OpenAI Realtime API WebSocket (`/v1/realtime`) con pipeline STT→LLM→TTS y VAD
- **WebSocket system_alert**: alertas de temperatura GPU y fallos de carga de modelos.
- UI Settings y Stats con tabs (Radix); dashboard con log de últimas peticiones.
- Configuración por modelo con estimación rápida de memoria y probe real de engine vLLM.
- Endpoints expuestos: `/ocabra/models/storage`, `/metrics`, `/health`, `/ready`, `/ocabra/services/*`, `/ocabra/host/stats`.
- Stats persistidos: `request_stats`, `gpu_stats`, `model_load_stats`.
- Compilación de engines TRT-LLM desde la UI: `CompileManager`, endpoints SSE, modal y página `TrtllmEngines`.
- Path traversal protegido con `_is_path_within_base()` en borrado de modelos y engines.
- `global_schedules` persistido en BD via `replace_global_schedules()`.
- `last_request_at` persistido en Redis; rehidratado al arrancar.
- CORS restringido a `localhost/127.0.0.1` via `allow_origin_regex`.
- Frontend servido por Nginx; Caddy como reverse proxy.
- **Tests**: 69 tests cubriendo path traversal, config patch, model manager config, worker lifecycle, y Langfuse tracer.

### Validaciones end-to-end confirmadas

- `llama.cpp`: `Qwen/Qwen2.5-0.5B-Instruct-GGUF` — registro, load, chat correctos.
- `SGLang`: `HuggingFaceTB/SmolLM2-135M-Instruct` — health/load correctos.
- `TensorRT-LLM`: `tensorrt_llm/Qwen3-8B-fp16` — carga, respuesta, descarga sin huérfanos.
- `vLLM`: `vllm/Qwen/Qwen3.5-0.8B` y `vllm/Qwen/Qwen3-32B-AWQ` (con `max_model_len=7800`).
- Compatibilidad Ollama: `/api/chat`, `/api/generate`, `/api/embed` (con fallback a `/api/embeddings` para Ollama < 0.3).
- Tests backend en verde: `test_service_manager.py`, `test_llama_cpp_backend.py`, `test_sglang_backend.py`, `test_tensorrt_llm_backend.py`, `test_path_traversal.py`, `test_config_patch.py`, `test_model_manager_config.py`, `test_worker_lifecycle.py`, `test_langfuse_tracer.py`.

### Referencia operativa

- Benchmark baseline: `docs/benchmarks/qwen3-backends-2026-04-03.md` (`vllm`, `tensorrt_llm`, `ollama` sobre Qwen3).

### Pendiente menor

Ver `docs/ROADMAP.md`:
- Documentación OpenAPI enriquecida
- Script first-run / instalación
- Tests e2e: flujos load/unload por backend, TRT-LLM compile mock
- Limpiar `tensorrt_llm/Qwen3-32B-AWQ-fp16` del inventario

---

## Arquitectura

```
                    ┌─────────────────────┐
                    │   LiteLLM Proxy     │  ← Routing/rate-limit opcional
                    │   (externo)         │
                    └─────────┬───────────┘
                              │ OpenAI API
                    ┌─────────▼───────────┐
                    │   oCabra Core API   │  FastAPI
                    │  ┌───────────────┐  │
                    │  │ OpenAI compat │  │  /v1/*
                    │  │ Ollama compat │  │  /api/*
                    │  │ oCabra admin  │  │  /ocabra/*
                    │  └───────┬───────┘  │
                    │  ┌───────▼───────┐  │
                    │  │  GPU Manager  │  │  pynvml, scheduling
                    │  │ Model Manager │  │  load/unload/pin
                    │  │ Worker Pool   │  │  process lifecycle
                    │  └───────┬───────┘  │
                    └─────────┼───────────┘
                              │
          ┌───────────────────┼──────────────────────┐
          │                   │                      │
   ┌──────▼──────┐   ┌────────▼───────┐   ┌─────────▼──────┐
   │ vLLM worker │   │Diffusers worker│   │ Whisper/TTS    │
   │ (por modelo)│   │ (por modelo)   │   │ worker         │
   │ GPU 0 o 1   │   │ GPU 0 o 1      │   │ GPU 0 o 1      │
   └─────────────┘   └────────────────┘   └────────────────┘

   Redis: event bus, queues, cache de metadatos
   PostgreSQL: config, stats, model registry
   Web UI: React SPA (Nginx), WebSocket → Core API
```

### Ciclo de vida de un modelo

```
DISCOVERED → CONFIGURED → [LOADING] → LOADED → [IDLE] → [UNLOADING] → UNLOADED
                               ↑                    │
                               └────────────────────┘  (reload bajo demanda / pin)
```

---

## Estructura de directorios

```
ocabra/
├── docker-compose.yml
├── .env.example
├── docs/
│   └── PLAN.md
│
├── backend/
│   ├── Dockerfile
│   ├── pyproject.toml
│   ├── alembic/
│   └── ocabra/
│       ├── main.py                  # FastAPI app, lifespan
│       ├── config.py                # Settings (pydantic-settings, .env)
│       ├── database.py              # SQLAlchemy async engine + session
│       ├── redis_client.py
│       │
│       ├── api/
│       │   ├── openai/              # /v1/* endpoints
│       │   │   ├── models.py
│       │   │   ├── chat.py
│       │   │   ├── completions.py
│       │   │   ├── embeddings.py
│       │   │   ├── images.py
│       │   │   ├── audio.py
│       │   │   └── pooling.py
│       │   ├── ollama/              # /api/* endpoints
│       │   │   ├── tags.py
│       │   │   ├── show.py
│       │   │   ├── pull.py
│       │   │   ├── generate.py
│       │   │   ├── chat.py
│       │   │   ├── embeddings.py
│       │   │   └── delete.py
│       │   └── internal/            # /ocabra/* endpoints
│       │       ├── models.py
│       │       ├── gpus.py
│       │       ├── stats.py
│       │       ├── config.py
│       │       ├── downloads.py
│       │       ├── registry.py
│       │       ├── services.py
│       │       ├── trtllm.py
│       │       └── ws.py
│       │
│       ├── core/
│       │   ├── gpu_manager.py       # Detección, monitoreo NVML, power stats
│       │   ├── model_manager.py     # Load/unload/pin, state machine
│       │   ├── scheduler.py         # GPU assignment, pressure eviction, schedules
│       │   ├── worker_pool.py       # Spawn/kill/proxy a workers
│       │   ├── model_ref.py         # IDs canónicas backend/model
│       │   ├── service_manager.py   # Orquestación de servicios interactivos
│       │   └── trtllm_compile_manager.py
│       │
│       ├── backends/
│       │   ├── base.py              # BackendInterface abstracta
│       │   ├── vllm_backend.py      # vLLM process manager + proxy
│       │   ├── diffusers_backend.py # Stable Diffusion / FLUX
│       │   ├── whisper_backend.py   # faster-whisper
│       │   ├── tts_backend.py       # Qwen3-TTS, Kokoro
│       │   ├── ollama_backend.py
│       │   ├── bitnet_backend.py
│       │   ├── acestep_backend.py
│       │   └── vllm_recipes.py
│       │
│       ├── registry/
│       │   ├── huggingface.py       # HF Hub API: buscar, metadata, download
│       │   ├── ollama_registry.py   # ollama.com model list
│       │   ├── bitnet_registry.py
│       │   └── local_scanner.py     # Escanear modelos locales
│       │
│       ├── integrations/
│       │   └── litellm_sync.py      # Auto-actualizar config de LiteLLM Proxy
│       │
│       ├── stats/
│       │   ├── collector.py         # Middleware de métricas por request
│       │   ├── gpu_power.py         # Energía, estimación coste por petición
│       │   └── aggregator.py        # Agregación periódica a BD
│       │
│       └── db/
│           ├── model_config.py      # SQLAlchemy: ModelConfig, EvictionSchedule
│           ├── stats.py             # SQLAlchemy: RequestStat, GpuStat, ModelLoadStat
│           └── trtllm.py            # SQLAlchemy: metadatos de compilación TRT-LLM
│
├── frontend/
│   ├── Dockerfile
│   ├── nginx.conf
│   ├── package.json
│   ├── vite.config.ts
│   └── src/
│       ├── App.tsx
│       ├── pages/
│       │   ├── Dashboard.tsx        # GPU cards, modelos activos, logs live
│       │   ├── Models.tsx           # Modelos instalados + acciones
│       │   ├── Explore.tsx          # Browser HuggingFace + Ollama
│       │   ├── Playground.tsx       # Chat/imagen/audio de prueba
│       │   ├── Stats.tsx            # Gráficos de uso, tokens, energía
│       │   ├── Settings.tsx         # Config servidor, GPUs, LiteLLM sync
│       │   └── TrtllmEngines.tsx    # Engines TRT-LLM compilados
│       ├── components/
│       │   ├── gpu/
│       │   │   ├── GpuCard.tsx
│       │   │   └── PowerGauge.tsx
│       │   ├── models/
│       │   │   ├── ModelCard.tsx
│       │   │   ├── ModelBadges.tsx   # Capabilities badges
│       │   │   └── DownloadQueue.tsx
│       │   ├── playground/
│       │   │   ├── ChatInterface.tsx
│       │   │   ├── ImageInterface.tsx
│       │   │   └── AudioInterface.tsx
│       │   └── ui/                  # shadcn/ui components
│       ├── api/
│       │   └── client.ts            # Typed API client (oCabra internal API)
│       ├── stores/
│       │   ├── gpuStore.ts
│       │   ├── modelStore.ts
│       │   ├── downloadStore.ts
│       │   └── serviceStore.ts
│       └── hooks/
│           └── useWebSocket.ts
│
└── workers/                         # Scripts standalone para backends
    ├── diffusers_worker.py
    ├── llama_cpp_worker.py
    ├── sglang_worker.py
    ├── tensorrt_llm_worker.py
    ├── tts_worker.py
    ├── vllm_worker.py               # Wraps vLLM OpenAI server + healthcheck
    └── whisper_worker.py
```

---

## Capacidades reportadas por la API

Cada modelo expone sus capacidades en `/v1/models` y `/api/show`:


**Convención de IDs de modelo (canónica):**
- `model_id` usa siempre el formato `backend/model`.
- Ejemplos: `vllm/meta-llama/Llama-3.3-70B-Instruct`, `ollama/mistral:7b`, `whisper/openai/whisper-large-v3`.
- `backend_model_id` es el id nativo del runtime sin prefijo (`model`).
- En OpenAI `/v1/*`, también se acepta `backend_model_id` como alias de `model`; si hay varios matches, se usa el primero encontrado.

```json
{
  "id": "vllm/mistral-7b-instruct",
  "capabilities": {
    "chat": true,
    "completion": true,
    "tools": true,
    "vision": false,
    "embeddings": false,
  "reasoning": false,
  "pooling": false,
  "rerank": false,
  "classification": false,
  "score": false,
  "image_generation": false,
  "audio_transcription": false,
    "music_generation": false,
    "tts": false,
    "streaming": true,
    "context_length": 32768
  },
  "status": "loaded",
  "ocabra": {
    "load_policy": "warm",
    "gpu": [1],
    "vram_used_mb": 8200,
    "display_name": "mistral-7b-instruct",
    "backend_model_id": "mistral-7b-instruct"
  }
}
```

---

## Políticas de carga de modelos

| Política | Carga | Descarga por idle | Descarga por presión | Schedule de evicción | Recarga automática |
|---|---|---|---|---|---|
| **`pin`** | Al arrancar oCabra | Nunca | Solo último recurso | Sí (se descarga en ventana horaria) | Sí, al terminar la ventana o cuando haya VRAM |
| **`warm`** | Primera petición | No | Sí (prioridad baja) | Sí (se descarga en ventana horaria) | Sí, en cuanto haya VRAM y fuera de ventana |
| **`on_demand`** | Primera petición | Sí (tras idle timeout) | Sí (primera en salir) | No aplica (ya se descarga sola) | No (se carga al llegar petición) |

**Orden de evicción** bajo presión (primero en salir):
1. `on_demand` con mayor idle time
2. `on_demand` recientes
3. `warm`
4. `pin` (solo si no queda otra opción)

**Schedules de evicción** (ventanas horarias configurables por modelo o globalmente):
- Durante la ventana: se descargan modelos `warm` y/o `pin` para liberar VRAM/energía.
- Al salir de la ventana: los modelos `pin` y `warm` con `auto_reload=true` se recargan.
- Múltiples schedules posibles (e.g., "lun-vie 02:00-06:00", "fines de semana todo el día").

## GPU Scheduler — Asignación

1. **Preferred GPU** configurado por servidor (default: GPU1 = 3090) y override por modelo.
2. **Fit check**: si el modelo no cabe en la GPU preferida, se intenta la alternativa.
3. **Tensor parallelism**: si el modelo no cabe en ninguna GPU individual, se usa span GPU0+GPU1.

---

## Estadísticas de energía

Por cada request se registra:
- Duración (ms)
- Tokens de entrada / salida
- GPU utilizada
- Δ energía estimada (W·s) = `power_draw_avg × duration_s`
- Coste energético estimado (configurable €/kWh)

Reportado en `/ocabra/stats/energy` y visible en la UI.

---

## Integración LiteLLM

Cuando se añade/elimina/activa/desactiva un modelo en oCabra:
- oCabra actualiza automáticamente el `config.yaml` de LiteLLM (vía API REST de LiteLLM o reescritura de fichero).
- El formato generado es compatible con LiteLLM proxy model list.
- Se puede configurar: endpoint base, api_key interna, modelo alias.

---

## Fases de implementación (paralelizable)

### FASE 0 — Fundación (secuencial, bloquea todo)

**Completada.**

- [x] Estructura de directorios completa
- [x] `docker-compose.yml` con servicios: api, frontend, postgres, redis, caddy
- [x] `pyproject.toml` con dependencias backend
- [x] `package.json` con dependencias frontend
- [x] `.env.example`
- [x] FastAPI app skeleton con lifespan (startup/shutdown)
- [x] SQLAlchemy async setup + modelos de BD + Alembic
- [x] Redis client + pub/sub helper
- [x] Config system (pydantic-settings)
- [x] React app base + routing (React Router) + TailwindCSS + shadcn/ui init
- [x] Layout principal con sidebar de navegación

---

### FASE 1 — Core paralelo (4 streams independientes tras Fase 0)

#### Stream A — GPU Manager + Scheduler
- pynvml: detección, polling VRAM, utilización, temperatura, power draw
- GPU state model (available VRAM, locked VRAM por modelo)
- Algoritmo de asignación (preferred → fallback → tensor parallel)
- Eviction scheduler: presión, idle timeout, horarios (loops `asyncio` + cron en BD)
- WebSocket event emitter para UI en tiempo real
- Endpoints: `GET /ocabra/gpus`, `GET /ocabra/gpus/{id}/stats`

#### Stream B — Model Manager + Worker Pool
- Máquina de estados del modelo (DISCOVERED → LOADED → UNLOADED)
- Worker Pool: spawn/kill subprocesos, health check, port assignment dinámico
- BackendInterface abstracta + registro de backends
- Model pin logic + auto-reload logic
- Endpoints: `GET/POST/DELETE /ocabra/models`, `POST /ocabra/models/{id}/load`, `/unload`, `PATCH`

#### Stream C — Model Registry + Download Manager
- HuggingFace Hub API: búsqueda, metadatos, archivos, descarga con progreso
- Ollama Registry: parsear ollama.com/library
- Local scanner: detectar modelos en carpeta configurada
- Download queue (Redis-backed) con progreso en tiempo real via SSE
- Endpoints: `GET /ocabra/registry/hf/*`, `GET /ocabra/registry/ollama/*`, `GET/POST /ocabra/downloads`

#### Stream D — Frontend Skeleton + Dashboard
- Zustand stores: gpuStore, modelStore, downloadStore, serviceStore
- Typed API client (`src/api/client.ts`)
- `useWebSocket` hook para eventos en tiempo real
- Página Dashboard: GpuCard (VRAM bar, temp, power gauge), lista modelos activos
- Componente DownloadQueue con barra de progreso SSE

---

### FASE 2 — Backends (5 streams + 1 opcional, requiere Stream B de Fase 1)

#### Stream A — vLLM Backend
- `vllm_worker.py`: lanza `vllm.entrypoints.openai.api_server` como subproceso
- Detección automática de capacidades del modelo (tools, vision, reasoning, context_length)
- Proxy de requests con headers correctos
- Gestión de errores y reintentos

#### Stream B — Diffusers Backend
- `diffusers_worker.py`: FastAPI interno que sirve `/generate`
- Soporte Stable Diffusion 1.5, XL, FLUX
- Parámetros: prompt, negative_prompt, steps, guidance, size, seed
- Devuelve imagen en base64 (compatible OpenAI `/v1/images/generations`)

#### Stream C — Whisper + TTS Backends
- `whisper_worker.py`: faster-whisper, endpoint `/transcribe`
- `tts_worker.py`: Qwen3-TTS / Kokoro, endpoint `/synthesize`
- Compatible con OpenAI `/v1/audio/transcriptions` y `/v1/audio/speech`

#### Stream D — llama.cpp Backend
- `llama_cpp_worker.py`: servidor OpenAI-compatible de llama.cpp (o wrapper propio)
- Soporte GGUF (CPU/GPU offload), chat/completions/embeddings según modelo
- Integración de capacidades (`tools`, `vision` si aplica, `context_length`)
- Soporte de ids canónicas `llama_cpp/<model>` + alias por nombre nativo

#### Stream E — SGLang Backend
- `sglang_worker.py`: arranque de runtime SGLang por modelo
- Soporte principal para chat/completions (OpenAI-compatible)
- Integración de capacidades y límites de contexto
- Soporte de ids canónicas `sglang/<model>` + alias por nombre nativo

#### Stream F (Opcional) — TensorRT-LLM Backend
- `tensorrt_llm_worker.py` con activación por feature flag/profile Docker
- Integración inicial para inferencia LLM optimizada en NVIDIA
- Fallback explícito a deshabilitado si no hay engine/modelo compatible

---

### FASE 3 — API Compatibility (2 streams paralelos, requiere Fase 2)

#### Stream A — OpenAI API
- `GET /v1/models` — lista con capabilities
- `POST /v1/chat/completions` — streaming SSE, tools, vision
- `POST /v1/completions`
- `POST /v1/embeddings`
- `POST /v1/images/generations`
- `POST /v1/audio/transcriptions`
- `POST /v1/audio/speech`

#### Stream B — Ollama API
- `GET /api/tags`
- `POST /api/show`
- `POST /api/pull` (delega a download manager)
- `POST /api/chat` (streaming NDJSON)
- `POST /api/generate`
- `POST /api/embeddings`

---

### FASE 4 — Frontend Features (paralelo, requiere Fase 3 para datos reales)

#### Stream A — Model Management UI
- Página Models: lista instalados, estado, GPU asignada, VRAM, botones load/unload/pin
- Página Explore: búsqueda HuggingFace + Ollama, filtros por tipo/tamaño/capabilities
- Modal de configuración por modelo (GPU preferred, pin, auto_reload, horarios)

#### Stream B — Playground
- Chat UI (markdown, streaming, tool calls rendering)
- Image UI (prompt, parámetros, galería de resultados)
- Audio UI (recorder, transcripción live, TTS player)
- Model selector con badge de capacidades

#### Stream C — Stats UI
- Gráficos tokens/s, latencia P50/P95, requests/minuto (Recharts)
- Panel energía: W actuales, kWh sesión, estimación coste
- Tabla top modelos por uso

#### Stream D — Settings UI
- Config global (carpeta modelos, GPU default, eviction policy)
- LiteLLM sync: URL, api_key, toggle auto-sync, botón sync manual
- Gestión de schedules de pin/eviction
- Variables de entorno cargadas (readonly)

---

### FASE 5 — Integrations + Polish (secuencial, requiere todo lo anterior)

- [x] LiteLLM auto-sync: detectar cambios de modelo → actualizar config LiteLLM via API
- [x] Logging estructurado (structlog) con nivel configurable
- [x] Healthcheck endpoints (`/health`, `/ready`)
- [x] Tests de integración para API OpenAI + Ollama
- [x] Tests de integración para `llama.cpp` y `SGLang`
- [x] Test smoke opcional para `TensorRT-LLM` (si profile habilitado)
- [ ] Documentación OpenAPI enriquecida
- [ ] Script de instalación / first-run

---

## Dependencias entre fases

```
Fase 0
  └── Fase 1 (A, B, C, D en paralelo)
        ├── A+B → Fase 2 (A, B, C, D, E + F opcional en paralelo)
        │          └── Fase 3 (A, B en paralelo)
        │                 └── Fase 4 (A, B, C, D en paralelo)
        │                        └── Fase 5
        └── C → Fase 2-C (parcialmente independiente)
        └── D → Fase 4 (puede empezar con mocks)
```

---

## Docker Compose — Servicios

```yaml
services:
  api:          # oCabra Core API (FastAPI) — GPU passthrough nvidia
  frontend:     # React SPA servida por Nginx
  postgres:     # PostgreSQL 16
  redis:        # Redis 7
  caddy:        # Reverse proxy: / → frontend, /v1 /api /ocabra → api
```

Workers de modelos son subprocesos gestionados por `api`, NO servicios Docker separados
(para poder asignar GPU dinámicamente y controlar el ciclo de vida).

---

## Próximos pasos inmediatos (Fase 0)

1. Crear estructura de directorios y ficheros base
2. `docker-compose.yml` funcional
3. FastAPI con lifespan + health endpoint
4. SQLAlchemy models + primera migración Alembic
5. React app con routing y layout

¿Arrancamos?
