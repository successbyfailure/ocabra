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
| Backend TTS | Transformers (Qwen3-TTS, Kokoro), Chatterbox (Resemble AI) |
| Frontend | React 18 + TypeScript + Vite + TailwindCSS + shadcn/ui |
| Charts | Recharts |
| Estado frontend | Zustand |
| Tiempo real | WebSockets + SSE (FastAPI nativo) |
| Contenedores | Docker Compose con perfiles |
| Frontend serve | Nginx |
| Reverse proxy | Caddy |

## Estado actual (2026-04-12)

**Fases 0–8 completadas e implementadas. Versión: 0.6.0**

El backlog de refactorización y hardening de seguridad está cerrado (ver `docs/REFACTOR_PLAN.md`).
El trabajo restante está en `docs/ROADMAP.md`.

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
- **Model Profiles (Fase 6)**: Separación modelos/perfiles. `ModelProfile` con CRUD, resolución por `profile_id`, fallback legacy, worker sharing por `(base_model_id, load_overrides_hash)`, assets de perfil, UI en Models.
- **Chatterbox TTS (Fase 7 parcial)**: Backend first-class para Chatterbox Multilingual (23 idiomas, voice cloning). Worker FastAPI, validación de `voice_ref`, detección en scanner/registry.
- **Resiliencia de backends (Bloque 11)**: Interfaz unificada multi-modal (`ModalityType`, `supported_modalities()`), evicción LRU + umbral VRAM, busy timeout con `ActiveRequest` tracking, `BackendProcessManager` con health checks y auto-restart.
- **Federación P2P (Bloque 12)**: Modo federado peer-to-peer completo. `FederationManager` con heartbeat, cifrado Fernet, proxy transparente, load balancing, inventario federado en `/v1/models` y `/api/tags`, UI de gestión de peers, operaciones remotas.
- **Tests**: 586+ tests cubriendo path traversal, config, model manager, worker lifecycle, Langfuse, profiles, modalities, eviction, busy timeout, process manager, y federación (54 tests).

### Validaciones end-to-end confirmadas

- `llama.cpp`: `Qwen/Qwen2.5-0.5B-Instruct-GGUF` — registro, load, chat correctos.
- `SGLang`: `HuggingFaceTB/SmolLM2-135M-Instruct` — health/load correctos.
- `TensorRT-LLM`: `tensorrt_llm/Qwen3-8B-fp16` — carga, respuesta, descarga sin huérfanos.
- `vLLM`: `vllm/Qwen/Qwen3.5-0.8B` y `vllm/Qwen/Qwen3-32B-AWQ` (con `max_model_len=7800`).
- Compatibilidad Ollama: `/api/chat`, `/api/generate`, `/api/embed` (con fallback a `/api/embeddings` para Ollama < 0.3).
- Tests backend en verde: `test_service_manager.py`, `test_llama_cpp_backend.py`, `test_sglang_backend.py`, `test_tensorrt_llm_backend.py`, `test_path_traversal.py`, `test_config_patch.py`, `test_model_manager_config.py`, `test_worker_lifecycle.py`, `test_langfuse_tracer.py`.

### Referencia operativa

- Benchmark baseline: `docs/benchmarks/qwen3-backends-2026-04-03.md` (`vllm`, `tensorrt_llm`, `ollama` sobre Qwen3).

### Próximas fases

- **Backends Modulares**: Cada backend instalable/desinstalable en runtime desde la UI. Imagen Docker slim + distribución OCI. Plan en `docs/tasks/modular-backends-plan.md`.
- **Fine-tuning de voz**: Motor genérico de fine-tuning con UI wizard (Chatterbox + Qwen3-TTS). Auto-crea perfiles al completar el entrenamiento.

### Pendiente menor

Ver `docs/ROADMAP.md`:
- Validación TRT-LLM multi-engine en producción (requiere prueba manual)
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


**Convención de IDs (post Fase 6):**
- `model_id` (interno): formato `backend/model` (ej: `vllm/Qwen/Qwen3-8B`). Solo visible para admins en `/ocabra/models`.
- `profile_id` (público): nombre corto sin `/` (ej: `chat`, `tts-glados`, `stt-diarized`). Es lo que los clientes usan en `/v1/*`.
- **`/v1/models` solo lista perfiles habilitados.** Un modelo sin perfiles no es accesible via API.
- **Pre-Fase 6 (actual):** `model_id` canónico y `backend_model_id` alias se aceptan directamente en `/v1/*`.

```json
// Pre-Fase 6 (actual): expone model_id canónico
{
  "id": "vllm/mistral-7b-instruct",
  "capabilities": {
    "chat": true, "completion": true, "tools": true, "vision": false,
    "embeddings": false, "reasoning": false, "streaming": true,
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

// Post-Fase 6: expone profile_id (el modelo base es interno)
{
  "id": "chat",
  "object": "model",
  "owned_by": "ocabra",
  "ocabra": {
    "category": "llm",
    "status": "loaded",
    "capabilities": { "chat": true, "tools": true, "streaming": true, "context_length": 8192 },
    "display_name": "Chat (Qwen3-8B)"
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

### FASE 6 — Model Profiles: Separación Modelos / Perfiles (requiere Fase 5)

#### Principio de diseño

**Los modelos son internos; los perfiles son la interfaz pública.**

El sistema actual expone los `model_id` canónicos (`backend/model`) directamente en `/v1/models`.
Esto mezcla detalles de infraestructura (qué backend, qué repo HF) con lo que los clientes
necesitan: un nombre estable para consumir. Además obliga a registrar N veces el mismo modelo
si se quieren configuraciones distintas.

Con Model Profiles se separan las dos capas:

```
┌──────────────────────────────────────────────────────┐
│  CAPA INTERNA — Modelos (solo visible para admins)   │
│  ──────────────────────────────────────────────────── │
│  tts/qwen3-tts              [loaded]  GPU 1  8 GB    │
│  vllm/Qwen/Qwen3-8B        [loaded]  GPU 1  16 GB   │
│  whisper/openai/large-v3    [configured]              │
│  chatterbox/resemble/turbo  [loaded]  GPU 0  4 GB    │
│                                                       │
│  → Gestión: descargar, cargar/descargar, borrar,     │
│    configurar GPU, load_policy, extra_config          │
│  → NO expuestos en /v1/models                        │
└────────────────────┬─────────────────────────────────┘
                     │ cada modelo tiene 0..N perfiles
┌────────────────────▼─────────────────────────────────┐
│  CAPA PÚBLICA — Perfiles (lo que ven los clientes)   │
│  ──────────────────────────────────────────────────── │
│  tts-es           → tts/qwen3-tts      lang=es       │
│  tts-en           → tts/qwen3-tts      lang=en       │
│  tts-glados       → tts/qwen3-tts      voice=clone   │
│  chat             → vllm/Qwen/Qwen3-8B               │
│  chat-long        → vllm/Qwen/Qwen3-8B max_ctx=32k   │
│  chat-fast        → vllm/Qwen/Qwen3-8B max_ctx=4k    │
│  stt              → whisper/large-v3                  │
│  stt-diarized     → whisper/large-v3   diarize=true   │
│  tts-cb-es        → chatterbox/turbo   lang=es        │
│                                                       │
│  → Expuestos en /v1/models como modelos normales     │
│  → Los clientes usan model="tts-glados" sin saber    │
│    nada del modelo base                              │
└──────────────────────────────────────────────────────┘
```

#### Reglas fundamentales

1. **`/v1/models` solo lista perfiles habilitados**, nunca modelos raw.
2. **Un modelo sin perfiles no es accesible** via API de inferencia.
3. **Al registrar un modelo** el admin puede auto-crear un perfil "default" con el nombre que elija.
4. **Borrar un modelo** borra sus perfiles en cascada.
5. **Los `model_id` canónicos no se exponen** a clientes. Si un cliente manda
   `model="vllm/Qwen/Qwen3-8B"` recibe 404 (debe usar un perfil).
6. **Los perfiles NO colisionan** con model_ids porque viven en namespace separado
   (tabla distinta, resolución exclusiva por `profile_id`).

#### Tipos de profile override

| Categoría | Ejemplos | Efecto |
|-----------|----------|--------|
| **Serving (load)** | `max_model_len`, `gpu_memory_utilization`, `tensor_parallel_size`, `quantization` | Merge con `extra_config` del modelo base al cargar. Perfiles con `load_overrides` distintos **lanzan workers separados**. |
| **Runtime (request)** | `temperature`, `top_p`, `max_tokens`, `system_prompt`, `stop_sequences` | Inyectados como defaults en el body de cada request. El cliente puede sobreescribirlos. |
| **TTS voice** | `language`, `voice`, `speed`, `voice_ref_audio` | Inyectados en body. `voice_ref_audio` referencia un asset subido. |
| **STT config** | `diarization_enabled`, `language`, `word_timestamps` | `load_overrides` si afecta al worker, `request_defaults` si no. |
| **Image gen** | `default_steps`, `default_size`, `default_guidance`, `scheduler` | Defaults en body. |

#### Workers compartidos vs dedicados

Cuando dos perfiles apuntan al mismo modelo base:

- **Mismos `load_overrides`** (o vacíos ambos) → **comparten worker**. Solo difieren en
  `request_defaults` que se inyectan en el forwarding. Cero coste extra de GPU.
- **Distintos `load_overrides`** → **workers separados**. Ej: `chat` con `max_model_len=8192`
  y `chat-long` con `max_model_len=32768` necesitan dos procesos vLLM distintos.

El ModelManager trackea workers por la tupla `(base_model_id, load_overrides_hash)`:

```python
# Dos perfiles, un worker:
tts-es   → base=tts/qwen3-tts, load_overrides={}  → worker key: "tts/qwen3-tts:default"
tts-en   → base=tts/qwen3-tts, load_overrides={}  → worker key: "tts/qwen3-tts:default"  ← mismo

# Dos perfiles, dos workers:
chat      → base=vllm/Qwen3-8B, load_overrides={max_model_len:8192}  → worker key: "vllm/Qwen3-8B:a3f2c1"
chat-long → base=vllm/Qwen3-8B, load_overrides={max_model_len:32768} → worker key: "vllm/Qwen3-8B:b7d4e9"
```

#### Esquema BD: `ModelProfile`

```python
class ModelProfile(Base):
    __tablename__ = "model_profiles"

    profile_id    = Column(String, primary_key=True)      # "tts-glados", "chat-long"
    base_model_id = Column(String, ForeignKey("model_config.model_id", ondelete="CASCADE"),
                           nullable=False, index=True)
    display_name  = Column(String)                         # Nombre en /v1/models
    description   = Column(Text)                           # Descripción para UI
    category      = Column(String)                         # "llm", "tts", "stt", "image", "music"

    # Overrides que afectan al worker (pueden crear worker dedicado)
    load_overrides   = Column(JSONB, default=dict)

    # Defaults inyectados en cada request (no afectan al worker)
    request_defaults = Column(JSONB, default=dict)

    # Assets asociados (audio de referencia, LoRA adapter, etc.)
    assets           = Column(JSONB, default=dict)         # {"voice_ref": "/data/voices/glados.wav"}

    # Control
    enabled       = Column(Boolean, default=True)
    is_default    = Column(Boolean, default=False)         # Perfil auto-creado al registrar modelo
    created_at    = Column(DateTime(timezone=True), server_default=func.now())
    updated_at    = Column(DateTime(timezone=True), onupdate=func.now())
```

Constraint: `UNIQUE(base_model_id)` donde `is_default=True` (solo un default por modelo).
FK con `ondelete="CASCADE"` → borrar modelo borra sus perfiles.

#### Resolución en API

`resolve_profile()` reemplaza a `resolve_model()`:

```python
async def resolve_profile(profile_id: str, ...) -> tuple[ModelProfile, ModelState]:
    """Resuelve un profile_id → (perfil, estado del modelo base)."""
    profile = await get_profile(session, profile_id)
    if not profile or not profile.enabled:
        raise HTTPException(404, f"Model '{profile_id}' not found")
    state = await ensure_loaded(model_manager, profile.base_model_id,
                                load_overrides=profile.load_overrides)
    return profile, state
```

El forwarding inyecta los defaults:

```python
async def forward_with_profile(profile: ModelProfile, state: ModelState, body: dict):
    # request_defaults como base, body del cliente sobreescribe
    merged = {**profile.request_defaults, **body}
    # Inyectar assets si aplica
    if "voice_ref" in profile.assets:
        merged["voice_ref"] = profile.assets["voice_ref"]
    return await worker_pool.forward_request(state.worker_key, path, merged)
```

#### Exposición en API

**OpenAI `/v1/models`** — solo perfiles:
```json
{
  "data": [
    {
      "id": "tts-glados",
      "object": "model",
      "created": 1712400000,
      "owned_by": "ocabra",
      "ocabra": {
        "category": "tts",
        "status": "loaded",
        "capabilities": { "tts": true, "streaming": true },
        "display_name": "TTS Glados Voice",
        "description": "English TTS with Glados voice clone"
      }
    },
    {
      "id": "chat",
      "object": "model",
      "created": 1712400000,
      "owned_by": "ocabra",
      "ocabra": {
        "category": "llm",
        "status": "loaded",
        "capabilities": { "chat": true, "tools": true, "streaming": true, "context_length": 8192 }
      }
    }
  ]
}
```

Los clientes no ven `base_model_id`, `load_overrides`, ni detalles internos.

**Internal API `/ocabra/models`** (admin) — sin cambios, sigue mostrando modelos raw con
estado, GPU, VRAM, etc. Ahora incluye los perfiles asociados:

```json
{
  "model_id": "tts/qwen3-tts",
  "status": "loaded",
  "current_gpu": [1],
  "vram_used_mb": 8192,
  "profiles": [
    { "profile_id": "tts-es", "enabled": true, "is_default": false },
    { "profile_id": "tts-en", "enabled": true, "is_default": true },
    { "profile_id": "tts-glados", "enabled": true, "is_default": false }
  ]
}
```

**Endpoints de perfiles** (anidados bajo modelos):
- `GET    /ocabra/models/{model_id}/profiles`              — listar perfiles del modelo
- `POST   /ocabra/models/{model_id}/profiles`              — crear perfil
- `GET    /ocabra/profiles/{profile_id}`                   — detalle de perfil (atajo)
- `PATCH  /ocabra/profiles/{profile_id}`                   — actualizar
- `DELETE /ocabra/profiles/{profile_id}`                   — eliminar
- `POST   /ocabra/profiles/{profile_id}/assets`            — subir asset (audio ref, LoRA)
- `DELETE /ocabra/profiles/{profile_id}/assets/{asset_key}` — eliminar asset

#### Ejemplos concretos

**Ejemplo 1 — LLM con diferentes configuraciones de serving:**

```json
// Modelo base: vllm/Qwen/Qwen3-8B (interno, no expuesto)

// Perfil: chat (configuración default, accesible como model="chat")
{
  "profile_id": "chat",
  "base_model_id": "vllm/Qwen/Qwen3-8B",
  "category": "llm",
  "is_default": true,
  "load_overrides": { "max_model_len": 8192 },
  "request_defaults": {}
}

// Perfil: chat-long (contexto máximo, worker separado)
{
  "profile_id": "chat-long",
  "base_model_id": "vllm/Qwen/Qwen3-8B",
  "category": "llm",
  "load_overrides": { "max_model_len": 32768, "gpu_memory_utilization": 0.9 },
  "request_defaults": {}
}

// Perfil: chat-creative (mismo worker que "chat", solo cambia defaults)
{
  "profile_id": "chat-creative",
  "base_model_id": "vllm/Qwen/Qwen3-8B",
  "category": "llm",
  "load_overrides": { "max_model_len": 8192 },
  "request_defaults": { "temperature": 1.2, "top_p": 0.95 }
}
```

**Ejemplo 2 — TTS con idioma, voz fija y voice cloning:**

```json
// Modelo base: tts/qwen3-tts (interno)

// Perfil: tts-es
{
  "profile_id": "tts-es",
  "base_model_id": "tts/qwen3-tts",
  "category": "tts",
  "is_default": true,
  "load_overrides": {},
  "request_defaults": { "language": "es", "voice": "nova" }
}

// Perfil: tts-en
{
  "profile_id": "tts-en",
  "base_model_id": "tts/qwen3-tts",
  "category": "tts",
  "load_overrides": {},
  "request_defaults": { "language": "en", "voice": "alloy" }
}

// Perfil: tts-glados (voice cloning con audio de referencia)
{
  "profile_id": "tts-glados",
  "base_model_id": "tts/qwen3-tts",
  "category": "tts",
  "load_overrides": {},
  "request_defaults": { "language": "en", "voice": "clone" },
  "assets": { "voice_ref": "/data/profiles/tts-glados/reference.wav" }
}
```

**Ejemplo 3 — STT con y sin diarización:**

```json
// Modelo base: whisper/openai/whisper-large-v3 (interno)

// Perfil: stt (transcripción simple)
{
  "profile_id": "stt",
  "base_model_id": "whisper/openai/whisper-large-v3",
  "category": "stt",
  "is_default": true,
  "load_overrides": {},
  "request_defaults": {}
}

// Perfil: stt-diarized (mismo modelo, diarización activada)
{
  "profile_id": "stt-diarized",
  "base_model_id": "whisper/openai/whisper-large-v3",
  "category": "stt",
  "load_overrides": { "diarization_enabled": true },
  "request_defaults": { "word_timestamps": true }
}
```

#### Migración desde el sistema actual

1. **Variantes `::diarize`** → se migran a perfiles automáticamente.
2. **Modelos existentes sin perfil** → migración Alembic crea un perfil default por modelo
   con `profile_id` = `display_name` (o derivado del `backend_model_id`).
3. **Clientes que usaban `model_id` canónico** → periodo de transición configurable:
   si `LEGACY_MODEL_ID_FALLBACK=true`, las requests con `model_id` canónico buscan
   el perfil default del modelo. Log de deprecation warning. Default: `true` en v0.6,
   `false` en v0.7.

#### Stream A — Backend

- [ ] Tabla `ModelProfile` + migración Alembic (con auto-creación de perfiles default)
- [ ] `core/profile_registry.py`: CRUD en BD, cache en memoria, resolución
- [ ] Refactor `resolve_model()` → `resolve_profile()` en `_deps.py`
- [ ] Worker key por `(base_model_id, load_overrides_hash)` en ModelManager
- [ ] Merge de `request_defaults` + `assets` en forwarding
- [ ] `/v1/models` lista solo perfiles habilitados
- [ ] `/ocabra/models` incluye `profiles[]` anidado
- [ ] Endpoints REST para perfiles (CRUD + assets upload)
- [ ] Fallback legacy `LEGACY_MODEL_ID_FALLBACK` con deprecation warning
- [ ] Migración automática de variantes `::diarize` a perfiles
- [ ] Grupos de acceso: vincular perfiles a groups (no modelos)
- [ ] Tests: resolución, merge, cascada, legacy fallback, workers compartidos/dedicados

#### Stream B — Frontend

- [ ] Tipo `ModelProfile` en `types/index.ts`
- [ ] API client: `api.profiles.*` (CRUD, upload asset)
- [ ] Refactor página Models:
  - Vista principal: lista de **modelos** con sus perfiles anidados (acordeón/expandible)
  - Cada modelo muestra: estado, GPU, VRAM, nº perfiles, acciones (load/unload/delete)
  - Cada perfil muestra: nombre, categoría, estado del worker, enabled toggle
  - Botón "Crear perfil" por modelo → modal
  - Borrar modelo → confirmar cascada de perfiles
- [ ] Modal de creación/edición de perfil:
  - `profile_id` (slug, validado: sin `/`, sin espacios, lowercase)
  - `display_name`, `description`, `category`
  - Editor de `load_overrides` (formulario inteligente según backend + JSON raw)
  - Editor de `request_defaults` (formulario según categoría + JSON raw)
  - Upload de assets (drag & drop audio, seleccionar LoRA)
  - Toggle `enabled`, checkbox `is_default`
  - Preview: "Los clientes verán: `model='tts-glados'`"
- [ ] Auto-crear perfil default al registrar modelo nuevo (checkbox en Explore/Add)

---

### FASE 7 — Chatterbox TTS Backend + Fine-tuning UI (paralelo a Fase 6)

#### Stream A — Chatterbox Backend

Nuevo backend `chatterbox` para Chatterbox Multilingual (Resemble AI).
Sigue el mismo patrón que `tts_backend.py` y `voxtral_backend.py`.

- [ ] `backend/ocabra/backends/chatterbox_backend.py`:
  - `ChatterboxBackend(BackendInterface)` con load/unload/health/capabilities/forward
  - VRAM estimate: ~4096 MB (Turbo) / ~8192 MB (full)
  - Capabilities: `tts=True, streaming=True`
  - Voice mapping: OpenAI voices → Chatterbox speakers
  - Soporte de voice cloning via `voice_ref` en request body
- [ ] `backend/workers/chatterbox_worker.py`:
  - FastAPI worker con endpoints `/health`, `/info`, `/voices`, `/synthesize`, `/synthesize/stream`
  - Carga `ChatterboxTurbo` de `resemble_ai/chatterbox`
  - Soporte 23 idiomas con detección automática o selección explícita
  - Voice cloning: acepta `voice_ref` (path a audio) para zero-shot cloning
  - Streaming por frases (mismo patrón que `tts_worker.py`)
  - Encoding: MP3, WAV, FLAC, Opus, AAC, PCM
- [ ] Registro del backend en `main.py` (`register_backend("chatterbox", ...)`)
- [ ] VRAM estimates en `model_ref.py` o config
- [ ] Voice mappings para Chatterbox en `tts_backend.py` o `chatterbox_backend.py`
- [ ] Detección automática en `huggingface.py` / `local_scanner.py`
- [ ] `requirements-chatterbox.txt` o sección en `pyproject.toml`
- [ ] Dockerfile update: instalar dependencias Chatterbox (o venv separado como Voxtral)
- [ ] Tests: load/unload, synthesize, streaming, voice cloning

#### Stream B — Fine-tuning Engine (Backend)

Motor genérico de fine-tuning que soporte múltiples backends TTS.
Fase inicial: Chatterbox + Qwen3-TTS. Extensible a Orpheus y otros.

**Esquema BD: `FineTuningJob`**

```python
class FineTuningJob(Base):
    __tablename__ = "finetuning_jobs"

    job_id        = Column(String, primary_key=True, default=lambda: f"ftjob-{uuid4().hex[:12]}")
    base_model_id = Column(String, ForeignKey("model_config.model_id"), nullable=False)
    display_name  = Column(String)                         # "Voz de Glados"
    status        = Column(String, default="pending")      # pending, preparing, training, completed, failed, cancelled
    backend_type  = Column(String)                         # "chatterbox", "qwen3-tts", "orpheus"

    # Configuración de entrenamiento
    training_config = Column(JSONB, default=dict)          # epochs, lr, batch_size, etc.
    dataset_config  = Column(JSONB, default=dict)          # audio_dir, transcript_file, format

    # Progreso y resultado
    progress      = Column(JSONB, default=dict)            # {"epoch": 45, "total_epochs": 150, "loss": 0.023}
    result        = Column(JSONB, default=dict)            # {"adapter_path": "/data/finetunes/glados/", "metrics": {...}}
    error_message = Column(Text)

    # GPU y recursos
    gpu_index     = Column(Integer)
    vram_used_mb  = Column(Integer)

    # Metadata
    created_by    = Column(String)                         # user_id
    created_at    = Column(DateTime(timezone=True), server_default=func.now())
    started_at    = Column(DateTime(timezone=True))
    completed_at  = Column(DateTime(timezone=True))
```

**`FineTuningManager`** en `core/`:

```
Flujo:
1. Usuario sube dataset (audio + transcripciones) → /data/finetune-datasets/{job_id}/
2. POST /ocabra/finetuning → crea FineTuningJob, valida dataset
3. Manager asigna GPU (via scheduler), lanza worker de fine-tuning
4. Worker reporta progreso via Redis pub/sub → SSE al frontend
5. Al completar: genera adapter/checkpoint → registra como ModelProfile automáticamente
6. El perfil resultante apunta al modelo base + adapter path en assets
```

- [ ] Tabla `FineTuningJob` + migración Alembic
- [ ] `core/finetuning_manager.py`:
  - Cola de jobs (uno a la vez por GPU)
  - Asignación de GPU via scheduler existente
  - Spawn worker de fine-tuning como subproceso
  - Progreso via Redis pub/sub
  - Cancelación limpia (SIGTERM al worker)
  - Auto-creación de ModelProfile al completar
- [ ] `workers/finetune_chatterbox_worker.py`:
  - Acepta: `audio_dir`, `training_config`
  - Usa toolkit de Chatterbox fine-tuning
  - Reporta progreso (epoch, loss, samples/s) a Redis
  - Guarda checkpoint/adapter en directorio configurado
- [ ] `workers/finetune_qwen_tts_worker.py`:
  - Fine-tuning de Qwen3-TTS (0.6B o 1.7B)
  - Solo 5 min de audio necesarios
  - Salida: adapter/checkpoint compatible con `tts_worker.py`
- [ ] Dataset management:
  - Upload endpoint: `POST /ocabra/finetuning/datasets`
  - Validación: formato audio, sample rate, transcripciones
  - Almacenamiento: `/data/finetune-datasets/{job_id}/`
- [ ] Endpoints REST:
  - `GET    /ocabra/finetuning`                     — listar jobs
  - `POST   /ocabra/finetuning`                     — crear job
  - `GET    /ocabra/finetuning/{job_id}`             — detalle + progreso
  - `POST   /ocabra/finetuning/{job_id}/cancel`      — cancelar
  - `DELETE /ocabra/finetuning/{job_id}`             — eliminar job + artefactos
  - `GET    /ocabra/finetuning/{job_id}/progress`    — SSE de progreso en tiempo real
  - `POST   /ocabra/finetuning/datasets`             — subir dataset
  - `GET    /ocabra/finetuning/datasets`             — listar datasets
- [ ] Tests: lifecycle, cancelación, progreso, dataset validation

#### Stream C — Fine-tuning UI (Frontend)

Nueva página `/fine-tuning` para gestionar todo el proceso de fine-tuning.

- [ ] Página `FineTuning.tsx`:
  - Vista principal: lista de jobs con estado, progreso, modelo base, fecha
  - Filtros: por estado, por modelo base, por backend
  - Acciones: crear, cancelar, eliminar, ver detalle
- [ ] Componentes:
  - `FineTuneWizard.tsx`: wizard paso a paso para crear un job:
    1. **Seleccionar modelo base** — dropdown de modelos que soportan fine-tuning
       (filtrado por capability `finetunable`)
    2. **Subir dataset** — drag & drop de archivos audio + transcripciones
       Formatos: carpeta con WAV/MP3 + CSV/TXT con transcripciones
       Validación en tiempo real (duración total, sample rate, formato)
    3. **Configurar entrenamiento** — presets (quick/balanced/quality) + modo avanzado
       - Quick: 50 epochs, lr=1e-4 (para probar rápido)
       - Balanced: 150 epochs, lr=5e-5 (recomendado)
       - Quality: 300 epochs, lr=2e-5 (máxima calidad)
       - Advanced: editor de todos los hiperparámetros
    4. **Seleccionar GPU** — dropdown de GPUs disponibles con VRAM libre
    5. **Revisar y lanzar** — resumen + estimación de tiempo
  - `FineTuneJobCard.tsx`: card con estado, barra de progreso, métricas en vivo
  - `FineTuneDetail.tsx`: vista detallada de un job:
    - Gráfico de loss en tiempo real (Recharts)
    - Log de eventos
    - Audio samples de preview (si el backend los genera)
    - Botón "Crear Perfil" para registrar el resultado como ModelProfile
  - `DatasetManager.tsx`: gestión de datasets subidos
    - Lista con duración total, número de muestras, idioma detectado
    - Reproductor de muestras individuales
    - Reutilización de datasets entre jobs
- [ ] API client: `api.finetuning.*` (CRUD, upload, progress SSE)
- [ ] Zustand store: `fineTuningStore.ts`
- [ ] Ruta en `App.tsx`: `/fine-tuning` con `minRole: "model_manager"`
- [ ] Entrada en Sidebar: icono `Sparkles` o `Wand2`
- [ ] SSE hook para progreso en tiempo real (mismo patrón que downloads)

---

## Dependencias entre fases

```
Fase 0
  └── Fase 1 (A, B, C, D en paralelo)
        ├── A+B → Fase 2 (A, B, C, D, E + F opcional en paralelo)
        │          └── Fase 3 (A, B en paralelo)
        │                 └── Fase 4 (A, B, C, D en paralelo)
        │                        └── Fase 5
        │                             ├── Fase 6 (A, B en paralelo: Model Profiles)
        │                             └── Fase 7 (A, B, C en paralelo: Chatterbox + Fine-tuning)
        │                                  └── Fase 7 depende de Fase 6 para auto-crear perfiles
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
