# oCabra — Roadmap

Última actualización: 2026-04-06

Fuente de verdad del trabajo pendiente. `docs/PLAN.md` documenta la arquitectura y las
fases completadas. `docs/REFACTOR_PLAN.md` recoge el estado del refactor (cerrado).

---

## Estado general

El sistema está en producción. Todas las fases del plan original (0–5) están completadas.
Los backends validados end-to-end son: `vllm`, `llama_cpp`, `sglang`, `tensorrt_llm`,
`ollama`, `diffusers`, `whisper`, `tts`, `bitnet`, `acestep`.
La UI de compilación de engines TRT-LLM también está implementada y cableada.

---

## ✅ Bloque 1 — Auth, usuarios y grupos (COMPLETADO)

Plan completo en `docs/tasks/auth-system-plan.md`.

### Implementado

- 3 roles: `user`, `model_manager`, `system_admin`
- Sesión JWT (cookie HTTP-only) para el dashboard
- API keys por usuario (`sk-ocabra-...`) para OpenAI y Ollama
- Modo sin key configurable por separado (anonymous → solo grupo default)
- Grupos de acceso a modelos con selector por backend en la UI
- Página de API Keys en el sidebar
- Auth interna reemplaza LiteLLM como capa de autenticación
- Gateway de servicios con directorio autenticado

---

## ✅ Bloque 2 — Fixes pendientes (COMPLETADO)

- [x] **[B6]** `ModelPatch` ya usa `model_dump(exclude_unset=True)` — permite resetear `preferred_gpu` a null
- [x] **[A4]** `_watch_and_reload` ya tiene `while not self._stopped` — check de shutdown implementado
- [x] **[M7]** `HFModelVariant.backend_type` Literal incluye los 11 backends
- [x] **[WS]** `system_alert` WebSocket event — emitido desde GPU manager (temperatura) y model manager (fallos de carga)

---

## ✅ Bloque 3 — Persistencia de settings en base de datos (COMPLETADO)

Implementado: tabla `server_config`, overrides cargados al arranque, `PATCH /ocabra/config` persiste en BD.

---

## ✅ Bloque 4 — Integración Langfuse (COMPLETADO)

**Opcional, desactivado por defecto.** Plan en `docs/tasks/langfuse-integration-plan.md`.

Implementado:
- `backend/ocabra/integrations/langfuse_tracer.py` — singleton, `trace_generation()`, `wrap_stream()`, `parse_sse_chunk()`
- Hook en `WorkerPool.forward_request` y `forward_stream`
- Settings `langfuse_*` en `config.py`, variables en `docker-compose.yml` y `.env.example`
- Dependencia `langfuse>=2.0` en `pyproject.toml`
- Tests: 11 casos en `backend/tests/integrations/test_langfuse_tracer.py`

---

## ✅ Bloque 5 — Cobertura de tests (COMPLETADO)

109 tests cubriendo paths críticos y flujos e2e:

| Test file | Función | Tests |
|-----------|---------|-------|
| `test_path_traversal.py` | `_delete_model_files()`, `_is_path_within_base()`, TRT-LLM engine deletion | 14 |
| `test_config_patch.py` | `patch_config()` schema + runtime | 12 |
| `test_model_manager_config.py` | `update_config()` whitelist | 17 |
| `test_worker_lifecycle.py` | Port lifecycle, worker registration, forward errors, backends | 15 |
| `test_langfuse_tracer.py` | Langfuse integration (disabled, streaming, content, sampling) | 11 |
| `test_load_unload_e2e.py` | Load/unload cycle, eviction, auto-reload, idle, policies | 17 |
| `test_trtllm_compile.py` | Compile jobs, Docker cmd, phases, cancel, failure handling | 23 |

---

## ✅ Bloque 6 — Operativo / deuda técnica menor (COMPLETADO)

- [x] Limpiar entrada `tensorrt_llm/Qwen3-32B-AWQ-fp16` del inventario (SQL directo)
- [x] Documentación OpenAPI enriquecida — summaries, descriptions y responses en todos los endpoints `/ocabra/*`
- [x] Script first-run (`setup.sh`) + guía de instalación (`docs/INSTALL.md`)
- [ ] Validación TRT-LLM con múltiples engines en producción (requiere prueba manual con hardware)

---

## ✅ Bloque 7 — Teams, stats ampliadas, admin UX (COMPLETADO)

Implementado:
- 7.1 `group_id` en ApiKeys y RequestStats (migración `0009`, collector, UserContext)
- 7.2 Admin crea keys para otros usuarios (`POST /ocabra/users/{user_id}/keys` + modal UI)
- 7.3 Endpoints de stats: `recent`, `by-user`, `by-group`, `my`, `my-group`
- 7.4 Settings en tabs (Radix tabs: General, GPUs, Backends, Almacenamiento, LiteLLM)
- 7.5 Stats en tabs (7 tabs con filtro por rol)
- 7.6 Dashboard log de últimas 20 peticiones (poll 30s)

---

## ✅ Bloque 8 — Voice Pipeline (COMPLETADO)

**Plan completo:** `docs/tasks/voice-pipeline-plan.md`

### Fase 1 — Tres endpoints oficiales funcionando correctamente (✅ COMPLETADA)

| Item | Archivo | Estado |
|------|---------|--------|
| [VP-1] TTS encoding real | `backend/workers/tts_worker.py` | ✅ MP3/WAV/PCM/FLAC via soundfile+ffmpeg |
| [VP-2] TTS streaming por frases | `backend/workers/tts_worker.py` + `api/openai/audio.py` | ✅ `/synthesize/stream` |
| [VP-3] STT verificar M4A | `backend/workers/whisper_worker.py` | ✅ faster-whisper acepta M4A |
| [VP-4] Voices endpoint | `backend/ocabra/api/openai/audio.py` | ✅ `GET /v1/audio/voices?model=` |

### Fase 1.5 — Voxtral TTS backend (vllm-omni) (✅ COMPLETADA)

| Item | Archivo | Estado |
|------|---------|--------|
| [VP-9] Voxtral worker | `backend/workers/voxtral_worker.py` | ✅ Wrapper FastAPI + vllm-omni subprocess |
| [VP-10] Voxtral backend | `backend/ocabra/backends/voxtral_backend.py` | ✅ BackendInterface |
| [VP-11] Registro backend | `backend/ocabra/main.py` + `config.py` | ✅ `voxtral_python_bin` |
| [VP-12] Instalar vllm-omni | `/opt/voxtral-venv/` vllm==0.18.0 + vllm-omni==0.18.0 | ✅ Instalado y testeado |

### Fase 2 — OpenAI Realtime API (`GET /v1/realtime`) (✅ COMPLETADA)

| Item | Archivo | Estado |
|------|---------|--------|
| [VP-5] Endpoint WebSocket | `backend/ocabra/api/openai/realtime.py` | ✅ WS `/v1/realtime?model=` con auth |
| [VP-6] RealtimeSession | `backend/ocabra/core/realtime_session.py` | ✅ Pipeline STT→LLM→TTS, protocolo completo |
| [VP-7] VAD servidor | `backend/ocabra/core/vad.py` | ✅ SimpleVAD por RMS |
| [VP-8] session.update | En `RealtimeSession` | ✅ Configuración de voz, modelo STT, VAD params |

---

## En progreso — Bloque 9 — Model Profiles (Fase 6)

Plan completo en `docs/PLAN.md` (Fase 6) y `docs/tasks/model-profiles-chatterbox-swarm-plan.md`.

Separación entre modelos base (internos, admin) y perfiles públicos (clientes).
Los perfiles son la interfaz pública de `/v1/models` y de toda la inferencia.

| Item | Descripción | Estado |
|------|-------------|--------|
| Tabla `ModelProfile` + migración | DB schema, FK a `model_configs`, cascada | En progreso |
| `core/profile_registry.py` | CRUD, cache, resolución de perfiles | En progreso |
| Endpoints REST `/ocabra/profiles/*` | CRUD + upload de assets | En progreso |
| `/ocabra/models` con `profiles[]` | Anidación de perfiles en respuesta admin | En progreso |
| `resolve_profile()` en `/v1/*` y `/api/*` | Resolución pública por `profile_id` | En progreso |
| Worker key por `(base_model_id, load_overrides_hash)` | Workers compartidos/dedicados | En progreso |
| Fallback legacy `LEGACY_MODEL_ID_FALLBACK` | Compatibilidad temporal con `model_id` canónico | En progreso |
| UI de perfiles en Models | CRUD, upload audio, toggle enabled/default | En progreso |
| Contratos y tests | `docs/CONTRACTS.md` sección 8, test stubs | En progreso |

---

## En progreso — Bloque 10 — Chatterbox TTS (Fase 7 parcial)

Plan completo en `docs/PLAN.md` (Fase 7, Stream A) y `docs/tasks/model-profiles-chatterbox-swarm-plan.md`.

Backend first-class para Chatterbox Multilingual (Resemble AI, MIT, 23 idiomas, voice cloning).
El motor de fine-tuning (Fase 7, Stream B/C) queda fuera de esta oleada.

| Item | Descripción | Estado |
|------|-------------|--------|
| `chatterbox_backend.py` | BackendInterface para Chatterbox | En progreso |
| `chatterbox_worker.py` | Worker FastAPI con synthesize/stream/voices | En progreso |
| Registro en `main.py` | `register_backend("chatterbox", ...)` | En progreso |
| Detección en scanner/registry | HuggingFace y local scanner | En progreso |
| Voice cloning via `voice_ref` | Path controlado por oCabra (asset de perfil) | En progreso |
| Tests backend | load/unload, synthesize, streaming, voice cloning | En progreso |

---

## Pendiente — Bloque 11 — Resiliencia de backends y gestión avanzada de recursos

**Plan detallado:** `docs/tasks/backend-resilience-plan.md`

Inspirado en análisis comparativo con LocalAI y AnythingLLM. Objetivo: mejorar la
estabilidad, el aislamiento de fallos y la eficiencia en el uso de VRAM.
Orden recomendado: 11.3 → 11.1 → 11.4 → 11.2 (de menor a mayor riesgo).

### 11.1 — Evicción LRU + umbral de VRAM

Añadir al `model_manager` un WatchDog que monitorice la VRAM real vía `gpu_manager` y
evicte automáticamente modelos WARM por LRU cuando la VRAM usada supere un umbral
configurable (`vram_eviction_threshold`, default 0.90). Antes de cargar un modelo nuevo,
comprobar si hay espacio suficiente y evictar preventivamente si no lo hay.

| Item | Descripción | Estado |
|------|-------------|--------|
| Setting `vram_eviction_threshold` | Configurable en `config.py` y `server_config` | Pendiente |
| LRU tracking en `model_manager` | Timestamp de último uso por modelo cargado | Pendiente |
| Pre-load VRAM check | Antes de `load()`, consultar VRAM disponible y evictar si falta | Pendiente |
| Background watchdog | Loop async que monitoriza VRAM y evicta si se supera el umbral | Pendiente |
| Tests | Evicción LRU, umbral, pre-load check | Pendiente |

### 11.2 — Aislamiento por proceso (gRPC/subprocess)

Formalizar el patrón de ejecución de backends como subprocesos independientes con
health checks y restart automático on crash. Un fallo en un backend (OOM CUDA, segfault)
no debe afectar al servidor principal ni a otros modelos cargados.

| Item | Descripción | Estado |
|------|-------------|--------|
| `BackendProcessManager` | Clase que gestiona el ciclo de vida de subprocesos backend | Pendiente |
| Health check periódico | Ping/heartbeat a cada worker, marcado ERROR si no responde | Pendiente |
| Auto-restart on crash | Reinicio automático con backoff exponencial | Pendiente |
| Crash isolation | Fallo de un backend no afecta a otros ni al servidor principal | Pendiente |
| Migrar backends existentes | Adaptar vLLM, Diffusers, Whisper, TTS al nuevo patrón | Pendiente |
| Tests | Crash recovery, health check failure, restart backoff | Pendiente |

### 11.3 — Interfaz unificada multi-modal en BackendInterface

Extender `BackendInterface` con métodos opcionales por modalidad (`generate_text`,
`generate_image`, `transcribe`, `synthesize_speech`, `embed`, `rerank`) y un
`get_capabilities()` que declare qué soporta cada backend. Simplifica el routing
desde las capas API (OpenAI / Ollama) sin conocer el tipo de backend.

| Item | Descripción | Estado |
|------|-------------|--------|
| Extender `BackendInterface` | Métodos opcionales por modalidad con `NotImplementedError` default | Pendiente |
| `get_capabilities()` | Retorna set de capacidades (`text_generation`, `embedding`, `tts`, etc.) | Pendiente |
| Routing por capability | Las capas API consultan capabilities en lugar de tipo de backend | Pendiente |
| Migrar backends existentes | Cada backend implementa solo los métodos de su modalidad | Pendiente |
| Tests | Capabilities reporting, routing, fallback en método no soportado | Pendiente |

### 11.4 — Busy timeout / health watchdog

Protección contra backends colgados. Si una inferencia supera un timeout configurable
por backend, marcar el modelo como ERROR, cancelar la request y reiniciar el worker.

| Item | Descripción | Estado |
|------|-------------|--------|
| Setting `busy_timeout_seconds` | Configurable por backend en `config.py` (default: 300s) | Pendiente |
| Request timeout tracking | Registrar inicio de cada request activa por worker | Pendiente |
| Watchdog loop | Async loop que detecta requests que exceden el timeout | Pendiente |
| Acción on timeout | Marcar modelo ERROR, cancelar request, reiniciar worker | Pendiente |
| Métricas | Contador de timeouts por modelo/backend en stats | Pendiente |
| Tests | Timeout detection, model ERROR transition, worker restart | Pendiente |

---

## Orden de ejecución sugerido

```
[✅ Hecho]  Auth + gateway + settings persistence
[✅ Hecho]  Bloque 2 — Quick fixes (B6, A4, M7, WS system_alert)
[✅ Hecho]  Bloque 3 — Settings en BD
[✅ Hecho]  Bloque 4 — Langfuse
[✅ Hecho]  Bloque 5 — Tests (paths críticos cubiertos; faltan e2e load/unload y TRT-LLM compile)
[✅ Hecho]  Bloque 7 — Teams, stats, admin UX
[✅ Hecho]  Bloque 8 — Voice Pipeline (Fase 1 + 1.5 + 2)

[En progreso] Bloque 9 — Model Profiles (Fase 6)
[En progreso] Bloque 10 — Chatterbox TTS (Fase 7 parcial)
[Pendiente]   Bloque 11 — Resiliencia de backends y gestión avanzada de recursos
[Pendiente]   Validación manual TRT-LLM multi-engine en producción
```
