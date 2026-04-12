# oCabra — Roadmap

Última actualización: 2026-04-12

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

## ✅ Bloque 9 — Model Profiles (Fase 6) (COMPLETADO)

Plan en `docs/tasks/model-profiles-chatterbox-swarm-plan.md`.

Implementado:
- `ModelProfile` DB + migración `0010`, `profile_registry.py` con CRUD y cache
- Endpoints REST `/ocabra/profiles/*` con upload de assets
- `resolve_profile()` en `/v1/*` y `/api/*` con fallback legacy configurable
- Worker sharing por `(base_model_id, load_overrides_hash)`
- UI de perfiles en Models (`ProfileModal.tsx`)
- Validación de `voice_ref` restringida a paths controlados
- Detección correcta de backends en local scanner

---

## ✅ Bloque 10 — Chatterbox TTS (Fase 7 parcial) (COMPLETADO)

Plan en `docs/tasks/model-profiles-chatterbox-swarm-plan.md`.

Implementado:
- `chatterbox_backend.py` — BackendInterface completo (304 líneas)
- `chatterbox_worker.py` — Worker FastAPI con synthesize/stream/voices (610 líneas)
- Registro en `main.py`, config con `chatterbox_python_bin`
- Detección en HuggingFace registry y local scanner
- Voice cloning via `voice_ref` con path traversal protection
- 28 idiomas, mapeo de voces OpenAI, formatos múltiples

---

## ✅ Bloque 11 — Resiliencia de backends (COMPLETADO)

Plan en `docs/tasks/backend-resilience-plan.md`.

### 11.1 — Evicción LRU + umbral de VRAM ✅
- `vram_eviction_threshold` en config, `_get_eviction_candidates()`, `_evict_for_space()`
- Pre-load VRAM check con evicción preventiva, `_vram_watchdog()` background loop
- 7 tests

### 11.2 — Aislamiento por proceso ✅
- `BackendProcessManager` con health loop, detección de PID muerto, auto-restart con backoff exponencial
- Settings: `worker_health_check_interval_seconds`, `auto_restart_workers`, `max_worker_restarts`
- Integrado en `main.py` lifespan. 7 tests

### 11.3 — Interfaz unificada multi-modal ✅
- `ModalityType` enum, métodos opcionales tipados en `BackendInterface`
- `supported_modalities()` en los 13 backends
- Helpers `get_backends_for_modality()` y `supports_modality()` en WorkerPool. 17 tests

### 11.4 — Busy timeout / health watchdog ✅
- `ActiveRequest` dataclass, `begin_request()` retorna request_id
- `_busy_watchdog()` loop, `_timeout_counts` metrics
- `busy_timeout_seconds` y `busy_timeout_action` configurables. 10 tests

---

## ✅ Bloque 12 — Federación P2P (COMPLETADO)

Plan en `docs/tasks/federation-plan.md`.

Implementado (5 fases):
- **Fase 1**: `FederationManager` con heartbeat, DB `federation_peers`, cifrado Fernet, CRUD endpoints, circuit breaker
- **Fase 2**: Proxy transparente (`proxy_request`/`proxy_stream`), `resolve_federated()`, load balancing con bias local, hooks en todos los endpoints OpenAI y Ollama
- **Fase 3**: Inventario federado en `/v1/models`, `/api/tags`, `/ocabra/models` con deduplicación
- **Fase 4**: UI completa — tab Federation en Settings, badges remotos en Models, panel en Dashboard, WebSocket events
- **Fase 5**: Operaciones remotas (load/unload/download/GPUs) para peers con `access_level=full`
- 54 tests de federación

---

## Pendiente — Bloque 13 — Backends Modulares

Plan en `docs/tasks/modular-backends-plan.md`.

Cada backend instalable/desinstalable en runtime desde la UI. Imagen Docker slim + distribución OCI.
5 fases: infraestructura → migrar backends → OCI → imagen slim → frontend.

---

## Orden de ejecución

```
[✅ Hecho]  Auth + gateway + settings persistence
[✅ Hecho]  Bloque 2 — Quick fixes
[✅ Hecho]  Bloque 3 — Settings en BD
[✅ Hecho]  Bloque 4 — Langfuse
[✅ Hecho]  Bloque 5 — Tests
[✅ Hecho]  Bloque 7 — Teams, stats, admin UX
[✅ Hecho]  Bloque 8 — Voice Pipeline
[✅ Hecho]  Bloque 9 — Model Profiles (Fase 6)
[✅ Hecho]  Bloque 10 — Chatterbox TTS (Fase 7 parcial)
[✅ Hecho]  Bloque 11 — Resiliencia de backends
[✅ Hecho]  Bloque 12 — Federación P2P
[Pendiente]   Bloque 13 — Backends Modulares
[Pendiente]   Validación manual TRT-LLM multi-engine en producción
```
