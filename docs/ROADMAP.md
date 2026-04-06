# oCabra — Roadmap

Última actualización: 2026-04-04

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

## Bloque 2 — Fixes pendientes (pequeños, independientes)

### [B6] `ModelPatch` no puede resetear `preferred_gpu` a `null`

- **Archivo:** `backend/ocabra/api/internal/models.py:150`
- **Fix:** `body.model_dump(exclude_unset=True)` en lugar del filtro `if v is not None`
- Effort: < 30 min

### [A4] `_watch_and_reload` sin check de shutdown del servidor

- **Archivo:** `backend/ocabra/core/model_manager.py:747`
- Ya tiene deadline y guards; falta verificar cierre del servidor para evitar loop en shutdown.
- **Fix:** Añadir `and not self._shutdown_event.is_set()` (o equivalent) al `while`
- Effort: < 30 min

### [M7] `HFModelVariant.backend_type` Literal incompleto

- **Archivo:** `backend/ocabra/schemas/registry.py:89`
- Faltan `llama_cpp`, `sglang`, `tensorrt_llm`, `acestep` en el Literal.
- **Fix:** Ampliar Literal o usar `str` con validador si el tipo ya está en `BackendType`
- Effort: < 30 min

### [WS] `system_alert` WebSocket event nunca emitido

- **Archivo:** `backend/ocabra/api/internal/ws.py` + `frontend/src/types/index.ts:539`
- El tipo está definido en frontend pero el backend nunca lo emite.
- **Fix:** Emitir desde el manejador de excepciones del lifespan o desde el scheduler
  cuando detecte condiciones críticas (GPU sin respuesta, VRAM exhausted, etc.)
- Effort: 1-2h

---

## Bloque 3 — Persistencia de settings en base de datos

**Prioridad: MEDIA**

Actualmente `PATCH /ocabra/config` muta el objeto `settings` en memoria; los cambios
se pierden al reiniciar el contenedor. El fichero `.env` solo establece valores iniciales.

### Objetivo

- Añadir tabla `server_config` en PostgreSQL con pares `key / value / updated_at`.
- Al arrancar, cargar los overrides de la tabla y aplicarlos sobre los valores leídos del `.env`.
- Cada `PATCH /ocabra/config` persiste los campos modificados en esa tabla.
- La tabla actúa como capa de override: `.env` → DB override → valor efectivo.
- Los campos de solo arranque (rutas de binarios, puertos, credenciales de infraestructura)
  se excluyen de la persistencia y siguen leyéndose únicamente del `.env`.

### Alcance estimado

| Archivo | Cambio |
|---------|--------|
| `backend/ocabra/db/server_config.py` | Nuevo modelo SQLAlchemy + helpers `load_overrides()` / `save_override()` |
| `backend/alembic/versions/XXXX_server_config.py` | Migración |
| `backend/ocabra/main.py` | `lifespan`: cargar overrides de DB tras arranque |
| `backend/ocabra/api/internal/config.py` | `patch_config()` llama a `save_override()` en lugar de solo mutar `settings` |

- Effort: 2-3h

---

## Bloque 4 — Integración Langfuse (observabilidad LLM)

**Prioridad: MEDIA — opcional, desactivado por defecto**

Plan detallado con código en `docs/tasks/langfuse-integration-plan.md`.

**Resumen de alcance:**
- `backend/ocabra/integrations/langfuse_tracer.py` — singleton, `trace_generation()`, `wrap_stream()`
- Hook en `WorkerPool.forward_request` y `forward_stream`
- Fix paralelo: tokens en streaming para `stats/collector.py` (independiente de Langfuse)
- Settings `langfuse_*` en `config.py`, variables en `docker-compose.yml` y `.env.example`
- Dependencia opcional `langfuse>=2.0` en `pyproject.toml`
- Tests completos (plan incluye lista de 11 casos)

**Archivos afectados:**
- `backend/ocabra/integrations/langfuse_tracer.py` (nuevo)
- `backend/ocabra/core/worker_pool.py`
- `backend/ocabra/stats/collector.py`
- `backend/ocabra/config.py`
- `backend/pyproject.toml`
- `docker-compose.yml`, `.env.example`

---

## Bloque 5 — Cobertura de tests

Paths críticos sin tests confirmados en el audit:

| Función | Riesgo cubierto |
|---------|----------------|
| `_delete_model_files()` en `models.py` | Path traversal |
| `delete_engine()` en `trtllm.py` | Path traversal |
| `update_config()` en `model_manager.py` | Whitelist de campos |
| `patch_config()` en `config.py` | Cambios de config en runtime |
| Flujos completos load/unload por backend | Regresión de integración |
| Compile job TRT-LLM: mock Docker, fases convert→build | Regresión compilación |

---

## Bloque 6 — Operativo / deuda técnica menor

| Item | Descripción | Effort |
|------|-------------|--------|
| Limpiar entrada `tensorrt_llm/Qwen3-32B-AWQ-fp16` | Registro en BD apunta a engine_dir inexistente; falla con error correcto pero ensucia el inventario | 5 min (SQL directo) |
| Validación TRT-LLM con múltiples engines en producción | Cargar/descargar 2+ engines sin huérfanos ni conflictos de puertos | Manual |
| Documentación OpenAPI enriquecida | Añadir ejemplos y descripciones ricas a endpoints `/ocabra/*` | 2-3h |
| Script first-run / instalación | Guía de puesta en marcha desde cero en nuevo host | 1-2h |

---

## Bloque 7 — Teams, stats ampliadas, admin UX (en implementación)

**Prioridad: ALTA**

### 7.1 Group_id en ApiKeys y RequestStats

- `api_keys`: añadir columna `group_id UUID NULL REFERENCES groups(id) ON DELETE SET NULL`
- `request_stats`: añadir columna `group_id UUID NULL` (sin FK; se conserva aunque se borre el grupo)
- Migración Alembic `0009_apikey_group_request_stat_group.py`
- `UserContext`: añadir `key_group_id: str | None` — se rellena cuando la auth es por API key
- `_record_stat` en `collector.py`: leer `auth_user.key_group_id` y persitirlo en `RequestStat.group_id`
- `CreateApiKeyRequest` (propio, `/ocabra/auth/keys`): añadir campo opcional `group_id`

### 7.2 Admin crea keys para otros usuarios

- **Nuevo endpoint:** `POST /ocabra/users/{user_id}/keys` (requiere `system_admin`)
  - Body: `{ name, expires_in_days?, group_id? }`
  - Response: `{ id, name, key_prefix, key, expires_at, created_at, group_id }` — key mostrada sólo una vez
- **UI Users.tsx:** botón "Crear key" por fila de usuario; modal con nombre, expiración y selector de grupo
- La key creada por el admin puede asignarse a un grupo para que el uso cuente en las stats de ese grupo

### 7.3 Nuevos endpoints de estadísticas

| Endpoint | Rol mínimo | Descripción |
|----------|-----------|-------------|
| `GET /ocabra/stats/recent?limit=20` | model_manager | Últimas N peticiones con info de usuario/grupo |
| `GET /ocabra/stats/by-user?from=&to=` | model_manager | Stats agregadas por usuario |
| `GET /ocabra/stats/by-group?from=&to=` | model_manager | Stats agregadas por grupo |
| `GET /ocabra/stats/my?from=&to=&model_id=` | user | Stats del usuario actual |
| `GET /ocabra/stats/my-group?from=&to=` | user | Stats del grupo del usuario actual |

**Respuestas:**
- `recent`: `{ requests: [{ id, modelId, backendType, requestKind, statusCode, startedAt, durationMs, inputTokens, outputTokens, error, userId, username, groupId, groupName }] }`
- `by-user`: `{ byUser: [{ userId, username, totalRequests, totalErrors, avgDurationMs, totalInputTokens, totalOutputTokens }] }`
- `by-group`: `{ byGroup: [{ groupId, groupName, totalRequests, totalErrors, avgDurationMs, totalInputTokens, totalOutputTokens }] }`
- `my`: mismo shape que `OverviewStats`
- `my-group`: `{ groupId, groupName, stats: OverviewStats }`

### 7.4 Settings en tabs (frontend)

Refactorizar `Settings.tsx` usando `@radix-ui/react-tabs`:

| Tab | Componentes |
|-----|-------------|
| General | GeneralSettings + ApiAccessSettings |
| GPUs | GPUSettings |
| Backends | BackendRuntimeSettings |
| Almacenamiento | StorageSettings + GlobalSchedules |
| LiteLLM | LiteLLMSettings |

### 7.5 Stats en tabs (frontend)

Refactorizar `Stats.tsx` con tabs:

| Tab | Visible para |
|-----|-------------|
| Resumen | todos (overview global para managers; propio para users) |
| Por modelo | managers/admins |
| Por usuario | managers/admins |
| Por grupo | managers/admins |
| Mis stats | todos (stats del usuario actual) |
| Mi grupo | todos (stats del grupo del usuario actual) |
| Log | managers/admins (últimas N peticiones con detalle) |

### 7.6 Dashboard — log de últimas peticiones

Sección al final del Dashboard: tabla con las últimas 20 peticiones.
Poll cada 30 s desde `GET /ocabra/stats/recent?limit=20`.
Columnas: tiempo, modelo, tipo, duración, tokens, usuario, estado.

---

## Bloque 8 — Voice Pipeline (en planificación)

**Prioridad: ALTA**  
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

### Fase 2 — OpenAI Realtime API (`GET /v1/realtime`)

| Item | Archivo | Descripción |
|------|---------|-------------|
| [VP-5] Endpoint WebSocket | `backend/ocabra/api/openai/realtime.py` (nuevo) | `GET /v1/realtime` con upgrade a WS, header `OpenAI-Beta: realtime=v1` |
| [VP-6] RealtimeSession | `backend/ocabra/core/realtime_session.py` (nuevo) | Gestión de sesión, pipeline STT→LLM→TTS, protocolo de eventos |
| [VP-7] VAD servidor | `backend/ocabra/core/vad.py` (nuevo) | SimpleVAD por RMS; interfaz para Silero futuro |
| [VP-8] session.update | En `RealtimeSession` | Configuración de voz, modelo STT, VAD params |

---

## Orden de ejecución sugerido

```
[✅ Hecho]  Auth + gateway + settings persistence

[En curso]  Bloque 7 (teams/stats/UX) — implementado con equipo de agentes
              · Backend: group_id en keys+stats, endpoints nuevos
              · Frontend: settings tabs, stats tabs, dashboard log, admin key UI

[Siguiente] B6 + A4 + M7  ←── quick wins, < 2h en total
            Langfuse       ←── feature autocontenida, plan listo
            Tests batch    ←── paths críticos

            Bloque 8 Voice Pipeline Fase 1  ←── VP-1 bloqueante para Android
            Bloque 8 Voice Pipeline Fase 2  ←── Realtime API WebSocket

[Cuando proceda] WS system_alert · OpenAPI docs · first-run script
```
