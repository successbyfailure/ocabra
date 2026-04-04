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

## Bloque 1 — Auth, usuarios y grupos (bloqueante para exposición pública)

**Prioridad: CRÍTICA**

Plan completo en `docs/tasks/auth-system-plan.md`.

### Resumen

- 3 roles jerárquicos: `user`, `model_manager`, `system_admin`
- Sesión JWT (cookie HTTP-only) para el dashboard; 24h por defecto, 30d con "recordarme"
- API keys por usuario para OpenAI/Ollama (`Authorization: Bearer sk-ocabra-...`)
- Modo sin key configurable por separado para OpenAI y Ollama (anonymous → solo grupo default)
- Grupos de acceso a modelos; los usuarios solo ven modelos de sus grupos en `/v1/models`
- Primer admin desde `.env` (`OCABRA_ADMIN_USER/PASS`, default `ocabra/ocabra`)
- Reemplaza LiteLLM como capa de auth (solo modelos locales)
- 5 fases de implementación + tests

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

## Orden de ejecución sugerido

```
[Ahora]     SEC-1 (auth)  ←── bloqueante para exponer en red
            B6 + A4 + M7  ←── quick wins, < 2h en total

[Siguiente] Settings DB    ←── .env solo como valor inicial, overrides en BD
            Langfuse       ←── feature autocontenida, plan listo
            Tests batch    ←── paths críticos

[Cuando proceda] WS system_alert · OpenAPI docs · first-run script
```
