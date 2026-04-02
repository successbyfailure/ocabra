# Auditoría de Documentación vs. Código — oCabra

Generado: 2026-04-02

Este documento compara `docs/PLAN.md`, `docs/CONTRACTS.md`, `docs/CONVENTIONS.md` y `AGENTS.md`
con el código real para identificar desalineaciones, funcionalidad no documentada e implementaciones faltantes.

## Estado tras la alineación parcial (2026-04-02)

Ya quedaron reflejados en documentación o código estable:
- `BackendCapabilities` con `pooling`, `rerank`, `classification`, `score` y `music_generation`
- `acestep` como backend first-class y en los tipos frontend
- `WorkerPool` con firmas reales síncronas para registro/gestión in-memory y helpers de forwarding
- `/v1/models` con extensión `ocabra` real
- `ServiceState` con `ui_url`, `start`, `runtime_loaded_when_alive` y campos runtime reales
- `downloadDir`/`maxTemperatureC` como overrides en memoria, `modelsDir` como valor inmutable de entorno y `globalSchedules` persistido en `eviction_schedules`
- `backend/ocabra/api/internal/gpus.py` y stats con respuestas actuales de listas planas / agregadas
- Stack real: frontend servido por Nginx; Caddy como reverse proxy
- Fase 0 y Fase 5 marcadas como completadas en `docs/PLAN.md`

Pendiente de cambios de código:
- Validación backend de `pytest` en un entorno con dependencias completas
- Validación productiva final de TensorRT-LLM con engines reales y toolchain objetivo

El resto del fichero conserva el snapshot de auditoría anterior para trazabilidad, pero esta sección es la que debe considerarse viva.

---

## A) Snapshot histórico previo a la alineación parcial (se conserva para trazabilidad)

### A1. `BackendCapabilities` tiene campos extra no documentados en CONTRACTS.md
- **CONTRACTS.md** define 11 booleans + `context_length` para `BackendCapabilities`.
- **Código real** (`backends/base.py:8-24`) añade 5 campos más: `pooling`, `rerank`, `classification`, `score`, `music_generation`.
- **Frontend** (`types/index.ts:46-62`) incluye `pooling`, `rerank`, `classification`, `score` pero **no** tiene `musicGeneration`.

### A2. `WorkerInfo.backend_type` incompleto en CONTRACTS.md
- **CONTRACTS.md** lista: `vllm | bitnet | diffusers | whisper | tts | ollama | llama_cpp | sglang | tensorrt_llm`
- **Código real:** `"acestep"` está registrado en `main.py:265` pero no está en el contrato ni en `KNOWN_BACKEND_TYPES`.

### A3. Métodos de `WorkerPool` con firmas erróneas en CONTRACTS.md
- `register_backend`, `release_port`, `get_worker` están documentados como `async def`.
- **Código real** (`core/worker_pool.py`): los tres son `def` síncronos.
- Métodos **no documentados** en CONTRACTS.md: `register_disabled_backend()`, `get_backend()`, `set_worker()`, `remove_worker()`, `forward_request()`, `forward_stream()`.

### A4. Endpoint GPU stats devuelve formato diferente al documentado
- **CONTRACTS.md sec. 5.2** documenta `GET /ocabra/gpus/{index}/stats` devolviendo `GPUStatHistory` (objeto con `gpuIndex`, `window`, `points`).
- **Código real** (`api/internal/gpus.py:37-68`): devuelve una lista plana con claves snake_case (`recorded_at`, `utilization_pct`, `vram_used_mb`, `power_draw_w`, `temperature_c`). Sin wrapper.

### A5. Respuesta de `unload` difiere del contrato
- **CONTRACTS.md sec. 5.1** documenta `POST /ocabra/models/{id}/unload` → `{"ok": true}`.
- **Código real** (`api/internal/models.py:120-129`): devuelve `ModelState.to_dict()` completo.

### A6. Respuesta de `DELETE /ocabra/models/{id}` tiene campo extra
- **CONTRACTS.md** documenta `{"ok": true}`.
- **Código real** (`models.py:169`): `{"ok": True, "deleted_path": deleted_path}`.

### A7. Formato de error sin campo `code`
- **CONTRACTS.md sec. 5** especifica errores como `{"detail": str, "code": str}`.
- **Código real**: usa `HTTPException` de FastAPI → `{"detail": str}` sin `"code"` en la mayoría de endpoints.

### A8. `/v1/models` usa subobject `"ocabra"` con nombres distintos a los documentados
- **PLAN.md:244-264** documenta campos top-level: `capabilities`, `gpu_assignment`, `status`, `vram_used_mb`, `pin`.
- **Código real** (`api/openai/models.py`): los campos van dentro de `"ocabra": { ... }` con `gpu` (no `gpu_assignment`), `load_policy` (no `pin`), y añade `display_name`, `backend_model_id`.

### A9. `ServiceState` usa `ui_url` en vez del documentado `ui_base_path`
- **CONTRACTS.md sec. 5.7** define `ui_base_path: str`.
- **Código real** (`core/service_manager.py:24,66`): el campo es `ui_url`. El nombre `ui_base_path` no existe.

### A10. `ServiceState` tiene campos extra no documentados
- Campos en código sin documentar: `idle_action`, `runtime_check_path`, `runtime_check_key`, `runtime_check_model_key`, `unload_method`, `unload_payload`, `post_unload_flush_path`, `docker_container_name`, `runtime_loaded_when_alive`.

### A11. Frontend type `ServerConfig` falta `idleEvictionCheckIntervalSeconds`
- **CONTRACTS.md sec. 5.6** lista este campo.
- **Backend** lo devuelve y acepta correctamente (`config.py:31,61,93-94`).
- **Frontend** (`types/index.ts:274`): el campo no está definido.

### A12. `downloadDir` y `maxTemperatureC` no son settings reales
- **CONTRACTS.md** los lista como campos `ServerConfig`.
- **Código real**: no existen en `config.py`. Son overrides en memoria (`config_overrides`) que se pierden al reiniciar. `downloadDir` default hardcoded a `f"{settings.models_dir}/downloads"`, `maxTemperatureC` a `88`.

### A13. Body de `POST /ocabra/downloads` difiere del contrato
- **CONTRACTS.md sec. 5.3** documenta `{ source: "huggingface"|"ollama", model_ref: str }`.
- **Código real** (`api/internal/downloads.py:25`): acepta también `"bitnet"` como source, y campos adicionales `artifact` y `register_config`.

---

## B) Docs describen X, el código no lo implementa (implementaciones faltantes)

### B1. `POST /ocabra/models/{id}/pin` no existe
- **CONTRACTS.md sec. 5.1** lo documenta.
- No hay tal endpoint. El pin se gestiona via `PATCH /ocabra/models/{id}` con `load_policy: "pin"`.

### B2. `POST /ocabra/services/{service_id}/touch` no existe
- **CONTRACTS.md sec. 5.7** lo documenta.
- El fichero `api/internal/services.py` define: `list`, `get`, `patch`, `refresh`, `patch_runtime`, `start`, `unload`. Sin `touch`.

### B3. `docker-compose.dev.yml` no existe
- **PLAN.md:121** lo lista en el árbol de directorios.

### B4. `statsStore.ts` no existe
- **PLAN.md:220** y **CONVENTIONS.md:141** documentan un Zustand store `statsStore`.
- Los stores existentes en `frontend/src/stores/`: `downloadStore.ts`, `gpuStore.ts`, `modelStore.ts`, `serviceStore.ts`. Sin `statsStore`.

### B5. `system_alert` WebSocket event nunca se emite
- **CONTRACTS.md sec. 5.8** y `types/index.ts:426` lo definen.
- Ningún código backend publica este evento.

### B6. `.prettierrc` no existe
- **CONVENTIONS.md:89** especifica config Prettier en `frontend/.prettierrc`.
- El fichero no existe en el frontend.

### B7. `eviction_schedules.cron_expr` vs. formato real de la API desconectados
- **CONTRACTS.md sec. 7** define la tabla con `cron_expr TEXT NOT NULL`.
- **API schema** (`api/internal/config.py:18-23`) expone `days: list[int]`, `start: str`, `end: str` y el runtime lo convierte a dos filas por ventana global (`evict_all`/`reload`) cuando persiste `globalSchedules`.

---

## C) El código implementa X, los docs no lo mencionan (funcionalidad sin documentar)

| Funcionalidad | Ubicación |
|--------------|-----------|
| Backend `acestep` completo (music generation) | `backends/acestep_backend.py`, `config.py:167-183` |
| Backend `bitnet` completo | `backends/bitnet_backend.py`, `registry/bitnet_registry.py` |
| `ollama` como backend first-class (no solo API compat) | `backends/ollama_backend.py`, `main.py` |
| Sistema de compilación TRT-LLM + UI | `core/trtllm_compile_manager.py`, `api/internal/trtllm.py`, `pages/TrtllmEngines.tsx` |
| Endpoints de pooling/rerank/classification/score | `api/openai/pooling.py` → `/v1/pooling`, `/v1/score`, `/v1/rerank`, `/v1/classify` |
| `POST /ocabra/services/{id}/start` | `api/internal/services.py:82` |
| `DELETE /api/delete` (Ollama compat) | `api/ollama/delete.py` |
| `POST /api/embed` (Ollama compat, adicional a `/api/embeddings`) | `api/ollama/embeddings.py` |
| `/health` y `/ready` endpoints | `api/health.py` (PLAN.md los marca como `- [ ]`) |
| Prometheus `/metrics` | `api/metrics.py` |
| `GET /ocabra/models/storage` | `api/internal/models.py:50` |
| `serviceStore.ts` Zustand store | `frontend/src/stores/serviceStore.ts` |
| `vllm_recipes.py` (configuración basada en recetas) | `backends/vllm_recipes.py` |
| `core/model_ref.py` (ID canónico de modelo) | `core/model_ref.py` — crítico, sin documentar |
| Soporte de variantes de diarización | `core/model_manager.py:89-100` |
| `GET /ocabra/registry/hf/{repo_id}/variants` | `api/internal/registry.py:37` |
| `downloadStore.ts` Zustand store | `frontend/src/stores/downloadStore.ts` |

---

## D) PLAN.md — checkboxes y estado desactualizados

### D1. Fase 0 toda sin marcar pero está implementada
- `PLAN.md:322-335` muestra todos los items de Fase 0 como `- [ ]`.
- El propio texto del PLAN (línea 42-43) confirma que Fases 0-4 están implementadas.

### D2. Fase 5 — items marcados como pendientes pero ya implementados
| Checkbox | Estado real |
|----------|------------|
| `- [ ] Healthcheck endpoints (/health, /ready)` | **Implementado** en `api/health.py` |
| `- [ ] LiteLLM auto-sync` | **Implementado** en `integrations/litellm_sync.py` |
| `- [ ] Logging estructurado (structlog)` | **Implementado** en `main.py:14-36` |

### D3. Árbol de directorios en PLAN.md severamente desactualizado
Ficheros/directorios existentes no listados (selección):
- `core/model_ref.py`, `core/service_manager.py`, `core/trtllm_compile_manager.py`
- `api/internal/trtllm.py`
- `backends/ollama_backend.py`, `backends/bitnet_backend.py`, `backends/acestep_backend.py`, `backends/vllm_recipes.py`
- `api/openai/pooling.py`, `api/openai/audio.py`, `api/ollama/delete.py`
- `schemas/` (directorio completo)
- `frontend/src/stores/downloadStore.ts`, `serviceStore.ts`
- `frontend/src/pages/TrtllmEngines.tsx`
- `frontend/src/components/settings/`, `components/common/`

### D4. Stack incorrecto — APScheduler vs asyncio
- **PLAN.md** hace referencia a APScheduler para eviction scheduling.
- **Código real**: loops `asyncio` planos en `main.py`. APScheduler no es dependencia del proyecto.

### D5. Stack incorrecto — Caddy vs Nginx
- **PLAN.md:38** dice "Caddy (interno, sirve API + frontend)".
- **Código real**: Nginx (`frontend/nginx.conf`, `frontend/Dockerfile`).

---

## E) Violaciones de CONVENTIONS.md encontradas en el código

### E1. Schemas Pydantic no están en `ocabra/schemas/`
- **CONVENTIONS.md:69** exige schemas en `ocabra/schemas/`.
- **Código real**: la mayoría de schemas están inline en los ficheros de endpoint:
  - `ModelPatch`, `AddModelRequest` → `api/internal/models.py`
  - `ServerConfigPatch`, `EvictionSchedulePayload` → `api/internal/config.py`
  - `DownloadCreateRequest` → `api/internal/downloads.py`
  - `ServiceRuntimePatch`, `ServicePatch` → `api/internal/services.py`
  - `DeleteRequest` → `api/ollama/delete.py`
- Solo `ocabra/schemas/registry.py` sigue la convención.

### E2. Nombres de schemas Pydantic no siguen la convención
- **CONVENTIONS.md:75-76**: request schemas deben ser `*Request` o `*Create/*Update`.
- Nombres reales: `ModelPatch`, `ServerConfigPatch`, `ServicePatch`, `EvictionSchedulePayload`.

### E3. Uso extensivo de `Any` fuera de código de proxy
- **CONVENTIONS.md:9**: "`Any` solo en código de proxy/reenvío".
- `Any` se importa en 27+ ficheros backend, incluyendo `service_manager.py`, `config.py`, `registry.py`.

### E4. GPU stats devuelve snake_case en vez de camelCase
- La convención de API (CONTRACTS.md, frontend types) usa camelCase.
- `api/internal/gpus.py:58-67` devuelve `utilization_pct`, `temperature_c`, `power_draw_w`, `vram_used_mb`, `recorded_at` en snake_case.

### E5. `ModelCapabilities` en frontend falta `musicGeneration`
- `backends/base.py:22`: `music_generation: bool = False`.
- `frontend/src/types/index.ts:46-62`: no incluye `musicGeneration`.

### E6. `to_dict()` manual en vez de `dataclasses.asdict()`
- `GPUState` usa `asdict()` correctamente.
- `BackendCapabilities` (`backends/base.py:26-44`) y `ModelState` (`core/model_manager.py:66-85`) usan `to_dict()` manual. Inconsistente y propenso a desincronizarse.

## Backlog residual real

- `SEC-1`: autenticación administrativa en `/ocabra/*`
- Validación productiva final de TensorRT-LLM con engines reales y toolchain CUDA/NVIDIA objetivo
- Batería backend de `pytest` en entorno completo de CI o contenedor
- Limpieza estructural adicional de módulos inflados y código muerto restante si se prioriza deuda técnica sobre nuevas features

---

## Resumen por impacto

### Contratos rotos (afectan integraciones y frontend)
- **A4** — GPU stats formato completamente diferente
- **A3** — WorkerPool métodos async/sync incorrectos en docs
- **A7** — Campo `code` en errores nunca presente
- **A9** — `ui_base_path` vs `ui_url` en ServiceState
- **A11** — `idleEvictionCheckIntervalSeconds` sin tipo en frontend

### Funcionalidad invisible (sin documentar)
- **C1-C3** — Tres backends enteros (`acestep`, `bitnet`, `ollama-as-backend`) sin documentar
- **C4** — Sistema TRT-LLM completo sin documentar en CONTRACTS.md
- **C5** — Endpoints pooling/rerank/score/classify sin documentar
- **C14** — `model_ref.py` (módulo crítico de IDs canónicos) sin documentar

### Docs obsoletas (crean confusión)
- **D3** — Árbol de directorios en PLAN.md desactualizado (~20 ficheros/dirs faltantes)
- **D1/D2** — Checkboxes de fases sin actualizar
- **D4/D5** — Stack (APScheduler, Caddy) diferente al real (asyncio, Nginx)

### Convenciones sistemáticamente ignoradas
- **E1/E2** — Schemas Pydantic inline en endpoints en vez de en `ocabra/schemas/`
