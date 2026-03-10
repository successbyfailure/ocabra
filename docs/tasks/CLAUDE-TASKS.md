# Tareas para Claude

Antes de empezar cualquier tarea: lee `CLAUDE.md`, `docs/PLAN.md` y `docs/CONTRACTS.md`.

---

## TAREA C-1 — Fase 0: Fundación

**Estado:** DISPONIBLE AHORA
**Rama:** `feat/0-foundation`
**Briefing completo:** `docs/agents/phase-0-foundation.md`

### Cuándo entrar
Inmediatamente. Es la primera tarea. No tiene dependencias.

### Qué hacer

1. Crear estructura de directorios completa según el briefing.
2. Crear `docker-compose.yml` con servicios: api, frontend, postgres, redis, caddy.
3. Crear `docker-compose.dev.yml` con overrides para desarrollo (volumes, reload).
4. Crear `.env.example` con TODAS las variables definidas en `docs/CONTRACTS.md §8`.
5. Crear `backend/Dockerfile` — imagen base `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`.
6. Crear `backend/pyproject.toml` con las dependencias del briefing.
7. Crear `backend/ocabra/main.py` — FastAPI app con `lifespan`, endpoint `GET /health`.
8. Crear `backend/ocabra/config.py` — pydantic-settings con todas las vars de entorno.
9. Crear `backend/ocabra/database.py` — SQLAlchemy async engine + session factory.
10. Crear `backend/ocabra/redis_client.py` — cliente async + helper pub/sub.
11. Crear modelos SQLAlchemy en `backend/ocabra/db/` — las 4 tablas de `docs/CONTRACTS.md §7`.
12. Configurar Alembic: `alembic.ini`, `alembic/env.py`, primera migración con las 4 tablas.
13. Crear `frontend/Dockerfile` — multi-stage: node build + nginx serve.
14. Crear `frontend/package.json` con dependencias del briefing.
15. Crear `frontend/vite.config.ts`, `tsconfig.json`, `tailwind.config.ts`.
16. Crear `frontend/src/App.tsx` — React Router con 6 rutas (Dashboard, Models, Explore, Playground, Stats, Settings).
17. Crear `frontend/src/components/layout/Layout.tsx` y `Sidebar.tsx` — layout con sidebar de navegación.
18. Crear páginas stub vacías para las 6 rutas.
19. Crear `caddy/Caddyfile` — proxy: `/v1/*`, `/api/*`, `/ocabra/*` → `api:8000`, resto → `frontend:80`.
20. Verificar que `docker compose up` levanta sin errores y `GET /health` retorna 200.

### Señal de finalización
- Rama `feat/0-foundation` con commit `feat(foundation): complete phase 0 scaffold`.
- Actualizar `docs/agents/phase-0-foundation.md` → marcar `[x] Completado`.

---

## TAREA C-2 — Stream 1-A: GPU Manager & Scheduler

**Estado:** ESPERAR a que C-1 esté mergeado en `main`
**Rama:** `feat/1-A-gpu-manager`
**Briefing completo:** `docs/agents/stream-1A-gpu-manager.md`

### Cuándo entrar
Cuando la rama `feat/0-foundation` esté mergeada en `main` (ver que `docs/agents/phase-0-foundation.md` tiene `[x] Completado`).

### Qué hacer

1. Implementar `backend/ocabra/core/gpu_manager.py`:
   - Clase `GPUManager` con `start()`, `stop()`, polling pynvml cada 2s.
   - Publica estado en Redis canal `gpu:stats` y key `gpu:state:{index}`.
   - Métodos `lock_vram()` y `unlock_vram()` para contabilizar VRAM por modelo.
   - `get_free_vram()` = total - used - locked - buffer 512 MB.

2. Implementar `backend/ocabra/core/scheduler.py`:
   - Clase `GPUScheduler` con lógica de asignación (preferred → fallback → tensor parallel).
   - Método `get_eviction_candidates()` — orden: on_demand idle > on_demand reciente > warm > pin.
   - Método `check_schedule_evictions()` — lee schedules de BD, descarga modelos en ventana activa.
   - Método `check_schedule_reloads()` — recarga modelos al salir de ventana.
   - Integración con APScheduler: tareas periódicas cada 30s.

3. Implementar `backend/ocabra/api/internal/gpus.py`:
   - `GET /ocabra/gpus` — lista GPUState de todas las GPUs.
   - `GET /ocabra/gpus/{index}` — estado de una GPU.
   - `GET /ocabra/gpus/{index}/stats?window=5m|1h|24h` — histórico de `gpu_stats`.

4. Agregar periódicamente stats de GPU a la tabla `gpu_stats` (cada 60s, promedio de los últimos 30 polls).

5. Registrar el router en `backend/ocabra/main.py` en la sección `# ROUTERS`.

6. Escribir tests en `backend/tests/test_gpu_manager.py` y `test_scheduler.py`:
   - Mock de pynvml.
   - Test de asignación con VRAM insuficiente en GPU preferida.
   - Test de orden de evicción.
   - Test de schedule (mock APScheduler).

### Señal de finalización
- Rama `feat/1-A-gpu-manager` mergeada.
- `docs/agents/stream-1A-gpu-manager.md` → `[x] Completado`.

---

## TAREA C-3 — Stream 1-B: Model Manager & Worker Pool

**Estado:** ESPERAR a que C-1 esté mergeado
**Rama:** `feat/1-B-model-manager`
**Briefing completo:** `docs/agents/stream-1B-model-manager.md`

### Cuándo entrar
Cuando `feat/0-foundation` esté mergeada. **Puede correr en paralelo con C-2.**

### Qué hacer

1. Crear `backend/ocabra/backends/base.py` — `BackendInterface` abstracta, `BackendCapabilities`, `WorkerInfo` (exactamente según `docs/CONTRACTS.md §1`).

2. Implementar `backend/ocabra/core/worker_pool.py`:
   - Registro de backends por tipo.
   - Asignación de puertos del rango configurado (sin colisiones).
   - Proxy de requests a workers via httpx.

3. Implementar `backend/ocabra/core/model_manager.py`:
   - Máquina de estados (DISCOVERED → LOADING → LOADED → UNLOADING → UNLOADED → ERROR).
   - `start()`: carga automática de modelos con `load_policy=pin` desde BD.
   - `load()`: llama al GPUScheduler (si no está listo aún, usa un mock), lanza el backend, actualiza estado, publica evento en Redis.
   - `unload()`: llama al backend, actualiza estado, libera VRAM en GPUManager.
   - `on_request()`: actualiza `last_request_at`; si UNLOADED, carga bajo demanda.
   - `check_idle_evictions()`: descarga modelos `on_demand` con idle > timeout.
   - `check_vram_pressure()`: si VRAM libre < umbral, evicta candidatos.
   - Auto-reload: si `auto_reload=true` y el modelo fue descargado por presión, recargarlo cuando haya VRAM suficiente.

4. Crear `MockBackend(BackendInterface)` en `backend/tests/` — simula load/unload sin procesos reales.

5. Implementar `backend/ocabra/api/internal/models.py`:
   - `GET /ocabra/models`, `GET /ocabra/models/{model_id}`
   - `POST /ocabra/models/{model_id}/load`
   - `POST /ocabra/models/{model_id}/unload`
   - `PATCH /ocabra/models/{model_id}`
   - `DELETE /ocabra/models/{model_id}`

6. Escribir tests en `backend/tests/test_model_manager.py` y `test_worker_pool.py`:
   - Ciclo completo con MockBackend.
   - Evicción por idle.
   - Auto-reload tras evicción.
   - WorkerPool: asignación de puertos sin colisiones.

### Señal de finalización
- Rama `feat/1-B-model-manager` mergeada.
- `docs/agents/stream-1B-model-manager.md` → `[x] Completado`.

---

## TAREA C-4 — Stream 2-A: vLLM Backend

**Estado:** ESPERAR a que C-3 esté mergeado
**Rama:** `feat/2-A-vllm`
**Briefing completo:** `docs/agents/stream-2A-vllm.md`

### Cuándo entrar
Cuando `feat/1-B-model-manager` esté mergeada (BackendInterface disponible).

### Qué hacer

1. Implementar `backend/ocabra/backends/vllm_backend.py`:
   - `load()`: lanza `vllm.entrypoints.openai.api_server` como subproceso con `CUDA_VISIBLE_DEVICES`, espera healthcheck hasta 120s.
   - `unload()`: SIGTERM → espera 30s → SIGKILL.
   - `health_check()`: GET al endpoint `/health` del proceso.
   - `get_capabilities()`: lee `config.json` del modelo, infiere capabilities por arquitectura (tabla en el briefing).
   - `get_vram_estimate_mb()`: suma safetensors × 1.2.
   - `forward_request()` y `forward_stream()`: proxy httpx.

2. Crear `workers/vllm_worker.py` — wrapper standalone con gestión de señales y healthcheck extendido.

3. Manejar errores críticos: OOM (código de salida 137), proceso caído inesperadamente → notificar al ModelManager → marcar ERROR → auto-reload si procede.

4. Escribir tests en `backend/tests/test_vllm_backend.py`:
   - Mock de subprocess: load/unload sin vLLM real.
   - Detección de capacidades con fixtures de config.json.
   - Manejo de OOM.

### Señal de finalización
- Rama `feat/2-A-vllm` mergeada.
- `docs/agents/stream-2A-vllm.md` → `[x] Completado`.

---

## TAREA C-5 — Stream 3-A: OpenAI API + Stats Middleware

**Estado:** ESPERAR a que C-4 esté mergeado
**Rama:** `feat/3-A-openai-api`
**Briefing completo:** `docs/agents/stream-3A-openai-api.md`

### Cuándo entrar
Cuando `feat/2-A-vllm`, `feat/2-B-diffusers` y `feat/2-C-audio` estén mergeados.

### Qué hacer

1. Implementar todos los endpoints de `backend/ocabra/api/openai/`:
   - `GET /v1/models` — con campo `ocabra` extendido.
   - `POST /v1/chat/completions` — streaming SSE, tools, vision, JSON mode.
   - `POST /v1/completions`.
   - `POST /v1/embeddings`.
   - `POST /v1/images/generations`.
   - `POST /v1/audio/transcriptions` (multipart).
   - `POST /v1/audio/speech`.

2. Implementar `ensure_loaded()` — carga bajo demanda con espera asíncrona.

3. Implementar middleware de stats en `backend/ocabra/stats/collector.py`:
   - Wrappea todos los endpoints `/v1/*`.
   - Registra duración, tokens (del último chunk en streaming), Δ energía.
   - Escribe en tabla `request_stats`.

4. Implementar errores en formato OpenAI estándar.

5. Tests con el SDK oficial de OpenAI apuntando al servidor de test:
   - Chat streaming.
   - Tool calls.
   - Carga bajo demanda.
   - Error de capability incorrecta.

### Señal de finalización
- Rama `feat/3-A-openai-api` mergeada.
- `docs/agents/stream-3A-openai-api.md` → `[x] Completado`.

---

## TAREA C-6 — Stream 5: Integrations & Polish

**Estado:** ESPERAR a que TODO lo anterior esté mergeado
**Rama:** `feat/5-integrations`
**Briefing completo:** `docs/agents/stream-5-integrations.md`

### Cuándo entrar
Cuando los streams 3-A, 3-B y todos los de frontend (4-A/B/C/D) estén mergeados.

### Qué hacer

1. Implementar `backend/ocabra/integrations/litellm_sync.py` — auto-sync al añadir/eliminar modelos.
2. Implementar endpoint Prometheus `GET /metrics`.
3. Configurar `structlog` con JSON en producción y contexto de request.
4. Implementar `GET /ready` con checks de postgres, redis, gpu_manager.
5. Escribir tests de integración end-to-end en `backend/tests/integration/`.
6. Crear `scripts/setup.sh` — script de first-run.
7. Revisar y completar documentación OpenAPI (descripciones de todos los endpoints).

### Señal de finalización
- Rama `feat/5-integrations` mergeada en `main`.
- `docs/agents/stream-5-integrations.md` → `[x] Completado`.
