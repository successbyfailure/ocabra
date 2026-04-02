# Plan Operativo Multi-Agente — REFACTOR + DOC_AUDIT

Generado: 2026-04-02
Fuente: `docs/REFACTOR_PLAN.md`, `docs/DOC_AUDIT.md`

## Objetivo

Convertir los hallazgos de refactor, seguridad y desalineación documental en un plan ejecutable por varios agentes en paralelo, respetando ownership de streams y minimizando conflictos sobre ficheros compartidos.

## Principios de ejecución

- No tocar ficheros fuera del stream asignado salvo coordinación explícita.
- Los cambios de contratos en `docs/CONTRACTS.md` se hacen antes o junto al primer PR que dependa de ellos.
- Los cambios de seguridad van antes que los refactors cosméticos.
- Todo cambio de comportamiento debe ir con test o ampliación de cobertura.
- `backend/ocabra/main.py`, `frontend/src/App.tsx`, `frontend/src/api/client.ts` y `docs/CONTRACTS.md` requieren ventana de integración coordinada.

## Lectura mínima para cada agente

- `CLAUDE.md`
- `AGENTS.md`
- `docs/CONTRACTS.md`
- `docs/CONVENTIONS.md`
- Este documento
- Su briefing de stream en `docs/agents/`

## Clasificación de trabajo

### Bloque 0 — Decisiones de arquitectura obligatorias

Sin estas decisiones no conviene lanzar los bloques críticos de seguridad ni la actualización documental profunda.

1. Autenticación administrativa:
   - Opción mínima: API key para `/ocabra/*`
   - Opción media: API key para admin + allowlist de orígenes para frontend
   - Opción fuerte: JWT o mTLS

2. Política de CORS:
   - Lista cerrada de orígenes
   - Modo dev vs prod

3. Fuente de verdad de configuración:
   - Confirmar qué campos son persistentes en DB
   - Confirmar qué campos son solo env y no mutables en runtime

4. Estrategia documental:
   - ¿Documentar el código actual y luego refactorizar?
   - ¿O aprovechar para alinear contratos con cambios de código en la misma oleada?

## Asignación propuesta por agentes

### Agente A — Seguridad transversal

Ownership principal:
- `backend/ocabra/main.py`
- `backend/ocabra/api/internal/*.py` solo en lo relativo a auth/guardas
- `backend/tests/` de seguridad

Tareas:
- `SEC-1` autenticación en `/ocabra/*`
- `SEC-2` CORS restringido
- `SEC-3` path traversal en borrado de modelos
- `SEC-4` path traversal en borrado de engines TRT-LLM
- `SEC-6` impedir mutación insegura de `models_dir`

Notas:
- Toca ficheros de varios streams. Debe actuar como tiger team con coordinación explícita.
- No debe mezclar refactors de estilo ni limpieza documental.

### Agente B — Core runtime / model lifecycle

Ownership principal:
- `backend/ocabra/core/model_manager.py`
- `backend/ocabra/core/worker_pool.py`
- `backend/ocabra/backends/base.py`
- `backend/tests/test_model_manager.py`
- `backend/tests/test_worker_pool.py`

Tareas:
- `A1` eliminar `on_request()`
- `A2` corregir race de `_in_flight`
- `A3` callbacks de errores en tareas fire-and-forget
- `A4` salida limpia de `_watch_and_reload`
- `A6` `forward_stream()` debe validar status HTTP
- `SEC-5` whitelist en `update_config()`
- `M5` usar `dataclasses.asdict()`
- `M8` persistir `last_request_at`
- `B3` evaluar extracción de tipos compartidos si queda bloqueado por imports circulares

### Agente C — Scheduler / GPU / metrics / stats

Ownership principal:
- `backend/ocabra/api/metrics.py`
- `backend/ocabra/core/gpu_manager.py`
- `backend/ocabra/core/scheduler.py`
- `backend/ocabra/stats/aggregator.py`
- `backend/ocabra/api/internal/gpus.py`
- `backend/tests/test_gpu_manager.py`
- `backend/tests/test_scheduler.py`

Tareas:
- `C1` await en `_update_gauges()`
- `A5` reconectar `global_schedules` con scheduler y DB
- `M6` usar o eliminar `vram_pressure_threshold_pct`
- `M9` percentil correcto
- `M10` energía con intervalo real
- `A4/A5` validar que la documentación de schedules deja de mencionar APScheduler si ya no aplica
- `DOC_AUDIT A4` alinear contrato de GPU stats

### Agente D — OpenAI / shared API semantics

Ownership principal:
- `backend/ocabra/api/openai/*`
- tests OpenAI

Tareas:
- `M2` helper compartido de errores streaming
- `M3` consolidación de `resolve_model()` con punto de extensión para Ollama
- `B5` docstring `ensure_loaded()`
- Ajustes de error envelope si se decide cumplir `{"detail","code"}` en admin API o mantener formato FastAPI y documentarlo
- Revisar `DOC_AUDIT A8` si se decide fijar contrato actual de `/v1/models`

### Agente E — Ollama compat + model refs

Ownership principal:
- `backend/ocabra/api/ollama/*`
- `backend/ocabra/core/model_ref.py`
- tests Ollama

Tareas:
- `C2` añadir `acestep` a `KNOWN_BACKEND_TYPES`
- `M1` extraer helpers compartidos de `chat.py` y `generate.py`
- `M3` delegación de resolución de modelo hacia OpenAI `_deps`
- `B7` decidir si `_DEFAULT_OLLAMA_TO_INTERNAL` se elimina o se documenta
- `DOC_AUDIT A2` actualizar contrato de backend types

### Agente F — Frontend types y cliente API

Ownership principal:
- `frontend/src/types/index.ts`
- `frontend/src/api/client.ts`
- `frontend/src/hooks/useSSE.ts`
- tests frontend relacionados

Tareas:
- `M7` tipos frontend incompletos
- `M11` `encodeURIComponent(jobId)` en SSE
- `B2` eliminar código muerto frontend (`useSSE.ts`, `formatTokenCount`, `api/types.ts`) si no se usa
- `DOC_AUDIT A11` añadir `idleEvictionCheckIntervalSeconds`
- `DOC_AUDIT E5` añadir `musicGeneration`

Notas:
- Si se elimina `useSSE.ts`, revisar referencias antes de borrar.
- `frontend/src/api/client.ts` es fichero compartido: evitar mezclar refactor estructural grande en la misma rama que correcciones de tipos.

### Agente G — Documentación y contratos

Ownership principal:
- `docs/PLAN.md`
- `docs/CONTRACTS.md`
- `docs/CONVENTIONS.md`
- `docs/agents/*` solo si hace falta actualizar estado o ownership

Tareas:
- Resolver desalineaciones del `DOC_AUDIT`
- Actualizar árbol de directorios, stack real y checkboxes de `PLAN.md`
- Corregir contratos rotos o documentar explícitamente desviaciones aceptadas
- Documentar backends y endpoints no reflejados: `acestep`, `bitnet`, `ollama` backend, TRT-LLM compile UI, pooling/rerank/classify/score, `/metrics`, `/health`, `/ready`, `model_ref.py`
- Decidir si `CONVENTIONS.md` se endurece o se suaviza sobre schemas inline

Notas:
- Este agente no debe “inventar” el contrato futuro. Debe trabajar contra decisiones cerradas y código resultante de las oleadas previas.

## Oleadas recomendadas

### Oleada 1 — Seguridad y bugs críticos

Objetivo:
- cerrar exposición remota
- eliminar bugs con impacto funcional real

Agentes:
- A, B, C, E, F

Items:
- `SEC-1`, `SEC-2`, `SEC-3`, `SEC-4`, `SEC-5`, `SEC-6`
- `C1`, `C2`
- `A2`, `A3`, `A4`, `A6`
- `M11`

Gate de salida:
- tests de seguridad añadidos
- tests de core en verde
- smoke manual de auth en `/ocabra/*`
- `/metrics` mostrando métricas GPU reales

### Oleada 2 — Consistencia funcional y persistencia

Objetivo:
- corregir comportamiento silenciosamente incorrecto
- alinear scheduler, config y persistencia

Agentes:
- B, C, D, E, F

Items:
- `A1`, `A5`
- `M1`, `M2`, `M3`, `M5`, `M6`, `M7`, `M8`, `M9`, `M10`
- `B5`, `B6`, `B7`

Gate de salida:
- flujos de carga/streaming sin regresiones
- `last_request_at` persistido
- schedules globales con fuente de verdad única

### Oleada 3 — Refactor estructural y limpieza

Objetivo:
- bajar acoplamiento
- dividir módulos inflados
- eliminar deuda muerta

Agentes:
- B, F, G

Items:
- `B1`, `B2`, `B3`, `M4`
- migración de schemas inline si se decide mantener la convención

Gate de salida:
- sin imports circulares nuevos
- estructura más modular sin romper contratos

### Oleada 4 — Documentación final y cierre

Objetivo:
- dejar docs como fuente fiable

Agentes:
- G con apoyo puntual de A-F

Items:
- todo `DOC_AUDIT` pendiente
- actualización de briefings y estado real
- checklist final de divergencias aceptadas

Gate de salida:
- `PLAN.md`, `CONTRACTS.md`, `CONVENTIONS.md` y `AGENTS.md` coherentes con el código

## Ficheros compartidos con alto riesgo de conflicto

- `backend/ocabra/main.py`
- `backend/ocabra/api/internal/config.py`
- `backend/ocabra/api/internal/models.py`
- `backend/ocabra/api/internal/trtllm.py`
- `backend/ocabra/core/model_manager.py`
- `frontend/src/api/client.ts`
- `frontend/src/types/index.ts`
- `docs/CONTRACTS.md`
- `docs/PLAN.md`

Regla:
- un solo agente activo por fichero compartido cada vez
- si hay dependencia cruzada, integrar por PRs pequeños y secuenciales

## Tests mínimos por bloque

### Seguridad
- test de auth requerida en `/ocabra/*`
- test de rechazo CORS a origen no permitido
- test de path traversal rechazado en borrado de modelos
- test de path traversal rechazado en borrado de engines

### Core
- concurrencia sobre `_in_flight`
- error propagation de tareas background
- streaming que falla con 4xx/5xx
- persistencia de `last_request_at`

### Scheduler / stats
- `global_schedules` persiste y se aplica
- percentiles correctos
- energía calculada con timestamps reales

### Frontend
- tipos alineados con payload backend
- SSE con `jobId` escapado

### Documentación
- checklist manual de endpoints y payloads contra código real

## Riesgos y dependencias

1. La autenticación administrativa cambia el contrato de todo `/ocabra/*`.
2. `global_schedules` no se debe tocar sin acordar primero la fuente de verdad.
3. `models_dir` no debe seguir siendo mutable si existen operaciones destructivas dependientes de él.
4. La limpieza de docs no debe adelantarse a cambios funcionales si va a quedar obsoleta en la siguiente rama.
5. La migración de schemas inline a `ocabra/schemas/` puede generar mucho churn; conviene decidir si entra ahora o se deja para una fase específica.

## Preguntas abiertas para cerrar antes de ejecutar

1. ¿La autenticación de `/ocabra/*` la quieres mínima con API key ya, o quieres diseñarla como solución definitiva?
2. ¿El frontend oficial va a convivir en el mismo origen que la API o habrá orígenes separados en producción?
3. ¿`models_dir` debe pasar a ser inmutable en runtime de forma definitiva?
4. ¿Quieres que `docs/CONTRACTS.md` refleje exactamente el código actual salvo bugs, o prefieres aprovechar para rediseñar contratos dudosos?
5. ¿La migración de schemas Pydantic inline a `ocabra/schemas/` entra en este esfuerzo o la dejamos fuera para no mezclar refactor estructural con correcciones críticas?
6. ¿Quieres priorizar salida rápida en 2 oleadas o limpieza profunda completa aunque aumente el tiempo y el churn?

## Recomendación de arranque

Orden recomendado:
1. Cerrar las 6 decisiones abiertas anteriores.
2. Lanzar Oleada 1 con 5 agentes.
3. Integrar y validar.
4. Lanzar Oleada 2.
5. Decidir si Oleada 3 entra en este ciclo o se programa aparte.

## Estado final

Este plan ya se ejecutó en la práctica en oleadas sucesivas.

Backlog residual real:
- autenticación administrativa en `/ocabra/*`
- validación productiva final de TensorRT-LLM con engines reales
- batería backend de `pytest` en entorno completo
- limpieza estructural adicional solo si supera al coste de churn
