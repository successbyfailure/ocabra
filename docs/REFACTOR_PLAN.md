# Plan de Refactorización y Seguridad — oCabra

Generado: 2026-04-02 · Actualizado: 2026-04-04

---

## Estado: completado

La ronda de refactorización y hardening está cerrada. Todos los hallazgos críticos,
altos y la mayoría de los medios han sido corregidos en el código.

### Resumen de lo resuelto

| ID | Descripción | Resolución |
|----|-------------|-----------|
| C1 | `get_all_states()` sin `await` en metrics.py | Corregido |
| C2 | `acestep` fuera de `KNOWN_BACKEND_TYPES` | Añadido |
| SEC-2 | CORS wildcard | Restringido a `localhost/127.0.0.1` via `allow_origin_regex` |
| SEC-3 | Path traversal en borrado de modelos | Protegido con `_is_path_within_base()` |
| SEC-4 | Path traversal en borrado de engines TRT-LLM | Protegido con `_is_path_within_base()` |
| SEC-5 | `setattr` sin whitelist en `update_config` | Whitelist explícita `allowed_fields` |
| A1 | `on_request()` código muerto | Eliminado |
| A2 | Race condition en `_in_flight` counter | Protegido con `threading.Lock` |
| A3 | Tasks fire-and-forget sin manejo de errores | `add_done_callback(_log_task_result)` añadido |
| A5 | `global_schedules` desconectado de la DB | Conectado via `replace_global_schedules()` |
| A6 | `forward_stream` sin comprobar status HTTP | `raise_for_status()` añadido |
| M4 | Código muerto: `ServerConfig`, `delete_key` | Eliminados |
| M5 | `BackendCapabilities.to_dict()` manual | Reemplazado con `dataclasses.asdict()` |
| M6 | `vram_pressure_threshold_pct` nunca leída | Usada en `scheduler.py` |
| M8 | `last_request_at` no persiste | Persiste en Redis; rehidratado al arrancar |
| M9 | Cálculo de percentil incorrecto | Corregido |
| M10 | Cálculo de energía con intervalo fijo | Corregido usando timestamps reales |
| M11 | `encodeURIComponent` faltante en SSE | Corregido en `client.ts` |
| B2 | Código muerto frontend (`useSSE`, `formatTokenCount`) | `useSSE` ya en uso (CompileModal); `formatTokenCount` revisada |
| B5 | Docstring incorrecto en `ensure_loaded()` | Corregido |
| B7 | `_DEFAULT_OLLAMA_TO_INTERNAL` hardcoded | Evaluado y limpiado |

---

## Pendiente residual

Tres items menores que no se han cerrado. Ver `docs/ROADMAP.md` para trazabilidad completa.

### [B6] `ModelPatch` no puede limpiar `preferred_gpu` a `None`

- **Archivo:** `backend/ocabra/api/internal/models.py:150`
- Filtro `{k: v for ... if v is not None}` impide enviar `null` para resetear `preferred_gpu`.
- **Fix:** Usar `body.model_dump(exclude_unset=True)` en lugar del filtro por `None`.

### [A4] `_watch_and_reload` sin check de shutdown

- **Archivo:** `backend/ocabra/core/model_manager.py:747`
- El loop ya tiene deadline y guards de estado, pero no verifica el cierre del servidor.
- **Fix:** Añadir comprobación de `self._shutdown_event` o equivalente al condition del `while`.

### [M7] `HFModelVariant.backend_type` Literal incompleto

- **Archivo:** `backend/ocabra/schemas/registry.py:89`
- Solo incluye `vllm, bitnet, diffusers, whisper, tts, ollama`. Faltan `llama_cpp`, `sglang`, `tensorrt_llm`, `acestep`.
- **Fix:** Ampliar el Literal o cambiar a `BackendType` si el tipo ya está centralizado.

---

## Items descartados

| ID | Motivo |
|----|--------|
| SEC-6 | `models_dir` ya es de solo lectura en la UI; el riesgo real queda cubierto por SEC-1 (auth) |
| M1/M2/M3 | Deduplicaciones de bajo impacto; aplazar hasta que haya más cambios en esas áreas |
| B1 | Módulos grandes pero coherentes; refactor de alcance no justificado ahora |
| B3 | Imports circulares contenidos; refactor estructural aplazado |
| B4 | Strings en español: cosmético, aplazado |
