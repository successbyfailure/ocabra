# Auditoría de Documentación vs. Código — oCabra

Actualizado: 2026-08-23

Este documento resume la desalineación residual entre documentación y código.
No intenta conservar planes antiguos ni snapshots históricos completos: el estado
vivo está aquí y en `docs/PLAN.md`.

## Estado vivo

Documentación ya alineada con el código:
- `BackendCapabilities` y tipos frontend incluyen `pooling`, `rerank`, `classification`, `score` y `music_generation`.
- `acestep`, `bitnet`, `ollama`, `llama_cpp`, `sglang` y `tensorrt_llm` están tratados como backends first-class.
- `WorkerPool`, `ServiceState`, `/metrics`, `/health`, `/ready`, `/ocabra/models/storage`, `/ocabra/services/*` y `globalSchedules` ya están reflejados con semántica real.
- `modelsDir` es de solo lectura en runtime; `downloadDir` y `maxTemperatureC` siguen siendo overrides en memoria.
- La configuración de modelos expone estimación de memoria rápida y, para `vllm`, probe real con el engine.
- `TensorRT-LLM` y `vLLM` han quedado validados en runtime real con carga, respuesta y descarga sin huérfanos.
- La compatibilidad Ollama ya traduce `max_tokens -> num_predict` y usa `backend_model_id` nativo al reenviar `/api/chat` y `/api/generate`.
- `TensorRT-LLM` deriva `context_length` desde el `config.json` real del engine y lo alinea con el estado del modelo.
- Existe un baseline de benchmark reproducible en `docs/benchmarks/qwen3-backends-2026-04-03.md`.
- `docs/agents/` y `docs/tasks/` se han reducido a estado actual o notas de archivo; ya no deben usarse como plan maestro.

## Cambios vivos recientes que ya deben asumirse

- Existe `POST /ocabra/models/{model_id}/memory-estimate`.
- La request acepta:
  - `preferred_gpu?: int | null`
  - `extra_config?: dict | null`
  - `run_probe?: bool`
- La respuesta puede incluir, según backend:
  - presupuesto estimado de memoria
  - tamaño de pesos o engine por GPU
  - estimación de KV cache
  - contexto máximo estimado
  - `status`, `warning`, `notes`
  - `source: "heuristic" | "runtime_probe"`
- En `vllm`, el probe real se apoya en el propio engine para detectar límites de contexto y fallos de inicialización antes de guardar cambios.

## Backlog residual real

- Bloque 18 de fiabilidad entregado: actividad Realtime STT, transiciones
  `loading/unloading`, resolución canónica de perfiles, errores accionables y
  hardening Bonsai/VibeASR.
- Bloque 19 de observabilidad Realtime entregado: identidad atribuible y filas
  diferenciadas para sesión, STT, chat y TTS en `request_stats`, con logs
  estructurados sin contenido ni credenciales.
- Validación dirigida del Bloque 19: 30 tests en verde. La ejecución local
  amplia (omitiendo `test_whisper_worker.py`, que requiere `numpy`) deja 972
  tests en verde, 2 omitidos y 14 fallos no relacionados: dependencias de
  servicios PostgreSQL/Ollama y expectativas antiguas de backends/GPU.
- Validación backend completa con `pytest` en CI o contenedor con dependencias
  y servicios completos.
- Validación productiva final de `TensorRT-LLM` con más de un engine y toolchain CUDA/NVIDIA objetivo.
- Limpieza del inventario persistido de entradas `tensorrt_llm/*` con `engine_dir` inexistente.
- Mejoras opcionales futuras:
  - revisión conjunta de estrategias de recursos, concurrencia y timeouts
  - decisión de soporte para OpenAI Responses API
  - más cobertura e2e de ciclos completos por backend

## Documentos que se consideran fuente de verdad

- `docs/PLAN.md`: arquitectura, estado global y backlog.
- `docs/CONTRACTS.md`: contratos API y entre módulos.
- `docs/CONVENTIONS.md`: naming y estilo.
- `AGENTS.md`: reglas de colaboración multi-agente.

## Documentos archivados o reducidos

- `docs/agents/*.md`: ya no son briefings de implementación; ahora son resúmenes de estado por stream.
- `docs/tasks/CLAUDE-TASKS.md`, `docs/tasks/QWEN-TASKS.md`, `docs/tasks/refactor-doc-audit-swarm-plan.md`, `docs/tasks/bitnet-implementation-plan.md`: documentos históricos.
- `docs/tasks/trtllm-compile-ui-plan.md` y `docs/tasks/langfuse-integration-plan.md`: siguen siendo planes activos.
