# Plan de Implementación: vLLM con soporte máximo en oCabra

## Instrucciones para el agente

Antes de empezar:

1. Lee `CLAUDE.md`.
2. Lee `AGENTS.md`.
3. Lee este documento completo.
4. Revisa `docs/CONTRACTS.md` y `docs/CONVENTIONS.md` antes de tocar contratos o UI.

Objetivo del trabajo:

- Convertir la integración de `vLLM` en `oCabra` en un backend de propósito general.
- Maximizar compatibilidad de modelos, no solo chat.
- Exprimir rendimiento sin sacrificar estabilidad.
- Mantener la UX coherente con la idea de `oCabra`: orquestar todos los modelos posibles, con la mejor receta disponible por backend.

Restricciones:

- Mantén compatibilidad con el hardware objetivo actual: `RTX 3060 12 GB + RTX 3090 24 GB`.
- Evita flags peligrosos por defecto si no hay medición.
- No clasifiques compatibilidad solo por tags de Hugging Face si ya hay una prueba runtime mejor.
- No rompas los flujos existentes de OpenAI-compatible y Ollama-compatible.

## Documentación de vLLM a revisar

Revisar estas secciones oficiales antes de implementar cada bloque:

- Supported Models
  https://docs.vllm.ai/en/stable/models/supported_models/
- Engine Args
  https://docs.vllm.ai/en/latest/configuration/engine_args/
- Optimization and Tuning
  https://docs.vllm.ai/en/latest/configuration/optimization/
- OpenAI-Compatible Server
  https://docs.vllm.ai/en/latest/serving/openai_compatible_server/
- Pooling Models
  https://docs.vllm.ai/en/stable/models/pooling_models/
- Tool Calling
  https://docs.vllm.ai/en/latest/features/tool_calling/
- Structured Outputs
  https://docs.vllm.ai/en/latest/features/structured_outputs/
- LoRA
  https://docs.vllm.ai/en/stable/features/lora.html
- Quantization
  https://docs.vllm.ai/en/latest/features/quantization/
- Compatibility Matrix
  https://docs.vllm.ai/en/v0.9.2/features/compatibility_matrix.html

## Estado actual del repo

### Lo que ya existe

- Backend `vLLM` funcional:
  - `backend/ocabra/backends/vllm_backend.py`
  - `workers/vllm_worker.py`
- Overrides por modelo en UI y backend para:
  - `tensor_parallel_size`
  - `max_model_len`
  - `max_num_seqs`
  - `max_num_batched_tokens`
  - `gpu_memory_utilization`
  - `enable_prefix_caching`
  - `trust_remote_code`
  - `enable_chunked_prefill`
  - `kv_cache_dtype`
  - `swap_space`
  - `enforce_eager`
- Registry HF con detección temprana de incompatibilidades de tokenizer en:
  - `backend/ocabra/registry/huggingface.py`
- UI de configuración por modelo con hints en:
  - `frontend/src/components/models/ModelConfigModal.tsx`

### Lo que falta

- Clasificación real de compatibilidad `vLLM`.
- Soporte explícito para `Transformers backend`.
- Soporte explícito para `pooling`, `rerank`, `classification`.
- `hf_overrides`, `chat_template`, `model_impl`, `runner`.
- `tool_call_parser`, `reasoning_parser`, `structured outputs`.
- `language_model_only` y mejores flags multimodales.
- Runtime probe real de compatibilidad.
- Recipes por familia/modelo.
- Matriz seria de quantization / compatibilidad por hardware.

## Objetivo funcional final

`oCabra` debe poder decidir, por modelo:

- si debe ir a `vLLM`
- si debe ir a implementación nativa o `Transformers backend`
- si es `generate` o `pooling`
- si necesita:
  - `trust_remote_code`
  - `hf_overrides`
  - `chat_template`
  - `tool_call_parser`
  - `reasoning_parser`
  - `language_model_only`
- con qué preset de rendimiento conviene correrlo

## Roadmap recomendado

## Fase 1. Modelo de compatibilidad y serving mode

### Objetivo

Dejar de tratar `vLLM` como backend binario `sí/no`.

### Cambios

Añadir un modelo de compatibilidad con estados como:

- `native_vllm`
- `transformers_backend`
- `pooling_only`
- `generate`
- `multimodal_generate`
- `multimodal_pooling`
- `needs_trust_remote_code`
- `needs_hf_overrides`
- `needs_chat_template`
- `needs_tool_parser`
- `needs_reasoning_parser`
- `unsupported`
- `unknown_until_runtime`

### Ficheros candidatos

- `backend/ocabra/schemas/registry.py`
- `backend/ocabra/registry/huggingface.py`
- `frontend/src/types/index.ts`
- `frontend/src/api/client.ts`
- `frontend/src/components/explore/HFModelCard.tsx`
- `frontend/src/pages/Explore.tsx`

### Criterios de aceptación

- El registry devuelve algo más útil que `suggested_backend=vllm`.
- La UI distingue:
  - compatible nativo
  - compatible vía transformers
  - necesita configuración extra
  - incompatible

## Fase 2. Nuevos parámetros críticos de vLLM

### Objetivo

Exponer la capa de control mínima para llegar a más modelos.

### Parámetros a soportar

- `model_impl`
  - `auto`
  - `vllm`
  - `transformers`
- `runner`
  - `generate`
  - `pooling`
- `hf_overrides`
- `chat_template`
- `chat_template_content_format` si aplica
- `generation_config`
- `override_generation_config`
- `tool_call_parser`
- `tool_parser_plugin`
- `reasoning_parser`
- `language_model_only`

### Ficheros candidatos

- `backend/ocabra/config.py`
- `.env`
- `.env.example`
- `docker-compose.yml`
- `backend/ocabra/backends/vllm_backend.py`
- `workers/vllm_worker.py`
- `frontend/src/types/index.ts`
- `frontend/src/api/client.ts`
- `frontend/src/components/models/ModelConfigModal.tsx`

### Criterios de aceptación

- Cada parámetro puede configurarse globalmente y por modelo.
- Los flags llegan al comando final de `vLLM`.
- La UI muestra hints claros y separa básico/avanzado.

## Fase 3. Tasks reales: generate, pooling, embeddings, rerank

### Objetivo

Que `oCabra` no trate todo como chat/completion.

### Cambios

Ampliar detección y capacidades para:

- `generate`
- `embeddings`
- `pooling`
- `rerank`
- `classification`
- `score`

### Implementación

- Revisar cómo `vLLM` expone estas tareas en el servidor OpenAI-compatible.
- Ajustar `ModelCapabilities`.
- Ajustar UI para mostrar la tarea real.
- Añadir rutas o adaptación donde haga falta.

### Ficheros candidatos

- `backend/ocabra/backends/vllm_backend.py`
- `backend/ocabra/api/openai/`
- `backend/ocabra/api/internal/models.py`
- `frontend/src/types/index.ts`
- `frontend/src/pages/Models.tsx`
- `frontend/src/pages/Playground.tsx`

### Criterios de aceptación

- Un modelo de embeddings no aparece como LLM genérico.
- Un reranker/pooling model aparece y se sirve con el modo correcto.

## Fase 4. Recipes por familia

### Objetivo

Reducir configuración manual y errores repetidos.

### Familias prioritarias

- Qwen
- Qwen VL
- DeepSeek
- Granite
- GLM
- FunctionGemma
- InternLM
- Jamba
- Nemotron
- MiniMax
- OLMo
- xLAM

### Qué debe poder definir una recipe

- `model_impl`
- `runner`
- `trust_remote_code`
- `hf_overrides`
- `chat_template`
- `tool_call_parser`
- `reasoning_parser`
- `language_model_only`
- preset de rendimiento

### Propuesta

Crear un módulo nuevo, por ejemplo:

- `backend/ocabra/backends/vllm_recipes.py`

o dentro de:

- `backend/ocabra/registry/`

### Criterios de aceptación

- Al menos las familias más comunes arrancan con menos intervención manual.
- La UI puede mostrar “recipe aplicada”.

## Fase 5. Runtime probe de compatibilidad

### Objetivo

Dejar de depender solo de heurísticas estáticas.

### Qué debe probar

- carga de tokenizer
- carga de config
- `model_impl` recomendado
- `runner` recomendado
- healthcheck
- endpoint mínimo
- si requiere:
  - `trust_remote_code`
  - `hf_overrides`
  - `tool parser`
  - `chat template`

### Diseño sugerido

- Probe corto y explícito.
- No descargar pesos nuevos innecesariamente.
- Ejecutar primero contra modelos ya descargados.
- Guardar resultado cacheado.

### Estados sugeridos

- `supported_native`
- `supported_transformers_backend`
- `supported_with_recipe`
- `needs_remote_code`
- `needs_hf_overrides`
- `needs_tool_parser`
- `needs_chat_template`
- `unsupported_tokenizer`
- `unsupported_architecture`
- `insufficient_vram`
- `unknown`

### Ficheros candidatos

- `backend/ocabra/registry/huggingface.py`
- nuevo servicio en `backend/ocabra/core/` o `backend/ocabra/registry/`
- UI en Explore para mostrar el resultado

### Criterios de aceptación

- El usuario puede distinguir “incompatible” de “no probado”.
- El sistema no sugiere instalar/cargar a ciegas modelos que ya sabe que fallarán.

## Fase 6. Tool calling, reasoning y structured outputs

### Objetivo

Hacer que `tools` y `reasoning` sean capacidades reales, no heurísticas.

### Añadir soporte por modelo

- `tool_call_parser`
- `tool_parser_plugin`
- `reasoning_parser`
- settings necesarias de structured outputs si hacen falta

### Cambios

- ampliar capabilities
- ampliar recipes
- ampliar UI
- revisar compatibilidad con endpoints OpenAI en `backend/ocabra/api/openai/`

### Criterios de aceptación

- Si un modelo necesita parser, `oCabra` lo sabe.
- Si el parser está configurado, las herramientas funcionan realmente.

## Fase 7. Multimodal serio

### Objetivo

Mejorar compatibilidad y consumo de memoria en modelos visuales/multimodales.

### Parámetros prioritarios

- `language_model_only`
- `skip_mm_profiling`
- `limit_mm_per_prompt`
- `mm_processor_cache_gb`
- `mm_processor_cache_type`

### Criterios de aceptación

- Los modelos multimodales pueden arrancar en modo solo lenguaje para ahorrar VRAM.
- La UI deja claro cuándo un modelo está corriendo sin parte multimodal.

## Fase 8. LoRA y adaptadores

### Objetivo

Soportar LoRA de forma explícita dentro del backend `vLLM`.

### Parámetros

- `enable_lora`
- `lora_modules`
- `max_loras`
- `max_lora_rank`
- `max_cpu_loras`

### Criterios de aceptación

- Se pueden declarar LoRAs por modelo base.
- La UI y el backend reflejan que el modelo corre con adaptadores.

## Fase 9. Quantization y compatibilidad por hardware

### Objetivo

No tratar todas las cuantizaciones igual.

### Clasificar

- FP8
- FP8 KV cache
- AWQ
- GPTQ
- Marlin
- bitsandbytes
- GGUF si llega a encajar por ruta concreta

### Reglas

- Cruzar con arquitectura GPU real.
- Marcar explícitamente lo que en Ampere es recomendable y lo que no.
- Evitar defaults agresivos globales.

### Criterios de aceptación

- El usuario ve warnings útiles cuando una cuantización existe pero no es ideal para su hardware.

## Fase 10. Presets de rendimiento y observabilidad

### Objetivo

Exprimir `vLLM` con configuración reproducible.

### Presets sugeridos

- `latency`
- `balanced`
- `throughput`
- `max_fit`

### Parámetros que deben ajustar

- `gpu_memory_utilization`
- `max_num_seqs`
- `max_num_batched_tokens`
- `enable_prefix_caching`
- `enable_chunked_prefill`
- `kv_cache_dtype`
- `max_model_len`

### Métricas a mostrar o guardar

- startup time
- load success/failure by recipe
- TTFT
- ITL
- throughput tokens/s
- preemptions
- OOM rate
- load failure classification

### Criterios de aceptación

- La UI puede sugerir presets razonables.
- El sistema permite comparar cambios de tuning.

## Orden de implementación recomendado

Implementar en este orden:

1. Fase 1
2. Fase 2
3. Fase 3
4. Fase 4
5. Fase 5
6. Fase 6
7. Fase 7
8. Fase 9
9. Fase 8
10. Fase 10

Motivo:

- primero compatibilidad y flags correctos
- luego cobertura funcional real
- después recipes y probes
- luego features avanzadas
- al final tuning fino y presets

## Entregables mínimos por fase

Cada fase debe incluir:

- código
- tests backend
- tests frontend si toca UI
- actualización de `.env.example`
- actualización de documentación de usuario si cambia UX

## Riesgos conocidos

- `Transformers backend` puede ampliar compatibilidad pero empeorar rendimiento.
- `trust_remote_code` aumenta superficie de riesgo.
- `hf_overrides` mal usados pueden hacer que un modelo “arranque” pero falle funcionalmente.
- multimodal y LoRA aumentan mucho la complejidad de recipes.
- no conviene activar globalmente flags de cuantización o memoria sin medir en esta máquina.

## Definición de “hecho”

El trabajo se considera completo cuando:

- Explore clasifica mejor los modelos `vLLM`.
- La configuración por modelo soporta los flags críticos adicionales.
- `oCabra` puede servir más tareas de `vLLM`, no solo chat.
- Hay recipes por familia para casos comunes.
- Existe runtime probe básico de compatibilidad.
- La UI explica mejor compatibilidad, parsers, templates y modo de ejecución.
- El usuario puede elegir entre soporte máximo y presets de rendimiento.

## Siguiente sprint recomendado

Si hay que empezar solo por un bloque manejable, hacer este sprint:

1. `model_impl`
2. `runner`
3. `hf_overrides`
4. `chat_template`
5. `tool_call_parser`
6. `reasoning_parser`
7. clasificación de compatibilidad `native/transformers/pooling`
8. probe runtime mínimo

Ese sprint ya da un salto muy grande en cobertura real de `vLLM`.
