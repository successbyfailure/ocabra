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

## Handoff

Si continúas este trabajo en otra iteración o con otro agente, usa:

- `docs/tasks/vllm-max-support-handoff.md`

## Estado actual del repo

## Seguimiento de implementación

### Actualización 2026-03-13

#### Sprint inicial ejecutado en esta iteración

Estado: **parcialmente completado con el bloque de mayor valor entregado end-to-end**.

Implementado:

- `model_impl` en config global, override por modelo, backend `vLLM`, worker wrapper y UI.
- `runner` en config global, override por modelo, backend `vLLM`, worker wrapper y UI.
- `hf_overrides` en config global, override por modelo, backend `vLLM` y UI.
- `chat_template` en config global, override por modelo, backend `vLLM` y UI.
- `tool_call_parser` en config global, override por modelo, backend `vLLM` y UI.
- `reasoning_parser` en config global, override por modelo, backend `vLLM` y UI.
- Clasificación de compatibilidad `native_vllm` / `transformers_backend` / `pooling` / `unsupported` / `unknown` en registry HF.
- Recomendación asociada de `model_impl`, `runner` y overrides requeridos en el payload de registry.
- Probe runtime mínimo basado en carga de `AutoConfig` y `AutoTokenizer` sin descargar pesos.
- UI de Explore actualizada para mostrar compatibilidad vLLM, overrides requeridos y estado del probe.
- UI de configuración por modelo actualizada para exponer los nuevos flags críticos.
- Manejo de variables `VLLM_*` opcionales vacías como `None` para evitar fallos de arranque en Docker.

Validado:

- Build de `frontend` dentro de Docker: OK.
- Test frontend nuevo para `HFModelCard`: OK.
- `ruff check` sobre los ficheros backend tocados: OK.
- Tests backend dirigidos para registry/vLLM backend y nuevos flags: OK.
- Reconstrucción y relanzado de servicios `api` y `frontend`: OK.
- `api` sano tras el despliegue (`healthy`) y `frontend` arrancado correctamente.

Pendiente dentro del sprint inicial:

- `chat_template_content_format`.
- `generation_config`.
- `override_generation_config`.
- `tool_parser_plugin`.
- clasificación funcional completa de tareas `rerank` / `classification` / `score`.
- probe runtime con arranque real de `vLLM` en lugar de probe ligero de config/tokenizer.

Límites detectados:

- La clasificación actual mejora mucho la decisión previa, pero sigue siendo híbrida: heurística + probe ligero.
- El probe no garantiza que el worker `vLLM` vaya a arrancar realmente; solo reduce falsos positivos obvios.
- La UI ya refleja compatibilidad y configuración, pero `Models` y `Playground` todavía no adaptan el flujo a modelos `pooling`.
- Siguen existiendo fallos previos, fuera de este sprint, en algunos tests completos de registry/inventario local/Ollama.

#### Segunda iteración del mismo día

Estado: **avance material sobre Fase 2 y Fase 3, todavía no plan completo**.

Implementado adicionalmente:

- `chat_template_content_format` en config global, override por modelo, backend `vLLM`, worker wrapper y UI.
- `generation_config` en config global, override por modelo, backend `vLLM`, worker wrapper y UI.
- `override_generation_config` en config global, override por modelo, backend `vLLM`, worker wrapper y UI.
- `tool_parser_plugin` en config global, override por modelo, backend `vLLM`, worker wrapper y UI.
- `language_model_only` en config global, override por modelo, backend `vLLM`, worker wrapper y UI.
- Extensión de `BackendCapabilities` para `pooling`, `rerank`, `classification` y `score`.
- Detección operativa de modelos `pooling`/`score` en backend `vLLM`.
- Endpoints OpenAI adicionales:
  - `/v1/pooling`
  - `/v1/score`
- `Models` ya distingue `pooling` como tipo propio.
- `Playground` deja de tratar modelos `pooling` como chat y expone una interfaz dedicada para `pooling` y `score`.

Validado adicionalmente:

- `ruff check` backend para los nuevos ficheros y tests OpenAI: OK.
- `pytest tests/test_openai_api.py tests/test_vllm_backend.py -q`: OK.
- Tests frontend nuevos para `PoolingInterface`: OK.
- Build frontend en contenedor tras los cambios de `Playground`/`Models`: OK.
- Reconstrucción y relanzado de `api` y `frontend` tras esta segunda iteración: OK.

Pendiente relevante tras esta iteración:

- `rerank` y `classification` siguen sin endpoint/UI dedicados.
- El probe runtime sigue siendo ligero; falta arranque real de `vLLM`.
- No existen recipes por familia.
- No hay soporte explícito de structured outputs.
- No hay matriz seria de quantization/hardware.

#### Tercera iteración del mismo día

Estado: **Fase 3 ampliada, Fase 5 subida a probe real con fallback y Fase 4 iniciada con recipes base**.

Implementado adicionalmente:

- Endpoints OpenAI adicionales sobre el bloque de pooling:
  - `/v1/rerank`
  - `/v1/classify`
- Comprobación de capability antes de reenviar tráfico a worker para `rerank` y `classification`.
- Normalización y validación temprana de payloads:
  - `classification`: string o lista de strings no vacíos
  - `rerank`: documentos como string o `{text: ...}` y validación de `top_n`
  - `score`: `queries`/`documents` como strings o listas paralelas; alias legacy `text_1`/`text_2`
- Detección más útil de capacidades en `VLLMBackend`:
  - arquitecturas `*ForSequenceClassification` ahora activan `classification`
  - `num_labels` / `id2label` permiten marcar modelos de clasificación aunque la arquitectura no esté en la tabla corta
  - heurística de nombre para `cross-encoder` / `rerank` activa `rerank` y `score`
- Playground ampliado para modelos de `pooling` con bloques dedicados de:
  - `pooling`
  - `score`
  - `rerank`
  - `classification`
- Tests backend añadidos para:
  - forwarding de `/v1/rerank`
  - forwarding de `/v1/classify`
  - rechazo cuando falta capability
  - inferencia de capacidades para sequence classification y rerankers
- Test frontend actualizado para reflejar los nuevos bloques del Playground.
- Servicio nuevo de probe runtime real de `vLLM`:
  - intenta arranque corto de `vLLM` contra artefactos locales o snapshots cacheados
  - usa healthcheck real y captura stderr
  - cachea resultados por repo
  - cae al probe ligero anterior si no hay artefacto local o el entorno no permite probe real
- `HuggingFaceRegistry` deja de concentrar toda la lógica del probe y delega en el servicio nuevo.
- Módulo inicial de recipes por familia:
  - `backend/ocabra/backends/vllm_recipes.py`
  - recipes base para `Qwen3`, `DeepSeek-R1`, `Granite`, `GLM` y `FunctionGemma`
  - el registry expone `recipe_id`, notas y `suggested_config` junto con overrides requeridos
- Explore muestra la recipe sugerida y la configuración recomendada.
- Explore envía una `register_config` junto a la descarga para que, al completarse, el modelo quede dado de alta con:
  - `load_policy` elegido
  - `display_name`
  - `extra_config.vllm` prellenado desde la recipe (`model_impl`, `runner`, `suggested_config`)
- La metadata de recipe queda persistida en `extra_config.vllm` y `ModelConfigModal` puede:
  - mostrar la recipe activa
  - enseñar notas/configuración sugerida
  - reaplicar sugerencias sobre los campos editables
- Las recipes distinguen ahora entre:
  - `required_overrides`: compatibilidad obligatoria
  - `suggested_config`: compatibilidad recomendada y preaplicable
  - `suggested_tuning`: tuning recomendado, visible y aplicable manualmente
- Refinamiento adicional de recipes por subfamilia:
  - `Qwen-VL` con sugerencia explícita de `language_model_only`
  - `Qwen3-MoE` con tuning más conservador de concurrencia
  - detección específica de `DeepSeek-V3` y `GLM-4` antes de caer en variantes más genéricas
- El registry traduce ahora un runtime probe `needs_remote_code` en `trust_remote_code` dentro de `required_overrides`, para que la UI y la instalación reflejen mejor ese requisito.

Validado adicionalmente:

- Backend:
  - `python -m pytest tests/test_openai_api.py tests/test_vllm_backend.py -q` dentro de `ocabra-api:latest` con extras `.[dev,vllm]`: OK (`44 passed`)
  - `python -m ruff check ...`: OK
  - `python -m pytest tests/test_vllm_runtime_probe.py tests/test_registry.py -q -k "...probe..."`: OK (`5 passed`)
  - `python -m pytest tests/test_registry.py -q -k "...recipe..."`: OK
- Frontend:
  - `vitest` dirigido para `PoolingInterface.test.tsx` con `--no-cache`: OK
  - `npx tsc -b`: OK
  - `vitest` dirigido para `HFModelCard.test.tsx` con `--no-cache`: OK
  - `vitest` dirigido para `ModelConfigModal.test.tsx` con `--no-cache`: OK

Límites detectados en esta iteración:

- `rerank`, `classification` y `score` ya no son pass-through ciego: validan y normalizan entrada antes del worker, pero la respuesta final sigue dependiendo del contrato upstream de vLLM.
- La detección de `rerank` sigue siendo parcialmente heurística; sirve para mejorar UX y gating, no como prueba definitiva de compatibilidad.
- El build completo de frontend no se reutilizó como señal de aceptación porque en este entorno hay ruido de permisos/cargador alrededor de `.vite` y `vite.config.ts`; la validación útil aquí fue `vitest` dirigido + `tsc`.
- El probe real solo se ejecuta si el modelo ya existe localmente o en snapshot cacheado; no descarga pesos nuevos ni intenta validar repos remotos desde cero.
- Parte del suite amplio de `test_registry.py` sigue teniendo fallos previos ajenos a este bloque (DB real / local scanner), así que la validación de esta iteración se hizo con tests dirigidos.
- Las recipes iniciales son conservadoras: recomiendan presets útiles, pero todavía no cubren exhaustivamente familias/variantes ni sustituyen una matriz de compatibilidad real.
- La preconfiguración automática aplica solo sugerencias explícitas de la recipe; aunque el registry ya marque `trust_remote_code` cuando el probe lo detecta, ese override delicado sigue requiriendo decisión manual.
- El tuning recomendado sigue siendo heurístico y opt-in; no se aplica automáticamente al registrar el modelo.

#### Ajuste incremental 2026-03-14

Estado: **refinamiento pequeño pero útil sobre Fase 4/Fase 5**.

Implementado adicionalmente:

- Recipes más finas para subfamilias de `vLLM`:
  - `qwen-vl`
  - `qwen3-moe`
  - `internvl-chat`
  - `llama4-multimodal`
- Priorización de recipes específicas antes de las genéricas para evitar que ciertas variantes caigan en presets demasiado amplios.
- Traducción directa del resultado de probe runtime `needs_remote_code` a `required_overrides += ["trust_remote_code"]` en el payload de soporte HF.
- El runtime probe puede corregir ahora `model_impl` y `runner` expuestos por el registry cuando su recomendación es más fiable que la heurística inicial o la recipe base.
- Explore y `ModelConfigModal` distinguen visualmente entre:
  - recipe persistida
  - recomendación verificada por probe
  - diferencia entre configuración activa y recomendación del probe cuando existe
- La metadata mínima del probe queda persistida en `extra_config.vllm` para no perder contexto al abrir la configuración manual tras instalar desde Explore.
- `ModelConfigModal` ya ofrece un botón específico para aplicar la recomendación del probe sobre `model_impl` y `runner`, separado de la recipe y del tuning.
- `Explore` muestra ahora un aviso explícito cuando la preconfiguración automática va a usar la recomendación verificada por probe.
- El contrato/UI preservan también `recipe_model_impl` y `recipe_runner` para poder mostrar claramente la diferencia entre recipe base y recomendación final del probe.
- `Explore` y `ModelConfigModal` muestran ya cuándo la recipe base no coincide con la recomendación final verificada por probe.
- El runtime probe persiste además `observed_at` / `observedAt`, para que la UI pueda enseñar cuándo se observó por última vez esa recomendación.
- El runtime probe clasifica ya más fallos reales de `vLLM` en señales accionables: `missing_chat_template`, `missing_tool_parser`, `missing_reasoning_parser` y `needs_hf_overrides`.
- El registry traduce esas señales a `required_overrides` concretos (`chat_template`, `tool_call_parser`, `reasoning_parser`, `hf_overrides`) y las expone a UI/instalación como warnings accionables.
- La UI ya no muestra solo `status` crudos del probe: cards, Explore y `ModelConfigModal` usan etiquetas legibles y hints directos del override sugerido.
- Existe prueba de flujo frontend tipo E2E con Vitest/jsdom para `Explore`, cubriendo apertura de modal, avisos de recipe/probe e instalación con persistencia de metadata.
- El repo ya incluye E2E de navegador con Playwright + Chrome para el flujo principal de `Explore`, además de una validación real ad hoc sobre la app levantada en `http://127.0.0.1:8484/explore`.
- El E2E de navegador cubre ya dos escenarios: guidance de recipe vs probe y caso accionable donde el probe exige un override concreto.

Validado adicionalmente:

- `python -m pytest tests/test_registry.py -q -k "qwen_vl or qwen3_moe or remote_code_adds_required_override or exposes_recipe_metadata or classifies_transformers_backend_and_probe"` dentro de `ocabra-api` con extras `.[dev,vllm]`: OK (`5 passed`)
- `python -m ruff check ocabra/backends/vllm_recipes.py ocabra/registry/huggingface.py tests/test_registry.py` dentro de `ocabra-api` con extras `.[dev,vllm]`: OK
- `python -m pytest tests/test_registry.py -q -k "internvl or llama4_multimodal or overrides_model_impl_and_runner or classifies_transformers_backend_and_probe or qwen3_moe or qwen_vl"` dentro de `ocabra-api` con extras `.[dev,vllm]`: OK (`6 passed`)
- `vitest --no-cache src/__tests__/HFModelCard.test.tsx src/__tests__/ModelConfigModal.test.tsx` en contenedor `node:20-alpine`: OK
- `npx tsc -b` en contenedor `node:20-alpine`: OK
- `vitest --no-cache src/__tests__/Explore.test.ts src/__tests__/HFModelCard.test.tsx src/__tests__/ModelConfigModal.test.tsx` en contenedor `node:20-alpine`: OK
- `vitest --no-cache src/__tests__/Explore.test.ts src/__tests__/ExploreFlow.test.tsx src/__tests__/HFModelCard.test.tsx src/__tests__/ModelConfigModal.test.tsx && npx tsc -b` en contenedor `node:20-alpine`: OK
- `./node_modules/.bin/playwright test --project=chrome` en `frontend/`: OK (`1 passed`)
- Validación real ad hoc con `/usr/bin/google-chrome` headless contra `http://127.0.0.1:8484/explore` tras `docker compose up -d --build api frontend`: OK
- `python -m pytest tests/test_vllm_runtime_probe.py tests/test_registry.py -q -k "remote_code or missing_chat_template or missing_tool_parser or missing_reasoning_parser or needs_hf_overrides"` en `ocabra-api` con extras `.[dev,vllm]`: OK (`9 passed`)
- `python -m ruff check ocabra/schemas/registry.py ocabra/registry/huggingface.py ocabra/registry/vllm_runtime_probe.py tests/test_vllm_runtime_probe.py tests/test_registry.py` en `ocabra-api`: OK

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
- Soporte explícito más sólido para `Transformers backend`.
- Endurecer contratos y compatibilidad real de `pooling`, `rerank`, `classification` y `score`.
- `structured outputs` end-to-end.
- Mejoras multimodales adicionales sobre `language_model_only`.
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

### Estado del roadmap a 2026-03-13

| Fase | Estado | Nota |
|------|--------|------|
| **Fase 1. Compatibilidad y serving mode** | **Parcial** | Clasificación y UI de Explore ya operativas. Falta afinar task coverage completa y estados multimodales/recipes. |
| **Fase 2. Parámetros críticos de vLLM** | **Casi completa** | Ya están `model_impl`, `runner`, `hf_overrides`, `chat_template`, `chat_template_content_format`, `generation_config`, `override_generation_config`, `tool_call_parser`, `tool_parser_plugin`, `reasoning_parser`, `language_model_only`. Queda separar mejor presets/UX avanzada y validar más casos reales. |
| **Fase 3. Tasks reales** | **Parcial** | `pooling`, `rerank`, `classification` y `score` ya tienen capacidad base, endpoints y Playground adaptado. Falta endurecer más la compatibilidad real y cubrir mejor las respuestas upstream. |
| **Fase 4. Recipes por familia** | **Muy avanzada** | Ya existen recipes por familia/subfamilia, separación entre compatibilidad y tuning, preconfiguración automática y UI final diferenciando recipe base vs recomendación del probe. Queda ampliar cobertura fina. |
| **Fase 5. Runtime probe** | **Avanzada** | Ya existe probe real con arranque corto de `vLLM` para modelos locales/cacheados, fallback al probe ligero y persistencia de `observed_at`. Falta enriquecer detección de recetas/overrides y cubrir más fallos reales. |
| **Fase 6. Tool calling / reasoning / structured outputs** | **Pendiente** | Solo está resuelta la capa de flags y clasificación básica; no la cobertura funcional end-to-end. |
| **Fase 7. Multimodal serio** | **Pendiente** | Sin `language_model_only` ni tuning MM específico. |
| **Fase 8. LoRA y adaptadores** | **Pendiente** | No iniciado. |
| **Fase 9. Quantization y hardware** | **Pendiente** | Solo warnings puntuales; sin matriz seria aún. |
| **Fase 10. Presets y observabilidad** | **Pendiente** | No iniciado. |

### Siguiente bloque recomendado

Siguiente bloque de mayor valor, manteniendo continuidad con lo ya entregado:

1. Terminar Fase 3 con `rerank`, `classification` y `score` más sólidos.
2. Subir Fase 5 desde probe ligero a probe con arranque real de `vLLM` para modelos ya descargados.
3. Empezar Fase 4 con recipes por familia para reducir configuración manual.

## Fase 1. Modelo de compatibilidad y serving mode

**Estado actual:** Parcialmente completada.

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

**Estado actual:** Casi completada.

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

**Estado actual:** Parcialmente completada.

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

**Estado actual:** Pendiente.

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
