# Handoff: vLLM Max Support

Fecha: 2026-03-13

## Estado actual

El trabajo completado hasta ahora cubre:

- Fase 1 parcial:
  - clasificación de compatibilidad HF para `vLLM`
  - `native_vllm` / `transformers_backend` / `pooling` / `unsupported` / `unknown`
  - UI de Explore actualizada
- Fase 2 casi completa:
  - flags críticos de `vLLM` en backend, worker, config global y override por modelo
  - UI de configuración por modelo actualizada
- Fase 3 parcial:
  - capacidades base para `pooling`, `rerank`, `classification`, `score`
  - endpoints `/v1/pooling` y `/v1/score`
  - `Models` y `Playground` adaptados para `pooling`
- Fase 3 ampliada en esta iteración:
  - endpoints `/v1/rerank` y `/v1/classify`
  - detección backend de sequence classification y rerankers
  - `Playground` adaptado para probar `rerank` y `classification`
  - normalización y validación temprana de payloads para `rerank`/`classification`
  - `score` endurecido con contrato principal `queries`/`documents` y compatibilidad legacy `text_1`/`text_2`
- Fase 5 parcial:
  - probe runtime mínimo con `AutoConfig` + `AutoTokenizer`
- Fase 5 ampliada en esta iteración:
  - probe runtime real con arranque corto de `vLLM` para artefactos locales/cacheados
  - fallback al probe ligero cuando no se puede ejecutar el probe real
  - servicio dedicado `backend/ocabra/registry/vllm_runtime_probe.py`
- Fase 4 iniciada en esta iteración:
  - módulo `backend/ocabra/backends/vllm_recipes.py`
  - recipes base para `Qwen3`, `DeepSeek-R1`, `Granite`, `GLM` y `FunctionGemma`
  - metadata de recipe y config sugerida expuesta en el registry/UI
  - preconfiguración automática al completar descarga mediante `register_config`
  - recipe persistida en `extra_config.vllm` y reutilizable desde `ModelConfigModal`
  - separación explícita entre compatibilidad requerida y tuning recomendado
- Refinamiento incremental posterior:
  - recipes específicas para `Qwen-VL` y `Qwen3-MoE`
  - recipes específicas para `InternVL` y `Llama 4` multimodal orientadas a uso textual
  - priorización mejor de subfamilias antes de recipes genéricas
  - si el probe runtime devuelve `needs_remote_code`, el registry añade `trust_remote_code` a `required_overrides`
  - si el probe runtime recomienda otro `model_impl` o `runner`, el registry corrige esos campos en `vllm_support`
  - Explore y `ModelConfigModal` muestran ahora la recomendación verificada por probe separada de la recipe base
  - esa metadata mínima del probe se persiste en `extra_config.vllm` al instalar desde Explore
  - `ModelConfigModal` incluye acción dedicada para aplicar `model_impl`/`runner` recomendados por probe
  - `Explore` advierte ya cuando la preconfiguración automática se apoya en la recomendación del probe
  - el contrato/UI preservan también `recipe_model_impl` y `recipe_runner`
  - el runtime probe persiste también `observed_at` / `observedAt` para exponer cuándo se observó por última vez la recomendación
  - el runtime probe clasifica ya fallos accionables adicionales: `missing_chat_template`, `missing_tool_parser`, `missing_reasoning_parser`, `needs_hf_overrides`
  - el registry traduce esas señales a `required_overrides` concretos para Explore, instalación y configuración manual
  - la UI usa ya etiquetas legibles del probe y muestra el override sugerido cuando el estado implica una acción concreta
  - `Explore` y `ModelConfigModal` muestran ya cuándo la recipe base no coincide con la recomendación final del probe
  - existe prueba de flujo frontend tipo E2E con Vitest/jsdom para el caso `Explore -> instalar con metadata de recipe + probe`
  - el repo incorpora además E2E de navegador con Playwright + Chrome para el flujo principal de `Explore`, incluyendo guidance recipe/probe y caso accionable con override sugerido

## Ficheros principales tocados

- `backend/ocabra/backends/base.py`
- `backend/ocabra/backends/vllm_backend.py`
- `backend/ocabra/backends/vllm_recipes.py`
- `backend/ocabra/config.py`
- `backend/ocabra/registry/huggingface.py`
- `backend/ocabra/registry/vllm_runtime_probe.py`
- `backend/ocabra/schemas/registry.py`
- `backend/ocabra/api/openai/__init__.py`
- `backend/ocabra/api/openai/pooling.py`
- `backend/ocabra/api/internal/downloads.py`
- `workers/vllm_worker.py`
- `backend/tests/test_vllm_backend.py`
- `backend/tests/test_vllm_runtime_probe.py`
- `backend/tests/test_downloads.py`
- `frontend/src/types/index.ts`
- `frontend/src/api/client.ts`
- `frontend/src/components/models/ModelConfigModal.tsx`
- `frontend/src/components/explore/HFModelCard.tsx`
- `frontend/src/components/models/ModelConfigModal.tsx`
- `frontend/src/components/common/CapabilityBadge.tsx`
- `frontend/src/components/models/ModelCard.tsx`
- `frontend/src/components/playground/PoolingInterface.tsx`
- `frontend/src/pages/Explore.tsx`
- `frontend/src/pages/Models.tsx`
- `frontend/src/pages/Playground.tsx`
- `frontend/src/__tests__/PoolingInterface.test.tsx`
- `frontend/src/__tests__/HFModelCard.test.tsx`
- `frontend/src/__tests__/ModelConfigModal.test.tsx`
- `frontend/src/__tests__/Explore.test.ts`
- `frontend/src/__tests__/ExploreFlow.test.tsx`
- `frontend/e2e/explore.spec.ts`
- `frontend/playwright.config.ts`
- `frontend/package.json`
- `frontend/package-lock.json`

## Validación ya hecha

- Backend:
  - `ruff check` en ficheros tocados
  - `pytest tests/test_openai_api.py tests/test_vllm_backend.py -q`
- Backend adicional en esta iteración:
  - `python -m pytest tests/test_openai_api.py tests/test_vllm_backend.py -q` ejecutado en `ocabra-api:latest` con extras `.[dev,vllm]` → `44 passed`
  - `python -m ruff check ...` ejecutado igual → OK
  - `python -m pytest tests/test_vllm_runtime_probe.py tests/test_registry.py -q -k "hf_detail_classifies_transformers_backend_and_probe or hf_variants_classifies_pooling_models or test_probe_runtime"` → `5 passed`
  - `python -m pytest tests/test_registry.py -q -k "vllm_recipe or exposes_recipe_metadata or hf_detail_classifies_transformers_backend_and_probe or hf_variants_classifies_pooling_models"` → `5 passed`
  - `python -m pytest tests/test_downloads.py tests/test_registry.py -q -k "auto_register_model_uses_register_config or vllm_recipe or exposes_recipe_metadata"` → `4 passed`
  - `python -m pytest tests/test_registry.py -q -k "qwen_vl or qwen3_moe or remote_code_adds_required_override or exposes_recipe_metadata or classifies_transformers_backend_and_probe"` ejecutado en `ocabra-api` con extras `.[dev,vllm]` → `5 passed`
  - `python -m ruff check ocabra/backends/vllm_recipes.py ocabra/registry/huggingface.py tests/test_registry.py` ejecutado igual → OK
  - `python -m pytest tests/test_registry.py -q -k "internvl or llama4_multimodal or overrides_model_impl_and_runner or classifies_transformers_backend_and_probe or qwen3_moe or qwen_vl"` ejecutado en `ocabra-api` con extras `.[dev,vllm]` → `6 passed`
  - `vitest --no-cache src/__tests__/HFModelCard.test.tsx src/__tests__/ModelConfigModal.test.tsx` en `node:20-alpine` → OK
  - `npx tsc -b` en `node:20-alpine` → OK
  - `vitest --no-cache src/__tests__/Explore.test.ts src/__tests__/HFModelCard.test.tsx src/__tests__/ModelConfigModal.test.tsx` en `node:20-alpine` → OK
  - `vitest --no-cache src/__tests__/Explore.test.ts src/__tests__/ExploreFlow.test.tsx src/__tests__/HFModelCard.test.tsx src/__tests__/ModelConfigModal.test.tsx && npx tsc -b` en `node:20-alpine` → OK
  - `./node_modules/.bin/playwright test --project=chrome` ejecutado en `frontend/` → `1 passed`
  - `python -m pytest tests/test_vllm_runtime_probe.py tests/test_registry.py -q -k "remote_code or missing_chat_template or missing_tool_parser or missing_reasoning_parser or needs_hf_overrides"` ejecutado en `ocabra-api` con extras `.[dev,vllm]` → `9 passed`
  - `python -m ruff check ocabra/schemas/registry.py ocabra/registry/huggingface.py ocabra/registry/vllm_runtime_probe.py tests/test_vllm_runtime_probe.py tests/test_registry.py` ejecutado igual → OK
- Frontend:
  - `vitest` para `HFModelCard.test.tsx` y `PoolingInterface.test.tsx`
  - `npm run build`
- Frontend adicional en esta iteración:
  - `vitest --no-cache src/__tests__/PoolingInterface.test.tsx` → OK
  - `npx tsc -b` → OK
  - `vitest --no-cache src/__tests__/HFModelCard.test.tsx` → OK
  - `vitest --no-cache src/__tests__/ModelConfigModal.test.tsx` → OK
- Docker:
  - `docker compose up -d --build api frontend`
  - `api` healthy
  - `frontend` running
- Navegador real:
  - validación ad hoc con `/usr/bin/google-chrome` headless contra `http://127.0.0.1:8484/explore` tras reconstruir `api` y `frontend` → OK

## Pendiente prioritario

Siguiente bloque recomendado:

1. Completar Fase 5:
   - enriquecer el probe real para detectar mejor recipes/overrides necesarios
   - clasificar más stderr reales de `vLLM` en señales accionables aparte de `needs_remote_code`
2. Continuar Fase 4:
   - ampliar recipes por familia
   - usar señales del probe runtime para seleccionar recipes más concretas cuando haya ambigüedad
   - ampliar la preconfiguración automática más allá del flujo actual de descarga
3. Siguiente mejora de UI/flujo:
   - decidir si conviene persistir además una versión/fingerprint del probe, además del `observed_at` ya expuesto, para saber cuándo la recomendación quedó obsoleta
   - ampliar la cobertura Playwright más allá del flujo principal de `Explore` si se quiere subir la regresión de navegador
3. Refinar Fase 3 si hace falta:
   - soportar más variantes upstream de payload/respuesta sin perder claridad contractual

## Tareas concretas sugeridas para el siguiente agente

### 1. Endurecer `rerank` y `classification`

- Verificar contra documentación oficial de vLLM los formatos exactos más útiles:
  - `classification`: entrada simple vs lista
  - `rerank`: variantes compatibles tipo Jina/Cohere
- Mantener la normalización actual:
  - `classification` acepta string o lista de strings no vacíos
  - `rerank` acepta documentos como strings o `{text: ...}` y valida `top_n`
- Añadir tests adicionales de traducción/shape si se soportan más variantes upstream.

### 2. Endurecer `score`

- Mantener `queries`/`documents` como contrato principal en oCabra.
- Conservar `text_1`/`text_2` solo como alias de compatibilidad.
- Si se soportan más variantes upstream, documentarlas y fijarlas con tests antes de exponerlas en UI.

### 3. Mejorar probe runtime real

- Ya existe servicio dedicado en `backend/ocabra/registry/vllm_runtime_probe.py`.
- Siguiente mejora:
  - detectar mejor cuándo faltan `hf_overrides`, `chat_template`, parser o recipe específica
  - clasificar mejor stderr reales de `vLLM`
  - decidir si conviene persistir cache de probe más allá del proceso actual

### 4. Recipes

- Ya existe módulo base.
- Siguiente mejora:
  - ampliar cobertura por familia/subfamilia
  - usar recipes para prellenar instalación/configuración de modelos
  - distinguir mejor entre override requerido y sugerencia de tuning

## Riesgos conocidos

- El probe real ya valida arranque corto de `vLLM`, pero solo para artefactos locales/cacheados.
- `rerank`, `classification` y `score` ya tienen endpoint, UI mínima y validación/normalización básica de entrada, pero siguen dependiendo del contrato runtime de vLLM para la respuesta final.
- Las recipes actuales son útiles pero iniciales; no cubren todavía todas las variantes reales ni validan end-to-end cada preset.
- La preconfiguración automática solo aplica campos sugeridos explícitos por recipe; los overrides delicados siguen requiriendo decisión manual.
- El tuning recomendado ya es visible y aplicable manualmente, pero sigue siendo heurístico y no debe tratarse como garantía de rendimiento óptimo.
- El suite amplio de `test_registry.py` sigue teniendo fallos previos no relacionados con este bloque (DB real / local scanner).
- No tocar `docker-compose.yml` salvo necesidad real.
- No mezclar esta iteración con fallos previos de tests completos de registry/Ollama/local scanner, salvo que bloqueen directamente el siguiente paso.

## Comandos útiles

Backend:

```bash
docker run --rm --entrypoint bash -v /docker/ocabra/backend:/app -w /app --network ocabra_default ocabra-api -lc 'python -m pip install -e ".[dev,vllm]" >/tmp/pip-install.log 2>&1 && python -m pytest tests/test_openai_api.py tests/test_vllm_backend.py -q'
```

Frontend:

```bash
docker run --rm --entrypoint sh -v /docker/ocabra/frontend:/app -w /app node:20-alpine -lc 'npm ci >/tmp/npm-ci.log 2>&1 && npx vitest run src/__tests__/HFModelCard.test.tsx src/__tests__/PoolingInterface.test.tsx && npm run build'
```

Servicios:

```bash
docker compose -f /docker/ocabra/docker-compose.yml up -d --build api frontend
docker compose -f /docker/ocabra/docker-compose.yml ps
```
