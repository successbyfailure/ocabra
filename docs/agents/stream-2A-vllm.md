# Estado: Stream 2-A — vLLM

Estado: completado con backlog de mejora opcional.

Área principal:
- `backend/ocabra/backends/vllm_backend.py`
- `workers/vllm_worker.py`
- `backend/ocabra/backends/vllm_recipes.py`

Resultado actual:
- backend `vllm` operativo para chat, completions, embeddings, pooling, rerank y clasificación según modelo
- recipes y overrides útiles integrados
- estimación heurística y probe runtime real disponibles para configuración de modelos
- validación real de `vllm/Qwen/Qwen3.5-0.8B` y `vllm/Qwen/Qwen3-32B-AWQ`

Pendiente real:
- seguir ampliando recipes o soporte de familias solo si aparece necesidad concreta

Referencia viva:
- `docs/PLAN.md`
- `docs/tasks/vllm-max-support-handoff.md`
