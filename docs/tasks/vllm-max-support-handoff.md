# Estado actual: vLLM Max Support

Estado: integrado.

Capacidades consolidadas:
- backend `vllm` usable como runtime generalista
- recipes y recomendaciones por familia de modelo
- soporte de `pooling`, `rerank`, `classification` y `score`
- probe runtime real para validar configuraciones de `vllm`
- estimación de memoria y límite de contexto visibles desde la configuración del modelo

Validación real destacada:
- `vllm/Qwen/Qwen3.5-0.8B`: carga, respuesta y descarga correctas
- `vllm/Qwen/Qwen3-32B-AWQ`: operativo con `max_model_len=7800`; `8000` falla correctamente por KV cache insuficiente

Pendiente real:
- ampliar recipes solo si aparecen familias nuevas
- seguir creciendo e2e backend/CI cuando el entorno completo esté disponible

Fuente viva:
- `docs/PLAN.md`
- `docs/CONTRACTS.md`
