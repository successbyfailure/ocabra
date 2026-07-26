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
- runtime actualizado y validado a vLLM 0.26.0; caché AOT/startup persistente
- Nemotron 3 Nano AWQ validado en RTX 3090 y pipeline 3060 + 3090
- Gemma 4, Qwen3.6 y Qwen3-Coder validados en RTX 3090, frente a Ollama y en
  configuraciones de dos GPU; resultados en
  `docs/benchmarks/vllm_026_gemma4_qwen36_coder.md`
- Qwen3.5 27B AWQ validado y registrado con contexto productivo de 64K y KV
  FP8 sobre FlashInfer;
  resultados en `docs/benchmarks/vllm_026_qwen35_context.md`
- KV FP8 extendida a Nemotron (256K), Qwen3-Coder (112K) y Qwen3.6 (64K);
  Gemma 4 conserva BF16 por incompatibilidad de kernels en SM86; resultados
  en `docs/benchmarks/vllm_026_fp8_kv_expansion.md`

Pendiente real:
- seguir ampliando recipes o soporte de familias solo si aparece necesidad concreta
- revalidar speculative decoding en Nemotron-H cuando se corrija la aserción Mamba

Referencia viva:
- `docs/PLAN.md`
- `docs/tasks/vllm-max-support-handoff.md`
