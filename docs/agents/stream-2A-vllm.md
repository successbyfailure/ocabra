# Briefing: Stream 2-A — vLLM Backend

**Prerequisito: Stream 1-B completado (BackendInterface disponible).**
**Rama:** `feat/2-A-vllm`

## Objetivo

Implementar el backend de vLLM: lanzar y gestionar procesos vLLM por modelo,
detectar capacidades automáticamente, y proxy de requests.

## Ficheros propios

```
backend/ocabra/backends/vllm_backend.py
workers/vllm_worker.py
backend/tests/test_vllm_backend.py
```

## Contrato a implementar

`BackendInterface` de `docs/CONTRACTS.md §1`. Todos los métodos abstractos.

## Cómo funciona vLLM en oCabra

Cada modelo cargado = un proceso vLLM separado con su propio puerto:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /data/models/{model_id} \
  --tensor-parallel-size {num_gpus} \
  --gpu-memory-utilization 0.90 \
  --port {assigned_port} \
  --host 127.0.0.1 \
  --served-model-name {model_id}
```

El proceso corre en background. oCabra hace proxy de requests hacia `http://127.0.0.1:{port}`.

## Implementación requerida

### vllm_backend.py

```python
class VLLMBackend(BackendInterface):

    async def load(self, model_id: str, gpu_indices: list[int], **kwargs) -> WorkerInfo:
        """
        1. Asigna puerto via worker_pool.assign_port()
        2. Construye el comando vllm con CUDA_VISIBLE_DEVICES={gpu_indices}
        3. Lanza asyncio.create_subprocess_exec()
        4. Espera a que el healthcheck /health pase (hasta 120s con backoff)
        5. Retorna WorkerInfo
        """

    async def unload(self, model_id: str) -> None:
        """Envía SIGTERM al proceso, espera hasta 30s, luego SIGKILL."""

    async def health_check(self, model_id: str) -> bool:
        """GET http://127.0.0.1:{port}/health → True si 200."""

    async def get_capabilities(self, model_id: str) -> BackendCapabilities:
        """
        Lee model config del directorio local:
        - config.json → architecture → infiere si es multimodal, reasoning, etc.
        - tokenizer_config.json → chat_template → sabe si tiene chat
        - generation_config.json → infiere tools si hay tool_choice
        También puede consultar GET /v1/models al proceso vLLM ya cargado.
        """

    async def get_vram_estimate_mb(self, model_id: str) -> int:
        """
        Estima VRAM a partir del tamaño de ficheros .safetensors en el directorio.
        Factor de overhead: x1.2 sobre el tamaño bruto de los pesos.
        Si no hay ficheros locales, usa la estimación de HuggingFace Hub.
        """

    async def forward_request(self, model_id: str, path: str, body: dict) -> Any:
        """httpx.AsyncClient POST a http://127.0.0.1:{port}{path}."""

    async def forward_stream(self, model_id: str, path: str, body: dict) -> AsyncIterator[bytes]:
        """httpx.AsyncClient stream POST, yield chunks."""
```

### Detección de capacidades por arquitectura

```python
ARCHITECTURE_CAPABILITIES = {
    # vision
    "LlavaNextForConditionalGeneration": {"vision": True, "chat": True},
    "Qwen2VLForConditionalGeneration": {"vision": True, "chat": True},
    "InternVLChatModel": {"vision": True, "chat": True},
    # reasoning (modelos con CoT largo)
    "DeepseekV3ForCausalLM": {"reasoning": True, "chat": True, "tools": True},
    "Qwen3ForCausalLM": {"reasoning": True, "chat": True, "tools": True},
    # texto general
    "LlamaForCausalLM": {"chat": True, "tools": True},
    "MistralForCausalLM": {"chat": True, "tools": True},
    "Phi3ForCausalLM": {"chat": True, "tools": True},
    # embeddings
    "BertModel": {"embeddings": True},
    "XLMRobertaModel": {"embeddings": True},
}
```

### workers/vllm_worker.py

Script standalone que puede lanzarse independientemente para debugging:

```python
#!/usr/bin/env python
"""
Wrapper de vLLM que añade healthcheck extendido y gestión de señales.
Uso: python vllm_worker.py --model-id mistral-7b --port 18001 --gpu 1
"""
```

## Variables de entorno del proceso vLLM

```bash
CUDA_VISIBLE_DEVICES=1          # GPU asignada
VLLM_WORKER_MULTIPROC_METHOD=spawn
HF_HOME=/data/hf_cache
```

## Errores a manejar

- `OOMError` durante carga → unload, publicar evento de error, notificar al Model Manager
- Proceso cae inesperadamente → detectar via poll(), marcar modelo como ERROR, intentar reload si auto_reload
- Timeout de startup (>120s) → matar proceso, lanzar excepción

## Tests requeridos

- Mock de `asyncio.create_subprocess_exec`: test de load/unload sin vLLM real
- Test de `get_capabilities`: fixtures de config.json de varios modelos
- Test de detección de arquitectura desconocida → fallback a `{"chat": True}`
- Test de manejo de OOM: proceso termina con código 137

## Estado

- [ ] En progreso
- [ ] Completado
