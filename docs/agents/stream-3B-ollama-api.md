# Briefing: Stream 3-B — Ollama API Compatibility

**Prerequisito: Streams 2-A completado (vLLM), 1-B completado.**
**Rama:** `feat/3-B-ollama-api`

## Objetivo

Implementar compatibilidad con la API de Ollama para que herramientas como
Open WebUI, Continue.dev, Obsidian AI, etc. funcionen sin configuración adicional.

## Ficheros propios

```
backend/ocabra/api/ollama/__init__.py
backend/ocabra/api/ollama/tags.py
backend/ocabra/api/ollama/show.py
backend/ocabra/api/ollama/pull.py
backend/ocabra/api/ollama/generate.py
backend/ocabra/api/ollama/chat.py
backend/ocabra/api/ollama/embeddings.py
backend/ocabra/api/ollama/delete.py
backend/tests/test_ollama_api.py
```

## Endpoints a implementar

Referencia: https://github.com/ollama/ollama/blob/main/docs/api.md

### GET /api/tags

Lista modelos instalados en formato Ollama:

```json
{
  "models": [
    {
      "name": "mistral:7b",
      "model": "mistral:7b",
      "modified_at": "2024-01-15T10:00:00Z",
      "size": 4108916384,
      "digest": "sha256:...",
      "details": {
        "parent_model": "",
        "format": "safetensors",
        "family": "mistral",
        "families": ["mistral"],
        "parameter_size": "7B",
        "quantization_level": "F16"
      }
    }
  ]
}
```

### POST /api/show

```json
// Request: { "name": "mistral:7b" }
// Response: modelfile, parameters, template, details, model_info
```

### POST /api/pull

```json
// Request: { "name": "llama3.2:3b", "stream": true }
// Response: stream NDJSON de progreso
{"status": "pulling manifest"}
{"status": "pulling layer", "digest": "sha256:...", "total": 2142590208, "completed": 241664}
{"status": "success"}
```

Delega al DownloadManager de 1-C.

### POST /api/generate

```json
// Request:
{
  "model": "mistral:7b",
  "prompt": "Why is the sky blue?",
  "stream": true,
  "options": { "temperature": 0.7, "num_predict": 100 }
}
// Response: stream NDJSON
{"model":"mistral:7b","created_at":"...","response":"The","done":false}
{"model":"mistral:7b","created_at":"...","response":"","done":true,
 "total_duration":5000000,"load_duration":100000,"prompt_eval_count":10,
 "eval_count":50,"eval_duration":4900000}
```

Proxy a vLLM `/v1/completions` traduciendo el formato.

### POST /api/chat

```json
// Request:
{
  "model": "mistral:7b",
  "messages": [{"role": "user", "content": "Hello"}],
  "stream": true
}
// Response: stream NDJSON
{"model":"mistral:7b","created_at":"...","message":{"role":"assistant","content":"Hi"},"done":false}
{"model":"mistral:7b","created_at":"...","message":{"role":"assistant","content":""},"done":true,...}
```

Proxy a vLLM `/v1/chat/completions` traduciendo SSE → NDJSON.

### POST /api/embeddings

```json
// Request: { "model": "nomic-embed-text", "input": "text here" }
// Response: { "model": "nomic-embed-text", "embeddings": [[0.1, 0.2, ...]] }
```

### DELETE /api/delete

```json
// Request: { "name": "mistral:7b" }
// Delega a model_manager.delete()
```

## Traducción de nombres de modelo

Ollama usa el formato `nombre:tag` (e.g., `llama3.2:3b`).
oCabra usa IDs de HuggingFace (e.g., `meta-llama/Llama-3.2-3B-Instruct`).

Implementar un mapa bidireccional:

```python
class OllamaNameMapper:
    def to_internal(self, ollama_name: str) -> str:
        """llama3.2:3b → meta-llama/Llama-3.2-3B-Instruct"""

    def to_ollama(self, model_id: str) -> str:
        """meta-llama/Llama-3.2-3B-Instruct → llama3.2:3b"""
```

El mapa se puede configurar en la config del servidor y se puede ampliar.

## Traducción SSE OpenAI → NDJSON Ollama

```python
async def openai_stream_to_ollama_ndjson(
    openai_stream: AsyncIterator[bytes],
    model_name: str
) -> AsyncIterator[bytes]:
    """
    Convierte el stream SSE de vLLM (data: {...}\n\n)
    al formato NDJSON de Ollama ({...}\n).
    Calcula eval_count, eval_duration, etc. del último chunk.
    """
```

## Traducción de opciones Ollama → parámetros vLLM

```python
OPTION_MAP = {
    "num_predict": "max_tokens",
    "num_ctx": "max_model_len",    # solo en el load del modelo
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "stop": "stop",
    "seed": "seed",
    "repeat_penalty": "repetition_penalty",
}
```

## Tests requeridos

- Test con `ollama` Python client apuntando al servidor de test
- Test de traducción NDJSON: verificar formato correcto del stream
- Test de POST /api/pull: verificar que delegoa a DownloadManager
- Test de name mapping: round-trip ollama_name → internal → ollama_name

## Estado

- [ ] En progreso
- [ ] Completado
