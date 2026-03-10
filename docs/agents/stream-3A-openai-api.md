# Briefing: Stream 3-A — OpenAI API Compatibility

**Prerequisito: Streams 2-A, 2-B, 2-C completados.**
**Rama:** `feat/3-A-openai-api`

## Objetivo

Implementar la capa de compatibilidad OpenAI completa. Cualquier cliente que
use la SDK de OpenAI apuntando a oCabra debe funcionar sin modificaciones.

## Ficheros propios

```
backend/ocabra/api/openai/__init__.py
backend/ocabra/api/openai/models.py
backend/ocabra/api/openai/chat.py
backend/ocabra/api/openai/completions.py
backend/ocabra/api/openai/embeddings.py
backend/ocabra/api/openai/images.py
backend/ocabra/api/openai/audio.py
backend/tests/test_openai_api.py
```

## Endpoints a implementar

### GET /v1/models

```python
# Response OpenAI format:
{
  "object": "list",
  "data": [
    {
      "id": "mistral-7b-instruct",
      "object": "model",
      "created": 1234567890,
      "owned_by": "ocabra",
      # extensión oCabra (no estándar, pero compatible):
      "ocabra": {
        "status": "loaded",
        "capabilities": { ... },
        "load_policy": "warm",
        "gpu": 1,
        "vram_used_mb": 8200
      }
    }
  ]
}
```

Solo lista modelos con status `loaded` o `configured`. Incluye todos los tipos.

### POST /v1/chat/completions

```python
# Soportar:
# - stream: true/false
# - tools + tool_choice
# - vision: messages con content=[{"type":"image_url","image_url":{"url":"..."}}]
# - response_format: {"type": "json_object"} | {"type": "text"}
# - max_tokens, temperature, top_p, stop, seed, etc.

# Flujo:
# 1. Identifica modelo → verifica que está LOADED (si no, carga bajo demanda)
# 2. Verifica que el modelo tiene capability chat=True
# 3. Proxy al worker vLLM en /v1/chat/completions
# 4. Si stream=true: SSE pass-through
# 5. Registra stats (tokens, duración, energía)
```

### POST /v1/completions

Similar a chat pero con prompt string. Proxy a vLLM `/v1/completions`.

### POST /v1/embeddings

```python
# Verifica capability embeddings=True
# Proxy a vLLM /v1/embeddings
# Response format estándar OpenAI
```

### POST /v1/images/generations

```python
# Verifica capability image_generation=True
# Proxy a DiffusersBackend
# Soporta: prompt, n, size, response_format (b64_json|url)
# Si response_format=url: guarda imagen temporalmente y sirve via /ocabra/files/{id}
```

### POST /v1/audio/transcriptions

```python
# multipart/form-data con "file" y "model"
# Verifica capability audio_transcription=True
# Proxy a WhisperBackend
# Soporta todos los response_format
```

### POST /v1/audio/speech

```python
# Verifica capability tts=True
# Proxy a TTSBackend
# Retorna audio streaming con Content-Type correcto
```

## Middleware de stats

Crea un middleware FastAPI que wrappea TODOS los endpoints `/v1/*`:

```python
async def stats_middleware(request: Request, call_next):
    start = time.monotonic()
    # Lee power_draw actual de la GPU del modelo
    response = await call_next(request)
    duration_ms = (time.monotonic() - start) * 1000
    # Extrae tokens del response body (si no es streaming)
    # Registra en BD: model_id, gpu, duration, tokens, energy
    return response
```

Para streaming, los tokens se cuentan desde el último chunk (`usage` field).

## Carga bajo demanda

Si llega una request a un modelo `configured` (no cargado):

```python
async def ensure_loaded(model_id: str) -> None:
    state = await model_manager.get_state(model_id)
    if state.status == ModelStatus.LOADED:
        return
    if state.status in (ModelStatus.CONFIGURED, ModelStatus.UNLOADED):
        await model_manager.load(model_id)
        # Esperar hasta LOADED o timeout de 120s
    elif state.status == ModelStatus.LOADING:
        # Esperar con polling hasta LOADED
    else:
        raise HTTPException(503, "Model unavailable")
```

## Errores estándar OpenAI

```python
# Formato de error OpenAI:
{
  "error": {
    "message": "The model 'mistral-7b' does not support tool calls.",
    "type": "invalid_request_error",
    "param": "tools",
    "code": "model_not_capable"
  }
}
```

## Tests requeridos

- Test con `openai` Python SDK apuntando al servidor de test
- Test de streaming: verificar que los chunks SSE llegan bien
- Test de tool calls: verificar formato correcto
- Test de carga bajo demanda: modelo configurado → se carga al recibir petición
- Test de error: modelo sin capability de imagen recibe request de imagen → 400

## Estado

- [ ] En progreso
- [ ] Completado
