# oCabra API Documentation

**Version**: 0.1.0
**Base URL**: `http://<host>:8000`
**Auth**: JWT via cookie (`access_token`) o header `Authorization: Bearer <token>`

---

## Autenticacion

Todos los endpoints (excepto `/health`, `/ready` y `/ocabra/auth/login`) requieren autenticacion.

```bash
# Login — obtener cookie de sesion
curl -c cookies.txt -X POST /ocabra/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"user","password":"pass"}'

# Usar cookie en requests posteriores
curl -b cookies.txt /ocabra/models

# O usar API key en header
curl -H "Authorization: Bearer sk-..." /v1/chat/completions
```

---

## System

| Method | Endpoint | Descripcion |
|--------|----------|-------------|
| GET | `/health` | Health check basico |
| GET | `/ready` | Readiness check |
| GET | `/metrics` | Metricas Prometheus |

---

## OpenAI Compatible API (`/v1`)

API compatible con el formato OpenAI. Los clientes como `openai-python`, `litellm`, etc. funcionan directamente.

El campo `model` acepta **profile_id** (recomendado) o model_id canonico (legacy).

### Chat & Completions

| Method | Endpoint | Descripcion |
|--------|----------|-------------|
| POST | `/v1/chat/completions` | Chat completion (streaming o no) |
| POST | `/v1/completions` | Text completion |
| GET | `/v1/models` | Listar modelos/perfiles disponibles |
| GET | `/v1/models/{model_id}` | Detalle de un modelo |

**POST /v1/chat/completions**
```json
{
  "model": "qwen3-8b",
  "messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello"}
  ],
  "max_tokens": 512,
  "temperature": 0.7,
  "stream": false
}
```

### Embeddings & Reranking

| Method | Endpoint | Descripcion |
|--------|----------|-------------|
| POST | `/v1/embeddings` | Generar embeddings |
| POST | `/v1/rerank` | Reranking de documentos |
| POST | `/v1/score` | Score de pares de texto |
| POST | `/v1/pooling` | Pooling sobre un modelo |
| POST | `/v1/classify` | Clasificacion de inputs |

**POST /v1/embeddings**
```json
{
  "model": "qwen3-embedding-8b",
  "input": ["Hello world", "Goodbye world"]
}
```

### Audio — TTS (Text-to-Speech)

| Method | Endpoint | Descripcion |
|--------|----------|-------------|
| POST | `/v1/audio/speech` | Generar audio a partir de texto |
| GET | `/v1/audio/voices` | Listar voces disponibles para un modelo |

**POST /v1/audio/speech**
```json
{
  "model": "kokoro-82m",
  "input": "Hola desde oCabra",
  "voice": "af_heart",
  "response_format": "wav",
  "speed": 1.0,
  "language": "Auto"
}
```

Parametros opcionales para voice cloning (modelos Base):
- `reference_audio`: string base64 del audio WAV de referencia (min 5s recomendado)
- `reference_text`: transcripcion del audio de referencia

Parametros para CustomVoice:
- `speaker`: nombre del speaker (ryan, vivian, etc.)
- `instruct`: instruccion de estilo ("Speak calmly and slowly")

Formatos soportados: `mp3` (default), `wav`, `opus`, `flac`, `pcm`, `aac`

### Audio — STT (Speech-to-Text)

| Method | Endpoint | Descripcion |
|--------|----------|-------------|
| POST | `/v1/audio/transcriptions` | Transcribir audio a texto |

**POST /v1/audio/transcriptions** (multipart/form-data)
```
model=whisper-base
file=@audio.wav
language=es           # opcional, auto-deteccion si vacio
response_format=json  # json, verbose_json, text, srt, vtt
temperature=0.0
diarize=true          # opcional, requiere perfil con diarizacion
```

### Audio — Music Generation

| Method | Endpoint | Descripcion |
|--------|----------|-------------|
| POST | `/v1/audio/generate` | Generar musica (ACE-Step) |

### Image Generation

| Method | Endpoint | Descripcion |
|--------|----------|-------------|
| POST | `/v1/images/generations` | Generar imagen a partir de texto |

**POST /v1/images/generations**
```json
{
  "model": "stable-diffusion",
  "prompt": "A cat sitting on a rainbow",
  "size": "512x512",
  "n": 1
}
```

---

## Ollama Compatible API (`/api`)

API compatible con el protocolo de Ollama. Clientes como `ollama-python` funcionan directamente.

| Method | Endpoint | Descripcion |
|--------|----------|-------------|
| POST | `/api/chat` | Chat completion |
| POST | `/api/generate` | Text generation |
| POST | `/api/embed` | Embeddings |
| POST | `/api/embeddings` | Embeddings (legacy) |
| GET | `/api/tags` | Listar modelos |
| POST | `/api/show` | Detalle de un modelo |
| POST | `/api/pull` | Descargar un modelo de Ollama |
| DELETE | `/api/delete` | Eliminar un modelo |

---

## oCabra Internal API (`/ocabra`)

API interna para gestion del servidor. Usada por el dashboard web.

### Models — Gestion de modelos

| Method | Endpoint | Descripcion |
|--------|----------|-------------|
| GET | `/ocabra/models` | Listar todos los modelos con estado |
| POST | `/ocabra/models` | Registrar un modelo nuevo |
| GET | `/ocabra/models/storage` | Uso de almacenamiento |
| GET | `/ocabra/models/{model_id}` | Estado de un modelo |
| PATCH | `/ocabra/models/{model_id}` | Actualizar config del modelo |
| DELETE | `/ocabra/models/{model_id}` | Eliminar modelo y sus ficheros |
| POST | `/ocabra/models/{model_id}/load` | Cargar modelo en GPU |
| POST | `/ocabra/models/{model_id}/unload` | Descargar modelo de GPU |
| POST | `/ocabra/models/{model_id}/memory-estimate` | Estimar VRAM necesaria |

**POST /ocabra/models** — Registrar modelo
```json
{
  "model_id": "vllm/Qwen/Qwen3-8B",
  "backend_type": "vllm",
  "display_name": "Qwen3 8B",
  "load_policy": "on_demand",
  "auto_reload": false,
  "preferred_gpu": 1,
  "extra_config": {}
}
```

**PATCH /ocabra/models/{model_id}** — Actualizar config
```json
{
  "load_policy": "pin",
  "auto_reload": true,
  "preferred_gpu": 0,
  "display_name": "Mi modelo custom",
  "extra_config": {"max_model_len": 8192}
}
```

Load policies:
- `pin`: Siempre cargado, inmune a eviccion
- `warm`: Cargado bajo demanda, no baja por idle
- `on_demand`: Cargado bajo demanda, baja por idle timeout

### Profiles — Perfiles de modelo

Los perfiles exponen variantes de un modelo base con defaults distintos.
Los clientes usan `profile_id` como valor de `model=` en las APIs.

| Method | Endpoint | Descripcion |
|--------|----------|-------------|
| GET | `/ocabra/models/{model_id}/profiles` | Perfiles de un modelo |
| POST | `/ocabra/models/{model_id}/profiles` | Crear perfil |
| GET | `/ocabra/profiles/{profile_id}` | Detalle de perfil |
| PATCH | `/ocabra/profiles/{profile_id}` | Actualizar perfil |
| DELETE | `/ocabra/profiles/{profile_id}` | Eliminar perfil |
| POST | `/ocabra/profiles/{profile_id}/assets` | Subir asset (multipart) |
| DELETE | `/ocabra/profiles/{profile_id}/assets/{key}` | Eliminar asset |

**POST /ocabra/models/{model_id}/profiles** — Crear perfil
```json
{
  "profile_id": "qwen3-8b-creative",
  "display_name": "Qwen3 8B Creative",
  "description": "Perfil creativo con alta temperatura",
  "category": "llm",
  "load_overrides": {},
  "request_defaults": {
    "temperature": 1.2,
    "top_p": 0.95,
    "max_tokens": 2048
  },
  "enabled": true,
  "is_default": false
}
```

Categorias: `llm`, `tts`, `stt`, `image`, `music`

Ejemplo de perfil TTS con voz fija:
```json
{
  "profile_id": "kokoro-heart",
  "display_name": "Kokoro Heart Voice",
  "category": "tts",
  "request_defaults": {
    "voice": "af_heart",
    "speed": 1.0,
    "response_format": "mp3"
  }
}
```

Ejemplo de perfil STT con diarizacion:
```json
{
  "profile_id": "whisper-diarized",
  "display_name": "Whisper con diarizacion",
  "category": "stt",
  "load_overrides": {
    "diarization_enabled": true,
    "whisper": {"diarizationEnabled": true}
  },
  "request_defaults": {
    "diarize": true
  }
}
```

**load_overrides vs request_defaults**:
- `load_overrides`: Afectan como se carga el modelo en GPU. Si difieren entre perfiles, se crea un worker separado (dedicado).
- `request_defaults`: Valores inyectados en cada request. El cliente puede sobreescribirlos.

### GPUs — Estado de GPUs

| Method | Endpoint | Descripcion |
|--------|----------|-------------|
| GET | `/ocabra/gpus` | Listar todas las GPUs |
| GET | `/ocabra/gpus/{index}` | Estado de una GPU |
| GET | `/ocabra/gpus/{index}/stats` | Historico de stats de GPU |

Respuesta de `/ocabra/gpus`:
```json
[
  {
    "index": 0,
    "name": "NVIDIA GeForce RTX 3060",
    "total_vram_mb": 12288,
    "free_vram_mb": 11867,
    "used_vram_mb": 421,
    "utilization_pct": 0.0,
    "temperature_c": 45.0,
    "power_draw_w": 12.5,
    "power_limit_w": 170.0,
    "locked_vram_mb": 0,
    "processes": []
  }
]
```

### Downloads — Descargas de modelos

| Method | Endpoint | Descripcion |
|--------|----------|-------------|
| GET | `/ocabra/downloads` | Listar descargas |
| POST | `/ocabra/downloads` | Iniciar descarga |
| DELETE | `/ocabra/downloads` | Limpiar historial |
| GET | `/ocabra/downloads/{job_id}` | Estado de descarga |
| DELETE | `/ocabra/downloads/{job_id}` | Cancelar descarga |
| GET | `/ocabra/downloads/{job_id}/stream` | SSE de progreso |

**POST /ocabra/downloads** — Iniciar descarga
```json
{
  "source": "huggingface",
  "model_ref": "Qwen/Qwen3-8B",
  "artifact": null,
  "register_config": {
    "backend_type": "vllm",
    "load_policy": "on_demand"
  }
}
```

Sources: `huggingface`, `ollama`, `bitnet`

### Registry — Busqueda de modelos

| Method | Endpoint | Descripcion |
|--------|----------|-------------|
| GET | `/ocabra/registry/hf` | Buscar en HuggingFace |
| GET | `/ocabra/registry/hf/search?q=qwen&task=text-generation` | Busqueda HF |
| GET | `/ocabra/registry/hf/{repo_id}` | Detalle de modelo HF |
| GET | `/ocabra/registry/hf/{repo_id}/variants` | Variantes (GGUF, AWQ, etc.) |
| GET | `/ocabra/registry/ollama/search?q=llama` | Buscar en Ollama |
| GET | `/ocabra/registry/ollama/{model}/variants` | Tags de Ollama |
| GET | `/ocabra/registry/bitnet/search?q=falcon` | Buscar modelos BitNet |
| GET | `/ocabra/registry/bitnet/{repo_id}/variants` | Variantes BitNet |
| GET | `/ocabra/registry/local` | Modelos descargados localmente |

### Services — Servicios interactivos

Servicios externos con UI propia (ComfyUI, A1111, Hunyuan3D, ACE-Step).

| Method | Endpoint | Descripcion |
|--------|----------|-------------|
| GET | `/ocabra/services` | Listar servicios |
| GET | `/ocabra/services/{id}` | Estado de servicio |
| PATCH | `/ocabra/services/{id}` | Habilitar/deshabilitar |
| POST | `/ocabra/services/{id}/refresh` | Refrescar estado |
| POST | `/ocabra/services/{id}/start` | Iniciar servicio |
| POST | `/ocabra/services/{id}/unload` | Descargar runtime |
| PATCH | `/ocabra/services/{id}/runtime` | Actualizar estado runtime |
| POST | `/ocabra/services/{id}/touch` | Marcar actividad |
| GET | `/ocabra/services/{id}/generations` | Historial de generaciones |

### Stats — Estadisticas

| Method | Endpoint | Descripcion |
|--------|----------|-------------|
| GET | `/ocabra/stats/overview` | Resumen general |
| GET | `/ocabra/stats/requests` | Estadisticas de requests |
| GET | `/ocabra/stats/tokens` | Estadisticas de tokens |
| GET | `/ocabra/stats/energy` | Consumo energetico |
| GET | `/ocabra/stats/performance` | Rendimiento por modelo |
| GET | `/ocabra/stats/recent` | Log de requests recientes |
| GET | `/ocabra/stats/by-user` | Stats por usuario |
| GET | `/ocabra/stats/by-group` | Stats por grupo |
| GET | `/ocabra/stats/my` | Mis stats |
| GET | `/ocabra/stats/my-group` | Stats de mi grupo |

Query params comunes: `from` (ISO), `to` (ISO), `model_id`

### Config — Configuracion del servidor

| Method | Endpoint | Descripcion |
|--------|----------|-------------|
| GET | `/ocabra/config` | Config actual |
| PATCH | `/ocabra/config` | Actualizar config |
| POST | `/ocabra/config/litellm/sync` | Sync con LiteLLM proxy |

Config keys principales (camelCase en REST):
- `defaultGpuIndex`, `idleTimeoutSeconds`, `vramBufferMb`
- `vramPressureThresholdPct`, `logLevel`
- `litellmBaseUrl`, `litellmAdminKey`, `litellmAutoSync`
- `energyCostEurKwh`, `maxTemperatureC`
- `vllmGpuMemoryUtilization`, `vllmMaxNumSeqs`, `vllmEnforceeager`
- `sglangMemFractionStatic`, `llamaCppGpuLayers`, `bitnetCtxSize`
- `globalSchedules`: array de schedules de eviccion cron
- `requireApiKeyOpenai`, `requireApiKeyOllama`

### Auth — Autenticacion y usuarios

| Method | Endpoint | Descripcion |
|--------|----------|-------------|
| POST | `/ocabra/auth/login` | Login |
| POST | `/ocabra/auth/logout` | Logout |
| GET | `/ocabra/auth/me` | Usuario actual |
| PUT | `/ocabra/auth/password` | Cambiar password |
| GET | `/ocabra/auth/keys` | Listar API keys propias |
| POST | `/ocabra/auth/keys` | Crear API key |
| DELETE | `/ocabra/auth/keys/{key_id}` | Revocar API key |

### Users — Gestion de usuarios (admin)

| Method | Endpoint | Descripcion |
|--------|----------|-------------|
| GET | `/ocabra/users` | Listar usuarios |
| POST | `/ocabra/users` | Crear usuario |
| GET | `/ocabra/users/{id}` | Detalle usuario |
| PATCH | `/ocabra/users/{id}` | Actualizar usuario |
| DELETE | `/ocabra/users/{id}` | Eliminar usuario |
| POST | `/ocabra/users/{id}/reset-password` | Reset password |
| GET | `/ocabra/users/{id}/keys` | API keys del usuario |
| POST | `/ocabra/users/{id}/keys` | Crear API key para usuario |
| DELETE | `/ocabra/users/{id}/keys/{key_id}` | Revocar API key |

Roles: `system_admin`, `model_manager`, `user`

### Groups — Grupos de acceso

| Method | Endpoint | Descripcion |
|--------|----------|-------------|
| GET | `/ocabra/groups` | Listar grupos |
| POST | `/ocabra/groups` | Crear grupo |
| PATCH | `/ocabra/groups/{id}` | Actualizar grupo |
| DELETE | `/ocabra/groups/{id}` | Eliminar grupo |
| GET | `/ocabra/groups/{id}/members` | Miembros del grupo |
| POST | `/ocabra/groups/{id}/members` | Anadir miembro |
| DELETE | `/ocabra/groups/{id}/members/{user_id}` | Quitar miembro |
| GET | `/ocabra/groups/{id}/models` | Modelos accesibles |
| POST | `/ocabra/groups/{id}/models` | Dar acceso a modelo |
| DELETE | `/ocabra/groups/{id}/models/{model_id}` | Quitar acceso |

### TensorRT-LLM — Compilacion de engines

| Method | Endpoint | Descripcion |
|--------|----------|-------------|
| GET | `/ocabra/trtllm/compile` | Listar jobs de compilacion |
| POST | `/ocabra/trtllm/compile` | Iniciar compilacion |
| DELETE | `/ocabra/trtllm/compile/{job_id}` | Cancelar compilacion |
| GET | `/ocabra/trtllm/compile/{job_id}/stream` | SSE de progreso |
| DELETE | `/ocabra/trtllm/engines/{name}` | Eliminar engine compilado |
| GET | `/ocabra/trtllm/estimate` | Estimar VRAM del engine |

### Host — Info del servidor

| Method | Endpoint | Descripcion |
|--------|----------|-------------|
| GET | `/ocabra/host/stats` | CPU, RAM, disco, uptime |

---

## WebSocket

```
WS /ocabra/ws
```

Eventos emitidos (JSON):
```json
{"type": "gpu_stats",        "data": [GPUState, ...]}
{"type": "model_event",      "data": {"event": "status_changed", "model_id": "...", "status": "..."}}
{"type": "service_event",    "data": {"event": "...", "service_id": "...", "status": "..."}}
{"type": "download_progress","data": {"job_id": "...", "pct": 0.5, "speed_mb_s": 120.0}}
{"type": "system_alert",     "data": {"level": "error", "message": "..."}}
```

---

## Backends soportados

| Backend | Tipo | Modelos | GPU |
|---------|------|---------|-----|
| **vllm** | LLM, Embeddings | HuggingFace transformers | Si |
| **sglang** | LLM, Embeddings | HuggingFace transformers | Si |
| **llama_cpp** | LLM, Embeddings | GGUF | CPU/GPU |
| **bitnet** | LLM | BitNet GGUF (1.58-bit) | CPU |
| **ollama** | LLM, Embeddings | Ollama registry | Externo |
| **tensorrt_llm** | LLM | TRT-LLM engines | Si |
| **whisper** | STT | faster-whisper, Whisper | Si |
| **tts** | TTS | Kokoro, Bark, Qwen3-TTS | Si |
| **chatterbox** | TTS | Chatterbox (voice clone) | Si |
| **voxtral** | TTS | Voxtral (vllm-omni) | Si |
| **diffusers** | Image | Stable Diffusion, FLUX | Si |
| **acestep** | Music | ACE-Step | Si |

---

## Codigos de error

| HTTP | Significado |
|------|-------------|
| 400 | Request invalido (parametros faltantes, JSON malformado) |
| 401 | No autenticado |
| 403 | Sin permisos (rol insuficiente o modelo no accesible) |
| 404 | Modelo/perfil/recurso no encontrado |
| 422 | Validacion fallida (Pydantic) |
| 500 | Error interno del servidor |
| 503 | Modelo no cargado / worker no disponible |

Formato de error OpenAI-compatible (`/v1/*`):
```json
{
  "error": {
    "message": "Model 'xxx' not found",
    "type": "invalid_request_error",
    "param": "model",
    "code": "model_not_found"
  }
}
```

Formato de error interno (`/ocabra/*`):
```json
{
  "detail": "Model 'xxx' not found"
}
```
