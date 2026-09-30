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

#### Cabeceras `X-Ocabra-*` — contrato unificado

**Todos los endpoints** de `/v1/*` que apuntan a un modelo (chat, completions,
embeddings, pooling/score/rerank/classify, audio/transcriptions, audio/speech,
audio/generate, images/generations, images/edits) exponen el mismo set de
cabeceras. Los endpoints streaming las flushean **antes** de bloquear en el
load, así el cliente ve la espera desde el primer byte; los no-streaming las
adjuntan a la respuesta final (útil para logs y para saber si la petición
pagó una carga en frío).

| Header | Cuándo | Valor |
|--------|--------|-------|
| `X-Ocabra-Model-Status` | siempre | Estado del worker en el momento de emitir (`configured`, `loading`, `loaded`, `unloaded`, `error`) |
| `X-Ocabra-Model-Id` | siempre | ID canónico del modelo resuelto (post-router) |
| `X-Ocabra-Expected-Wait-Seconds` | cuando no `loaded` y hay histórico | Mediana de las últimas 5 cargas del modelo, en segundos |
| `X-Ocabra-Load-Queue-Depth` | cuando hay presión | Total de cargas encoladas (activas + esperando el mutex de carga cross-backend) |
| `X-Ocabra-Load-Active` | cuando hay presión | Nº de loads actualmente ejecutándose |
| `X-Ocabra-Was-Cold-Start` | post-load y `pre_status != loaded` | `1` cuando la petición pagó un cold-start |
| `X-Ocabra-Load-Duration-Ms` | junto a `Was-Cold-Start` | Tiempo real que el request esperó al load, en milisegundos |
| `X-Ocabra-Router` | routers | `profile_id` del router al que el cliente apuntó |
| `X-Ocabra-Router-Target` | routers | `profile_id` del target que finalmente sirvió la petición |

Ademas, en respuestas SSE (`stream=true`) el servidor intercala dos comentarios
oCabra-aware con el mismo shape:

```
: {"event": "ocabra.model_loading", "model_id": "...", "worker_key": "...",
   "status": "configured", "expected_wait_seconds": 12}

: {"event": "ocabra.model_ready", "model_id": "...", "worker_key": "...",
   "load_duration_ms": 11820, "was_cold_start": true}
```

Las líneas empiezan con `:` (comentario SSE) para no romper parsers estrictos
tipo OpenAI SDK; los oCabra-aware pueden extraer el JSON tras el `:`. Si el
load falla después de haber flusheado headers, el error viaja como
`data: {"error": ...}` en el mismo stream (no como HTTP status).

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

### Image Generation & Editing

| Method | Endpoint | Descripcion |
|--------|----------|-------------|
| POST | `/v1/images/generations` | Generar imagen a partir de texto (`image_generation`) |
| POST | `/v1/images/edits` | Editar imagen guiado por prompt (`image_editing`) |
| GET | `/v1/images/files/{name}` | Servir imagen generada (respuesta `response_format=url`) |

**POST /v1/images/generations**
```json
{
  "model": "qwen-image-2.1",
  "prompt": "A cat sitting on a rainbow",
  "size": "1024x1024",
  "n": 1,
  "num_inference_steps": 20,
  "guidance_scale": 4.0,
  "response_format": "url"
}
```

Tamaños recomendados para Qwen-Image 2.1: `1024x1024`, `1152x864`, `864x1152`,
`1216x832`, `832x1216`, `1344x768`, `768x1344`.

**POST /v1/images/edits** (multipart/form-data)
```
model=qwen-image-edit-plus
prompt=change the neon text to say ELECTRIC
image=@input.png
image_ref_1=@ref2.png    # opcional (multi-referencia; hasta 3 refs)
image_ref_2=@ref3.png    # opcional
mask=@mask.png           # opcional; PNG con alfa (transparente = zona a editar)
num_inference_steps=20
guidance_scale=4.0
strength=0.75
response_format=url
n=1
```

Códigos de error específicos:
- `400 edit_unsupported` — el pipeline del modelo cargado no tiene variante
  img2img (p.ej. `Flux2KleinPipeline`, `ZImagePipeline`, `QwenImage21Pipeline`).
- `400 mask_unsupported` — enviaste una máscara pero el pipeline es
  edit-nativo sin soporte de inpainting (p.ej. `QwenImageEditPlusPipeline`).
- `400 model_not_capable` — el modelo tiene `image_editing=false`.

Capabilities relevantes en `/v1/models/{id}`:
- `image_generation` — soporta text-to-image (`/v1/images/generations`).
- `image_editing` — soporta img2img o edición prompt-based (`/v1/images/edits`).
  Un modelo puede tener uno, otro, ambos o ninguno según el pipeline del backend.

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
| GET | `/api/ps` | Listar modelos cargados en memoria |
| POST | `/api/show` | Detalle de un modelo |
| POST | `/api/pull` | Descargar un modelo de Ollama |
| DELETE | `/api/delete` | Eliminar un modelo |

**GET /api/ps**

Devuelve solo los modelos en estado `LOADED`. Cada entrada incluye los campos
estandar Ollama (`name`, `model`, `size`, `size_vram`, `digest`, `details`,
`expires_at`) mas un campo extra `expected_load_seconds` (mediana historica)
util para estimar el cold-start de modelos que **no** salgan en esta lista.

```json
{
  "models": [
    {
      "name": "qwen3-8b",
      "model": "qwen3-8b",
      "size": 5137025024,
      "size_vram": 5368709120,
      "digest": "sha256:...",
      "details": {"family": "llm", "format": "safetensors", ...},
      "expires_at": "2026-05-10T18:42:00Z",
      "expected_load_seconds": 12
    }
  ]
}
```

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

### Status — Estado del servidor

| Method | Endpoint | Descripcion |
|--------|----------|-------------|
| GET | `/ocabra/status` | Foto compacta de carga: cola, cargas activas y workers residentes |

Endpoint pensado para banners y widgets que quieran renderizar "N cargando /
Q en cola" sin sondear `/ocabra/models` completo. Requiere solo rol `user`.

```json
{
  "loads": {
    "queue_depth": 2,
    "waiting": 1,
    "active": 1,
    "in_progress": ["diffusers/qwen-image-2.1"],
    "waiting_for_service_gpu": [
      {"modelId": "diffusers/qwen-image-2.1", "blockedBy": "hunyuan3d"}
    ]
  },
  "workers": {
    "loaded_count": 2,
    "loaded_ids": ["ollama/gemma4:26b-ctx160k", "whisper/openai/whisper-base"],
    "in_flight_requests": 3
  }
}
```

Semántica:
- `loads.active` — número de `backend.load()` en ejecución (serializados por
  un mutex global; nunca > 1 con la config actual).
- `loads.waiting` — peticiones esperando ese mutex.
- `loads.queue_depth` — `waiting + active`.
- `loads.in_progress` — model_ids canónicos de los loads activos.
- `loads.waiting_for_service_gpu` — peticiones que abortaron por VRAM ocupada
  por un servicio externo (Hunyuan, TRELLIS.2…) y están reintentando en
  bucle hasta que libere. Cada entrada tiene `modelId` y `blockedBy`.
- `workers.in_flight_requests` — suma de requests en vuelo por modelo,
  útil para saber si algo se está sirviendo aunque no haya loads activos.

Poll recomendado: cada 4-5 s.

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
